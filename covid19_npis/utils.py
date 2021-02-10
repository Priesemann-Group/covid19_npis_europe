import tensorflow as tf
import logging
import datetime
import os
import collections
import tensorflow_probability as tfp
import numpy as np
from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.bijectors.bijector import ldj_reduction_shape
from tensorflow_probability.python.bijectors.bijector import _autodiff_log_det_jacobian

log = logging.getLogger(__name__)
SKIP_DTYPE_CHECKS = False


def force_cpu_for_tensorflow():
    """
    Sets the used device for tensorflow to the cpu.
    """
    my_devices = tf.config.experimental.list_physical_devices(device_type="CPU")
    tf.config.experimental.set_visible_devices(devices=my_devices, device_type="CPU")
    tf.config.set_visible_devices([], "GPU")


def split_cpu_in_logical_devices(n):
    """
    Splits into multiple logical devices i.e. virtual CPUs
    """

    # Retrieve list of cpus in system
    physical_devices = tf.config.list_physical_devices("CPU")
    log.info(f"Found {len(physical_devices)} CPUs in system.")

    # Specify 2 virtual CPUs. Note currently memory limit is not supported.
    tf.config.set_logical_device_configuration(
        physical_devices[0], [tf.config.LogicalDeviceConfiguration(),] * n,
    )
    logical_devices = tf.config.list_logical_devices("CPU")

    log.info(f"Created {len(logical_devices)} virtual CPUs")


def setup_colored_logs():
    import coloredlogs

    logger = logging.getLogger()
    coloredlogs.install(level=logger.level)

    coloredlogs.install(logger=tf.get_logger(), level=tf.get_logger().level)


def save_trace(
    trace, modelParams, store="./", name=None, trace_prior=None,
):
    """
    Saves a traces from our model run. Adds data and prior traces to InferenceData before
    saving.

    Parameters
    ----------
    trace: arviz.InferenceData
        Trace from a model run.
    modelParams: modelParams
        ModelParams for observed data.
    fpath: str, optional
        Filepath i.e. where to save the trace.
        |default| "./"
    name: str, optional
        Name of the file using timestapm if none.
        |default| None
    trace_prior: arviz.InferenceData, optional
        Trace from prior sampling. pm.sample_prior_predictive

    """
    import copy

    trace = copy.copy(trace)

    # Combine additional data if given
    if trace_prior is not None:
        trace.extend(trace_prior, join="right")

    if not os.path.exists(store):
        os.makedirs(store)

    # Name of file
    if name is None:
        name = datetime.datetime.now().strftime("%y_%m_%d_%H_%M")

    # Save via pickle
    import pickle

    pickle.dump(
        [modelParams, trace], open(f"{store}/{name}", "wb"),
    )
    log.info(f"Saved trace as {name} in {store}!")

    return name, store


def load_trace(name, fpath="./"):
    """
    Loads a trace previously saved with `save_trace`
    """
    import pickle

    with open(f"{fpath}/{name}", "rb") as f:
        modelParams, trace = pickle.load(f)

    return modelParams, trace


def save_trace_zarr(
    trace, modelParams, store=None, name=None, trace_prior=None,
):
    """
    Saves trace using new experimental arviz backend zarr!

    Parameters
    ----------
    trace: arviz.InferenceData
        Trace from a model run.
    modelParams: modelParams
        ModelParams for observed data.
    store: str, optional
        Filepath i.e. where to save the traces datagroups.
        |default| "./traces/[TIMESTAMP]"
    name: str, optional
        Name of the file using timestapm if none.
        |default| None
    trace_prior: arviz.InferenceData, optional
        Trace from prior sampling. pm.sample_prior_predictive
    """
    # This is where the files are stored!
    if name is None:
        name = datetime.datetime.now().strftime("%y_%m_%d_%H")
    if store is None:
        store = f"./traces/{name}"

    # Additional trace given
    if trace_prior is not None:
        trace.extend(trace_prior, join="right")

    # Check if folder exists
    if not os.path.exists(store):
        os.makedirs(store)
    trace.to_zarr(store)

    # Save modelParams
    import pickle

    pickle.dump(
        modelParams, open(f"{store}/modelParams.pickle", "wb"),
    )
    log.info(f"Saved trace & modelParams in {store}!")
    return store


def load_trace_zarr(store):
    """
    Load trace using new experimental arviz backend zarr
    & modelParams using pickle.

    Parameters
    ----------
    store: str
        Filepath i.e. where to save the traces datagroups or
        zarr store.

    Returns
    -------
    trace: arviz.InferenceData
        Trace object
    modelParams: covid19_npis.modelParams.modelParams
        Modelparams class
    """

    import arviz, pickle

    # Load trace
    trace = arviz.InferenceData.from_zarr(store)

    # Load modeParams
    with open(f"{store}/modelParams.pickle", "rb") as f:
        modelParams = pickle.load(f)

    return modelParams, trace


VarMap = collections.namedtuple("VarMap", "slice, shape, flat_shape, dtype")

UNSPECIFIED = object()


class StructuredBijector(tfp.bijectors.Bijector):
    """Bijector which applies one bijector to a structure.
    The event_ndims of the bijector has to be one

  """

    def __init__(self, structure, bijector=None, name=None):
        """Instantiates `StructuredBijector` bijector.
    Args:
      structure: Structure
      bijector: bijector with event_ndims == 1
    """

        if name is None:
            name = "Struct_" + bijector.name
            name = name.replace("/", "")
        with tf.name_scope(name) as name:
            # Structured dtypes are based on the non-wrapped input.
            # Keep track of the non-wrapped structure of bijectors to correctly
            # wrap inputs/outputs in _walk methods.
            self._bijector = bijector
            self._nested_structure = self._no_dependency(
                tf.nest.map_structure(lambda b: None, structure)
            )
            self._list_shapes = []
            size = 0
            x_list = []
            for tensor in tf.nest.flatten(structure):
                flat_shape = int(np.prod(tensor.shape))
                x_list.append(tf.reshape(tensor, shape=(-1)))
                slc = slice(size, size + flat_shape)
                self._list_shapes.append(
                    VarMap(slc, list(tensor.shape), flat_shape, tensor.dtype)
                )
                size += flat_shape
            x_flat = tf.concat(x_list, axis=0)
            self._size = size
            self._flat_struct = x_flat

            super(StructuredBijector, self).__init__(
                forward_min_event_ndims=tf.nest.map_structure(lambda _: 1, structure),
                inverse_min_event_ndims=tf.nest.map_structure(lambda _: 0, structure),
                is_constant_jacobian=bijector.is_constant_jacobian,
                validate_args=bijector.validate_args,
                dtype=tf.nest.map_structure(lambda _: tf.float32, structure),
                parameters=bijector.parameters,
                name=name,
            )

    def _call_fn(self, fn, struct, *args, **kwargs):
        nested_structure = tf.nest.map_structure(lambda b: None, struct)

        x_list = []
        size = 0
        list_shapes_local = []
        for i, tensor in enumerate(
            tf.__internal__.nest.flatten_up_to(self._nested_structure, struct)
        ):
            if hasattr(tensor, "shape"):
                x_list.append(
                    tf.reshape(tensor, shape=(-1, self._list_shapes[i].flat_shape))
                )
                flat_shape = int(x_list[-1].shape[-1])
                slc = slice(size, size + flat_shape)
                list_shapes_local.append(
                    VarMap(slc, tensor.shape, flat_shape, tensor.dtype)
                )
                size += flat_shape
            else:
                x_list.append(tensor)
                flat_shape = 1
                slc = slice(size, size + flat_shape)
                list_shapes_local.append(VarMap(slc, None, None, None))
                size += flat_shape

        if hasattr(x_list[0], "shape"):
            x_flat = tf.concat(x_list, axis=1)
            res_flat = fn(x_flat, *args, **kwargs)
        else:
            res_flat = [fn(x, *args, **kwargs) for x in x_list]

        struct_flat = []
        for var in list_shapes_local:
            slc, shape, flat_shape, dtype = var
            if shape is not None:
                struct_flat.append(
                    tf.cast(tf.reshape(res_flat[..., slc], shape), dtype)
                )
            else:
                struct_flat.append(res_flat[slc][0])
        return tf.nest.pack_sequence_as(nested_structure, struct_flat)

    def _call_jac(self, fn, struct, *args, **kwargs):
        nested_structure = tf.nest.map_structure(lambda b: None, struct)

        x_list = []
        size = 0
        list_shapes_local = []
        for i, tensor in enumerate(
            tf.__internal__.nest.flatten_up_to(self._nested_structure, struct)
        ):
            if hasattr(tensor, "shape"):
                x_list.append(
                    tf.reshape(tensor, shape=(-1, self._list_shapes[i].flat_shape))
                )
                flat_shape = int(x_list[-1].shape[-1])
                slc = slice(size, size + flat_shape)
                list_shapes_local.append(
                    VarMap(slc, tensor.shape, flat_shape, tensor.dtype)
                )
                size += flat_shape
            else:
                x_list.append(tensor)
                flat_shape = 1
                slc = slice(size, size + flat_shape)
                list_shapes_local.append(VarMap(slc, None, None, None))
                size += flat_shape

        if hasattr(x_list[0], "shape"):
            x_flat = tf.concat(x_list, axis=1)
            res_flat = fn(x_flat, *args, **kwargs)
        else:
            res_flat = [fn(x, *args, **kwargs) for x in x_list]

        return res_flat

    def _forward(self, x, **kwargs):
        return self._call_fn(self._bijector.forward, x, **kwargs)

    def _inverse(self, x, **kwargs):
        return self._call_fn(self._bijector.inverse, x, **kwargs)

    def inverse_log_det_jacobian(self, y, event_ndims, **kwargs):
        return self._call_jac(
            self._bijector.inverse_log_det_jacobian, y, event_ndims=1, **kwargs
        )

    def forward_log_det_jacobian(self, x, event_ndims, **kwargs):
        return self._call_jac(
            self._bijector.forward_log_det_jacobian, x, event_ndims=1, **kwargs
        )

    def inverse_event_ndims(self, nd, **kwargs):
        return tf.nest.map_structure(lambda _: 1, nd)

    """
    def inverse_dtype(self, dtype=UNSPECIFIED, name="inverse_dtype", **kwargs):
        Returns the dtype returned by `inverse` for the provided input.
        with tf.name_scope("{}/{}".format(self.name, name)):
            if dtype is UNSPECIFIED:
                # We pass the the broadcasted output structure through `_inverse_dtype`
                # rather than directly returning the input structure, allowing
                # subclasses to alter results based on `**kwargs`.
                output_dtype = nest_util.broadcast_structure(
                    self._nested_structure, self.dtype
                )
            else:
                # Make sure inputs are compatible with statically-known dtype.
                output_dtype = tf.__internal__.nest.map_structure_up_to(
                    self._nested_structure,
                    lambda y: dtype_util.convert_to_dtype(y, dtype=self.dtype),
                    nest_util.coerce_structure(self._nested_structure, dtype),
                    check_types=False,
                )

            input_dtype = self._inverse_dtype(output_dtype, **kwargs)
            try:
                # kwargs may alter dtypes themselves, but we currently require
                # structure to be statically known.
                tf.nest.assert_same_structure(
                    self._nested_structure, input_dtype, check_types=False
                )
            except Exception as err:
                raise NotImplementedError(
                    "Changing output structure in `inverse_dtype` "
                    "at runtime is not currently supported:\n" + str(err)
                )
            return input_dtype
    """

    def forward_dtype(self, dtype=UNSPECIFIED, name="forward_dtype", **kwargs):
        """Returns the dtype returned by `forward` for the provided input."""
        with tf.name_scope("{}/{}".format(self.name, name)):
            if dtype is UNSPECIFIED:
                # We pass the the broadcasted input structure through `_forward_dtype`
                # rather than directly returning the output structure, allowing
                # subclasses to alter results based on `**kwargs`.
                input_dtype = nest_util.broadcast_structure(
                    self._nested_structure, self.dtype
                )
            else:
                # Make sure inputs are compatible with statically-known dtype.
                input_dtype = tf.__internal__.nest.map_structure_up_to(
                    self._nested_structure,
                    lambda x, static_dtype: dtype_util.convert_to_dtype(
                        x, dtype=static_dtype
                    ),
                    nest_util.coerce_structure(self._nested_structure, dtype),
                    nest_util.coerce_structure(self._nested_structure, self.dtype),
                    check_types=False,
                )

            output_dtype = self._forward_dtype(input_dtype, **kwargs)
            try:
                # kwargs may alter dtypes themselves, but we currently require
                # structure to be statically known.
                tf.nest.assert_same_structure(
                    self._nested_structure, output_dtype, check_types=False
                )
            except Exception as err:
                raise NotImplementedError(
                    "Changing output structure in `forward_dtype` "
                    "at runtime is not currently supported:\n" + str(err)
                )
            return output_dtype

    def inverse_dtype(self, dtype=UNSPECIFIED, name="inverse_dtype", **kwargs):
        """Returns the dtype returned by `inverse` for the provided input."""
        with tf.name_scope("{}/{}".format(self.name, name)):
            if dtype is UNSPECIFIED:
                # We pass the the broadcasted output structure through `_inverse_dtype`
                # rather than directly returning the input structure, allowing
                # subclasses to alter results based on `**kwargs`.
                output_dtype = nest_util.broadcast_structure(
                    self._nested_structure, self.dtype
                )
            else:
                # Make sure inputs are compatible with statically-known dtype.
                output_dtype = tf.__internal__.nest.map_structure_up_to(
                    self._nested_structure,
                    lambda x, static_dtype: dtype_util.convert_to_dtype(
                        x, dtype=static_dtype
                    ),
                    nest_util.coerce_structure(self._nested_structure, dtype),
                    nest_util.coerce_structure(self._nested_structure, self.dtype),
                    check_types=False,
                )

            input_dtype = self._inverse_dtype(output_dtype, **kwargs)
            try:
                # kwargs may alter dtypes themselves, but we currently require
                # structure to be statically known.
                tf.nest.assert_same_structure(
                    self._nested_structure, input_dtype, check_types=False
                )
            except Exception as err:
                raise NotImplementedError(
                    "Changing output structure in `inverse_dtype` "
                    "at runtime is not currently supported:\n" + str(err)
                )
            return input_dtype

    def _call_forward_log_det_jacobian(self, x, event_ndims, name, **kwargs):
        """Wraps call to _forward_log_det_jacobian, allowing extra shared logic.

        Specifically, this method
          - adds a name scope,
          - performs validations,
          - handles the special case of non-injective Bijector (forward jacobian is
            ill-defined in this case and we raise an exception)

        so that sub-classes don't have to worry about this stuff.

        Args:
          x: same as in `forward_log_det_jacobian`
          event_ndims: same as in `forward_log_det_jacobian`
          name: same as in `forward_log_det_jacobian`
          **kwargs: same as in `forward_log_det_jacobian`

        Returns:
          fldj: the forward log det jacobian at `x`. Also updates the cache as
          needed.
        """
        if not self._is_injective:
            raise NotImplementedError(
                "forward_log_det_jacobian cannot be implemented for non-injective "
                "transforms."
            )

        if not self.has_static_min_event_ndims:
            raise NotImplementedError(
                "Subclasses without static `forward_min_event_ndims` must override "
                "`_call_forward_log_det_jacobian`."
            )

        with self._name_and_control_scope(name):
            dtype = self.inverse_dtype(**kwargs)
            x = nest_util.convert_to_nested_tensor(
                x,
                name="x",
                dtype_hint=dtype,
                dtype=None if SKIP_DTYPE_CHECKS else dtype,
                allow_packing=True,
            )

            reduce_shape, assertions = ldj_reduction_shape(
                nest_util.convert_to_nested_tensor(ps.shape(self._flat_struct)),
                event_ndims=nest_util.coerce_structure(
                    self.forward_min_event_ndims, event_ndims
                ),
                min_event_ndims=self._forward_min_event_ndims,
                parameter_batch_shape=self._parameter_batch_shape,
                allow_event_shape_broadcasting=self._allow_event_shape_broadcasting,
                validate_args=self.validate_args,
            )

            # Make sure we have validated reduce_shape before continuing on.
            with tf.control_dependencies(assertions):
                # Make sure the unreduced ILDJ is in the cache.
                attrs = self._cache.forward_attributes(x, **kwargs)
                if "ildj" in attrs:
                    ildj = attrs["ildj"]
                elif hasattr(self, "_forward_log_det_jacobian"):
                    ildj = attrs["ildj"] = -self._forward_log_det_jacobian(x, **kwargs)
                elif hasattr(self, "_inverse_log_det_jacobian"):
                    y = self.forward(x, **kwargs)  # Fall back to computing `ildj(y)`
                    ildj = attrs["ildj"] = self._inverse_log_det_jacobian(y, **kwargs)
                elif self._is_scalar:
                    ildj = -_autodiff_log_det_jacobian(self._forward, x)
                else:
                    raise NotImplementedError(
                        "Neither _forward_log_det_jacobian nor _inverse_log_det_jacobian "
                        "is implemented. One or the other is required."
                    )

                return self._reduce_jacobian_det_over_shape(-ildj, reduce_shape)


"""
    def _walk_forward(self, step_fn, xs, **kwargs):
        Applies `transform_fn` to `x` in parallel over nested bijectors.
        # Set check_types to False to support bij-structures wrapped by Trackable.
        nested_structure = tf.nest.map_structure(lambda b: None, xs)

        x_list = []
        size = 0
        list_shapes_local = []
        for i, tensor in enumerate(
            tf.__internal__.nest.flatten_up_to(self._nested_structure, xs)
        ):
            if hasattr(tensor, "shape"):
                x_list.append(
                    tf.reshape(tensor, shape=(-1, self._list_shapes[i].flat_shape))
                )
                flat_shape = int(x_list[-1].shape[-1])
                slc = slice(size, size + flat_shape)
                list_shapes_local.append(
                    VarMap(slc, tensor.shape, flat_shape, tensor.dtype)
                )
                size += flat_shape
            else:
                x_list.append(tensor)
                flat_shape = 1
                slc = slice(size, size + flat_shape)
                list_shapes_local.append(VarMap(slc, None, None, None))
                size += flat_shape
        bij = self._bijectors[0]
        if hasattr(x_list[0], "shape"):
            x_flat = tf.concat(x_list, axis=1)
            res_flat = step_fn(bij, x_flat, **kwargs.get(bij.name, {}))
        else:
            res_flat = [step_fn(bij, x) for x in x_list]

        struct_flat = []
        for var in list_shapes_local:
            slc, shape, flat_shape, dtype = var
            if shape is not None:
                struct_flat.append(
                    tf.cast(tf.reshape(res_flat[..., slc], shape), dtype)
                )
            else:
                struct_flat.append(res_flat[slc][0])
        return tf.nest.pack_sequence_as(nested_structure, struct_flat)

    def _walk_inverse(self, step_fn, ys, **kwargs):
        Applies `transform_fn` to `y` in parallel over nested bijectors.
        # Set check_types to False to support bij-structures wrapped by Trackable.
        return self._walk_forward(step_fn, ys, **kwargs)
"""
