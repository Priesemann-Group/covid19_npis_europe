import copy
import sys

import numpy as np
import tensorflow_probability as tfp
import pymc4 as pm
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb


# The code in this module raises an error during cleanup. This hook catches it such that
# it doesn't get printed
def unraisablehook(unraisable, **kwargs):
    if "WeakStructRef" in f"{unraisable.object!r}":
        pass
    else:
        print(f"{unraisable.err_msg}: {unraisable.object!r}")


sys.unraisablehook = unraisablehook

# deprecated
def create_bijector_fn(shift_and_log_scale_fn):
    # def bijector_fn(x, **condition_kwargs):
    #     params = shift_and_log_scale_fn(x, **condition_kwargs)
    #     shift, scale = tf.unstack(params, num=2, axis=-1)
    #
    #     bijectors = []
    #     scale = tf.math.sigmoid(scale)
    #     bijectors.append(tfb.Shift(shift * (1 - scale)))
    #     bijectors.append(tfb.Scale(scale=scale + 1e-5))
    #     return tfb.Chain(bijectors, validate_event_size=False)

    def bijector_fn(x, **condition_kwargs):
        params = shift_and_log_scale_fn(x, **condition_kwargs)
        shift, log_scale = tf.unstack(params, num=2, axis=-1)

        bijectors = []
        bijectors.append(tfb.Shift(shift))
        bijectors.append(tfb.Scale(log_scale=log_scale))
        return tfb.Chain(bijectors, validate_event_size=False)

    return bijector_fn


def build_iaf(values_iaf_dict, order_list, values_exclude_dict=None):

    values_iaf_dict = copy.deepcopy(values_iaf_dict)
    values_exclude_dict = copy.deepcopy(values_exclude_dict)

    init_iaf_struct = {
        k: i for k, i in zip(values_iaf_dict.keys(), range(len(values_iaf_dict)))
    }

    size_iaf = sum([int(np.prod(tensor.shape)) for tensor in values_iaf_dict.values()])

    size_splits = [int(np.prod(v.shape)) for v in values_iaf_dict.values()]
    iaf_split = tfb.Split(size_splits, axis=-1)

    iaf_restructure = tfb.Restructure(init_iaf_struct)
    iaf_reshape = tfb.JointMap(
        {
            k: tfb.Reshape(v.shape, (int(np.prod(v.shape)),))
            for k, v in values_iaf_dict.items()
        }
    )
    iaf_reorder = tfb.Chain([iaf_reshape, iaf_restructure, iaf_split])

    bijectors_iaf_list = []

    for order in order_list:
        bijectors_iaf_list.append(
            tfb.Invert(
                tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=tfp.bijectors.AutoregressiveNetwork(
                        params=2,
                        hidden_units=[size_iaf, size_iaf],
                        input_order=order,
                        activation="elu",
                    )
                )
            )
        )

    bijector_iaf = tfb.Chain(bijectors_iaf_list)
    bijector_iaf_total = tfb.Chain([iaf_reorder, bijector_iaf, tfb.Invert(iaf_reorder)])

    ### Make identity bijector for excluded values:

    if values_exclude_dict is not None:
        init_rest_struct = {
            k: i
            for k, i in zip(
                values_exclude_dict.keys(),
                range(
                    len(values_iaf_dict),
                    len(values_iaf_dict) + len(values_exclude_dict),
                ),
            )
        }
        init_struct = {**init_iaf_struct, **init_rest_struct}

        bijector_rest = tfb.JointMap(
            {name: tfb.Identity() for name, tensor in values_exclude_dict.items()}
        )

        # Join the two bijectors:

        restructure_split = tfb.Restructure(
            [init_iaf_struct, init_rest_struct], init_struct
        )
        restructure_merge = tfb.Invert(restructure_split)
        bijector = tfb.Chain(
            [
                restructure_merge,
                tfb.JointMap([bijector_iaf_total, bijector_rest]),
                restructure_split,
            ],
        )

    else:
        bijector = bijector_iaf_total

    return bijector


def build_approximate_posterior(model):

    order_list = ["right-to-left", "left-to-right"]
    _, state = pm.evaluate_model_transformed(model)
    state, _ = state.as_sampling_state()

    values_dict = dict(state.all_unobserved_values)
    transformed_names = list(values_dict.keys())

    noise_vars = ("main_model|noise_R", "main_model|noise_R_age")
    values_without_noise = {k: v for k, v in values_dict.items() if k not in noise_vars}
    values_with_noise = {k: v for k, v in values_dict.items() if k in noise_vars}

    normal_base = tfd.JointDistributionNamed(
        {
            name: tfd.Sample(tfd.Normal(loc=0.0, scale=1.0), sample_shape=tensor.shape)
            for name, tensor in values_dict.items()
        },
        validate_args=False,
        name=None,
    )

    bijectors_list = []
    for vals, orders, vals_exclude in [
        (values_without_noise, order_list, values_with_noise),
        (values_dict, order_list, None),
        (values_without_noise, order_list, values_with_noise),
    ]:
        bijectors_list.append(build_iaf(vals, orders, vals_exclude))

    bijector = tfb.Chain(bijectors_list)
    posterior_approx = tfd.TransformedDistribution(normal_base, bijector=bijector)
    return posterior_approx, bijector, transformed_names
