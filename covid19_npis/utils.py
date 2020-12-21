import tensorflow as tf
import logging
import datetime

log = logging.getLogger(__name__)


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
    trace, modelParams, fpath="./", name=None, trace_prior=None,
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

    # Name of file
    if name is None:
        name = datetime.datetime.now().strftime("%y_%m_%d_%H")

    # Save via pickle
    import pickle

    pickle.dump(
        [modelParams, trace], open(f"{fpath}/{name}", "wb"),
    )
    log.info(f"Saved trace as {name} in {fpath}!")

    return name, fpath


def load_trace(name, fpath="./"):
    """
    Loads a trace previously saved with `save_trace`
    """
    import pickle

    with open(f"{fpath}/{name}", "rb") as f:
        modelParams, trace = pickle.load(f)

    return modelParams, trace
