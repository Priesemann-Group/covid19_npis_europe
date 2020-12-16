import tensorflow as tf
import logging


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
