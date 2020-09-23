import tensorflow as tf


def force_cpu_for_tensorflow():
    """
        Sets the used device for tensorflow to the cpu.
    """
    my_devices = tf.config.experimental.list_physical_devices(device_type="CPU")
    tf.config.experimental.set_visible_devices(devices=my_devices, device_type="CPU")
    tf.config.set_visible_devices([], "GPU")
