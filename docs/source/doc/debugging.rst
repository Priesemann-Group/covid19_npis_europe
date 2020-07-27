Debugging
---------

This is a small list of debug code snippets.

Enable tensorflow eager execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

	tf.config.run_functions_eagerly(True)
	tf.debugging.enable_check_numerics(stack_height_limit=30, path_length_limit=50)


Basic usage of logger
^^^^^^^^^^^^^^^^^^^^^

::

	# Change to debug mode i.e all log.debug is printed
	logging.basicConfig(level=logging.DEBUG)

	# Use log.debug instead of print
	log.debug(f"My var {var}")


Force cpu or other device
^^^^^^^^^^^^^^^^^^^^^^^^^

::

	my_devices = tf.config.experimental.list_physical_devices(device_type="CPU")
	tf.config.experimental.set_visible_devices(devices=my_devices, device_type="CPU")
	tf.config.set_visible_devices([], "GPU")