Debugging
---------

This is a small list of debug code snippets.

Debugging nans with tensorflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is a little problematic, because some nans occur during the
runtime without being an error. Often these are cases where an
operation has different implementations based on the value of the
input, because it would otherwise lead to a loss of precision.

Therefore we wrote some patches, which put try-except blocks
around these code parts and if a error occurs, disable check_numeric
for this part.

For patching tensorflow_probability
(replace the variables by the correct path):

::

    cd scripts/debugging_patches
    patch -d {$CONDA_PATH}/envs/{$ENVIRONMENT_NAME}/ -p 0 < filter_nan_errors1.patch
    patch -d {$CONDA_PATH}/envs/{$ENVIRONMENT_NAME}/ -p 0 < filter_nan_errors2.patch
    patch -d {$CONDA_PATH}/envs/{$ENVIRONMENT_NAME}/ -p 0 < filter_nan_errors3.patch

And then uncomment these line of codes in the run_script. Check_numerics
has to enabled only before the optimization, not before the initial
sampling, because the nan occuring during the sampling of the gamma
distribution hasn't been patched.

::

    tf.config.run_functions_eagerly(True)
    tf.debugging.enable_check_numerics(stack_height_limit=30, path_length_limit=50)

For debugging the VI, it is reasonable to increase the step size, to run
more quickly into errors


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