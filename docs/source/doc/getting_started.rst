Getting started
---------------


Installation
^^^^^^^^^^^^
We use some functions of our `toolbox for inference and forecast of the spread of the Coronavirus <https://github.com/Priesemann-Group/covid19_inference/>`_. We supply this toolbox via a github submodule, which can to be initialized while cloning the repository. Alternatively our toolbox can be installed with pip.

.. code-block:: console

		git clone --recurse-submodules git@github.com:Priesemann-Group/covid19_npis_europe.git

Next install all required python libaries.

.. code-block:: console

		pip install -r requirements.txt


Now we should be setup and ready to run our code or look into the source code.



Running the simulation
^^^^^^^^^^^^^^^^^^^^^^
If you want to run our code there is a work flow you can follow.

#. Create or generate a dataset.
	We supply multiple scripts to generate different datasets. All of them can be found in [insert folder].
	You can run them easily with

	.. code-block:: python

		python script.py

	Alternatively you create your own dataset, we wrote are short guide which can help you get your data inserted into our model see `here <guide_for_creating_dataset.html>`_.
#. Load dataset
	Todo
#. Generate model with data
	Todo
#. Sampling
	Todo
#. Plotting
	Todo


Understanding our model
^^^^^^^^^^^^^^^^^^^^^^^
We supply our model which we used in our publication (wip). If you want to know how it works in detail have a look into our Methods section in the publication and the documentation here. You can also use our functions to create your own model but that could take some effort.

We suggest you start with the :py:class:`covid19_npis.model.main_model` and work your way threw from top to bottom. It is always helpful to have the `tensorflow documentation <https://www.tensorflow.org/api_docs/python/>`_. opened. We use :py:class:`tf.einsum` so you should have a look at `Einstein notation <https://en.wikipedia.org/wiki/Einstein_notation>`_ if you don't know it by heart yet.

