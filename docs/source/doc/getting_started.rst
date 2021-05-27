Getting started
---------------


Installation
^^^^^^^^^^^^
We use some functions of our `toolbox for inference and forecast of the spread of the Coronavirus <https://github.com/Priesemann-Group/covid19_inference/>`_. We supply this toolbox via a github submodule, which can to be initialized while cloning the repository. Alternatively our toolbox can be installed with pip.

.. code-block:: console

		git clone --recurse-submodules git@github.com:Priesemann-Group/covid19_npis_europe.git

Next install all required python libaries. Tensorflow isn't put in the
requirements.txt to allow the installation of another distribution (tensorflow-gpu for instance)

.. code-block:: console

    pip install tensorflow==2.5.0
    pip install -r requirements.txt


Now we should be setup and ready to run our code or look into the source code.



Running the simulation(s)
^^^^^^^^^^^^^^^^^^^^^^^^^
If you want to run our code there is a work flow you can follow.

#. Create or generate a dataset.
	We supply multiple scripts to generate different datasets. All of them can be found in the `scripts/data_generators/` folder. All of them create the data files inside the `data` folder.
	You can run them easily with

	.. code-block:: python

		python script.py

	Alternatively you create your own dataset, we wrote are short guide which can help you get your data inserted into our model see `here <guide_for_creating_dataset.html>`_. 
#. Load dataset
	Before we can start to fit with our model we have to load our data files. There are multiple ways to do this but all of them rely on the :py:class:`covid19_npis.ModelParams` class. Have a look into the constructors to see all possibilities.

	The easiest way is to load a complete data folder e.g. `data/Germany_bundesländer` (generated with the Germany_bundesländer.py script).

	.. code-block:: python

		modelParams = covid19_npis.ModelParams.from_folder("./data/Germany_bundesländer/")


#. Generate model with data
	Now according to the dimensions of our data (i.e. time,number of countries...) we create our model. This is done by passing the `modelParams` to our pymc4 model instance.

	.. code-block:: python

		this_model = covid19_npis.model.main_model(modelParams)


#. Sampling
	Sampling is done with the :py:func:`pymc4.sample` function and our model from above. The sampling function generates an  :py:class:`arviz.InferenceData` object, which we can later use for the plotting or for other sample stats.

	.. code-block:: python

		# A typical sample function call
		begin_time = time.time()
		log.info("Start sampling")
		trace = pm.sample(
			this_model,
			burn_in=200,
			num_samples=100,
			num_chains=2,
			xla=True,
			step_size=0.01,
		)
		end_time = time.time()
		log.info("running time: {:.1f}s".format(end_time - begin_time))

		Best practise is to measure the time the sampling takes and to save the trace after sampling.

		# Save the trace
		name, fpath = covid19_npis.utils.save_trace(
		    trace, modelParams, fpath="./traces",
		)

#. Plotting
	Todo


Understanding our model
^^^^^^^^^^^^^^^^^^^^^^^
We supply our model which we used in our publication (wip). If you want to know how it works in detail have a look into our Methods section in the publication and the documentation here. You can also use our functions to create your own model but that could take some effort.

We suggest you start with the :py:class:`covid19_npis.model.main_model` and work your way threw from top to bottom. It is always helpful to have the `tensorflow documentation <https://www.tensorflow.org/api_docs/python/>`_. opened. We use :py:class:`tf.einsum` so you should have a look at `Einstein notation <https://en.wikipedia.org/wiki/Einstein_notation>`_ if you don't know it by heart yet.

