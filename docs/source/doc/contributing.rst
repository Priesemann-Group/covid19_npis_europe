Contributing
------------


Code formatting
^^^^^^^^^^^^^^^
We use black https://github.com/psf/black as automatic code formatter.
Please run your code through it before you open a pull request.

We do not check for formatting in the testing (travis) but have a config in the repository that uses `black as a pre-commit hook <https://black.readthedocs.io/en/stable/version_control_integration.html>`_.

This snippet should get you up and running:

.. code-block:: python

    conda install -c conda-forge black
    conda install -c conda-forge pre-commit
    pre-commit install
..


Try to stick to `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_.
You can use `type annotations <https://www.python.org/dev/peps/pep-0484/>`_
if you want, but it is not necessary or encouraged.

Testing
^^^^^^^

We use travis and pytest. To check your changes locally:

.. code-block:: python

    python -m pytest --log-level=INFO --log-cli-level=INFO
..

It would be great if anything that is added to the code-base has an according test in the ``tests`` folder. We are not there yet, but it is on the todo. Be encouraged to add tests :)


Documentation
^^^^^^^^^^^^^

The documentation is built using Sphinx from the docstrings. To test it before
submitting, navigate with a terminal to the docs/ directory.
Install (if necessary) the required python packages for the documentation and compile the
documentation.

.. code-block:: bash

		cd docs
		pip install -r piprequirements.txt
		make html
..

The documentation can now be be accessed locally in ``docs/_build/html/index.html``. As an example for the docstring formatting you can look at the documentation of :func:`covid19_npis.model.disease_spread`. We try to use the `numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`_ style.



