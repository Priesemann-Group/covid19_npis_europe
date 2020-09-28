Model
-----

Disease spread
^^^^^^^^^^^^^^

.. autofunction:: covid19_npis.model.disease_spread.InfectionModel

.. autofunction:: covid19_npis.model.disease_spread.construct_generation_interval

.. autofunction:: covid19_npis.model.disease_spread.construct_h_0_t

.. autofunction:: covid19_npis.model.disease_spread.construct_delay_kernel

Reproduction number
^^^^^^^^^^^^^^^^^^^

.. automodule:: covid19_npis.model.reproduction_number
	:members:
	:private-members: _create_distributions

Utility
^^^^^^^

.. automodule:: covid19_npis.model.utils
	:members:


TESTING DRAFT(WIP)
^^^^^^^^^^^^^^^^^^

Functions with the prefix "calc" take pymc4 generator functions as input and calculate new
(from the generator distributions depending) variables.

Functions with the prefix "construct" generate pymc4 generator functions for different distributions, without a lot of mathematical logic.



.. automodule:: covid19_npis.model.number_of_tests
	:members:
	:private-members:
