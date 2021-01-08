.. _how-to-data:

How to build a dataset for our model
------------------------------------

To use our model you may want to create your own dataset. In the following we try to guide you through the process of creating your own dataset. Feel free to take a look into our `script`_. we use to create our dataset.

.. _script: https://github.com/Priesemann-Group/covid19_npis_europe/blob/master/scripts/download_new_data.py


We use a hierarchical for our data as for our model. To add new country or region to our model we first create a folder containing the data.

::

	  mkdir test_country


Next we create a config.json file inside this folder. The json has to contain a unique name for the country/region and the age group brackets. You can add any number of age groups, the name of the groups should be the same across all countries though! We use four different age groups for most of our analysis as follows.


.. code-block:: json

    {
        "name": "test_country",
        "age_groups": {
        	"age_group_0" : [0,34],
        	"age_group_1" : [35,59],
        	"age_group_2" : [60,79],
        	"age_group_3" : [80,100]
        }
    }


- config.json, dict:
    - name : "country_name"
    - age_groups : dict 
        - "column_name" : [age_lower, age_upper]


Population data
^^^^^^^^^^^^^^^

Each dataset for a country/region needs to contain population data for every age from 0 to 100. The data should be saved as population.csv! Most of the population data can be found on the `UN website`_.

.. _UN website: https://population.un.org/wpp/Download/

+-------+------------+
| age   | PopTotal   |
+=======+============+
| 0     | 831175     |
+-------+------------+
| 1     | 312190     |
+-------+------------+
| ...   | ...        |
+-------+------------+

- Age column named "age"
- Column Number of people per age named "PopTotal"

New cases/ Positive tests data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We supply the number of positive tested persons per day and age group as a csv file for our country/region. The file has to be named "new_cases.csv" and has to contain the same column names as defined in the config.json! That is the age groups.
Date Format should be "%d.%m.%y".


+----------+-------------+-------------+-------------+-------------+
| date     | age_group_0 | age_group_1 | age_group_2 | age_group_3 |
+==========+=============+=============+=============+=============+
| 01.01.20 |     103     |     110     |     13      |     130     |
+----------+-------------+-------------+-------------+-------------+
| 02.01.20 |     103     |     103     |     103     |     103     |
+----------+-------------+-------------+-------------+-------------+
| ...      |     ...     |     ...     |     ...     |     ...     |
+----------+-------------+-------------+-------------+-------------+

- Time/Date column has to be named "date" or "time"
- Age group columns have to be named consistent between different data and countries! 


Total tests data
^^^^^^^^^^^^^^^^

The number of total tests performed per day in the country/region is also supplied as
a csv file called "tests.csv". The format should be as follows: 


+----------+-------------+
| date     |      tests  |
+==========+=============+
| 01.01.20 |   10323     |
+----------+-------------+
| 02.01.20 |   13032     |
+----------+-------------+
| ...      |     ...     |
+----------+-------------+

- Time/Date column has to be named "date" or "time"
- Daily performed tests column with name "tests"


Number of deaths data
^^^^^^^^^^^^^^^^^^^^^

The number of deaths per day in the country/region also supplied as csv file nameed "deaths.csv".


+----------+-------------+
| date     |     deaths  |
+==========+=============+
| 01.01.20 |   10        |
+----------+-------------+
| 02.01.20 |   35        |
+----------+-------------+
| ...      |     ...     |
+----------+-------------+

- Time/Date column has to be named "date" or "time"
- Daily deaths column has to be named "deaths"
- Optional(not working yet): Daily deaths per age group same column names as in new_cases

Interventions data
^^^^^^^^^^^^^^^^^^

The intervention is also added as csv file. The file has to be named "interventions.csv" and can contain any number of interventions. We use the the `oxford response tracker`_ for this purpose, but you can also construct your own time series.

.. _oxford response tracker: https://www.bsg.ox.ac.uk/research/research-projects/coronavirus-government-response-tracker


You can call/name the interventions whatever you like. The index should be an integer though.

+----------+----------------+---------------+-------------+-------------+
| date     | school_closing | cancel_events | curfew      | ...         |
+==========+================+===============+=============+=============+
| 01.01.20 |     1          |     0         |     0       |     ...     |
+----------+----------------+---------------+-------------+-------------+
| 02.01.20 |     1          |     0         |     0       |     ...     |
+----------+----------------+---------------+-------------+-------------+
| 03.01.20 |     1          |     2         |     3       |     ...     |
+----------+----------------+---------------+-------------+-------------+
| 04.01.20 |     2          |     2         |     3       |     ...     |
+----------+----------------+---------------+-------------+-------------+
| 05.01.20 |     2          |     1         |     0       |     ...     |
+----------+----------------+---------------+-------------+-------------+
| ...      |     ...        |     ...       |     ...     |     ...     |
+----------+----------------+---------------+-------------+-------------+

- Time/Date column has to be named "date" or "time"
- Different intervention as additional columns with intervention name as column name
