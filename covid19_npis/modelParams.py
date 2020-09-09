import datetime
import numpy as np
import pandas as pd

import logging

log = logging.getLogger(__name__)


class ModelParams:
    """ 
        Class for all model parameters is used to among others add data and 
        distrbution names/dimensions to the model.
        Is also used by the plotting routines as references to distribution names etc.

        Distribution params can be changed by overwriting the defaults

        .. code-block::

            params = ModelParams()
            params.distributions["I_0"]["name"] = "my new fancy name"

        This class also contains the data used for fitting. `dataframe` is the original
        dataframe. `data_tensor` is a tensor in the correct shape (time x countries x age)
        with values replace by nans when no data is available.

        Parameters
        ----------
        countries: list of covid19_npis.data.Country objects
            Data objects for multiple countries
    """

    def __init__(
        self, countries, min_offset_sim_data=20, minimal_daily_cases=40, dtype="float32"
    ):

        self._dtype = dtype
        self._min_offset_sim_data = min_offset_sim_data
        self._minimal_daily_cases = minimal_daily_cases

        # Save data objects and calculate all other variables
        self.countries = countries

        # Configs for distribution

        # Make global accessible since only one instance should be active at any time
        globals()["modelParams"] = self

    @property
    def countries(self):
        return self._countries

    @countries.setter
    def countries(self, countries):
        """
        Every time the countries are set we want to update every other,
        data variable i.e. dataframe, data summary and data_tensor. 
        This is done here!
        """
        self._countries = countries

        """ # Update dataframe
        Join all dataframes from the country objects
        """
        for i, country in enumerate(self.countries):
            if i > 0:
                _df = _df.join(country.data_new_cases)
            else:
                _df = country.data_new_cases
        self._dataframe = _df

        # Join all interventions dataframes
        for i, country in enumerate(self.countries):
            if i > 0:
                _int = _int.join(country.data_interventions)
            else:
                _int = country.data_interventions
        self._interventions = _int

        """ # Update Data summary
        """
        data = {  # Is set on init
            "begin": _df.index.min(),
            "end": _df.index.max(),
            "age_groups": [],
            "countries": [],
            "interventions": [],
        }
        # Create countries lookup list dynamic from data dataframe
        for i in range(len(_df.columns.levels[0])):
            data["countries"].append(_df.columns.levels[0][i])
        # Create age group list dynamic from data dataframe
        for i in range(len(_df.columns.levels[1])):
            data["age_groups"].append(_df.columns.levels[1][i])
        # Create interventions list dynamic from interventions dataframe
        for i in range(len(_int.columns.levels[1])):
            data["interventions"].append(_int.columns.levels[1][i])

        self._data_summary = data

        """ # Update Data Tensor
        set data tensor, replaces values smaller than 40 by nans.
        """
        data_tensor = (
            self.dataframe.to_numpy()
            .astype(self.dtype)
            .reshape((-1, len(self.countries), len(self.age_groups)))
        )
        i_data_begin_list = []
        for c in range(data_tensor.shape[1]):
            mask = np.sum(data_tensor[:, c, :], axis=-1) > self._minimal_daily_cases
            i_data_begin = np.min(np.nonzero(mask)[0])
            i_data_begin_list.append(i_data_begin)
        i_data_begin_list = np.array(i_data_begin_list)
        i_data_begin_list = np.maximum(i_data_begin_list, self._min_offset_sim_data)
        self._indices_begin_data = i_data_begin_list
        for i in i_data_begin_list:
            data_tensor[:i] = np.nan
        self._data_tensor = data_tensor

    @property
    def dataframe(self):
        """
        New cases as multiColumn dataframe level 0 = country/region and
        level 1 = age group.
        """
        return self._dataframe

    @property
    def data_summary(self):
        """
        Data summary for all countries
        """
        return self._data_summary

    @property
    def data_tensor(self):
        """
        Tensor of input dataframe i.e. daily new cases for countries/regions
        and age groups.
        |shape| time, country, agegroup 
        """
        return self._data_tensor

    # ------------------------------------------------------------------------------ #
    # Additional properties
    # ------------------------------------------------------------------------------ #

    @property
    def dtype(self):
        return self._dtype

    @property
    def age_groups(self):
        return self.data_summary["age_groups"]

    @property
    def num_age_groups(self):
        return len(self.data_summary["age_groups"])

    @property
    def num_countries(self):
        return len(self.data_summary["countries"])

    @property
    def indices_begin_sim(self):
        return self._indices_begin_data

    @property
    def min_offset_sim_data(self):
        return self._min_offset_sim_data

    @property
    def length(self):
        return len(self._dataframe)

    # ------------------------------------------------------------------------------ #
    # Methods
    # ------------------------------------------------------------------------------ #

    def date_to_index(self, date):
        return (date - self._data_summary["begin"]).days
