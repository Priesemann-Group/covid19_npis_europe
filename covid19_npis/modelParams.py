import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import logging

log = logging.getLogger(__name__)


class ModelParams:
    """ 
        This is a class for all model parameters. It is mainly used to have a convenient
        to access data in model wide parameters e.g. start date for simulation.
        

        This class also contains the data used for fitting. `dataframe` is the original
        dataframe. `data_tensor` is a tensor in the correct shape (time x countries x age)
        with values replace by nans when no data is available.

        Parameters
        ----------
        countries: list of covid19_npis.data.Country objects
            Data objects for multiple countries
    """

    def __init__(
        self,
        countries,
        min_offset_sim_data=20,
        minimal_daily_cases=40,
        spline_degree=3,
        dtype="float32",
    ):

        self._dtype = dtype
        self._min_offset_sim_data = min_offset_sim_data
        self._minimal_daily_cases = minimal_daily_cases
        self._spline_degree = spline_degree

        # Save data objects and calculate all other variables
        self.countries = countries

        # Configs for distribution

        # Make global accessible since only one instance should be active at any time
        globals()["modelParams"] = self

    @property
    def countries(self):
        """
        Return
        ------
        :
            List of all country objects.
            See :py:class:`covid19_npis.data.Country`.
        """
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

    @property
    def knots(self):
        knots = np.arange(0, self.length, 7)
        knots = np.insert(knots, 0, [0] * self._spline_degree, axis=0)
        knots = np.insert(knots, -1, [knots[-1]] * self._spline_degree, axis=0)
        return knots

    @property
    def date_data_tensor(self):
        """
        Creates a tensor with dimension intervention, country, change_points
        Padded with 0.0 for none existing change points
        """
        max_num_cp = self.max_num_cp
        data = []
        for i, intervention in enumerate(
            self.countries[0].interventions
        ):  # Should be same across all countries -> 0
            d_c = []
            for c, country in enumerate(self.countries):
                d_cp = []
                for p, cp in enumerate(country.change_points[intervention.name]):
                    d_cp.append(self.date_to_index(cp.date_data))
                if len(d_cp) < max_num_cp:
                    d_cp.append([0.0] * (max_num_cp - len(d_cp)))
                d_c.append(d_cp)
            data.append(d_c)
        return tf.constant(data, dtype="float32")

    @property
    def gamma_data_tensor(self):
        """
        Creates a ragged tensor with dimension intervention, country, change_points
        The change points dimension can have different sizes.
        """
        max_num_cp = self.max_num_cp
        data = []
        for i, intervention in enumerate(
            self.countries[0].interventions
        ):  # Should be same across all countries -> 0
            d_c = []
            for c, country in enumerate(self.countries):
                d_cp = []
                for p, cp in enumerate(country.change_points[intervention.name]):
                    d_cp.append(cp.gamma_max)
                if len(d_cp) < max_num_cp:
                    d_cp.append([0.0] * (max_num_cp - len(d_cp)))
                d_c.append(d_cp)
            data.append(d_c)
        return tf.constant(data, dtype="float32")

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
    def num_interventions(self):
        return len(self.data_summary["interventions"])

    @property
    def num_knots(self):
        return len(self.knots) - 2 * (self._spline_degree - 1)

    @property
    def indices_begin_sim(self):
        return self._indices_begin_data

    @property
    def min_offset_sim_data(self):
        return self._min_offset_sim_data

    @property
    def length(self):
        return len(self._dataframe)

    @property
    def max_num_cp(self):
        data = []
        for i, intervention in enumerate(
            self.countries[0].interventions
        ):  # Should be same across all countries -> 0
            for c, country in enumerate(self.countries):
                index = 0
                for p, cp in enumerate(country.change_points[intervention.name]):
                    index = index + 1
                data.append(index)

        return max(data)

    # ------------------------------------------------------------------------------ #
    # Methods
    # ------------------------------------------------------------------------------ #

    def date_to_index(self, date):
        return (date - self._data_summary["begin"]).days
