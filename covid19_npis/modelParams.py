import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
from scipy.interpolate import BSpline

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
        min_offset_sim_death_data=40,
        minimal_daily_deaths=40,
        spline_degree=3,
        spline_stride=7,
        dtype="float32",
    ):

        self._dtype = dtype
        self._min_offset_sim_data = min_offset_sim_data
        self._minimal_daily_cases = minimal_daily_cases
        self._min_offset_sim_death_data = min_offset_sim_death_data
        self._minimal_daily_deaths = minimal_daily_deaths
        self._spline_degree = spline_degree
        self._spline_stride = spline_stride

        # Save data objects and calculate all other variables
        self.countries = countries

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
        # Get if csv exits:
        check = self.countries[0].exist
        for i, country in enumerate(self.countries):
            for key in country.exist:
                check[key] &= country.exist[key]

        # Join positive tests
        if check["/new_cases.csv"]:
            for i, country in enumerate(self.countries):
                if i > 0:
                    _df = _df.join(country.data_new_cases)
                else:
                    _df = country.data_new_cases
            self._dataframe = _df
        else:
            log.error("Supply /new_cases.csv!")

        # Join total tests
        if check["/tests.csv"]:
            for i, country in enumerate(self.countries):
                if i > 0:
                    _df_t = _df_t.join(country.data_total_tests)
                else:
                    _df_t = country.data_total_tests
            self._dataframe_total_tests = _df_t
        else:
            self._dataframe_total_tests = None

        # Join deaths
        if check["/deaths.csv"]:
            for i, country in enumerate(self.countries):
                if i > 0:
                    _df_d = _df_d.join(country.data_deaths)
                else:
                    _df_d = country.data_deaths
            self._dataframe_deaths = _df_d
        else:
            self._dataframe_deaths = None

        # Join population
        if check["/population.csv"]:
            for i, country in enumerate(self.countries):
                if i > 0:
                    _df_p = _df_p.join(country.data_population)
                else:
                    _df_p = country.data_population
            self._dataframe_population = _df_p
        else:
            self._dataframe_population = None

        # Join all interventions dataframes
        if check["/interventions.csv"]:
            for i, country in enumerate(self.countries):
                if i > 0:
                    _int = _int.join(country.data_interventions)
                else:
                    _int = country.data_interventions
            self._interventions = _int
        else:
            log.error("Supply /interventions.csv!")
        self._check = check  # For data summary
        """ # Update data tensor
        """
        self._update_data_summary()

        """ # Update positive test data tensor/df
        set data tensor, replaces values smaller than 40 by nans.
        """
        data_tensor = (
            self._dataframe.to_numpy()
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

        """ # Update deaths data tensor/df
        set data tensor, replaces values smaller than 40 by nans.
        """
        deaths_tensor = (
            self._dataframe_deaths.to_numpy()
            .astype(self.dtype)
            .reshape((-1, len(self.countries)))  ## assumes non-age-stratified data
        )
        i_data_begin_list = []
        for c in range(deaths_tensor.shape[1]):
            mask = deaths_tensor[:, c] > self._minimal_daily_deaths
            i_data_begin = np.min(np.nonzero(mask)[0])
            i_data_begin_list.append(i_data_begin)
        i_data_begin_list = np.array(i_data_begin_list)
        i_data_begin_list = np.maximum(
            i_data_begin_list, self._min_offset_sim_death_data
        )
        self._indices_begin_data = np.maximum(
            i_data_begin_list, self._indices_begin_data
        )

        for i in i_data_begin_list:
            data_tensor[:i] = np.nan
            deaths_tensor[:i] = np.nan
        self._pos_tests_data_tensor = data_tensor
        self._deaths_data_tensor = deaths_tensor

    def _update_data_summary(self):
        """# Update Data summary"""
        data = {  # Is set on init
            "begin": self.data_begin,
            "end": self.data_end,
            "age_groups": [],
            "countries": [],
            "interventions": [],
            "files": self._check,
        }
        # Create countries lookup list dynamic from data dataframe
        for country_name in self.pos_tests_dataframe.columns.get_level_values(
            level="country"
        ).unique():
            data["countries"].append(country_name)
        # Create age group list dynamic from data dataframe
        for age_group_name in self.pos_tests_dataframe.columns.get_level_values(
            level="age_group"
        ).unique():
            data["age_groups"].append(age_group_name)
        # Create interventions list dynamic from interventions dataframe
        for i in self._interventions.columns.get_level_values(
            level="intervention"
        ).unique():
            data["interventions"].append(i)

        self._data_summary = data

    @property
    def spline_basis(self):
        stride = self._spline_stride
        degree = self._spline_degree
        knots = np.arange(
            self.length + degree * stride, 0 - (degree + 1) * stride, -stride
        )
        knots = knots[::-1]
        num_splines = len(knots) - 2 * (degree - 1)
        spl = BSpline(knots, np.eye(num_splines), degree, extrapolate=False)
        spline_basis = spl(np.arange(0, self.length))
        return spline_basis  # shape : modelParams.length x modelParams.num_splines

    @property
    def data_summary(self):
        """
        Data summary for all countries
        """
        return self._data_summary

    # ------------------------------------------------------------------------------ #
    # Interventions
    # ------------------------------------------------------------------------------ #
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
                    for x in range(max_num_cp - len(d_cp)):
                        d_cp.append(0.0)
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
                    for x in range(max_num_cp - len(d_cp)):
                        d_cp.append(0.0)
                d_c.append(d_cp)
            data.append(d_c)
        return tf.constant(data, dtype="float32")

    # ------------------------------------------------------------------------------ #
    # Positive tests
    # ------------------------------------------------------------------------------ #
    @property
    def pos_tests_dataframe(self):
        """
        New cases as multiColumn dataframe level 0 = country/region and
        level 1 = age group.
        """
        return self._dataframe

    @property
    def pos_tests_data_tensor(self):
        """
        Tensor of daily new cases / positive tests for countries/regions
        and age groups.
        |shape| time, country, agegroup
        """
        return self._pos_tests_data_tensor.astype(self.dtype)

    # ------------------------------------------------------------------------------ #
    # Total tests
    # ------------------------------------------------------------------------------ #
    @property
    def total_tests_dataframe(self):
        """
        Dataframe of total tests in all countries. Datetime index and country columns
        as Multiindex.
        """
        return self._dataframe_total_tests

    @property
    def total_tests_data_tensor(self):
        """
        |shape| time, country
        """
        return self._dataframe_total_tests.to_numpy().astype(self.dtype)

    # ------------------------------------------------------------------------------ #
    # Number of deaths
    # ------------------------------------------------------------------------------ #
    @property
    def deaths_dataframe(self):
        """
        Dataframe of deaths in all countries. Datetime index and country columns
        as Multiindex.
        """
        return self._dataframe_deaths

    @property
    def deaths_data_tensor(self):
        """

        |shape| time, country
        """
        return self._deaths_data_tensor.astype(self.dtype)

    # ------------------------------------------------------------------------------ #
    # Population
    # ------------------------------------------------------------------------------ #
    @property
    def N_dataframe(self):
        """
        Dataframe of population in all countries. Datetime index and country columns
        as Multiindex.
        """
        return self._dataframe_population

    @property
    def N_data_tensor(self):
        """
        Creates the population tensor with
        automatically calculated age strata/brackets.
        |shape| country, age_groups
        """
        data = []
        for c, country in enumerate(self.countries):
            d_c = []

            # Get real age groups from country config
            age_dict = country.age_groups

            for age_group in self.age_groups:
                # Select age range from config and sum over it
                lower, upper = age_dict[age_group]
                d_c.append(country.data_population[lower:upper].sum().values[0])
            data.append(d_c)
        return tf.constant(data, dtype="float32")

    @property
    def N_data_tensor_total(self):
        """
        Creates the population tensor for every age.
        |shape| country, age
        """
        data = []
        for c, country in enumerate(self.countries):
            data.append(country.data_population.values[:, 0].tolist())
        return tf.constant(data, dtype="float32", shape=[self.num_countries, 101])

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
    def num_splines(self):
        return self.spline_basis.shape[1]

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

    @property
    def data_begin(self):
        return self.pos_tests_dataframe.index.min()

    @property
    def data_end(self):
        return self.pos_tests_dataframe.index.max()

    # ------------------------------------------------------------------------------ #
    # Methods
    # ------------------------------------------------------------------------------ #

    def date_to_index(self, date):
        return (date - self._data_summary["begin"]).days
