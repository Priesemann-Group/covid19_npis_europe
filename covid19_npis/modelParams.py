import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
from scipy.interpolate import BSpline
import pprint

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
        offset_sim_data=20,
        minimal_daily_cases=40,
        min_offset_sim_death_data=40,
        minimal_daily_deaths=10,
        spline_degree=3,
        spline_stride=7,
        dtype="float32",
    ):
        self._dtype = dtype
        self._offset_sim_data = offset_sim_data
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

        def join_dataframes(key, check_dict, attribute_name):
            """Joins the dataframes for each country
            if the key exists in a given dictionary.

            Parameters
            ----------
            key: str
            check_dict: dict
            attribute_name: str

            Returns
            -------
            df (joined)
            """
            if not check_dict[key]:
                return None
            for i, country in enumerate(self._countries):
                if i > 0:
                    df = df.join(getattr(country, attribute_name))
                else:
                    df = getattr(country, attribute_name)
            return df

        self._countries = countries

        # Create dictionary and add existing csv files
        check = self._countries[0].exist
        for i, country in enumerate(self._countries):
            for key in country.exist:
                check[key] &= country.exist[key]
        self._check = check  # Save for data summary

        """ Update dataframes
        """
        # Positive tests
        self._dataframe_new_cases = join_dataframes(
            key="/new_cases.csv", check_dict=check, attribute_name="data_new_cases"
        )

        # Total tests
        self._dataframe_total_tests = join_dataframes(
            key="/tests.csv", check_dict=check, attribute_name="data_total_tests"
        )

        # Deaths
        self._dataframe_deaths = join_dataframes(
            key="/deaths.csv", check_dict=check, attribute_name="data_deaths"
        )

        # Population
        self._dataframe_population = join_dataframes(
            key="/population.csv", check_dict=check, attribute_name="data_population"
        )

        # Interventions
        self._dataframe_interventions = join_dataframes(
            key="/interventions.csv",
            check_dict=check,
            attribute_name="data_interventions",
        )

        """ Update data summary
        """
        self._update_data_summary()

        """ Calculate positive tests data tensor (tensorflow)
        Set data tensor, replaces values smaller than 40 by nans.
        """
        self.pos_tests_data_tensor = self._dataframe_new_cases  # Uses setter below!

        """ # Calculate total tests data tensor (tensorflow)
        Set data tensor, replaces values smaller than 40 by nans.
        """
        self.total_tests_data_tensor = self._dataframe_total_tests  # Uses setter below!

        """ # Update deaths data tensor
        set data tensor, replaces values smaller than 10 by nans.
        """
        self.deaths_data_tensor = self._dataframe_total_tests  # Uses setter below!

    # ------------------------------------------------------------------------------ #
    # Data Summary
    # ------------------------------------------------------------------------------ #
    def _update_data_summary(self):
        """# Update Data summary"""
        data = {  # Is set on init
            "data begin": self.date_data_begin,
            "data end": self.date_data_end,
            "sim begin": self.date_data_begin
            - datetime.timedelta(days=self._offset_sim_data),
            "sim end": self.date_data_end,
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
        for i in self._dataframe_interventions.columns.get_level_values(
            level="intervention"
        ).unique():
            data["interventions"].append(i)

        self._data_summary = data

    @property
    def data_summary(self):
        """
        Data summary for all countries
        """
        return self._data_summary

    def __str__(self):
        """
        Nicely converted string of the data summary if the object is printed.
        """
        return pprint.pformat(self.data_summary)

    def __repr__(self):
        return self.__str__()

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
        return tf.constant(data, dtype=self.dtype)

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
        return tf.constant(data, dtype=self.dtype)

    # ------------------------------------------------------------------------------ #
    # Positive tests
    # ------------------------------------------------------------------------------ #

    @property
    def pos_tests_dataframe(self):
        """
        New cases as multiColumn dataframe level 0 = country/region and
        level 1 = age group.
        """
        return self._dataframe_new_cases

    @property
    def pos_tests_data_tensor(self):
        """
        Tensor of daily new cases / positive tests for countries/regions
        and age groups.

        Returns
        -------
        tf.Tensor:
            |shape| time, country, agegroup
        """
        return self._tensor_pos_tests

    @pos_tests_data_tensor.setter
    def pos_tests_data_tensor(self, df):
        """
        Setter for the data tensor

        Parameters
        ----------
        df: pd.DataFrame
            Positive tests dataframe
        """
        new_cases_tensor = (
            df.to_numpy()
            .astype(self.dtype)
            .reshape((-1, len(self.countries), len(self.age_groups)))
        )
        new_cases_tensor = np.concatenate(
            [
                np.empty(
                    (self._offset_sim_data, len(self.countries), len(self.age_groups))
                ),
                new_cases_tensor,
            ]
        )
        i_data_begin_list = []
        for c in range(new_cases_tensor.shape[1]):
            mask = (
                np.sum(new_cases_tensor[:, c, :], axis=-1) > self._minimal_daily_cases
            )
            i_data_begin = np.min(np.nonzero(mask)[0])
            i_data_begin_list.append(i_data_begin)

        i_data_begin_list = np.array(i_data_begin_list)
        i_data_begin_list = np.maximum(i_data_begin_list, self._offset_sim_data)
        self._indices_begin_data = i_data_begin_list

        for c, i in enumerate(self._indices_begin_data):
            new_cases_tensor[:i, c, :] = np.nan

        self._tensor_pos_tests = tf.constant(new_cases_tensor, dtype=self.dtype)

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
        Returns
        -------
        tf.Tensor:
            |shape| time, country
        """
        return self._tensor_total_tests

    @total_tests_data_tensor.setter
    def total_tests_data_tensor(self, df):
        """
        Setter for the total tests data tensor

        Parameters
        ----------
        df: pd.DataFrame
            Total tests dataframe
        """
        total_tests_tensor = (
            self._dataframe_total_tests.to_numpy()
            .astype(self.dtype)
            .reshape((-1, len(self.countries)))
        )
        total_tests_tensor = np.concatenate(
            [
                np.empty((self._offset_sim_data, len(self.countries))),
                total_tests_tensor,
            ]
        )
        for c, i in enumerate(self._indices_begin_data):
            total_tests_tensor[:i, c] = np.nan
        self._tensor_total_tests = tf.constant(total_tests_tensor, dtype=self.dtype)

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
        Returns
        -------
        tf.Tensor:
            |shape| time, country
        """
        return self._tensor_deaths

    @deaths_data_tensor.setter
    def deaths_data_tensor(self, df):
        """
        Setter for the deaths data tensor

        Parameters
        ----------
        df: pd.DataFrame
            Deaths tests dataframe
        """
        deaths_tensor = (
            df.to_numpy()
            .astype(self.dtype)
            .reshape((-1, len(self.countries)))  ## assumes non-age-stratified data
        )
        deaths_tensor = np.concatenate(
            [
                np.empty((self._offset_sim_data, len(self.countries))),
                deaths_tensor,
            ]
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
        self._indices_begin_data_deaths = np.maximum(
            i_data_begin_list, self._indices_begin_data
        )
        for c, i in enumerate(self._indices_begin_data_deaths):
            deaths_tensor[:i, c] = np.nan

        self._tensor_deaths = tf.constant(deaths_tensor, dtype=self.dtype)

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
    def offset_sim_data(self):
        return self._offset_sim_data

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
    def indices_begin_data(self):
        """
        Returns the index of every country when the first case is reported. It could
        be that for some countries, the index is later than self.offset_sim_data.
        """
        return self._indices_begin_data

    @property
    def length_data(self):
        """
        Returns
        -------
        :number
            Length of the inserted/loaded data in days
        """
        return len(self._dataframe_new_cases)

    @property
    def length_sim(self):
        """
        Returns
        -------
        :number
            Length of the simulation in days.
        """
        return len(self._tensor_pos_tests)

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
    def date_data_begin(self):
        return self.pos_tests_dataframe.index.min()

    @property
    def date_sim_begin(self):
        return self.pos_tests_dataframe.index.min() - datetime.timedelta(
            days=self._offset_sim_data
        )

    @property
    def date_data_end(self):
        return self.pos_tests_dataframe.index.max()

    @property
    def spline_basis(self):
        """
        Calculates B-spline basis.

        Return
        ------
        |shape| modelParams.length_sim, modelParams.num_splines
        """
        stride = self._spline_stride
        degree = self._spline_degree
        knots = np.arange(
            self.length_sim + degree * stride, 0 - (degree + 1) * stride, -stride
        )
        knots = knots[::-1]
        num_splines = len(knots) - 2 * (degree - 1)
        spl = BSpline(knots, np.eye(num_splines), degree, extrapolate=False)
        spline_basis = spl(np.arange(0, self.length_sim))
        return spline_basis

    # ------------------------------------------------------------------------------ #
    # Other Methods
    # ------------------------------------------------------------------------------ #

    def date_to_index(self, date):
        return (date - self.date_data_begin).days + self.offset_sim_data

    def get_weekdays(self):
        self._weekdays_data_tensor = tf.constant(
            pd.date_range(start=self.date_data_begin, end=self.date_data_begin).weekday,
            tf.float32,
        )
        return self._weekdays_data_tensor
