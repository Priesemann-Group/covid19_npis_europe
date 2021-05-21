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
    countries:  list, :py:class:`covid19_npis.data.Country`
        Data objects for multiple countries
    """

    def __init__(
        self,
        countries,
        const_contact=True,  # set 'true' for a constant contact matrix (without age-group interaction)
        R_interval_time=5,  # time interval over which the reproduction number is calculated
        offset_sim_data=20,
        minimal_daily_cases=40,
        min_offset_sim_death_data=40,
        minimal_daily_deaths=10,
        spline_degree=3,
        spline_stride=7,
        dtype="float32",
    ):
        self._dtype = dtype
        self._const_contact = const_contact
        self._R_interval_time = R_interval_time
        self._offset_sim_data = offset_sim_data
        self._minimal_daily_cases = minimal_daily_cases
        self._min_offset_sim_death_data = min_offset_sim_death_data
        self._minimal_daily_deaths = minimal_daily_deaths
        self._spline_degree = spline_degree
        self._spline_stride = spline_stride
        self._indices_begin_data = None
        # Save data objects and calculate all other variables
        self.countries = countries

        # Make global accessible since only one instance should be active at any time
        globals()["modelParams"] = self

    @classmethod
    def from_folder(cls, fpath, **kwargs):
        """
        Create modelParams class from folder containing differet regions or countrys
        """
        import os
        from .data import Country

        c = []
        for entry in os.scandir(fpath):
            if os.path.isdir(entry):
                c.append(Country(entry.path))
        return cls(countries=c, **kwargs)

    @property
    def countries(self):
        """
        Data objectes for each country.

        Return
        ------
        :
           List of all country object
        """
        return self._countries

    def country_by_name(self, name):
        """
        Returns country by name
        """
        for country in self._countries:
            if country.name == name:
                return country

        # Error
        raise Error("Name not found in country list")

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

            # df = None
            # df_total = pd.DataFrame()
            for i, country in enumerate(self._countries):
                # if ((not accumulate) | (accumulate & (len(getattr(country,attribute_name).columns)>1))):
                if i > 0:
                    df = df.join(getattr(country, attribute_name))
                else:
                    df = getattr(country, attribute_name)
                # if accumulate:
                # df_total[country.name] = getattr(country, attribute_name).sum(axis=1)
            return df  # , df_total

        # sort countries alphabetically to have consistent indexes
        c_sort = np.argsort([c.name for c in countries])
        self._countries = []
        for c in c_sort:
            self._countries.append(countries[c])

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
        # log.info(f'data_summary:\n{self.data_summary}')

        # self._adjust_stratification(attributes=['_dataframe_new_cases'])

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
        self.deaths_data_tensor = self._dataframe_deaths  # Uses setter below!

        self._set_data_mask()  # prepare data masks for use in likelihood computation

        """ # Update intervetions data tensor
        """
        self.date_data_tensor = self._dataframe_interventions  # Uses setter below!
        self.gamma_data_tensor = self._dataframe_interventions

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
            "age_groups_ref": [],
            "age_groups_summarized": [],
            "countries": [],
            "interventions": [],
            "files": self._check,
        }
        # Create countries lookup list dynamic from data dataframe
        # for country_name in sorted(self.pos_tests_dataframe.columns.get_level_values(level='country').unique()):
        for country in self._countries:
            country_name = country.name
            # age_groups = count
            data["countries"].append(country_name)
            ### added
            age_groups = self.pos_tests_dataframe[country_name].columns
            if len(age_groups) > 1:
                if len(data["age_groups"]):
                    assert len(data["age_groups"]) == len(
                        age_groups
                    ), "data with different number of age groups provided - please provide either similiar age stratification, or summarized data"
                else:
                    data["age_groups"] = list(age_groups)
                    data["age_groups_ref"] = country.age_groups
                data["age_groups_summarized"].append(False)
            else:
                data["age_groups_summarized"].append(True)
        data["age_groups_summarized"] = np.array(data["age_groups_summarized"])
        # data["age_group_data"][country_name] = list(self.pos_tests_dataframe[country_name].columns)
        # Create age group list dynamic from data dataframe
        # for age_group_name in self.pos_tests_dataframe.columns.get_level_values(
        #     level="age_group"
        # ).unique():
        #     data["age_groups"].append(age_group_name)
        # Create interventions list dynamic from interventions dataframe
        for i in self._dataframe_interventions.columns.get_level_values(
            level="intervention"
        ).unique():
            data["interventions"].append(i)

        self._data_summary = data

    @property
    def data_summary(self):
        """
        Data summary for modelParams object.
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
        return self._date_data_tensor

    @property
    def gamma_data_tensor(self):
        """
        Creates a ragged tensor with dimension intervention, country, change_points
        The change points dimension can have different sizes.
        """
        return self._gamma_data_tensor

    @gamma_data_tensor.setter
    def gamma_data_tensor(self, df):
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
        self._gamma_data_tensor = tf.constant(data, dtype=self.dtype)

    @date_data_tensor.setter
    def date_data_tensor(self, df):
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
        self._date_data_tensor = tf.constant(data, dtype=self.dtype)

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
        tf.Tensor
            |shape| time, country, agegroup
        """
        return self._tensor_pos_tests

    @property
    def pos_tests_total_data_tensor(self):
        """
        Tensor of daily new cases / positive tests for countries/regions
        and age groups.

        Returns
        -------
        tf.Tensor
            |shape| time, country, agegroup
        """
        return self._tensor_pos_tests_total

    # @property
    # def pos_tests_data_array(self):
    #     """
    #     Numpy Array of daily new cases / positive tests for countries/regions
    #     and age groups.
    #
    #     Returns
    #     -------
    #     tf.Tensor
    #         |shape| time, country, agegroup
    #     """
    #     return self._array_pos_tests.astype(self.dtype)

    @pos_tests_data_tensor.setter
    def pos_tests_data_tensor(self, df):
        """
        Setter for the data tensor

        Parameters
        ----------
        df: pd.DataFrame
            Positive tests dataframe
        """

        # create tensor with stratified (provided or artificial) case data for all countries
        new_cases = np.zeros(
            (self.pos_tests_dataframe.shape[0], 0, self.num_age_groups)
        )
        for i, c in enumerate(self.countries):
            new_cases_tmp = self.pos_tests_dataframe[c.name].to_numpy()
            if self.data_summary["age_groups_summarized"][
                i
            ]:  ## write existing data into array
                new_cases_tmp = np.repeat(
                    new_cases_tmp / self.num_age_groups, self.num_age_groups, axis=1
                )  ## could be further refined according to demographics - but not suuuper important

            new_cases = np.concatenate(
                [new_cases, new_cases_tmp[:, np.newaxis, :]], axis=1
            )

        # prepend data with zeros for simulation offset
        new_cases = np.concatenate(
            [
                np.zeros(
                    (self._offset_sim_data, self.num_countries, self.num_age_groups)
                ),
                new_cases,
            ],
            axis=0,
        )

        i_data_begin_list = []
        for c in range(new_cases.shape[1]):
            mask = np.nansum(new_cases[:, c, :], axis=-1) > self._minimal_daily_cases
            if mask.sum() == 0:  # [False,False,False]
                i_data_begin = len(mask) - 1
            else:
                i_data_begin = np.min(np.nonzero(mask)[0])
            i_data_begin_list.append(i_data_begin)
        i_data_begin_list = np.array(i_data_begin_list)
        # i_data_begin_list = np.maximum(i_data_begin_list, self._offset_sim_data)
        self._indices_begin_data = i_data_begin_list

        for c, i in enumerate(self.indices_begin_data):
            new_cases[:i, c, :] = np.nan

        self._tensor_pos_tests = tf.constant(new_cases, dtype=self.dtype)
        self._tensor_pos_tests_total = tf.constant(
            new_cases.sum(axis=-1), dtype=self.dtype
        )

        # self._tensor_pos_tests = tf.constant(new_cases_tensor, dtype=self.dtype)

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
        if not self.countries[0].exist["/tests.csv"]:
            self._tensor_total_tests = None
            return
        total_tests_tensor = (
            self._dataframe_total_tests.to_numpy()
            .astype(self.dtype)
            .reshape((-1, len(self.countries)))
        )
        total_tests_tensor = np.concatenate(
            [
                np.zeros((self._offset_sim_data, len(self.countries))),
                total_tests_tensor,
            ]
        )
        for c, i in enumerate(self.indices_begin_data):
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
        if len(df.columns.names) == 1:
            deaths_tensor = (
                df.to_numpy()
                .astype(self.dtype)
                .reshape((-1, len(self.countries)))  ## assumes non-age-stratified data
            )
            deaths_tensor = np.concatenate(
                [
                    np.zeros((self._offset_sim_data, len(self.countries),)),
                    deaths_tensor,
                ]
            )

        if len(df.columns.names) == 2:
            deaths_tensor = (
                df.to_numpy()
                .T.astype(self.dtype)
                .reshape(
                    (-1, len(self.countries), len(self.age_groups))
                )  ## assumes non-age-stratified data
            )

            deaths_tensor = np.concatenate(
                [
                    np.zeros(
                        (
                            self._offset_sim_data,
                            len(self.countries),
                            len(self.age_groups),
                        )
                    ),
                    deaths_tensor,
                ]
            )
        i_data_begin_list = []
        for c in range(deaths_tensor.shape[1]):
            mask = deaths_tensor[:, c] > self._minimal_daily_deaths
            if mask.sum() == 0:  # [False,False,False]
                i_data_begin = len(mask) - 1
            else:
                i_data_begin = np.min(np.nonzero(mask)[0])
            i_data_begin_list.append(i_data_begin)
        i_data_begin_list = np.array(i_data_begin_list)
        i_data_begin_list = np.maximum(
            i_data_begin_list, self._min_offset_sim_death_data
        )
        self._indices_begin_data_deaths = np.maximum(
            i_data_begin_list, self.indices_begin_data
        )
        for c, i in enumerate(self._indices_begin_data_deaths):
            deaths_tensor[:i, c] = np.nan

        self._tensor_deaths = tf.constant(deaths_tensor, dtype=self.dtype)

    # ------------------------------------------------------------------------------ #
    # masks
    # ------------------------------------------------------------------------------ #
    def _set_data_mask(self):

        self.data_stratified_mask = np.argwhere(
            (
                ~np.isnan(self.pos_tests_data_tensor)
                & ~self.data_summary["age_groups_summarized"][np.newaxis, :, np.newaxis]
            ).flatten()
        )
        self.data_summarized_mask = np.argwhere(
            (
                ~np.isnan(self.pos_tests_total_data_tensor)
                & self.data_summary["age_groups_summarized"][np.newaxis, :]
            ).flatten()
        )

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

            # Get age groups from country config
            if self.data_summary["age_groups_summarized"][c]:
                # If age group not present in this country data
                age_groups_c = self.data_summary["age_groups_ref"]
            else:
                age_groups_c = country.age_groups

            for age_group in self.age_groups:
                # Select age range from config and sum over it
                lower, upper = age_groups_c[age_group]
                d_c.append(country.data_population[lower:upper].sum().values[0])

            data.append(d_c)
        return tf.constant(data, dtype=tf.float32)

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
        return len(np.unique(self.data_summary["age_groups"]))
        # return len(self.data_summary["age_groups"])

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
        if self._indices_begin_data is None:
            self.pos_tests_data_tensor = self.pos_tests_dataframe
        return self._indices_begin_data

    @property
    def length_data(self):
        """
        Returns
        -------
        :number
            Length of the inserted/loaded data in days
        """
        # return len(self._dataframe_new_cases)
        # return tf.size(self._dataframe_new_cases)
        return self._dataframe_new_cases.shape[-2]

    @property
    def length_sim(self):
        """
        Returns
        -------
        :number
            Length of the simulation in days.
        """
        # return len(self._tensor_pos_tests)
        return self._tensor_pos_tests.shape[-3]

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

    def _make_global(self):
        """
        Run once if you want to make the modelParams global. Used in plotting
        """
        globals()["modelParams"] = self
