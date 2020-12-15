import logging
import pandas as pd
import os
import json

log = logging.getLogger(__name__)
from . import modelParams

from .plot.utils import check_for_shape_label, get_shape_from_dataframe


def convert_trace_to_dataframe_list(trace, sample_state):
    r"""
    Converts the pymc4 arviz trace to multiple pandas dataframes.
    Also sets the right labels for the  dimensions i.e splits data by
    country and age group.

    Do not look too much into this function if you want to keep your sanity!

    Parameters
    ----------
    trace: arivz InferenceData

    sample_state: pymc4 sample state

    Returns
    -------
    list of pd.DataFrame
        Multiindex dataframe containing all samples by chain and other dimensions
        defined in config.py

    """

    # Try to get posterior and prior data
    data = []
    if hasattr(trace, "posterior"):
        log.info("Found posterior in trace.")
        data.append(trace.posterior.data_vars)
    if hasattr(trace, "prior_predictive"):
        log.info("Found prior_predictive in trace.")
        data.append(trace.prior_predictive.data_vars)

    # Get list of all distributions
    dists = [key for key in sample_state.distributions]
    determs = [key for key in sample_state.deterministics]

    # Create dataframe from xarray
    dfs = []
    for d in data:
        for dist in dists:
            dfs.append(convert_trace_to_dataframe(trace, sample_state, dist))
        for deter in determs:
            dfs.append(convert_trace_to_dataframe(trace, sample_state, deter))
    return dfs


def convert_trace_to_dataframe(trace, sample_state, key, data_type=None):
    r"""
    Converts the pymc4 arviz trace for a single key to a pandas dataframes.
    Also sets the right labels for the  dimensions i.e splits data by
    country and age group.

    Do not look too much into this function if you want to keep your sanity!

    Parameters
    ----------
    trace: arivz InferenceData

    sample_state: pymc4 sample state

    key: str
        Name of variable in modelParams

    data_type: str
        Type of trace, gets detected automatically normally.
        Possible values are: "posterior", "prior_predictive", "posterior_predictive".
        Overwrites automatic behaviour!
        default: None
    Returns
    -------
    pd.DataFrame
        Multiindex dataframe containing all samples by chain and other dimensions
        defined in modelParams.py

    """
    # Try to get posterior and prior data
    data = []
    if data_type is None:
        if hasattr(trace, "posterior"):
            data_type = "posterior"
        if hasattr(trace, "prior_predictive"):
            data_type = "prior_predictive"

    if data_type == "posterior":
        log.info("Found posterior in trace.")
        data = trace.posterior.data_vars
    elif data_type == "prior_predictive":
        log.info("Found prior_predictive in trace.")
        data = trace.prior_predictive.data_vars
    elif data_type == "posterior_predictive":
        log.info("Using posterior_predictive from trace as data!")
        data = trace.posterior_predictive.data_vars

    # Get model and var names a bit hacky but works
    for var in data:
        model_name, var_name = var.split("/")
        break

    # Check key value
    dists_and_determs = [
        dist.split("/")[1] for dist in sample_state.continuous_distributions
    ]
    for deter in sample_state.deterministics:
        dists_and_determs.append(deter.split("/")[1])

    assert key in dists_and_determs, f"Key '{key}' not found! Check for typos."

    # Get distribution
    try:
        dist = sample_state.continuous_distributions[model_name + "/" + key]
    except Exception as e:
        dist = sample_state.deterministics[model_name + "/" + key]

    # Check if it has shape and shape_label
    check_for_shape_label(dist)
    # convert to dataframe
    df = data[f"{model_name}/{dist.name}"].to_dataframe()

    num_of_levels = len(df.index.levels)
    # Rename level to dimension labels
    """
    0 is always chain
    1 is always sample
    ... somthing else sometimes
    -3 is modelParams.distribution...shape[0]
    -2 is modelParams.distribution...shape[1]
    -1 is modelParams.distribution...shape[2]

    The last 3 can shift up to the number of labels 
    """
    ndim = len(get_shape_from_dataframe(df))

    # Rename dimensions if shape labels are present
    if hasattr(dist, "shape_label"):
        for i in range(ndim):
            if isinstance(dist.shape_label, (list, tuple,),):
                label = dist.shape_label[i]
            else:
                label = dist.shape_label
            if label is None:
                # Drop each level with "None" shape label
                df.index = df.index.droplevel(i)
            else:
                df.index.rename(
                    label, level=i - ndim, inplace=True,
                )

    # Rename country index to country names
    if r"country" in df.index.names:
        df.index = df.index.set_levels(
            modelParams.modelParams.data_summary["countries"], level="country"
        )

    # Rename age_group index to age_group names
    if r"age_group" in df.index.names:
        df.index = df.index.set_levels(
            modelParams.modelParams.data_summary["age_groups"], level="age_group"
        )
    if r"age_group_i" in df.index.names:
        df.index = df.index.set_levels(
            modelParams.modelParams.data_summary["age_groups"], level="age_group_i"
        )
    if r"age_group_j" in df.index.names:
        df.index = df.index.set_levels(
            modelParams.modelParams.data_summary["age_groups"], level="age_group_j"
        )

    if r"intervention" in df.index.names:
        df.index = df.index.set_levels(
            modelParams.modelParams.data_summary["interventions"], level="intervention"
        )

    # Convert time index to datetime starting at model begin
    if r"time" in df.index.names:
        df.index = df.index.set_levels(
            pd.date_range(
                modelParams.modelParams.date_sim_begin,
                modelParams.modelParams.date_data_end,
            ),
            level="time",
        )

    # Last rename column
    df = df.rename(columns={f"{model_name}/{dist.name}": dist.name})

    return df


def select_from_dataframe(df, axis=0, **kwargs):
    for key, value in kwargs.items():
        df = df.xs(value, level=key, axis=axis)
    return df


class Country(object):
    """Country data class!
    Contains death, new_cases/positive tests, daily tests, interventions and config data for a specific country.
    Retrieves this data from a gives folder. There are the following specifications for the data:

    - new_cases.csv
        - Time/Date column has to be named "date" or "time"
        - Age group columns have to be named consistent between different data and countries
    - interventions.csv
        - Time/Date column has to be named "date" or "time"
        - Different intervention as additional columns with intervention name as column name
    - tests.csv
        - Time/Date column has to be named "date" or "time"
        - Daily performed tests column with name "tests"
    - deaths.csv
        - Time/Date column has to be named "date" or "time"
        - Daily deaths column has to be named "deaths"
        - Optional: Daily deaths per age group same column names as in new_cases
    - population.csv
        - Age column named "age"
        - Column Number of people per age named "PopTotal"
    - config.json, dict:
        - name : "country_name"
        - age_groups : dict
            - "column_name" : [age_lower, age_upper]


    Also calculates change points and interventions automatically on init.

    Parameters
    ----------
    path_to_folder : string
        Filepath to the folder, which holds all the data for the country!
        Should be something like "../data/Germany".
        That is new_cases.csv, interventions.csv, population.csv

    """

    """
    Class attribute for the interventions, get set for all instances
    of the country class!
    list of intervention objects
    """
    interventions = []

    def __init__(self, path_to_folder):
        self.path_to_folder = path_to_folder

        # Check if files exist
        self.exist = self.__check_files_exist()

        # Load file if it exists
        self.__load_files(self.exist)

        self.__check_for_age_group_names()

        # Create change_points from interventions time series
        self.change_points = {}
        for column in self.data_interventions.columns:
            self.change_points.update(
                self.create_change_points(self.data_interventions[column])
            )

    def __check_files_exist(self):
        def helper(file):
            if os.path.isfile(self.path_to_folder + file):
                return True
            else:
                log.warning(
                    f"Could not find {self.path_to_folder + file} file. Trying to continue without it!"
                )
                return False

        files = [
            "/new_cases.csv",
            "/tests.csv",
            "/deaths.csv",
            "/interventions.csv",
            "/population.csv",
            "/config.json",
        ]
        ret = {}
        for file in files:
            ret[file] = helper(file)
        return ret

    def __load_files(self, exist):
        """
        Loads files if they exist
        """
        if exist["/config.json"]:
            with open(self.path_to_folder + "/config.json") as json_file:
                data = json.load(json_file)
                self.name = data["name"]
                self.age_groups = data["age_groups"]
        else:
            self.name = "Fix config"
            self.age_groups = {}

        if exist["/new_cases.csv"]:
            self.data_new_cases = self._to_iso(
                self._load_csv_with_date_index(self.path_to_folder + "/new_cases.csv"),
                "age_group",
            )
        else:
            self.data_new_cases = None

        if exist["/tests.csv"]:
            self.data_total_tests = self._to_iso(
                self._load_csv_with_date_index(self.path_to_folder + "/tests.csv"),
            )
        else:
            self.data_total_tests = None

        if exist["/deaths.csv"]:
            self.data_deaths = self._to_iso(
                self._load_csv_with_date_index(self.path_to_folder + "/deaths.csv"),
            )
        else:
            self.data_deaths = None

        if exist["/interventions.csv"]:
            self.data_interventions = self._to_iso(
                self._load_csv_with_date_index(
                    self.path_to_folder + "/interventions.csv"
                ),
                "intervention",
            )
        else:
            self.data_interventions = None

        if exist["/population.csv"]:
            self.data_population = self._to_iso(
                pd.read_csv(self.path_to_folder + "/population.csv", index_col="age")
            )
        else:
            self.data_population = None

        log.info(f"Loaded data for {self.name}.")

    def __check_for_age_group_names(self):
        for age_group in self.age_groups:
            if age_group not in self.data_new_cases.columns.get_level_values(
                level="age_group"
            ):
                message = "Age group missmatch in config.json and new_cases.csv!"
                raise NameError(
                    f"{self.name}: {message} {self.path_to_folder} {age_group}"
                )

    def _load_csv_with_date_index(self, filepath):
        """
        Loads csv file with date column
        """
        data = pd.read_csv(filepath)
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"], format="%d.%m.%y")
            data = data.set_index("date")
        elif "time" in data.columns:
            data["date"] = pd.to_datetime(data["time"], format="%d.%m.%y")
            data = data.set_index("date")

        return data

    def _to_iso(self, df, name=None):
        """
        Create multicolumn from normal columns with country at level 0
        and name for level 1. Or only constructs one level if no name is supplied.

        Parameters
        ----------
        df : pandas.DataFrame

        name: str, optional
        """

        if name is None:
            if len(df.columns) != 1:
                log.warning(f"Multiple columns found in {df.name}! Using first one!")
            df.columns = pd.MultiIndex(
                levels=[[self.name,],], codes=[[0,],], names=["country"],
            )
            return df

        cols = []
        for column in df.columns:
            cols.append((self.name, column))
        df.columns = pd.MultiIndex.from_tuples(cols, names=["country", name])
        return df

    def create_change_points(self, df):
        """
        Create change points for a single intervention and also adds
        interventions if they do not exist yet.

        Parameters
        ----------
        df : pandas.DataFrame
            Single intervention column with datetime index.

        Returns
        -------
        :
            Change points dict :code:`{name:[cps]}`
        """
        # Add intervention also checks if it exists
        num_stages = df.max()  # 0,1,2,3 -> 4
        self.add_intervention(df.name[1], num_stages)

        # Get intervention
        interv = self.get_intervention_by_name(df.name[1])

        """# Generate change points
        - iterate over time in df and detect changes in value
        - if change is positive: gamma_max = num/stages
        """
        change_points = []
        previous_value = df[0]
        for row in range(1, len(df)):
            # Calc delta:
            delta = df[row] - previous_value
            if delta != delta:
                log.error(f"Found nan in intervetions! {self.name}")
            if delta != 0:
                change_points.append(
                    Change_point(
                        date_data=df.index[row],
                        gamma_max=delta
                        / (interv.num_stages),  # +1 because it starts at 0
                    )
                )
            # Set new previous value
            previous_value = df[row]

        cp_dict = {interv.name: change_points}
        return cp_dict

    @classmethod
    def add_intervention(cls, name, num_stages):
        """
        Constructs and adds intervention to the class attributes if that
        intervention does not exist yet! This is done by name check.

        Parameters
        ----------
        name : string
            Name of the intervention

        time_series : pandas.DataFrame
            Intervention indexs as time series with datetime index!
        """

        # Break if intervention does already exist!
        for interv in cls.interventions:
            if name == interv.name:
                # Update number of stages (maybe there are more stages)
                if num_stages > interv.num_stages:
                    interv.num_stages = num_stages
                return

        # construct intervention
        intervention = Intervention(name, num_stages)
        # Add to interventions class attribute
        cls.interventions.append(intervention)

    @classmethod
    def set_intervention_alpha_prior(cls, name, prior_loc, prior_scale):
        """
        Manual set prior for effectivity alpha for a intervention via the name.
        That is it set prior_alpha_loc and prior_alpha_scale of a Intervention instance.

        Parameters
        ----------
        name: string
            Name of intervention

        prior_loc : number

        prior_scale: number
        """

        for interv in cls.interventions:
            if name == interv.name:
                interv.prior_alpha_loc = prior_loc
                interv.prior_alpha_scale = prior_scale
                return

    @classmethod
    def get_intervention_by_name(cls, name):
        """
        Gets intervention from interventions array via name

        Returns
        -------
        :
            Intervention
        """
        for interv in cls.interventions:
            if name == interv.name:
                return interv


class Intervention(object):
    """
    Parameters
    ----------
    name : string
        Name of the intervention

    num_stages: int,
        Number of different stages the intervention can have.

    prior_alpha_loc : number, optional

    prior_alpha_scale : number, optional
    """

    def __init__(self, name, num_stages, prior_alpha_loc=0.1, prior_alpha_scale=0.1):

        # Name
        self.name = name

        # Get number of stages
        self.num_stages = num_stages

        # Hyper prior effectivity
        self.prior_alpha_loc = prior_alpha_loc
        self.prior_alpha_scale = prior_alpha_scale


class Change_point(object):
    """
    Parameters
    ----------
    prior_date_loc : number
        Mean of prior distribution for the location (date) of the change point.

    gamma_max:
        Gamma max value for change point

    length : number, optional
        Length of change point

    prior_date_scale : number, optional
        Scale of prior distribution for the location (date) of the change point.
    """

    def __init__(self, date_data, gamma_max):

        self.date_data = date_data
        self.gamma_max = gamma_max
