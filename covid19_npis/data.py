import logging
import pandas as pd

log = logging.getLogger(__name__)
from . import modelParams

from .plot.utils import check_for_shape_and_shape_label


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


def convert_trace_to_dataframe(trace, sample_state, key):
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
        
    Returns
    -------
    pd.DataFrame
        Multiindex dataframe containing all samples by chain and other dimensions
        defined in modelParams.py

    """
    # Try to get posterior and prior data
    if hasattr(trace, "posterior"):
        log.info("Found posterior in trace.")
        data = trace.posterior.data_vars
    if hasattr(trace, "prior_predictive"):
        log.info("Found prior_predictive in trace.")
        data = trace.prior_predictive.data_vars

    # Get model and var names a bit hacky but works
    for var in data:
        model_name, var_name = var.split("/")
        break

    # Check key value
    dists_and_determs = [dist.split("/")[1] for dist in sample_state.distributions]
    for deter in sample_state.deterministics:
        dists_and_determs.append(deter.split("/")[1])

    assert key in dists_and_determs, f"Key '{key}' not found! Check for typos."

    # Get distribution
    try:
        dist = sample_state.distributions[model_name + "/" + key]
    except Exception as e:
        dist = sample_state.deterministics[model_name + "/" + key]

    # Check if it has shape and shape_label
    check_for_shape_and_shape_label(dist)
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
    if isinstance(dist.shape, (tuple, list)):
        ndim = len(dist.shape)
    else:
        ndim = 1

    # Rename dimensions if shape labels are present
    if hasattr(dist, "shape_label"):
        for i in range(ndim):
            if isinstance(dist.shape_label, (list, tuple,)):
                df.index.rename(
                    dist.shape_label[i], level=i - ndim, inplace=True,
                )
            else:
                df.index.rename(
                    dist.shape_label, level=i - ndim, inplace=True,
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

    # Convert time index to datetime starting at model begin
    if r"time" in df.index.names:
        df.index = df.index.set_levels(
            pd.date_range(
                modelParams.modelParams.dataframe.index.min(),
                modelParams.modelParams.dataframe.index.max(),
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
    """ Country data class, that contains new_cases and intervention data for a speciffic country.
        Also calculates change points and interventions automatically on init.

        There is a global class attribute which holds the interventions.

        Parameters
        ----------
        name: string
            Name of the country
            
        fp_new_cases : string
            Filepath to the daily new cases as csv, should have age group columns. The age group
            names get parsed from the column names!

        fp_interventions : string
            Filepath to the interventions csv file. The column names get parsed as intervention names!
    """

    """
    Class attribute for the interventions, get set for all instances
    of the country class!
    list of intervention objects
    """
    interventions = []

    def __init__(self, name, fp_new_cases, fp_interventions):
        self.name = name
        # Retrieve files and parse them!
        self.data_new_cases = self._to_iso(
            self._load_csv_with_date_index(fp_new_cases), "age_group"
        )
        self.data_interventions = self._to_iso(
            self._load_csv_with_date_index(fp_interventions), "intervention"
        )
        # Create change_points from interventions time series
        self.change_points = {}
        for column in self.data_interventions.columns:
            self.change_points.update(
                self.create_change_points(self.data_interventions[column])
            )

    def _load_csv_with_date_index(self, filepath):
        """
            Loads csv file with date column
        """
        data = pd.read_csv(filepath)
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"], format="%d.%m.%y")
            data = data.set_index("date")

        return data

    def _to_iso(self, df, name):
        """
            Create multicolumn from normal columns with country at level 0
            and name for level 1

            Parameters
            ----------
            df : pandas.DataFrame

            name: str
        """
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
            if delta != 0:
                change_points.append(
                    Change_point(
                        prior_date_loc=df.index[row],
                        gamma_max=delta
                        / interv.num_stages,  # +1 because it starts at 0
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

    def __init__(self, prior_date_loc, gamma_max, length=4, prior_date_scale=2):

        # Priors
        self.prior_date_loc = prior_date_loc
        self.prior_date_scale = prior_date_scale

        self.gamma_max = gamma_max
        self.length = length
