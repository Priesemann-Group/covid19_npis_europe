import logging
import pandas as pd

log = logging.getLogger(__name__)


def convert_trace_to_dataframe_list(trace, config):
    r"""
    Converts the pymc4 arviz trace to multiple pandas dataframes.
    Also sets the right labels for the  dimensions i.e splits data by
    country and age group. 

    Do not look too much into this function if you want to keep your sanity!

    Parameters
    ----------
    trace: arivz InferenceData

    config: cov19_npis.config.Config
        
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
    dists = [key for key in config.distributions]

    # Create dataframe from xarray
    dfs = []
    for d in data:
        for dist in dists:
            dfs.append(convert_trace_to_dataframe(trace, config, dist))
    return dfs


def convert_trace_to_dataframe(trace, modelParams, key):
    r"""
    Converts the pymc4 arviz trace for a single key to a pandas dataframes.
    Also sets the right labels for the  dimensions i.e splits data by
    country and age group. 

    Do not look too much into this function if you want to keep your sanity!

    Parameters
    ----------
    trace: arivz InferenceData

    modelParams: cov19_npis.modelParams.modelParams

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
    assert key in [
        modelParams.distributions[dist]["name"] for dist in modelParams.distributions
    ], f"Key '{key}' not found! Is it added to modelParams.py?"

    # Get distribution
    dist = modelParams.get_distribution_by_name(key)

    df = data[f"{model_name}/{dist['name']}"].to_dataframe()
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
    if isinstance(dist["shape_label"], (tuple, list)):
        ndim = len(dist["shape_label"])
    else:
        ndim = 1

    for i in range(ndim):
        df.index.rename(
            dist["shape_label"][i], level=i - ndim, inplace=True,
        )

    # Rename country index to country names
    if r"country" in df.index.names:
        df.index = df.index.set_levels(
            modelParams.data_summary["countries"], level="country"
        )

    # Rename age_group index to age_group names
    if r"age_group" in df.index.names:
        df.index = df.index.set_levels(
            modelParams.data_summary["age_groups"], level="age_group"
        )
    if r"age_group_i" in df.index.names:
        df.index = df.index.set_levels(
            modelParams.data_summary["age_groups"], level="age_group_i"
        )
    if r"age_group_j" in df.index.names:
        df.index = df.index.set_levels(
            modelParams.data_summary["age_groups"], level="age_group_j"
        )

    # Convert time index to datetime starting at model begin
    if r"time" in df.index.names:
        df.index = df.index.set_levels(
            pd.date_range(
                modelParams.dataframe.index.min(), modelParams.dataframe.index.max()
            ),
            level="time",
        )

    # Last rename column
    df = df.rename(columns={f"{model_name}/{dist['name']}": dist["name"]})

    return df


def select_from_dataframe(df, axis=0, **kwargs):
    for key, value in kwargs.items():
        df = df.xs(value, level=key, axis=axis)
    return df


class Country(object):
    """docstring for Country"""

    def __init__(self, arg):
        super(Country, self).__init__()
        self.arg = arg


class Intervention(object):
    """docstring for Intervention"""

    def __init__(self, arg):
        super(Intervention, self).__init__()
        self.arg = arg


class Change_point(object):
    """docstring for Change_point"""

    def __init__(self, arg):
        super(Change_point, self).__init__()
        self.arg = arg
