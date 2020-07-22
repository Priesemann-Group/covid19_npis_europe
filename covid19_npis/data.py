import logging

log = logging.getLogger(__name__)


def convert_trace_to_pandas_list(model, trace, config):
    r"""
    Converts the pymc4 arviz trace to multiple pandas dataframes.
    Also sets the right labels for the  dimensions i.e splits data by
    country and age group. 

    Do not look too much into this function if you want to keep your sanity!

    Parameters
    ----------
    model: pymc4 model instance

    trace: arivz InferenceData

    config: cov19_npis.config.Config
        
    Returns
    -------
    :array of

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

    # Get model and var names a bit hacky but works
    for d in data:
        print(d)
        for var in d:
            model_name, var_name = var.split("/")
            break

    # Create dataframe from xarray
    dfs = []
    for d in data:
        for dist in dists:
            df = d[f"{model_name}/{config.distributions[dist]['name']}"].to_dataframe()
            num_of_levels = len(df.index.levels)

            # Rename level to dimension labels
            for i in range(num_of_levels):
                """
                i=0 is always chain
                i=1 is always sample
                i=2 is config.distribution...shape[0]
                i=3 is config.distribution...shape[1]
                """
                if i == 0 or i == 1:
                    continue
                df.index.rename(
                    config.distributions[dist]["shape_label"][i - 2],
                    level=i,
                    inplace=True,
                )
            # Rename country and age_group entries in index
            if r"country" in df.index.names:
                df.index = df.index.set_levels(
                    config.data["countries"], level="country"
                )

            if r"age_group" in df.index.names:
                df.index = df.index.set_levels(
                    config.data["age_groups"], level="age_group"
                )
            if r"age_group_i" in df.index.names:
                df.index = df.index.set_levels(
                    config.data["age_groups"], level="age_group_i"
                )
            if r"age_group_j" in df.index.names:
                df.index = df.index.set_levels(
                    config.data["age_groups"], level="age_group_j"
                )
            dfs.append(df)
    return dfs
