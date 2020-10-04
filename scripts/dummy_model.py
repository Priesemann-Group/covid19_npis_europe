import sys
import logging
import time
import os

"""
os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
"""

import pymc4 as pm
import tensorflow as tf
import numpy as np
import time
import os

"""
old_opts = tf.config.optimizer.get_experimental_options()
print(old_opts)
tf.config.optimizer.set_experimental_options(
    {
        "autoparallel_optimizer": True,
        "layout_optimizer": True,
        "loop_optimizer": True,
        "dependency_optimizer": True,
        "shape_optimizer": True,
        "function_optimizer": True,
        "constant_folding_optimizer": True,
    }
)
print(tf.config.optimizer.get_experimental_options())
"""


sys.path.append("../")

# Needed to set logging level before importing other modules
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

import covid19_npis
from covid19_npis.model import main_model


""" # Debugging and other snippets
"""
# For eventual debugging:
tf.config.run_functions_eagerly(True)
# tf.debugging.enable_check_numerics(stack_height_limit=50, path_length_limit=50)

# Force CPU
covid19_npis.utils.force_cpu_for_tensorflow()


""" # 1. Data Retrieval
    Load data for different countries/regions, for now we have to define every
    country by hand maybe we want to automatize that at some point.

    TODO: maybe we want to outsource that to a different file at some point
"""

# Load our data from csv files into our own custom data classes
"""
c1 = covid19_npis.data.Country(
    "test-country-1",  # name
    "../data/test_country_1/new_cases.csv",  # new_Cases per age groups in country
    "../data/test_country_1/interventions.csv",  # interventions timeline with stringency index
)
c2 = covid19_npis.data.Country(
    "test-country-2",
    "../data/test_country_2/new_cases.csv",
    "../data/test_country_2/interventions.csv",
)
"""

c1 = covid19_npis.data.Country(
    "test-country-1",  # name
    "../data/test_data_from_model/test-country-1/new_cases.csv",  # new_Cases per age groups in country
    "../data/test_data_from_model/test-country-1/interventions.csv",  # interventions timeline with stringency index
)
c2 = covid19_npis.data.Country(
    "test-country-2",
    "../data/test_data_from_model/test-country-2/new_cases.csv",
    "../data/test_data_from_model/test-country-2/interventions.csv",
)

# Construct our modelParams from the data.
modelParams = covid19_npis.ModelParams(countries=[c1, c2])


""" # 2. MCMC Sampling
"""

begin_time = time.time()

trace = pm.sample(
    main_model(modelParams),
    num_samples=100,
    burn_in=200,
    use_auto_batching=False,
    num_chains=3,
    xla=True,
)

end_time = time.time()
log.info("running time: {:.1f}s".format(end_time - begin_time))


""" # 3. Plotting
"""
import matplotlib.pyplot as plt
import pandas as pd

""" ## Sample for prior plots and also covert to nice format
"""
trace_prior = pm.sample_prior_predictive(
    main_model(modelParams), sample_shape=(1000,), use_auto_batching=False
)
_, sample_state = pm.evaluate_model(main_model(modelParams))


""" ## Plot distributions
    Function returns a list of figures which can be shown by fig[i].show() each figure being one country.
"""


# Plot by name.
dist_names = [
    "R_0",
    "I_0_diff_base",
    "g_mu",
    "g_theta",
    "sigma",
    "alpha_i_c_a",
    "l_i_sign",
    "d_i_c_p",
]

dist_fig = {}
dist_axes = {}
for name in dist_names:
    dist_fig[name], dist_axes[name] = covid19_npis.plot.distribution(
        trace, trace_prior, sample_state=sample_state, key=name
    )
    # Save figure
    for i, fig in enumerate(dist_fig[name]):
        if len(dist_fig[name]) > 1:
            subname = f"_{i}"
        else:
            subname = ""
        fig.savefig(
            f"figures/dist_{name}" + f"{subname}.pdf", dpi=300, transparent=True
        )


""" ## Plot time series
"""
ts_names = ["new_I_t", "R_t", "h_0_t"]
ts_fig = {}
ts_axes = {}
for name in ts_names:
    ts_fig[name], ts_axes[name] = covid19_npis.plot.timeseries(
        trace, sample_state=sample_state, key=name, plot_chain_separated=True,
    )
    # plot data into new_cases
    if name == "new_I_t":
        for i, c in enumerate(modelParams.data_summary["countries"]):
            for j, a in enumerate(modelParams.data_summary["age_groups"]):
                ts_axes["new_I_t"][j][i] = covid19_npis.plot.time_series._timeseries(
                    modelParams.dataframe.index[:],
                    modelParams.dataframe[(c, a)].to_numpy()[:],
                    ax=ts_axes["new_I_t"][j][i],
                    alpha=0.5,
                )

    # plot R_t data into R_t plot --> testing
    if name == "R_t":
        # Load data
        for i, c in enumerate(modelParams.data_summary["countries"]):
            data = pd.read_csv(f"../data/test_country_{i+1}/reproduction_number.csv")
            data["date"] = pd.to_datetime(data["date"], format="%d.%m.%y")
            data = data.set_index("date")
            for j, age_group in enumerate(data.columns):
                ts_axes["R_t"][j][i] = covid19_npis.plot.time_series._timeseries(
                    data.index, data[age_group], ax=ts_axes["R_t"][j][i], alpha=0.5,
                )
    # Save figures
    for i, fig in enumerate(ts_fig[name]):
        if len(ts_fig[name]) > 1:
            subname = f"_{i}"
        else:
            subname = ""
        fig.savefig(f"figures/ts_{name}" + f"{subname}.pdf", dpi=300, transparent=True)
