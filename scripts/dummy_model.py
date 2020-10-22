import sys
import logging


import time
import itertools
import os

"""
os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
"""

import pymc4 as pm
import tensorflow as tf

# Mute Tensorflow warnings ...
logging.getLogger("tensorflow").setLevel(logging.ERROR)

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


import covid19_npis
from covid19_npis.model import main_model


""" # Debugging and other snippets
"""

# Logs setup
log = logging.getLogger()
# Needed to set logging level before importing other modules
log.setLevel(logging.DEBUG)
covid19_npis.utils.setup_colored_logs()
logging.getLogger("parso.python.diff").disabled = True

# For eventual debugging:
tf.config.run_functions_eagerly(True)
# tf.debugging.enable_check_numerics(stack_height_limit=50, path_length_limit=50)

# Force CPU
covid19_npis.utils.force_cpu_for_tensorflow()
covid19_npis.utils.split_cpu_in_logical_devices(4)


""" # 1. Data Retrieval
    Load data for different countries/regions, for now we have to define every
    country by hand maybe we want to automatize that at some point.

    TODO: maybe we want to outsource that to a different file at some point
"""

# Load our data from csv files into our own custom data classes

c1 = covid19_npis.data.Country("../data/Germany",)  # name
c2 = covid19_npis.data.Country("../data/France",)


"""
c1 = covid19_npis.data.Country("../data/Germany",)  # Data folder for germany
c2 = covid19_npis.data.Country("../data/France",)
"""
# Construct our modelParams from the data.
modelParams = covid19_npis.ModelParams(countries=[c1, c2])

# Test shapes, should be all 3:
def print_dist_shapes(st):
    for name, dist in itertools.chain(
        st.discrete_distributions.items(), st.continuous_distributions.items(),
    ):
        print(dist.log_prob(st.all_values[name]).shape, name)
    for p in st.potentials:
        print(p.value.shape, p.name)


_, sample_state = pm.evaluate_model_transformed(
    main_model(modelParams), sample_shape=(3,)
)
print_dist_shapes(sample_state)

""" # 2. MCMC Sampling
"""

begin_time = time.time()

trace = pm.sample(
    main_model(modelParams),
    num_samples=200,
    burn_in=400,
    use_auto_batching=False,
    num_chains=2,
    xla=True,
    # sampler_type="nuts",
)

end_time = time.time()
log.info("running time: {:.1f}s".format(end_time - begin_time))

# Save trace
import datetime
import arviz

today = datetime.datetime.now()
trace.posterior.to_netcdf(
    f"./traces/{today.strftime('%y_%m_%d_%H')}.h5", engine="scipy"
)

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
    "sigma_likelihood_pos_tests",
    "alpha_i_c_a",
    "l_i_sign",
    "d_i_c_p",
    "C",
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
ts_names = ["positive_tests", "R_t", "h_0_t"]
ts_fig = {}
ts_axes = {}
for name in ts_names:
    ts_fig[name], ts_axes[name] = covid19_npis.plot.timeseries(
        trace, sample_state=sample_state, key=name, plot_chain_separated=True,
    )
    # plot data into new_cases
    if name == "new_I_t" or name == "positive_tests":
        for i, c in enumerate(modelParams.data_summary["countries"]):
            for j, a in enumerate(modelParams.data_summary["age_groups"]):
                ts_axes[name][j][i] = covid19_npis.plot.time_series._timeseries(
                    modelParams.pos_tests_dataframe.index[:],
                    modelParams.pos_tests_dataframe[(c, a)].to_numpy()[:],
                    ax=ts_axes[name][j][i],
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
