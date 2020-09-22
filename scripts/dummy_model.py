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
# logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

import covid19_npis
from covid19_npis import transformations

#  from covid19_npis.benchmarking import benchmark
from covid19_npis.model.distributions import (
    LKJCholesky,
    Deterministic,
    Gamma,
    HalfCauchy,
    Normal,
    LogNormal,
)


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
"""

# Load our data from csv files into our own custom data classes
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

# Construct our modelParams from the data.
modelParams = covid19_npis.ModelParams(countries=[c1, c2])


""" # Construct pymc4 model
"""


@pm.model()
def test_model(modelParams):
    event_shape = (modelParams.num_countries, modelParams.num_age_groups)
    len_gen_interv_kernel = 12

    # Create Reproduction Number for every age group

    R_0 = yield covid19_npis.model.reproduction_number.construct_R_0(
        name="R_0",
        loc=2.0,
        scale=0.5,
        hn_scale=0.3,  # Scale parameter of HalfNormal for each country
        modelParams=modelParams,
    )
    """
    R_0 = yield covid19_npis.model.reproduction_number.construct_R_0_old(
        "R_0", 2.5, 2.0, modelParams
    )
    """
    log.debug(f"R_0:\n{R_0}")

    # Create interventions and change points from model parameters. Combine to R_t
    R_t = yield covid19_npis.model.reproduction_number.construct_R_t(R_0, modelParams)
    log.debug(f"R_t:\n{R_t}")

    # Create Contact matrix
    # Use Cholesky version as the non Cholesky version uses tf.linalg.slogdet which isn't implemented in JAX
    C = yield LKJCholesky(
        name="C_cholesky",
        dimension=modelParams.num_age_groups,
        concentration=4,  # eta
        conditionally_independent=True,
        event_stack=modelParams.num_countries,
        validate_args=True,
        transform=transformations.CorrelationCholesky(),
        shape_label=("country", "age_group_i", "age_group_j"),
    )  # |shape| batch_dims, num_countries, num_age_groups, num_age_groups

    C = yield Deterministic(
        name="C",
        value=tf.einsum("...an,...bn->...ab", C, C),
        shape_label=("country", "age_group_i", "age_group_j"),
    )
    # Normalize C
    C, _ = tf.linalg.normalize(C, ord=1, axis=-1)

    log.debug(f"C_normalized:\n{C}")

    # Create normalized pdf of generation interval
    (
        gen_kernel,  # shape: countries x len_gen_interv,
        mean_gen_interv,  #  shape g_mu: countries x 1
    ) = yield covid19_npis.model.construct_generation_interval(l=len_gen_interv_kernel)
    log.debug(f"gen_interv:\n{gen_kernel}")

    # Generate exponential distribution with initial infections as external input

    h_0_t = yield covid19_npis.model.construct_h_0_t(
        modelParams=modelParams,
        len_gen_interv_kernel=len_gen_interv_kernel,
        R_t=R_t,
        mean_gen_interv=mean_gen_interv,
        mean_test_delay=0,
    )  # |shape| time, batch, countries, age_groups

    yield Deterministic(
        "h_0_t",
        tf.einsum("t...ca->...tca", h_0_t),
        shape_label=("time", "country", "age_group"),
    )

    """
    h_0 = yield HalfCauchy(
        name="I_0",
        event_stack=(modelParams.num_countries, modelParams.num_age_groups),
        scale=20,
        transformation=transformations.SoftPlus(reinterpreted_batch_ndims=2),
        shape_label=("country", "age_group"),
        conditionally_independent=True,
    )
    if len(R_0.shape) == 3:
        h_0 = tf.ones((3, 2, 4), dtype="float32")
    else:
        h_0 = tf.ones((2, 4), dtype="float32")
    h_0_t = tf.stack([h_0] * 50)
    """

    # Create N tensor (vector)
    # should be done earlier in the real model
    N = tf.convert_to_tensor([1e12, 1e12, 1e12, 1e12] * modelParams.num_countries)
    N = tf.reshape(N, event_shape)
    log.debug(f"N:\n{N}")
    # Calculate new cases
    new_I_t = covid19_npis.model.InfectionModel(
        N=N, h_0_t=h_0_t, R_t=R_t, C=C, gen_kernel=gen_kernel  # default valueOp:AddV2
    )
    log.debug(f"new_I_t:\n{new_I_t[0,:]}")  # dimensons=t,c,a

    # Clip in order to avoid infinities
    new_I_t = tf.clip_by_value(new_I_t, 1e-7, 1e9)

    new_I_t = yield Deterministic(
        name="new_I_t", value=new_I_t, shape_label=("time", "country", "age_group"),
    )

    mean_delay = yield LogNormal(
        name="mean_delay",
        loc=np.log(12, dtype="float32"),
        scale=0.1,
        event_shape=(modelParams.num_countries, 1),
    )

    # kernel =
    # new_cases = covid19_npis.model.utils.convolution_with_fixed_kernel(new_I_t, kernel)

    likelihood = yield covid19_npis.model.studentT_likelihood(modelParams, new_I_t)

    return likelihood


""" # MCMC Sampling
"""

begin_time = time.time()

trace = pm.sample(
    test_model(modelParams),
    num_samples=50,
    burn_in=100,
    use_auto_batching=False,
    num_chains=3,
    xla=True,
)

end_time = time.time()
log.info("running time: {:.1f}s".format(end_time - begin_time))


""" # Plotting
"""
import matplotlib.pyplot as plt
import pandas as pd

""" ## Sample for prior plots and also covert to nice format
"""
trace_prior = pm.sample_prior_predictive(
    test_model(modelParams), sample_shape=1000, use_auto_batching=False
)
_, sample_state = pm.evaluate_model(test_model(modelParams))


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
