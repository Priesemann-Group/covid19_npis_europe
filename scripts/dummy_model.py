import sys
import logging
import time
import os

import pymc4 as pm
import tensorflow as tf
import numpy as np
import time
import os

sys.path.append("../")

# Needed to set logging level before importing other modules
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

import covid19_npis
from covid19_npis import transformations
from covid19_npis.benchmarking import benchmark
from covid19_npis.model.distributions import LKJCholesky, Deterministic, Gamma


""" # Debugging and other snippets
"""
# For eventual debugging:
# tf.config.run_functions_eagerly(True)
# tf.debugging.enable_check_numerics(stack_height_limit=50, path_length_limit=100)

# Force CPU
covid19_npis.utils.force_cpu_for_tensorflow()


""" # 1. Data Retrieval
    Load data for different countries/regions, for now we have to define every
    country by hand maybe we want to automatize that at some point.
"""

# Load our data from csv files into our own custom data classes
c1 = covid19_npis.data.Country(
    "test_country_1",  # name
    "../data/test_country_1/new_cases.csv",  # new_Cases per age groups in country
    "../data/test_country_1/interventions.csv",  # interventions timeline with stringency index
)
c2 = covid19_npis.data.Country(
    "test_country_2",
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
    mean_R_0 = 2.5
    beta_R_0 = 2.0
    R_0 = yield Gamma(
        name="R_0",
        concentration=mean_R_0 * beta_R_0,
        rate=beta_R_0,
        conditionally_independent=True,
        event_stack=event_shape,
        transform=transformations.SoftPlus(reinterpreted_batch_ndims=len(event_shape)),
        shape_label=("country", "age_group"),
    )
    log.debug(f"R_0:\n{R_0}")

    # Create interventions and change points from model parameters. Combine to R_t
    R_t = yield covid19_npis.model.reproduction_number.construct_R_t(R_0, modelParams)
    # R_t = tf.stack([R_0] * 50)
    log.info(f"R_t:\n{R_t}")

    # Create Contact matrix
    # Use Cholesky version as the non Cholesky version uses tf.linalg.slogdet which isn't implemented in JAX
    C = yield LKJCholesky(
        name="C_cholesky",
        dimension=modelParams.num_age_groups,
        concentration=4,  # eta
        conditionally_independent=True,
        event_stack=modelParams.num_countries,
        validate_args=True,
        transform=transformations.CorrelationCholesky(reinterpreted_batch_ndims=1),
        shape_label=("country", "age_group_i", "age_group_j"),
    )  # |shape| batch_dims, num_countries, num_age_groups, num_age_groups
    log.info(f"C chol:\n{C}")

    C = yield Deterministic(
        name="C",
        value=tf.einsum("...an,...bn->...ab", C, C),
        shape_label=("country", "age_group_i", "age_group_j"),
    )
    log.debug(f"C:\n{C}")

    # Create normalized pdf of generation interval
    (
        gen_kernel,  # shape: countries x len_gen_interv,
        mean_gen_interv,  #  shape g_mu: countries x 1
    ) = yield covid19_npis.model.construct_generation_interval(l=len_gen_interv_kernel)
    log.info(f"gen_interv:\n{gen_kernel}")

    # Generate exponential distribution with initial infections as external input
    h_0_t = yield covid19_npis.model.construct_h_0_t(
        modelParams=modelParams,
        len_gen_interv_kernel=len_gen_interv_kernel,
        R_t=R_t,
        mean_gen_interv=mean_gen_interv,
        mean_test_delay=0,
    )  # |shape| time, batch, countries, age_groups

    # Create N tensor (vector)
    # should be done earlier in the real model
    N = tf.convert_to_tensor([1e12, 1e12, 1e12, 1e12] * modelParams.num_countries)
    N = tf.reshape(N, event_shape)
    log.debug(f"N:\n{N}")
    # Calculate new cases
    new_cases = covid19_npis.model.InfectionModel(
        N=N, h_0_t=h_0_t, R_t=R_t, C=C, gen_kernel=gen_kernel  # default valueOp:AddV2
    )
    log.info(f"new_cases:\n{new_cases[0,:]}")  # dimensons=t,c,a

    # Clip in order to avoid infinities
    new_cases = tf.clip_by_value(new_cases, 1e-7, 1e9)

    new_cases = yield Deterministic(
        name="new_cases",
        value=new_cases,
        shape=modelParams.data_tensor.shape,
        shape_label=("time", "country", "age_group"),
    )

    likelihood = yield covid19_npis.model.studentT_likelihood(modelParams, new_cases)

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

""" ## Sample for prior plots and also covert to nice format
"""
trace_prior = pm.sample_prior_predictive(
    test_model(modelParams), sample_shape=1000, use_auto_batching=False
)
_, sample_state = pm.evaluate_model(test_model(modelParams))

""" ## Plot distributions
    Function returns a list of figures which can be shown by fig[i].show() each figure being one country.
"""
dist_names = ["R_0", "I_0_diff_base", "g_mu", "g_theta", "sigma"]
fig = {}
for name in dist_names:
    fig[name] = covid19_npis.plot.distribution(
        trace, trace_prior, sample_state=sample_state, key=name
    )
    # Save figure
    plt.savefig("figures/dist_" + name + ".pdf", dpi=300, transparent=True)

""" ## Plot time series for "new_cases" and "R_t"
"""
fig_new_cases = covid19_npis.plot.timeseries(
    trace, sample_state=sample_state, key="new_cases"
)

# plot data into the axes
for i, c in enumerate(modelParams.data_summary["countries"]):
    for j, a in enumerate(modelParams.data_summary["age_groups"]):
        fig_new_cases[j][i] = covid19_npis.plot.time_series._timeseries(
            modelParams.dataframe.index[:],
            modelParams.dataframe[(c, a)].to_numpy()[:],
            ax=fig_new_cases[j][i],
            alpha=0.5,
        )

# Save figure
plt.savefig("figures/ts_new_cases.pdf", dpi=300, transparent=True)


fig_R_t = covid19_npis.plot.timeseries(trace, sample_state=sample_state, key="R_t")

# Save figure
plt.savefig("figures/ts_R_t.pdf", dpi=300, transparent=True)
