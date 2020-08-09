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

""" # Debugging and other snippets
"""
# For eventual debugging:
# tf.config.run_functions_eagerly(True)
# tf.debugging.enable_check_numerics(stack_height_limit=50, path_length_limit=50)

# Force CPU
my_devices = tf.config.experimental.list_physical_devices(device_type="CPU")
tf.config.experimental.set_visible_devices(devices=my_devices, device_type="CPU")
tf.config.set_visible_devices([], "GPU")

# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

""" # Data Retrieval
    Retrieves some dummy/test data and creates ModelParams object
"""
# Get test dataset with time dependent reproduction number
"""
I_new_1, interventions_1 = covid19_npis.test_data.simple_new_I_with_R_t(1)
I_new_2, interventions_2 = covid19_npis.test_data.simple_new_I_with_R_t(0.9)
I_new = I_new_1.join(I_new_2)
interventions = [interventions_1, interventions_2]
"""
I_new = covid19_npis.test_data.simple_new_I(1)
I_new = I_new.join(covid19_npis.test_data.simple_new_I(1))


# Create model params
"""
We create our own model params object which holds names of the distributions,
shape label and the observed data. This is necessary for the data converter
later on.
"""
modelParams = covid19_npis.ModelParams(I_new)
# modelParams.interventions = interventions
""" # Construct pymc4 model
"""


@pm.model
def test_model(modelParams):
    event_shape = (modelParams.num_countries, modelParams.num_age_groups)
    len_gen_interv_kernel = 12

    # Create Reproduction Number for every age group
    mean_R_0 = 2.5
    beta_R_0 = 2.0
    R_0 = yield pm.Gamma(
        name=modelParams.distributions["R"]["name"],
        concentration=mean_R_0 * beta_R_0,
        rate=beta_R_0,
        conditionally_independent=True,
        event_stack=event_shape,
        transform=transformations.SoftPlus(reinterpreted_batch_ndims=len(event_shape)),
    )
    log.debug(f"R_0:\n{R_0}")
    # Interventions = covid19_npis.model.reproduction_number.create_interventions(
    #    modelParams
    # )
    # log.info(f"Interventions:\n{Interventions}")
    # R_t = yield covid19_npis.model.reproduction_number.construct_R_t(R_0, Interventions)
    R_t = tf.stack([R_0] * 50)
    log.debug(f"R_t:\n{R_t}")

    # Create Contact matrix
    # Use Cholesky version as the non Cholesky version uses tf.linalg.slogdet which isn't implemented in JAX
    C = yield pm.LKJCholesky(
        name="C_cholesky",
        dimension=modelParams.num_age_groups,
        concentration=4,  # eta
        conditionally_independent=True,
        event_stack=modelParams.num_countries,
        validate_args=True,
        transform=transformations.CorrelationCholesky(reinterpreted_batch_ndims=1),
    )  # |shape| batch_dims, num_countries, num_age_groups, num_age_groups
    log.debug(f"C chol:\n{C}")

    C = yield pm.Deterministic(
        name=modelParams.distributions["C"]["name"],
        value=tf.einsum("...an,...bn->...ab", C, C),
    )
    log.debug(f"C:\n{C}")

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
    )

    # Create N tensor (vector)
    # should be done earlier in the real model
    N = tf.convert_to_tensor([1e12, 1e12, 1e12, 1e12] * modelParams.num_countries)
    N = tf.reshape(N, event_shape)
    log.debug(f"N:\n{N}")
    # Calculate new cases
    new_cases = covid19_npis.model.InfectionModel(
        N=N, h_0_t=h_0_t, R_t=R_t, C=C, gen_kernel=gen_kernel  # default valueOp:AddV2
    )
    log.debug(f"new_cases:\n{new_cases[0,:]}")  # dimensons=t,c,a

    # Clip in order to avoid infinities
    new_cases = tf.clip_by_value(new_cases, 1e-7, 1e9)

    new_cases = yield pm.Deterministic(name="new_cases", value=new_cases)

    likelihood = yield covid19_npis.model.studentT_likelihood(modelParams, new_cases)

    return likelihood


""" # MCMC Sampling
"""

begin_time = time.time()
trace = pm.sample(
    test_model(modelParams),
    num_samples=500,
    burn_in=500,
    use_auto_batching=False,
    num_chains=2,
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

""" ## Plot distributions
    Function returns a list of figures which can be shown by fig[i].show() each figure being one country.
"""
dist_names = ["R", "I_0_diff_base", "g_mu", "g_theta", "sigma"]
fig = {}
for name in dist_names:
    fig[name] = covid19_npis.plot.distribution(
        trace, trace_prior, modelParams=modelParams, key=name
    )
    # Save figure
    plt.savefig("figures/dist_" + name + ".pdf", dpi=300, transparent=True)

""" ## Plot time series for "new_cases"
"""
fig_new_cases = covid19_npis.plot.timeseries(
    trace, modelParams=modelParams, key="new_cases"
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
