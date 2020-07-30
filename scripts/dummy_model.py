import sys
import pymc4 as pm
import tensorflow as tf
import numpy as np
import time
import os

import logging

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)

sys.path.append("../")
import covid19_npis
from covid19_npis import transformations
from covid19_npis.benchmarking import benchmark

##For eventual debugging:
# tf.config.run_functions_eagerly(True)
# tf.debugging.enable_check_numerics(
#    stack_height_limit=30, path_length_limit=50
# )

""" Force CPU
"""
my_devices = tf.config.experimental.list_physical_devices(device_type="CPU")
tf.config.experimental.set_visible_devices(devices=my_devices, device_type="CPU")
tf.config.set_visible_devices([], "GPU")

# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

""" # Data Retrieval
    Retrieves some dummy/test data
"""
# Fixed R matrix for now one country one age group
I_new = covid19_npis.test_data.simple_new_I(1)
I_new = I_new.join(covid19_npis.test_data.simple_new_I(1))
num_age_groups = 4
num_countries = 2

""" # Construct pymc4 model
    We create our own config object which holds names of the distributions,
    shape label and the observed data. This is necessary for the data converter
    later on.
"""
config = covid19_npis.Config(I_new)


@pm.model
def test_model(config):
    event_shape = (num_countries, num_age_groups)

    # Create I_0
    I_0 = yield pm.HalfCauchy(
        name=config.distributions["I_0"]["name"],
        loc=10.0,
        scale=25,
        conditionally_independent=True,
        event_stack=event_shape,
        transform=transformations.Log(reinterpreted_batch_ndims=len(event_shape)),
    )
    I_0 = tf.clip_by_value(I_0, 1e-12, 1e12)

    # Create Reproduction Number for every age group
    R_0 = yield pm.LogNormal(
        name=config.distributions["R"]["name"],
        loc=1,
        scale=2.5,
        conditionally_independent=True,
        event_stack=event_shape,
        transform=transformations.Log(reinterpreted_batch_ndims=len(event_shape)),
    )
    log.debug(f"R_0:\n{R_0}")

    # Create interventions for dummy model
    cp1_1 = covid19_npis.model.reproduction_number.Change_point(
        "cp1_1", date_loc=7, date_scale=2, gamma_max=0.5, length=2
    )
    cp1_2 = covid19_npis.model.reproduction_number.Change_point(
        "cp1_2", date_loc=30, date_scale=2, gamma_max=-0.5, length=2
    )
    cp2_1 = covid19_npis.model.reproduction_number.Change_point(
        "cp2_1", date_loc=24, date_scale=2, gamma_max=1, length=2
    )
    Interventions = [
        covid19_npis.model.reproduction_number.Intervention(
            "inter_1",
            length_loc=3,
            length_scale=2,
            alpha_loc=0.1,
            alpha_scale=0.2,
            change_points=[cp1_1, cp1_2],
        ),
        covid19_npis.model.reproduction_number.Intervention(
            "inter_2",
            length_loc=3,
            length_scale=2,
            alpha_loc=0.2,
            alpha_scale=0.2,
            change_points=[cp2_1],
        ),
    ]

    R_t = yield covid19_npis.model.reproduction_number.construct_R_t(R_0, Interventions)
    log.info(R_t)
    # Create Contact matrix

    # Use Cholesky version as the non Cholesky version uses tf.linalg.slogdet which isn't implemented in JAX
    C = yield pm.LKJCholesky(
        name="C_cholesky",
        dimension=num_age_groups,
        concentration=2,  # eta
        conditionally_independent=True,
        event_stack=num_countries,
    )  # dimensions: batch_dims x num_countries x num_age_groups x num_age_groups
    C = yield pm.Deterministic(
        name=config.distributions["C"]["name"],
        value=tf.einsum("...ab,...ba->...ab", C, C),
    )

    # Create normalized pdf of generation interval
    g_p = yield covid19_npis.model.construct_generation_interval()
    log.debug(f"g_p:\n{g_p}")

    # Create N tensor (vector)
    # should be done earlier in the real model
    N = tf.convert_to_tensor([1e12, 1e12, 1e12, 1e12] * num_countries)
    N = tf.reshape(N, event_shape)
    log.debug(f"N:\n{N}")
    # Calculate new cases
    new_cases = covid19_npis.model.InfectionModel(
        N=N, I_0=I_0, R_t=R_t, C=C, g_p=g_p  # default valueOp:AddV2
    )
    log.debug(f"new_cases:\n{new_cases[0,:]}")  # dimensons=t,c,a

    # Clip in order to avoid infinities
    new_cases = tf.clip_by_value(new_cases, 1e-7, 1e9)

    # Get scale of likelihood
    sigma = yield pm.HalfCauchy(
        name=config.distributions["sigma"]["name"],
        scale=50.0,
        event_stack=(1, num_countries),
        conditionally_independent=True,
        transform=transformations.SoftPlus(reinterpreted_batch_ndims=2),
    )
    sigma = sigma[..., tf.newaxis]  # same across age groups
    new_cases = yield pm.Deterministic(name="new_cases", value=new_cases)
    # Likelihood of the data
    likelihood = yield pm.StudentT(
        name="like",
        loc=new_cases,
        scale=sigma * tf.sqrt(new_cases) + 1,
        df=4,
        observed=config.get_data()
        .to_numpy()
        .astype("float32")
        .reshape((50, num_countries, num_age_groups))[16:],
        reinterpreted_batch_ndims=3,
    )
    return likelihood


""" # MCMC Sampling
"""
begin_time = time.time()
trace = pm.sample(
    test_model(config),
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
    test_model(config), sample_shape=5000, use_auto_batching=False
)

""" ## Plot distributions
    Function returns a list of figures which can be shown by fig[i].show() each figure being one country.
"""
dist_names = ["R", "I_0", "g_mu", "g_theta", "sigma"]
fig = {}
for name in dist_names:
    fig[name] = covid19_npis.plot.distribution(
        trace, trace_prior, config=config, key=name
    )
    # Save figure
    plt.savefig("figures/dist_" + name + ".pdf", dpi=300, transparent=True)

""" ## Plot time series for "new_cases"
"""
fig_new_cases = covid19_npis.plot.timeseries(trace, config=config, key="new_cases")

# plot data into the axes
for i, c in enumerate(config.data["countries"]):
    for j, a in enumerate(config.data["age_groups"]):
        fig_new_cases[j][i] = covid19_npis.plot.time_series._timeseries(
            config.df.index[:-16],
            config.df[(c, a)].to_numpy()[16:],
            ax=fig_new_cases[j][i],
            alpha=0.5,
        )

# Save figure
plt.savefig("figures/ts_new_cases.pdf", dpi=300, transparent=True)
