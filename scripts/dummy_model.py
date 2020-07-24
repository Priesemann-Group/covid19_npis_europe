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

""" Force GPU
"""
# my_devices = tf.config.experimental.list_physical_devices(device_type="CPU")
# tf.config.experimental.set_visible_devices(devices=my_devices, device_type="CPU")
# tf.config.set_visible_devices([], "GPU")

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

""" # Data Retrieval
    Retrieves some dummy/test data
"""
# Fixed R matrix for now one country one age group
I_new = covid19_npis.test_data.simple_new_I(0.3)
I_new = I_new.join(covid19_npis.test_data.simple_new_I(0.3))
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
    I_0 = tf.clip_by_value(I_0, 1e-9, 1e10)

    # Create Reproduction Number for every age group
    R = yield pm.LogNormal(
        name=config.distributions["R"]["name"],
        loc=1,
        scale=2.5,
        conditionally_independent=True,
        event_stack=event_shape,
        transform=transformations.Log(reinterpreted_batch_ndims=len(event_shape)),
    )
    log.debug(f"R:\n{R}")

    R_t = tf.stack(
        [R] * 50
    )  # R_t has dimensions time x batch_dims x num_countries x num_age_groups

    # Create Contact matrix

    # Use Cholesky version as the non Cholesky version uses tf.linalg.slogdet which isn't implemented in JAX
    C = yield pm.LKJCholesky(
        name=config.distributions["C"]["name"],
        dimension=num_age_groups,
        concentration=2,  # eta
        conditionally_independent=True,
        event_stack=num_countries
        # event_stack = num_countries,
        # batch_stack=batch_stack
    )  # dimensions: batch_dims x num_countries x num_age_groups x num_age_groups
    C = tf.einsum("...ab,...ba->...ab", C, C)

    # Create normalized pdf of generation interval
    g_p = yield covid19_npis.model.construct_generation_interval()
    log.debug(f"g_p:\n{g_p}")

    # Create N tensor (vector)
    # should be done earlier in the real model
    N = tf.convert_to_tensor([10e5, 10e5, 10e5, 10e5] * 2)
    N = tf.reshape(N, event_shape)
    # log.debug(f"N:\n{N}")
    # Calculate new cases
    new_cases = covid19_npis.model.InfectionModel(
        N=N, I_0=I_0, R_t=R_t, C=C, g_p=g_p  # default valueOp:AddV2
    )
    # log.debug(f"new_cases:\n{new_cases[0,:]}")  # dimensons=t,c,a

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
        observed=config.get_data().to_numpy().astype("float32").reshape((50, 2, 4)),
        reinterpreted_batch_ndims=3,
    )
    """

    psi = yield pm.HalfCauchy(
        name=config.distributions["sigma"]["name"],
        scale=50.0,
        event_stack=(1, num_countries),  # same across time
        conditionally_independent=True,
        transform=transformations.SoftPlus(reinterpreted_batch_ndims=2),
    )

    psi = psi[..., tf.newaxis]  # same across age groups

    def convert(mu, psi):
        p = mu / (mu + psi)
        return tf.clip_by_value(p, 1e-9, 1.0)

    p = convert(new_cases, psi)
    log.debug(f"r:{p}")
    new_cases = yield pm.Deterministic(name="new_cases", value=new_cases)
    likelihood = yield pm.NegativeBinomial(
        name="like",
        total_count=psi,
        probs=p,
        observed=config.get_data().to_numpy().astype("float32").reshape((50, 2, 4)),
        allow_nan_stats=True,
        reinterpreted_batch_ndims=3,
    )
    """

    return likelihood


# a = pm.sample_prior_predictive(test_model(data), sample_shape=1000, use_auto_batching=False)
begin_time = time.time()
trace = pm.sample(
    test_model(config),
    num_samples=50,
    burn_in=50,
    use_auto_batching=False,
    num_chains=2,
    xla=True,
)
"""
benchmark(
    test_model(config),
    only_xla=False,
    iters=10,
    num_chains=(4,),
    parallelize=True,
    n_evals=100,
)
"""
end_time = time.time()
log.info("running time: {:.1f}s".format(end_time - begin_time))


""" # Convert trace to nicely format (easier plotting)
    Function returns list with samples for each distribution in the config
    (see config.py)
"""

# posteriors = covid19_npis.convert_trace_to_pandas_list(test_model, trace, config)


""" # Sample for prior plots and also covert to nice format
"""
trace_prior = pm.sample_prior_predictive(
    test_model(config), sample_shape=1000, use_auto_batching=False
)

""" # Plot distributions
    Function returns a list of figures which can be shown by fig[i].show() each figure beeing one country.
"""
fig_R = covid19_npis.plot.distribution(trace, trace_prior, config=config, key="R")
fig_new_cases = covid19_npis.plot.timeseries(trace, config=config, key="new_cases")

# plot data onto axes of new_cases
for i, c in enumerate(config.data["countries"]):
    for j, a in enumerate(config.data["age_groups"]):
        fig_new_cases[j][i] = covid19_npis.plot.time_series._timeseries(
            config.df.index,
            config.df[(c, a)].to_numpy(),
            ax=fig_new_cases[j][i],
            alpha=0.5,
        )
