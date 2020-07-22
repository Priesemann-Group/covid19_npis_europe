import sys
import pymc4 as pm
import tensorflow as tf
import numpy as np
import time
import os

import logging

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

#tf.config.threading.set_inter_op_parallelism_threads(2)
#tf.config.threading.set_intra_op_parallelism_threads(2)
os.environ['XLA_FLAGS']="--xla_force_host_platform_device_count=2"

""" # Data Retrieval
    Retries some dum)my/test data
"""
# Fixed R matrix for now one country one age group
I_new = covid19_npis.test_data.simple_new_I(0.35)
I_new = I_new.join(covid19_npis.test_data.simple_new_I(0.3))
num_age_groups = 4
num_countries = 2

""" # Construct pymc4 model
    We create our own config object which holds names of the distributions
    ,shape label and the observed data. This is necessary for the data converter
    later on.
"""

config = covid19_npis.Config(I_new)


@pm.model
def test_model(config):
    # Create I_0
    I_0 = yield pm.HalfCauchy(
        name=config.distributions["I_0"]["name"],
        loc=10.0,
        scale=25,
        conditionally_independent=True,
        event_stack=(num_countries, num_age_groups),
        transform=transformations.Log(reinterpreted_batch_ndims=2),
    )
    I_0 = tf.clip_by_value(I_0, 1e-9, 1e10)
    log.info(f"I_0:\n{I_0} {type(I_0)}")

    # Create Reproduction Number for every age group
    R = yield pm.LogNormal(
        name=config.distributions["R"]["name"],
        loc=1,
        scale=2.5,
        conditionally_independent=True,
        event_stack=(num_countries, num_age_groups),
        transform=transformations.Log(reinterpreted_batch_ndims=2),
    )

    R_t = tf.stack(
        [R] * 50
    )  # R_t has dimensions time x batch_dims x num_countries x num_age_groups

    log.info(f"R:\n{R_t.shape} {type(R_t)}")

    # Create Contact matrix

    # Use Cholesky version as the non Cholesky version uses tf.linalg.slogdet which isn't implemented in JAX
    C = yield pm.LKJCholesky(
        name=config.distributions["C"]["name"],
        dimension=num_age_groups,
        concentration=tf.constant(2.),  # eta
        conditionally_independent=True,
        event_stack=num_countries
        # event_stack = num_countries,
        # batch_stack=batch_stack
    )  # dimensions: batch_dims x num_countries x num_age_groups x num_age_groups
    C = tf.einsum("...ab,...ba->...ab", C, C)
    #C = tf.eye(num_age_groups)

    log.info(f"C:\n{C} {type(C)}")
    # C, norm = tf.linalg.normalize(C, 1)
    # log.info(f"C:\n{C.shape}\n{C}")

    g_p = yield covid19_npis.model.construct_generation_interval()

    log.info(f"g_p:\n{g_p} {type(g_p)}")

    # Create N tensor (vector) should be done earlier in the real model
    N = tf.convert_to_tensor([10e5, 10e5, 10e5, 10e5] * 2)
    N = tf.reshape(N, (num_countries, num_age_groups))
    log.info(f"N:\n{N} {type(N)}")
    new_cases = covid19_npis.model.InfectionModel(
        N=N, I_0=I_0, R_t=R_t, C=C, g_p=g_p  # default valueOp:AddV2
    )
    log.info(f"new_cases:\n{new_cases[0,:]}")  # dimensons=t,c,a
    # tf.print(f"new_cases tf:\n{new_cases[-1,0]}")
    # log.info(f"new_cases:\n{new_cases[:,0,:]}")

    new_cases = tf.clip_by_value(new_cases, 1e-7, 1e9)

    sigma = yield pm.HalfCauchy(name=config.distributions["sigma"]["name"], scale=50)
    for i in range(3):
        sigma = tf.expand_dims(sigma, axis=-1)

    likelihood = yield pm.StudentT(
        name="like",
        loc=new_cases,
        scale=sigma * tf.sqrt(new_cases) + 1,
        df=4,
        observed=config.get_data().to_numpy().astype("float32").reshape((50, 2, 4)),
        reinterpreted_batch_ndims=3,
    )

    """ UNUSED NEGATIVE BINOMIAL CODE
    def convert(mu, var):
        r = 1 / var
        p = mu / (mu + r)
        return r, p

    r, p = convert(new_cases, 0.2)

    log.info(f"r:{r}")
    log.info(f"p:{p}")
    log.info(f"data:{data.shape}")
    likelihood = yield pm.NegativeBinomial(
        name="like",
        total_count=r,
        probs=p,
        observed=data.astype("float32"),
        allow_nan_stats=True,
        reinterpreted_batch_ndims=3
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
    num_chains=3,
    xla=True,
)
#benchmark(test_model(config), only_xla=False, iters=3, num_chains=(2,20,50), parallelize=True, n_evals=100)
end_time = time.time()
print("running time: {:.1f}s".format(end_time - begin_time))
