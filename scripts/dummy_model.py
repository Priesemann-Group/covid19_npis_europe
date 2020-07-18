import sys
import pymc4 as pm
import tensorflow as tf
import numpy as np

import logging

log = logging.getLogger(__name__)

sys.path.append("../")
import covid19_npis
from covid19_npis import transformations

##For eventual debugging:
#tf.config.run_functions_eagerly(True)
#tf.debugging.enable_check_numerics(
#    stack_height_limit=30, path_length_limit=50
#)

""" # Data Retrieval
    Retries some dum)my/test data
"""
# Fixed R matrix for now one country one age group
I_new = covid19_npis.test_data.simple_new_I(0.35)
I_new = I_new.join(covid19_npis.test_data.simple_new_I(0.3), lsuffix="C1", rsuffix="C2")
num_age_groups = 4
num_countries = 2

""" # Construct pymc4 model
"""
data = I_new.to_numpy().reshape((50, 2, 4))


@pm.model
def test_model(data):
    # Create I_0
    I_0 = yield pm.HalfCauchy(name="I_0", loc=10.0, scale=num_countries*[num_age_groups*[25]], conditionally_independent=True,
                              reinterpreted_batch_ndims=2, transform=transformations.Log(reinterpreted_batch_ndims=2))

    log.info(f"I_0:\n{I_0}")

    # Create Reproduktion number for every age group
    R = yield pm.LogNormal(name="R_age_groups", loc=num_countries*[num_age_groups*[1]], scale=2.5, conditionally_independent=True,
                           reinterpreted_batch_ndims=2, transform=transformations.Log(reinterpreted_batch_ndims=2))

    R_t = tf.stack([R] * 50)

    log.info(f"R:\n{R_t.shape}")

    # Create Contact matrix
    shape_C = num_countries
    #batch_stack = None if len(R_t.shape) == 3 else 3
    C = yield pm.LKJ(
        name="Contact_matrix",
        dimension=num_age_groups,
        concentration=[2]*num_countries,  # eta
        conditionally_independent = True,
        reinterpreted_batch_ndims=1
        #event_stack = num_countries,
        #batch_stack=batch_stack
    )

    log.info(f"C:\n{C}")
    #C, norm = tf.linalg.normalize(C, 1)
    #log.info(f"C:\n{C.shape}\n{C}")

    # Create N tensor (vector) should be done earlier in the real model
    N = tf.convert_to_tensor([10e5, 10e5, 10e5, 10e5] * 2)
    N = tf.reshape(N, (num_countries, num_age_groups))
    log.info(f"N:\n{N}")
    new_cases = covid19_npis.model.InfectionModel(
        N=N, I_0=I_0, R_t=R_t, C=C, g=None, l=16  # default valueOp:AddV2
    )
    log.info(f"new_cases:\n{new_cases[0,:]}")  # dimensons=t,c,a
    #tf.print(f"new_cases tf:\n{new_cases[-1,0]}")
    #log.info(f"new_cases:\n{new_cases[:,0,:]}")

    def convert(mu, var):
        r = 1 / var
        p = mu / (mu + r)
        return r, p

    r, p = convert(new_cases, 0.2)

    log.info(f"r:{r}")
    log.info(f"p:{p}")
    log.info(f"data:{data.shape}")

    sigma = yield pm.HalfCauchy(name='scale_likelihood', scale=50)
    for i in range(3):
        sigma=tf.expand_dims(sigma, axis=-1)

    likelihood = yield pm.StudentT(
        name="like", loc=new_cases, scale=sigma*tf.sqrt(new_cases)+1, df=4, observed=data.astype("float32"),
        reinterpreted_batch_ndims=3
    )

    #likelihood = yield pm.NegativeBinomial(
    #    name="like",
    #    total_count=r,
    #    probs=p,
    #    observed=data.astype("float32"),
    #    allow_nan_stats=True,
    #    reinterpreted_batch_ndims=3
   #)


    return likelihood
#a = pm.sample_prior_predictive(test_model(data), sample_shape=1000, use_auto_batching=False)

trace = pm.sample(test_model(data), num_samples=10, burn_in=10, use_auto_batching=False, num_chains=1)


