import sys
import pymc4 as pm
import tensorflow as tf
import numpy as np

import logging

log = logging.getLogger(__name__)

sys.path.append("../")
import covid19_npis


""" # Data Retrieval
    Retries some dummy/test data
"""
# Fixed R matrix for now one country one age group
I_new = covid19_npis.test_data.simple_new_I(1)
I_new = I_new.join(covid19_npis.test_data.simple_new_I(2), lsuffix="C1", rsuffix="C2")
num_age_groups = 4
num_countries = 2

""" # Construct pymc4 model
"""
data = I_new.to_numpy().reshape((50, 2, 4))


@pm.model
def test_model(data):
    # Create I_0
    shape_I_0 = num_age_groups * num_countries
    I_0 = yield pm.HalfCauchy(name="I_0", loc=10.0, scale=[25] * shape_I_0)
    I_0 = tf.reshape(I_0, (num_countries, num_age_groups))
    log.info(f"I_0:\n{I_0}")

    # Create Reproduktion number for every age group
    shape_R = num_age_groups * num_countries
    R = yield pm.LogNormal(name="R_age_groups", loc=[0] * shape_R, scale=2.5,)
    R = tf.reshape(R, (num_countries, num_age_groups))  # 1==time
    R_t = tf.stack([R] * 50)

    log.info(f"R:\n{R_t.shape}")

    # Create Contact matrix
    shape_C = num_countries
    C = yield pm.LKJ(
        name="Contact_matrix",
        dimension=num_age_groups,
        concentration=[2] * shape_C,  # eta
    )
    log.info(f"C:\n{C}")
    C, norm = tf.linalg.normalize(C, 1)
    log.info(f"C:\n{C}")

    # Create N tensor (vector) should be done earlier in the real model
    N = tf.convert_to_tensor([10e5, 10e5, 10e5, 10e5] * 2)
    N = tf.reshape(N, (num_countries, num_age_groups))
    log.info(f"N:\n{N}")
    new_cases = covid19_npis.model.InfectionModel(
        N=N, I_0=I_0, R_t=R_t, C=C, g=None, l=16  # default value
    )
    log.info(f"new_cases:\n{new_cases}")  # dimensons=t,c,a
    log.info(f"new_cases:\n{new_cases[:,0,:]}")

    def convert(mu, var):
        r = 1 / var
        p = mu / (mu + r)
        return r, p

    r, p = convert(new_cases, 0.2)

    log.info(f"r:{r}")
    log.info(f"p:{p}")

    likelihood = yield pm.StudentT(
        name="like", loc=new_cases, scale=100, df=4, observed=data
    )
    """
    likelihood = yield pm.NegativeBinomial(
        name="like",
        total_count=r,
        probs=p,
        observed=data.astype("float32"),
        allow_nan_stats=True,
    )
    """


trace = pm.sample(test_model(data), num_samples=50, burn_in=80, num_chains=1)
