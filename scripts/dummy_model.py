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


@pm.model
def test_model(data):
    # Create I_0
    shape_I_0 = num_age_groups * num_countries
    I_0 = yield pm.HalfCauchy(name="I_0", loc=10.0, scale=[25] * shape_I_0)
    I_0 = tf.reshape(I_0, (num_countries, num_age_groups))
    log.info(f"I_0:\n{I_0}")

    # Create Reproduktion number for every age group
    shape_R = num_age_groups * num_countries
    R = yield pm.Normal(name="R_age_groups", loc=[5] * shape_R, scale=2.5,)
    R = tf.reshape(R, (num_countries, num_age_groups))  # 1==time
    log.info(f"R:\n{R}")

    # Create Contact matrix
    C = yield pm.LKJ(
        name="Contact_matrix",
        dimension=num_age_groups,
        concentration=[2] * num_countries,  # eta
    )
    log.info(f"C:\n{C}")

    # Create N tensor (vector) should be done earlier in the real model
    N = tf.convert_to_tensor([10e5, 10e5, 10e5, 10e5] * 2)
    N = tf.reshape(N, (num_countries, num_age_groups))
    log.info(f"N:\n{N}")
    new_cases = yield covid19_npis.model.InfectionModel(
        N=N, I_0=I_0, R_t=R, C=C, g=None, l=16  # default value
    )
    log.info(f"new_cases:\n{new_cases}")  # dimensons=t,c,a
    log.info(f"new_cases:\n{new_cases[:,0,:]}")
    """
    def convert(mu):
        r = 1 / 1e-12
        p = mu / (mu + r)
        return r, 1 - p

    print(f"r:{r}")
    print(p)

    likelihood = yield pm.NegativeBinomial(
        name="like",
        total_count=tf.cast(r, "int32"),
        probs=p,
        observed=data.astype("int32"),
        allow_nan_stats=True,
    )

    """
    # Reshape data to fit
    data = data.to_numpy().reshape((50, 2, 4))  # To match tca fromat
    print(f"data:\n{data}")
    """
    likelihood = yield pm.LogNormal(
        name="like", loc=new_cases, scale=100, observed=data
    )
    """


trace = pm.sample(test_model(I_new), num_samples=50, burn_in=80, num_chains=8)
