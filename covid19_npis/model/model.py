import logging

import pymc4 as pm
import tensorflow as tf
import numpy as np

# Needed to set logging level before importing other modules
# logging.basicConfig(level=logging.DEBUG)

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
from covid19_npis.model.utils import convolution_with_fixed_kernel, gamma

log = logging.getLogger(__name__)


@pm.model()
def main_model(modelParams):

    """ # Create initial Reproduction Number R_0:
    The returned R_0 tensor has the |shape| batch, country, age_group.
    """
    R_0 = yield covid19_npis.model.reproduction_number.construct_R_0(
        name="R_0",
        loc=2.0,
        scale=0.5,
        hn_scale=0.3,  # Scale parameter of HalfNormal for each country
        modelParams=modelParams,
    )
    log.debug(f"R_0:\n{R_0}")

    """ # Create time dependent reproduction number R(t):
    Create interventions and change points from model parameters and initial reproduction number.
    Finally combine to R(t).
    The returned R(t) tensor has the |shape| time, batch, country, age_group.
    """
    R_t = yield covid19_npis.model.reproduction_number.construct_R_t(R_0, modelParams)
    log.debug(f"R_t:\n{R_t}")

    """ # Create Contact matrix C:
    We use the Cholesky version as the non Cholesky version uses tf.linalg.slogdet which isn't implemented in JAX.
    The returned tensor has the |shape| batch, country, age_group, age_group.
    """
    C = yield LKJCholesky(
        name="C_cholesky",
        dimension=modelParams.num_age_groups,
        concentration=4,  # eta
        conditionally_independent=True,
        event_stack=modelParams.num_countries,
        validate_args=True,
        transform=transformations.CorrelationCholesky(),
        shape_label=("country", "age_group_i", "age_group_j"),
    )
    log.debug(f"C:\n{C}")
    # We add C to the trace via Deterministics
    C = yield Deterministic(
        name="C",
        value=tf.einsum("...an,...bn->...ab", C, C),
        shape_label=("country", "age_group_i", "age_group_j"),
    )
    # Finally we normalize C
    C, _ = tf.linalg.normalize(C, ord=1, axis=-1)
    log.debug(f"C_normalized:\n{C}")

    """ # Create generation interval g:
    """
    len_gen_interv_kernel = 12
    # Create normalized pdf of generation interval
    (
        gen_kernel,  # shape: countries x len_gen_interv,
        mean_gen_interv,  #  shape g_mu: countries x 1
    ) = yield covid19_npis.model.construct_generation_interval(l=len_gen_interv_kernel)
    log.debug(f"gen_interv:\n{gen_kernel}")

    """ # Generate exponential distribution initial infections h_0(t):
    We need to generate initial infectious before our data starts, because we do a convolution
    in the infectiousmodel loops. This convolution needs start values which we do not want
    to set to 0!
    The returned h_0(t) tensor has the |shape| time, batch, country, age_group.
    """
    h_0_t = yield covid19_npis.model.construct_h_0_t(
        modelParams=modelParams,
        len_gen_interv_kernel=len_gen_interv_kernel,
        R_t=R_t,
        mean_gen_interv=mean_gen_interv,
        mean_test_delay=0,
    )
    # Add h_0(t) to trace
    yield Deterministic(
        "h_0_t",
        tf.einsum("t...ca->...tca", h_0_t),
        shape_label=("time", "country", "age_group"),
    )
    log.debug(f"h_0(t):\n{h_0_t}")

    """ # Create population size tensor (vector) N:
    Should be done earlier in the real model i.e. in the modelParams
    The N tensor has the |shape| country, age_group.
    """
    N = tf.convert_to_tensor([1e12, 1e12, 1e12, 1e12] * modelParams.num_countries)
    N = tf.reshape(N, (modelParams.num_countries, modelParams.num_age_groups))
    log.debug(f"N:\n{N}")

    """ # Create new cases new_I(t):
    This is done via Infection dynamics in InfectionModel, see describtion
    The returned tensor has the |shape| batch, time,country, age_group.
    """
    new_I_t = covid19_npis.model.InfectionModel(
        N=N, h_0_t=h_0_t, R_t=R_t, C=C, gen_kernel=gen_kernel  # default valueOp:AddV2
    )
    log.debug(f"new_I_t:\n{new_I_t[0,:]}")  # dimensons=t,c,a

    # Clip in order to avoid infinities
    new_I_t = tf.clip_by_value(new_I_t, 1e-7, 1e9)

    # Add new_I_t to trace
    new_I_t = yield Deterministic(
        name="new_I_t", value=new_I_t, shape_label=("time", "country", "age_group"),
    )

    """ # Number of tests
    """
    # Bsplines
    B = covid19_npis.model.number_of_tests.construct_Bsplines_basis(modelParams)

    (
        phi,
        eta,
        xi,
        m,
        theta_delay,
    ) = yield covid19_npis.model.number_of_tests.construct_testing_state(
        "state", modelParams, num_knots=B.shape[-1]
    )

    phi_t = covid19_npis.model.number_of_tests.calculate_Bsplines(phi, B)
    eta_t = covid19_npis.model.number_of_tests.calculate_Bsplines(eta, B)
    xi_t = covid19_npis.model.number_of_tests.calculate_Bsplines(xi, B)
    m_t = covid19_npis.model.number_of_tests.calculate_Bsplines(m, B)

    """ # Reporting delay d:
    """
    length_kernel = 14
    log.info(f"m_t\n{m_t.shape}")
    log.info(f"theta_delay\n{theta_delay.shape}")
    log.info(f"m_t / theta_delay\n{(m_t / theta_delay).shape}")
    # Time tensor
    t = tf.range(0.01, length_kernel + 0.01, 1.0, dtype="float32")
    delay = gamma(
        t[..., tf.newaxis, tf.newaxis], m_t / theta_delay + 1.0, 1.0 / theta_delay
    )
    log.info(f"delay\n{delay.shape}")
    log.debug(f"new_cases\n{new_cases}")
    likelihood = yield covid19_npis.model.studentT_likelihood(modelParams, new_cases)

    return likelihood
