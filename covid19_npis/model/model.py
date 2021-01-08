import logging

import pymc4 as pm
import tensorflow as tf
import numpy as np

# Needed to set logging level before importing other modules
# logging.basicConfig(level=logging.DEBUG)

from . import *

from .. import transformations

#  from covid19_npis.benchmarking import benchmark

from .distributions import (
    LKJCholesky,
    Deterministic,
    Gamma,
    HalfCauchy,
    Normal,
    LogNormal,
)
from .utils import convolution_with_varying_kernel, gamma

log = logging.getLogger(__name__)


@pm.model()
def main_model(modelParams):
    """
    ToDo:
    -----
    Create Docstring for this function.
    """

    """# Create initial Reproduction Number R_0:
    The returned R_0 tensor has the |shape| batch, country, age_group.
    """
    R_0 = yield reproduction_number.construct_R_0(
        name="R_0_c",
        modelParams=modelParams,
        loc=3.3,
        scale=0.5,
        hn_scale=0.3,  # Scale parameter of HalfNormal for each country
    )

    """ # Create time dependent reproduction number R(t):
    Create interventions and change points from model parameters and initial reproduction number.
    Finally combine to R(t).
    The returned R(t) tensor has the |shape| time, batch, country, age_group.
    """
    R_t = yield reproduction_number.construct_R_t(
        name="R_t", modelParams=modelParams, R_0=R_0
    )
    log.debug(f"R_t:\n{R_t}")

    """ # Create Contact matrix C:
    We use the Cholesky version as the non Cholesky version uses tf.linalg.slogdet which isn't implemented in JAX.
    The returned tensor has the |shape| batch, country, age_group, age_group.
    """
    C = yield construct_C(name="C", modelParams=modelParams)
    log.debug(f"C:\n{C}")

    """ # Create generation interval g:
    """
    len_gen_interv_kernel = 12
    # Create normalized pdf of generation interval
    (
        gen_kernel,  # shape: countries x len_gen_interv,
        mean_gen_interv,  #  shape g_mu: countries x 1
    ) = yield construct_generation_interval(l=len_gen_interv_kernel)
    log.debug(f"gen_interv:\n{gen_kernel}")

    """ # Generate exponential distribution initial infections E_0(t):
    We need to generate initial infectious before our data starts, because we do a convolution
    in the infectiousmodel loops. This convolution needs start values which we do not want
    to set to 0!
    The returned E_0(t) tensor has the |shape| time, batch, country, age_group.
    """
    E_0_t = yield construct_E_0_t(
        modelParams=modelParams,
        len_gen_interv_kernel=len_gen_interv_kernel,
        R_t=R_t,
        mean_gen_interv=mean_gen_interv,
        mean_test_delay=0,
    )
    # Add E_0(t) to trace
    yield Deterministic(
        name="E_0_t",
        value=tf.einsum("t...ca->...tca", E_0_t),
        shape_label=("time", "country", "age_group"),
    )
    log.debug(f"E_0(t):\n{E_0_t}")

    """ # Get population size tensor from modelParams:
    Should be done earlier in the real model i.e. in the modelParams
    The N tensor has the |shape| country, age_group.
    """
    N = modelParams.N_data_tensor
    log.debug(f"N:\n{N}")

    """ # Create new cases new_E(t):
    This is done via Infection dynamics in InfectionModel, see describtion
    The returned tensor has the |shape| batch, time,country, age_group.
    """
    new_E_t = InfectionModel(
        N=N, E_0_t=E_0_t, R_t=R_t, C=C, gen_kernel=gen_kernel  # default valueOp:AddV2
    )
    log.debug(f"new_E_t:\n{new_E_t[0,:]}")  # dimensons=t,c,a

    # Clip in order to avoid infinities
    new_E_t = tf.clip_by_value(new_E_t, 1e-7, 1e9)

    # Add new_E_t to trace
    new_E_t = yield Deterministic(
        name="new_E_t", value=new_E_t, shape_label=("time", "country", "age_group"),
    )
    log.debug(f"new_E_t\n{new_E_t.shape}")

    """ # Number of tests and deaths
        We simulate our reported cases i.e positiv test and totalnumber of tests total
        and deaths.
    """
    # Tests
    total_tests, positive_tests = yield number_of_tests.generate_testing(
        name_total="total_tests",
        name_positive="positive_tests",
        modelParams=modelParams,
        new_E_t=new_E_t,
    )

    # Deaths

    # Infection fatality ratio
    death_Phi = yield deaths._calc_Phi_IFR(name="IFR", modelParams=modelParams)
    # Death reporting delay
    death_m, death_theta = yield deaths._construct_reporting_delay(
        name="delay_deaths", modelParams=modelParams
    )
    # Calculate new deaths delayed
    deaths_delayed = yield deaths.calc_delayed_deaths(
        name="deaths",
        new_cases=new_E_t,
        Phi_IFR=death_Phi,
        m=death_m,
        theta=death_theta,
    )

    """ Likelihood
    TODO    - description on fitting data
            - add deaths and total tests
    """

    likelihood = yield studentT_likelihood(
        modelParams, positive_tests, total_tests, deaths_delayed
    )
    return likelihood
