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
from covid19_npis.model.utils import convolution_with_varying_kernel, gamma

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
    log.debug(f"new_I_t\n{new_I_t.shape}")

    """ # Number of tests
    TODO - short description here/ maybe but all of that into a helper function
    """
    # Get basic functions for b-splines (used later)
    B = covid19_npis.model.number_of_tests.construct_Bsplines_basis(modelParams)

    (
        phi,
        eta,
        xi,
        m_star,
    ) = yield covid19_npis.model.number_of_tests.construct_testing_state(
        "testing_state", modelParams, num_knots=B.shape[-1]
    )

    # Transform m_star and construct reporting delay
    m, theta = yield covid19_npis.model.number_of_tests._construct_reporting_delay(
        name="delay", modelParams=modelParams, m_ast=m_star
    )

    # Construct Bsplines
    phi_t = covid19_npis.model.number_of_tests.calculate_Bsplines(phi, B)
    eta_t = covid19_npis.model.number_of_tests.calculate_Bsplines(eta, B)
    xi_t = covid19_npis.model.number_of_tests.calculate_Bsplines(xi, B)
    m_t = covid19_npis.model.number_of_tests.calculate_Bsplines(m, B)

    # Reporting delay d:
    delay = covid19_npis.model.number_of_tests.calc_reporting_kernel(m_t, theta)
    delay = yield Deterministic(
        "delay_kernel", delay, shape_label=("country", "kernel", "time")
    )
    log.info(f"kernel\n{delay}")  # batch, country, kernel, time
    log.info(f"new_I_t\n{new_I_t}")  # batch, time, country, age_group

    filter_axes_data = covid19_npis.model.utils.get_filter_axis_data_from_dims(
        len(new_I_t.shape)
    )
    new_cases_delayed = convolution_with_varying_kernel(
        data=new_I_t, kernel=delay, data_time_axis=-3, filter_axes_data=filter_axes_data
    )
    new_cases_delayed = yield Deterministic(
        "new_cases_delayed",
        new_cases_delayed,
        shape_label=("time", "country", "age_group"),
    )
    log.debug(f"new_cases_delayed\n{new_cases_delayed}")

    # Calc positive tests
    phi_age = yield covid19_npis.model.number_of_tests._construct_phi_age(
        "phi_age", modelParams
    )
    positive_tests = covid19_npis.model.number_of_tests.calc_positive_tests(
        new_cases_delayed=new_cases_delayed,
        phi_plus=phi_t,
        phi_age=phi_age,
        modelParams=modelParams,
    )
    positive_tests = yield Deterministic(
        "positive_tests", positive_tests, shape_label=("time", "country", "age_group")
    )
    log.debug(f"positive_tests\n{positive_tests}")

    # Calc total number of tests
    phi_tests_reported = yield covid19_npis.model.number_of_tests._construct_phi_tests_reported(
        name="phi_tests_reported", modelParams=modelParams
    )
    total_tests = covid19_npis.model.number_of_tests.calc_total_number_of_tests_performed(
        new_cases_delayed=new_cases_delayed,
        phi_tests_reported=phi_tests_reported,
        phi_plus=phi_t,
        eta=eta_t,
        xi=xi_t,
        modelParams=modelParams,
    )
    total_tests = yield Deterministic(
        "total_tests", total_tests, shape_label=("time", "country", "age_group")
    )
    log.debug(f"total_tests\n{total_tests}")

    likelihood = yield covid19_npis.model.studentT_likelihood(
        modelParams, positive_tests
    )

    return likelihood
