import logging

import tensorflow as tf
import pymc4 as pm
import numpy as np

from covid19_npis import transformations

from covid19_npis.model.distributions import HalfCauchy, StudentT

log = logging.getLogger(__name__)


def studentT_likelihood(modelParams, pos_tests, total_tests, deaths):
    """
        Highlevel function for the likelihood of our model.
        At the moment there are 3 function calls inside this function:
            - positive tests :py:func:`_studentT_positive_tests`
            - total tests :py:func:`_studentT_total_tests`
            - deaths :py:func:`_studentT_deaths`


        Parameters
        ----------
        modelParams: :py:class:`covid19_npis.ModelParams`
            Instance of modelParams, mainly used for number of age groups and
            number of countries.

        pos_tests: tf.Tensor
            Inferred tensor for the number of positive tests recorded. 
            |shape| batch, time, country, age_group

        total_tests: tf.Tensor
            Inferred tensor for the number of total tests recorded. 
            |shape| batch, time, country, age_group

        deaths: tf.Tensor
            Inferred tensor for the number of total tests recorded. 
            |shape| batch, time, country, age_group

    """

    likelihood = yield _studentT_positive_tests(modelParams, pos_tests)

    if modelParams.data_summary["files"]["/tests.csv"]:
        likelihood_total_tests = yield _studentT_total_tests(modelParams, total_tests)
        # likelihood = tf.stack([likelihood, likelihood_total_tests], axis=-1)

    # if modelParams.data_summary["files"]["/deaths.csv"]:
    # likelihood_deaths = yield _studentT_deaths(modelParams, deaths)
    # likelihood = tf.stack([likelihood, likelihood_deaths], axis=-1)
    log.debug(f"likelihood:\n{likelihood}")
    return likelihood


def _studentT_positive_tests(modelParams, pos_tests):
    r"""
        Creates studentT likelihood for the recorded positive tests.

        .. math::

            \text{Add math}

        Parameters
        ----------
        modelParams: :py:class:`covid19_npis.ModelParams`
            Instance of modelParams, mainly used for number of age groups and
            number of countries.

        pos_tests: tf.Tensor
            Inferred tensor for the number of positive tests recorded. 
            |shape| batch, time, country, age_group
    """

    # Scale of the likelihood sigma
    sigma = yield HalfCauchy(
        name="sigma_likelihood_pos_tests",
        scale=50.0,
        event_stack=modelParams.num_countries,
        conditionally_independent=True,
        transform=transformations.SoftPlus(),
        shape_label="country",
    )
    sigma = sigma[..., tf.newaxis, :, tf.newaxis]  # add time and age group dimension
    log.debug(f"sigma:\n{sigma}")

    # Retrieve data from the modelParameters and create a boolean mask
    data = modelParams.pos_tests_data_tensor
    mask = ~np.isnan(data)

    len_batch_shape = len(pos_tests.shape) - 3
    likelihood = yield StudentT(
        name="likelihood_pos_tests",
        loc=tf.boolean_mask(pos_tests, mask, axis=len_batch_shape),
        scale=tf.boolean_mask(
            sigma * tf.sqrt(pos_tests) + 1, mask, axis=len_batch_shape
        ),
        df=4,
        observed=tf.boolean_mask(data, mask),
        reinterpreted_batch_ndims=1,
    )
    log.debug(f"likelihood_pos_tests:\n{likelihood}")
    tf.debugging.check_numerics(
        likelihood, "Nan in likelihood", name="likelihood_pos_tests"
    )
    return likelihood


def _studentT_total_tests(modelParams, total_tests):
    """
        Creates studentT likelihood for the recorded total tests.

        .. math::

            \text{Add math}

        Parameters
        ----------
        modelParams: :py:class:`covid19_npis.ModelParams`
            Instance of modelParams, mainly used for number of age groups and
            number of countries.

        total_tests: tf.Tensor
            Inferred tensor for the number of total tests recorded. 
            |shape| batch, time, country, age_group
    """

    # Sadly we do not have age strata for the total performed test. We sum over the
    # age groups to get a value for all ages. We can add an exception later if we find
    # data for that.
    total_tests_without_age = tf.reduce_sum(total_tests, axis=-1)

    # Scale of the likelihood sigma for each country
    sigma = yield HalfCauchy(
        name="sigma_likelihood_total_tests",
        scale=50.0,
        event_stack=modelParams.num_countries,
        conditionally_independent=True,
        transform=transformations.SoftPlus(),
        shape_label="country",
    )
    sigma = sigma[..., tf.newaxis, :]  # Add time dimension

    log.debug(f"likelihood_total_tests sigma:\n{sigma}")

    # Retrieve data from the modelParameters and create a boolean mask
    data = modelParams.total_tests_data_tensor
    mask = ~np.isnan(data)

    # Create studentT likelihood
    len_batch_shape = len(total_tests_without_age.shape) - 2
    likelihood = yield StudentT(
        name="likelihood_total_tests",
        loc=tf.boolean_mask(total_tests_without_age, mask, axis=len_batch_shape),
        scale=tf.boolean_mask(
            sigma * tf.sqrt(total_tests_without_age) + 1, mask, axis=len_batch_shape
        ),
        df=4,
        observed=tf.boolean_mask(data, mask),
        reinterpreted_batch_ndims=1,
    )
    log.debug(f"likelihood_total_tests:\n{likelihood}")
    tf.debugging.check_numerics(
        likelihood, "Nan in likelihood", name="likelihood_total"
    )
    return likelihood[..., tf.newaxis]


def _studentT_deaths(modelParams, deaths):
    """
        Creates studentT likelihood for the recorded deaths.

        .. math::

            \text{Add math}

        Parameters
        ----------
        modelParams: :py:class:`covid19_npis.ModelParams`
            Instance of modelParams, mainly used for number of age groups and
            number of countries.

        total_tests: tf.Tensor
            Inferred tensor for the number of total tests recorded. 
            |shape| batch, time, country, age_group
    """

    # Sadly we do not have age strata for the number of deaths in most countries.
    # We sum over the age groups to get a value for all ages.
    # We can add an exception later.
    deaths_without_age = tf.reduce_sum(deaths, axis=-1)

    # Scale of the likelihood sigma for each country
    sigma = yield HalfCauchy(
        name="sigma_likelihood_deaths",
        scale=50.0,
        event_stack=modelParams.num_countries,
        conditionally_independent=True,
        transform=transformations.SoftPlus(),
        shape_label="country",
    )
    log.debug(f"likelihood_deaths sigma:\n{sigma}")
    sigma = sigma[..., tf.newaxis, :]  # Add time dimension
    # Retrieve data from the modelParameters and create a boolean mask
    data = modelParams.deaths_data_tensor
    mask = ~np.isnan(data)

    # Create studentT likelihood
    len_batch_shape = len(deaths_without_age.shape) - 2
    likelihood = yield StudentT(
        name="likelihood_deaths",
        loc=tf.boolean_mask(deaths_without_age, mask, axis=len_batch_shape),
        scale=tf.boolean_mask(
            sigma * tf.sqrt(deaths_without_age) + 1, mask, axis=len_batch_shape
        ),
        df=4,
        observed=tf.boolean_mask(data, mask),
        reinterpreted_batch_ndims=1,
    )
    log.debug(f"likelihood_deaths:\n{likelihood}")
    tf.debugging.check_numerics(
        likelihood, "Nan in likelihood", name="likelihood_deaths"
    )
    return likelihood[..., tf.newaxis]
