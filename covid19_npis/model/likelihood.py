import logging

import tensorflow as tf
import pymc4 as pm
import numpy as np

from .. import transformations

from .distributions import HalfCauchy, StudentT

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

    if modelParams.data_summary["files"]["/deaths.csv"]:
        likelihood_deaths = yield _studentT_deaths(modelParams, deaths)

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

    # log.info('pos tests')
    # log.info(f"\n{pos_tests.shape}")

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
    log.debug(f"pos_tests_data_tensor:\n{data}")
    mask = np.argwhere(~np.isnan(data).flatten())

    log.info(f"data:\n{index_mask(data, mask)}")
    log.info(f"data shape:\n{index_mask(data, mask).shape}")

    len_batch_shape = len(pos_tests.shape) - 3
    likelihood = yield StudentT(
        name="likelihood_pos_tests",
        loc=index_mask(pos_tests, mask, batch_dims=len_batch_shape),
        scale=index_mask(
            sigma * tf.sqrt(pos_tests + 1), mask, batch_dims=len_batch_shape
        ),
        df=4,
        observed=index_mask(data, mask),
        reinterpreted_batch_ndims=1,
    )
    log.debug(f"likelihood_pos_tests:\n{likelihood}")
    log.info(f"likelihood_pos_tests:\n{likelihood}")
    log.info(f"likelihood_pos_tests shape:\n{likelihood.shape}")
    tf.debugging.check_numerics(
        likelihood, "Nan in likelihood", name="likelihood_pos_tests"
    )
    return likelihood


def index_mask(x, mask, batch_dims=0):
    new_shape = tuple(x.shape[:batch_dims]) + (-1,)
    x_flattened = tf.reshape(x, new_shape)
    gathered = tf.gather(x_flattened, mask, axis=-1)[..., 0]
    return gathered


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
    # log.info('total tests')
    # log.info(f"\n{total_tests.shape}")
    total_tests_without_age = tf.reduce_sum(total_tests, axis=-1)
    # log.info(f"\n{total_tests_without_age.shape}")
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
    mask = np.argwhere(~np.isnan(data).flatten())

    # Create studentT likelihood
    len_batch_shape = len(total_tests_without_age.shape) - 2
    likelihood = yield StudentT(
        name="likelihood_total_tests",
        loc=index_mask(total_tests_without_age, mask, batch_dims=len_batch_shape),
        scale=index_mask(
            sigma * tf.sqrt(total_tests_without_age + 1),
            mask,
            batch_dims=len_batch_shape,
        ),
        df=4,
        observed=index_mask(data, mask),
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

    data = modelParams.deaths_data_tensor
    # Scale of the likelihood sigma for each country
    sigma = yield HalfCauchy(
        name="sigma_likelihood_deaths",
        scale=5,
        event_stack=modelParams.num_countries,
        conditionally_independent=True,
        transform=transformations.SoftPlus(),
        shape_label="country",
    )

    # First check if we have age groups in deaths
    if data.ndim == 3:
        sigma = sigma[
            ..., tf.newaxis, :, tf.newaxis
        ]  # add time and age group dimension
        len_batch_shape = len(deaths.shape) - 3
    else:
        deaths = tf.reduce_sum(deaths, axis=-1)  # Remove age dimension via sum
        sigma = sigma[..., tf.newaxis, :]  # Add time dimension
        len_batch_shape = len(deaths.shape) - 2

    # We sum over the age groups to get a value for all ages.
    # We can add an exception later.

    log.debug(f"likelihood_deaths sigma:\n{sigma}")

    # Retrieve data from the modelParameters and create a boolean mask

    mask = np.argwhere(~np.isnan(data).flatten())

    # Create studentT likelihood

    likelihood = yield StudentT(
        name="likelihood_deaths",
        loc=index_mask(deaths, mask, batch_dims=len_batch_shape),
        scale=index_mask(sigma * tf.sqrt(deaths + 1), mask, batch_dims=len_batch_shape),
        df=4,
        observed=index_mask(data, mask),
        reinterpreted_batch_ndims=1,
    )
    log.debug(f"likelihood_deaths:\n{likelihood}")
    tf.debugging.check_numerics(
        likelihood, "Nan in likelihood", name="likelihood_deaths"
    )
    return likelihood[..., tf.newaxis]
