import logging

import tensorflow as tf
import pymc4 as pm
import numpy as np

from covid19_npis import transformations

log = logging.getLogger(__name__)


def studentT_likelihood(modelParams, new_cases):
    # Get scale of likelihood

    sigma = yield pm.HalfCauchy(
        name=modelParams.distributions["sigma"]["name"],
        scale=50.0,
        event_stack=(1, modelParams.num_countries),
        conditionally_independent=True,
        transform=transformations.SoftPlus(reinterpreted_batch_ndims=2),
    )
    sigma = sigma[..., tf.newaxis]  # same across age groups
    # Likelihood of the data
    data = modelParams.data_tensor
    mask = ~np.isnan(data)
    len_batch_shape = len(new_cases.shape) - 3

    log.debug(f"data:\n{tf.boolean_mask(data, mask)}")
    log.debug(f"new_cases without mask:\n{new_cases}")

    log.debug(
        f"new_cases w. mask:\n{tf.boolean_mask(new_cases, mask, axis=len_batch_shape)}"
    )

    likelihood = yield pm.StudentT(
        name="like",
        loc=tf.boolean_mask(new_cases, mask, axis=len_batch_shape),
        scale=tf.boolean_mask(
            sigma * tf.sqrt(new_cases) + 1, mask, axis=len_batch_shape
        ),
        df=4,
        observed=tf.boolean_mask(data, mask),
        reinterpreted_batch_ndims=1,
    )
    return likelihood
