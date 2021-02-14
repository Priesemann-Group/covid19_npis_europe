import logging

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from covid19_npis.model.distributions import Deterministic, HalfNormal, Normal

log = logging.getLogger(__name__)


def construct_K(
    name, modelParams, mean_K=-0.5, sigma_K=1, sigma_country=0.5, sigma_age=0.5,
):
    """
    Constructs country contact matrix :math:`K` in hierachical manner.

    TODO
    ----
    Docstring

    Parameters
    ----------
    name: str
    modelParams: 
    """

    # Hyperdistributions i.e. variance for the following Normal distributions
    K_country_sigma = yield HalfNormal(
        name=f"{name}_country_sigma",
        scale=sigma_country,
        conditionally_independent=True,
        event_stack=(1, 1),
    )
    K_age_sigma = yield HalfNormal(
        name=f"{name}_age_sigma",
        scale=sigma_age,
        conditionally_independent=True,
        event_stack=(1, 1),
    )

    #
    Delta_K_country = (
        yield Normal(
            name=f"Delta_{name}_country",
            loc=0,
            scale=1,
            conditionally_independent=True,
            event_stack=(
                modelParams.num_countries * (modelParams.num_countries - 1) // 2,
                1,
            ),
            shape_label=("country cross terms", None),
        )
    ) * K_country_sigma
    Delta_K_age = (
        yield Normal(
            name=f"Delta_{name}_age",
            loc=0,
            scale=1,
            conditionally_independent=True,
            event_stack=(1, modelParams.num_age_groups,),
            shape_label=(None, "age groups"),
        )
    ) * K_age_sigma

    Base_K = (
        yield Normal(
            name=f"Base_{name}",
            loc=0,
            scale=sigma_K,
            conditionally_independent=True,
            event_stack=(1, 1),
        )
    ) + mean_K

    # Put everything together
    K_array = Base_K + Delta_K_age + Delta_K_country

    # Transform
    K_array = tf.math.sigmoid(K_array)
    K_array = tf.clip_by_value(
        K_array, 0, 0.99
    )  # ensures off diagonal terms are smaller than diagonal terms

    size = modelParams.num_countries
    transf_array = lambda arr: normalize_matrix(
        _subdiagonal_array_to_matrix(arr, size) + tf.linalg.eye(size, dtype=arr.dtype)
    )

    # We need to put the country dimensions to the back
    # and than to the front again

    # transpose array to get country to the back!
    K_array = tf.einsum("...ca->...ac", K_array)
    K_matrix = transf_array(K_array)  # Create matrix from array
    # transpose back i.e. age_group into the back!
    K_matrix = tf.einsum("...aij->...ija", K_matrix)
    K_matrix = yield Deterministic(
        name=f"{name}",
        value=K_matrix,
        shape_label=("country_i", "country_j", "age_group"),
    )
    return K_matrix


def construct_C(
    name, modelParams, mean_C=-0.5, sigma_C=1, sigma_country=0.5, sigma_age=0.5,
):
    """
    Constructs inter-age contact matrix C, 
    
    TODO
    ----
    Docstring

    Parameters
    ----------
    name: str
    modelParams:
    mean_C:
    sigma_C:
    sigma_country:
    sigma_age:
    dim_type: str, optional
        Matrix type, possible values are 'age' and 'country'
    """

    C_country_sigma = yield HalfNormal(
        name=f"{name}_country_sigma",
        scale=sigma_country,
        conditionally_independent=True,
        event_stack=(1, 1),
    )
    C_age_sigma = yield HalfNormal(
        name=f"{name}_age_sigma",
        scale=sigma_age,
        conditionally_independent=True,
        event_stack=(1, 1),
    )

    Delta_C_country = (
        yield Normal(
            name=f"Delta_{name}_country",
            loc=0,
            scale=1,
            conditionally_independent=True,
            event_stack=(modelParams.num_countries, 1),
            shape_label=("country", None),
        )
    ) * C_country_sigma
    Delta_C_age = (
        yield Normal(
            name=f"Delta_{name}_age",
            loc=0,
            scale=1,
            conditionally_independent=True,
            event_stack=(
                1,
                modelParams.num_age_groups * (modelParams.num_age_groups - 1) // 2,
            ),
            shape_label=(None, "age groups cross terms"),
        )
    ) * C_age_sigma

    Base_C = (
        yield Normal(
            name=f"Base_{name}",
            loc=0,
            scale=sigma_C,
            conditionally_independent=True,
            event_stack=(1, 1),
        )
    ) + mean_C
    C_array = Base_C + Delta_C_age + Delta_C_country
    C_array = tf.math.sigmoid(C_array)
    C_array = tf.clip_by_value(
        C_array, 0, 0.99
    )  # ensures off diagonal terms are smaller than diagonal terms

    size = modelParams.num_age_groups
    transf_array = lambda arr: normalize_matrix(
        _subdiagonal_array_to_matrix(arr, size) + tf.linalg.eye(size, dtype=arr.dtype)
    )

    C_matrix = transf_array(C_array)
    C_matrix = yield Deterministic(
        name=f"{name}",
        value=C_matrix,
        shape_label=("country", "age_group_i", "age_group_j"),
    )
    yield Deterministic(
        name=f"{name}_mean",
        value=transf_array(tf.math.sigmoid(Base_C + Delta_C_age))[..., 0, :, :],
        shape_label=("age_group_i", "age_group_j"),
    )

    return C_matrix


def normalize_matrix(matrix):
    size = matrix.shape[-1]
    diag = tf.linalg.diag_part(matrix)
    lower_triang = tf.linalg.band_part(matrix, -1, 0)
    sub_diag = tf.linalg.set_diag(
        lower_triang, tf.zeros(matrix.shape[:-1], dtype=matrix.dtype)
    )
    sub_diag = (
        sub_diag / tf.sqrt(diag)[..., tf.newaxis, :] / tf.sqrt(diag)[..., tf.newaxis]
    )
    # sum_sub_diag = tf.reduce_sum(sub_diag, axis=(-2, -1), keepdims=True)
    sum_rows_non_diag = tf.reduce_sum(
        sub_diag + tf.linalg.matrix_transpose(sub_diag), axis=-1
    )
    norm_by = tf.reduce_max(sum_rows_non_diag, axis=-1)[..., tf.newaxis, tf.newaxis]

    diag_transf = (
        tf.eye(size, dtype=matrix.dtype) - sum_rows_non_diag[..., tf.newaxis] + norm_by
    ) / (norm_by + 1)
    sub_diag_transf = sub_diag / (norm_by + 1)
    return (
        tf.linalg.band_part(diag_transf, 0, 0)
        + sub_diag_transf
        + tf.linalg.matrix_transpose(sub_diag_transf)
    )


def _subdiagonal_array_to_matrix(array, size):
    """
    Transforms an array containing the subdiagonal elements of a matrix into a symmetric
    matrix. Keeps prepended batch_dims in the generated matrix
    Parameters
    ----------
    array: array with shape (batch_dims,) + (num_dims * (num_dims - 1) // 2,)
    size: Size of square matrix

    Returns
    -------
    matrix with shape (batch_dims,) + (num_dims, num_dims)

    """
    assert array.shape[-1] == size * (size - 1) // 2
    i = 0
    diag_indices = [0]
    for _ in range(size // 2 + 2):
        i += size
        diag_indices += [i, i + 1]
        i += 1
    diag_indices = diag_indices[:size]
    mask = np.ones(size * (size + 1) // 2, dtype=np.bool)
    mask[diag_indices] = 0
    indices = np.arange(len(mask))[mask]
    array_scattered = tf.scatter_nd(
        indices[..., None],
        tf.transpose(
            array, perm=(len(array.shape) - 1,) + tuple(range(len(array.shape) - 1))
        ),
        shape=(size * (size + 1) // 2,) + array.shape[:-1],
    )
    array_scattered = tf.transpose(
        array_scattered, perm=tuple(range(1, len(array_scattered.shape))) + (0,)
    )
    triangular = tfp.math.fill_triangular(array_scattered, upper=True)
    matrix = triangular + tf.linalg.matrix_transpose(triangular)
    return matrix
