from collections.abc import Iterable
import logging

import tensorflow as tf
import numpy as np


log = logging.getLogger(__name__)


# Un-normalized distribution pdf for generation interval
def gamma(x, alpha, beta):
    return tf.math.pow(x, (alpha - 1)) * tf.exp(-beta * x)


# Un-normalized distribution pdf for generation interval
def weibull(x, alpha, beta):
    return tf.math.pow(x, (alpha - 1)) * tf.exp(-tf.math.pow(beta * x, alpha))


def positive_axes(axes, ndim):
    """
    Given a list of axes, returns them written as positive numbers
    Parameters
    ----------
    axes : array-like, int
        list of axes, positive or negative
    ndim : int
        number of dimensions of the array
    Returns
    -------
    positive list of axes
    """
    return (ndim + np.array(axes)) % ndim


def match_axes(tensor, target_axes, ndim=None):
    """
    Extend and transpose dimensions, such that the dimension i of `tensor` is at the
    position target_axes[i]. Missing dimension are filled with size 1 dimensions.
    This is therefore a generalization of tf.expand_dims and tf.transpose and
    implemented using these. If ndim is None, the number of the dimensions of the result
    is the minimum fullfilling the requirements of target_axes

    Parameters
    ----------
    tensor : tf.Tensor
        The input tensor with len(tensor.dims) == len(target_axes)
    target_axes : list of ints
        Target positions of the dimensions. Can be negative.
    Returns
    -------
    tensor :
        The transposed and expanded tensor.

    """

    ### Preparation

    # One larger than positive values or equal to the negative values
    ndim_inferred = max(max(target_axes) + 1, max(-np.array(target_axes)))
    if ndim is None:
        ndim = ndim_inferred
    else:
        assert ndim >= max(max(target_axes) + 1, max(-np.array(target_axes)))
    target_axes = np.array(positive_axes(target_axes, ndim))
    if not len(set(target_axes)) == len(target_axes):
        raise RuntimeError(f"An axis is doubly targeted: {target_axes}")
    target_sorted = np.sort(target_axes)
    i_sorted = np.argsort(target_axes)
    i_sorted_inv = np.argsort(i_sorted)

    lacking_dims = np.diff([-1] + list(target_sorted)) - 1

    ### Permutation part:
    perm = target_axes - np.cumsum(lacking_dims)[i_sorted_inv]
    tensor = tf.transpose(tensor, perm=perm)

    ### Expansion part:
    # if len(lacking_dims) < ndim:
    #    lacking_dims = np.concatenate([lacking_dims, [ndim - len(lacking_dims)]])
    pos_dims_lacking = np.where(lacking_dims > 0)[0]
    for pos in pos_dims_lacking[::-1]:
        num_missing = lacking_dims[pos]
        for j in range(num_missing):
            tensor = tf.expand_dims(tensor, axis=pos)
    return tensor


def einsum_indexed(
    tensor1,
    tensor2,
    inner1=(),
    inner2=(),
    outer1=(),
    outer2=(),
    targ_outer1=(),
    targ_outer2=(),
):
    """
    Calling tf.einsum with indices instead of a string. For example
    einsum_indexed(t1, t2, inner1=1, inner2=0, outer1=0, outer2=1) corresponds to the
    `tf.einsum` string "ab...,bc...->ac...".

    Parameters
    ----------
    tensor1 : tensor
        Input tensor 1
    tensor2 : tensor
        Input tensor 2
    inner1 : int or list
        The axes in tensor 1 over which a inner product is taken
    inner2 : int or list
        The axes indices in tensor 2 over which a inner product is taken
    outer1 : int or list
        The axes indices in tensor 1 over which a outer product is taken
    outer2 : int or list
        The axes indices in tensor 2 over which a outer product is taken
    targ_outer1 : int or list
        The axes indices in the result where the outer product axes of tensor 1 is
        mapped to. If omitted, the position is inferred such that the order stays the
        same, and the indices of tensor 1 are to the left of the indices of tensor2 for
        outer products.
    targ_outer2
        The axes indices in the result where the outer product axes of tensor 2 is
        mapped to. If omitted, the position is inferred such that the order stays the
        same, and the indices of tensor 1 are to the left of the indices of tensor2 for
        outer products.

    Returns
    -------
    tensor

    """

    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRZTUVWXYZ"

    ind_inputs1 = ["-" for _ in tensor1.shape]
    ind_inputs2 = ["-" for _ in tensor2.shape]

    normalize_input = lambda x: tuple((x,)) if not isinstance(x, Iterable) else x
    inner1 = normalize_input(inner1)
    inner2 = normalize_input(inner2)
    outer1 = normalize_input(outer1)
    outer2 = normalize_input(outer2)
    targ_outer1 = normalize_input(targ_outer1)
    targ_outer2 = normalize_input(targ_outer2)

    assert len(inner1) == len(inner2)
    assert len(outer1) == len(outer2)
    assert len(targ_outer1) == len(targ_outer2)

    len_output = max(len(tensor1.shape), len(tensor1.shape)) + len(outer1) - len(inner1)
    ind_output = ["-" for _ in range(len_output)]

    for pos1, pos2 in zip(inner1, inner2):
        ind_inputs1[pos1] = alphabet[0]
        ind_inputs2[pos2] = alphabet[0]
        alphabet = alphabet[1:]

    for i, (pos1, pos2) in enumerate(zip(outer1, outer2)):
        if targ_outer1:
            ind_inputs1[pos1] = alphabet[0]
            ind_output[targ_outer1[i]] = alphabet[0]
            alphabet = alphabet[1:]
        else:
            ind_inputs1[pos1] = "!"
        if targ_outer2:
            ind_inputs2[pos2] = alphabet[0]
            ind_output[targ_outer2[i]] = alphabet[0]
            alphabet = alphabet[1:]
        else:
            ind_inputs2[pos2] = "!"

    letters_broadcasting = []
    for i in range(len_output - 1, -1, -1):
        if ind_output[i] == "-":
            ind_output[i] = alphabet[0]
            letters_broadcasting.append(alphabet[0])
            alphabet = alphabet[1:]

    broadcasting_letter = ""
    broadcasting_tensor = 0
    outer_tensor = 0
    for i in range(1, max(len(tensor1.shape), len(tensor2.shape)) + 1):
        if ind_inputs2[-i] == "!":
            if not outer_tensor:
                outer_tensor = 1
                ind_inputs2[-i] = letters_broadcasting.pop(0)
            elif outer_tensor == 2:
                outer_tensor = 0
                ind_inputs2[-i] = letters_broadcasting.pop(0)
            else:
                raise RuntimeError("Wrong parametrization of einsum")

        if ind_inputs1[-i] == "!":
            if not outer_tensor:
                outer_tensor = 2
                ind_inputs1[-i] = letters_broadcasting.pop(0)
            elif outer_tensor == 1:
                outer_tensor = 0
                ind_inputs1[-i] = letters_broadcasting.pop(0)
            else:
                raise RuntimeError("Wrong parametrization of einsum")

        if ind_inputs2[-i] == "-":
            if broadcasting_tensor == 0:
                broadcasting_tensor = 1
                broadcasting_letter = letters_broadcasting.pop(0)
                ind_inputs2[-i] = broadcasting_letter
            elif broadcasting_tensor == 2:
                broadcasting_tensor = 0
                ind_inputs2[-i] = broadcasting_letter
                broadcasting_letter = ""
            else:
                raise RuntimeError("Wrong parametrization of einsum")

        if ind_inputs1[-i] == "-":
            if broadcasting_tensor == 0:
                broadcasting_tensor = 2
                broadcasting_letter = letters_broadcasting.pop(0)
                ind_inputs1[-i] = broadcasting_letter
            elif broadcasting_tensor == 1:
                broadcasting_tensor = 0
                ind_inputs1[-i] = broadcasting_letter
                broadcasting_letter = ""
            else:
                raise RuntimeError("Wrong parametrization of einsum")

    if "-" in ind_inputs1 or "-" in ind_inputs2:
        raise RuntimeError("Wrong parametrization of einsum")
    if "!" in ind_inputs1 or "!" in ind_inputs2:
        raise RuntimeError("Wrong parametrization of einsum")

    string_einsum = (
        "".join(ind_inputs1) + "," + "".join(ind_inputs2) + "->" + "".join(ind_output)
    )

    log.debug(string_einsum)
    print(string_einsum)

    return tf.einsum(string_einsum, tensor1, tensor2)


def concatenate_axes(tensor, axis1, axis2):
    """
    Concatenates two consecutive axess
    Parameters
    ----------
    tensor : tensor
        input
    axis1 : int
        first axis
    axis2 : int
        second axis

    Returns
    -------
    Concatenated tensor

    """
    assert axis2 == axis1 + 1
    shape = tensor.shape
    return tf.reshape(
        tensor, shape[:axis1] + (shape[axis1] * shape[axis2],) + shape[axis2 + 1 :]
    )


def convolution_with_fixed_kernel(
    data, kernel, data_time_axis=0, conv_axes_data=(), padding=None
):
    """

    Parameters
    ----------
    data
    kernel : time x conv_axes
    data_time_axis : has to be to the left of the conv_axes
    conv_axes_data :
    padding

    Returns
    -------

    """
    len_kernel = kernel.shape[0]
    data_time_axis = positive_axes(data_time_axis, ndim=len(data.shape))

    if padding is None:
        padding = len_kernel

    kernel_for_frame = tf.repeat(kernel[..., tf.newaxis], repeats=padding, axis=-1)
    kernel_for_frame = tf.linalg.diag(
        kernel_for_frame, k=(-len_kernel + 1, 0), num_rows=len_kernel + padding - 1
    )  # dimensions: copies for frame (padding) x time x conv_axes

    kernel_for_frame = match_axes(
        kernel_for_frame,
        target_axes=[data_time_axis, data_time_axis + 1] + list(conv_axes_data),
        ndim=len(data.shape) + 1,
    )

    data_framed = tf.signal.frame(
        data,
        frame_length=len_kernel + padding - 1,
        frame_step=padding,
        pad_end=True,
        axis=data_time_axis,
    )

    result = einsum_indexed(
        data_framed,
        kernel_for_frame,
        inner1=data_time_axis + 1,
        inner2=data_time_axis,
        outer1=data_time_axis,
        outer2=data_time_axis + 1,
    )

    result = concatenate_axes(result, data_time_axis, data_time_axis + 1)

    return result
