from collections.abc import Iterable
import logging

import tensorflow as tf
import numpy as np


log = logging.getLogger(__name__)


def gamma(x, alpha, beta):
    """
    Returns a gamma kernel evaluated at x. The implementation is the same as defined
    in the tfp.gamma distribution which is probably quiet numerically stable.
    Parameters
    ----------
    x
    alpha
    beta

    Returns
    -------

    """

    log_unnormalized_prob = tf.math.xlogy(alpha - 1.0, x) - beta * x
    log_normalization = tf.math.lgamma(alpha) - alpha * tf.math.log(beta)
    return tf.exp(log_unnormalized_prob - log_normalization)


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
        assert ndim >= max(
            max(target_axes) + 1, max(-np.array(target_axes))
        ), "ndim is smaller then the number of inferred axes from target_axes"

    target_axes = np.array(positive_axes(target_axes, ndim))
    if not len(set(target_axes)) == len(target_axes):
        raise RuntimeError(f"An axis is doubly targeted: {target_axes}")
    target_sorted = np.sort(target_axes)
    i_sorted = np.argsort(target_axes)
    i_sorted_inv = np.argsort(i_sorted)

    lacking_dims = np.diff([-1] + list(target_sorted)) - 1

    ### Permutation part:
    perm = target_axes - np.cumsum(lacking_dims)[i_sorted_inv]
    tensor = tf.transpose(tensor, perm=np.argsort(perm))

    ### Expansion part:
    append_to_end = ndim - sum(lacking_dims) - len(tensor.shape)
    lacking_dims = np.concatenate([lacking_dims, [append_to_end]])

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
    vec1=(),
    vec2=(),
    targ_outer1=(),
    targ_outer2=(),
):
    """
    Calling tf.einsum with indices instead of a string. For example
    einsum_indexed(t1, t2, inner1=1, inner2=0, outer1=0, outer2=1) corresponds to the
    `tf.einsum` string "ab...,bc...->ac..." (Matrix product) and a matrix vector product
    "...ab,...b,->...a" is parameterized by
    einsum_indexed(t1, t2, inner1=-1, inner2=-1, vec1=-2)

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
    vec1 : int or list
        The axes indices of the matrix in a matrix-vector product which are "staying"
        in the result. This is for the case where tensor1 corresponds to the matrix.
    vec2 : int or list
        The axes indices of the matrix in a matrix-vector product which are "staying"
        in the result. This is for the case where tensor2 corresponds to the matrix.
    targ_outer1 : int or list
        The axes indices in the result where the outer product axes of tensor 1 is
        mapped to. If omitted, the position is inferred such that the order stays the
        same, and, if equal, the indices of tensor 1 are to the left of the indices of tensor2 for
        outer products.
    targ_outer2 : int or list
        The axes indices in the result where the outer product axes of tensor 2 is
        mapped to. If omitted, the position is inferred such that the order stays the
        same, and, if equal, the indices of tensor 1 are to the left of the indices of tensor2 for
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
    vec1 = normalize_input(vec1)
    vec2 = normalize_input(vec2)
    targ_outer1 = normalize_input(targ_outer1)
    targ_outer2 = normalize_input(targ_outer2)

    assert len(inner1) == len(inner2)
    assert len(outer1) == len(outer2)
    assert len(targ_outer1) == len(targ_outer2)

    len_output = (
        min(len(tensor1.shape), len(tensor2.shape))
        + len(outer1)
        - len(inner1)
        + len(vec1)
        + len(vec2)
    )
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

    for pos in vec1:
        ind_inputs1[pos] = "*"
    for pos in vec2:
        ind_inputs2[pos] = "*"

    letters_broadcasting = []
    for i in range(len_output - 1, -1, -1):
        if ind_output[i] == "-":
            ind_output[i] = alphabet[0]
            letters_broadcasting.append(alphabet[0])
            alphabet = alphabet[1:]

    broadcasting_letter = ""
    broadcasting_tensor = 0
    broadcasting_index = None
    outer_tensor = 0
    tensor1_broadcast_to = [None for _ in range(len(tensor1.shape))]
    tensor2_broadcast_to = [None for _ in range(len(tensor2.shape))]
    """
    print(ind_inputs1)
    print(ind_inputs2)
    """
    for i in range(1, max(len(tensor1.shape), len(tensor2.shape)) + 1):
        input1_end = i > len(tensor1.shape)
        input2_end = i > len(tensor2.shape)

        if not input2_end and ind_inputs2[-i].isalpha():
            tensor2_broadcast_to[-i] = tensor2.shape[-i]
        if not input1_end and ind_inputs1[-i].isalpha():
            tensor1_broadcast_to[-i] = tensor1.shape[-i]

        if not input2_end and ind_inputs2[-i] == "*":
            ind_inputs2[-i] = letters_broadcasting.pop(0)
            tensor2_broadcast_to[-i] = tensor2.shape[-i]
        if not input1_end and ind_inputs1[-i] == "*":
            ind_inputs1[-i] = letters_broadcasting.pop(0)
            tensor1_broadcast_to[-i] = tensor1.shape[-i]

        if not input2_end and ind_inputs2[-i] == "!":
            if not outer_tensor:
                outer_tensor = 1
                ind_inputs2[-i] = letters_broadcasting.pop(0)
            elif outer_tensor == 2:
                outer_tensor = 0
                ind_inputs2[-i] = letters_broadcasting.pop(0)
            else:
                raise RuntimeError("Wrong parametrization of einsum")
            tensor2_broadcast_to[-i] = tensor2.shape[-i]

        if not input1_end and ind_inputs1[-i] == "!":
            if not outer_tensor:
                outer_tensor = 2
                ind_inputs1[-i] = letters_broadcasting.pop(0)
            elif outer_tensor == 1:
                outer_tensor = 0
                ind_inputs1[-i] = letters_broadcasting.pop(0)
            else:
                raise RuntimeError("Wrong parametrization of einsum")
            tensor1_broadcast_to[-i] = tensor1.shape[-i]

        if not input2_end and ind_inputs2[-i] == "-":
            if broadcasting_tensor == 0:
                broadcasting_tensor = 1
                broadcasting_letter = letters_broadcasting.pop(0)
                broadcasting_index = -i
                ind_inputs2[-i] = broadcasting_letter
            elif broadcasting_tensor == 2:
                broadcasting_tensor = 0
                ind_inputs2[-i] = broadcasting_letter
                broadcast_dim = max(
                    tensor2.shape[-i], tensor1.shape[broadcasting_index]
                )
                # print(broadcast_dim)
                tensor2_broadcast_to[-i] = broadcast_dim
                tensor1_broadcast_to[broadcasting_index] = broadcast_dim

                broadcasting_index = None
                broadcasting_letter = ""
            else:
                raise RuntimeError("Wrong parametrization of einsum")

        if not input1_end and ind_inputs1[-i] == "-":
            if broadcasting_tensor == 0:
                broadcasting_tensor = 2
                broadcasting_letter = letters_broadcasting.pop(0)
                broadcasting_index = -i
                ind_inputs1[-i] = broadcasting_letter
            elif broadcasting_tensor == 1:
                broadcasting_tensor = 0
                ind_inputs1[-i] = broadcasting_letter
                broadcast_dim = max(
                    tensor1.shape[-i], tensor2.shape[broadcasting_index]
                )
                tensor1_broadcast_to[-i] = broadcast_dim
                tensor2_broadcast_to[broadcasting_index] = broadcast_dim

                broadcasting_index = None
                broadcasting_letter = ""
            else:
                raise RuntimeError("Wrong parametrization of einsum")

    if "-" in ind_inputs1 or "-" in ind_inputs2:
        raise RuntimeError("Wrong parametrization of einsum")
    if "!" in ind_inputs1 or "!" in ind_inputs2:
        raise RuntimeError("Wrong parametrization of einsum")

    # Necessary because tf.einsum doesn't accept axis size 1 and >1 respectively for
    # the two inputs when not doing when the broadcasting is not parametrized with "..."
    tensor1 = tf.broadcast_to(tensor1, tensor1_broadcast_to)
    tensor2 = tf.broadcast_to(tensor2, tensor2_broadcast_to)

    string_einsum = (
        "".join(ind_inputs1) + "," + "".join(ind_inputs2) + "->" + "".join(ind_output)
    )

    log.debug("inferred einsum string: :", string_einsum)

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


def slice_of_axis(tensor, axis, begin, end):
    """
    Returns the tensor where the axis `axis` is sliced from `begin` to `end`
    
    Parameters
    ----------
    tensor : tensor
    axis : int
    begin : int
    end : int

    Returns
    -------
    sliced tensor

    """

    begin_axis = begin % tensor.shape[axis]
    size_axis = (end - begin) % tensor.shape[axis]

    begin_arr = np.zeros(len(tensor.shape), dtype="int")
    size_arr = np.array(tensor.shape)

    begin_arr[axis] = begin_axis
    size_arr[axis] = size_axis

    return tf.slice(tensor, begin=begin_arr, size=size_arr)


def convolution_with_fixed_kernel(
    data, kernel, data_time_axis, filter_axes_data=(), padding=None
):
    """
    Convolve data with a time independent kernel. The returned shape is equal to the shape
    of data. In order avoid constructing a time_length x time_length kernel, the data
    is decomposed in overlapping frames, with a stride of `padding`, allowing to construct
    a only padding x time_length sized kernel.

    Parameters
    ----------
    data : tensor
        The input tensor
    kernel : tensor
        Has as shape filter_axes x time. filter_axes can be several axes, where in each
        dimension a difference kernel is located
    data_time_axis : int
        the axis of data which corresponds to the time axis
    filter_axes_data : tuple
        the axes of `data`, to which the `filter_axes` of `kernel` should be mapped to.
        Each of this dimension is therefore subject to a different filter
    padding : int
        By default, the padding is set to length of data_time_axis divided by 4.
    
    Returns
    -------
    A convolved tensor with the same shape as data.
    """
    len_kernel = kernel.shape[-1]
    len_time = data.shape[data_time_axis]

    # Add batch shapes to filter axes
    while len(filter_axes_data) < len(data.shape) - 1:
        filter_axes_data = [0] + list(filter_axes_data)

    data_time_axis = positive_axes(data_time_axis, ndim=len(data.shape))

    if padding is None:
        padding = np.ceil(len_time / 4).astype("int")

    kernel_for_frame = tf.repeat(kernel[..., tf.newaxis], repeats=padding, axis=-1)

    kernel_for_frame = tf.linalg.diag(
        kernel_for_frame, k=(-len_kernel + 1, 0), num_rows=len_kernel + padding - 1
    )  # dimensions: conv_axes x copies for frame (padding) x time

    # if a filter_axis is larger then the data_time_axis, it has to be increased by one, as
    # the kernel gained a dimension:
    if filter_axes_data:
        filter_axes_data = positive_axes(filter_axes_data, len(data.shape))
        filter_axes_data_for_frame = filter_axes_data
        filter_axes_data_for_frame[filter_axes_data_for_frame > data_time_axis] += 1
    else:
        filter_axes_data_for_frame = ()

    kernel_for_frame = match_axes(
        kernel_for_frame,
        target_axes=list(filter_axes_data_for_frame)
        + [data_time_axis, data_time_axis + 1],
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

    result = slice_of_axis(result, data_time_axis, begin=0, end=len_time)

    return result


def convolution_with_varying_kernel(data, kernel, data_time_axis, filter_axes_data=()):
    """
    Convolve data with a time dependent kernel. The returned shape is equal to the shape
    of data. In this implementation, the kernel will be augmented by a time_data axis,
    and then the inner product with the date will be taken. This is not an optimal
    implementation, as the most of the entries of the kernel inner product
    matrix will be zero.

    Parameters
    ----------
    data : tensor
        The input tensor
    kernel : tensor
        Has as shape filter_axes x time_kernel x time_data. filter_axes can be several
        axes, where in each dimension a difference kernel is located
    data_time_axis : int
        the axis of data which corresponds to the time axis
    filter_axes_data : tuple
        the axes of `data`, to which the `filter_axes` of `kernel` should be mapped to.
        Each of this dimension is therefore subject to a different filter
    
    Returns
    -------
    A convolved tensor with the same shape as data.
    """
    len_kernel = kernel.shape[-2]
    len_time = data.shape[data_time_axis]
    assert (
        len_time == kernel.shape[-1]
    ), "kernel time axis is not equal to data time axis"

    # Add batch shapes to filter axes
    while len(filter_axes_data) < len(data.shape) - 1:
        filter_axes_data = [0] + list(filter_axes_data)
    data_time_axis = positive_axes(data_time_axis, ndim=len(data.shape))

    kernel = tf.linalg.diag(
        kernel, k=(-len_kernel + 1, 0), num_rows=len_time
    )  # dimensions: conv_axes x copies for frame (padding) x time

    # if a filter_axis is larger then the data_time_axis, it has to be increased by one, as
    # the kernel gained a dimension:
    if filter_axes_data:
        filter_axes_data = positive_axes(filter_axes_data, len(data.shape))
        filter_axes_data_for_conv = filter_axes_data
        filter_axes_data_for_conv[filter_axes_data_for_conv > data_time_axis] += 1
    else:
        filter_axes_data_for_conv = ()

    kernel = match_axes(
        kernel,
        target_axes=list(filter_axes_data_for_conv)
        + [data_time_axis, data_time_axis + 1],
        ndim=len(data.shape) + 1,
    )

    result = einsum_indexed(
        data,
        kernel,
        inner1=data_time_axis,
        inner2=data_time_axis + 1,
        vec2=data_time_axis,
    )
    return result


def convolution_with_map(data, kernel, modelParams):
    """
    Parameters
    ----------
    data: 
    """
    kernel_len = kernel.shape[-1]
    shape_padding = []
    for i in data.shape:
        shape_padding.append(i)
    shape_padding[-1] = kernel_len
    data_shift = tf.concat(
        values=[tf.zeros(shape_padding, dtype="float32"), data], axis=-1, name="concat"
    )
    log.info(data_shift.shape)
    log.info(kernel.shape)

    convolution = tf.map_fn(
        fn=lambda tau: tf.einsum(
            "...cat,...ct->...ca", data_shift[..., tau - kernel_len : tau], kernel,
        ),
        elems=tf.convert_to_tensor(
            np.arange(kernel_len, modelParams.length + kernel_len)
        ),
        dtype="float32",
    )
    log.info(convolution.shape)
    convolution = tf.einsum("t...ca->...tca", convolution)
    return convolution
