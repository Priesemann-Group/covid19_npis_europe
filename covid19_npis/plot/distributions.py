from .rcParams import *
from ..data import convert_trace_to_dataframe, select_from_dataframe
from .. import modelParams

from .utils import (
    get_model_name_from_sample_state,
    get_dist_by_name_from_sample_state,
    check_for_shape_label,
    get_math_from_name,
    get_shape_from_dataframe,
)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats

import logging

log = logging.getLogger(__name__)
plt.rcParams.update({"figure.max_open_warning": 0})


def _plot_prior(x, ax=None, **kwargs):
    """
    Low level plotting function, plots the prior as line for sampling data by using kernel density estimation.
    For more references see `scipy documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html>`_.
    
    It is highly recommended to pass an axis otherwise the xlim may be a bit wonky. 

    Parameters
    ----------
    x : 
        Input values, from sampling

    ax : mpl axes element, optional
        Plot into an existing axes element
        |default| :code:`None`

    kwargs : dict, optional
        Directly passed to plotting mpl.
    """
    reset_xlim = False
    if ax is None:
        fig, ax = plt.subplots()
        xlim = [x.min(), x.max()]
    else:
        # may need to convert axes values, and restore xlimits after adding prior
        xlim = ax.get_xlim()
        reset_xlim = True

    prior = stats.kde.gaussian_kde(x)
    x_for_ax = np.linspace(*xlim, num=1000)
    x_for_pr = x_for_ax

    ax.plot(
        x_for_ax,
        prior(x_for_ax),
        label="Prior",
        color=rcParams.color_prior,
        linewidth=3,
        **kwargs,
    )

    if reset_xlim:
        ax.set_xlim(*xlim)

    return ax


def _plot_posterior(x, bins=50, ax=None, **kwargs):
    """
    Low level plotting function to plot an sampling data as histogram.

    Parameters
    ----------
    x: 
        Input values, from sampling

    bins: int, optional 
        Defines the number of equal-width bins in the range.
        |default| 50

    ax : mpl axes element, optional
        Plot into an existing axes element
        |default| :code:`None`

    kwargs : dict, optional
        Directly passed to plotting mpl.
    """
    if ax is None:
        fig, ax = plt.subplots()

    ax.hist(
        x,
        bins=bins,
        color=rcParams.color_posterior,
        label="Posterior",
        alpha=0.7,
        zorder=-5,
        density=True,
        **kwargs,
    )

    return ax


def distribution(trace_posterior, trace_prior, sample_state, key):
    """
    High level function for creating plot overview for a variable. Works for
    one and two dimensional variable at the moment.

    TODO
    ----
    - ndim=3
    

    Parameters
    ----------
    trace_posterior,trace_prior : arivz InferenceData
        Raw data from pymc4 sampling

    sample_state : pymc4 sample state

    key : str
        Name of the variable to plot


    Returns
    -------
    array of mpl figures
        one figure for each country

    """
    log.info(f"Creating distibution plot for {key}")

    # Get prior and posterior data for key
    posterior = convert_trace_to_dataframe(trace_posterior, sample_state, key)
    prior = convert_trace_to_dataframe(trace_prior, sample_state, key)

    # Get other model parameters which we need
    model_name = get_model_name_from_sample_state(sample_state)
    dist = get_dist_by_name_from_sample_state(sample_state, key)
    check_for_shape_label(dist)

    # Get shape from data, should be the same for posterior and prior
    shape = get_shape_from_dataframe(posterior)

    def dist_ndim_1():
        # E.g. only age groups or only one value over all age groups
        rows = shape[0]
        cols = 1

        if hasattr(dist, "shape_label"):
            label1 = dist.shape_label
        else:
            label1 = f"{model_name}/{dist.name}_dim_0"

        fig, ax = plt.subplots(rows, cols, figsize=(4.5 / 3 * cols, rows * 1))
        if rows == 1:
            # Flatten chains and other sampling dimensions of df into one array
            array_posterior = posterior.to_numpy().flatten()
            array_prior = prior.to_numpy().flatten()
            return (
                fig,
                _distribution(
                    array_posterior=array_posterior,
                    array_prior=array_prior,
                    dist_name=dist.name,
                    dist_math=get_math_from_name(dist.name),
                    ax=ax,
                ),
            )
        else:
            for i, value in enumerate(
                posterior.index.get_level_values(label1).unique()
            ):
                # Flatten chains and other sampling dimensions of df into one array
                array_posterior = posterior.xs(value, level=label1).to_numpy().flatten()
                array_prior = prior.xs(value, level=label1).to_numpy().flatten()
                ax[i] = _distribution(
                    array_posterior=array_posterior,
                    array_prior=array_prior,
                    dist_name=dist.name,
                    dist_math=get_math_from_name(dist.name),
                    ax=ax[i],
                    suffix=f"{i}",
                )

            # Set labels on y-axis
            for i in range(cols):
                ax[i].set_xlabel(posterior.index.get_level_values(label1).unique()[i])

            return fig, ax

    def dist_ndim_2():

        # In the default case: label1 should be country and label2 should age_group
        if hasattr(dist, "shape_label"):
            label1, label2 = dist.shape_label
        else:
            label1 = model_name + "/" + dist.name + "_dim_0"
            label2 = model_name + "/" + dist.name + "_dim_1"

        # First label is rows
        # Second label is columns
        cols, rows = shape

        fig, ax = plt.subplots(
            rows, cols, figsize=(4.5 / 3 * cols, rows * 1), constrained_layout=True,
        )
        for i, value1 in enumerate(posterior.index.get_level_values(label1).unique()):
            for j, value2 in enumerate(
                posterior.index.get_level_values(label2).unique()
            ):
                # Select values from datafram
                arry_posterior = (
                    select_from_dataframe(posterior, **{label1: value1, label2: value2})
                    .to_numpy()
                    .flatten()
                )

                array_prior = (
                    select_from_dataframe(prior, **{label1: value1, label2: value2})
                    .to_numpy()
                    .flatten()
                )

                ax[j][i] = _distribution(
                    array_posterior=arry_posterior,
                    array_prior=array_prior,
                    dist_name=dist.name,
                    dist_math=get_math_from_name(dist.name),
                    ax=ax[j][i],
                    suffix=f"{i},{j}",
                )

        # Set labels on y-axis
        for i in range(cols):
            ax[0][i].set_xlabel(posterior.index.get_level_values(label1).unique()[i])
        # Set labels on x-axis
        for i in range(rows):
            ax[i][0].set_ylabel(posterior.index.get_level_values(label2).unique()[i])

        return fig, ax

    def dist_ndim_3():
        # In the default case: label2 should be country and label3 should age_group
        if hasattr(dist, "shape_label"):
            label1, label2, label3 = dist.shape_label
        else:
            label1 = model_name + "/" + dist.name + "_dim_0"
            label2 = model_name + "/" + dist.name + "_dim_1"
            label3 = model_name + "/" + dist.name + "_dim_2"

        # First label is rows
        # Second label is columns
        z, cols, rows = shape
        cols = cols
        rows = rows * z

        fig, ax = plt.subplots(rows, cols, figsize=(5 / 3 * cols, rows * 2),)
        for i, value1 in enumerate(posterior.index.get_level_values(label1).unique()):
            for j, value2 in enumerate(
                posterior.index.get_level_values(label2).unique()
            ):  # Cols
                for k, value3 in enumerate(
                    posterior.index.get_level_values(label3).unique()
                ):  # Rows
                    # Select values from datafram
                    arry_posterior = (
                        select_from_dataframe(
                            posterior,
                            **{label1: value1, label2: value2, label3: value3},
                        )
                        .to_numpy()
                        .flatten()
                    )
                    array_prior = (
                        select_from_dataframe(
                            prior, **{label1: value1, label2: value2, label3: value3}
                        )
                        .to_numpy()
                        .flatten()
                    )

                    # Flatten to 2d
                    ax[k + rows * i][j] = _distribution(
                        array_posterior=arry_posterior,
                        array_prior=array_prior,
                        dist_name=dist.name,
                        dist_math=get_math_from_name(dist.name),
                        ax=ax[k + rows * i][j],
                        suffix=f"{j},{k + rows * i}",
                    )
                    ax[k + rows * i][j].set_title(f"{value1}\n{value2}\n{value3}")

        return fig, ax

    # ------------------------------------------------------------------------------ #
    # CASES
    # ------------------------------------------------------------------------------ #
    if len(shape) == 1:
        fig, axes = dist_ndim_1()
    elif len(shape) == 2:
        fig, axes = dist_ndim_2()
    elif len(shape) == 3:
        fig, axes = dist_ndim_3()
    # ------------------------------------------------------------------------------ #
    # Titles and other
    # ------------------------------------------------------------------------------ #
    fig.suptitle(key)

    return axes


def _distribution(
    array_posterior, array_prior, dist_name, dist_math, suffix="", ax=None
):
    """
    Low level function to plots posterior and prior from arrays.

    Parameters
    ----------
    array_posterior,array_prior : array
        Sampling data as array, should be filtered beforehand.

    dist_name: str
        name of distribution for plotting
    dist_math: str
        math of distribution for plotting

    suffix: str,optional
        Suffix for the plot title e.g. "age_group_1"
        |default| ""

    ax : mpl axes element, optional
        Plot into an existing axes element
        |default| :code:`None`


    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5 / 3, 1), constrained_layout=True)

    # ------------------------------------------------------------------------------ #
    # Plot
    # ------------------------------------------------------------------------------ #
    ax = _plot_posterior(x=array_posterior, ax=ax)
    ax = _plot_prior(x=array_prior, ax=ax)

    # ------------------------------------------------------------------------------ #
    # Annotations
    # ------------------------------------------------------------------------------ #
    # add the overlay with median and CI values. these are two strings
    text_md, text_ci = _string_median_CI(array_posterior, prec=2)
    text_md = f"${dist_math}^{{{suffix}}}={text_md}$"

    # create the inset text elements, and we want a bounding box around the compound
    try:
        tel_md = ax.text(
            0.6,
            0.9,
            text_md,
            fontsize=12,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="center",
            zorder=100,
        )
        x_min, x_max, y_min, y_max = _get_mpl_text_coordinates(tel_md, ax)
        tel_ci = ax.text(
            0.6,
            y_min * 0.9,  # let's have a ten perecent margin or so
            text_ci,
            fontsize=9,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="center",
            zorder=101,
        )
        _add_mpl_rect_around_text(
            [tel_md, tel_ci], ax, facecolor="white", alpha=0.5, zorder=99,
        )
    except Exception as e:
        log.info(f"Unable to create inset with {dist_name} value: {e}")

    # ------------------------------------------------------------------------------ #
    # Additional plotting settings
    # ------------------------------------------------------------------------------ #
    ax.xaxis.set_label_position("top")
    # ax.set_xlabel(dist["name"] + suffix)
    ax.set_xlim(0)

    ax.tick_params(labelleft=False)
    ax.set_rasterization_zorder(rcParams.rasterization_zorder)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    return ax


# ------------------------------------------------------------------------------ #
# Formating and util
# ------------------------------------------------------------------------------ #


def _string_median_CI(arr, prec=2):
    f_trunc = lambda n: _truncate_number(n, prec)
    med = f_trunc(np.median(arr))
    perc1, perc2 = (
        f_trunc(np.percentile(arr, q=2.5)),
        f_trunc(np.percentile(arr, q=97.5)),
    )
    # return "Median: {}\nCI: [{}, {}]".format(med, perc1, perc2)
    return f"{med}", f"[{perc1}, {perc2}]"


def _truncate_number(number, precision):
    return "{{:.{}f}}".format(precision).format(number)


def _get_mpl_text_coordinates(text, ax):
    """
        helper to get coordinates of a text object in the coordinates of the
        axes element [0,1].
        used for the rectangle backdrop.

        Returns:
        x_min, x_max, y_min, y_max
    """
    fig = ax.get_figure()

    try:
        fig.canvas.renderer
    except Exception as e:
        log.debug(e)
        # otherwise no renderer, needed for text position calculation
        fig.canvas.draw()

    x_min = None
    x_max = None
    y_min = None
    y_max = None

    # get bounding box of text
    transform = ax.transAxes.inverted()
    try:
        bb = text.get_window_extent(renderer=fig.canvas.get_renderer())
    except:
        bb = text.get_window_extent()
    bb = bb.transformed(transform)
    x_min = bb.get_points()[0][0]
    x_max = bb.get_points()[1][0]
    y_min = bb.get_points()[0][1]
    y_max = bb.get_points()[1][1]

    return x_min, x_max, y_min, y_max


def _add_mpl_rect_around_text(text_list, ax, x_padding=0.05, y_padding=0.05, **kwargs):
    """
        add a rectangle to the axes (behind the text)

        provide a list of text elements and possible options passed to
        mpl.patches.Rectangle
        e.g.
        facecolor="grey",
        alpha=0.2,
        zorder=99,
    """

    x_gmin = 1
    y_gmin = 1
    x_gmax = 0
    y_gmax = 0

    for text in text_list:
        x_min, x_max, y_min, y_max = _get_mpl_text_coordinates(text, ax)
        if x_min < x_gmin:
            x_gmin = x_min
        if y_min < y_gmin:
            y_gmin = y_min
        if x_max > x_gmax:
            x_gmax = x_max
        if y_max > y_gmax:
            y_gmax = y_max

    # coords between 0 and 1 (relative to axes) add 10% margin
    y_gmin = np.clip(y_gmin - y_padding, 0, 1)
    y_gmax = np.clip(y_gmax + y_padding, 0, 1)
    x_gmin = np.clip(x_gmin - x_padding, 0, 1)
    x_gmax = np.clip(x_gmax + x_padding, 0, 1)

    rect = mpl.patches.Rectangle(
        (x_gmin, y_gmin),
        x_gmax - x_gmin,
        y_gmax - y_gmin,
        transform=ax.transAxes,
        **kwargs,
    )

    ax.add_patch(rect)
