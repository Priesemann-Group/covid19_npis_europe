from .rcParams import *
from ..data import convert_trace_to_dataframe

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats

import logging

log = logging.getLogger(__name__)


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


def distribution(trace_posterior, trace_prior, config, key):
    """
    High level function for creating plot overview for a variable.
    

    Parameters
    ----------
    trace_posterior,trace_prior : arivz InferenceData
        Raw data from pymc4 sampling

    config : cov19_npis.config.Config

    key : str
        Name of the variable to plot


    Returns
    -------
    array of mpl figures
        one figure for each country

    """

    # Get prior and posterior data for key
    posterior = convert_trace_to_dataframe(trace_posterior, config, key)
    prior = convert_trace_to_dataframe(trace_prior, config, key)

    dist = config.get_distribution_by_name(key)

    # Get ndim of distribution i.e. event stack ndim
    if isinstance(dist["shape"], int):
        ndim = 1
    else:
        ndim = len(dist["shape"])

    def dist_ndim_1():
        # E.g. only age groups or only one value over all age groups
        rows = dist["shape"]
        cols = 1
        label1 = dist["shape_label"]

        fig, ax = plt.subplots(rows, cols, figsize=(4.5 / 3 * cols, rows * 1))
        if rows == 1:
            # Flatten chains and other sampling dimensions of df into one array
            array_posterior = posterior.to_numpy().flatten()
            array_prior = prior.to_numpy().flatten()
            return _distribution(array_posterior, array_prior, dist, ax=ax)
        else:
            for i, value in enumerate(
                posterior.index.get_level_values(label1).unique()
            ):
                # Flatten chains and other sampling dimensions of df into one array
                array_posterior = posterior.xs(value, level=label1).to_numpy().flatten()
                array_prior = prior.xs(value, level=label1).to_numpy().flatten()
                ax[i] = _distribution(array_posterior, array_prior, dist, ax=ax[i])
            return ax

    def dist_ndim_2():

        # In the default case: label1 should be country and label2 should age_group
        label2, label1 = dist["shape_label"]

        # First label is rows
        # Second label is columns
        cols, rows = dist["shape"]

        fig, ax = plt.subplots(
            rows, cols, figsize=(4.5 / 3 * cols, rows * 1), constrained_layout=True,
        )
        for i, value1 in enumerate(posterior.index.get_level_values(label1).unique()):
            # Select first level
            temp_posterior = posterior.xs(value1, level=label1)
            temp_prior = prior.xs(value1, level=label1)
            for j, value2 in enumerate(
                posterior.index.get_level_values(label2).unique()
            ):
                # Select second level and convert to numpy array
                array_posterior = (
                    temp_posterior.xs(value2, level=label2).to_numpy().flatten()
                )
                array_prior = temp_prior.xs(value2, level=label2).to_numpy().flatten()

                ax[i][j] = _distribution(
                    array_posterior, array_prior, dist, ax=ax[i][j],
                )

        # Set labels on y-axis
        for i in range(rows):
            ax[i, 0].set_ylabel(posterior.index.get_level_values(label1).unique()[i])
        # Set labels on x-axis
        for i in range(cols):
            ax[0, i].set_xlabel(posterior.index.get_level_values(label1).unique()[i])

        return ax

    def dist_ndim_3():
        return "TODO"

    # ------------------------------------------------------------------------------ #
    # CASES
    # ------------------------------------------------------------------------------ #
    if ndim == 1:
        return dist_ndim_1()
    elif ndim == 2:
        return dist_ndim_2()
    elif ndim == 3:
        return dist_ndim_3()


def _distribution(array_posterior, array_prior, distribution_dict, suffix="", ax=None):
    """
    Low level function to plots posterior and prior from arrays.

    Parameters
    ----------
    array_posterior,array_prior : array
        Sampling data as array, should be filtered beforehand.

    distribution_dict: dict
        Config.distibution["name"] dictionary, get via config.get_distribution_by_name("name")
    
    suffix: str,optional
        Suffix for the plot title e.g. "age_group_1"
        |default| ""

    ax : mpl axes element, optional
        Plot into an existing axes element
        |default| :code:`None`
    

    Example
    -------
    .. :code: 

        # With given trace for prior and posterior
        df_posterior = convert_trace_to_dataframe(trace_posterior,"R")
        df_prior = convert_trace_to_dataframe(trace_prior,"R")

        _distribution(
            df_posterior,
            df_prior,
            config=config,
            country="my_country_name",
            age_group="my_age_group")

        plt.show()


    """
    dist = distribution_dict

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5 / 3, 1))

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
    text_md = f"${dist['math']}={text_md}$"

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
        log.info(f"Unable to create inset with {dist['name']} value: {e}")

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
