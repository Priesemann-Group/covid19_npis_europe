from scipy import stats
import matplotlib.pyplot as plt
from .rcParams import *
from ..data import convert_trace_to_dataframe
import numpy as np

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
    shape_label = dist["shape_label"]
    additional_dimensions = len(dist["shape_label"]) - 1  # -1 because of country

    figures = []
    for country in posterior.index.get_level_values("country").unique():
        # Case one additional dimension
        if additional_dimensions == 1:
            rows = dist["shape"][1]
            cols = 1
            fig, axes = plt.subplots(
                rows, cols, figsize=(4.5 / 3 * cols, rows), constrained_layout=True
            )
            for i, key_i in enumerate(
                posterior.index.get_level_values(dist["shape_label"][1]).unique()
            ):  # rows
                axes[i] = _distribution(
                    posterior,
                    prior,
                    config,
                    country,
                    ax=axes[i],
                    **{dist["shape_label"][1]: key_i},
                )
        # Case two additional dimensions
        elif additional_dimensions == 2:
            rows = dist["shape"][1]
            cols = dist["shape"][2]
            fig, axes = plt.subplots(rows, cols, figsize=(4.5 / 3 * cols, rows))
            for i, key_i in enumerate(
                posterior.index.get_level_values(dist["shape_label"][1]).unique()
            ):  # y
                for j, key_j in enumerate(
                    posterior.index.get_level_values(dist["shape_label"][2]).unique()
                ):  # x
                    axes[i][j] = _distribution(
                        posterior,
                        prior,
                        config,
                        country,
                        ax=axes[i][j],
                        **{
                            dist["shape_label"][1]: key_i,
                            dist["shape_label"][2]: key_j,
                        },
                    )
        # Append figure to list
        figures.append(fig)
    return figures


def _distribution(df_posterior, df_prior, config, country, ax=None, **dimensions):
    """
    Plots a single distribution from a give country and additional dimensions (e.g. aga_groups),
    if no dimension is given the function uses all available samples in the dataframes
    to create the plot.
    

    Parameters
    ----------
    df_posterior:

    df_prior:

    country: str

    ax : mpl axes element, optional
        Plot into an existing axes element
        |default| :code:`None`

    dimensions: str, optional, kwarg
        Use with care there is no value checks for this,
        e.g. :code:`age_group="a0-10"`. See config shape_label
        for possibilities.

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

    # ------------------------------------------------------------------------------ #
    # Filter data
    # ------------------------------------------------------------------------------ #

    # Get country and other dimensions
    df_posterior = df_posterior.xs(country, level="country")
    df_prior = df_prior.xs(country, level="country")
    suffix = ""
    for level, key in dimensions.items():
        df_posterior = df_posterior.xs(key, level=level)
        df_prior = df_prior.xs(key, level=level)
        suffix += r"$_{" + key + "}$"

    data_posterior = df_posterior.to_numpy().flatten()
    data_prior = df_prior.to_numpy().flatten()

    dist = config.get_distribution_by_name(df_posterior.columns[0])

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5 / 3, 1))

    # ------------------------------------------------------------------------------ #
    # Plot
    # ------------------------------------------------------------------------------ #

    ax = _plot_posterior(x=data_posterior, ax=ax)
    ax = _plot_prior(x=data_prior, ax=ax)

    # add the overlay with median and CI values. these are two strings
    text_md, text_ci = _string_median_CI(data_posterior, prec=2)
    test_md = f"${dist['math']}={text_md}$"

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
        log.debug(f"Unable to create inset with {dist['name']} value: {e}")

    # finalize
    ax.tick_params(labelleft=False)
    ax.set_rasterization_zorder(rcParams.rasterization_zorder)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.set_xlabel(dist["name"] + suffix)
    ax.xaxis.set_label_position("top")
    ax.set_xlim(0)

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
