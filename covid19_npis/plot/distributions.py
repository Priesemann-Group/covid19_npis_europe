import os, sys
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy import stats

from .rcParams import *

from .utils import (
    get_posterior_prior_from_trace,
    get_math_from_name,
    get_dist_by_name_from_sample_state,
)

mpl.rc("figure", max_open_warning=0)
log = logging.getLogger(__name__)


def distribution(
    trace,
    sample_state,
    key,
    dir_save=None,
    plot_age_groups_together=True,
    force_matrix=False,
):
    """
        High level plotting function for distributions,
        plots prior and posterior if they are given in the trace.

        Parameters
        ----------
        trace: arivz.InferenceData
            Raw data from pymc4 sampling, can contain both posterior data
            and prior data. Or only one of both!

        sample_state : pymc4 sample state
            Used mainly for shape labels

        key : str
            Name of the variable to plot

        dir_save: str, optional
            where to save the the figures (expecting a folder). Does not save if None
            |default| None
        force_matrix: bool, optional
            Forces matrix plotting behaviour for last two dimensions
            |default| False

        Returns
        -------
        array of mpl figures
            one figure for each country
    """
    log.debug(f"Creating distibution plot for {key}")

    # Check for special behaviour
    if key in ["C", "C_mean", "Sigma"] or force_matrix:
        return distribution_matrix(trace, sample_state, key, dir_save=dir_save)

    # Convert trace to dataframe with index levels and values changed to
    # values specified in model and modelParams
    # Uses `data.convert_trace_to_dataframe`
    posterior, prior = get_posterior_prior_from_trace(
        trace, sample_state, key, drop_chain_draw=True
    )
    if posterior is not None:
        main = posterior
    elif prior is not None:
        main = prior
    else:
        raise ValueError("Posterior and prior none!!")

    # Get distribution object instance
    dist = get_dist_by_name_from_sample_state(sample_state, key)

    # Check if directory exists
    if dir_save is not None:
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)
        if not os.path.exists(dir_save + f"/{key}"):
            os.makedirs(dir_save + f"/{key}")

    # Progress bar
    pbar = tqdm(
        total=len(main.index.unique()), desc=f"Plotting {key}", position=1, leave=False,
    )

    axes = {}

    def helper_plot(posterior, prior, name_str):
        if posterior is not None:
            df = posterior
        else:
            df = prior

        if plot_age_groups_together and ("age_group" in df.index.names):
            unq_age = df.index.get_level_values("age_group").unique()
            fig, ax = plt.subplots(
                len(unq_age),
                1,
                figsize=(
                    2.2,
                    2.2 * len(unq_age),
                ),
                squeeze=False,
            )
            ax = ax[:, 0]
            for i, ag in enumerate(unq_age):
                # Create pivot table i.e. time on index and draw on columns
                if posterior is not None:
                    posterior_t = posterior.xs(ag).to_numpy().flatten()
                else:
                    posterior_t = None
                if prior is not None:
                    prior_t = prior.xs(ag).to_numpy().flatten()
                else:
                    prior_t = None

                ax_now = ax[i] if len(unq_age)>1 else ax
                # Plot
                _distribution(
                    array_posterior=posterior_t,
                    array_prior=prior_t,
                    dist_name=dist.name,
                    dist_math=get_math_from_name(dist.name),
                    suffix=f"{i}",
                    ax=ax_now,
                )
                # Set title for axis
                ax_now.set_title(ag)
        elif len(df.index.names) == 1:
            fig, ax = plt.subplots(
                len(df.index.get_level_values(df.index.names[0]).unique()),
                1,
                figsize=(
                    2.2,
                    2.2 * len(df.index.get_level_values(df.index.names[0]).unique()),
                ),
                squeeze=False,
            )
            ax = ax[:, 0]
            for i, ag in enumerate(
                df.index.get_level_values(df.index.names[0]).unique()
            ):
                if posterior is not None:
                    posterior_t = posterior.xs(ag).to_numpy().flatten()
                else:
                    posterior_t = None
                if prior is not None:
                    prior_t = prior.xs(ag).to_numpy().flatten()
                else:
                    prior_t = None

                # Plot
                _distribution(
                    array_posterior=posterior_t,
                    array_prior=prior_t,
                    dist_name=dist.name,
                    dist_math=get_math_from_name(dist.name),
                    suffix=f"{i}",
                    ax=ax[i],
                )
                ax[i].set_title(ag)
        else:
            i = 0
            if posterior is not None:
                posterior_t = posterior.to_numpy().flatten()
            else:
                posterior_t = None
            if prior is not None:
                prior_t = prior.to_numpy().flatten()
            else:
                prior_t = None

            fig, ax = plt.subplots(1, 1, figsize=(2.2, 2.2))
            # Plot
            _distribution(
                array_posterior=posterior_t,
                array_prior=prior_t,
                dist_name=dist.name,
                dist_math=get_math_from_name(dist.name),
                suffix=f"{i}",
                ax=ax,
            )

        # Suptitle
        fig.suptitle(
            f"{key}:\n" + name_str.replace("/", "\n"),
            verticalalignment="top",
            ha="left",
            fontweight="bold",
            x=0.01,
            y=0.99,
        )
        fig.tight_layout(h_pad=1.5, w_pad=1.5)
        # Save figure if save_dir is defined
        if dir_save is not None:
            if not os.path.exists(os.path.dirname(f"{dir_save}/{key}/{name_str}.png")):
                os.makedirs(os.path.dirname(f"{dir_save}/{key}/{name_str}.png"))
            # Get path to folder
            if f"{dir_save}/{key}/{name_str}.png".split("/")[-1] == ".png":
                name_str = name_str + "all"
            fig.savefig(
                f"{dir_save}/{key}/{name_str}.png", transparent=True, dpi=300,
            )
        axes[name_str] = ax
        plt.close(fig)
        pbar.update(i + 1)

    # Some kind of recursion to unfold every other dimension
    def recursion(posterior, prior, name_str):
        if posterior is not None:
            index = posterior.index
        else:
            index = prior.index
        if len(index.names) > 1:
            if posterior is not None:
                levels = posterior.index.names
            else:
                levels = prior.index.names
            for lev in levels:
                # Iterate over all level values
                if plot_age_groups_together and lev == "age_group":
                    continue
                for i, value in enumerate(index.get_level_values(lev).unique()):
                    if posterior is not None:
                        posterior_t = posterior.xs(value, level=lev)
                    else:
                        posterior_t = None
                    if prior is not None:
                        prior_t = prior.xs(value, level=lev)
                    else:
                        prior_t = None
                    recursion(
                        posterior_t, prior_t, name_str + "/" + str(value),
                    )
                return
        # else
        name_str = name_str[1:]
        helper_plot(posterior, prior, name_str)

    #
    # START RECURSION
    recursion(posterior, prior, "")
    pbar.close()
    return axes


def distribution_matrix(trace, sample_state, key, dir_save=None):
    """
    High level function to create a distribution plot
    for matrix like variables e.g. 'C'.
    Uses last two dimensions for matrix like plot.

    Parameters
    ----------
        trace: arivz.InferenceData
            Raw data from pymc4 sampling, can contain both posterior data
            and prior data. Or only one of both!

        sample_state : pymc4 sample state
            Used mainly for shape labels

        key : str
            Name of the variable to plot

        dir_save: str, optional
            where to save the the figures (expecting a folder). Does not save if None
            |default| None

    Returns
    -------
    axes
    """
    # Get dataframes
    posterior, prior = get_posterior_prior_from_trace(
        trace, sample_state, key, drop_chain_draw=True
    )

    if posterior is not None:
        main = posterior
    elif prior is not None:
        main = prior
    else:
        raise ValueError("Posterior and prior none!!")

    # Get unique entries for last two dimensions
    rows = main.index.get_level_values(main.index.names[-1]).unique()
    cols = main.index.get_level_values(main.index.names[-2]).unique()

    # Get distribution object instance
    dist = get_dist_by_name_from_sample_state(sample_state, key)

    # Check if directory exists
    if dir_save is not None:
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)
        if not os.path.exists(dir_save + f"/{key}"):
            os.makedirs(dir_save + f"/{key}")

    # Progress bar
    pbar = tqdm(
        total=len(main.index.unique()), desc=f"Plotting {key}", position=1, leave=False,
    )

    axes = {}

    def helper_plot(posterior, prior, name_str):
        """
        Plots matrix from the last two dimensions
        """
        fig, ax = plt.subplots(
            len(rows),
            len(cols),
            figsize=(2.2 * len(cols), 2.2 * len(rows)),
            squeeze=False,
        )
        for x, row in enumerate(rows):
            ax_row = ax[x] if len(rows)>1 else ax
            if posterior is not None:
                posterior_x = posterior.xs(row, level=posterior.index.names[-1])
            if prior is not None:
                prior_x = prior.xs(row, level=prior.index.names[-1])
            for y, col in enumerate(cols):
                ax_col = ax_row[y] if len(cols)>1 else ax_row
                if posterior is not None:
                    posterior_xy = posterior_x.xs(col).to_numpy().flatten()
                else:
                    posterior_xy = None
                if prior is not None:
                    prior_xy = prior_x.xs(col).to_numpy().flatten()
                else:
                    prior_xy = None

                # Plot
                _distribution(
                    array_posterior=posterior_xy,
                    array_prior=prior_xy,
                    dist_name=dist.name,
                    dist_math=get_math_from_name(dist.name),
                    suffix=f"{x},{y}",
                    ax=ax_col,
                )

        # Create titles for the axes
        for x, row in enumerate(rows):
            ax_row = ax[x] if len(rows)>1 else ax
            ax_col = ax_row[0] if len(cols)>1 else ax_row
            ax_col.set_ylabel(row)
        for y, col in enumerate(cols):
            ax_col = ax[x] if len(cols)>1 else ax
            ax_row = ax_col[0] if len(rows)>1 else ax_col
            ax_row.set_title(col)

        # Suptitle
        fig.suptitle(
            f"{key.replace('_', ' ')}:\n{name_str}",
            verticalalignment="top",
            ha="left",
            fontweight="bold",
            x=0.01,
            y=0.99,
        )

        # Save figure if save_dir is defined
        fig.tight_layout(h_pad=1.5, w_pad=1.5)
        if dir_save is not None:
            if not os.path.exists(os.path.dirname(f"{dir_save}/{key}/{name_str}.png")):
                os.makedirs(os.path.dirname(f"{dir_save}/{key}/{name_str}.png"))
            if f"{dir_save}/{key}/{name_str}.png".split("/")[-1] == ".png":
                name_str = name_str + "all"
            fig.savefig(
                f"{dir_save}/{key}/{name_str}", transparent=True, dpi=300,
            )
        axes[name_str] = ax

        plt.close(fig)
        pbar.update(len(rows) * len(cols))

    # Some kind of recursion to unfold every other dimension
    def recursion(posterior, prior, name_str):
        if posterior is not None:
            index = posterior.index
        else:
            index = prior.index
        if len(index.names) > 2:
            if posterior is not None:
                levels = posterior.index.names[0:-2]
            else:
                levels = prior.index.names[0:-2]
            for lev in levels:
                # Iterate over all level values
                for i, value in enumerate(index.get_level_values(lev).unique()):
                    if posterior is not None:
                        posterior_t = posterior.xs(value, level=lev)
                    else:
                        posterior_t = None
                    if prior is not None:
                        prior_t = posterior.xs(value, level=lev)
                    else:
                        prior_t = None
                    recursion(posterior_t, prior_t, name_str + "/" + value)
                return
        # else
        name_str = name_str[1:]
        helper_plot(posterior, prior, name_str)

    # START RECURSION
    recursion(posterior, prior, "")
    pbar.close()
    return axes


# ------------------------------------------------------------------------------ #
# Low level functions
# ------------------------------------------------------------------------------ #


def _distribution(
    array_posterior, array_prior, dist_name, dist_math, suffix="", ax=None
):
    """
    Low level function to plots posterior and prior from arrays.

    Parameters
    ----------
    array_posterior, array_prior : array or None
        Sampling data as array, should be filtered beforehand. If none
        it does not get plotted!

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
        fig, ax = plt.subplots(figsize=(4.5 / 3, 1),)

    # ------------------------------------------------------------------------------ #
    # Plot
    # ------------------------------------------------------------------------------ #
    if array_posterior is not None:
        ax = _plot_posterior(x=array_posterior, ax=ax)
    if array_prior is not None:
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
        log.debug(f"Unable to create inset with {dist_name} value: {e}")

    # ------------------------------------------------------------------------------ #
    # Additional plotting settings
    # ------------------------------------------------------------------------------ #
    ax.xaxis.set_label_position("top")
    # ax.set_xlabel(dist["name"] + suffix)

    ax.tick_params(labelleft=False)
    ax.set_rasterization_zorder(rcParams.rasterization_zorder)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    return ax


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
    try:
        prior = stats.kde.gaussian_kde(x)
    except Exception as e:  # Probably only one value of x
        return ax
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
