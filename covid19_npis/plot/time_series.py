from .rcParams import *
from .. import data
from .utils import (
    get_model_name_from_sample_state,
    get_dist_by_name_from_sample_state,
    check_for_shape_label,
    get_shape_from_dataframe,
)
from .. import modelParams

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats
import locale

import logging
import os

log = logging.getLogger(__name__)


def timeseries_new(
    trace,
    sample_state,
    key,
    sampling_type="posterior",
    plot_observed=False,
    plot_chain_separated=False,
    dir_save=None,
):
    """
    High level plotting fucntion to create time series for a a give variable,
    i.e. plot for every additional dimension.
    Can only be done for variables with a time or date in shape_labels!

    Parameters
    ----------
    trace_posterior, trace_prior : arivz InferenceData
        Raw data from pymc4 sampling

    sample_state : pymc4 sample stae

    key : str
        Name of the timeseries variable to plot. Same name as in the model definitions.

    sampling_type: str, optional
        Name of the type (group) in the arviz inference data. |default| posterior

    plot_observed: bool, optional
        Do you want to plot the new cases? May not work for 1 and 2 dim case.

    dir_save: str, optional
        where to save the the figures (expecting a folder). Does not save if None
        |default| None 

    """
    log.debug(f"Creating timeseries plot for {key}")

    # Check type of arviz trace
    types = trace.groups()
    if sampling_type not in types:
        raise KeyError("sampling_type '{sampling_type}' not found in trace!")

    # Convert trace to dataframe with index levels and values changed to
    # values specified in model and modelParams
    df = data.convert_trace_to_dataframe(trace, sample_state, key)

    # Sanity check for "time" or "date" in index
    if ("time" not in df.index.names) and ("date" not in df.index.names):
        raise ValueError(
            "No time or date found in variable dimensions!\n (Is the distribution shape_label set?!)"
        )

    # Drop chains dimension if seperated!
    if not plot_chain_separated:
        df.index = df.index.droplevel("chain")

    # Drop the number of draws
    # df.index = df.index.droplevel("draw")

    # Define recursive plotting fuction
    axes = {}

    def recursive_plot(df, name_str):
        """
        Every call of this function reduces dimensions by one
        """

        # We start x function calls depending on the number of dimensions, going from
        # left to right i.e. country than agegroup
        if len(df.index.names) > 1:

            # Iterate over all levels expect time
            levels = df.index.names
            for lev in levels:
                if lev == "time":
                    continue
                if lev == "draw":
                    continue

                # Iterate over all level values
                for i, value in enumerate(df.index.get_level_values(lev).unique()):
                    # create new dataframe for next recursion
                    df_t = df.xs(value, level=lev)
                    recursive_plot(df_t, name_str + "_" + value)

                return  # Stop theses recursions

        # Create pivot table i.e. time on index and draw on columns
        df = df.reset_index().pivot_table(index="time", columns="draw")

        # Remove "_" from name
        name_str = name_str[1:]

        # Plot this dimension!
        axes[name_str] = _timeseries(df.index, df.to_numpy(), what="model",)

    recursive_plot(df, "")

    if dir_save is not None:
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)
        if not os.path.exists(dir_save + f"/{key}"):
            os.makedirs(dir_save + f"/{key}")
        for name, ax in axes.items():
            fig = ax.get_figure()
            plt.tight_layout()
            fig.savefig(
                f"{dir_save}/{key}/{name}", transparent=True, dpi=300,
            )

    return axes


def timeseries(
    trace, sample_state, key, plot_observed=False, plot_chain_separated=False
):
    """
    Create time series overview for a a give variable, i.e. plot for every additional dimension.
    Should only done to variables with a time shape label at position 0!

    Parameters
    ----------
    trace_posterior, trace_prior : arivz InferenceData
        Raw data from pymc4 sampling

    sample_state : pymc4 sample stae

    key : str
        Name of the timeseries variable to plot. Same name as in the model definitions.

    plot_observed: bool, optional
        Do you want to plot the new cases? May not work for 1 and 2 dim case.
    """

    log.debug(f"Creating timeseries plot for {key}")

    # Convert trace to dataframe
    df = data.convert_trace_to_dataframe(trace, sample_state, key)
    # log.info(df)
    # Get other important properties
    model_name = get_model_name_from_sample_state(sample_state)
    dist = get_dist_by_name_from_sample_state(sample_state, key)
    check_for_shape_label(dist)

    shape = get_shape_from_dataframe(df)
    # Determine ndim:

    def timeseries_ndim_1():
        """
        Only a time dimension nothing else
        """
        # Since we work with a multiindex pandas dataframe we need to unstack
        # our time values and transpose

        df = df.unstack(level="time").T
        df.index = df.index.droplevel(0)
        # Plot model once for each chain
        fig, axes = plt.subplots(1, 1, figsize=(3, 1.5))
        # Plot each chain
        axes = _timeseries(df.index, df.to_numpy(), ax=axes, what="model")

        if plot_chain_separated:
            for c in df.column.get_level_values("chain").unique():
                axes = _timeseries(
                    df.index,
                    df.xs(c, level="chain", axis=1).to_numpy(),
                    ax=axes,
                    what="model",
                    color=None,
                    label=f"Chain {c}",
                )
                axes.legend()
        return fig, axes

    def timeseries_ndim_2():
        """
        Time and an additional dimension
        """
        # In the default case: label1 should be time and label2 should age_group
        if hasattr(dist, "shape_label"):
            time, label1 = dist.shape_label
        else:
            time = model_name + "/" + dist.name + "_dim_0"
            label1 = model_name + "/" + dist.name + "_dim_1"
        cols = shape[1]
        rows = 1  # not time

        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 1.5 * rows),)
        for i, value in enumerate(df.index.get_level_values(label1).unique()):
            df_t = df.xs(value, level=label1)
            df_t = df_t.unstack(level="time").T
            df_t.index = df_t.index.droplevel(0)
            # Plot model
            axes[i] = _timeseries(df_t.index, df_t.to_numpy(), ax=axes[i], what="model")

            if plot_chain_separated:
                for c in df.index.get_level_values("chain").unique():
                    axes[i] = _timeseries(
                        df_t.index,
                        df_t.xs(c, level="chain", axis=1).to_numpy(),
                        ax=axes[i],
                        what="model",
                        color=None,
                        label=f"Chain {c}",
                    )
                    axes[i].legend()

        # Set labels on y-axis
        for i in range(cols):
            axes[i].set_title(df.index.get_level_values(label1).unique()[i])

        return fig, axes

    def timeseries_ndim_3():
        if hasattr(dist, "shape_label"):
            time, label1, label2 = dist.shape_label
        else:
            time = model_name + "/" + dist.name + "_dim_0"
            label1 = model_name + "/" + dist.name + "_dim_1"
            label2 = model_name + "/" + dist.name + "_dim_2"

        # shape[0] == time
        cols = shape[1]
        rows = shape[2]

        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 1.5 * rows),)

        # Loop threw all dimensions of variable
        for i, value1 in enumerate(df.index.get_level_values(label1).unique()):
            for j, value2 in enumerate(df.index.get_level_values(label2).unique()):

                # Plot model
                model = data.select_from_dataframe(
                    df, **{label1: value1, label2: value2}
                )
                model = model.unstack(level="time").T
                model.index = model.index.droplevel(0)
                axes[j][i] = _timeseries(
                    model.index, model.to_numpy(), ax=axes[j][i], what="model"
                )

                if plot_observed:
                    axes[j][i] = _timeseries(
                        modelParams.modelParams.dataframe.index,
                        modelParams.modelParams.dataframe[(value1, value2)],
                        ax=axes[j][i],
                        what="data",
                    )
                if plot_chain_separated:
                    for c in df.index.get_level_values("chain").unique():
                        axes[j][i] = _timeseries(
                            model.index,
                            model.xs(c, level="chain", axis=1).to_numpy(),
                            ax=axes[j][i],
                            what="model",
                            color=None,
                            label=f"Chain {c}",
                        )
                        axes[j][i].legend()

        # Set labels on y-axis
        for i in range(rows):
            axes[i][0].set_ylabel(df.index.get_level_values(label2).unique()[i])
        # Set labels on x-axis
        for i in range(cols):
            axes[0][i].set_title(df.index.get_level_values(label1).unique()[i])

        return fig, axes

    # ------------------------------------------------------------------------------ #
    # CASES
    # ------------------------------------------------------------------------------ #
    if len(shape) == 1:
        fig, axes = timeseries_ndim_1()
    elif len(shape) == 2:
        fig, axes = timeseries_ndim_2()
    elif len(shape) == 3:
        fig, axes = timeseries_ndim_3()

    # ------------------------------------------------------------------------------ #
    # Title and other
    # ------------------------------------------------------------------------------ #
    fig.suptitle(
        key, verticalalignment="top", fontweight="bold",
    )

    return [fig], axes


def _timeseries(
    x,
    y,
    ax=None,
    what="data",
    draw_ci_95=None,
    draw_ci_75=None,
    draw_ci_50=None,
    **kwargs,
):
    """
    low-level function to plot anything that has a date on the x-axis.

    Parameters
    ----------
    x : array of datetime.datetime
        times for the x axis

    y : array, 1d or 2d
        data to plot. if 2d, we plot the CI as fill_between (if CI enabled in rc
        params)
        if 2d, then first dim is realization and second dim is time matching `x`
        if 1d then first tim is time matching `x`

    ax : mpl axes element, optional
        plot into an existing axes element. default: None

    what : str, optional
        what type of data is provided in x. sets the style used for plotting:
        * `data` for data points
        * `fcast` for model forecast (prediction)
        * `model` for model reproduction of data (past)

    kwargs : dict, optional
        directly passed to plotting mpl.

    Returns
    -------
        ax
    """

    # ------------------------------------------------------------------------------ #
    # Default parameter
    # ------------------------------------------------------------------------------ #

    if draw_ci_95 is None:
        draw_ci_95 = rcParams["draw_ci_95"]

    if draw_ci_75 is None:
        draw_ci_75 = rcParams["draw_ci_75"]

    if draw_ci_50 is None:
        draw_ci_50 = rcParams["draw_ci_50"]

    if ax is None:
        figure, ax = plt.subplots(figsize=(6, 4))

    # still need to fix the last dimension being one
    # if x.shape[0] != y.shape[-1]:
    #     log.exception(f"X rows and y rows do not match: {x.shape[0]} vs {y.shape[0]}")
    #     raise KeyError("Shape mismatch")

    if y.ndim == 2:
        data = np.median(y, axis=1)
    elif y.ndim == 1:
        data = y
    else:
        log.exception(f"y needs to be 1 or 2 dimensional, but has shape {y.shape}")
        raise KeyError("Shape mismatch")

    # ------------------------------------------------------------------------------ #
    # kwargs
    # ------------------------------------------------------------------------------ #

    if what == "data":
        if "color" not in kwargs:
            kwargs = dict(kwargs, color=rcParams["color_data"])
        if "marker" not in kwargs:
            kwargs = dict(kwargs, marker="d")
        if "ls" not in kwargs and "linestyle" not in kwargs:
            kwargs = dict(kwargs, ls="None")
    elif what == "fcast":
        if "color" not in kwargs:
            kwargs = dict(kwargs, color=rcParams["color_model"])
        if "ls" not in kwargs and "linestyle" not in kwargs:
            kwargs = dict(kwargs, ls="--")
    elif what == "model":
        if "color" not in kwargs:
            kwargs = dict(kwargs, color=rcParams["color_model"])
        if "ls" not in kwargs and "linestyle" not in kwargs:
            kwargs = dict(kwargs, ls="-")

    # ------------------------------------------------------------------------------ #
    # plot
    # ------------------------------------------------------------------------------ #
    ax.plot(x, data, **kwargs)

    # overwrite some styles that do not play well with fill_between
    if "linewidth" in kwargs:
        del kwargs["linewidth"]
    if "marker" in kwargs:
        del kwargs["marker"]
    if "alpha" in kwargs:
        del kwargs["alpha"]
    if "label" in kwargs:
        del kwargs["label"]
    kwargs["lw"] = 0
    kwargs["alpha"] = 0.1

    if draw_ci_95 and y.ndim == 2:
        ax.fill_between(
            x,
            np.percentile(y, q=2.5, axis=1),
            np.percentile(y, q=97.5, axis=1),
            **kwargs,
        )

    if draw_ci_75 and y.ndim == 2:
        ax.fill_between(
            x,
            np.percentile(y, q=12.5, axis=1),
            np.percentile(y, q=87.5, axis=1),
            **kwargs,
        )

    del kwargs["alpha"]
    kwargs["alpha"] = 0.2

    if draw_ci_50 and y.ndim == 2:
        ax.fill_between(
            x,
            np.percentile(y, q=25.0, axis=1),
            np.percentile(y, q=75.0, axis=1),
            **kwargs,
        )

    # ------------------------------------------------------------------------------ #
    # formatting
    # ------------------------------------------------------------------------------ #
    _format_date_xticks(ax, interval=2)

    return ax


# ------------------------------------------------------------------------------ #
# Formating and util
# ------------------------------------------------------------------------------ #
def _format_date_xticks(ax, minor=None, interval=1):
    # ensuring utf-8 helps on some setups
    locale.setlocale(locale.LC_ALL, rcParams.locale + ".UTF-8")
    ax.xaxis.set_major_locator(
        mpl.dates.WeekdayLocator(interval=interval, byweekday=mpl.dates.SU)
    )
    if minor is None:
        # overwrite local argument with rc params only if default.
        minor = rcParams["date_show_minor_ticks"]
    if minor is True:
        ax.xaxis.set_minor_locator(mpl.dates.DayLocator())
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter(rcParams["date_format"]))

    for label in ax.get_xticklabels():
        label.set_rotation(rcParams["timeseries_xticklabel_rotation"])
        label.set_ha("center")
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=, ha='right')
