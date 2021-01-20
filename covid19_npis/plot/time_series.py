from .rcParams import *
from .. import data
from .utils import (
    get_model_name_from_sample_state,
    get_dist_by_name_from_sample_state,
    check_for_shape_label,
    get_shape_from_dataframe,
    number_formatter,
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
from tqdm.auto import tqdm

log = logging.getLogger(__name__)


def timeseries(
    trace,
    sample_state,
    key,
    sampling_type="posterior",
    plot_chain_separated=False,
    plot_age_groups_together=True,
    dir_save=None,
    observed=None,
):
    """
    High level plotting fucntion to create time series for a a give variable,
    i.e. plot for every additional dimension.
    Can only be done for variables with a time or date in shape_labels!

    Does NOT plot observed cases, these have to be added manually for now.

    Parameters
    ----------
    trace_posterior, trace_prior : arivz InferenceData
        Raw data from pymc4 sampling

    sample_state : pymc4 sample stae

    key : str
        Name of the timeseries variable to plot. Same name as in the model definitions.

    sampling_type: str, optional
        Name of the type (group) in the arviz inference data. |default| posterior

    dir_save: str, optional
        where to save the the figures (expecting a folder). Does not save if None
        |default| None

    observed: pd.DataFrame, optional
        modelParams dataframe for the corresponding observed values for the
        variable e.g. modelParams.pos_tests_dataframe
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

    # Drop chains index if seperated!
    if not plot_chain_separated:
        df.index = df.index.droplevel("chain")

    # Drop the number of draws
    # df.index = df.index.droplevel("draw")

    # Define recursive plotting fuction
    axes = {}

    def recursive_plot(df, name_str, observed=None):
        """
        Every call of this function reduces dimensions by one if
        plot age_groups_together is defined it skips agegroup
        dimension and creates subplots for each age group.
        """

        # We start x function calls depending on the number of dimensions, going from
        # left to right i.e. country than agegroup
        if len(df.index.names) > 1:

            # Iterate over all levels expect time,draw,age_group
            levels = df.index.names
            for lev in levels:
                if lev == "time":
                    continue
                if lev == "draw":
                    continue
                if plot_age_groups_together and lev == "age_group":
                    continue

                # Iterate over all level values
                for i, value in enumerate(df.index.get_level_values(lev).unique()):
                    # create new dataframe for next recursion
                    df_t = df.xs(value, level=lev)
                    if observed is not None:
                        # I hope the dataframes have the same format
                        _observed = observed.xs(value, level=lev, axis=1)
                    else:
                        _observed = None
                    recursive_plot(df_t, name_str + "_" + value, _observed)

                return  # Stop theses recursions

        # Remove "_" from name
        name_str = name_str[1:]

        if plot_age_groups_together and "age_group" in df.index.names:
            fig, a_axes = plt.subplots(
                len(df.index.get_level_values("age_group").unique()),
                1,
                figsize=(4, 1.5 * len(df.index.get_level_values("age_group").unique())),
                constrained_layout=True,
            )
            for i, ag in enumerate(df.index.get_level_values("age_group").unique()):
                temp = df.xs(ag, level="age_group")

                # Create pivot table i.e. time on index and draw on columns
                temp = temp.reset_index().pivot_table(index="time", columns="draw")

                # Plot data
                _timeseries(temp.index, temp.to_numpy(), what="model", ax=a_axes[i])

                # Plot observed
                if observed is not None:
                    _timeseries(
                        observed.index, observed.to_numpy(), what="data", ax=a_axes[i]
                    )
                # Set title for axis
                a_axes[i].set_title(ag)

            axes[name_str] = a_axes
        else:

            # Create pivot table i.e. time on index and draw on columns
            df = df.reset_index().pivot_table(index="time", columns="draw")

            # Plot this dimension!
            axes[name_str] = _timeseries(df.index, df.to_numpy(), what="model",)
            # Plot observed
            if observed is not None:
                _timeseries(
                    observed.index, observed.to_numpy(), what="data", ax=axes[name_str]
                )
            else:
                _observed = None
            axes[name_str].set_title(name_str.replace("_", " "))

    recursive_plot(df, "", observed)

    # Create figure supertitle with key and dimensions
    for name, ax in axes.items():
        if type(ax) == np.ndarray:
            fig = ax[0].get_figure()
        else:
            fig = ax.get_figure()
        fig.suptitle(
            f"{key.replace('_', ' ')}:\n{name}",
            verticalalignment="top",
            ha="left",
            fontweight="bold",
            x=0.1,
        )

    if dir_save is not None:
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)
        if not os.path.exists(dir_save + f"/{key}"):
            os.makedirs(dir_save + f"/{key}")
        for name, ax in tqdm(
            axes.items(),
            desc=f"Saving figures to '{dir_save}/{key}'",
            position=1,
            leave=False,
        ):
            if type(ax) == np.ndarray:
                fig = ax[0].get_figure()
            else:
                fig = ax.get_figure()
            fig.savefig(
                f"{dir_save}/{key}/{name}", transparent=True, dpi=300,
            )
    plt.close(fig)
    return axes


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

    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(number_formatter))

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
        # OLD: ax.xaxis.set_minor_locator(mpl.dates.DayLocator())
        ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=16))
        ax.xaxis.set_major_formatter(mpl.ticker.NullFormatter())
    # OLD: ax.xaxis.set_major_formatter(mpl.dates.DateFormatter(rcParams["date_format"]))
    ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
    ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter("%b"))

    for label in ax.get_xticklabels():
        label.set_rotation(rcParams["timeseries_xticklabel_rotation"])
        label.set_ha("center")
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=, ha='right')
