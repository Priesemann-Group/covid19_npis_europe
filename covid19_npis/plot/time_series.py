from .rcParams import *
from ..data import convert_trace_to_dataframe, select_from_dataframe

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats
import locale

import logging

log = logging.getLogger(__name__)


def timeseries(trace, modelParams, key, plot_observed=False):
    """
    Create time series overview for a a give variable, i.e. plot for every additional dimension.
    Can only done to variables with a time shape label. See modelParams.py

    Parameters
    ----------
    trace_posterior,trace_prior : arivz InferenceData
        Raw data from pymc4 sampling

    modelParams : cov19_npis.modelParams.modelParams

    key : str
        Name of the variable to plot

    plot_observed: bool, optional
        Do you want to plot the new cases? May not work for 1 and 2 dim case.
    """

    df = convert_trace_to_dataframe(trace, modelParams, key)
    dist = modelParams.get_distribution_by_name(key)

    # Determine ndim:
    if isinstance(dist["shape"], int):
        ndim = 1
    else:
        ndim = len(dist["shape"])

    def timeseries_ndim_1():
        """
        Only a time dimension nothing else
        """
        # Since we work with a multiindex pandas dataframe we need to unstack
        # our time values and transpose

        df = df.unstack(level="time").T
        df.index = df.index.droplevel(0)
        # Plot model
        return _timeseries(df.index, df.to_numpy(), what="model")

    def timeseries_ndim_2():
        """
        Time and an additional dimension
        """
        # In the default case: label1 should be time and label2 should age_group
        time, label1 = dist["shape_label"]
        cols = 1
        rows = dist["shape"][1]  # not time

        fig, axes = plt.subplots(rows, cols, figsize=(6, 3 * rows))
        for i, value in enumerate(df.index.get_level_values(label1).unique()):
            df_t = df.xs(value, level=label1)
            df_t = df_t.unstack(level="time").T
            df_t.index = df_t.index.droplevel(0)
            # Plot model
            axes[i] = _timeseries(df_t.index, df_t.to_numpy(), ax=axes[i], what="model")
            # Plot data
        return axes

    def timeseries_ndim_3():
        time, label1, label2 = dist["shape_label"]
        cols = dist["shape"][1]
        rows = dist["shape"][2]

        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows))

        # Loop threw all dimensions of variable
        for i, value1 in enumerate(df.index.get_level_values(label1).unique()):
            for j, value2 in enumerate(df.index.get_level_values(label2).unique()):

                # Plot model
                model = select_from_dataframe(df, **{label1: value1, label2: value2})
                model = model.unstack(level="time").T
                model.index = model.index.droplevel(0)
                axes[j][i] = _timeseries(
                    model.index, model.to_numpy(), ax=axes[j][i], what="model"
                )

                if plot_observed:
                    axes[j][i] = _timeseries(
                        modelParams.dataframe.index,
                        modelParams.dataframe[(value1, value2)],
                        ax=axes[j][i],
                        what="data",
                    )

                # Set labels on y-axis
        for i in range(rows):
            axes[i][0].set_ylabel(df.index.get_level_values(label2).unique()[i])
        # Set labels on x-axis
        for i in range(cols):
            axes[0][i].set_title(df.index.get_level_values(label1).unique()[i])

        return axes

    # ------------------------------------------------------------------------------ #
    # CASES
    # ------------------------------------------------------------------------------ #
    if ndim == 1:
        return timeseries_ndim_1()
    elif ndim == 2:
        return timeseries_ndim_2()
    elif ndim == 3:
        return timeseries_ndim_3()


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
        figure, ax = plt.subplots(figsize=(6, 3))

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

    if what is "data":
        if "color" not in kwargs:
            kwargs = dict(kwargs, color=rcParams["color_data"])
        if "marker" not in kwargs:
            kwargs = dict(kwargs, marker="d")
        if "ls" not in kwargs and "linestyle" not in kwargs:
            kwargs = dict(kwargs, ls="None")
    elif what is "fcast":
        if "color" not in kwargs:
            kwargs = dict(kwargs, color=rcParams["color_model"])
        if "ls" not in kwargs and "linestyle" not in kwargs:
            kwargs = dict(kwargs, ls="--")
    elif what is "model":
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
    _format_date_xticks(ax)

    return ax


# ------------------------------------------------------------------------------ #
# Formating and util
# ------------------------------------------------------------------------------ #
def _format_date_xticks(ax, minor=None):
    # ensuring utf-8 helps on some setups
    locale.setlocale(locale.LC_ALL, rcParams.locale + ".UTF-8")
    ax.xaxis.set_major_locator(
        mpl.dates.WeekdayLocator(interval=1, byweekday=mpl.dates.SU)
    )
    if minor is None:
        # overwrite local argument with rc params only if default.
        minor = rcParams["date_show_minor_ticks"]
    if minor is True:
        ax.xaxis.set_minor_locator(mpl.dates.DayLocator())
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter(rcParams["date_format"]))
