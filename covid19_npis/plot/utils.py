# ------------------------------------------------------------------------------ #
# @Author:        Sebastian B. Mohr
# @Email:
# @Created:       2020-08-17 10:35:59
# @Last Modified: 2021-01-27 13:20:26
# ------------------------------------------------------------------------------ #

import logging
import json

from .. import data

log = logging.getLogger(__name__)


def get_model_name_from_sample_state(sample_state):
    dists = list(sample_state.continuous_distributions.items())
    model_name, dist_name = dists[0][0].split("|")
    return model_name


def get_dist_by_name_from_sample_state(sample_state, name):
    model_name = get_model_name_from_sample_state(sample_state)
    try:
        dist = sample_state.continuous_distributions[model_name + "|" + name]
    except Exception as e:
        dist = sample_state.deterministics[model_name + "|" + name]
    return dist


def get_shape_from_dataframe(df):
    """
    Returns shape tuple from dataframe.

    Return
    ------
    tuple: int

    """
    # Ndim of the dataframe minus chain and samples
    ndim = len(df.index.levels) - 2

    shape = ()
    if ndim == 0:
        shape = (1,)
    else:
        for i in reversed(range(0, ndim)):
            shape = shape + (len(df.index.levels[-1 - i]),)
    log.debug(f"Found shape '{shape}' for dataframe '{df.columns[0]}'.")
    return shape


def check_for_shape_label(dist):
    try:
        shape_label = dist.shape_label
    except AttributeError:
        log.warning(
            f"'shape_label' not found in distribution {dist.name}, could yield to strange behaviour in plotting!"
        )


def get_math_from_name(name):
    """
    Gets math string from distribution name.
    Returns "name" if no mathkey is found!

    Parameters
    ----------
    name :
        Name of the distribution/timeseries.

    Returns
    -------
    : str
        Mathstring for plotting
    """

    # Latex $..$ get casted before plotting
    math_keys = {
        "g_mu": r"g_{\mu}",
        "g_theta": r"g_{\theta}",
        "E_0_diff_base": r"E_{0,diff base}",
        "R_0": r"R_{0}",
        "R_0_c": r"R_{0, c}",
        "sigma": r"\sigma",
        "new_cases": r"N",
        "R_t": r"R_{t}",
        "alpha_i_c_a": r"\alpha_{i,c,a}",
        "d_i_c_p": r"d_{i,c,p}",
        "l_i_sign": r"l_{i,sign(\Delta\gamma)}",
        "C": r"C_c",
        "C_mean": "C",
        "Phi_IFR": r"\Phi_{IFR}",
        "phi_age": r"\phi_{age}",
        "phi_tests_reported": r"\phi_{tests reported}",
        "mu_testing_state": r"\mu_{testing state}",
        "delay_deaths_m": r"d_{deaths, m}",
        "delay_deaths_theta": r"d_{deaths, \theta}",
    }

    if name not in math_keys:
        log.debug(
            f"Math key for distribution with name '{name}' not found! Expect strange behaviour."
        )
        return name
    else:
        return math_keys[name]


def number_formatter(number, pos=None):
    """
    Converts number to magnitude format
    Taken from https://flynn.gg/blog/better-matplotlib-charts/
    """
    magnitude = 0
    while abs(number) >= 1000:
        magnitude += 1
        number /= 1000.0
    return "%.1f%s" % (number, ["", "K", "M", "B", "T", "Q"][magnitude])


def get_posterior_prior_from_trace(trace, sample_state, key, drop_chain_draw=False):
    """ Returns prior posterior tuple is None if not found in trace
    """

    # Detect datatype
    if "posterior" in trace.groups():
        posterior = data.convert_trace_to_dataframe(
            trace, sample_state, key, data_type="posterior"
        )
        if drop_chain_draw and len(posterior.index.names) > 2:
            posterior.index = posterior.index.droplevel("chain")
            posterior.index = posterior.index.droplevel("draw")
        elif drop_chain_draw:
            print("Hi")
            posterior = posterior.reset_index()
            posterior = posterior[key]
    else:
        posterior = None

    if "prior_predictive" in trace.groups():
        prior = data.convert_trace_to_dataframe(
            trace, sample_state, key, data_type="prior_predictive"
        )
        if drop_chain_draw and len(prior.index.names) > 2:
            prior.index = prior.index.droplevel("chain")
            prior.index = prior.index.droplevel("draw")
        elif drop_chain_draw:
            prior = prior[key]
    else:
        prior = None

    return (posterior, prior)
