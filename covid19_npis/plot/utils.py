# ------------------------------------------------------------------------------ #
# @Author:        Sebastian B. Mohr
# @Email:
# @Created:       2020-08-17 10:35:59
# @Last Modified: 2020-09-09 13:32:53
# ------------------------------------------------------------------------------ #

import logging
import json

log = logging.getLogger(__name__)


def get_model_name_from_sample_state(sample_state):
    dists = list(sample_state.continuous_distributions.items())
    model_name, dist_name = dists[0][0].split("/")
    return model_name


def get_dist_by_name_from_sample_state(sample_state, name):
    model_name = get_model_name_from_sample_state(sample_state)
    try:
        dist = sample_state.continuous_distributions[model_name + "/" + name]
    except Exception as e:
        dist = sample_state.deterministics[model_name + "/" + name]
    return dist


def check_for_shape_and_shape_label(dist):
    try:
        shape_label = dist.shape_label
    except AttributeError:
        log.warning(
            f"'shape_label' not found in distribution {dist.name}, could yield to strange behaviour in plotting!"
        )
    try:
        shape = dist.shape
    except Exception as e:
        log.warning(
            f"'shape' not found in distribution {dist.name}, could yield to strange behaviour in plotting!"
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
        "I_0_diff_base": r"I_{0,diff base}",
        "R_0": r"R_{0}",
        "sigma": r"\sigma",
        "new_cases": r"N",
        "R_t": r"R_{t}",
        "alpha_i_c_a": r"\alpha_{i,c,a}",
    }

    if name not in math_keys:
        log.warning(
            f"Math key for distribution with name '{name}' not found! Expect strange behaviour."
        )
        return name
    else:
        return math_keys[name]
