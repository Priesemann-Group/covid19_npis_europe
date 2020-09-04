# ------------------------------------------------------------------------------ #
# @Author:        Sebastian B. Mohr
# @Email:
# @Created:       2020-08-17 10:35:59
# @Last Modified: 2020-08-17 12:41:54
# ------------------------------------------------------------------------------ #

import logging

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
