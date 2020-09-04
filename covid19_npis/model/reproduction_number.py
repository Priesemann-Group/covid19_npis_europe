import tensorflow as tf
import logging
import pymc4 as pm
from .utils import gamma
import tensorflow_probability as tfp
import numpy as np

log = logging.getLogger(__name__)

from covid19_npis import transformations
from covid19_npis.model.distributions import (
    Normal,
    LogNormal,
    Deterministic,
    HalfNormal,
)
from .. import modelParams


def _fsigmoid(t, l, d):
    r"""
        Calculates and returns

        .. math::

            \frac{1}{1+e^{-4/l*(t-d)}}

        Parameters
        ----------
        t:
            Time, "variable"
        l:
            Length of the change point, determines scale
        d:
            Date of the change point, determines location
    """
    # Prep dimensions
    d = tf.expand_dims(d, axis=-1)
    # Factors of the exponent
    log.debug(f"d in _fsigmoid\n{d.shape}")
    log.debug(f"t in _fsigmoid\n{t.shape}")
    inside_exp_1 = -4.0 / l
    inside_exp_2 = t - d
    log.debug(f"t-d\n{inside_exp_2.shape}")

    return 1.0 / (1.0 + tf.exp(inside_exp_1 * inside_exp_2))


class Change_point(object):
    """
        Change point class. Should contain date d, gamma_max and function to return gamma at point t.

        Parameters
        ----------
        name: str
            Name of the change point, get passed to the pymc4 distribution for the date.

        date_loc : number
            Prior for the location of the date of the change point.

        date_scale : number
            Prior for the scale of the date of the change point.

        gamma_max : number
            Maximum gamma value for the change point, i.e the value the logistic function gamma_t converges to. [-1,1]
    """

    def __init__(self, name, date_loc, date_scale, gamma_max):
        self.name = name
        self.prior_date_loc = date_loc
        self.prior_date_scale = date_scale
        self.gamma_max = gamma_max

        self._d = Normal(
            "date_" + self.name,
            self.prior_date_loc,
            self.prior_date_scale,
            event_stack=modelParams.modelParams.num_age_groups,
            shape_label=("age_group"),
            conditionally_independent=True,
        )

        log.debug(f"Created changepoint at prior_d={self.prior_date_loc}")

    @property
    def date(self):
        r"""
        Returns pymc4 generator for the date :math:`d`, i.e. a normal distribution. The priors
        are set at init of the object.
        """

        return (yield self._d)

    def gamma_t(self, t, l):
        """
        Returns gamma value at t with given length :math:`l`. The length :math:`l` should be
        passed from the intervention class.
        """
        log.debug("gamma_t_change_point")
        d = yield self.date
        sigmoid = _fsigmoid(t, l, d)
        return sigmoid * self.gamma_max


class Intervention(object):
    """
        Intervention class, contains every variable that is only intervention dependent
        and the change point for the intervention, i.e. the hyperprior distributions.

        Parameters
        ----------
        name: str
            Name of the intervention, get passed to the pymc4 functions with suffix.

        alpha_loc_loc:
            Location of hyperprior location for the effectivity of the intervention.

        alpha_loc_scale:
            Scale of hyperprior location for the effectivity of the intervention.

        alpha_scale_loc:
            Location of hyperprior sale for the effectivity of the intervention.

        alpha_scale_scale:
             Scale of hyperprior sale for the effectivity of the intervention.
    """

    def __init__(
        self, name,
    ):
        self.name = name

        # Distribution for alpha^cross
        self._alpha_cross = Normal(
            "_alpha_cross_" + self.name,
            loc=-1.0,  # See publication for reasoning behind -1 and 2
            scale=2.0,
            conditionally_independent=True,
        )

        # Distribution for delta_l_cross sigma_l gets multiplied at a later point for better sampling.
        self._delta_l_cross = Normal(
            "_delta_l_cross_" + self.name,
            loc=0.0,
            scale=1.0,
            conditionally_independent=True,
        )

        self._delta_d = Normal(
            "_delta_d_" + self.name, loc=0.0, scale=1.0, conditionally_independent=True,
        )

        log.debug(f"Created intervention with name: {name}")

    @property
    def alpha_cross(self):
        r"""
        Returns pymc4 generator for the effetivity :math:`\alpha_i`, i.e. a normal distribution. The priors
        are set to loc=-1.0 and scale=2.0.
        """

        return (yield self._alpha_cross)

    @property
    def delta_l_cross(self):
        r"""
        Returns pymc4 generator for the effetivity :math:`\Delta l^\cross_i`, i.e. a normal distribution. The priors
        are set to loc=0.0, scale=1.0. Should be multiplied with hyperdist.
        """

        return (yield self._delta_l_cross)

    @property
    def delta_d(self):
        r"""
        Returns pymc4 generator for the effetivity :math:`\Delta d_i`, i.e. a normal distribution. The priors
        are set to loc=0.0, scale=1.0. Should be multiplied with hyperdist.
        """

        return (yield self._delta_d)


def _create_distributions(modelParams):
    """
    Returns a dict of distributions and interventions for further processing/sampling.

    Parameter
    ---------
    modelParams

    Return
    ------
    :
        interventions, distributions
    """
    log.debug("_create_distributions")

    """
    Get all interventions from the countries data objects
    """
    interventions_data = modelParams.countries[0].interventions

    """
    Create distributions for each interventions that beeing 
    alpha_i and Δl_i^cross. Happens in the intervention class.
    """
    interventions = {}
    for i in interventions_data:
        interventions[i.name] = Intervention(name=i.name)

    """
        Δ Alpha cross for each country and age group with hyperdistributions
    """
    sigma_a_c = HalfNormal("sigma_alpha_country", scale=0.1,)
    sigma_a_a = HalfNormal("sigma_alpha_age_group", scale=0.1,)
    # We need to multiply sigma_a_c and sigma_a_a later. (See construct R_t)
    delta_alpha_cross_c = Normal(
        "delta_alpha_cross_c",
        loc=0.0,
        scale=1.0,
        event_stack=(modelParams.num_countries),
        shape_label=("country"),
    )
    delta_alpha_cross_a = Normal(
        "delta_alpha_cross_a",
        loc=0.0,
        scale=1.0,
        event_stack=(modelParams.num_age_groups),
        shape_label=("age_group"),
    )

    """
        l distributions
    """
    sigma_l_interv = HalfNormal("sigma_l_interv", scale=1.0)
    # Δl_i^cross was created in intervention class see above
    l_positive_cross = Normal("l_positive_cross", loc=3.0, scale=1.0,)
    l_negative_cross = Normal("l_negative_cross", loc=5.0, scale=2.0,)

    """
        date d distributions
    """
    sigma_d_interv = HalfNormal("sigma_d_interv", scale=0.3)
    sigma_d_country = HalfNormal("sigma_d_country", scale=0.3)
    # delta_d_i was set in intervention class see above
    delta_d_c = Normal(
        "delta_d_c",
        loc=0.0,
        scale=1.0,
        event_stack=(modelParams.num_countries),
        shape_label=("country"),
    )

    # We create a dict here to pass all distributions to another function
    distributions = {}
    distributions["sigma_a_c"] = sigma_a_c
    distributions["sigma_a_a"] = sigma_a_a
    distributions["delta_alpha_cross_c"] = delta_alpha_cross_c
    distributions["delta_alpha_cross_a"] = delta_alpha_cross_a
    distributions["sigma_l_interv"] = sigma_l_interv
    distributions["l_positive_cross"] = l_positive_cross
    distributions["l_negative_cross"] = l_negative_cross
    distributions["sigma_d_interv"] = sigma_d_interv
    distributions["sigma_d_country"] = sigma_d_country
    distributions["delta_d_c"] = delta_d_c

    return interventions, distributions


def construct_R_t(R_0, modelParams):
    """
    Constructs the time dependent reproduction number :math:`R(t)` for every country and age group.

    Parameter
    ---------

    R_0:
        |shape| batch, country, age group

    modelParams: 

    Return
    ------
    R_t:
        Reproduction number matrix.
        |shape| time, batch, country, age group
    """
    # Create distributions for date and hyperpriors.
    interventions, distributions = _create_distributions(modelParams)

    log.debug("construct_R_t")

    """ Construct alpha_i_c_a
    """
    # Multiply distributions by hyperpriors
    delta_alpha_cross_c = (yield distributions["delta_alpha_cross_c"]) * (
        yield distributions["sigma_a_c"]
    )
    delta_alpha_cross_a = (yield distributions["delta_alpha_cross_a"]) * (
        yield distributions["sigma_a_a"]
    )

    # Stack distributions for easy addition in the loop (is there a better way?)
    delta_alpha_cross_c = tf.stack(
        [delta_alpha_cross_c] * modelParams.num_age_groups, axis=-1
    )  # shape c,a
    delta_alpha_cross_a = tf.stack(
        [delta_alpha_cross_a] * modelParams.num_countries, axis=-2
    )  # shape c,a

    # For each intervention create alpha tensor of shape country, age_group and add it to the dict
    alpha = {}  # i.e alpha_i_c_a
    for i_name, intervention in interventions.items():
        # Draw from distributions
        alpha_cross_i = yield intervention.alpha_cross
        # Add the country age and intervention distribution
        alpha_cross_i_c_a = alpha_cross_i + delta_alpha_cross_c + delta_alpha_cross_a
        alpha[i_name] = 1.0 / (1.0 + tf.exp(-1.0 * alpha_cross_i_c_a))
        log.debug(f"Alpha_{i_name}\n{alpha[i_name]}")

    """ Construct l_{i,sign}
        Create length of the changepoints
    """
    sigma_l_interv = yield distributions["sigma_l_interv"]
    l_positive_cross = yield distributions["l_positive_cross"]
    l_negative_cross = yield distributions["l_negative_cross"]
    # For each interventions we create a length l
    length = {}  # i.e.
    for i_name, intervention in interventions.items():
        delta_l_cross = (yield intervention.delta_l_cross) * sigma_l_interv

        # TODO:  NEED TO DETECT WHEATHER TO USE POSITIVE OR NEGATIVE l_cross
        l_cross_i_sign = l_positive_cross + delta_l_cross

        length[i_name] = tf.math.log(1.0 + tf.exp(l_cross_i_sign))
        log.debug(f"Length_{i_name}\n{length[i_name]}")

    """ Construct d_i_c_p
        Create dates for each change point
    """
    sigma_d_interv = yield distributions["sigma_d_interv"]
    delta_d_c = (yield distributions["delta_d_c"]) * (
        yield distributions["sigma_d_country"]
    )

    d = {}  # i.e. d_i_c_p
    for i_name, intervention in interventions.items():
        d[i_name] = {}

        # Sample from intervention distribution
        delta_d_i = (yield intervention.delta_d) * sigma_d_interv

        country_index = 0
        for country in modelParams.countries:
            d[i_name][country.name] = []
            # Get value from country distribution
            delta_d_c_gather = tf.gather(delta_d_c, country_index, axis=-1)
            country_index = country_index + 1

            for cp_index, change_point in enumerate(country.change_points[i_name]):
                d_data = modelParams.date_to_index(
                    country.change_points[i_name][cp_index].date_data
                )
                # Add everything and append to dict
                d[i_name][country.name].append(d_data + delta_d_i + delta_d_c_gather)
            log.debug(f"Date_{i_name}_{country.name}\n{d[i_name][country.name]}")

    """ Construct gamma_i_c_p
        Loop over interventions countries changepoints
    """
    # Create time index tensor of length modelParams.simlength
    t = tf.range(0, modelParams.length, dtype="float32")

    gamma_i_c_p = {}  # i.e. gamma_i_c_p
    for i_name, intervention in interventions.items():
        gamma_i_c_p[i_name] = {}
        for country in modelParams.countries:
            gamma_i_c_p[i_name][country.name] = []
            for cp_index, change_point in enumerate(country.change_points[i_name]):
                delta_gamma_max_data = country.change_points[i_name][cp_index].gamma_max
                gamma_i_c_p[i_name][country.name].append(
                    delta_gamma_max_data
                    / (
                        1.0
                        + tf.exp(
                            -4
                            / length[i_name]
                            * (t - d[i_name][country.name][cp_index])
                        )
                    )
                )
            log.debug(
                f"Gamma_i_c_p_{i_name}_{country.name}\n{gamma_i_c_p[i_name][country.name]}"
            )

    """ Calculate gamma_i_c

        Iterate over all changepoints in an intervention and sum up gamma
    """
    gamma_i_c = {}
    for i_name, intervention in interventions.items():
        gamma_i_c[i_name] = []  # shape country
        for country in modelParams.countries:
            gamma_i_c[i_name].append(
                tf.reduce_sum(gamma_i_c_p[i_name][country.name], axis=0)
            )
        gamma_i_c[i_name] = tf.convert_to_tensor(
            gamma_i_c[i_name]
        )  # shape country,time
        log.debug(f"Gamma_i_c_{i_name}\n{gamma_i_c[i_name]}")

    """ Calculate R_eff

        Iterate over all interventions
    """
    _sum = []
    for i_name, intervention in interventions.items():
        _sum.append(tf.einsum("...ct,...ca->...cat", gamma_i_c[i_name], alpha[i_name]))

    R_eff = tf.einsum("...ca,...cat->...tca", R_0, tf.exp(tf.reduce_sum(_sum, axis=0)))
    log.debug(f"R_eff\n{R_eff}")

    R_t = yield Deterministic(
        name="R_t",
        value=R_eff,
        shape=(
            modelParams.length,
            modelParams.num_countries,
            modelParams.num_age_groups,
        ),
        shape_label=("time", "country", "age_group"),
    )

    if len(R_t.shape) == 4:
        R_t = tf.transpose(R_t, perm=(1, 0, 2, 3))

    return R_t
