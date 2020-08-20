import tensorflow as tf
import logging
import pymc4 as pm
from .utils import gamma
import tensorflow_probability as tfp
import numpy as np

log = logging.getLogger(__name__)

from covid19_npis import transformations
from covid19_npis.model.distributions import Normal, LogNormal, Deterministic
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
        self, name, alpha_loc_loc, alpha_loc_scale, alpha_scale_loc, alpha_scale_scale
    ):
        self.name = name

        # Distributions
        self.prior_alpha_loc_loc = alpha_loc_loc
        self.prior_alpha_loc_scale = alpha_loc_scale
        self.prior_alpha_scale_loc = alpha_scale_loc
        self.prior_alpha_scale_scale = alpha_scale_scale

        log.debug(f"tsty\n{self.prior_alpha_loc_scale}")
        # Init distributions
        # TODO: Add scale
        self._alpha_loc = LogNormal(
            "alpha_loc_" + self.name,
            np.log(self.prior_alpha_loc_loc, dtype="float32"),
            np.log(self.prior_alpha_loc_scale, dtype="float32"),
            conditionally_independent=True,
            transformation=transformations.SoftPlus(
                scale=20, reinterpreted_batch_ndims=0
            ),
        )

        self._alpha_scale = LogNormal(
            "alpha_scale_" + self.name,
            np.log(self.prior_alpha_scale_loc, dtype="float32"),
            np.log(self.prior_alpha_scale_scale, dtype="float32"),
            conditionally_independent=True,
            transformation=transformations.SoftPlus(
                scale=20, reinterpreted_batch_ndims=0
            ),
        )
        log.debug(f"Create intervention with name: {name}")

    @property
    def alpha_loc(self):
        r"""
        Returns pymc4 generator for the effetivity :math:`\alpha`, i.e. a normal distribution. The priors
        are set at init of the object.
        """

        return (yield self._alpha_loc)

    @property
    def alpha_scale(self):
        r"""
        Returns pymc4 generator for the effetivity :math:`\alpha`, i.e. a normal distribution. The priors
        are set at init of the object.
        """

        return (yield self._alpha_scale)


def _create_distributions(modelParams):
    """
    Returns a list of interventions :py:func:`covid19_npis.reproduction_number.Intervention` from
    a change point dict.

    Parameter
    ---------
    change_points : dict
        Dict housing every parameter for the interventions, i.e.
        :math:`l_{i}`
        :math:`d_{i,n}`
        :math:`alpha_{i}`
        :math:`i` being interventions and :math:`n` being a change point.

    Return
    ------
    :
        Interventions array like
        |shape| country, interventions
    """
    log.debug("_create_distributions")
    """
    Get all interventions from the countries data objects
    """
    interventions_data = modelParams.countries[0].interventions

    """
    Create hyperprior for each intervention
    """
    interventions = {}
    for i in interventions_data:
        interventions[i.name] = Intervention(
            name=i.name,
            alpha_loc_loc=i.prior_alpha_loc,
            alpha_loc_scale=i.prior_alpha_loc / 5,
            alpha_scale_loc=i.prior_alpha_scale,
            alpha_scale_scale=i.prior_alpha_scale / 5,
        )

    """
        Create dict with distributions
    """
    countries = {}
    for country in modelParams.countries:
        countries[country.name] = {}
        for intervention_name, change_points in country.change_points.items():
            countries[country.name][intervention_name] = []
            # Create changepoint
            for i, change_point in enumerate(change_points):
                countries[country.name][intervention_name].append(
                    Change_point(
                        name=f"{country.name}_{intervention_name}_{i}",
                        date_loc=modelParams.date_to_index(change_point.prior_date_loc),
                        date_scale=change_point.prior_date_scale,
                        gamma_max=change_point.gamma_max,
                    )
                )
    return interventions, countries


def construct_R_t(R_0, modelParams):
    """
    Constructs the time dependent reproduction number :math:`R(t)` for every country and age group.

    Parameter
    ---------

    R_0:
        |shape| batch, country, age group

    Interventions: array like covid19_npis.reproduction_number.Intervention
        |shape| country

    Return
    ------
    R_t:
        Reproduction number matrix.
        |shape| time, batch, country, age group
    """
    log.debug("construct_R_t")

    # Create distributions for date and hyperpriors.
    interventions, countries = _create_distributions(modelParams)

    # Get alpha for each intervention,country event stack of country size
    alpha = {}
    for i_name, intervention in interventions.items():
        # Get hyperpriors
        alpha_loc = yield intervention.alpha_loc
        alpha_scale = yield intervention.alpha_scale
        log.debug(f"Hyperdist_Alpha:\nloc:{alpha_loc}\nscale:{alpha_scale}")

        # Create alpha for each country
        alpha[i_name] = yield Normal(
            name=f"alpha_{i_name}",
            loc=alpha_loc,
            scale=alpha_scale,
            shape_label=("country"),
            event_stack=modelParams.num_countries,
        )
        log.debug(f"Alpha_{i_name}\n{alpha[i_name]}")

    """ We want to create a time dependent R_t for each country and age group
        We iterate over country and interventions.
    """
    # Create time index tensor of length modelParams.simlength
    t = tf.range(0, modelParams.length, dtype="float32")

    exp_to_multi = []
    country_index = 0
    for country_name, country_interventions in countries.items():
        _sum = []
        for intervention_name, changepoints in country_interventions.items():

            alpha_interv = tf.gather(alpha[intervention_name], country_index, axis=-1)
            log.debug(
                f"Alpha_sliced {country_name} {intervention_name}\n{alpha_interv}"
            )
            # Calculate the gamma value for each cp and sum them up
            gammas_cp = []
            for cp in changepoints:
                gamma_cp = yield cp.gamma_t(t, l=5.0)
                gammas_cp.append(gamma_cp)
            log.debug(f"gammas_cp:\n{gammas_cp}")
            gamma_t = tf.reduce_sum(gammas_cp, axis=0)
            log.debug(f"gamma_t reduced_sum:\n{gammas_cp}")
            # Append gamma*alpha to array
            _sum.append(tf.einsum("...ai,...->...ai", gamma_t, alpha_interv))

        # We sum over all interventions in a country and append to list for countries
        print(f"sum:{_sum}")
        exp_to_multi.append(tf.exp(tf.reduce_sum(_sum, axis=0)))
        country_index = country_index + 1
    exp_to_multi = tf.convert_to_tensor(exp_to_multi)

    if len(exp_to_multi.shape) == 4:
        # before |shape| country, batch, agegroup, time
        exp_to_multi = tf.transpose(exp_to_multi, perm=(1, 0, 2, 3))
        # after |shape| batch, country, agegroup, time

    """
    Multiply R_0 with the exponent, i.e.
        R(t) = R_0 * exp(sum_i(gamma_i(t)*alpha_i))
        R_0: |shape| batch, country, agegroup
        exp: |shape| batch, country, agegroup, time
    """
    log.debug(f"country exponential function:\n{exp_to_multi.shape}")
    log.debug(f"R_0:\n{R_0.shape}")

    R_t = tf.einsum(
        "...ca,...cat->...tca", R_0, exp_to_multi
    )  # Reshape to |shape| batch, time. country, age group here
    log.debug(f"R_t_inside:\n{R_t.shape}")

    R_t = yield Deterministic(
        name="R_t",
        value=R_t,
        shape=(50, 2, 4),
        shape_label=("time", "country", "age_group"),
    )

    if len(R_t.shape) == 4:
        R_t = tf.transpose(R_t, perm=(1, 0, 2, 3))

    return R_t
