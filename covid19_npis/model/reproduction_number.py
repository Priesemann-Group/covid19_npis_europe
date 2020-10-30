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
    Gamma,
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
    log.debug(f"d in _fsigmoid\n{d}")
    log.debug(f"t in _fsigmoid\n{t}")
    inside_exp_1 = 4.0 / l
    inside_exp_2 = t - d
    log.debug(f"t-d\n{inside_exp_2}")
    inside_exp_1 = tf.expand_dims(inside_exp_1, axis=-1)
    return tf.math.sigmoid(inside_exp_1 * inside_exp_2)


def _create_distributions(modelParams):
    r"""
        Returns a dict of distributions for further processing/sampling with the following priors:

        .. math::

            \alpha^\dagger_i &\sim \mathcal{N}\left(-1, 2\right)\quad \forall i,\\
            \Delta \alpha^\dagger_c &\sim \mathcal{N}\left(0, \sigma_{\alpha, \text{country}}\right) \quad \forall c, \\
            \Delta \alpha^\dagger_a &\sim \mathcal{N}\left(0, \sigma_{\alpha, \text{age}}\right)\quad \forall a, \\
            \sigma_{\alpha, \text{country}}  &\sim HalfNormal\left(0.1\right),\\
            \sigma_{\alpha, \text{age}} &\sim HalfNormal\left(0.1\right)
        
        .. math::

            l^\dagger_{\text{positive}} &\sim \mathcal{N}\left(3, 1\right),\\
            l^\dagger_{\text{negative}} &\sim \mathcal{N}\left(5, 2\right),\\
            \Delta l^\dagger_i &\sim \mathcal{N}\left(0,\sigma_{l, \text{interv.}} \right)\quad \forall i,\\
            \sigma_{l, \text{interv.}}&\sim HalfNormal\left(1\right)

        .. math::

            \Delta d_i  &\sim \mathcal{N}\left(0, \sigma_{d, \text{interv.}}\right)\quad \forall i,\\
            \Delta d_c &\sim \mathcal{N}\left(0, \sigma_{d, \text{country}}\right)\quad \forall c,\\
            \sigma_{d, \text{interv.}}  &\sim HalfNormal\left(0.3\right),\\
            \sigma_{d, \text{country}} &\sim HalfNormal\left(0.3\right)

        Parameters
        ----------
        modelParams: :py:class:`covid19_npis.ModelParams`
            Instance of modelParams, mainly used for number of age groups and
            number of countries. 

        Return
        ------
        :
            interventions, distributions
    """
    log.debug("_create_distributions")

    """
        Δ Alpha cross for each country and age group with hyperdistributions
    """
    sigma_a_c = HalfNormal(
        "sigma_alpha_country",
        scale=0.1,
        transform=transformations.SoftPlus(scale=0.1),
        conditionally_independent=True,
    )
    sigma_a_a = HalfNormal(
        "sigma_alpha_age_group",
        scale=0.1,
        transform=transformations.SoftPlus(scale=0.1),
        conditionally_independent=True,
    )
    # We need to multiply sigma_a_c and sigma_a_a later. (See construct R_t)
    delta_alpha_cross_c = Normal(
        "delta_alpha_cross_c",
        loc=0.0,
        scale=1.0,
        event_stack=(1, modelParams.num_countries, 1),  # intervention country age_group
        shape_label=(None, "country", None),
        conditionally_independent=True,
    )
    delta_alpha_cross_a = Normal(
        "delta_alpha_cross_a",
        loc=0.0,
        scale=1.0,
        event_stack=(
            1,
            1,
            modelParams.num_age_groups,
        ),  # intervention country age_group
        shape_label=(None, None, "age_group"),
        conditionally_independent=True,
    )
    alpha_cross_i = Normal(
        "alpha_cross_i",
        loc=-1.0,  # See publication for reasoning behind -1 and 2
        scale=2.0,
        conditionally_independent=True,
        event_stack=(
            modelParams.num_interventions,
            1,
            1,
        ),  # intervention country age_group
        shape_label=("intervention", None, None),
    )

    """
        l distributions
    """
    sigma_l_interv = HalfNormal(
        "sigma_l_interv",
        scale=1.0,
        transform=transformations.SoftPlus(),
        conditionally_independent=True,
    )

    delta_l_cross_i = Normal(
        "delta_l_cross_i",
        loc=0.0,
        scale=1.0,
        conditionally_independent=True,
        event_stack=(modelParams.num_interventions,),
        shape_label=("intervention"),
    )
    log.debug(f"sigma_l_interv\n{sigma_l_interv}")
    # Δl_i^cross was created in intervention class see above
    l_positive_cross = Normal(
        "l_positive_cross",
        loc=3.0,
        scale=1.0,
        conditionally_independent=True,
        event_stack=(1,),
    )
    l_negative_cross = Normal(
        "l_negative_cross",
        loc=5.0,
        scale=2.0,
        conditionally_independent=True,
        event_stack=(1,),
    )

    """
        date d distributions
    """
    sigma_d_interv = HalfNormal(
        "sigma_d_interv",
        scale=0.3,
        transform=transformations.SoftPlus(scale=0.3),
        conditionally_independent=True,
    )
    sigma_d_country = HalfNormal(
        "sigma_d_country",
        scale=0.3,
        transform=transformations.SoftPlus(scale=0.3),
        conditionally_independent=True,
    )
    delta_d_i = Normal(
        "delta_d_i",
        loc=0.0,
        scale=1.0,
        event_stack=(modelParams.num_interventions, 1, 1),
        shape_label=("intervention", None, None),
        conditionally_independent=True,
    )
    delta_d_c = Normal(
        "delta_d_c",
        loc=0.0,
        scale=1.0,
        event_stack=(1, modelParams.num_countries, 1),
        shape_label=(None, "country", None),
        conditionally_independent=True,
    )

    # We create a dict here to pass all distributions to another function
    distributions = {}
    distributions["sigma_a_c"] = sigma_a_c
    distributions["sigma_a_a"] = sigma_a_a
    distributions["delta_alpha_cross_c"] = delta_alpha_cross_c
    distributions["delta_alpha_cross_a"] = delta_alpha_cross_a
    distributions["alpha_cross_i"] = alpha_cross_i
    distributions["sigma_l_interv"] = sigma_l_interv
    distributions["l_positive_cross"] = l_positive_cross
    distributions["l_negative_cross"] = l_negative_cross
    distributions["delta_l_cross_i"] = delta_l_cross_i
    distributions["sigma_d_interv"] = sigma_d_interv
    distributions["sigma_d_country"] = sigma_d_country
    distributions["delta_d_i"] = delta_d_i
    distributions["delta_d_c"] = delta_d_c

    return distributions


def construct_R_t(R_0, modelParams):
    r"""
        Constructs the time dependent reproduction number :math:`R(t)` for every country and age group.
        There are a lot of things happening here be sure to check our paper for more indepth explanations! 

        We build the effectivity in an hierarchical manner in the unbounded space:

        .. math::

            \alpha_{i,c,a} &= \frac{1}{1+e^{-\alpha^\dagger_{i,c,a}}},\\
            \alpha^\dagger_{i,c,a} &= \alpha^\dagger_i + \Delta \alpha^\dagger_c + \Delta \alpha^\dagger_{a}
        
        The length of the change point depends on the intervention and whether the
        strength is increasing or decreasing:

        .. math::

            l_{i, \text{sign}\left(\Delta \gamma\right)} &= \ln\left(1 + e^{l^\dagger_{i, \text{sign}\left(\Delta \gamma\right)}}\right),\\
            l^\dagger_{i, \text{sign}\left(\Delta \gamma\right)} &= l^\dagger_{\text{sign}\left(\Delta \gamma\right)} + \Delta l^\dagger_i,

        The date of the begin of the intervention is also allowed to vary slightly around the date :math:`d^{\text{data}}_{i,c}`
        given by the Oxford government response tracker:

        .. math::

            d_{i,c,p} &= d^{\text{data}}_{i,c,p} + \Delta d_i +\Delta d_c

        And finally the time dependent reproduction number :math:`R^*_e`:

        .. math::
            
            \gamma_{i,c,p}(t) &= \frac{1}{1+e^{-4/l_{i, \text{sign}\left(\Delta \gamma\right)} \cdot (t - d_{i,c,p})}} \cdot \Delta \gamma_{i,c,p}^{\text{data}}\\
            \gamma_{i,c}(t) &= \sum_p \gamma_{i,c,p}(t)\\
            R^*_e &= R^*_0 e^{-\sum_i^{N_i}\alpha_{i, c, a} \gamma_{i,c}(t)}

        We also sometimes call the time dependent reproduction number R_t!


        Parameters
        ----------

        R_0: tf.tensor
            Initial reproduction number. Should be constructed using 
            :py:func:`construct_R_0` or :py:func:`construct_R_0_old`.
            |shape| batch, country, age group

        modelParams: :py:class:`covid19_npis.ModelParams`
            Instance of modelParams, mainly used for number of age groups and
            number of countries. 

        Return
        ------
        :
            Time dependent reproduction number tensor :math:`R(t)`.
            |shape| time, batch, country, age group
    """
    # Create distributions for date and hyperpriors.
    distributions = _create_distributions(modelParams)

    log.debug("construct_R_t")

    def alpha():
        """
        Helper function to construct the alpha_i_c_a tensor
        """
        delta_alpha_cross_c = yield distributions["delta_alpha_cross_c"]
        sigma_a_c = yield distributions["sigma_a_c"]

        delta_alpha_cross_c = tf.einsum(  # Multiply distribution by hyperprior
            "...ica,...->...ica", delta_alpha_cross_c, sigma_a_c
        )

        delta_alpha_cross_a = yield distributions["delta_alpha_cross_a"]
        sigma_a_a = yield distributions["sigma_a_a"]
        delta_alpha_cross_a = tf.einsum(  # Multiply distribution by hyperprior
            "...ica,...->...ica", delta_alpha_cross_a, sigma_a_a
        )
        # Draw for the interventions
        alpha_cross_i = yield distributions["alpha_cross_i"]
        # Add all together, dimensions are defined in _create_distributions
        alpha_cross_i_c_a = alpha_cross_i + delta_alpha_cross_c + delta_alpha_cross_a
        return tf.math.sigmoid(alpha_cross_i_c_a)

    alpha_i_c_a = yield Deterministic(
        name="alpha_i_c_a",
        value=(yield alpha()),
        shape_label=("intervention", "country", "age_group",),
    )
    log.debug(f"alpha_i_c_a\n{alpha_i_c_a}")

    def length():
        """
        Helper function to construct the l_i,sign(Δγ) tensor
        """
        delta_l_cross_i = yield distributions["delta_l_cross_i"]
        sigma_l_interv = yield distributions["sigma_l_interv"]
        delta_l_cross_i = tf.einsum(  # Multiply distribution by hyperprior
            "...i,...->...i", delta_l_cross_i, sigma_l_interv
        )
        l_positive_cross = yield distributions["l_positive_cross"]
        l_negative_cross = yield distributions["l_negative_cross"]
        # TODO:  NEED TO DETECT WHEATHER TO USE POSITIVE OR NEGATIVE l_cross
        l_cross_i_sign = l_positive_cross + delta_l_cross_i

        return tf.math.softplus(l_cross_i_sign)

    l_i_sign = yield Deterministic(
        name="l_i_sign",
        value=(yield length()),
        shape_label=("intervention"),  # intervention
    )
    log.debug(f"l_i_sign\n{l_i_sign}")

    def date():
        """
        Helper funtion to return the date tensors.
        Returns multiple tensors one for each change point.
        """
        delta_d_i = yield distributions["delta_d_i"]
        sigma_d_interv = yield distributions["sigma_d_interv"]
        delta_d_i = tf.einsum(  # Multiply distribution by hyperprior
            "...ica,...->...ica", delta_d_i, sigma_d_interv
        )

        delta_d_c = yield distributions["delta_d_c"]
        sigma_d_country = yield distributions["sigma_d_country"]
        delta_d_c = tf.einsum(  # Multiply distribution by hyperprior
            "...ica,...->...ica", delta_d_c, sigma_d_country
        )
        # Get data tensor padded with 0 if the cp does not exist for an intervention/country combo
        d_data = (
            modelParams.date_data_tensor
        )  # shape intervention, country, change_points

        return d_data + delta_d_i + delta_d_c

    d_i_c_p = yield Deterministic(
        name="d_i_c_p",
        value=(yield date()),
        shape_label=("intervention", "country", "change_point",),
    )
    log.debug(f"d_i_c_p\n{d_i_c_p}")

    def gamma(d_i_c_p, l_i_sign):
        """
        Helper function to construct gamma_i_c_p and calculate gamma_i_c
        """
        # Create time index tensor of length modelParams.simlength
        t = tf.range(0, modelParams.length, dtype="float32")

        # We need to expand the dims of d_icp because we need a additional time dimension
        # for "t - d_icp"
        d_i_c_p = tf.expand_dims(d_i_c_p, axis=-1)
        inner_sigmoid = tf.einsum("...i,...icpt->...icpt", 4.0 / l_i_sign, t - d_i_c_p)
        log.debug(f"inner_sigmoid\n{inner_sigmoid}")
        gamma_i_c_p = tf.einsum(
            "...icpt,icp->...icpt",
            tf.math.sigmoid(inner_sigmoid),
            modelParams.gamma_data_tensor,
        )
        log.debug(
            f"gamma_i_c_p\n{gamma_i_c_p}"
        )  # shape inter, country, changepoint, time

        """ Calculate gamma_i_c from gamma_i_c_p
            Iterate over all changepoints in an intervention and sum up over every change point.
            We padded the date tensor earlier so we do NOT want to sum over the additional entries!
        """
        list_gamma_i_c = []
        for i, intervention in enumerate(
            modelParams.countries[0].interventions
        ):  # Should be same across all countries -> 0
            list_gamma_c = []
            gamma_c_p = tf.gather(gamma_i_c_p, i, axis=-4)
            for c, country in enumerate(modelParams.countries):
                gamma_p = tf.gather(gamma_c_p, c, axis=-3)

                # Cut gamma_p to get only the used change points values
                # i.e remove padding!
                num_change_points = len(country.change_points[intervention.name])
                gamma_p = gamma_p[..., 0:num_change_points, :]

                # Calculate the sum over all change points
                gamma_values = tf.math.reduce_sum(gamma_p, axis=-2)
                list_gamma_c.append(gamma_values)
            list_gamma_i_c.append(list_gamma_c)

        gamma = tf.convert_to_tensor(list_gamma_i_c)

        # Transpose batch into front
        gamma = tf.einsum("ic...t -> ...ict", gamma)

        return gamma

    gamma_i_c = gamma(
        d_i_c_p, l_i_sign
    )  # no yield because we do not sample anything in this function
    log.debug(f"gamma_i_c\n{gamma_i_c}")
    log.debug(f"alpha_i_c_a\n{alpha_i_c_a}")
    """ Calculate R_eff
    """
    exponent = tf.einsum("...ict,...ica->...cat", gamma_i_c, alpha_i_c_a)

    R_eff = tf.einsum("...ca,...cat->...tca", R_0, tf.exp(-exponent))
    log.debug(f"R_eff\n{R_eff}")

    R_t = yield Deterministic(
        name="R_t", value=R_eff, shape_label=("time", "country", "age_group"),
    )

    R_t = tf.einsum("...tca -> t...ca", R_t)

    return R_t  # shape time, batch, country, age_group


def construct_R_0(name, loc, scale, hn_scale, modelParams):
    r"""
        Constructs R_0 in the following hierarchical manner:

        .. math::

            R^*_{0,c} &= R^*_0 + \Delta R^*_{0,c}, \\
            R^*_0 &\sim \mathcal{N}\left(2,0.5\right)\\
            \Delta R^*_{0,c} &\sim \mathcal{N}\left(0, \sigma_{R^*, \text{country}}\right)\quad \forall c,\\
            \sigma_{R^*, \text{country}} &\sim HalfNormal\left(0.3\right)

        Parameters
        ----------
        name: str
            Name of the distribution (gets added to trace).
        loc: number
            Location parameter of the R^*_0 Normal distribution.
        scale: number
            Scale paramter of the R^*_0 Normal distribution.
        hn_scale: number
            Scale parameter of the \sigma_{R^*, \text{country}} HaflNormal distribution.

        Returns
        -------
        :
            R_0 tensor |shape| batch, country, age_group
    """

    R_0_star = yield Normal(
        name="R_0_star",
        loc=0.0,
        scale=scale,
        conditionally_independent=True,
        # transform=transformations.Normal(shift=loc),
    )

    sigma_R_0_c = yield HalfNormal(
        name="sigma_R_0_c",
        scale=hn_scale,
        conditionally_independent=True,
        transform=transformations.SoftPlus(scale=hn_scale),
    )

    delta_R_0_c = (
        yield Normal(
            name="delta_R_0_c",
            loc=0.0,
            scale=1.0,
            event_stack=(modelParams.num_countries),
            shape_label=("country"),
            conditionally_independent=True,
        )
    ) * sigma_R_0_c[..., tf.newaxis]

    # Add to trace via deterministic
    R_0 = R_0_star[..., tf.newaxis] + delta_R_0_c
    # Softplus because we want to make sure that R_0 > 0.
    R_0 = 0.1 * tf.math.softplus(10 * R_0)
    R_0 = yield Deterministic(name=name, value=R_0, shape_label=("country"),)

    return tf.repeat(R_0[..., tf.newaxis], repeats=modelParams.num_age_groups, axis=-1)


def construct_R_0_old(name, mean, beta, modelParams):
    r"""
    Old constructor of :math:`R_0` using a gamma distribution:

    .. math::

        R_0 &\sim Gamma\left(\mu=2.5,\beta=2.0\right)

    Parameters
    ----------
    name: string
        Name of the distribution for trace and debugging.
    mean:
        Mean :math:`\mu` of the gamma distribution.
    beta:
        Rate :math:`\beta` of the gamma distribution.

    Returns
    -------
    :
        R_0 tensor |shape| batch, country, age_group

    """
    event_shape = (modelParams.num_countries, modelParams.num_age_groups)
    R_0 = yield Gamma(
        name=name,
        concentration=mean * beta,
        rate=beta,
        conditionally_independent=True,
        event_stack=event_shape,
        transform=transformations.SoftPlus(),
        shape_label=("country", "age_group"),
    )
    return R_0
