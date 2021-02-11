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
    alpha_sigma_c = HalfNormal(
        name="alpha_sigma_country",
        scale=0.1,
        transform=transformations.SoftPlus(scale=0.1),
        conditionally_independent=True,
    )
    alpha_sigma_a = HalfNormal(
        name="alpha_sigma_age_group",
        scale=0.1,
        transform=transformations.SoftPlus(scale=0.1),
        conditionally_independent=True,
    )
    # We need to multiply alpha_sigma_c and alpha_sigma_a later. (See construct R_t)
    delta_alpha_cross_c = Normal(
        name="delta_alpha_cross_c",
        loc=0.0,
        scale=1.0,
        event_stack=(1, modelParams.num_countries, 1),  # intervention country age_group
        shape_label=(None, "country", None),
        conditionally_independent=True,
    )
    delta_alpha_cross_a = Normal(
        name="delta_alpha_cross_a",
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
        name="alpha_cross_i",
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
    l_sigma_interv = HalfNormal(
        name="l_sigma_interv",
        scale=1.0,
        transform=transformations.SoftPlus(),
        conditionally_independent=True,
    )

    delta_l_cross_i = Normal(
        name="delta_l_cross_i",
        loc=0.0,
        scale=1.0,
        conditionally_independent=True,
        event_stack=(modelParams.num_interventions,),
        shape_label=("intervention"),
    )
    log.debug(f"l_sigma_interv\n{l_sigma_interv}")
    # Δl_i^cross was created in intervention class see above
    l_positive_cross = Normal(
        name="l_positive_cross",
        loc=3.0,
        scale=1.0,
        conditionally_independent=True,
        event_stack=(1,),
    )
    l_negative_cross = Normal(
        name="l_negative_cross",
        loc=5.0,
        scale=2.0,
        conditionally_independent=True,
        event_stack=(1,),
    )

    """
        date d distributions
    """
    d_sigma_interv = HalfNormal(
        name="d_sigma_interv",
        scale=0.3,
        transform=transformations.SoftPlus(scale=0.3),
        conditionally_independent=True,
    )
    d_sigma_country = HalfNormal(
        name="d_sigma_country",
        scale=0.3,
        transform=transformations.SoftPlus(scale=0.3),
        conditionally_independent=True,
    )
    delta_d_i = Normal(
        name="delta_d_i",
        loc=0.0,
        scale=1.0,
        event_stack=(modelParams.num_interventions, 1, 1),
        shape_label=("intervention", None, None),
        conditionally_independent=True,
    )
    delta_d_c = Normal(
        name="delta_d_c",
        loc=0.0,
        scale=1.0,
        event_stack=(1, modelParams.num_countries, 1),
        shape_label=(None, "country", None),
        conditionally_independent=True,
    )

    # We create a dict here to pass all distributions to another function
    distributions = {}
    distributions["alpha_sigma_c"] = alpha_sigma_c
    distributions["alpha_sigma_a"] = alpha_sigma_a
    distributions["delta_alpha_cross_c"] = delta_alpha_cross_c
    distributions["delta_alpha_cross_a"] = delta_alpha_cross_a
    distributions["alpha_cross_i"] = alpha_cross_i
    distributions["l_sigma_interv"] = l_sigma_interv
    distributions["l_positive_cross"] = l_positive_cross
    distributions["l_negative_cross"] = l_negative_cross
    distributions["delta_l_cross_i"] = delta_l_cross_i
    distributions["d_sigma_interv"] = d_sigma_interv
    distributions["d_sigma_country"] = d_sigma_country
    distributions["delta_d_i"] = delta_d_i
    distributions["delta_d_c"] = delta_d_c

    return distributions


def construct_R_t(name, modelParams, R_0):
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

        name: str
            Name of the distribution (gets added to trace).
        modelParams: :py:class:`covid19_npis.ModelParams`
            Instance of modelParams, mainly used for number of age groups and
            number of countries.
        R_0: tf.tensor
            Initial reproduction number. Should be constructed using
            :py:func:`construct_R_0` or :py:func:`construct_R_0_old`.
            |shape| batch, country, age group

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
        alpha_sigma_c = yield distributions["alpha_sigma_c"]

        delta_alpha_cross_c = tf.einsum(  # Multiply distribution by hyperprior
            "...ica,...->...ica", delta_alpha_cross_c, alpha_sigma_c
        )

        delta_alpha_cross_a = yield distributions["delta_alpha_cross_a"]
        alpha_sigma_a = yield distributions["alpha_sigma_a"]
        delta_alpha_cross_a = tf.einsum(  # Multiply distribution by hyperprior
            "...ica,...->...ica", delta_alpha_cross_a, alpha_sigma_a
        )
        # Draw for the interventions
        alpha_cross_i = yield distributions["alpha_cross_i"]
        # Add all together, dimensions are defined in _create_distributions
        alpha_cross_i_c_a = alpha_cross_i + delta_alpha_cross_c + delta_alpha_cross_a
        yield Deterministic(
            name="alpha_i_a",
            value=tf.math.sigmoid(alpha_cross_i + delta_alpha_cross_a)[..., :, 0, :],
            shape_label=("intervention", "age_group"),
        )

        return tf.math.sigmoid(alpha_cross_i_c_a)

    def length():
        """
        Helper function to construct the l_i,sign(Δγ) tensor
        """
        delta_l_cross_i = yield distributions["delta_l_cross_i"]
        l_sigma_interv = yield distributions["l_sigma_interv"]
        delta_l_cross_i = tf.einsum(  # Multiply distribution by hyperprior
            "...i,...->...i", delta_l_cross_i, l_sigma_interv
        )
        l_positive_cross = yield distributions["l_positive_cross"]
        l_negative_cross = yield distributions["l_negative_cross"]

        # TODO:  NEED TO DETECT WHETHER TO USE POSITIVE OR NEGATIVE l_cross
        l_cross_i_sign = l_positive_cross + delta_l_cross_i

        return tf.math.softplus(l_cross_i_sign)

    def date():
        """
        Helper funtion to return the date tensors.
        Returns multiple tensors one for each change point.
        """
        delta_d_i = yield distributions["delta_d_i"]
        d_sigma_interv = yield distributions["d_sigma_interv"]
        delta_d_i = tf.einsum(  # Multiply distribution by hyperprior
            "...ica,...->...ica", delta_d_i, d_sigma_interv
        )

        delta_d_c = yield distributions["delta_d_c"]
        d_sigma_country = yield distributions["d_sigma_country"]
        delta_d_c = tf.einsum(  # Multiply distribution by hyperprior
            "...ica,...->...ica", delta_d_c, d_sigma_country
        )
        # Get data tensor padded with 0 if the cp does not exist for an intervention/country combo
        d_data = (
            modelParams.date_data_tensor
        )  # shape intervention, country, change_points

        return d_data + delta_d_i + delta_d_c

    def gamma(d_i_c_p, l_i_sign):
        """
        Helper function to construct gamma_i_c_p and calculate gamma_i_c
        """
        # Create time index tensor of the simulation length
        t = tf.range(0, modelParams.length_sim, dtype=tf.float32)

        # We need to expand the dims of d_icp because we need a additional time dimension
        # for "t - d_icp"
        d_i_c_p = tf.expand_dims(d_i_c_p, axis=-1)

        # Get the sigmoid and multiply it with our gamma tensor
        sigmoid = tf.math.sigmoid(
            tf.einsum("...i,...icpt->...icpt", 4.0 /  (l_i_sign + 1e-3), (t - d_i_c_p))
        )
        gamma_i_c_p = tf.einsum(
            "...icpt,icp->...icpt", sigmoid, modelParams.gamma_data_tensor,
        )
        log.debug(
            f"gamma_i_c_p\n{gamma_i_c_p}"
        )  # shape inter, country, changepoint, time

        """ Calculate gamma_i_c from gamma_i_c_p
            Iterate over all changepoints in an intervention and sum up over every change point.
            We padded the date tensor earlier so we do NOT want to sum over the additional entries!
        """
        gamma_list_i_c = []
        for i, intervention in enumerate(
            modelParams.data_summary["interventions"]
        ):  # Should be same across all countries -> 0
            gamma_list_c = []
            for c, country in enumerate(modelParams.countries):

                # Cut gamma_p to get only the used change points values
                # remove padding!
                num_change_points = len(country.change_points[intervention])
                gamma_p = gamma_i_c_p[..., i, c, 0:num_change_points, :]
                log.debug(f"gamma_p\n{gamma_p}")

                # Calculate the sum over all change points
                gamma_values = tf.math.reduce_sum(gamma_p, axis=-2)
                log.debug(f"gamma_values\n{gamma_values}")

                # Add all countries
                gamma_list_c.append(gamma_values)
            # Add all interventions
            gamma_list_i_c.append(gamma_list_c)

        gamma = tf.convert_to_tensor(gamma_list_i_c)
        log.debug(f"gamma_asd\n{gamma}")
        # Transpose batch into front
        gamma = tf.einsum("ic...t -> ...ict", gamma)

        return gamma

    """ Use helper functions to get basic tensors
    """
    alpha_i_c_a = yield Deterministic(
        name="alpha_i_c_a",
        value=(yield alpha()),
        shape_label=("intervention", "country", "age_group",),
    )
    log.debug(f"alpha_i_c_a\n{alpha_i_c_a}")

    l_i_sign = yield Deterministic(
        name="l_i_sign", value=(yield length()), shape_label=("intervention",),
    )
    log.debug(f"l_i_sign\n{l_i_sign}")

    d_i_c_p = yield Deterministic(
        name="d_i_c_p",
        value=(yield date()),
        shape_label=("intervention", "country", "change_point",),
    )
    log.debug(f"d_i_c_p\n{d_i_c_p}")

    # Superposition of change points
    gamma_i_c = gamma(
        d_i_c_p, l_i_sign
    )  # no yield because we do not sample anything in this function
    log.debug(f"gamma_i_c\n{gamma_i_c}")
    log.debug(f"alpha_i_c_a\n{alpha_i_c_a}")

    """ Calculate effectiveness and strength sum.
    """
    summation = tf.einsum("...ict,...ica->...cat", gamma_i_c, alpha_i_c_a)

    """ Calculate R_eff
    In the following we multiply the previously calculated sum with the basic
    reproduction number. Additionally we add some noise to add robustness
    """

    # Multiply R_0 and sum
    R_eff = tf.einsum("...ca,...cat->...tca", R_0, tf.exp(-summation))
    log.debug(f"R_eff\n{R_eff}")

    R_t = yield Deterministic(
        name=name, value=R_eff, shape_label=("time", "country", "age_group"),
    )
    sum_noise = yield construct_noise("noise_R", modelParams)

    # Add noise to R, this softplus has the same value and slope at 0 than exp but
    # grows more slowly.
    R_t = R_t * tf.nn.softplus(sum_noise / tf.math.log(2.0)) / tf.math.log(2.0)

    # Swap dimensions to follow specifications
    R_t = tf.einsum("...tca -> t...ca", R_t)

    return R_t  # shape time, batch, country, age_group


def construct_R_0(name, modelParams, loc, scale, hn_scale):
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
        modelParams: :py:class:`covid19_npis.ModelParams`
            Instance of modelParams, mainly used for number of age groups and
            number of countries.
        loc: number
            Location parameter of the R^*_0 Normal distribution.
        scale: number
            Scale parameter of the R^*_0 Normal distribution.
        hn_scale: number
            Scale parameter of the \sigma_{R^*, \text{country}} HaflNormal distribution.

        Returns
        -------
        :
            R_0 tensor |shape| batch, country, age_group
    """

    R_0 = (
        yield Normal(name="R_0", loc=0.0, scale=scale, conditionally_independent=True,)
    ) + loc
    log.debug(f"R_0:\n{R_0}")

    R_0_sigma_c = (
        yield HalfNormal(
            name="R_0_sigma_c",
            scale=1.0,
            conditionally_independent=True,
            transform=transformations.SoftPlus(),
        )
    ) * hn_scale

    delta_R_0_c = (
        yield Normal(
            name="delta_R_0_c",
            loc=0.0,
            scale=1.0,
            event_stack=(modelParams.num_countries),
            shape_label=("country"),
            conditionally_independent=True,
        )
    ) * R_0_sigma_c[..., tf.newaxis]
    log.debug(f"delta_R_0_c:\n{delta_R_0_c}")

    # Add to trace via deterministic
    R_0_c = R_0[..., tf.newaxis] + delta_R_0_c
    log.debug(f"R_0_c before softplus:\n{R_0_c}")

    # Softplus because we want to make sure that R_0 > 0.
    R_0_c = tf.math.softplus(R_0_c)
    R_0_c = yield Deterministic(name=name, value=R_0_c, shape_label=("country"),)
    log.debug(f"R_0_c:\n{R_0_c}")

    # for robustness
    tf.clip_by_value(R_0_c, 1, 5)

    return tf.repeat(
        R_0_c[..., tf.newaxis], repeats=modelParams.num_age_groups, axis=-1
    )


def construct_noise(name, modelParams, sigma=0.05, sigma_age=0.02):

    noise_R_sigma = (
        yield HalfNormal(
            name=f"{name}_sigma",
            scale=1.0,
            conditionally_independent=True,
            event_stack=(modelParams.num_countries,),
            shape_label=("country"),
            transform=transformations.SoftPlus(scale=100),
        )
    ) * sigma

    noise_R_sigma_age = (
        yield HalfNormal(
            name=f"{name}_sigma_age",
            scale=1.0,
            conditionally_independent=True,
            event_stack=(modelParams.num_countries, modelParams.num_age_groups),
            shape_label=("country", "age_group"),
            transform=transformations.SoftPlus(scale=100),
        )
    ) * sigma_age

    noise_R = (
        yield Normal(
            name=f"{name}",
            loc=0.0,
            scale=1.0,
            event_stack=(modelParams.length_sim, modelParams.num_countries,),
            shape_label=("time", "country"),
            conditionally_independent=True,
        )
    ) * noise_R_sigma[..., tf.newaxis, :]

    noise_R_age = (
        yield Normal(
            name=f"{name}_age",
            loc=0.0,
            scale=1.0,
            event_stack=(
                modelParams.length_sim,
                modelParams.num_countries,
                modelParams.num_age_groups,
            ),
            shape_label=("time", "country", "age_group"),
            conditionally_independent=True,
        )
    ) * noise_R_sigma_age[..., tf.newaxis, :, :]

    sum_noise_R = tf.math.cumsum(
        noise_R[..., tf.newaxis] + noise_R_age, exclusive=True, axis=-2
    )
    return sum_noise_R


def construct_R_0_old(name, modelParams, mean, beta):
    r"""
    Old constructor of :math:`R_0` using a gamma distribution:

    .. math::

        R_0 &\sim Gamma\left(\mu=2.5,\beta=2.0\right)

    Parameters
    ----------
    name: string
        Name of the distribution for trace and debugging.
    modelParams: :py:class:`covid19_npis.ModelParams`
        Instance of modelParams, mainly used for number of age groups and
        number of countries.
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
