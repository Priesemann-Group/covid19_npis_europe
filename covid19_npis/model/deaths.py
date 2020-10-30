import tensorflow as tf
import tensorflow_probability as tfp
import logging
import pymc4 as pm
import numpy as np

log = logging.getLogger(__name__)

from .distributions import (
    HalfCauchy,
    Normal,
    StudentT,
    HalfNormal,
    LKJCholesky,
    MvStudentT,
    Deterministic,
)
from .. import transformations
from .utils import gamma, get_filter_axis_data_from_dims, convolution_with_fixed_kernel


def _construct_reporting_delay(
    name,
    modelParams,
    theta_sigma_scale=0.3,
    theta_mu_loc=1.5,
    theta_mu_scale=0.3,
    m_sigma_scale=4.0,
    m_mu_loc=21.0,
    m_mu_scale=2.0,
):
    r"""
        .. math::

            m_{D_\text{death}, c} &= \ln \left(1 + e^{m^*_{D_\text{death}, c}} \right)\\
            m^*_{D_\text{death}, c} &\sim \mathcal{N} (\mu_{m_{D_\text{death}}}, \sigma_{m_{D_\text{death}}}), \\
            \mu_{m_{D_\text{death}}}&\sim \mathcal{N}(21, 2), \\
            \sigma_{m_{D_\text{test}}} &\sim HalfNormal(4), \label{eq:prior_delta_m_delay}\\
            \theta_{D_\text{death}, c} &=\frac{1}{4} \ln \left(1 + e^{4\theta^*_{D_\text{death}, c}} \right)\\
            \theta^*_{D_\text{death}, c} &\sim \mathcal{N}(\mu_{\theta_{D_\text{test}}},\sigma_{\theta_{D_\text{test}}}),\\
            \mu_{\theta_{D_\text{death}}} &\sim \mathcal{N}(1.5, 0.3),\\
            \sigma_{\theta_{D_\text{death}}} &\sim HalfNormal(0.3).

        Parameters
        ----------
        name: str
            Name of the reporting delay variable :math:`m_{D_\text{test},c,b}.`

        modelParams: :py:class:`covid19_npis.ModelParams`
            Instance of modelParams, mainly used for number of age groups and
            number of countries.

        theta_sigma_scale: optional
            Scale parameter for the Normal distribution :math:`\sigma_{\theta_{D_\text{death}}}.`
            |default| 0.3
        theta_mu_loc: optional
            Location parameter for the Normal distribution :math:`\mu_{\theta_{D_\text{death}}}.`
            |default| 1.5
        theta_mu_scale: optional
            Scale parameter for the HalfNormal distribution :math:`\mu_{\theta_{D_\text{death}}}.`
            |default| 0.3

        m_sigma_scale: optional
            Scale parameter for the HalfNormal distribution :math:`\sigma_{m_{D_\text{test}}}.`
            |default| 4.0
        m_mu_loc: optional
            Location parameter for the Normal distribution :math:`\mu_{m_{D_\text{death}}}.`
            |default| 21.0
        m_mu_scale: optional
            Scale parameter for the Normal distribution :math:`\mu_{m_{D_\text{death}}}.`
            |default| 2.0

        Returns
        -------
        :
            (m, theta)
            |shape| (batch, country) x 2
    """

    """ # Theta
    """
    theta_sigma = yield HalfNormal(
        name=f"{name}_theta_sigma",
        scale=theta_sigma_scale,
        conditionally_independent=True,
    )

    theta_mu = (
        yield Normal(
            name=f"{name}_theta_mu",
            loc=0.0,
            scale=theta_mu_scale,
            conditionally_independent=True,
        )
    ) + theta_mu_loc

    # theta_dagger = N(μ,θ)
    theta_dagger = (
        tf.einsum(
            "...c,...->...c",
            (
                yield Normal(
                    name=f"{name}_theta_dagger",
                    loc=0.0,
                    scale=1.0,
                    event_stack=modelParams.num_countries,
                    shape_label="country",
                    conditionally_independent=True,
                )
            ),
            theta_sigma,
        )
        + theta_mu[..., tf.newaxis]
    )

    theta = yield Deterministic(
        name=f"{name}_theta",
        value=0.25 * tf.math.softplus(4 * theta_dagger),
        shape_label=("country"),
    )

    """ # Mean m
    """
    m_sigma = yield HalfNormal(
        name=f"{name}_m_sigma",
        scale=m_sigma_scale,
        conditionally_independent=True,
    )
    mu_m = (
        yield Normal(
            name=f"{name}_mu_m",
            loc=0.0,
            scale=m_mu_scale,
            conditionally_independent=True,
        )
    ) + m_mu_loc
    # m_dagger = N(μ,θ)
    m_dagger = (
        tf.einsum(
            "...c,...->...c",
            (
                yield Normal(
                    name=f"{name}_m_dagger",
                    loc=0.0,
                    scale=1.0,
                    event_stack=modelParams.num_countries,
                    shape_label="country",
                    conditionally_independent=True,
                )
            ),
            m_sigma,
        )
        + mu_m[..., tf.newaxis]
    )

    m = yield Deterministic(
        name=f"{name}_m", value=tf.math.softplus(m_dagger), shape_label=("country")
    )

    return m, theta


def _calc_Phi_IFR(
    name,
    modelParams,
    alpha_loc=0.119,
    alpha_scale=0.003,
    beta_loc=-7.53,
    beta_scale=0.4,
):
    r"""
    Calculates and construct the IFR and Phi_IFR:

    .. math::

        \beta_{\text{IFR,c}} &= \mathcal{N}\left(-7.53, 0.4\right) \\
        \alpha_\text{IFR} &= \mathcal{N}\left(0.119, 0.003\right)


    .. math::

        \text{IFR}_c(a^*) &= \frac{1}{100} \exp{\left(\beta_{\text{IFR,c}} + \alpha_\text{IFR} \cdot a\right)} \\

    .. math::

        \phi_{\text{IFR}, c,a} = \frac{1}{\sum_{a^* = a_\text{beg}(a)}^{a_\text{end}(a)}  N_\text{pop}\left(a^*\right)}\sum_{a^* = a_\text{beg}(a)}^{a_\text{end}(a)} N_{\text{pop}, c}\left(a^*\right) \text{IFR}_c\left(a^* \right),

    Parameters
    ----------
    name: str
        Name of the infection fatatlity ratio variable :math:`\phi_{\text{IFR}, c,a}.`

    modelParams: :py:class:`covid19_npis.ModelParams`
        Instance of modelParams, mainly used for number of age groups and
        number of countries.

    alpha_loc: optional
        |default| 0.119

    alpha_scale: optional
        |default| 0.003

    beta_loc: optional
        |default| -7.53

    beta_scale: optional
        |default| 0.4

    Returns
    -------
    :
        Phi_IFR
        |shape| batch, country, age brackets
    """

    alpha = yield Normal(
        name=f"{name}_alpha",
        loc=alpha_loc,
        scale=alpha_scale,
        conditionally_independent=True,
    )

    beta = yield Normal(
        name=f"{name}_beta",
        loc=beta_loc,
        scale=beta_scale,
        conditionally_independent=True,
        event_stack=modelParams.num_countries,
        shape_label="country",
    )

    ages = tf.range(0.0, 101.0, delta=1.0, dtype="float32")  # [0...100]
    log.debug(f"ages\n{ages.shape}")
    log.debug(f"beta\n{beta.shape}")
    log.debug(f"alpha\n{alpha[..., tf.newaxis].shape}")

    IFR = 0.01 * tf.exp(
        beta[..., tf.newaxis] + tf.einsum("...,a->...a", alpha[..., tf.newaxis], ages)
    )  # |shape| batch,coutry,ages
    log.debug(f"IFR\n{IFR.shape}")

    N_total = modelParams.N_data_tensor_total  # |shape| coutry,ages
    N_agegroups = modelParams.N_data_tensor
    log.debug(f"N_total\n{N_total}")
    log.debug(f"N_agegroups\n{N_agegroups}")

    # Multiply N_pop(a) * IFR(a) for every age group and country
    product = tf.einsum("...ca,ca->...ca", IFR, N_total)
    log.debug(f"product\n{product}")
    # for each  country and age group:
    phi = []
    for c, country in enumerate(modelParams.countries):
        phi_c = []
        for age_group in modelParams.age_groups:
            # Get lower/upper bound for age groups in selected country
            lower, upper = country.age_groups[age_group]  # inclusive

            phi_a = tf.math.reduce_sum(product[..., c, lower : upper + 1], axis=-1)
            log.debug(f"phi_a\n{phi_a.shape}")
            phi_c.append(phi_a)
        phi.append(phi_c)
    log.debug(f"phi\n{tf.convert_to_tensor(phi).shape}")

    phi = tf.einsum("ca...->...ca", tf.convert_to_tensor(phi))  # Transpose

    Phi_IFR = tf.einsum("ca,...ca->...ca", 1.0 / N_agegroups, phi)
    log.debug(f"Phi_IFR\n{Phi_IFR.shape}")

    return Phi_IFR


def calc_delayed_deaths(name, new_cases, Phi_IFR, m, theta, length_kernel=14):
    r"""
    Calculates delayed deahs from IFR and delay kernel.

    .. math::

        \Tilde{E}_{\text{delayDeath}, c, a}(t) = \phi_{\text{IFR}, c,a} \sum_{\tau=0}^T \Tilde{E}_{c,a}(t-\tau) \cdot f_{c,t}(\tau) \\
        f_{c,t}(\tau) = Gamma(\tau ; \alpha = \frac{m_{D_{\text{death}, c}}}{\theta_{D_\text{death}}}+ 1, \beta=\frac{1}{\theta_{D_\text{death}}})

    Parameters
    ----------
    name: str
        Name of the delayed deaths variable :math:`\Tilde{E}_{\text{delayDeath}, c, a}(t).`

    new_cases: tf.Tensor
        New cases without reporting delay :math:`\Tilde{E}_{c,a}(t).`
        |shape| batch, time, country, age_group

    Phi_IFR: tf.Tensor
        Infection fatality ratio of the age brackets :math:`\phi_{\text{IFR}, c,a}.`
        |shape| batch, country, age_group

    m: tf.Tensor
        Median fatality delay for the delay kernel :math:`m_{D_{\text{death}, c}}.`
        |shape| batch, country

    theta: tf.Tensor
        Scale fatality delay for the delay kernel :math:`\theta_{D_\text{death}}.`
        |shape| batch

    length_kernel : optional
        Length of the kernel in days
        |default| 14 days

    Returns
    -------
    :
        :math;`\Tilde{E}_{\text{delayDeath}, c, a}(t)`
        |shape| batch, time, country, age_group
    """

    """ # Construct delay kernel f
    """
    # Time tensor
    tau = tf.range(
        0.01, length_kernel + 0.01, 1.0, dtype="float32"
    )  # The gamma function does not like 0!

    # Get shapes right we want c,t
    m = m[..., tf.newaxis, tf.newaxis]  # |shape| batch, country, age, time
    theta = theta[..., tf.newaxis, tf.newaxis]  # |shape| batch, country, age, time
    # Calculate pdf
    kernel = gamma(
        tau,
        m / theta + 1.0,
        1.0 / theta,
    )  # add age group dimension
    log.debug(f"kernel deaths\n{kernel}")
    log.debug(f"new_cases deaths\n{new_cases}")

    """ # Calc delayed deaths
    """
    if len(new_cases.shape) == 5:
        filter_axes_data = [-5, -4, -2, -1]
    elif len(new_cases.shape) == 4:
        filter_axes_data = [-4, -2, -1]
    else:
        filter_axes_data = [-2, -1]

    dd = convolution_with_fixed_kernel(
        data=new_cases,
        kernel=kernel,
        data_time_axis=-3,
        filter_axes_data=filter_axes_data,
    )
    log.debug(f"dd\n{dd.shape}")
    delayed_deaths = yield Deterministic(
        name=name,
        value=tf.einsum("...ca,...tca->...tca", Phi_IFR, dd),
        shape_label=("time", "country", "age_group"),
    )

    return delayed_deaths
