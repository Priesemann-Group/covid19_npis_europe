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
from .utils import gamma


def _construct_reporting_delay(
    name, modelParams,
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

        sigma_theta_scale: optional
            Scale parameter for the Normal distribution :math:`\sigma_{\theta_{D_\text{death}}}.`
            |default| 0.3
        mu_theta_loc: optional
            Location parameter for the Normal distribution :math:`\mu_{\theta_{D_\text{death}}}.`
            |default| 1.5
        mu_theta_scale: optional
            Scale parameter for the HalfNormal distribution :math:`\mu_{\theta_{D_\text{death}}}.`
            |default| 0.3

        sigma_m_scale: optional
            Scale parameter for the HalfNormal distribution :math:`\sigma_{m_{D_\text{test}}}.`
            |default| 4.0
        mu_m_loc: optional
            Location parameter for the Normal distribution :math:`\mu_{m_{D_\text{death}}}.`
            |default| 21.0
        mu_m_scale: optional
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
    sigma_theta = yield HalfNormal(
        name=f"sigma_theta_{name}",
        scale=sigma_theta_scale,
        conditionally_independent=True,
    )

    mu_theta = (
        yield Normal(
            name=f"mu_theta_{name}",
            loc=0.0,
            scale=mu_theta_scale,
            conditionally_independent=True,
        )
    ) + mu_theta_loc

    # theta_dagger = N(μ,θ)
    theta_dagger = (
        tf.einsum(
            "...c,...->...c",
            (
                yield Normal(
                    name=f"theta_dagger_{name}",
                    loc=0.0,
                    scale=1.0,
                    event_stack=modelParams.num_countries,
                    shape_label="country",
                    conditionally_independent=True,
                )
            ),
            sigma_theta,
        )
        + mu_theta[..., tf.newaxis]
    )

    theta = yield Deterministic(
        name=f"theta_{name}",
        value=0.25 * tf.math.softplus(4 * theta_dagger),
        shape_label=("country"),
    )

    """ # Mean m
    """
    sigma_m = yield HalfNormal(
        name=f"sigma_m_{name}", scale=sigma_m_scale, conditionally_independent=True,
    )
    mu_m = (
        yield Normal(
            name=f"mu_m_{name}",
            loc=0.0,
            scale=mu_m_scale,
            conditionally_independent=True,
        )
    ) + mu_m_loc
    # m_dagger = N(μ,θ)
    m_dagger = (
        tf.einsum(
            "...c,...->...c",
            (
                yield Normal(
                    name=f"m_dagger_{name}",
                    loc=0.0,
                    scale=1.0,
                    event_stack=modelParams.num_countries,
                    shape_label="country",
                    conditionally_independent=True,
                )
            ),
            sigma_m,
        )
        + mu_m[..., tf.newaxis]
    )

    m = yield Deterministic(
        name=f"m_{name}", value=tf.math.softplus(m_dagger), shape_label=("country")
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


    """

    alpha = yield Normal(
        name=f"alpha_{name}",
        loc=alpha_loc,
        scale=alpha_scale,
        conditionally_independent=True,
    )

    beta = yield Normal(
        name=f"beta_{name}",
        loc=beta_loc,
        scale=beta_scale,
        conditionally_independent=True,
        event_stack=modelParams.num_countries,
        shape_label="country",
    )

    ages = tf.range(0.0, 101.0, delta=1.0, dtype="float32")  # [0...100]
    log.info(f"ages\n{ages.shape}")
    log.info(f"beta\n{beta.shape}")
    log.info(f"alpha\n{alpha[..., tf.newaxis].shape}")

    IFR = 0.01 * tf.exp(
        beta[..., tf.newaxis] + tf.einsum("...,a->...a", alpha[..., tf.newaxis], ages)
    )
    log.info(f"IFR\n{IFR.shape}")

    N_total = modelParams.N_data_tensor_total
    N_agegroups = modelParams.N_data_tensor

    # for each age group: N_pop(a) * IFR(a)
    sum_ = []
    for c, country in enumerate(modelParams.countries):
        sum_c = []
        for age_group in modelParams.age_groups:
            lower, upper = country.age_groups[
                age_group
            ]  # Get lower/upper bound for age groups
            sum_a = tf.einsum(
                "...a,a->...",
                IFR[..., c, lower : upper + 1],
                N_total[c, lower : upper + 1],
            )
            sum_c.append(sum_a)
        sum_.append(sum_c)

    sum_ = tf.einsum("ca...->...ca", tf.convert_to_tensor(sum_))  # Transpose

    log.info(f"SUM\n{sum_.shape}")

    Phi_IFR = tf.einsum("ca,...ca->...ca", 1.0 / N_agegroups, sum_)
    log.info(f"Phi_IFR\n{Phi_IFR.shape}")

    return Phi_IFR
