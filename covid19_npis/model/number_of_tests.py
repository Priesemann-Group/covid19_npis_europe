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
    MvNormalCholesky,
    Deterministic,
    VonMises,
)
from .. import transformations
from . import utils


def weekly_modulation(name, modelParams, cases):
    r"""
    Adds a weekly modulation of the number of new cases:

    .. math::

        \text{cases\_modulated} &= \text{cases} \cdot (1-f(t))\,, \qquad\text{with}\\
        f(t) &= (1-w) \cdot \left(1 - \left|\sin\left(\frac{\pi}{7} t- \frac{1}{2}\Phi_w\right)\right| \right)


    The modulation is assumed to be the same for all age-groups within one country and determined by
    the "weight" and "offset" parameters. The weight follows a sigmoidal distribution with normal prior
    of "weight_cross". The "offset" follows a VonMises distribution centered
    around 0 (Mondays) and a wide SD (concentration parameter = 2).
    
    Parameters
    ----------
    name : str or None,
        The name of the cases to be modulated (gets added to trace).
    modelParams: :py:class:`covid19_npis.ModelParams`
        Instance of modelParams, mainly used for number of age groups and
        number of countries.
    cases : tf.tensor
        The input array of daily new cases for countries and age groups

    Returns
    -------
    cases_modulated : tf.tensor

    TODO
    ----
        - check prior parameters
        - different modulations across: age, country?
        - check: are (cumulative) case numbers same as in unmodulated case? need some kind of normalization?
        - store and plot parameters at end
    """

    log.debug("Week modulation")

    # offset-distribution of weekly modulation minimum
    offset = yield VonMises(
        name=name + "_modulation_offset",
        loc=0,
        concentration=2,
        conditionally_independent=True,
        event_stack=(1, modelParams.num_countries, 1),
        shape_label=(None, "country", None),
    )

    # amplitude of weekly modulation
    weight_cross = yield Normal(
        name=name + "_modulation_weight",
        loc=0,
        scale=2,
        conditionally_independent=True,
        event_stack=(1, modelParams.num_countries, 1),
        shape_label=(None, "country", None),
        # transform=transformations.SoftPlus(scale=0.5),
    )
    weight = tf.math.sigmoid(weight_cross)

    t = modelParams.get_weekdays()  # get array with weekdays
    f = (1 - weight) * (
        1
        - tf.math.abs(
            tf.math.sin(
                tf.reshape(t, (1, -1, 1, 1)) / 7 * tf.constant(np.pi) + offset / 2
            )
        )
    )
    # modulation factor
    cases_modulated = cases * (1 - f)  # total modulation

    yield Deterministic(
        name=name + "_modulated",
        value=cases_modulated,
        shape_label=("time", "country", "age_group"),
    )
    return cases_modulated


def generate_testing(name_total, name_positive, modelParams, new_E_t):
    r"""
    High level function for generating/simulating testing behaviour.

    Constructs B splines
    Delay cases

    Parameters
    ----------
    name_total: str,
        Name for the total tests performed

    name_positive: str,
        Name for the positive tests performed

    modelParams: :py:class:`covid19_npis.ModelParams`
        Instance of modelParams, mainly used for number of age groups and
        number of countries.

    new_E_t: tf.Tensor
        New cases :math:`E_{\text{age}, a}.`
        |shape| batch, time, country, age_group

    Returns
    -------
    :
        (:math:`n_{\Sigma, c,a}(t)`, :math:`n_{{+}, {c,a}}(t)`
        Total and positive tests by age group and country
        |shape| (batch, time, country, age_group) x 2

    ToDo
    -----
    - Add more documenation for this function
    """

    # Get basic functions for b-splines (used later)
    B = construct_Bsplines_basis(modelParams)

    """ Construct correlated fraction of positive tests (phi),
    traced persons per case (eta), base rate of testing (xi)
    and testing delay (m star)
    """
    (phi, eta, xi, m_ast) = yield construct_testing_state(
        name_phi="phi_plus",
        name_eta="eta",
        name_xi="xi",
        name_m_ast="m_ast",
        modelParams=modelParams,
        num_knots=B.shape[-1],
    )

    # Transfrom m_ast with reporting delay
    m, theta = yield _construct_reporting_delay(
        name="testing_delay", modelParams=modelParams, m_ast=m_ast
    )

    # Calculate time dependent variables via bsplines
    phi_t = _calculate_Bsplines(phi, B)
    eta_t = _calculate_Bsplines(eta, B)
    xi_t = _calculate_Bsplines(xi, B)
    m_t = _calculate_Bsplines(m, B)

    log.debug(f"phi_t {phi_t}")
    log.debug(f"eta_t {eta_t}")
    log.debug(f"xi_t {xi_t}")
    log.debug(f"m_t {m_t}")

    # Construct gamma kernel from delay parameter m and add to trace
    delay_kernel = yield _calc_reporting_delay_kernel(
        name="reporting_delay_kernel", m=m_t, theta=theta
    )

    filter_axes_data = utils.get_filter_axis_data_from_dims(len(new_E_t.shape))
    # Convolution with gamma kernel
    new_E_t_delayed = utils.convolution_with_varying_kernel(
        data=new_E_t,
        kernel=delay_kernel,
        data_time_axis=-3,
        filter_axes_data=filter_axes_data,
    )
    new_E_t_delayed = yield Deterministic(
        name=f"new_E_t_delayed",
        value=new_E_t_delayed,
        shape_label=("time", "country", "age_group"),
    )
    log.debug(f"new_E_t_delayed\n{new_E_t_delayed}")

    """ # Postive tests
    """
    phi_age = yield _construct_phi_age(name="phi_age", modelParams=modelParams)
    phi_age = tf.debugging.check_numerics(phi_age, f"phi_age:\n{phi_age}")

    positive_tests = _calc_positive_tests(
        new_E_t_delayed=new_E_t_delayed,
        phi_plus=phi_t,
        phi_age=phi_age,
    )

    positive_tests = yield weekly_modulation(
        name=name_positive,
        modelParams=modelParams,
        cases=positive_tests,
    )

    log.debug(f"positive_tests\n{positive_tests}")
    positive_tests = yield Deterministic(
        name=name_positive,
        value=positive_tests,
        shape_label=("time", "country", "age_group"),
    )

    """ # Total tests
    """
    phi_tests_reported = yield _construct_phi_tests_reported(
        name="phi_tests_reported", modelParams=modelParams
    )
    total_tests = _calc_total_number_of_tests_performed(
        new_E_t_delayed=new_E_t_delayed,
        phi_tests_reported=phi_tests_reported,
        phi_plus=phi_t,
        eta=eta_t,
        xi=xi_t,
    )
    total_tests = yield Deterministic(
        name=name_total, value=total_tests, shape_label=("time", "country", "age_group")
    )
    total_tests_compact = yield Deterministic(
        name=f"{name_total}_compact",
        value=tf.reduce_sum(total_tests, axis=-1),
        shape_label=("time", "country"),
    )
    log.debug(f"total_tests\n{total_tests}")
    return (total_tests, positive_tests)


def _calc_positive_tests(new_E_t_delayed, phi_plus, phi_age):
    r"""
    .. math::

        n_{{+}, {c,a}}(t) =\Tilde{E}_{\text{delayTest}, {c,a}}(t) \cdot \phi_{+, c}(t) \cdot \phi_{\text{age}, a},


    Parameters
    -----------

    name: str
        Name of the variable for the new positive cases :math:`n_{{+}, {c,a}}(t)`
        in the trace.

    new_E_t_delayed: tf.Tensor
        New cases with reporting delay :math:`\Tilde{E}_{\text{delayTest}, c,a}(t).`
        |shape| batch, time, country, age_group

    phi_plus: tf.Tensor
        Fraction of positive tests :math:`\phi_{+, c}(t).`
        |shape| batch, time, country

    phi_age: tf.Tensor
        Fraction of positive tests :math:`\phi_{\text{age}, a}.`
        |shape| batch, age_group

    Returns
    -------
    :
        :math:`n_{{+}, {c,a}}(t)`
        |shape| batch, time, country, age_group
    """

    n_plus = tf.einsum("...tca,...tc,...a->...tca", new_E_t_delayed, phi_plus, phi_age)
    return n_plus


def _calc_total_number_of_tests_performed(
    new_E_t_delayed, phi_tests_reported, phi_plus, eta, xi
):
    r"""
        .. math::

            n_{\Sigma, c,a}(t) &= \phi_{\text{tests reported}, c}\nonumber \\
            \cdot   (\, & \Tilde{E}_{\text{delayTest}, c,a}(t) \cdot  \phi_{+, c}(t) \nonumber \\
            +  \, &\Tilde{E}_{\text{delayTest}, c,a}(t) \cdot  \phi_{+, c}(t) \cdot \eta_{\text{traced}, c}(t)\nonumber \\
            +\,  &\xi_c(t))

        Parameters
        ----------
        name: str
            Name of the variable for the total number of tests performed :math:`n_{\Sigma, c,a}(t)`
            in the trace.

        new_E_t_delayed: tf.Tensor
            New cases with reporting delay :math:`\Tilde{E}_{\text{delayTest}, c,a}(t).`
            |shape| batch, time, country, age_group

        phi_tests_reported: tf.Tensor
            Difference in fraction for different countries :math:`\phi_{\text{tests reported}, c}`
            |shape| batch, country

        phi_plus: tf.Tensor
            Fraction of positive tests :math:`\phi_{+, c}(t).`
            |shape| batch, time, country

        eta: tf.Tensor
            Number of traced persons per case with subsequent negative test per case
            :math:`\eta_{\text{traced}, c}(t).`
            |shape| batch, time, country

        xi: tf.Tensor
            Base rate of testing per day that leads to negative tests
            :math:`\xi_c(t).`
            |shape| batch, time, country

        Returns
        -------
        :
            :math:`n_{\Sigma, c,a}(t)`
            |shape| batch, time, country, age_group
    """

    inner = (
        tf.einsum("...tca,...tc->...tca", new_E_t_delayed, phi_plus)
        + tf.einsum("...tca,...tc,...tc->...tca", new_E_t_delayed, phi_plus, eta)
        + xi[..., tf.newaxis]
    )
    n_Sigma = tf.einsum("...c,...tca->...tca", phi_tests_reported, inner)

    return n_Sigma


def _construct_phi_tests_reported(
    name, modelParams, mu_loc=1.0, mu_scale=1.0, sigma_scale=1.0
):
    r"""
        Construct the different of the fraction of tests for each country in the following
        hierarchical manner:

        .. math::

            \phi_{\text{tests reported}, c} &= \frac{e^{\phi^\dagger_{\text{tests reported}, c}}}{e^{\phi^\dagger_{\text{tests reported}, c}} + 1},\label{tests_reported}\\
            \phi^\dagger_{\text{tests reported}, c} &\sim \mathcal{N}(\mu_{\phi^\dagger_{\text{tests reported}}}, \sigma_{\phi^\dagger_{\text{tests reported}}}),\\
            \mu_{\phi^\dagger_{\text{tests reported}}} &\sim \mathcal{N}(1,1),\\
            \sigma_{\phi^\dagger_{\text{tests reported}}} &\sim HalfCauchy(1).

        Parameters
        ----------
        name: str
            Name of the variable :math:`\phi_{\text{tests reported}, c}`. Will also
            appear in the trace with this name.

        modelParams: :py:class:`covid19_npis.ModelParams`
            Instance of modelParams, mainly used for number of age groups and
            number of countries.

        mu_loc: optional
            Location parameter for the Normal distribution :math:`\mu_{\phi^\dagger_{\text{tests reported}}}`.
            |default| 1.0

        mu_scale: optional
            Scale parameter for the Normal distribution :math:`\mu_{\phi^\dagger_{\text{tests reported}}}`.
            |default| 1.0

        sigma_scale: optional
            Scale parameter for the :math:`\sigma_{\phi^\dagger_{\text{tests reported}}}` HalfCauchy
            distribution.
            |default| 1.0


        Returns
        -------
        :
            :math:`\phi_{\text{tests reported}, c}`
            |shape| batch, country
    """

    sigma = yield HalfCauchy(
        name=f"{name}_sigma",
        scale=sigma_scale,
        conditionally_independent=True,
    )
    mu = yield Normal(
        name=f"{name}_mu",
        loc=mu_loc,
        scale=mu_scale,
        conditionally_independent=True,
    )
    phi_dagger = yield Normal(
        name=f"{name}_dagger",
        loc=mu,
        scale=sigma,
        event_stack=modelParams.num_countries,
        shape_label="country",
    )

    phi = yield Deterministic(
        name=name, value=tf.math.sigmoid(phi_dagger), shape_label="country"
    )

    return phi


def _construct_phi_age(name, modelParams, sigma_scale=0.2):
    r"""
        Fraction of positive tests :math:`\phi_{\text{age}, a}.`

        .. math::

            \phi_{\text{age}, a} &= e^{\phi^\dagger_{\text{age},a}} \label{eq:phi_age}\\
            \phi^\dagger_{\text{age},a} &= \mathcal{N}\left(0, \sigma_{\phi^\dagger_{\text{age},a}}\right)\\
            \sigma_{\phi^\dagger_{\text{age},a}}&=HalfNormal\left(0.2\right)

        Parameters
        ----------
        name: str
            Name of the variable :math:`\phi_{\text{age}, a}`. Will also
            appear in the trace with this name.

        modelParams: :py:class:`covid19_npis.ModelParams`
            Instance of modelParams, mainly used for number of age groups and
            number of countries.

        sigma_scale:
            Scale parameter for the HalfNormal distribution :math:`\sigma_{\phi^\dagger_{\text{age},a}}.`
            |default| 0.2

        Returns
        -------
        :
            :math:`\phi_{\text{age}, a}`
            |shape| batch, age_group
    """

    sigma = yield HalfNormal(
        name=f"{name}_sigma",
        scale=sigma_scale,
        event_stack=modelParams.num_age_groups,
        conditionally_independent=True,
        shape_label="age_group",
        transform=transformations.SoftPlus(scale=sigma_scale),
    )
    log.debug(f"sigma_phi_age {sigma}")

    phi_cross = yield Normal(
        name=f"{name}_cross",
        loc=0.0,
        scale=1.0,
        shape_label="age_group",
        event_stack=modelParams.num_age_groups,
        conditionally_independent=True,
    )
    phi_cross = tf.einsum("...a,...a->...a", phi_cross, sigma)
    log.debug(f"phi_age_cross{phi_cross}")

    # Transform
    phi = yield Deterministic(
        name=name, value=tf.math.softplus(phi_cross), shape_label="age_group"
    )
    log.debug(f"phi_age{phi}")

    return phi


def _construct_reporting_delay(
    name,
    modelParams,
    m_ast,
    mu_loc=1.5,
    mu_scale=0.4,
    theta_sigma_scale=0.2,
    m_sigma_scale=3.0,
):
    r"""
        .. math::

            m_{D_\text{test},c,b} &= m^\ast_{D_\text{test}, c,b} + \Delta m_{D_\text{test}, c}\\

        .. math::

            \Delta m_{D_\text{test}, c} &\sim \mathcal{N} (0, \sigma_{m_{D\, \text{test}}}), \\
            \sigma_{m_{D\, \text{test}}} &\sim HalfNormal(3), \label{eq:prior_delta_m_delay}\\
            \theta_{D_\text{test}, c} &\sim \mathcal{N}(\mu_{\theta_{D_\text{test}}},\sigma_{\theta_{D_\text{test}}}),\\
            \mu_{\theta_{D_\text{test}}} &\sim \mathcal{N}(1.5, 0.4),\\
            \sigma_{\theta_{D_\text{test}}} &\sim HalfNormal(0.2).

        Parameters
        ----------
        name: str
            Name of the reporting delay variable :math:`m_{D_\text{test},c,b}.`

        modelParams: :py:class:`covid19_npis.ModelParams`
            Instance of modelParams, mainly used for number of age groups and
            number of countries.

        m_ast: tf.Tensor
            :math:`m^\ast_{D_\text{test}, c,b}`
            |shape| batch, country, spline

        mu_loc: optional
            Location parameter for the Normal distribution :math:`\mu_{\theta_{D_\text{test}}}.`
            |default| 1.5

        mu_scale: optional
            Scale parameter for the Normal distribution :math:`\mu_{\theta_{D_\text{test}}}.`
            |default| 0.4

        theta_sigma_scale: optional
            Scale parameter for the HalfNorml distribution :math:`\sigma_{\theta_{D_\text{test}}}.`
            |default| 0.2

        m_sigma_scale: optional
            Scale parameter for the HalfNorml distribution :math:`\sigma_{m_{D\, \text{test}}}.`
            |default| 3.0

        Returns
        -------
        :
            :math:`m_{D_\text{test},c,b}`
            |shape| batch, country, spline
    """

    # Theta
    theta_sigma = yield HalfNormal(
        name=f"{name}_theta_sigma",
        scale=theta_sigma_scale,
        conditionally_independent=True,
    )
    mu = yield Normal(
        name=f"{name}_mu",
        loc=0.0,
        scale=mu_scale,
        conditionally_independent=True,
    )
    mu = mu + mu_loc
    log.debug(f"mu delta m:\n{mu}")
    log.debug(f"theta_sigma\n{theta_sigma}")
    theta = (
        tf.einsum(
            "...c,...->...c",
            (
                yield Normal(
                    name=f"{name}_theta",
                    loc=0.0,
                    scale=1.0,
                    event_stack=modelParams.num_countries,
                    shape_label="country",
                    conditionally_independent=True,
                )
            ),
            theta_sigma,
        )
        + mu[..., tf.newaxis]
    )

    # m
    m_sigma = yield HalfNormal(
        name=f"{name}_m_sigma", scale=m_sigma_scale, conditionally_independent=True
    )
    delta_m = tf.einsum(
        "...c,...->...c",
        (
            yield Normal(
                name=f"delta_{name}",
                loc=0.0,
                scale=1.0,
                event_stack=modelParams.num_countries,
                shape_label="country",
                conditionally_independent=True,
            )
        ),
        m_sigma,
    )

    m = tf.math.softplus(m_ast + delta_m[..., tf.newaxis])
    theta = 0.5 * tf.math.softplus(2 * theta)
    theta = tf.clip_by_value(theta, clip_value_min=0.1, clip_value_max=10)

    # We need to add the spline dimension at some point i.e. prop. expand delta_m
    m = yield Deterministic(
        name=name,
        value=m,
        shape_label=("country", "spline"),
    )
    log.debug(f"m_spline:\n{m}")
    return (m, theta)


def _calc_reporting_delay_kernel(name, m, theta, length_kernel=14):
    r"""
    Calculates the pdf for the gamma reporting kernel.

    .. math::

        f_{c,t}(\tau) =  Gamma(\tau ; \alpha = \frac{m_{D_{\text{test}},c}(t)}{\theta_{D_\text{test}},c}
        + 1, \beta=\frac{1}{\theta_{D_\text{test},c}}),\nonumber\\
        \text{with $f_{c,t}$ normalized such that} \:\: \sum_{\tau=0}^T f_{c,t}(\tau) = 1.

    Parameters
    ----------
    name:
        Name of the reporting delay kernel :math:`f_{c,t}(\tau)`
    m:
        |shape| batch, time, country
    theta:
        |shape| batch, country
    length_kernel: optional
        Length of the kernel in days
        |default| 14 days

    Returns
    -------
    :
        |shape| batch,country, kernel, time
    """

    # Time tensor
    t = tf.range(
        0.01, length_kernel + 0.01, 1.0, dtype="float32"
    )  # The gamma function does not like 0!

    # Get shapes right we want c,t
    m = tf.einsum("...tc->...ct", m)[
        ..., tf.newaxis
    ]  # Add empty kernel axis -> batch country time kernel
    theta = theta[..., tf.newaxis, tf.newaxis]  # Add a empty time axis, and kernel axis
    log.debug(f"m\n{m}")
    log.debug(f"theta\n{theta}")

    # Calculate pdf
    kernel = utils.gamma(
        t,
        m / theta + 1.0,
        1.0 / theta,
    )
    kernel = tf.einsum("...ctk->...ckt", kernel)

    kernel = yield Deterministic(
        name=name, value=kernel, shape_label=("country", "kernel", "time")
    )
    log.debug(f"reportin delay kernel\n{kernel}")  # batch, country, kernel, time

    return kernel


def construct_testing_state(
    name_phi,
    name_eta,
    name_xi,
    name_m_ast,
    modelParams,
    num_knots,
    mu_cross_loc=0.0,
    mu_cross_scale=10.0,
    m_mu_loc=12.0,
    m_mu_scale=2.0,
    sigma_cross_scale=10.0,
    m_sigma_scale=1.0,
):
    r"""
        .. math::

            (\phi^\dagger_{\text{tested},c,b},
            \: \eta^\dagger_{\text{traced},c,b},
            \: \xi^\dagger_{c,b},
            \: m^\ast_{D_\text{test},c,b})
            &\sim StudentT_{\nu=4} \left(\boldsymbol{\mu}, \mathbf{\Sigma}\right)

        where

        .. math::

            \boldsymbol{\mu} &= \left(\mu_{\phi^\dagger_+},
            \mu_{\eta^\dagger_\text{traced}},
            \mu_{\xi^\dagger},
            \mu_{m_{D_\text{test}}} \right) \\

            \mathbf{\Sigma} &\sim LKJ(\eta=2,
            \boldsymbol{\sigma} = \left(\sigma_\phi, \sigma_\eta,\sigma_\xi,\sigma_m)\right)


        with the distributions parametarized in the following hierarchical manner:

        .. math::

            \mu_{\phi^\dagger_+},\: \mu_{\eta^\dagger_\text{traced}},\: \mu_{\xi^\dagger} &\sim \mathcal {N}(0, 10),\\
            \mu_{m_{D_\text{test}}} &\sim \mathcal {N}(12, 2), \\
            \sigma_\phi, \sigma_\eta, \sigma_\xi &\sim HalfCauchy(10), \\
            \sigma_m &\sim HalfNormal(1) \label{eq:prior_sigma_delay_time}\\

        at last we transform the variables :math:`\phi_{+,c,b},\: \eta_{\text{traced},c,b},\: \xi_{c,b}`

        .. math::

            \phi_{+,c,b} &= \frac{e^{\phi^\dagger_{+,c,b}}}{e^{\phi^\dagger_{+,c,b}} + 1},\\
            \eta_{\text{traced},c,b} &= \ln \left(1 + e^{ \eta^\dagger_{\text{traced},c,b}} \right),\\
            \xi_{c,b} &= \ln \left(1 + e^{\xi_{c,b}^\dagger}\right)\frac{n_\text{inhabitants}}{10 000}\\


        Parameters
        ----------
        name_phi: str
            Name of the fraction of positive tests variable :math:`\phi_{+,c,b}.`

        name_eta: str
            Name of the number of traced persons variable :math:`\eta_{\text{traced},c,b}.`

        name_xi: str
            Name of the base tests rate variable :math:`\xi_{c,b}.`

        name_m_ast: str
            Name of the testing delay variable :math:`m^*_{D_{test},c,b}.`

        modelParams: :py:class:`covid19_npis.ModelParams`
            Instance of modelParams, mainly used for number of age groups and
            number of countries.

        num_knots:
            Number of knots for the Bspline dimension.

        mu_cross_loc: optional
            Location parameter for the three Normal distributions :math:`\mu_{\phi^\dagger_+},\: \mu_{\eta^\dagger_\text{traced}},\: \mu_{\xi^\dagger}.`
            |default| 0.0

        mu_cross_scale: optional
            Scale parameter for the three Normal distributions :math:`\mu_{\phi^\dagger_+},\: \mu_{\eta^\dagger_\text{traced}},\: \mu_{\xi^\dagger}.`
            |default| 10.0

        m_mu_loc: optional
            Location parameter for the Normal distribution :math:`\mu_{m_{D_\text{test}}}.`
            |default| 12.0

        m_mu_scale: optional
            Scale parameter for the Normal distribution :math:`\mu_{m_{D_\text{test}}}.`
            |default| 2.0

        sigma_cross_scale: optional
            Scale parameter for the three HalfCauchy distributions :math:`\sigma_\phi, \sigma_\eta, \sigma_\xi.`
            |default| 10.0

        m_sigma_scale: optional
            Scale parameter for the HalfNormal distribution :math:`\sigma_m.`
            |default| 1.0

        Returns
        -------
        :
            Testing state tuple :math:`(\phi_{+,c,b},
            \: \eta_{\text{traced},c,b},\: \xi_{c,b},\: m_{D_\text{test},c,b}),\: \theta_{D_\text{test}}.`
            |shape| 4 x (batch, country, spline),
    """

    """ First construct all hierachical variables: m,phi,xi,eta
    """
    # m
    m_sigma = yield HalfNormal(
        name=f"{name_m_ast}_sigma",
        scale=m_sigma_scale,
        conditionally_independent=True,
        transform=transformations.SoftPlus(scale=m_sigma_scale),
    )
    m_mu = yield Normal(
        name=f"{name_m_ast}_mu",
        loc=m_mu_loc,
        scale=m_mu_scale,
        conditionally_independent=True,
    )
    log.debug(f"m_sigma{m_sigma}")
    log.debug(f"m_mu{m_mu}")

    # Fraction of positive tests phi
    phi_sigma = yield HalfCauchy(
        name=f"{name_phi}_sigma",
        scale=sigma_cross_scale,
        conditionally_independent=True,
        transform=transformations.SoftPlus(scale=sigma_cross_scale),
    )
    phi_mu_cross = yield Normal(
        name=f"{name_phi}_mu_cross",
        loc=mu_cross_loc,
        scale=mu_cross_scale,
        conditionally_independent=True,
    )
    log.debug(f"phi_sigma{phi_sigma}")
    log.debug(f"phi_mu_cross{phi_mu_cross}")

    # Eta
    eta_sigma = yield HalfCauchy(
        name=f"{name_eta}_sigma",
        scale=sigma_cross_scale,
        conditionally_independent=True,
        transform=transformations.SoftPlus(scale=sigma_cross_scale),
    )
    eta_mu_cross = yield Normal(
        name=f"{name_eta}_mu_cross",
        loc=mu_cross_loc,
        scale=mu_cross_scale,
        conditionally_independent=True,
    )
    log.debug(f"eta_sigma{eta_sigma}")
    log.debug(f"eta_mu_cross{eta_mu_cross}")

    # Xi
    xi_sigma = yield HalfCauchy(
        name=f"{name_xi}_sigma",
        scale=10.0,
        conditionally_independent=True,
        transform=transformations.SoftPlus(scale=10.0),
    )
    xi_mu_cross = yield Normal(
        name=f"{name_xi}_mu_cross",
        loc=mu_cross_loc,
        scale=mu_cross_scale,
        conditionally_independent=True,
    )

    log.debug(f"xi_sigma{xi_sigma}")
    log.debug(f"xi_mu_cross{xi_mu_cross}")

    """ Correlate with cholsky and multivariant normal distribution
    """
    Sigma = yield LKJCholesky(
        name="Sigma_cholesky",
        dimension=4,
        concentration=2.0,  # eta
        # validate_args=True,
        transform=transformations.CorrelationCholesky(),
        conditionally_independent=True,
    )
    Sigma = tf.einsum(
        "...ij,...i->...ij",  # todo look at i,j
        Sigma,
        tf.stack([phi_sigma, eta_sigma, xi_sigma, m_sigma], axis=-1),
    )
    Sigma = yield Deterministic(
        name=f"Sigma",
        value=Sigma,
        shape_label=("testing_state_vars", "testing_state_vars"),
    )
    log.debug(f"Sigma state:\n{Sigma}")

    # Stack all means for multivariant distribution
    mu = tf.stack([phi_mu_cross, eta_mu_cross, xi_mu_cross, m_mu], axis=-1)
    state = yield MvNormalCholesky(
        name="testing_MvNormalCholesky",
        loc=mu,
        scale_tril=Sigma,
        validate_args=True,
        event_stack=(modelParams.num_countries, num_knots),
        shape_label=("country", "spline"),
    )
    log.debug(f"state:\n{state}")

    """ Transform and add to trace
    """

    # Get variables from state to transform them
    phi_cross = tf.gather(state, 0, axis=-1)
    eta_cross = tf.gather(state, 1, axis=-1)
    xi_cross = tf.gather(state, 2, axis=-1)
    m_ast = tf.gather(state, 3, axis=-1)

    # Transform variables
    phi = tf.math.sigmoid(phi_cross)
    eta = tf.math.softplus(eta_cross)
    xi = tf.einsum(
        "...cb,c->...cb",
        tf.math.softplus(xi_cross),
        tf.reduce_sum(modelParams.N_data_tensor, axis=-1) / 10000,
    )

    # Add all vars to the trace
    phi_det = yield Deterministic(
        name=name_phi,
        value=phi,
    )
    eta_det = yield Deterministic(
        name=name_eta,
        value=eta,
    )
    xi_det = yield Deterministic(
        name=name_xi,
        value=xi,
    )
    m_ast_det = yield Deterministic(
        name=name_m_ast,
        value=m_ast,
    )

    return (phi, eta, xi, m_ast)


def construct_Bsplines_basis(modelParams):
    r"""
    Function to construct the basis functions for all BSplines, should only be called
    once. Uses splipy python library.

    Parameters
    ----------
    modelParams: :py:class:`covid19_npis.ModelParams`
        Instance of modelParams, mainly used for number of age groups and
        number of countries.

    degree: optional
        degree corresponds to exponent of the splines i.e. degree of three corresponds
        to a cubic spline.
        |default| 3

    knots: list, optional
        Knots array used for constructing the BSplines.
        |default| one knot every 7 days

    Returns
    -------
    :
        |shape| time, knots?
    """

    B = modelParams.spline_basis
    log.debug(f"spline basis:\n{B}")
    return tf.constant(B, dtype="float32")


def _calculate_Bsplines(coef, basis):
    r"""
    Calculates the Bsplines given the basis functions B and the coefficients x.

    .. math::

        x(t) = \sum_{b} x_b B_b(t)


    Parameters
    ----------

    coef:
        Coefficients :math:`x`.
        |shape| ...,country, spline

    basis:
        Basis functions tensor :math:`B.`
        |shape| time, spline

    Returns
    -------
    :
        :math:`x(t)`
        |shape| ...,time, country

    """

    return tf.einsum("...cb,tb->...tc", coef, basis)
