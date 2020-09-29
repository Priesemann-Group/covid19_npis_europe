from .distributions import HalfCauchy, Normal, StudentT, HalfNormal


def calc_positive_tests(name, new_cases_delayed, phi_plus, phi_age, modelParams):
    r"""
        .. math::

            n_{{+}, {c,a}}(t) =\Tilde{E}_{\text{delayTest}, {c,a}}(t) \cdot \phi_{+, c}(t) \cdot \phi_{\text{age}, a},


        Parameters
        -----------

        name: str
            Name of the variable for the new positive cases :math:`n_{{+}, {c,a}}(t)`
            in the trace.

        new_cases_delayed: tf.Tensor
            New cases with reporting delay :math:`\Tilde{E}_{\text{delayTest}, c,a}(t).`
            |shape| batch, time, country, age_group

        phi_plus: tf.Tensor
            Fraction of positive tests :math:`\phi_{+, c}(t).`
            |shape| batch, time, country

        phi_age: tf.Tensor
            Fraction of positive tests :math:`\phi_{\text{age}, a}.`
            |shape| batch, age_group

        modelParams: :py:class:`covid19_npis.ModelParams`
            Instance of modelParams, mainly used for number of age groups and
            number of countries.

        Returns
        -------
        :
            :math:`n_{{+}, {c,a}}(t)`
            |shape| batch, time, country, age_group
    """
    return


def calc_total_number_of_tests_performed(
    name, new_cases_delayed, phi, eta, xi, modelParams
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

        new_cases_delayed: tf.Tensor
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

        modelParams: :py:class:`covid19_npis.ModelParams`
            Instance of modelParams, mainly used for number of age groups and
            number of countries.

        Returns
        -------
        :
            :math:`n_{\Sigma, c,a}(t)`
            |shape| batch, time, country, age_group
    """
    return


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
        name=f"sigma_{name}", scale=sigma_scale, conditionally_independent=True,
    )
    mu = yield Normal(
        name=f"mu_{name}", loc=mu_loc, scale=mu_scale, conditionally_independent=True,
    )
    phi_dagger = yield Normal(
        name=f"{name}_dagger",
        loc=mu,
        scale=sigma,
        event_stack=(modelParams.num_countries),
        shape_label="country",
        conditionally_independent=True,
    )

    phi = yield Deterministic(
        name=name, value=phi_dagger * tf.math.sigmoid(phi_dagger), shape_label="country"
    )

    return phi


def _construct_phi_age(name, modelParams, sigma_scale=0.2):
    r"""

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
        name=f"sigma_{name}",
        scale=sigma_scale,
        event_stack=modelParams.num_age_groups,
        conditionally_independent=True,
        shape_label="age_group",
    )
    phi_cross = yield Normal(
        name=f"{name}_cross",
        loc=0,
        scale=sigma,
        shape_label=age_group,
        conditionally_independent=True,
    )

    phi = yield Deterministic(
        name=name, value=tf.math.exp(phi_cross), shape_label="age_group"
    )

    return phi


def _construct_reporting_delay(
    name,
    modelParams,
    m_ast,
    mu_loc=1.5,
    mu_scale=0.4,
    sigma_theta_scale=0.2,
    sigma_m_scale=3.0,
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

        sigma_theta_scale: optional
            Scale parameter for the HalfNorml distribution :math:`\sigma_{\theta_{D_\text{test}}}.`
            |default| 0.2

        sigma_m_scale: optional
            Scale parameter for the HalfNorml distribution :math:`\sigma_{m_{D\, \text{test}}}.`
            |default| 3.0

        Returns
        -------
        :
            :math:`m_{D_\text{test},c,b}`
            |shape| batch, country, spline
    """
    mu = yield Normal(
        name=f"mu_{name}", loc=mu_loc, scale=mu_scale, conditionally_independent=True
    )
    sigma_theta = yield HalfNormal(
        name=f"sigma_theta_{name}",
        scale=sigma_theta_scale,
        conditionally_independent=True,
    )
    theta = yield Normal(
        name=f"theta_{name}",
        loc=mu,
        scale=sigma_theta,
        event_stack=modelParams.num_countries,
        shape_label="country",
        conditionally_independent=True,
    )
    sigma_m = yield HalfNormal(
        name=f"sigma_m_{name}", scale=sigma_m_scale, conditionally_independent=True
    )
    delta_m = yield Normal(
        name=f"delta_{name}",
        loc=0.0,
        scale=sigma_m,
        event_stack=modelParams.num_countries,
        shape_label="country",
        conditionally_independent=True,
    )

    # We need to add the spline dimension at some point i.e. prop. expand delta_m
    m = yield Deterministic(
        name=name, value=m_ast + delta_m, shape_label=("country", "spline")
    )

    return m


def _construct_testing_state(
    name,
    modelParams,
    mu_cross_loc=0.0,
    mu_cross_scale=10.0,
    mu_m_loc=12.0,
    mu_m_scale=2.0,
    sigma_cross_scale=10.0,
    sigma_m_scale=1.0,
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
        name: str
            Name of the studentT distribution variable :math:`(\phi^\dagger_{\text{tested},c,b},
            \: \eta^\dagger_{\text{traced},c,b},\: \xi^\dagger_{c,b},\: m^\ast_{D_\text{test},c,b}).`

        modelParams: :py:class:`covid19_npis.ModelParams`
            Instance of modelParams, mainly used for number of age groups and
            number of countries.

        mu_cross_loc: optional
            Location parameter for the three Normal distributions :math:`\mu_{\phi^\dagger_+},\: \mu_{\eta^\dagger_\text{traced}},\: \mu_{\xi^\dagger}.`
            |default| 0.0

        mu_cross_scale: optional
            Scale parameter for the three Normal distributions :math:`\mu_{\phi^\dagger_+},\: \mu_{\eta^\dagger_\text{traced}},\: \mu_{\xi^\dagger}.`
            |default| 10.0

        mu_m_loc: optional
            Location parameter for the Normal distribution :math:`\mu_{m_{D_\text{test}}}.`
            |default| 12.0

        mu_m_scale: optional
            Scale parameter for the Normal distribution :math:`\mu_{m_{D_\text{test}}}.`
            |default| 2.0

        sigma_cross_scale: optional
            Scale parameter for the three HalfCauchy distributions :math:`\sigma_\phi, \sigma_\eta, \sigma_\xi.`
            |default| 10.0

        sigma_m_scale: optional
            Scale parameter for the HalfNormal distribution :math:`\sigma_m.`
            |default| 1.0

        Returns
        -------
        : 
            Testing state tuple :math:`(\phi_{+,c,b},
            \: \eta_{\text{traced},c,b},\: \xi_{c,b},\: m^\ast_{D_\text{test},c,b}).`
            |shape| 4 x (batch, country * spline)
    """
    return


def spline():
    """
    TODO
    ----
    - Write docstring
    - implement
    - @Jonas :)
    """
