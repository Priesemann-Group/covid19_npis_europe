def construct_positive_tests(name, new_cases_delayed, phi, modelParams):
    r"""
        .. math::

            n_{{+}, {c,a}}(t) =\Tilde{E}_{\text{delayTest}, {c,a}}(t) \cdot \phi_{+, c}(t)


        Parameters
        -----------

        name: str
            Name of the variable for the new positive cases :math:`n_{{+}, {c,a}}(t)`
            in the trace.

        new_cases_delayed: tf.Tensor
            New cases with reporting delay :math:`\Tilde{E}_{\text{delayTest}, c,a}(t).`
            |shape| batch, time, country, age_group

        phi: tf.Tensor
            Fraction of positive tests :math:`\phi_{+, c}(t).`
            |shape| batch, time, country

        modelParams: :py:class:`covid19_npis.ModelParams`
            Instance of modelParams, mainly used for number of age groups and
            number of countries.

    """
    return


def construct_total_number_of_tests_performed(
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

        phi: tf.Tensor
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
    """
    return
