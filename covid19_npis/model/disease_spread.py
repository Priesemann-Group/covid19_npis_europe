def _construct_generation_interval_gamma(
    mu_k=4.8 / 0.04, mu_theta=0.04, theta_k=0.8 / 0.1, theta_theta=0.1,
):
    r"""
    Returns a generator for the generation interval distribution g.

    .. math::

        g(\tau) = Gamma(\tau;
        k = \frac{\mu_{D_{\text{gene}}}}{\theta_{D_\text{gene}}},
        \theta=\theta_{D_\text{gene}})

    
    whereby the underlying distribution are modeled as follows

    .. math::
    
        \mu_{D_{\text{gene}}} &\sim Gamma(k = 4.8/0.04, \theta=0.04) \\
        \theta_{D_\text{gene}} &\sim Gamma(k = 0.8/0.1, \theta=0.1)


    Parameters
    ----------
    mu_k : number, optional
        k/concentration parameter for underlying gamma distribution of mu (:math:`\mu_{D_{\text{gene}}}`).
    mu_theta : number, optional
        theta/scale parameter for underlying gamma distribution of mu (:math:`\mu_{D_{\text{gene}}}`).
    theta_k : number, optional
        k/concentration parameter for underlying gamma distribution of theta (:math:`\theta_{D_\text{gene}}`).
    theta_theta : number, optional
        theta/scale parameter for underlying gamma distribution of theta (:math:`\theta_{D_\text{gene}}`).
    """
    g = {}

    """ The Shape parameter k of generation interval distribution is
        a gamma distribution with:
    """
    g["mu"] = {}
    g["mu"]["k"] = mu_k
    g["mu"]["θ"] = mu_theta

    # Pymc4 and tf use alpha and beta as parameters so we need to take 1/θ as rate.
    # See https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Gamma
    # and https://en.wikipedia.org/wiki/Gamma_distribution
    g_mu = yield pm.Gamma(name="g_mu", concentration=g["k"]["k"], rate=1 / g["mu"]["θ"])

    """ Shape parameter k of generation interval distribution is
        a gamma distribution with:
    """
    g["θ"] = {}
    g["θ"]["k"] = theta_k
    g["θ"]["θ"] = theta_theta

    g_theta = yield pm.Gamma(
        name="g_theta", concentration=g["θ"]["k"], rate=1 / g["θ"]["θ"]
    )

    """ Construct generation interval gamma distribution from underlying
        generation distriibutions (see above)
        k = alpha = mu/θ
    """
    g = yield pm.Gamma(
        # Emil: How do I make this time dependent?
        # Sebastian: Not too use, we also want it normalized. Maybe Jonas can help with that.
        name="g",
        concentration=g_mu / g_theta,
        rate=1 / g_theta,
        # batch_stack = time ?
    )
    return g
