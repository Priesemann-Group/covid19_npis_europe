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

    Returns
    -------
    : 
        Generator for the generation interval distribution :math:`g(\tau)`

    TODO
    ----
    - g time dependent 
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
        generation distributions (see above)
        k = alpha = mu/θ
    """
    g = tfp.distributions.Gamma(
        # Emil: How do I make this time dependent?
        # Sebastian: Not too use, we also want it normalized. Maybe Jonas can help with that.
        # Matthias: We're not interested in drawing samples from this distribution, but need the pdf
        #   fortunately tfp is simple in that regard g.pdf([0,1,2,3,....,10]) gives the 'weights' of the generation interval
        # Sebastian: Indeed, that is very nice! FYI the function is called .prob([]) As far as i can see one
        #   cant call it within the pymc4 model context.
        #   try to add .prob(np.arange(0,20,0.001))
        name="g",
        concentration=g_mu / g_theta,
        rate=1 / g_theta,
    )

    return g


def InfectionModel(N, I_0, R_T, C, g ,l=16):
    r"""
    .. math::

         \tilde{I_l}(t) = \frac{S_l(t)}{N_{\text{pop}, j}} R_l(t)\sum_{\tau=0}^{t} \tilde{I_l}(t-\tau) g(\tau) \label{eq:I_gene}
    

    TODO
    ----
    - documentation
    - implement I_0
    - tf scan function
    - write code

    Parameters
    ----------
    I_0:
        Initial number of infectious.
    R_T:
        Reproduction number matrix. (time x country x age_group) 
    g:
        Generation interval
    N:
        Initial population
    C:
        inter-age-group Contact-Matrix (see 8)
    l: number, optional
        Length of generation interval i.e :math:`t` in the formula above

    Returns
    -------
    :
        Sample from distribution of new, daily cases
    """

    """ Old pseudocode:
    def new_infectious_cases_next_day(S_t, Ĩ_t):
        
        #Calculate new newly infectious per day
        #Sebastian: This will probably not work like that. Someone else should look over
        #it since im not too sure how to do that.
        #Matthias: Wip see notebook, needs some variable-renaming.
        
        
        #New susceptible pool
        

        Ĩ_t_new = tf.tensordot(Ĩ_t, g)

        S_t_new = S_t - Ĩ_t_new  # eq 4

        return [S_t_new, Ĩ_t_new]

    S_t, Ĩ_t = tf.scan(
        fn=new_infectious_cases_next_day,
        elems=[],
        initializer=[S_0, I_0],  # S_0 should be population size i.e. N
    )
    """

    # Normalize
    g_p = g.prob(tf.range(1e-12,l+1e-12,dtype=g.dtype)) # shift range by 1e-12 to allow distributions which are undefined for \tau = 0
    g_p /= tf_sum(g_p)

    def new_infectious_cases_next_day(a, R_t):
        # Unpack a:
        # Old I_next is I_lastv now
        I_t, I_lastv, S_t = a
        f = S_t / N
        
        # Calc "infectious" people, weighted by serial_p (country x age_group)
        infectious = tf.einsum("tca,t->ca",E_lastv,g_p) 
        
        # Calculate effective R_t [country,age_group] from Contact-Matrix C [country,age_group,age_group]
        R_sqrt = tf.sqrt(R_t)
        R_diag = tf.einsum("ij,ci->cij",tf.eye(R_t.shape[-1],dtype=R_t.dtype), R_sqrt)
        R_eff = tf.einsum("cij,ik,ckl->cil", R_diag, C, R_diag) # Effective growth
        
        # Calculate new infections
        new = tf.einsum("nj,njk,nk->nk",infectious,R_eff,f)
        
        new_v = tf.reshape(
            new, [1 ,new.shape[0], new.shape[1]]
        )  # add dimension for concatenation
        I_nextv = tf.concat(
            [new_v, I_lastv[:-1, :, :]], 0
        )  # Create new infected population for new step, insert latest at front

        return [new, I_nextv, S_t - new]
    
    # Generate exponential distributed intial I_0_t, sum = I_0
    # I't not the real thing right now, as the slope of the exponential doesn't match R_t,
    # but close enough to avoid starting oscillations
    exp_r = tf.range(start=l,limit=0.,delta=-1.,dtype=g.dtype)
    exp_d = tf.math.exp(exp_r)
    exp_d = exp_d * g_p # wieght by serial_p
    exp_d /= tf_sum(exp_d,axis=0)   # normalize by dividing by sum over time-dimension
    I_0_t = tf.einsum("ca,t->tca",I_0,exp_d)
    #    I_0_t = tf.tensordot(I_0, exp_d, axes=0) # Calculate the outer product (axes=0) --> initial distribution
    
    # Exchanged for I_0_t
    initial = [tf.zeros(N.shape,dtype=np.float64), I_0_t, N]

    # Initialze the internal state for the scan function
    initial = [
        tf.zeros(N.shape, dtype=np.float64),
        tf.zeros([N.shape[0], N.shape[1], l], dtype=np.float64),
        N,
    ]

    out = tf.scan(fn=new_infectious_cases_next_day, elems=R_T, initializer=initial)
    return out

