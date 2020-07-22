import tensorflow as tf
import logging
import pymc4 as pm
import tensorflow_probability as tfp

log = logging.getLogger(__name__)

# Distribution pdf for generation interval
def gamma(x, alpha, beta):
    return tf.math.pow(x, (alpha - 1)) * tf.exp(-beta * x)


def _construct_I_0_t(I_0, l=16):
    r"""
    Generates a exponential distributed :math:`I_t`, which goes :math:`l` steps into the past
    i.e has a length of :math:`l`. This is needed because of the convolution with the generation
    interval inside the time step function.

    The :math:`I_t` is normalized by the initial :math:`I_0` values:

    .. math::

        \sum_t{I_t} = I_0

    TODO
    ----
    - slope of exponent should match R_0
    - add more in depth explanation

    Parameters
    ----------
        I_0:
            Tensor of initial I_0 values.
        l:number,optional
            Number of time steps we need into the past
            |default| 16
    Returns
    -------
    :
        I_t
    """

    # Construct exponential function
    exp = tf.math.exp(tf.range(start=l, limit=0.0, delta=-1.0, dtype=I_0.dtype))

    # Normalize to one
    exp, norm = tf.linalg.normalize(tensor=exp, ord=1, axis=0)

    # sums every given I_0 with the exponential function values
    I_0_t = tf.einsum("...ca,t->t...ca", I_0, exp)
    I_0_t = tf.clip_by_value(I_0_t, 1e-7, 1e9)


    return I_0_t

def construct_generation_interval(mu_k=4.8 / 0.04, mu_theta=0.04, theta_k=0.8 / 0.1, theta_theta=0.1, l=16):
    # Generate generation interval
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
    g_mu = yield pm.Gamma(
        name="g_mu",
        concentration=g["mu"]["k"],
        rate=1.0 / g["mu"]["θ"],
        conditionally_independent=True,
    )

    """ Shape parameter k of generation interval distribution is
        a gamma distribution with:
    """
    g["θ"] = {}
    g["θ"]["k"] = theta_k
    g["θ"]["θ"] = theta_theta

    g_theta = yield pm.Gamma(
        name="g_theta",
        concentration=g["θ"]["k"],
        rate=1.0 / g["θ"]["θ"],
        conditionally_independent=True,
    )

    """ Construct generation interval gamma distribution from underlying
        generation distribution
    """
    if len(g_theta.shape)>0:
        g_theta = tf.expand_dims(g_theta, axis=-1)
        g_mu = tf.expand_dims(g_mu, axis=-1)
    g = gamma(tf.range(1, l, dtype=g_mu.dtype), g_mu / g_theta, 1.0 / g_theta)

    # Get the pdf and normalize
    g_p, norm = tf.linalg.normalize(g, 1)
    return g_p


#@tf.function(autograph=False, experimental_compile=True)
def InfectionModel(N, I_0, R_t, C, g_p):
    r"""
    This function combines a variety of different steps:

        #. Generates the generation interval with two underlying gamma distributions for mu and theta
            .. math::

                g(\tau) = Gamma(\tau;
                k = \frac{\mu_{D_{\text{gene}}}}{\theta_{D_\text{gene}}},
                \theta=\theta_{D_\text{gene}})


            whereby the underlying distribution are modeled as follows

            .. math::

                \mu_{D_{\text{gene}}} &\sim Gamma(k = 4.8/0.04, \theta=0.04) \\
                \theta_{D_\text{gene}} &\sim Gamma(k = 0.8/0.1, \theta=0.1)

        #. Converts the given :math:`I_0` values  to an exponential distributed initial :math:`I_{0_t}` with an
           length of :math:`l` this can be seen in :py:func:`_construct_I_0_t`.

        #. Calculates :math:`R_{eff}` for each time step using the given contact matrix :math:`C`:

            .. math::
                R_{diag} &= \text{diag}(\sqrt{R}) \\
                R_{eff}  &= R_{diag} \cdot C \cdot R_{diag}

        #. Calculates the :math:`\tilde{I}` arrays i.e. new infectious for each age group and
           country, with the efficient reproduction matrix :math:`R_{eff}`, the susceptible pool
           :math:`S`, the population size :math:`N` and the generation interval :math:`g(\tau)`.
           This is done recursive for every time step.

            .. math::
                    \tilde{I}(t) &= \frac{S(t)}{N} \cdot R_{eff} \cdot \sum_{\tau=0}^{t} \tilde{I}(t-1-\tau) g(\tau) \\
                    S(t) &= S(t-1) - \tilde{I}(t-1)

    TODO
    ----
    - rewrite while loop to tf.scan function


    Parameters
    ----------
    I_0:
        Initial number of infectious.
        |shape| batch_dims, country, age_group
    R_t:
        Reproduction number matrix.
        |shape| time, batch_dims, country, age_group
    N:
        Initial population
    C:
        inter-age-group Contact-Matrix (see 8)
        |shape| country, age_group, age_group
    l: number, optional
        Length of generation interval i.e :math:`t` in the formula above
        |default| 16
    mu_k : number, optional
        Concentration/k parameter for underlying gamma distribution of mu (:math:`\mu_{D_{\text{gene}}}`).
        |default| 120
    mu_theta : number, optional
        Scale/theta parameter for underlying gamma distribution of mu (:math:`\mu_{D_{\text{gene}}}`).
        |default| 0.04
    theta_k : number, optional
        Concentration/k parameter for underlying gamma distribution of theta (:math:`\theta_{D_\text{gene}}`).
        |default| 8
    theta_theta : number, optional
        Scale/theta parameter for underlying gamma distribution of theta (:math:`\theta_{D_\text{gene}}`).
        |default| 0.1

    Returns
    -------
    :
        Sample from distribution of new, daily cases
    """

    l = g_p.shape[-1]+1

    # Generate exponential distributed intial I_0_t
    I_0_t = _construct_I_0_t(I_0, l)
    # Clip in order to avoid infinities
    I_0_t = tf.clip_by_value(I_0_t, 1e-7, 1e9)

    #@tf.function(autograph=False)
    def new_infectious_cases_next_day(i, new_infections, S_t):

        # Internal state
        f = S_t / N
        R = tf.gather(R_t, i, axis=0)

        # These are the infections over which the convolution is done
        I_array = new_infections.stack(name='stack')[:-l:-1]

        # Calc "infectious" people, weighted by serial_p (country x age_group)
        #I_array = tf.ones((15,2,4))
        infectious = tf.einsum("t...ca,...t->...ca", I_array, g_p)

        # Calculate effective R_t [country,age_group] from Contact-Matrix C [country,age_group,age_group]
        # log.info(f"R_t inside scan:\n{R}")
        # log.info(f"I_t inside scan:\n{I_array}")
        R_sqrt = tf.math.sqrt(R)
        # log.info(f"R_sqrt:\n{R_sqrt}")
        R_diag = tf.linalg.diag(R_sqrt)
        # log.info(f"R_diag:\n{R_diag}")
        # log.info(f"C:\n{C}")
        R_eff = tf.einsum(
            "...cij,...cik,...ckl->...cil", R_diag, C, R_diag
        )  # Effective growth number
        # log.info(f"R_eff:\n{R_eff}")
        # Calculate new infections
        # log.info(f"infectious:\n{infectious}")
        # log.info(f"f:\n{f}")
        new = tf.einsum("...ci,...cij,...cj->...cj", infectious, R_eff, f)

        log.info(f"new:\n{new}")
        new_infections = new_infections.write(i, new)

        S_t = S_t - new

        return i + 1, new_infections, S_t

    """ # Generate exponential distributed intial I_0_t,
        whereby t goes l days into the past
        I_0_t should be in sum = I_0

        TODO
        ----
        - slope of exponenet should match R_0
    """

    exp_r = tf.range(
        start=l,
        limit=0.0,
        delta=-1.0,
        dtype=R_t.dtype,
        name='exp_range'
    )
    exp_d = tf.math.exp(exp_r)
    # exp_d = exp_d * g_p  # weight by serial_p
    exp_d, norm = tf.linalg.normalize(
        exp_d, axis=0
    )  # normalize by dividing by sum over time-dimension

    log.info(f"I_0_t:\n{I_0_t}")


    log.info(f"R_t outside scan:\n{R_t}")
    total_days = R_t.shape[0]

    #Create an Tensor array and initalize the first l elements
    new_infections = tf.TensorArray(
        dtype=R_t.dtype,
        size=total_days,
        element_shape=R_t.shape[1:]
    )
    for i in range(l):
        new_infections = new_infections.write(i, I_0_t[i])


    cond = lambda i, *_: i < total_days

    S_initial = N - tf.reduce_sum(I_0_t, axis=0)

    _, daily_infections_final, last_S_t = tf.while_loop(
        cond,
        new_infectious_cases_next_day,
        (l, new_infections, S_initial),
        maximum_iterations=total_days-l,
        name='spreading_loop'
    )

    daily_infections_final = daily_infections_final.stack()
    if len(daily_infections_final.shape) == 4:
        daily_infections_final = tf.transpose(daily_infections_final, perm=(1,0,2,3))

    return daily_infections_final # batch_dims x time x country x age
