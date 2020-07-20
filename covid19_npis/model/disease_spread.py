import tensorflow as tf
import logging
import pymc4 as pm
import tensorflow_probability as tfp

log = logging.getLogger(__name__)


def gamma(x, alpha, beta):
    return tf.math.pow(x, (alpha - 1)) * tf.exp(-beta * x)


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
    g_mu = yield pm.Gamma(
        name="g_mu", concentration=g["mu"]["k"], rate=1.0 / g["mu"]["θ"],conditionally_independent=True,
    )

    """ Shape parameter k of generation interval distribution is
        a gamma distribution with:
    """
    g["θ"] = {}
    g["θ"]["k"] = theta_k
    g["θ"]["θ"] = theta_theta

    g_theta = yield pm.Gamma(
        name="g_theta", concentration=g["θ"]["k"], rate=1.0 / g["θ"]["θ"],conditionally_independent=True,
    )

    """ Construct generation interval gamma distribution from underlying
        generation distributions (see above)
        k = alpha = mu/θ
    """

    return g_mu, g_theta



@tf.function
def InfectionModel(N, I_0, R_t, C, g=None, l=16):
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
        Initial number of infectious. (batch_dims x country x age_group)
    R_t:
        Reproduction number matrix. (time x batch_dims x country x age_group)
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

    if g is None:
        g_mu, g_theta = _construct_generation_interval_gamma()

    g = gamma(tf.range(1, l, dtype=g_mu.dtype), 4, 1 / 0.5)
    # Get the pdf and normalize
    g_p, norm = tf.linalg.normalize(g, 1)

    # shift range by 1e-12 to allow distributions which are undefined for \tau = 0


    def new_infectious_cases_next_day(i, new_infections, S_t):

        # Internal state
        f = S_t / N
        R = tf.gather(R_t,  i, axis=0)

        #These are the infections over which the convolution is done
        I_array = new_infections.gather(indices=tf.range(i-1,i-l, -1))

        # Calc "infectious" people, weighted by serial_p (country x age_group)
        infectious = tf.einsum("t...ca,t->...ca", I_array, g_p)

        # Calculate effective R_t [country,age_group] from Contact-Matrix C [country,age_group,age_group]
        #log.info(f"R_t inside scan:\n{R}")
        #log.info(f"I_t inside scan:\n{I_array}")
        R_sqrt = tf.math.sqrt(R)
        #log.info(f"R_sqrt:\n{R_sqrt}")
        R_diag = tf.linalg.diag(R_sqrt)
        #log.info(f"R_diag:\n{R_diag}")
        #log.info(f"C:\n{C}")
        R_eff = tf.einsum(
            "...cij,...cik,...ckl->...cil", R_diag, C, R_diag
        )  # Effective growth number
        #log.info(f"R_eff:\n{R_eff}")
        # Calculate new infections
        #log.info(f"infectious:\n{infectious}")
        #log.info(f"f:\n{f}")
        new = tf.einsum("...ci,...cij,...cj->...cj", infectious, R_eff, f)
        new = tf.clip_by_value(new, 1e-7, 1e9)

        log.info(f"new:\n{new}")
        new_infections.write(i, new)

        S_t = S_t - new

        return i+1, new_infections, S_t

    """ # Generate exponential distributed intial I_0_t,
        whereby t goes l days into the past
        I_0_t should be in sum = I_0

        TODO
        ----
        - slope of exponenet should match R_0
    """

    exp_r = tf.range(start=l, limit=0.0, delta=-1.0, dtype=g.dtype)
    exp_d = tf.math.exp(exp_r)
    # exp_d = exp_d * g_p  # wieght by serial_p
    exp_d, norm = tf.linalg.normalize(
        exp_d, axis=0
    )  # normalize by dividing by sum over time-dimension

    I_0_t = tf.einsum("...ca,t->t...ca", I_0, exp_d)
    log.info(f"I_0_t:\n{I_0_t}")


    log.info(f"R_t outside scan:\n{R_t}")
    total_days = R_t.shape[0]

    #Create an Tensor array and initalize the first l elements
    new_infections = tf.TensorArray(
      dtype=R_t.dtype, size=total_days, element_shape=R_t.shape[1:])
    for i in range(l):
        new_infections.write(i, I_0_t[i])


    cond = lambda i, *_: i < total_days

    S_initial=N - tf.reduce_sum(I_0_t, axis=0)

    _, daily_infections_final, last_S_t = tf.while_loop(
        cond, new_infectious_cases_next_day,
        (l, new_infections, S_initial),
        maximum_iterations=total_days-l)

    daily_infections_final = daily_infections_final.stack()
    if len(daily_infections_final.shape) == 4:
        daily_infections_final = tf.transpose(daily_infections_final, perm=(1,0,2,3))

    return daily_infections_final #batch_dims x time x country x age

