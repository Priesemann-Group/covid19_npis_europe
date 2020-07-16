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
        name="g_mu", concentration=g["mu"]["k"], rate=1.0 / g["mu"]["θ"]
    )

    """ Shape parameter k of generation interval distribution is
        a gamma distribution with:
    """
    g["θ"] = {}
    g["θ"]["k"] = theta_k
    g["θ"]["θ"] = theta_theta

    g_theta = yield pm.Gamma(
        name="g_theta", concentration=g["θ"]["k"], rate=1.0 / g["θ"]["θ"]
    )

    """ Construct generation interval gamma distribution from underlying
        generation distributions (see above)
        k = alpha = mu/θ
    """

    return g_mu, g_theta


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
        Initial number of infectious.
    R_t:
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

    if g is None:
        g_mu, g_theta = _construct_generation_interval_gamma()

    g = gamma(tf.range(1e-12, l + 1e-12, dtype=g_mu.dtype), 4, 1 / 0.5)
    # Get the pdf and normalize
    g_p, norm = tf.linalg.normalize(g, 1)
    # shift range by 1e-12 to allow distributions which are undefined for \tau = 0

    def new_infectious_cases_next_day(a, R):
        # Unpack a:
        # Old I_next is I_lastv now
        I_t, I_lastv, S_t = a  # Internal state
        f = S_t / N

        # Calc "infectious" people, weighted by serial_p (country x age_group)
        infectious = tf.einsum("tca,t->ca", I_lastv, g_p)

        # Calculate effective R_t [country,age_group] from Contact-Matrix C [country,age_group,age_group]
        log.info(f"R_t inside scan:\n{R}")
        log.info(f"I_t inside scan:\n{I_t}")
        R_sqrt = tf.math.sqrt(R)
        log.info(f"R_sqrt:\n{R_sqrt}")
        R_diag = tf.linalg.diag(R_sqrt)
        log.info(f"R_diag:\n{R_diag}")
        log.info(f"C:\n{C}")
        R_eff = tf.einsum(
            "cij,cik,ckl->cil", R_diag, C, R_diag
        )  # Effective growth number
        log.info(f"R_eff:\n{R_eff}")
        # Calculate new infections
        log.info(f"infectious:\n{infectious}")
        log.info(f"f:\n{f}")
        new = tf.einsum("ci,cij,cj->cj", infectious, R_eff, f)
        log.info(f"new:\n{new}")
        new_v = tf.reshape(
            new, [1, new.shape[0], new.shape[1]]
        )  # add dimension for concatenation
        I_nextv = tf.concat(
            [
                new_v,
                tf.slice(
                    I_lastv,
                    [0, 0, 0],
                    [I_lastv.shape[0] - 1, I_lastv.shape[1], I_lastv.shape[2]],
                ),
            ],
            0,
        )  # Create new infected population for new step, insert latest at front

        return [new, I_nextv, S_t - new]

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

    I_0_t = tf.einsum("ca,t->tca", I_0, exp_d)
    log.info(f"I_0_t:\n{I_0_t}")

    initial = [tf.zeros(N.shape, dtype=R_t.dtype), I_0_t, N]

    log.info(f"initial:\n{initial[0]}\n{initial[1]}\n{initial[2]}")
    log.info(f"R_t outside scan:\n{R_t}")
    out = tf.scan(fn=new_infectious_cases_next_day, elems=R_t, initializer=initial)
    return out[0]
