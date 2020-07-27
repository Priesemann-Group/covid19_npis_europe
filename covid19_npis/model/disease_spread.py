import tensorflow as tf
import logging
import pymc4 as pm
from .utils import gamma
import tensorflow_probability as tfp


log = logging.getLogger(__name__)


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
        l: number,optional
            Number of time steps we need to go into the past
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
    I_0_t = tf.clip_by_value(I_0_t, 1e-12, 1e12)

    return I_0_t


def construct_generation_interval(
    mu_k=4.8 / 0.04, mu_theta=0.04, theta_k=0.8 / 0.1, theta_theta=0.1, l=16
):
    r"""
    Generates the generation interval with two underlying gamma distributions for mu and theta
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
    l: number, optional
        Length of generation interval i.e :math:`t` in the formula above
        |default| 16

    Returns
    -------
    :
        Normalized generation interval pdf
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

    log.debug(f"g_mu:\n{g_mu}")
    log.debug(f"g_mu:\n{g_theta}")

    # Avoid error related to possible batch dimensions
    if len(g_theta.shape) > 0:
        g_theta = tf.expand_dims(g_theta, axis=-1)
        g_mu = tf.expand_dims(g_mu, axis=-1)

    """ Construct generation interval gamma distribution from underlying
        generation distribution
    """

    g = gamma(tf.range(1, l, dtype=g_mu.dtype), g_mu / g_theta, 1.0 / g_theta)
    # g = weibull(tf.range(1, l, dtype=g_mu.dtype), g_mu / g_theta, 1.0 / g_theta)

    # Get the pdf and normalize
    # g_p, norm = tf.linalg.normalize(g, 1)
    if len(g.shape) > 1:
        g = g / tf.expand_dims(tf.reduce_sum(g, axis=-1), axis=-1)
    else:
        g = g / tf.reduce_sum(g)
    return g


@tf.function(autograph=False, experimental_compile=True)
def InfectionModel(N, I_0, R_t, C, g_p):
    r"""
    This function combines a variety of different steps:

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
        Total population per country
        |shape| country, age_group
    C:
        inter-age-group Contact-Matrix (see 8)
        |shape| country, age_group, age_group
    g_p:
        Normalized PDF of the generation interval
        |shape| batch_dims(?), l

    Returns
    -------
    :
        Sample from distribution of new, daily cases
    """

    # @tf.function(autograph=False)
    def new_infectious_cases_next_day(params, R):
        # Unpack a:
        # Old I_next is I_lastv now
        I_t, I_lastv, S_t = params  # Internal state

        # Internal state
        f = S_t / N

        # These are the infections over which the convolution is done
        log.debug(f"I_lastv: {I_lastv}")

        # Calc "infectious" people, weighted by serial_p (country x age_group)
        infectious = tf.einsum("t...ca,...t->...ca", I_lastv, g_p)

        # Calculate effective R_t [country,age_group] from Contact-Matrix C [country,age_group,age_group]
        R_sqrt = tf.math.sqrt(R)
        R_diag = tf.linalg.diag(R_sqrt)
        R_eff = tf.einsum(
            "...cij,...cik,...ckl->...cil", R_diag, C, R_diag
        )  # Effective growth number

        log.debug(f"infectious: {infectious}")
        log.debug(f"R_eff:\n{R_eff}")
        log.debug(f"f:\n{f}")

        # Calculate new infections
        new = tf.einsum("...ci,...cij,...cj->...cj", infectious, R_eff, f)
        log.debug(f"new:\n{new}")
        I_nextv = tf.concat(
            [new[tf.newaxis, ...], I_lastv[:-1, ...],], axis=0,
        )  # Create new infected population for new step, insert latest at front

        S_t = S_t - new

        return new, I_nextv, S_t

    # Number of days that we look into the past for our convolution
    l = g_p.shape[-1] + 1

    # Generate exponential distributed intial I_0_t
    I_0_t = _construct_I_0_t(I_0, l - 1)
    # Clip in order to avoid infinities
    I_0_t = tf.clip_by_value(I_0_t, 1e-12, 1e12)
    log.debug(f"I_0_t:\n{I_0_t}")

    # TO DO: Documentation
    # log.info(f"R_t outside scan:\n{R_t}")
    total_days = R_t.shape[0]

    # Initial susceptible population = total - infected
    S_initial = N - tf.reduce_sum(I_0_t, axis=0)

    """ Calculate time evolution of new, daily infections
        as well as time evolution of susceptibles
        as lists
    """
    initial = (tf.zeros(S_initial.shape, dtype=S_initial.dtype), I_0_t, S_initial)
    out = tf.scan(fn=new_infectious_cases_next_day, elems=R_t, initializer=initial)
    daily_infections_final = out[0]

    # Transpose tensor in order to have batch dim before time dim
    if len(daily_infections_final.shape) == 4:
        daily_infections_final = tf.transpose(daily_infections_final, perm=(1, 0, 2, 3))

    return daily_infections_final  # batch_dims x time x country x age
