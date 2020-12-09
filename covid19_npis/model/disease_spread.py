import tensorflow as tf
import logging
import pymc4 as pm
from .utils import gamma, convolution_with_fixed_kernel
import numpy as np
import tensorflow_probability as tfp


from covid19_npis import transformations
from covid19_npis.model.distributions import (
    HalfCauchy,
    Normal,
    Gamma,
    LogNormal,
    Deterministic,
)


log = logging.getLogger(__name__)


def construct_h_0_t(
    modelParams, len_gen_interv_kernel, R_t, mean_gen_interv, mean_test_delay=10,
):
    r"""
    Generates a prior for E_0_t, based on the observed number of cases during the first
    5 days. Currently it is implemented to take the first value of R_t, and multiply the
    inverse of R_t with first observed values until the begin of the simulation is reached.
    This is then used as a prior for a lognormal distribution which set the h_0_t.


    Parameters
    ----------
        modelParams: :py:class:`covid19_npis.ModelParams`
            Instance of modelParams, mainly used for number of age groups and
            number of countries.
        len_gen_interv_kernel: number
            ...some description
        R_t: tf.tensor
            Time dependent reproduction number tensor :math:`R(t)`.
            |shape| time, batch, country, age group
        mean_gen_interv: countries
            ...some description
        mean_test_delay: number, optional
            ...some description
            |default| 10
    Returns
    -------
    :
        h_0_t:
            some description
            |shape| time, batch, country, age_group
    """
    batch_dims = tuple(R_t.shape)[:-3]
    data = modelParams.pos_tests_data_tensor
    diff_sim_data = modelParams.min_offset_sim_data
    assert data.ndim == 3
    assert (
        diff_sim_data > len_gen_interv_kernel + mean_test_delay
    ), "min_offset_sim_data is to small"
    i_data_begin_list = modelParams.indices_begin_sim

    # eigvals, _ = tf.linalg.eigh(R_t[..., i_data_begin, :, :])
    # largest_eigval = eigvals[-1]
    R_t_rescaled = (
        R_t - tf.ones(R_t.shape, dtype=R_t.dtype)
    ) / mean_gen_interv + tf.ones(R_t.shape, dtype=R_t.dtype)
    R_inv = 1 / R_t_rescaled
    R_inv = tf.clip_by_value(R_inv, clip_value_min=0.7, clip_value_max=1.2)
    """
    R = R_t_rescaled[0]
    R_sqrt = tf.math.sqrt(R)
    R_diag = tf.linalg.diag(R_sqrt)
    R_eff = R_diag @ C @ R_diag
    log.debug(f"R_eff for h_0_t construction {R_eff.shape}:\n{R_eff}")
    R_eff_inv = tf.linalg.pinv(R_eff)
    log.debug(f"R_eff_inv for h_0_t construction:\n{R_eff_inv}")
    """

    i_sim_begin_list = i_data_begin_list - diff_sim_data
    avg_cases_begin = []
    for c in range(data.shape[1]):
        avg_cases_begin.append(
            np.nanmean(data[i_data_begin_list[c] : i_data_begin_list[c] + 5, c], axis=0)
        )
    avg_cases_begin = np.array(avg_cases_begin)
    h_0_t_mean = []
    E_t = avg_cases_begin
    log.debug(f"avg_cases_begin:\n{avg_cases_begin}")

    for i in range(diff_sim_data - len_gen_interv_kernel - mean_test_delay):
        """
        R = tf.gather(
            R_t_rescaled,
            i_sim_begin_list + diff_sim_data - mean_test_delay - i,
            axis=-3,
            batch_dims=1,
        )
        """
        # E_t = tf.linalg.matvec(R_eff_inv, E_t)
        E_t = R_inv[0] * E_t

        log.debug(f"i, E_t:{i}\n{E_t}")

    h_0_t_mean = [None for _ in range(len_gen_interv_kernel - 1, -1, -1)]
    for i in range(len_gen_interv_kernel - 1, -1, -1):
        # R = tf.gather(R_t_rescaled, i_sim_begin_list + i, axis=-3, batch_dims=1,))
        # E_t = tf.linalg.matvec(R_eff_inv, E_t)
        E_t = R_inv[0] * E_t
        log.debug(f"i, E_t:{i}\n{E_t}")
        h_0_t_mean[i] = E_t
    h_0_t_mean = tf.stack(h_0_t_mean, axis=-3) / len_gen_interv_kernel
    h_0_t_mean = tf.clip_by_value(h_0_t_mean, 1e-5, 1e6)
    log.debug(f"h_0_t_mean:\n{h_0_t_mean.shape}")

    h_0_base = h_0_t_mean[..., 0:1, :, :] * tf.exp(
        (
            yield Normal(
                name="E_0_diff_base",
                loc=0.0,
                scale=3.0,
                conditionally_independent=True,
                event_stack=tuple(h_0_t_mean[..., 0:1, :, :].shape[-3:]),
            )
        ),
    )
    h_0_mean_diff = h_0_t_mean[..., 1:, :, :] - h_0_t_mean[..., :-1, :, :]
    h_0_base_add = h_0_mean_diff * tf.exp(
        (
            yield Normal(
                name="E_0_diff_add",
                loc=0.0,
                scale=1.0,
                conditionally_independent=True,
                event_stack=tuple(h_0_mean_diff.shape[-3:]),
            )
        ),
    )
    log.debug(f"h_0_base:\n{h_0_base.shape}")
    log.debug(f"h_0_base_add:\n{h_0_base_add.shape}")
    log.debug(f"R_t:\n{R_t.shape}")

    h_0_t_rand = tf.math.cumsum(
        tf.concat([h_0_base, h_0_base_add,], axis=-3,), axis=-3,
    )  # shape:  batch_dims x len_gen_interv_kernel x countries x age_groups

    h_0_t_rand = tf.einsum(
        "...kca->k...ca", h_0_t_rand
    )  # Now: shape:  len_gen_interv_kernel x batch_dims x countries x age_groups

    log.debug(f"h_0_t_rand:\n{h_0_t_rand.shape}")
    h_0_t = []
    batch_shape = R_t.shape[1:-2]
    log.debug(f"batch_shape:\n{batch_shape}")
    total_len = R_t.shape[0]
    age_shape = R_t.shape[-1:]
    for i, i_begin in enumerate(i_sim_begin_list):
        h_0_t.append(
            tf.concat(
                [
                    tf.zeros((i_begin,) + batch_shape + (1,) + age_shape),
                    h_0_t_rand[..., i : i + 1, :],
                    tf.zeros(
                        (total_len - len_gen_interv_kernel - i_begin,)
                        + batch_shape
                        + (1,)
                        + age_shape
                    ),
                ],
                axis=0,
            )
        )
    h_0_t = tf.concat(h_0_t, axis=-2)
    return h_0_t


def _construct_E_0_t_transposed(E_0, l=16):
    r"""
    Generates a exponential distributed :math:`E_t`, which goes :math:`l` steps into the past
    i.e has a length of :math:`l`. This is needed because of the convolution with the generation
    interval inside the time step function.

    The :math:`E_t` is normalized by the initial :math:`E_0` values:

    .. math::

        \sum_t{E_t} = E_0

    TODO
    ----
    - slope of exponent should match R_0
    - add more in depth explanation

    Parameters
    ----------
        E_0:
            Tensor of initial E_0 values.
        l: number,optional
            Number of time steps we need to go into the past
            |default| 16
    Returns
    -------
    :
        E_t
    """

    # Construct exponential function
    exp = tf.math.exp(tf.range(start=l, limit=0.0, delta=-1.0, dtype=E_0.dtype))

    # Normalize to one
    exp, norm = tf.linalg.normalize(tensor=exp, ord=1, axis=0)

    # sums every given E_0 with the exponential function values
    E_0_t = tf.einsum("...ca,t->...tca", E_0, exp)
    E_0_t = tf.clip_by_value(E_0_t, 1e-7, 1e9)

    return E_0_t


def construct_generation_interval(
    name="g", mu_k=4.8 / 0.04, mu_theta=0.04, theta_k=0.8 / 0.1, theta_theta=0.1, l=16
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
    name: string
        Name of the distribution for trace and debugging.
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

    g_mu = yield Gamma(
        name=f"{name}_mu",
        concentration=g["mu"]["k"],
        rate=1.0 / g["mu"]["θ"],
        conditionally_independent=True,
        validate_args=True,
    )

    # g_mu = tf.constant(5.0)
    # g_theta = tf.constant(1.0)

    """ Shape parameter k of generation interval distribution is
        a gamma distribution with:
    """
    g["θ"] = {}
    g["θ"]["k"] = theta_k
    g["θ"]["θ"] = theta_theta

    g_theta = yield Gamma(
        name=f"{name}_theta",
        concentration=g["θ"]["k"],
        rate=1.0 / g["θ"]["θ"],
        conditionally_independent=True,
        validate_args=True,
    )

    log.debug(f"g_mu:\n{g_mu}")
    log.debug(f"g_theta:\n{g_theta}")

    # Add a small number here to prevent zeros. Could happen in the sampling
    # at some point and we divide at a later point by these tensors,
    # which could yield nans otherwise.
    g_theta = tf.expand_dims(g_theta, axis=-1) + 1e-8
    g_mu = tf.expand_dims(g_mu, axis=-1) + 1e-8

    """ Construct generation interval gamma distribution from underlying
        generation distribution
    """

    g = gamma(tf.range(0.1, l + 0.1, dtype=g_mu.dtype), g_mu / g_theta, 1.0 / g_theta)
    # g = weibull(tf.range(1, l, dtype=g_mu.dtype), g_mu / g_theta, 1.0 / g_theta)

    g = yield Deterministic(name=name, value=g)
    return (
        g,
        tf.expand_dims(g_mu, axis=-1),
    )  # shape g: batch_shape x len_gen_interv, shape g_mu: batch_shape x 1 x 1


def InfectionModel(N, h_0_t, R_t, C, gen_kernel):
    r"""
    This function combines a variety of different steps:

        #. Converts the given :math:`E_0` values  to an exponential distributed initial :math:`E_{0_t}` with an
           length of :math:`l` this can be seen in :py:func:`_construct_E_0_t`.

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

    Parameters
    ----------
    E_0:
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
    def loop_body(params, elems):
        # Unpack a:
        # Old E_next is E_lastv now
        R, h = elems
        E_t, E_lastv, S_t = params  # Internal state

        # Internal state
        f = S_t / N

        # Convolution:

        log.debug(f"E_t {E_t}")
        # Calc "infectious" people, weighted by serial_p (country x age_group)

        infectious = tf.einsum("t...ca,...t->...ca", E_lastv, gen_kernel)  # Convolution

        # Calculate effective R_t [country,age_group] from Contact-Matrix C [country,age_group,age_group]
        R_sqrt = tf.math.sqrt(R)
        R_diag = tf.linalg.diag(R_sqrt)
        R_eff = tf.einsum(
            "...cij,...cik,...ckl->...cil", R_diag, C, R_diag
        )  # Effective growth number

        # log.debug(f"infectious: {infectious}")
        # log.debug(f"R_eff:\n{R_eff}")
        # log.debug(f"f:\n{f}")
        # log.debug(f"h:\n{h}")

        # Calculate new infections
        new = tf.einsum("...ci,...cij,...cj->...cj", infectious, R_eff, f) + h
        new = tf.clip_by_value(new, 0, 1e9)

        log.debug(f"new:\n{new}")  # kernel_time,batch,country,age_group
        E_nextv = tf.concat(
            [new[tf.newaxis, ...], E_lastv[:-1, ...],], axis=0,
        )  # Create new infected population for new step, insert latest at front

        S_t = S_t - new

        return new, E_nextv, S_t

    # Number of days that we look into the past for our convolution
    len_gen_interv_kernel = gen_kernel.shape[-1]

    E_0_initial = tf.zeros((len_gen_interv_kernel,) + R_t.shape[1:], dtype=R_t.dtype)
    E_0_initial = E_0_initial + h_0_t[:len_gen_interv_kernel]
    S_initial = N - tf.reduce_sum(h_0_t, axis=0)

    R_t_for_loop = R_t[len_gen_interv_kernel:]
    h_t_for_loop = h_0_t[len_gen_interv_kernel:]
    # Initial susceptible population = total - infected

    """ Calculate time evolution of new, daily infections
        as well as time evolution of susceptibles
        as lists
    """

    initial = (tf.zeros(S_initial.shape, dtype=S_initial.dtype), E_0_initial, S_initial)
    out = tf.scan(fn=loop_body, elems=(R_t_for_loop, h_t_for_loop), initializer=initial)
    daily_infections_final = out[0]
    daily_infections_final = tf.concat(
        [h_0_t[:len_gen_interv_kernel], daily_infections_final], axis=0
    )

    # Transpose tensor in order to have batch dim before time dim
    daily_infections_final = tf.einsum("t...ca->...tca", daily_infections_final)

    return daily_infections_final  # batch_dims x time x country x age


def InfectionModel_unrolled(N, E_0, R_t, C, g_p):
    r"""
    This function unrolls the loop. It compiling time is slower (about 10 minutes)
    but the running time is faster and more parallel:

        #. Converts the given :math:`E_0` values to an exponential distributed initial :math:`E_{0_t}` with an
           length of :math:`l` this can be seen in :py:func:`_construct_E_0_t`.

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
    E_0:
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

    # Number of days that we look into the past for our convolution
    l = g_p.shape[-1] + 1

    # Generate exponential distributed intial E_0_t
    E_0_t = _construct_E_0_t_transposed(E_0, l - 1)
    # Clip in order to avoid infinities
    E_0_t = tf.clip_by_value(E_0_t, 1e-7, 1e9)
    log.debug(f"E_0_t:\n{E_0_t}")

    # TO DO: Documentation
    # log.debug(f"R_t outside scan:\n{R_t}")
    total_days = R_t.shape[0]

    # Initial susceptible population = total - infected
    S_initial = N - tf.reduce_sum(E_0_t, axis=-3)

    """ Calculate time evolution of new, daily infections
        as well as time evolution of susceptibles
        as lists
    """
    S_t = S_initial
    E_t = E_0_t  # has shape batch x time x coutry x age_group

    for i in range(l - 1, total_days):

        # Internal state
        f = S_t / N

        # These are the infections over which the convolution is done
        # log.debug(f"E_lastv: {E_t}")

        # Calc "infectious" people, weighted by serial_p (country x age_group)
        infectious = tf.einsum("...tca,...t->...ca", E_t[..., -1:-l:-1, :, :], g_p)

        # Calculate effective R_t [country,age_group] from Contact-Matrix C [country,age_group,age_group]
        R_sqrt = tf.math.sqrt(R_t[i])
        R_diag = tf.linalg.diag(R_sqrt)
        R_eff = tf.einsum(
            "...cij,...cik,...ckl->...cil", R_diag, C, R_diag
        )  # Effective growth number

        # log.debug(f"infectious: {infectious}")
        # log.debug(f"R_eff:\n{R_eff}")
        # log.debug(f"f:\n{f}")

        # Calculate new infections
        new_E = tf.einsum("...ci,...cij,...cj->...cj", infectious, R_eff, f)
        # log.debug(f"new_E:\n{new_E}")
        E_t = tf.concat([E_t, new_E[..., tf.newaxis, :, :]], axis=-3,)

        S_t = S_t - new_E

    return E_t  # batch_dims x time x country x age


def construct_delay_kernel(name, modelParams, loc, scale, length_kernel):
    r"""
        Constructs delay :math:`d` in hierarchical manner:

        .. math::

            \mu_c^d &\sim \text{LogNormal}\left(\mu=2.5,\sigma=0.1\right) \quad \forall c \\
            \sigma^d_c &\sim \\
            d_{c} &= \text{PDF-Gamma}(\mu^d_c,\sigma_d)

        Parameters
        ----------
        name:
            Name of the delay distribution
        modelParams: :py:class:`covid19_npis.ModelParams`
            Instance of modelParams, mainly used for number of age groups and
            number of countries.
        loc:
            Location of the hierarchical Lognormal distribution for the mean of the delay.
        scale:
            Theta parameter for now#
        length_kernel:
            Length of the delay kernel in days.

        Returns
        -------
        :
            Generator for gamma probability density function.
            |shape| batch, country, kernel(time)

        TODO
        ----
        Think about sigma distribution and how to parameterize it. Also implement that.

    """
    delay_mean = yield LogNormal(
        name="delay_mean",
        loc=np.log(loc, dtype="float32"),
        scale=0.1,
        event_stack=(
            modelParams.num_countries,
            1,
        ),  # country, time placeholder -> we do not want to do tf.expanddims
        conditionally_independent=True,
    )
    delay_theta = scale  # For now

    # Time tensor
    t = tf.range(
        0.1, length_kernel + 0.1, 1.0, dtype="float32"
    )  # The gamma function does not like 0!
    log.debug(f"time\n{t}")
    # Create gamma pdf from sampled mean and scale.
    delay = gamma(t, delay_mean / delay_theta, 1.0 / delay_theta)
    log.debug(f"delay\n{delay}")
    # Reshape delay i.e. add age group such that |shape| batch, time, country, age group
    delay = tf.stack([delay] * modelParams.num_age_groups, axis=-1)
    delay = tf.einsum("...cta->...cat", delay)

    return delay
