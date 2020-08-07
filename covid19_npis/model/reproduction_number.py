import tensorflow as tf
import logging
import pymc4 as pm
from .utils import gamma
import tensorflow_probability as tfp
import numpy as np

log = logging.getLogger(__name__)


num_age_groups = 4
num_countries = 2


def _fsigmoid(t, l, d):
    # Prep dimensions
    d = tf.expand_dims(d, axis=-1)
    l = tf.expand_dims(l, axis=-1)

    inside_exp_1 = -4.0 / l
    log.info("hi")
    inside_exp_2 = -d + t
    log.info(f"-4/l\n{inside_exp_1.shape}")
    log.info(f"t-d\n{inside_exp_2.shape}")
    return 1.0 / (1.0 + tf.exp(inside_exp_1 * inside_exp_2))


class Change_point(object):
    """
        Change point class. Should contain date d, gamma_max and function to return gamma at point t.

        Parameters
        ----------
        name: str
            Name of the change point, get passed to the pymc4 distribution for the date.

        date_loc : number
            Prior for the location of the date of the change point.

        date_scale : number
            Prior for the scale of the date of the change point.

        gamma_max : number
            Maximum gamma value for the change point, i.e the value the logistic function gamma_t converges to. [-1,1]
    """

    def __init__(self, name, date_loc, date_scale, gamma_max):
        self.name = name
        self.prior_date_loc = date_loc
        self.prior_date_scale = date_scale
        self.gamma_max = gamma_max

        log.info(f"{self.prior_date_scale},{self.prior_date_loc}")

        self._d = pm.Normal(
            self.name,
            self.prior_date_loc,
            self.prior_date_scale,
            conditionally_independent=True,
            batch_stack=num_age_groups,
        )  # Test if it works like this or if we need yield statement here already.

    @property
    def date(self):
        r"""
        Returns pymc4 generator for the date :math:`d`, i.e. a normal distribution. The priors
        are set at init of the object.
        """

        return (yield self._d)

    def gamma_t(self, t, l):
        """
        Returns gamma value at t with given length :math:`l`. The length :math:`l` should be
        passed from the intervention class.
        """
        log.info("gamma_t_change_point")
        d = yield self.date
        sigmoid = _fsigmoid(t, l, d)
        return sigmoid * self.gamma_max


class Intervention(object):
    """
        Intervention class, contains every variable that is only intervention dependent
        and the change point for the intervention.

        Parameters
        ----------
        name: str
            Name of the intervention, get passed to the pymc4 functions with suffix '_length' or
            '_alpha'.

        length_loc:
            Prior for the location of the length. Set to one overarching value for all
            change points.

        length_scale:
            Prior for the scale of the length. Set to one overarching value for all
            change points.

        alpha_loc:
            Prior for the location of the effectivity for the intervention.

        alpha_scale:
            Prior for the scale of the effectivity for the intervention.

        change_points: dict, optional
            Constructs Change_points object from the dict TODO: think about that a bit more.
            |default| :code:`None`
    """

    def __init__(
        self, name, length_loc, length_scale, alpha_loc, alpha_scale, change_points=None
    ):
        self.name = name
        log.info(name)
        # Distributions
        self.prior_length_loc = length_loc
        self.prior_length_scale = length_scale
        self.prior_alpha_loc = alpha_loc
        self.prior_alpha_scale = alpha_scale

        self.change_points = []
        # Add to change points
        if change_points is not None:
            for change_point in change_points:
                self.add_change_point(change_point)

        # Init distributions

        self._l = pm.LogNormal(
            self.name + "_length",
            np.log(self.prior_length_loc).astype("float32"),
            self.prior_length_scale,
            conditionally_independent=True,
            batch_stack=num_age_groups,
        )

        self._alpha = pm.Normal(
            self.name + "_alpha",
            self.prior_alpha_loc,
            self.prior_alpha_scale,
            conditionally_independent=True,
            batch_stack=(1),
        )

    @property
    def length(self):
        r"""
        Returns pymc4 generator for the length :math:`l`, i.e. a normal distribution. The priors
        are set at init of the object.
        """
        return (yield self._l)

    @property
    def alpha(self):
        r"""
        Returns pymc4 generator for the effetivity :math:`\alpha`, i.e. a normal distribution. The priors
        are set at init of the object.
        """

        return (yield self._alpha)

    def add_change_point(self, change_point):
        """
        Adds a change point to the intervention by dictionary or by passing the class
        itself.
        """
        if isinstance(change_point, Change_point):
            self.change_points.append(change_point)
        elif isinstance(change_point, dict):
            assert "name" in change_point, f"Change point dict must have 'name' key"
            assert (
                "date_loc" in change_point
            ), f"Change point dict must have 'date_loc' key"
            assert (
                "date_scale" in change_point
            ), f"Change point dict must have 'date_scale' key"
            assert (
                "gamma_max" in change_point
            ), f"Change point dict must have 'gamma_max' key"
            self.change_points.append(
                Change_point(
                    change_point["name"],
                    change_point["date_loc"],
                    change_point["date_scale"],
                    self.length,
                    change_point["gamma_max"],
                )
            )

    def gamma_t(self, t):  # Intervention
        """
        Returns the gamma (strength) value of an intervention at the time t.
        Sum over all change points

        Parameter
        ---------
        t: number
            Time
        """

        log.info("gamma_t_intervetion")
        l = yield self.length

        # Sum over change points
        _sum = []
        for cp in self.change_points:
            gamma_cp = yield cp.gamma_t(t, l)
            _sum.append(gamma_cp)

        ret = tf.reduce_sum(_sum, axis=0)

        return ret


def create_interventions(modelParams):
    """
    Returns a list of interventions :py:func:`covid19_npis.reproduction_number.Intervention` from
    a change point dict.

    Parameter
    ---------
    change_points : dict
        Dict housing every parameter for the interventions, i.e.
        :math:`l_{i}`
        :math:`d_{i,n}`
        :math:`alpha_{i}`
        :math:`i` being interventions and :math:`n` being a change point.

    Return
    ------
    :
        Interventions array like
        |shape| country, interventions
    """
    log.info("create_interventions")
    ret = []
    for c, C in enumerate(modelParams.interventions):  # Country
        interventions = []
        for i in C:  # interventions
            cps = []
            for cp in C[i]:  # changepoints
                cps.append(
                    Change_point(
                        name=f"{c}_{i}_{cp}",
                        date_loc=C[i][cp]["date"],
                        date_scale=2,
                        gamma_max=C[i][cp]["gamma_max"],
                    )
                )
            interventions.append(
                Intervention(
                    name=f"{c}_{i}",
                    length_loc=C[i][cp]["length"],
                    length_scale=2.5,
                    alpha_loc=C[i][cp]["alpha"],
                    alpha_scale=0.5,
                    change_points=cps,
                )
            )
        ret.append(interventions)
    return ret


def construct_R_t(R_0, Interventions):
    """
    Constructs the time dependent reproduction number :math:`R(t)` for every country and age group.


    Parameter
    ---------

    R_0:
        |shape| batch, country, age group

    Interventions: array like covid19_npis.reproduction_number.Intervention
        |shape| country

    Return
    ------
    R_t:
        Reproduction number matrix.
        |shape| time, batch, country, age group
    """

    # Create tensorflow R_t for now hardcoded to 50 timesteps
    t = tf.range(0, 50, dtype="float32")

    """ We want to create a time dependent R_t for each country and age group
        We iterate over country and interventions.
    """

    exp_to_multi = []
    for c in range(num_countries):
        _sum = []
        for i in Interventions[c]:
            # Idee:
            _alpha = yield i.alpha
            gamma_t = yield i.gamma_t(t)
            log.info(f"gamma_t\n{gamma_t.shape}")
            log.info(f"alpha\n{_alpha.shape}")
            _sum.append(tf.einsum("...ai,...j->...ai", gamma_t, _alpha))

        # We sum over all interventions in a country and append to list for countries
        exp_to_multi.append(tf.exp(tf.reduce_sum(_sum, axis=0)))

    exp_to_multi = tf.convert_to_tensor(exp_to_multi)

    if len(exp_to_multi.shape) == 4:
        # before |shape| country, batch, agegroup, time
        exp_to_multi = tf.transpose(exp_to_multi, perm=(1, 0, 2, 3))
        # after |shape| batch, country, agegroup, time

    """
    Multiplicate R_0 with the exponent i.e.
        R(t) = R_0 * exp(sum_i(gamm_i(t)*alpha_i))
        R_0: |shape| batch, country, agegroup
        exp: |shape| batch, country, agegroup, time
    """
    log.info(f"country exponential function:\n{exp_to_multi.shape}")
    log.info(f"R_0:\n{R_0.shape}")

    R_t = tf.einsum(
        "...ca,...cat->t...ca", R_0, exp_to_multi
    )  # Reshape to |shape| time, batch, country, age group here
    log.info(f"R_t:\n{R_t.shape}")

    return R_t
