import tensorflow as tf
import logging
import pymc4 as pm
from .utils import gamma
import tensorflow_probability as tfp


log = logging.getLogger(__name__)


def _fsigmoid(t, l, d):
    return 1.0 / (1.0 + tf.exp(-4.0 / l * (t - d)))


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

    def __init__(self, name, date_loc, date_scale, length, gamma_max):
        self.name = name
        self.prior_date_loc = date_loc
        self.prior_date_scale = date_scale
        self.length = length
        self.gamma_max = gamma_max

    @property
    def date(self):
        r"""
        Returns pymc4 generator for the date :math:`d`, i.e. a normal distribution. The priors
        are set at init of the object.
        """
        yield pm.Normal(
            self.name, self.prior_date_loc, self.prior_date_scale
        )  # Test if it works like this or if we need yield statement here already.

    def gamma_t(self, t):
        """
        Returns gamma value at t with given length :math:`l`. The length :math:`l` should be
        passed from the intervention class.
        """
        yield _fsigmoid(t, self.length, self.date) * self.gamma_max


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

    @property
    def length(self):
        r"""
        Returns pymc4 generator for the length :math:`l`, i.e. a normal distribution. The priors
        are set at init of the object.
        """
        yield pm.Normal(
            self.name + "_length", self.prior_length_loc, self.prior_length_scale
        )

    @property
    def alpha(self):
        r"""
        Returns pymc4 generator for the effetivity :math:`\alpha`, i.e. a normal distribution. The priors
        are set at init of the object.
        """
        yield pm.Normal(
            self.name + "_alpha", self.prior_alpha_loc, self.prior_alpha_scale
        )

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

    def gamma_t(self, t):
        """
        Returns the gamma (strength) value of an intervention at the time t.

        Parameter
        ---------
        t: number
            Time
        """
        _sum = 0.0
        for cp in self.change_points:
            _sum += cp.gamma_t(t)
        return _sum


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
    ret = []
    for c, C in enumerate(modelParams.interventions):  # Country
        interventions = []
        for i in C:  # interventions
            cps = []
            for cp in C[i]:  # changepoints
                cps.append(
                    Change_point(
                        name=C[i][cp]["name"],
                        date_loc=C[i][cp]["date"],
                        date_scale=2,
                        gamma_max=C[i][cp]["gamma_max"],
                    )
                )
            interventions.append(
                Intervention(
                    name=i,
                    length_loc=C[i][cp]["length"],
                    length_scale=2.5,
                    alpha_loc=C[i][cp]["alpha"],
                    alpha_scale=0.5,
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
        |shape| batch, time, country, age group
    """

    # Create tensorflow R_t for now hardcoded to 50 timesteps
    R_t = tf.stack([R_0] * 50)

    def _sum_interventions(t):
        _sum = 0.0
        for i in Interventions:
            _sum += i.alpha * i.gamma_t(t)
        return R_0 * tf.exp(_sum)

    R_t = _sum_interventions(tf.range(0, 50, dtype="float32"))

    # That could work like  that im not sure tho. -> has to be tested
    # tf.map_fn(_sum_interventions, tf.range(0, 50, delta=1, dtype="float32"))
    R_t = tf.stack([R_0] * 50)
    return R_t
