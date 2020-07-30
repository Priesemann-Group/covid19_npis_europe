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
        
        TODO
        ----
        - Documentation
        - implement name
    """

    def __init__(self, date_loc, date_scale, gamma_max):
        self.prior_date_loc = date_loc
        self.prior_date_scale = date_scale
        self.gamma_max = gamma_max

    @property
    def date(self):
        r"""
        Returns pymc4 generator for the date :math:`d`, i.e. a normal distribution. The priors
        are set at init of the object.
        """
        return pm.Normal(self.name, self.prior_date_loc, self.prior_date_scale)

    def gamma_t(self, t, l):
        """
            Returns gamma value at t with parameters l and d.
        """
        return _fsigmoid(t, l, self.date) * self.gamma_max


class Intervention(object):
    """
        Intervention class, contains every variable that is only intervention dependent
        and the change point for the intervention.

        Parameters
        ----------
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
            |default| None


        TODO
        ----
        - implement name
        - method to add an change point
    """

    def __init__(
        self, length_loc, length_scale, alpha_loc, alpha_scale, change_points=None
    ):

        self.prior_length_loc = length_loc
        self.prior_length_scale = length_scale

        self.prior_alpha_loc = alpha_loc
        self.prior_alpha_scale = alpha_scale

        # TODO change point construct logic and name

    @property
    def length(self):
        r"""
        Returns pymc4 generator for the length :math:`l`, i.e. a normal distribution. The priors
        are set at init of the object.
        """
        return pm.Normal(
            self.name + "_length", self.prior_length_loc, self.prior_length_scale
        )

    @property
    def alpha(self):
        r"""
        Returns pymc4 generator for the effetivity :math:`\alpha`, i.e. a normal distribution. The priors
        are set at init of the object.
        """
        return pm.Normal(
            self.name + "_alpha", self.prior_alpha_loc, self.prior_alpha_scale
        )

    def gamma_t(self, t):
        """
        Returns the gamma (strength) value of an intervention at the time t.

        Parameter
        ---------
        t: number
            Time
        """
        _sum = 0
        for cp in self.change_points:
            _sum += cp.gamma_t(t, self.length)
        return _sum


def create_interventions():
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
    """


def construct_R_t(R_0, Interventions):
    """
    Constructs the time dependent reproduction number :math:`R(t)` for every country and age group.


    Parameter
    ---------

    R_0:
        |shape| batch, countries, age group

    Interventions: array like covid19_npis.reproduction_number.Intervention
        
    Return
    ------
    R_t:
        Reproduction number matrix.
        |shape| time, batch_dims, country, age_group
    """
