"""
README:

We inherit the distribution classes from pymc4 to add some additional
functionalities i.e. our DistributionAdditions class, which is mainly
used for plotting.

We dynamically create every distribution defined in __all__
this makes adding additional distribution later very easy.
They have to be called the same name as in pymc4 e.g.
'pm.LogNormal'->'LogNormal' !!
"""
import logging

log = logging.getLogger(__name__)
import pymc4 as pm
from pymc4 import Distribution
import types
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import tensorflow as tf

dists_to_modify = [
    "Deterministic",
    "LogNormal",
    "Normal",
    "LKJCholesky",
    "HalfCauchy",
    "StudentT",
    "Gamma",
    "HalfNormal",
    "Deterministic",
]


class DistributionAdditions:
    """
        Additional kwargs for every distribution.
        Every distribution class should inherit this class. 
    """

    def __init__(self, *args, **kwargs):

        if "shape_label" in kwargs:
            self.shape_label = kwargs.get("shape_label")
            event_ndim = len(self.shape_label)
            del kwargs["shape_label"]

        super().__init__(*args, **kwargs)


# ------------------------------------------------------------------------------ #
# Dynamically create classes
# ------------------------------------------------------------------------------ #
module = types.ModuleType("distributions")
for dist_name in dists_to_modify:
    # Get pymc4 class
    pmdist = getattr(pm, dist_name)
    # Add our own class to module scope
    vars()[dist_name] = types.new_class(dist_name, (DistributionAdditions, pmdist))

# ------------------------------------------------------------------------------ #
# If we want to add some special behaviour
# for a specific class we can do that here
# ------------------------------------------------------------------------------ #

# Example:
def other_init(self, *args, **kwargs):
    kwargs["validate_args"] = True
    super(self.__class__, self).__init__(*args, **kwargs)
    # print("This is a modified __init__")


"""
class Deterministic(Distribution):
    def __init__(self, name, value, **kwargs):
        with tf.name_scope(name):
            super().__init__(name, loc=value, **kwargs)

    @staticmethod
    def _init_distribution(conditions, **kwargs):
        loc = conditions["loc"]
        return tfp.distributions.Deterministic(loc=loc, **kwargs)


vars()["Deterministic"] = types.new_class(
    "Deterministic", (DistributionAdditions, Deterministic)
)
__all__ = dists_to_modify + ["Deterministic"]
"""

LogNormal.__init__ = other_init
Normal.__init__ = other_init
LKJCholesky.__init__ = other_init
HalfCauchy.__init__ = other_init
StudentT.__init__ = other_init
Gamma.__init__ = other_init


# Own implementation of  Multivariate Student's t-distribution can be removed as soon as the pymc4
# pull reqeust is merged
class MvStudentT(pm.distributions.distribution.ContinuousDistribution):
    r"""
    Multivariate Student's t-distribution

    .. math::

        f(x) =
            \frac{(1 + ||y||^{2} / \nu)^{-0.5 (\nu + k)}}
            {Z}, \\
        y = \Sigma^{-1} (x - \mu),\\
        Z = \frac{|\det(\Sigma)| \sqrt(\nu \pi)^{k}} \Gamma(0.5 \nu)} {\Gamma(0.5 \nu)+k}


    ========  ==========================
    Support   :math:`x \in \mathbb{R}^k`
    Mean      :math:`\mu` if :math:`\nu>1`
    Variance  :math:`\frac{\nu}{\nu-2}\Sigma` if :math:`\nu > 2`
    ========  ==========================


    Parameters
    ----------
    df : positive scalar
        The degrees of freedom :math:`\nu.`
    loc : array_like
        Vector of means :math:`\mu.`
    scale : 
        Lower triangular matrix, such that scale @ scale.T is positive
        semi-definite :math:`\Sigma.`


    """

    def __init__(self, name, df, loc, scale, **kwargs):
        super().__init__(name, df=df, loc=loc, scale=scale, **kwargs)

    @staticmethod
    def _init_distribution(conditions, **kwargs):
        df, loc, scale = conditions["df"], conditions["loc"], conditions["scale"]
        return tfd.MultivariateStudentTLinearOperator(
            df=df, loc=loc, scale=scale, **kwargs
        )
