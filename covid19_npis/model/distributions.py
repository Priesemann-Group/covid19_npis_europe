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

import pymc4 as pm

__all__ = [
    "LogNormal",
    "Normal",
    "LKJCholesky",
    "HalfCauchy",
    "StudentT",
    "Gamma",
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
            del kwargs["shape_label"]

        super().__init__(*args, **kwargs)


# ------------------------------------------------------------------------------ #
# Dynamically create classes
# ------------------------------------------------------------------------------ #

for dist_name in __all__:
    # Get pymc4 class
    pmdist = getattr(pm, dist_name)
    # Create constructor/init function
    def __constructor__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)

    # Add our own class to module scope
    vars()[dist_name] = type(
        dist_name, (DistributionAdditions, pmdist), {"__init__": __constructor__}
    )

# ------------------------------------------------------------------------------ #
# If we want to add some special behaviour
# for a specific class we can do that here
# ------------------------------------------------------------------------------ #

# Example:
def other_init(self, *args, **kwargs):
    super(self.__class__, self).__init__(*args, **kwargs)
    # print("This is a modified __init__")


LogNormal.__init__ = other_init
