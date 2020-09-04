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

__all__ = [
    "LogNormal",
    "Normal",
    "LKJCholesky",
    "HalfCauchy",
    "StudentT",
    "Gamma",
    "Deterministic",
    "HalfNormal",
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

        if "event_stack" in kwargs:
            self.shape = kwargs.get("event_stack")

        if "shape" in kwargs:
            self.shape = kwargs.get("shape")
            del kwargs["shape"]

        if "transformation" in kwargs and "shape" in kwargs:
            if kwargs["transformation"]._reinterpreted_batch_ndims != len(self.shape):
                log.warning(
                    f"Automatically setting reinterpreted_batch_ndims to length of event_stack in transofrmation for {self.name}"
                )
                kwargs["transformation"]._reinterpreted_batch_ndims = len(self.shape)

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
