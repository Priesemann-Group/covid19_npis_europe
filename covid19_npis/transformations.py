from pymc4.distributions.transforms import BackwardTransform, JacobianPreference
from tensorflow_probability import bijectors as tfb


class Normal(BackwardTransform):
    name = "Normal"
    JacobianPreference = JacobianPreference.Backward

    def __init__(self, shift=None, scale=None, **kwargs):
        if scale is None:
            scaling = tfb.Identity()
        else:
            scaling = tfb.Scale(scale)
        if shift is None:
            shifting = tfb.Identity()
        else:
            shifting = tfb.Shift(shift)
        transform = tfb.Chain([scaling, shifting])
        super().__init__(transform, **kwargs)


class SoftPlus(BackwardTransform):
    name = "SoftPlus"
    JacobianPreference = JacobianPreference.Backward

    def __init__(self, scale=None, hinge_softness=None, **kwargs):
        if scale is None:
            scaling = tfb.Scale(10)
        else:
            scaling = tfb.Scale(scale=scale)
        transform = tfb.Chain(
            [tfb.Shift(1e-7), scaling, tfb.Softplus(hinge_softness=hinge_softness)]
        )
        super().__init__(transform, **kwargs)


class SoftPlus_SinhArcsinh(BackwardTransform):
    name = "SoftPlus-SinhTanh"
    JacobianPreference = JacobianPreference.Backward

    def __init__(self, scale=None, skewness=None, tailweight=None, **kwargs):
        if scale is None:
            scaling = tfb.Identity()
        else:
            scaling = tfb.Scale(scale)
        transform = tfb.Chain(
            [scaling, tfb.Softplus(), tfb.SinhArcsinh(skewness, tailweight,),]
        )
        super().__init__(transform, **kwargs)


class Exp_SinhArcsinh(BackwardTransform):
    name = "Exp-SinhTanh"
    JacobianPreference = JacobianPreference.Backward

    def __init__(self, scale1=0.5, scale2=1e4, skewness=-10, tailweight=0.02, **kwargs):

        scaling1 = tfb.Scale(scale1)
        scaling2 = tfb.Scale(scale2)

        transform = tfb.Chain(
            [scaling2, tfb.Exp(), scaling1, tfb.SinhArcsinh(skewness, tailweight,),]
        )
        super().__init__(transform, **kwargs)


class CorrelationCholesky(BackwardTransform):
    name = "CorrelationCholesky"
    JacobianPreference = JacobianPreference.Backward

    def __init__(self, **kwargs):
        transform = tfb.CorrelationCholesky()
        super().__init__(transform, **kwargs)


class LogScale(BackwardTransform):
    name = "LogScale"
    JacobianPreference = JacobianPreference.Backward

    def __init__(self, scale=None, **kwargs):
        if scale is None:
            scaling = tfb.Identity()
        else:
            scaling = tfb.Scale(scale)
        transform = tfb.Chain([scaling, tfb.Log()])
        super().__init__(transform, **kwargs)
