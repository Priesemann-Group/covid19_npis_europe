from pymc4.distributions.transforms import BackwardTransform, JacobianPreference
from tensorflow_probability import bijectors as tfb



class Normal(BackwardTransform):
    name = "Normal"
    JacobianPreference = JacobianPreference.Backward

    def __init__(self, shift=None, scale=None,  **kwargs):
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

    def __init__(self, scale=None, **kwargs):
        if scale is None:
            scaling = tfb.Identity()
        else:
            scaling = tfb.Scale(scale)
        transform = tfb.Chain([scaling, tfb.Softplus()])
        super().__init__(transform, **kwargs)




class SoftPlus_SinhArcsinh(BackwardTransform):
    name = "SoftPlus_SinhTanh"
    JacobianPreference = JacobianPreference.Backward

    def __init__(
        self, scale=None, skewness=None, tailweight=None, **kwargs
    ):
        if scale is None:
            scaling = tfb.Identity()
        else:
            scaling = tfb.Scale(scale)
        transform = tfb.Chain(
            [scaling, tfb.Softplus(), tfb.SinhArcsinh(skewness, tailweight,),]
        )
        super().__init__(transform, **kwargs)




class Deterministic(BackwardTransform):
    name = "SoftPlus_SinhTanh"
    JacobianPreference = JacobianPreference.Backward

    def __init__(
        self, scale=None, skewness=None, tailweight=None, **kwargs
    ):
        if scale is None:
            scaling = tfb.Identity()
        else:
            scaling = tfb.Scale(scale)
        transform = tfb.Chain(
            [scaling, tfb.Softplus(), tfb.SinhArcsinh(skewness, tailweight,),]
        )
        super().__init__(transform, **kwargs)




class CorrelationCholesky(BackwardTransform):
    name = "CorrelationCholesky"
    JacobianPreference = JacobianPreference.Backward

    def __init__(self, **kwargs):
        transform = tfb.CorrelationCholesky()
        super().__init__(transform, **kwargs)


