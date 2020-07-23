from pymc4.distributions.transforms import Transform, JacobianPreference
from tensorflow_probability import bijectors as tfb


class Log(Transform):
    name = "log"
    JacobianPreference = JacobianPreference.Backward

    def __init__(self, reinterpreted_batch_ndims=0):
        # NOTE: We actually need the inverse to match PyMC3, do we?
        self._transform = tfb.Exp()
        self._reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def forward(self, x):
        return self._transform.inverse(x)

    def inverse(self, z):
        return self._transform.forward(z)

    def forward_log_det_jacobian(self, x):
        return self._transform.inverse_log_det_jacobian(
            x, self._transform.inverse_min_event_ndims + self._reinterpreted_batch_ndims
        )

    def inverse_log_det_jacobian(self, z):
        return self._transform.forward_log_det_jacobian(
            z, self._transform.forward_min_event_ndims + self._reinterpreted_batch_ndims
        )


class SoftPlus(Transform):
    name = "SoftPlus"
    JacobianPreference = JacobianPreference.Backward

    def __init__(self, scale=None, reinterpreted_batch_ndims=0):
        if scale is None:
            scaling = tfb.Identity()
        else:
            scaling = tfb.Scale(scale)
        self._transform = tfb.Chain([scaling, tfb.Softplus()])
        self._reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def forward(self, x):
        return self._transform.inverse(x)

    def inverse(self, z):
        return self._transform.forward(z)

    def forward_log_det_jacobian(self, x):
        return self._transform.inverse_log_det_jacobian(
            x, self._transform.inverse_min_event_ndims + self._reinterpreted_batch_ndims
        )

    def inverse_log_det_jacobian(self, z):
        return self._transform.forward_log_det_jacobian(
            z, self._transform.forward_min_event_ndims + self._reinterpreted_batch_ndims
        )


class SoftPlus_SinhArcsinh(Transform):
    name = "SoftPlus_SinhTanh"
    JacobianPreference = JacobianPreference.Backward

    def __init__(
        self, scale=None, skewness=None, tailweight=None, reinterpreted_batch_ndims=0
    ):
        if scale is None:
            scaling = tfb.Identity()
        else:
            scaling = tfb.Scale(scale)
        self._transform = tfb.Chain(
            [scaling, tfb.Softplus(), tfb.SinhArcsinh(skewness, tailweight,),]
        )
        self._reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def forward(self, x):
        return self._transform.inverse(x)

    def inverse(self, z):
        return self._transform.forward(z)

    def forward_log_det_jacobian(self, x):
        return self._transform.inverse_log_det_jacobian(
            x, self._transform.inverse_min_event_ndims + self._reinterpreted_batch_ndims
        )

    def inverse_log_det_jacobian(self, z):
        return self._transform.forward_log_det_jacobian(
            z, self._transform.forward_min_event_ndims + self._reinterpreted_batch_ndims
        )


class Deterministic(Transform):
    name = "SoftPlus_SinhTanh"
    JacobianPreference = JacobianPreference.Backward

    def __init__(
        self, scale=None, skewness=None, tailweight=None, reinterpreted_batch_ndims=0
    ):
        if scale is None:
            scaling = tfb.Identity()
        else:
            scaling = tfb.Scale(scale)
        self._transform = tfb.Chain(
            [scaling, tfb.Softplus(), tfb.SinhArcsinh(skewness, tailweight,),]
        )
        self._reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def forward(self, x):
        return self._transform.inverse(x)

    def inverse(self, z):
        return self._transform.forward(z)

    def forward_log_det_jacobian(self, x):
        return self._transform.inverse_log_det_jacobian(
            x, self._transform.inverse_min_event_ndims + self._reinterpreted_batch_ndims
        )

    def inverse_log_det_jacobian(self, z):
        return self._transform.forward_log_det_jacobian(
            z, self._transform.forward_min_event_ndims + self._reinterpreted_batch_ndims
        )
