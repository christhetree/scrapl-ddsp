from kymatio.frontend.torch_frontend import ScatteringTorch
from kymatio.scattering1d.frontend.base_frontend import ScatteringBase1D
from kymatio.scattering1d.frontend.torch_frontend import TimeFrequencyScatteringTorch
from .base_frontend import TimeFrequencyScraplBase


class TimeFrequencyScraplTorch(TimeFrequencyScraplBase, TimeFrequencyScatteringTorch):
    def __init__(
        self,
        *,
        J,
        J_fr,
        shape,
        Q,
        T=None,
        stride=None,
        Q_fr=1,
        F=None,
        stride_fr=None,
        out_type='array',
        format='joint',
        backend='torch'
    ):
        ScatteringTorch.__init__(self)
        TimeFrequencyScraplBase.__init__(
            self,
            J=J,
            J_fr=J_fr,
            shape=shape,
            Q=Q,
            T=T,
            stride=stride,
            Q_fr=Q_fr,
            F=F,
            stride_fr=stride_fr,
            out_type=out_type,
            format=format,
            backend=backend,
        )
        ScatteringBase1D._instantiate_backend(self, "kymatio.scattering1d.backend.")
        TimeFrequencyScraplBase.build(self)
        TimeFrequencyScraplBase.create_filters(self)
        self.register_filters()

    def scattering_singlepath(self, x, n2, n_fr):
        self.load_filters()
        return super().scattering_singlepath(x, n2, n_fr)


__all__ = ["TimeFrequencyScraplTorch"]