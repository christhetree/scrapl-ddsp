from kymatio.frontend.jax_frontend import ScatteringJax
from kymatio.scattering1d.frontend.base_frontend import ScatteringBase1D
from kymatio.scattering1d.frontend.jax_frontend import TimeFrequencyScatteringJax
from .base_frontend import TimeFrequencyScraplBase


class TimeFrequencyScraplJax(TimeFrequencyScraplBase, TimeFrequencyScatteringJax):
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
        backend='jax'
    ):
        ScatteringJax.__init__(self)
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


__all__ = ["TimeFrequencyScraplJax"]