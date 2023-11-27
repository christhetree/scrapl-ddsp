from ..core.timefrequency_scattering import jtfs_singlepath, jtfs_singlepath_average_and_format
from kymatio.scattering1d.frontend.base_frontend import TimeFrequencyScatteringBase


class TimeFrequencyScraplBase(TimeFrequencyScatteringBase):
    def __init__(self, **kwargs):
        kwargs['out_type'] = 'array'
        kwargs['format'] = 'joint'
        super(TimeFrequencyScraplBase, self).__init__(**kwargs)
    
    def scattering_singlepath(self, x, n2, n_fr):
        TimeFrequencyScatteringBase._check_runtime_args(self)
        TimeFrequencyScatteringBase._check_input(self, x)

        x_shape = self.backend.shape(x)
        batch_shape, signal_shape = x_shape[:-1], x_shape[-1:]
        x = self.backend.reshape_input(x, signal_shape)
        U_0 = self.backend.pad(
            x, pad_left=self.pad_left, pad_right=self.pad_right)

        filters = [self.phi_f, self.psi1_f, self.psi2_f]
        path = jtfs_singlepath(U_0, self.backend,
            filters, self.log2_stride, (self.average=='local'),
            self.filters_fr, self.log2_stride_fr, (self.average_fr=='local'),
            n2, n_fr)
                
        path = jtfs_singlepath_average_and_format(path, self.backend,
            self.phi_f, self.log2_stride, self.average,
            self.filters_fr[0], self.log2_stride_fr, self.average_fr)

        # TODO(cm): causing problems
        # Unpad.
        # if self.average != 'global':
        #     res = max(path['j'][-1], 0)
        #     path['coef'] = self.backend.unpad(
        #         path['coef'], self.ind_start[res], self.ind_end[res])

        # # Reshape path to batch shape.
        # path['coef'] = self.backend.reshape_output(path['coef'],
        #     batch_shape, n_kept_dims=(1 + (self.format=='joint')))

        return path


__all__ = ['TimeFrequencyScraplBase']
