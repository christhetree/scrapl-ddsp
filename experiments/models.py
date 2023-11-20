import logging
import os
from typing import Optional, List, Tuple

import torch as tr
from torch import Tensor as T
from torch import nn
from torchaudio.transforms import MelSpectrogram, FrequencyMasking, TimeMasking

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class Spectral2DCNN(nn.Module):
    def __init__(self,
                 in_ch: int = 1,
                 n_samples: int = 88200,
                 sr: float = 44100,
                 n_fft: int = 1024,
                 hop_len: int = 256,
                 n_mels: int = 256,
                 kernel_size: Tuple[int, int] = (5, 13),
                 out_channels: Optional[List[int]] = None,
                 bin_dilations: Optional[List[int]] = None,
                 temp_dilations: Optional[List[int]] = None,
                 pool_size: Tuple[int, int] = (3, 1),
                 latent_dim: int = 1,
                 freq_mask_amount: float = 0.0,
                 time_mask_amount: float = 0.0,
                 use_ln: bool = True,
                 eps: float = 1e-7) -> None:
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.n_mels = n_mels
        self.kernel_size = kernel_size
        assert pool_size[1] == 1
        self.pool_size = pool_size
        self.latent_dim = latent_dim
        self.freq_mask_amount = freq_mask_amount
        self.time_mask_amount = time_mask_amount
        self.use_ln = use_ln
        self.eps = eps
        if out_channels is None:
            out_channels = [64] * 5
        self.out_channels = out_channels
        if bin_dilations is None:
            bin_dilations = [1] * len(out_channels)
        self.bin_dilations = bin_dilations
        if temp_dilations is None:
            temp_dilations = [2 ** idx for idx in range(len(out_channels))]
        self.temp_dilations = temp_dilations
        assert len(out_channels) == len(bin_dilations) == len(temp_dilations)

        self.spectrogram = MelSpectrogram(sample_rate=int(sr),
                                          n_fft=n_fft,
                                          hop_length=hop_len,
                                          normalized=False,
                                          n_mels=n_mels,
                                          center=True)
        n_bins = n_mels
        n_frames = n_samples // hop_len + 1
        temporal_dims = [n_frames] * len(out_channels)

        self.freq_masking = FrequencyMasking(freq_mask_param=int(freq_mask_amount * n_bins))
        self.time_masking = TimeMasking(time_mask_param=int(time_mask_amount * n_frames))

        layers = []
        for out_ch, b_dil, t_dil, temp_dim in zip(out_channels, bin_dilations, temp_dilations, temporal_dims):
            if use_ln:
                layers.append(nn.LayerNorm([n_bins, temp_dim], elementwise_affine=False))
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride=(1, 1), dilation=(b_dil, t_dil), padding="same"))
            layers.append(nn.MaxPool2d(kernel_size=pool_size))
            layers.append(nn.PReLU(num_parameters=out_ch))
            in_ch = out_ch
            n_bins = n_bins // pool_size[0]
        self.cnn = nn.Sequential(*layers)

        # TODO(cm): change from regression to classification
        self.output = nn.Conv1d(out_channels[-1], self.latent_dim, kernel_size=(1,))

    def forward(self, x: T) -> (T, T):
        assert x.ndim == 3
        x = self.spectrogram(x)

        if self.training:
            if self.freq_mask_amount > 0:
                x = self.freq_masking(x)
            if self.time_mask_amount > 0:
                x = self.time_masking(x)

        x = tr.clip(x, min=self.eps)
        x = tr.log(x)
        x = self.cnn(x)
        x = tr.mean(x, dim=-2)
        latent = x

        x = self.output(x)
        x = tr.sigmoid(x)
        return x, latent