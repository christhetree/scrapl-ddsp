import logging
import os
from typing import Optional, List, Tuple

import torch as tr
from torch import Tensor as T
from torch import nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class Spectral2DCNN(nn.Module):
    def __init__(self,
                 n_bins: int,
                 n_frames: int,
                 in_ch: int = 1,
                 kernel_size: Tuple[int, int] = (3, 3),
                 out_channels: Optional[List[int]] = None,
                 bin_dilations: Optional[List[int]] = None,
                 temp_dilations: Optional[List[int]] = None,
                 pool_size: Tuple[int, int] = (2, 2),
                 latent_dim: int = 32,
                 use_ln: bool = True) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.n_frames = n_frames
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.latent_dim = latent_dim
        self.use_ln = use_ln

        if out_channels is None:
            out_channels = [64] * 5
        self.out_channels = out_channels
        if bin_dilations is None:
            bin_dilations = [1] * len(out_channels)
        self.bin_dilations = bin_dilations
        if temp_dilations is None:
            temp_dilations = [1] * len(out_channels)
        self.temp_dilations = temp_dilations
        assert len(out_channels) == len(bin_dilations) == len(temp_dilations)

        layers = []
        for out_ch, b_dil, t_dil in zip(out_channels, bin_dilations, temp_dilations):
            if use_ln:
                layers.append(nn.LayerNorm([n_bins, n_frames], elementwise_affine=False))
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride=(1, 1), dilation=(b_dil, t_dil), padding="same"))
            layers.append(nn.MaxPool2d(kernel_size=pool_size))
            layers.append(nn.PReLU(num_parameters=out_ch))
            in_ch = out_ch
            n_bins = n_bins // pool_size[0]
            n_frames = n_frames // pool_size[1]
        self.cnn = nn.Sequential(*layers)

        self.fc = nn.Linear(out_channels[-1], latent_dim)
        self.fc_prelu = nn.PReLU(num_parameters=latent_dim)
        self.out_density = nn.Linear(latent_dim, 1)
        self.out_slope = nn.Linear(latent_dim, 1)

    def forward(self, x: T) -> (T, T):
        assert x.ndim == 3
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = tr.mean(x, dim=(2, 3))
        x = self.fc(x)
        x = self.fc_prelu(x)
        latent = x

        density_hat = self.out_density(latent)
        density_hat = tr.sigmoid(density_hat).squeeze(1)
        slope_hat = self.out_slope(latent)
        slope_hat = tr.tanh(slope_hat).squeeze(1)
        return density_hat, slope_hat
