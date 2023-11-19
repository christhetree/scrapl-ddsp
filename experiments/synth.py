import logging
import os
from typing import Optional

import numpy as np
import torch as tr
from torch import Tensor as T, nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class ChirpTextureSynth(nn.Module):
    def __init__(self,
                 sr: float,
                 n_samples: int,
                 n_grains: int,
                 grain_n_samples: int,
                 f0_min_hz: float,
                 f0_max_hz: float,
                 Q: int,
                 hop_len: int,
                 seed: Optional[int] = None):
        super().__init__()
        assert n_samples > grain_n_samples
        assert f0_max_hz > f0_min_hz
        self.sr = sr
        self.n_samples = n_samples
        self.n_grains = n_grains
        self.grain_n_samples = grain_n_samples
        self.f0_min_hz = f0_min_hz
        self.f0_max_hz = f0_max_hz
        self.Q = Q
        self.hop_len = hop_len

        self.log2_f0_min = tr.log2(tr.tensor(f0_min_hz))
        self.log2_f0_max = tr.log2(tr.tensor(f0_max_hz))
        self.const_log2 = tr.log2(tr.tensor(2.0))
        self.grain_dur_s = grain_n_samples / sr
        support = tr.arange(self.grain_n_samples) / self.sr - (self.grain_dur_s / 2)
        self.grain_support = support.repeat(self.n_grains, 1)
        self.rand_gen = tr.Generator()
        if seed is not None:
            self.rand_gen.manual_seed(seed)

    def sample_onsets(self) -> T:
        # TODO(cm): add support for edge padding
        onsets = tr.rand(self.n_grains, generator=self.rand_gen) * (self.n_samples - self.grain_n_samples)
        onsets = onsets.long()
        return onsets

    def sample_f0_freqs(self) -> T:
        log2_f0_freqs = tr.rand(self.n_grains, generator=self.rand_gen) * (self.log2_f0_max - self.log2_f0_min) + self.log2_f0_min
        f0_freqs = tr.pow(2.0, log2_f0_freqs)
        f0_freqs = f0_freqs.view(-1, 1)
        return f0_freqs

    def calc_amplitudes(self, theta_density: T) -> T:
        assert theta_density.ndim == 0
        offset = 0.25 * theta_density + 0.75 * theta_density ** 2
        grain_indices = tr.arange(self.n_grains)
        sigmoid_operand = (1 - theta_density) * self.n_grains * (grain_indices / self.n_grains - offset)
        amplitudes = 1 - tr.sigmoid(2 * sigmoid_operand)
        amplitudes = amplitudes / tr.max(amplitudes)
        amplitudes = amplitudes.view(-1, 1)
        return amplitudes

    def calc_slope(self, theta_slope: T) -> T:
        """
        theta_slope --> ±1 correspond to a near-vertical line.
        theta_slope = 0 corresponds to a horizontal line.
        The output is measured in octaves per second.
        """
        assert theta_slope.ndim == 0
        typical_slope = self.sr / (self.Q * self.hop_len)
        return tr.tan(theta_slope * np.pi / 2) * typical_slope / 4

    def forward(self,
                theta_density: T,
                theta_slope: T,
                seed: Optional[int] = None) -> T:
        if seed is not None:
            self.rand_gen.manual_seed(seed)

        # Create chirplet grains
        f0_freqs_hz = self.sample_f0_freqs()
        amplitudes = self.calc_amplitudes(theta_density)
        window = self.make_hann_window(self.grain_n_samples)

        phase = self.grain_support
        gamma = self.calc_slope(theta_slope)  # TODO
        if gamma != 0:
            phase = tr.expm1(gamma * self.const_log2 * phase) / (gamma * self.const_log2)
        grains = tr.sin(2 * tr.pi * f0_freqs_hz * phase) * amplitudes * window
        grains /= tr.sqrt(f0_freqs_hz)

        # Create audio
        paddings = tr.zeros((self.n_grains, self.n_samples - self.grain_n_samples))
        onsets = self.sample_onsets()
        x = []
        for grain, padding, onset in zip(grains, paddings, onsets):
            grain = tr.cat((grain, padding))
            x.append(tr.roll(grain, shifts=onset.item()))
        x = tr.stack(x, dim=0)
        x = tr.sum(x, dim=0)
        x = x / tr.norm(x, p=2)  # TODO
        return x

    @staticmethod
    def make_hann_window(n_samples: int) -> T:
        x = tr.arange(n_samples)
        y = tr.sin(tr.pi * x / n_samples) ** 2
        return y


if __name__ == "__main__":
    sr = 2 ** 13
    duration = 2 ** 2
    grain_duration = 2 ** -1
    n_grains = 2 ** 2
    f0_min_hz = 2 ** 8
    f0_max_hz = 2 ** 11

    n_samples = int(duration * sr)
    grain_n_samples = int(grain_duration * sr)

    synth = ChirpTextureSynth(sr=sr,
                              n_samples=n_samples,
                              n_grains=n_grains,
                              grain_n_samples=grain_n_samples,
                              f0_min_hz=f0_min_hz,
                              f0_max_hz=f0_max_hz)

    x = synth.forward(theta_density=tr.tensor(1.0), theta_slope=tr.tensor(0.5))
