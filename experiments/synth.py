import logging
import os
from typing import Optional

import numpy as np
import torch as tr
import torch.nn as nn
from torch import Tensor as T

from experiments.paths import OUT_DIR

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
                 max_theta_slope: float = 0.95,
                 seed: Optional[int] = None):
        super().__init__()
        assert n_samples >= grain_n_samples
        assert f0_max_hz >= f0_min_hz
        self.sr = sr
        self.n_samples = n_samples
        self.n_grains = n_grains
        self.grain_n_samples = grain_n_samples
        self.f0_min_hz = f0_min_hz
        self.f0_max_hz = f0_max_hz
        self.Q = Q
        self.hop_len = hop_len
        self.max_theta_slope = max_theta_slope  # This prevents instabilities near +/-1

        self.log2_f0_min = tr.log2(tr.tensor(f0_min_hz))
        self.log2_f0_max = tr.log2(tr.tensor(f0_max_hz))
        self.grain_dur_s = grain_n_samples / sr
        support = tr.arange(grain_n_samples) / sr - (self.grain_dur_s / 2)
        grain_support = support.repeat(n_grains, 1)
        self.register_buffer("grain_support", grain_support)
        grain_indices = tr.arange(n_grains)
        self.register_buffer("grain_indices", grain_indices)
        window = self.make_hann_window(grain_n_samples)
        self.register_buffer("window", window)
        log2_f0_freqs = tr.empty((self.n_grains,))
        self.register_buffer("log2_f0_freqs", log2_f0_freqs)
        onsets = tr.empty((self.n_grains,))
        self.register_buffer("onsets", onsets)
        paddings = tr.zeros((self.n_grains, self.n_samples - self.grain_n_samples))
        self.register_buffer("paddings", paddings)

        # TODO(cm): use only one generator, seems to be a PyTorch limitation
        self.rand_gen_cpu = tr.Generator(device="cpu")
        self.rand_gen_gpu = None
        if seed is not None:
            self.rand_gen_cpu.manual_seed(seed)
        if tr.cuda.is_available():
            self.rand_gen_gpu = tr.Generator(device="cuda")
            if seed is not None:
                self.rand_gen_gpu.manual_seed(seed)

    def get_rand_gen(self, device: str) -> tr.Generator:
        if device == "cpu":
            return self.rand_gen_cpu
        else:
            return self.rand_gen_gpu

    def sample_onsets(self, rand_gen: tr.Generator) -> T:
        # TODO(cm): add support for edge padding
        onsets = self.onsets.uniform_(generator=rand_gen)
        onsets.fill_(0.5)  # TODO(cm): tmp
        onsets = onsets * (self.n_samples - self.grain_n_samples)
        onsets = onsets.long()
        return onsets

    def sample_f0_freqs(self, rand_gen: tr.Generator) -> T:
        log2_f0_freqs = self.log2_f0_freqs.uniform_(generator=rand_gen)
        log2_f0_freqs = log2_f0_freqs * (self.log2_f0_max - self.log2_f0_min) + self.log2_f0_min
        f0_freqs = tr.pow(2.0, log2_f0_freqs)
        f0_freqs = f0_freqs.view(-1, 1)
        return f0_freqs

    def calc_amplitudes(self, theta_density: T) -> T:
        assert theta_density.ndim == 0
        offset = 0.25 * theta_density + 0.75 * theta_density ** 2
        sigmoid_operand = (1 - theta_density) * self.n_grains * (self.grain_indices / self.n_grains - offset)
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
        # theta_slope = self.max_theta_slope * theta_slope
        typical_slope = self.sr / (self.Q * self.hop_len)
        slope = tr.tan(
            self.max_theta_slope * theta_slope * np.pi / 2) * typical_slope / 4
        return slope

    def forward(self,
                theta_density: T,
                theta_slope: T,
                seed: Optional[T] = None) -> T:
        assert theta_density.ndim == theta_slope.ndim == 0
        rand_gen = self.get_rand_gen(device=self.grain_support.device.type)
        if seed is not None:
            rand_gen.manual_seed(int(seed.item()))

        # Create chirplet grains
        f0_freqs_hz = self.sample_f0_freqs(rand_gen)
        amplitudes = self.calc_amplitudes(theta_density)
        gamma = self.calc_slope(theta_slope)

        inst_freq = f0_freqs_hz * (2 ** (gamma * self.grain_support)) / self.sr
        phase = 2 * tr.pi * tr.cumsum(inst_freq, dim=1)
        grains = tr.sin(phase) * amplitudes * self.window
        grains /= tr.sqrt(f0_freqs_hz)

        # Create audio
        onsets = self.sample_onsets(rand_gen)
        x = []
        for grain, padding, onset in zip(grains, self.paddings, onsets):
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
    grain_duration = 2 ** 2
    n_grains = 2 ** 0
    f0_min_hz = 2 ** 8
    f0_max_hz = 2 ** 11

    n_samples = int(duration * sr)
    grain_n_samples = int(grain_duration * sr)

    synth = ChirpTextureSynth(sr=sr,
                              n_samples=n_samples,
                              n_grains=n_grains,
                              grain_n_samples=grain_n_samples,
                              f0_min_hz=f0_min_hz,
                              f0_max_hz=f0_max_hz,
                              Q=12,
                              hop_len=256)

    x = synth.forward(theta_density=tr.tensor(1.0), theta_slope=tr.tensor(0.5))

    save_path = "chirp_texture.wav"
    import soundfile as sf
    sf.write(os.path.join(OUT_DIR, save_path), x.numpy(), samplerate=sr)
