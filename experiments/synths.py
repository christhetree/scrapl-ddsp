import logging
import math
import os
from typing import Optional

import numpy as np
import torch as tr
import torch.nn as nn
from torch import Tensor as T

from experiments.paths import OUT_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ChirpletSynth(nn.Module):
    def __init__(
        self,
        sr: float,
        n_samples: int,
        bw_n_samples: int,
        f0_min_hz: float,
        f0_max_hz: float,
        J_cqt: int = 5,
        Q: int = 12,
        hop_len: int = 256,
        am_hz_min: float = 4.0,
        am_hz_max: float = 16.0,
        fm_oct_hz_min: float = 0.5,
        fm_oct_hz_max: float = 4.0,
        delta_frac_min: Optional[float] = None,
        delta_frac_max: Optional[float] = None,
        sigma0: float = 0.1,
    ):
        super().__init__()
        assert bw_n_samples <= n_samples
        bw_frac = bw_n_samples / n_samples
        max_delta_frac = (1.0 - bw_frac) / 2.0
        if delta_frac_min is None:
            delta_frac_min = -max_delta_frac
        if delta_frac_max is None:
            delta_frac_max = max_delta_frac
        assert -max_delta_frac <= delta_frac_min <= delta_frac_max <= max_delta_frac
        assert f0_max_hz >= f0_min_hz
        self.sr = sr
        self.n_samples = n_samples
        self.bw_n_samples = bw_n_samples
        self.f0_min_hz = f0_min_hz
        self.f0_max_hz = f0_max_hz
        self.J_cqt = J_cqt
        self.Q = Q
        self.hop_len = hop_len
        self.am_hz_min = am_hz_min
        self.am_hz_max = am_hz_max
        self.fm_oct_hz_min = fm_oct_hz_min
        self.fm_oct_hz_max = fm_oct_hz_max
        self.delta_frac_min = delta_frac_min
        self.delta_frac_max = delta_frac_max
        self.sigma0 = sigma0

        # Derived params
        self.f0_min_hz_log2 = tr.log2(tr.tensor(f0_min_hz))
        self.f0_max_hz_log2 = tr.log2(tr.tensor(f0_max_hz))
        self.f0_hz = None
        if f0_min_hz == f0_max_hz:
            self.f0_hz = f0_min_hz
        self.am_hz_min_log2 = tr.log2(tr.tensor(am_hz_min))
        self.am_hz_max_log2 = tr.log2(tr.tensor(am_hz_max))
        self.fm_oct_hz_min_log2 = tr.log2(tr.tensor(fm_oct_hz_min))
        self.fm_oct_hz_max_log2 = tr.log2(tr.tensor(fm_oct_hz_max))
        self.delta_min = int(delta_frac_min * n_samples)
        self.delta_max = int(delta_frac_max * n_samples)
        self.rand_gen = tr.Generator(device="cpu")

        # Temporal support
        support = (tr.arange(n_samples) - (n_samples // 2)) / sr
        self.register_buffer("support", support)

    def forward(self, theta_am_hz: T, theta_fm_hz: T, seed: Optional[T] = None) -> T:
        assert theta_am_hz.ndim == theta_fm_hz.ndim == 0
        if seed is not None:
            assert seed.ndim == 0 or seed.shape == (1,)
            self.rand_gen.manual_seed(int(seed.item()))
        delta = tr.randint(
            self.delta_min, self.delta_max + 1, (1,), generator=self.rand_gen
        ).item()

        if self.f0_hz is None:
            f0_hz_log2 = (
                tr.rand((1,), generator=self.rand_gen)
                * (self.f0_max_hz_log2 - self.f0_min_hz_log2)
                + self.f0_min_hz_log2
            )
            f0_hz = (2**f0_hz_log2).item()
        else:
            f0_hz = self.f0_hz

        chirplet = self.generate_am_chirp(
            self.support,
            f0_hz,
            theta_am_hz,
            theta_fm_hz,
            self.n_samples,
            self.bw_n_samples,
            delta,
            self.sigma0,
        )
        return chirplet

    def make_x_from_theta(self, theta_am_hz_0to1: T, theta_fm_hz_0to1: T, seed: T) -> T:
        assert theta_am_hz_0to1.ndim == theta_fm_hz_0to1.ndim == 1
        assert theta_am_hz_0to1.min() >= 0.0
        assert theta_am_hz_0to1.max() <= 1.0
        assert theta_fm_hz_0to1.min() >= 0.0
        assert theta_fm_hz_0to1.max() <= 1.0
        theta_am_hz_log2 = (
            theta_am_hz_0to1 * (self.am_hz_max_log2 - self.am_hz_min_log2)
            + self.am_hz_min_log2
        )
        theta_am_hz = 2**theta_am_hz_log2
        theta_fm_hz_log2 = (
            theta_fm_hz_0to1 * (self.fm_oct_hz_max_log2 - self.fm_oct_hz_min_log2)
            + self.fm_oct_hz_min_log2
        )
        theta_fm_hz = 2**theta_fm_hz_log2
        x = []
        for idx in range(theta_am_hz.size(0)):
            curr_x = self.forward(theta_am_hz[idx], theta_fm_hz[idx], seed[idx])
            x.append(curr_x)
        x = tr.stack(x, dim=0).unsqueeze(1)  # Unsqueeze channel dim
        return x

    @staticmethod
    def generate_am_chirp(
        support: T,
        f0_hz: float | T,
        am_hz: float | T,
        fm_oct_hz: float | T,
        n_samples: int,
        bw_n_samples: int,
        delta: int = 0,
        sigma0: float = 0.1,
    ) -> T:
        # t = (tr.arange(n_samples) - (n_samples // 2)) / sr
        assert support.ndim == 1
        t = support
        phi = f0_hz / (fm_oct_hz * math.log(2)) * (2 ** (fm_oct_hz * t) - 1)
        carrier = tr.sin(2 * tr.pi * phi)
        modulator = tr.sin(2 * tr.pi * am_hz * t)
        window_std = sigma0 * bw_n_samples / fm_oct_hz
        window = tr.signal.windows.gaussian(n_samples, std=window_std, sym=True)
        chirp = carrier * fm_oct_hz * window
        if am_hz > 0:
            chirp = chirp * modulator
        if delta != 0:
            chirp = tr.roll(chirp, shifts=delta)
        return chirp


class ChirpTextureSynth(nn.Module):
    def __init__(
        self,
        sr: float,
        n_samples: int,
        n_grains: int,
        grain_n_samples: int,
        f0_min_hz: float,
        f0_max_hz: float,
        J_cqt: int = 5,
        Q: int = 12,
        hop_len: int = 256,
        max_theta_slope: float = 0.95,
    ):
        super().__init__()
        assert n_samples >= grain_n_samples
        assert f0_max_hz >= f0_min_hz
        self.sr = sr
        self.n_samples = n_samples
        self.n_grains = n_grains
        self.grain_n_samples = grain_n_samples
        self.f0_min_hz = f0_min_hz
        self.f0_max_hz = f0_max_hz
        self.J_cqt = J_cqt
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
        if tr.cuda.is_available():
            self.rand_gen_gpu = tr.Generator(device="cuda")

    def get_rand_gen(self, device: str) -> tr.Generator:
        if device == "cpu":
            return self.rand_gen_cpu
        else:
            return self.rand_gen_gpu

    def sample_onsets(self, rand_gen: tr.Generator) -> T:
        # TODO(cm): add support for edge padding
        onsets = self.onsets.uniform_(generator=rand_gen)
        onsets = onsets * (self.n_samples - self.grain_n_samples)
        onsets = onsets.long()
        return onsets

    def sample_f0_freqs(self, rand_gen: tr.Generator) -> T:
        log2_f0_freqs = self.log2_f0_freqs.uniform_(generator=rand_gen)
        log2_f0_freqs = (
            log2_f0_freqs * (self.log2_f0_max - self.log2_f0_min) + self.log2_f0_min
        )
        f0_freqs = tr.pow(2.0, log2_f0_freqs)
        f0_freqs = f0_freqs.view(-1, 1)
        return f0_freqs

    def calc_amplitudes(self, theta_density: T) -> T:
        assert theta_density.ndim == 0
        offset = 0.25 * theta_density + 0.75 * theta_density**2
        sigmoid_operand = (
            (1 - theta_density)
            * self.n_grains
            * (self.grain_indices / self.n_grains - offset)
        )
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
        slope = (
            tr.tan(self.max_theta_slope * theta_slope * np.pi / 2) * typical_slope / 4
        )
        return slope

    def forward(self, theta_density: T, theta_slope: T, seed: Optional[T] = None) -> T:
        assert theta_density.ndim == theta_slope.ndim == 0
        rand_gen = self.get_rand_gen(device=self.grain_support.device.type)
        if seed is not None:
            assert seed.ndim == 0 or seed.shape == (1,)
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

    def make_x_from_theta(
        self, theta_d_0to1: T, theta_s_0to1: T, seed: T, seed_words: Optional[T] = None
    ) -> T:
        # TODO(cm): add batch support to synth
        assert theta_d_0to1.min() >= 0.0
        assert theta_d_0to1.max() <= 1.0
        assert theta_s_0to1.min() >= 0.0
        assert theta_s_0to1.max() <= 1.0
        theta_density = theta_d_0to1
        theta_slope = theta_s_0to1 * 2.0 - 1.0
        x = []
        for idx in range(theta_density.size(0)):
            curr_x = self.forward(theta_density[idx], theta_slope[idx], seed[idx])
            x.append(curr_x)
        x = tr.stack(x, dim=0).unsqueeze(1)  # Unsqueeze channel dim
        return x

    @staticmethod
    def make_hann_window(n_samples: int) -> T:
        x = tr.arange(n_samples)
        y = tr.sin(tr.pi * x / n_samples) ** 2
        return y


if __name__ == "__main__":
    sr = 2**13
    duration = 2**2
    grain_duration = 2**2
    n_grains = 2**0
    f0_min_hz = 2**8
    f0_max_hz = 2**11

    n_samples = int(duration * sr)
    grain_n_samples = int(grain_duration * sr)

    synth = ChirpTextureSynth(
        sr=sr,
        n_samples=n_samples,
        n_grains=n_grains,
        grain_n_samples=grain_n_samples,
        f0_min_hz=f0_min_hz,
        f0_max_hz=f0_max_hz,
        Q=12,
        hop_len=256,
    )

    x = synth.forward(theta_density=tr.tensor(1.0), theta_slope=tr.tensor(0.5))

    save_path = "chirp_texture.wav"
    import soundfile as sf

    sf.write(os.path.join(OUT_DIR, save_path), x.numpy(), samplerate=sr)
