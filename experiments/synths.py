import logging
import math
import os
from typing import Optional

import numpy as np
import torch as tr
import torch.nn as nn
import torchaudio
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
        bw_oct: float,  # TODO(cm)
        f0_min_hz: float,
        f0_max_hz: float,
        J_cqt: int = 5,
        Q: int = 12,
        hop_len: int = 256,
        am_hz_min: float = 4.0,
        am_hz_max: float = 16.0,
        fm_oct_hz_min: float = 0.5,
        fm_oct_hz_max: float = 4.0,
        delta_min: int = 0,
        delta_max: int = 0,
        sigma0: float = 0.1,
    ):
        super().__init__()
        assert -n_samples <= delta_min <= delta_max <= n_samples
        assert f0_max_hz >= f0_min_hz
        self.sr = sr
        self.n_samples = n_samples
        self.bw_oct = bw_oct
        self.f0_min_hz = f0_min_hz
        self.f0_max_hz = f0_max_hz
        self.J_cqt = J_cqt
        self.Q = Q
        self.hop_len = hop_len
        self.am_hz_min = am_hz_min
        self.am_hz_max = am_hz_max
        self.fm_oct_hz_min = fm_oct_hz_min
        self.fm_oct_hz_max = fm_oct_hz_max
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.sigma0 = sigma0
        log.info(f"am_hz_min: {am_hz_min}, am_hz_max: {am_hz_max}, "
                 f"fm_hz_min: {fm_oct_hz_min}, fm_hz_max: {fm_oct_hz_max}")

        # Derived params
        self.bw_n_samples = int(bw_oct * sr)
        self.f0_min_hz_log2 = tr.log2(tr.tensor(f0_min_hz))
        self.f0_max_hz_log2 = tr.log2(tr.tensor(f0_max_hz))
        self.f0_hz = None
        if f0_min_hz == f0_max_hz:
            self.f0_hz = f0_min_hz
        self.am_hz_min_log2 = None
        self.am_hz_max_log2 = None
        self.am_hz = None
        if am_hz_min == am_hz_max:
            self.am_hz = am_hz_min
        else:
            self.am_hz_min_log2 = tr.log2(tr.tensor(am_hz_min))
            self.am_hz_max_log2 = tr.log2(tr.tensor(am_hz_max))
        self.fm_oct_hz_min_log2 = None
        self.fm_oct_hz_max_log2 = None
        self.fm_oct_hz = None
        if fm_oct_hz_min == fm_oct_hz_max:
            self.fm_oct_hz = fm_oct_hz_min
        else:
            self.fm_oct_hz_min_log2 = tr.log2(tr.tensor(fm_oct_hz_min))
            self.fm_oct_hz_max_log2 = tr.log2(tr.tensor(fm_oct_hz_max))
        self.delta = None
        if delta_min == delta_max:
            self.delta = self.delta_min
        self.rand_gen = tr.Generator(device="cpu")

        # Temporal support
        support = (tr.arange(n_samples) - (n_samples // 2)) / sr
        self.register_buffer("support", support)
        # Window support
        win_support = self.create_gaussian_window_support(n_samples)
        self.register_buffer("win_support", win_support)

    def make_x(self, theta_am_hz: T, theta_fm_hz: T, seed: Optional[T] = None) -> T:
        assert theta_am_hz.ndim == theta_fm_hz.ndim == 0
        if seed is not None:
            assert seed.ndim == 0 or seed.shape == (1,)
            self.rand_gen.manual_seed(int(seed.item()))
        if self.delta is None:
            delta = tr.randint(
                self.delta_min, self.delta_max + 1, (1,), generator=self.rand_gen
            ).item()
        else:
            delta = self.delta

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
            self.bw_n_samples,
            delta,
            self.sigma0,
            win_support=self.win_support,
        )
        return chirplet

    def forward(self, theta_am_hz_0to1: T, theta_fm_hz_0to1: T, seed: T) -> T:
        assert theta_am_hz_0to1.ndim == theta_fm_hz_0to1.ndim == 1
        assert theta_am_hz_0to1.min() >= 0.0
        assert theta_am_hz_0to1.max() <= 1.0
        assert theta_fm_hz_0to1.min() >= 0.0
        assert theta_fm_hz_0to1.max() <= 1.0

        if self.am_hz is None:
            theta_am_hz_log2 = (
                theta_am_hz_0to1 * (self.am_hz_max_log2 - self.am_hz_min_log2)
                + self.am_hz_min_log2
            )
            theta_am_hz = 2**theta_am_hz_log2
        else:
            theta_am_hz = tr.full_like(theta_am_hz_0to1, self.am_hz)
        if self.fm_oct_hz is None:
            theta_fm_hz_log2 = (
                theta_fm_hz_0to1 * (self.fm_oct_hz_max_log2 - self.fm_oct_hz_min_log2)
                + self.fm_oct_hz_min_log2
            )
            theta_fm_hz = 2**theta_fm_hz_log2
        else:
            theta_fm_hz = tr.full_like(theta_fm_hz_0to1, self.fm_oct_hz)
        x = []
        for idx in range(theta_am_hz.size(0)):
            curr_x = self.make_x(theta_am_hz[idx], theta_fm_hz[idx], seed[idx])
            x.append(curr_x)
        x = tr.stack(x, dim=0).unsqueeze(1)  # Unsqueeze channel dim
        return x

    @staticmethod
    def create_gaussian_window_support(
        n_samples: int, sym: bool = True, device: Optional[tr.device] = None
    ) -> T:
        assert n_samples > 0
        start = -(n_samples if not sym and n_samples > 1 else n_samples - 1) / 2.0
        end = start + (n_samples - 1)
        win_support = tr.linspace(start=start, end=end, steps=n_samples, device=device)
        return win_support

    @staticmethod
    def create_gaussian_window(
        std: float,
        support: Optional[T] = None,
        n_samples: Optional[int] = None,
        sym: bool = True,
        device: Optional[tr.device] = None,
    ) -> T:
        assert std > 0, f"std must be positive, got {std}"
        if support is None:
            assert n_samples is not None
            support = ChirpletSynth.create_gaussian_window_support(
                n_samples, sym=sym, device=device
            )
        constant = 1.0 / (std * math.sqrt(2.0))
        window = support * constant
        window = tr.exp(-(window**2))
        return window

    @staticmethod
    def generate_am_chirp(
        support: T,
        f0_hz: float | T,
        am_hz: float | T,
        fm_oct_hz: float | T,
        bw_n_samples: int,
        delta: int = 0,
        sigma0: float = 0.1,
        win_support: Optional[T] = None,
    ) -> T:
        # t = (tr.arange(n_samples) - (n_samples // 2)) / sr
        assert support.ndim == 1
        assert am_hz >= 0
        n_samples = support.size(0)
        t = support
        if fm_oct_hz == 0.0:
            phi = f0_hz * t
            window_std = float(sigma0 * bw_n_samples)
        else:
            phi = f0_hz / (fm_oct_hz * math.log(2)) * (2 ** (fm_oct_hz * t) - 1)
            window_std = abs(float(sigma0 * bw_n_samples / fm_oct_hz))
        carrier = tr.sin(2 * tr.pi * phi)
        # Divide am_hz by 2 since we're using a sinusoid as the modulator
        modulator = tr.sin(2 * tr.pi * am_hz / 2.0 * t)
        window = ChirpletSynth.create_gaussian_window(
            window_std,
            support=win_support,
            n_samples=n_samples,
            sym=True,
            device=support.device,
        )
        if fm_oct_hz == 0.0:
            chirp = carrier * window
        else:
            chirp = carrier * fm_oct_hz * window
        if am_hz > 0:
            chirp = chirp * modulator
        if delta != 0:
            chirp = tr.roll(chirp, shifts=delta)
            if delta > 0:
                chirp[:delta] = 0.0
            else:
                chirp[delta:] = 0.0
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

    def make_x(self, theta_density: T, theta_slope: T, seed: Optional[T] = None) -> T:
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

    def forward(self, theta_d_0to1: T, theta_s_0to1: T, seed: T) -> T:
        # TODO(cm): add batch support to synth
        assert theta_d_0to1.min() >= 0.0
        assert theta_d_0to1.max() <= 1.0
        assert theta_s_0to1.min() >= 0.0
        assert theta_s_0to1.max() <= 1.0
        theta_density = theta_d_0to1
        theta_slope = theta_s_0to1 * 2.0 - 1.0
        x = []
        for idx in range(theta_density.size(0)):
            curr_x = self.make_x(theta_density[idx], theta_slope[idx], seed[idx])
            x.append(curr_x)
        x = tr.stack(x, dim=0).unsqueeze(1)  # Unsqueeze channel dim
        return x

    @staticmethod
    def make_hann_window(n_samples: int) -> T:
        x = tr.arange(n_samples)
        y = tr.sin(tr.pi * x / n_samples) ** 2
        return y


class AMFMSynth(nn.Module):
    def __init__(
        self,
        sr: int,
        n_samples: int,
        f0_hz_min: float,
        f0_hz_max: float,
        am_hz_min: float,
        am_hz_max: float,
        fm_hz_min: float,
        fm_hz_max: float,
        mi_am_min: float = 0.0,
        mi_am_max: float = 1.0,
        mi_fm_min: float = 0.0,
        mi_fm_max: float = 1.0,
        theta_am_is_mod_index: bool = False,
        theta_fm_is_mod_index: bool = True,
        J_cqt: int = 5,
        Q: int = 12,
        hop_len: int = 256,
    ):
        super().__init__()
        assert 0.0 < f0_hz_min <= f0_hz_max
        assert 0.0 < am_hz_min <= am_hz_max
        assert 0.0 < fm_hz_min <= fm_hz_max
        self.sr = sr
        self.n_samples = n_samples
        self.f0_hz_min = f0_hz_min
        self.f0_hz_max = f0_hz_max
        self.am_hz_min = am_hz_min
        self.am_hz_max = am_hz_max
        self.fm_hz_min = fm_hz_min
        self.fm_hz_max = fm_hz_max
        self.mi_am_min = mi_am_min
        self.mi_am_max = mi_am_max
        self.mi_fm_min = mi_fm_min
        self.mi_fm_max = mi_fm_max
        self.theta_am_is_mod_index = theta_am_is_mod_index
        self.theta_fm_is_mod_index = theta_fm_is_mod_index
        self.J_cqt = J_cqt
        self.Q = Q
        self.hop_len = hop_len

        # Derived params
        self.nyquist = sr / 2.0
        self.f0_hz_min_log2 = tr.log2(tr.tensor(f0_hz_min))
        self.f0_hz_max_log2 = tr.log2(tr.tensor(f0_hz_max))
        self.am_hz_min_log2 = tr.log2(tr.tensor(am_hz_min))
        self.am_hz_max_log2 = tr.log2(tr.tensor(am_hz_max))
        self.fm_hz_min_log2 = tr.log2(tr.tensor(fm_hz_min))
        self.fm_hz_max_log2 = tr.log2(tr.tensor(fm_hz_max))
        self.rand_gen = tr.Generator(device="cpu")
        self.register_buffer("t", tr.arange(n_samples) / sr)
        self.register_buffer("ones", tr.ones((n_samples,)))

    def make_x(
        self,
        theta_am: T,
        theta_fm: T,
        seed: Optional[T] = None,
        seed_target: Optional[T] = None,
    ) -> T:
        assert theta_am.ndim == theta_fm.ndim == 0
        # Deal with randomness here
        if seed is not None:
            assert seed.ndim == 0 or seed.shape == (1,)
            self.rand_gen.manual_seed(int(seed.item()))
        phase = (tr.rand((1,), generator=self.rand_gen) * 2 * tr.pi).to(self.t.device)
        phase_am = (tr.rand((1,), generator=self.rand_gen) * 2 * tr.pi).to(
            self.t.device
        )
        phase_fm = (tr.rand((1,), generator=self.rand_gen) * 2 * tr.pi).to(
            self.t.device
        )
        theta_am_hz = (tr.rand((1,), generator=self.rand_gen)).to(self.t.device)
        theta_am_mi = (tr.rand((1,), generator=self.rand_gen)).to(self.t.device)
        theta_fm_mi = (tr.rand((1,), generator=self.rand_gen)).to(self.t.device)
        theta_fm_hz = (tr.rand((1,), generator=self.rand_gen)).to(self.t.device)
        theta_f0_hz = (tr.rand((1,), generator=self.rand_gen)).to(self.t.device)

        # if seed_target is not None:
        #     assert seed_target.ndim == 0 or seed_target.shape == (1,)
        #     self.rand_gen.manual_seed(int(seed_target.item()))
        # elif seed is not None:
        #     self.rand_gen.manual_seed(int(seed.item()))
        # theta_fm_hz = (tr.rand((1,), generator=self.rand_gen)).to(self.t.device)
        # theta_f0_hz = (tr.rand((1,), generator=self.rand_gen)).to(self.t.device)

        if self.theta_am_is_mod_index:
            mi_am = theta_am
            am_hz_log2 = (
                theta_am_hz * (self.am_hz_max_log2 - self.am_hz_min_log2)
                + self.am_hz_min_log2
            )
            am_hz = 2**am_hz_log2
        else:
            mi_am = theta_am_mi * (self.mi_am_max - self.mi_am_min) + self.mi_am_min
            am_hz = theta_am
        if self.theta_fm_is_mod_index:
            mi_fm = theta_fm
            fm_hz_log2 = (
                theta_fm_hz * (self.fm_hz_max_log2 - self.fm_hz_min_log2)
                + self.fm_hz_min_log2
            )
            fm_hz = 2**fm_hz_log2
        else:
            mi_fm = theta_fm_mi * (self.mi_fm_max - self.mi_fm_min) + self.mi_fm_min
            fm_hz = theta_fm

        f0_hz_log2 = (
            theta_f0_hz * (self.f0_hz_max_log2 - self.f0_hz_min_log2)
            + self.f0_hz_min_log2
        )
        f0_hz = 2**f0_hz_log2

        am_sig = tr.sin(2 * tr.pi * am_hz * self.t + phase_am) * mi_am
        # from matplotlib import pyplot as plt
        # plt.plot(am_sig.numpy())
        # plt.title("am_sig")
        # plt.show()
        fm_sig = tr.sin(2 * tr.pi * fm_hz * self.t + phase_fm) * mi_fm
        # from matplotlib import pyplot as plt
        # plt.plot(fm_sig.numpy())
        # plt.title("fm_sig")
        # plt.show()
        f0_hz = self.ones * f0_hz.item()
        if mi_fm > 0:  # FM
            f0_hz = f0_hz + fm_sig * f0_hz
        if f0_hz.min() < 0.0:
            log.warning(f"FM modulation is negative: {f0_hz.min():.0f}")
        if f0_hz.max() > self.nyquist:
            log.warning(f"FM modulation exceeds Nyquist: {f0_hz.max():.0f}")
        arg = tr.cumsum(2 * tr.pi * f0_hz / self.sr, dim=0) + phase
        # Multiply by 0.5 to avoid clipping from AM modulation
        x = 0.5 * tr.sin(arg)
        if mi_am > 0:  # AM
            x = x + am_sig * x
        return x

    def forward(
        self,
        theta_am_0to1: T,
        theta_fm_0to1: T,
        seed: Optional[T] = None,
        seed_target: Optional[T] = None,
    ) -> T:
        assert theta_am_0to1.ndim == theta_fm_0to1.ndim == 1
        assert theta_am_0to1.min() >= 0.0
        assert theta_am_0to1.max() <= 1.0
        assert theta_fm_0to1.min() >= 0.0
        assert theta_fm_0to1.max() <= 1.0
        if seed is not None:
            assert seed.shape == theta_am_0to1.shape
        if seed_target is not None:
            assert seed_target.shape == theta_am_0to1.shape

        if self.theta_am_is_mod_index:
            theta_am = (
                theta_am_0to1 * (self.mi_am_max - self.mi_am_min) + self.mi_am_min
            )
        else:
            theta_am = (
                theta_am_0to1 * (self.am_hz_max_log2 - self.am_hz_min_log2)
                + self.am_hz_min_log2
            )
            theta_am = 2**theta_am

        if self.theta_fm_is_mod_index:
            theta_fm = (
                theta_fm_0to1 * (self.mi_fm_max - self.mi_fm_min) + self.mi_fm_min
            )
        else:
            theta_fm = (
                theta_fm_0to1 * (self.fm_hz_max_log2 - self.fm_hz_min_log2)
                + self.fm_hz_min_log2
            )
            theta_fm = 2**theta_fm
        x = []
        for idx in range(theta_am.size(0)):
            curr_seed = None
            if seed is not None:
                curr_seed = seed[idx]
            curr_seed_target = None
            if seed_target is not None:
                curr_seed_target = seed_target[idx]
            curr_x = self.make_x(
                theta_am[idx], theta_fm[idx], curr_seed, curr_seed_target
            )
            x.append(curr_x)
        x = tr.stack(x, dim=0).unsqueeze(1)
        return x


if __name__ == "__main__":
    sr = 44100
    n_samples = 2 * sr
    f0_min_hz = 440.0
    f0_max_hz = 440.0
    am_hz_min = 2.0
    am_hz_max = 16.0
    fm_hz_min = 220.0
    fm_hz_max = 880.0
    mi_am_min = 0.0
    mi_am_max = 0.0
    mi_fm_min = 1.0
    mi_fm_max = 1.0

    synth = AMFMSynth(
        sr=sr,
        n_samples=n_samples,
        f0_hz_min=f0_min_hz,
        f0_hz_max=f0_max_hz,
        am_hz_min=am_hz_min,
        am_hz_max=am_hz_max,
        fm_hz_min=fm_hz_min,
        fm_hz_max=fm_hz_max,
        mi_am_min=mi_am_min,
        mi_am_max=mi_am_max,
        mi_fm_min=mi_fm_min,
        mi_fm_max=mi_fm_max,
        theta_am_is_mod_index=False,
        theta_fm_is_mod_index=False,
    )
    # theta_am_0to1 = tr.tensor([0.5])
    # theta_am_0to1 = tr.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    theta_fm_0to1 = tr.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    theta_am_0to1 = tr.full_like(theta_fm_0to1, 0.5)
    # theta_fm_0to1 = tr.full_like(theta_am_0to1, 0.0)
    seed = tr.full_like(theta_fm_0to1, 1).long()
    x = synth.forward(
        theta_am_0to1=theta_am_0to1, theta_fm_0to1=theta_fm_0to1, seed=seed
    )
    for idx, curr_x in enumerate(x):
        save_path = os.path.join(OUT_DIR, f"am_fm_{idx}.wav")
        torchaudio.save(save_path, curr_x, sr)
    exit()

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
