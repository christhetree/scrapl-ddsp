import json
import logging
import os
from typing import Optional

import numpy as np
import torch as tr
import torch.nn as nn
import torchaudio
from torch import Tensor as T
from tqdm import tqdm

from data import get_text_embedding
from denoiser import Denoiser
from experiments.paths import OUT_DIR, CONFIGS_DIR, DATA_DIR, MODELS_DIR
from flowtron import Flowtron
from text.cmudict import CMUDict

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ChirpTextureSynth(nn.Module):
    def __init__(
        self,
        sr: float,
        n_samples: int,
        n_grains: int,
        grain_n_samples: int,
        f0_min_hz: float,
        f0_max_hz: float,
        Q: int,
        hop_len: int,
        max_theta_slope: float = 0.95,
        seed: Optional[int] = None,
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
        # onsets.fill_(0.5)  # TODO(cm): tmp
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

    def make_x_from_theta(self, theta_density: T, theta_slope: T, seed: T) -> T:
        # TODO(cm): add batch support to synth
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


class FlowtronSynth(nn.Module):
    def __init__(
        self,
        config_path: str,
        model_path: str,
        waveglow_path: str,
        theta_e_path: str,
        text: str = "What is going on?!",
        n_frames: int = 128,
        max_sigma: float = 1.0,
        waveglow_sigma: float = 0.75,
        denoiser_strength: float = 0.01,
        cache_dir: Optional[str] = None,
        waveglow_seed: Optional[int] = None,
    ):
        super().__init__()
        self.text = text
        self.n_frames = n_frames
        self.max_sigma = max_sigma
        self.waveglow_sigma = waveglow_sigma
        self.denoiser_strength = denoiser_strength
        self.cache_dir = cache_dir
        self.waveglow_seed = waveglow_seed

        if self.cache_dir is not None:
            log.info(f"Using Flowtron cache directory: {self.cache_dir}")
            os.makedirs(self.cache_dir, exist_ok=True)
        self.use_cache = cache_dir is not None

        self.rand_gen_cpu = tr.Generator(device="cpu")
        self.rand_gen_gpu = None
        if tr.cuda.is_available():
            self.rand_gen_gpu = tr.Generator(device="cuda")

        self.waveglow_rand_gen_cpu = tr.Generator(device="cpu")
        self.waveglow_rand_gen_gpu = None
        if tr.cuda.is_available():
            self.waveglow_rand_gen_gpu = tr.Generator(device="cuda")

        with open(config_path, "r") as f:
            config = json.load(f)
        self.data_config = config["data_config"]
        self.model_config = config["model_config"]
        self.sr = self.data_config["sampling_rate"]
        self.z_dim = self.model_config["n_mel_channels"]
        self.hop_len = self.data_config["hop_length"]
        self.Q = 12  # TODO(cm): tmp

        log.info(f"Started loading Flowtron models")
        state_dict = tr.load(model_path, map_location="cpu")["state_dict"]
        model = Flowtron(**self.model_config)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        waveglow = tr.load(waveglow_path, map_location="cpu")["model"]
        waveglow.eval()
        denoiser = Denoiser(waveglow)
        denoiser.eval()
        log.info(f"Finished loading Flowtron models")

        self.model = model
        self.waveglow = waveglow
        self.denoiser = denoiser

        self.text_cleaners = self.data_config["text_cleaners"]
        cmudict_path = self.data_config["cmudict_path"]
        assert os.path.isfile(cmudict_path)
        keep_ambiguous = self.data_config["keep_ambiguous"]
        self.cmudict = CMUDict(cmudict_path, keep_ambiguous=keep_ambiguous)
        self.p_arpabet = self.data_config["p_arpabet"]

        # TODO(cm): make dynamic
        text_encoded = get_text_embedding(
            text, self.text_cleaners, self.cmudict, self.p_arpabet, self.rand_gen_cpu
        )
        self.register_buffer("text_encoded", text_encoded)
        self.register_buffer("speaker_id", tr.tensor([0]).long())

        theta_e_mu = tr.load(theta_e_path)
        assert theta_e_mu.shape == (self.z_dim, 1)
        self.register_buffer("theta_e_mu", theta_e_mu)
        # self.register_buffer("z", tr.zeros((1, self.z_dim, self.n_frames)))

    def get_rand_gen(self, device: str) -> tr.Generator:
        if device == "cpu":
            return self.rand_gen_cpu
        else:
            return self.rand_gen_gpu

    def get_waveglow_rand_gen(self, device: str) -> tr.Generator:
        if device == "cpu":
            return self.waveglow_rand_gen_cpu
        else:
            return self.waveglow_rand_gen_gpu

    def _forward(self, theta_sig: T, theta_e: T, seed: Optional[T] = None) -> T:
        assert theta_sig.ndim == theta_e.ndim == 1
        rand_gen = self.get_rand_gen(device=self.speaker_id.device.type)
        if seed is not None:
            assert seed.shape == theta_sig.shape

        z_s = []
        for idx, (sig_0to1, e_0to1) in enumerate(zip(theta_sig, theta_e)):
            assert 0 < sig_0to1 <= 1
            assert 0 <= e_0to1 <= 1
            if seed is not None:
                rand_gen.manual_seed(int(seed[idx].item()))
            sig = sig_0to1 * self.max_sigma
            mu = e_0to1 * self.theta_e_mu
            mu = mu.view(-1, 1).expand(-1, self.n_frames)
            z = tr.normal(mu, sig, generator=rand_gen)
            z_s.append(z)
        z_s = tr.stack(z_s, dim=0)

        bs = z_s.size(0)
        # Prepare speaker_id and text_encoded
        speaker_id = self.speaker_id.expand(bs)
        text_encoded = self.text_encoded.expand(bs, -1)
        # Calc Mel posterior
        mel_posterior, _ = self.model.infer(z_s, speaker_id, text_encoded)
        # Calc audio from Mel posterior
        waveglow_rand_gen = self.get_waveglow_rand_gen(device=mel_posterior.device.type)
        if self.waveglow_seed is not None:
            waveglow_rand_gen.manual_seed(self.waveglow_seed)
        audio = self.waveglow.infer(
            mel_posterior, sigma=self.waveglow_sigma, rand_gen=waveglow_rand_gen
        )
        # Denoise audio
        audio_denoised = self.denoiser(audio, strength=self.denoiser_strength)
        return audio_denoised

    def _forward_cached(self, theta_sig: T, theta_e: T, seed: T) -> T:
        assert theta_sig.shape == theta_e.shape == seed.shape
        audios = []
        for sig, e, s in tqdm(zip(theta_sig, theta_e, seed)):
            cache_name = f"flowtron_{s.item()}_{sig.item():.6f}_{e.item():.6f}.wav"
            cache_path = os.path.join(self.cache_dir, cache_name)
            if os.path.isfile(cache_path):
                audio, sr = torchaudio.load(cache_path)
                assert sr == self.sr
                audio = audio.unsqueeze(0)
            else:
                audio = self._forward(sig.view(1), e.view(1), s.view(1))
                audio_2 = self._forward(sig.view(1), e.view(1), s.view(1))
                log.info(f"audio.max(): {audio.max()}, audio_2.max(): {audio_2.max()}")
                assert tr.allclose(audio, audio_2)
                torchaudio.save(cache_path, audio.squeeze(0), self.sr)
            audios.append(audio)
        audio = tr.cat(audios, dim=0)
        return audio

    def forward(self, theta_sig: T, theta_e: T, seed: Optional[T] = None) -> T:
        if self.use_cache:
            return self._forward_cached(theta_sig, theta_e, seed)
        else:
            return self._forward(theta_sig, theta_e, seed)

    def make_x_from_theta(self, theta_sig: T, theta_e: T, seed: T) -> T:
        theta_e = (theta_e + 1.0) / 2.0  # TODO(cm): tmp
        audio = self.forward(theta_sig, theta_e, seed)
        return audio


if __name__ == "__main__":
    config_path = os.path.join(CONFIGS_DIR, "flowtron/config.json")
    model_path = os.path.join(MODELS_DIR, "flowtron_ljs.pt")
    waveglow_path = os.path.join(MODELS_DIR, "waveglow_256channels_universal_v5.pt")
    theta_e_path = os.path.join(DATA_DIR, "z_80_surprised.pt")

    synth = FlowtronSynth(
        config_path=config_path,
        model_path=model_path,
        waveglow_path=waveglow_path,
        theta_e_path=theta_e_path,
    )

    theta_sig = tr.tensor([0.1, 0.5, 1.0])
    # theta_sig = tr.tensor([0.001, 0.001, 0.001])
    theta_e = tr.tensor([0.0, 0.5, 1.0])
    # theta_e = tr.tensor([1.0, 1.0, 1.0])
    # theta_sig = tr.tensor([0.5])
    # theta_e = tr.tensor([0.2])
    seed = tr.tensor([43])
    # seed = tr.tensor([123, 456, 789])

    with tr.no_grad():
        audio = synth(theta_sig, theta_e, seed)

    for idx, a in enumerate(audio):
        save_path = os.path.join(OUT_DIR, f"flowtron_{idx}.wav")
        torchaudio.save(save_path, a, synth.sr)

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
