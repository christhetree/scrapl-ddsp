import json
import logging
import os
from typing import Optional

import numpy as np
import torch as tr
import torch.nn as nn
import torchaudio
import yaml
from torch import Tensor as T

from experiments.paths import OUT_DIR, CONFIGS_DIR, MODELS_DIR
from flowtron.data import get_text_embedding
from flowtron.flowtron import Flowtron
from flowtron.text.cmudict import CMUDict
from hifigan.env import AttrDict
from hifigan.models import Generator
from speechbrain.inference import FastSpeech2

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


class FlowtronSynth(nn.Module):
    def __init__(
        self,
        config_path: str,
        model_path: str,
        vocoder_path: str,
        theta_d_path: str,
        theta_s_path: str,
        default_text: str = "what is going on",
        wordlist_path: Optional[str] = None,
        n_words: int = 4,
        n_frames: int = 128,
        max_sigma: float = 1.0,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.default_text = default_text
        self.n_words = n_words
        self.n_frames = n_frames
        self.max_sigma = max_sigma
        self.cache_dir = cache_dir

        if self.cache_dir is not None:
            log.info(f"Using Flowtron cache directory: {self.cache_dir}")
            os.makedirs(self.cache_dir, exist_ok=True)

        self.rand_gen_cpu = tr.Generator(device="cpu")
        self.rand_gen_gpu = None
        if tr.cuda.is_available():
            self.rand_gen_gpu = tr.Generator(device="cuda")

        with open(config_path, "r") as f:
            config = json.load(f)
        self.data_config = config["data_config"]
        self.model_config = config["model_config"]
        self.sr = self.data_config["sampling_rate"]
        self.z_dim = self.model_config["n_mel_channels"]
        self.hop_len = self.data_config["hop_length"]
        self.Q = 12  # TODO(cm): tmp

        log.info(f"Loading Flowtron")
        state_dict = tr.load(model_path, map_location="cpu")["state_dict"]
        model = Flowtron(**self.model_config)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        # model.encoder.lstm.train()
        # for flow in model.flows:
        #     if isinstance(flow, AR_Step):
        #         flow.lstm.train()
        #         flow.attention_lstm.train()
        #     elif isinstance(flow, AR_Back_Step):
        #         flow.ar_step.lstm.train()
        #         flow.ar_step.attention_lstm.train()
        #     else:
        #         raise ValueError(f"Unknown flow type: {type(flow)}")
        self.model = model

        log.info(f"Loading Vocoder")
        vocoder_state_dict = tr.load(vocoder_path, map_location="cpu")["generator"]
        vocoder_config_path = os.path.join(MODELS_DIR, "hifigan_t2_v3_config.json")
        with open(vocoder_config_path, "r") as f:
            vocoder_config = json.load(f)
            hyperparams = AttrDict(vocoder_config)
        vocoder = Generator(hyperparams)
        vocoder.load_state_dict(vocoder_state_dict)
        vocoder.eval()
        vocoder.remove_weight_norm()
        self.vocoder = vocoder
        log.info(f"Finished loading models")

        # Text embedding init
        self.text_cleaners = self.data_config["text_cleaners"]
        cmudict_path = self.data_config["cmudict_path"]
        assert os.path.isfile(cmudict_path)
        keep_ambiguous = self.data_config["keep_ambiguous"]
        self.cmudict = CMUDict(cmudict_path, keep_ambiguous=keep_ambiguous)
        self.p_arpabet = self.data_config["p_arpabet"]

        self.wordlist = []
        if wordlist_path is not None:
            with open(wordlist_path, "r") as f:
                self.wordlist = json.load(f)
            log.info(f"Wordlist contains {len(self.wordlist)} words")

        if not self.wordlist:
            log.info(f"Using default text: {self.default_text}")
            default_text_emb = self.get_text_embedding(
                self.default_text, self.rand_gen_cpu
            )
            self.register_buffer("default_text_emb", default_text_emb)

        # Other init
        self.register_buffer("speaker_id", tr.tensor([0]).long())

        theta_d_mu = tr.load(theta_d_path)
        assert theta_d_mu.shape == (self.z_dim, self.n_frames)
        self.register_buffer("theta_d_mu", theta_d_mu)
        theta_s_mu = tr.load(theta_s_path)
        assert theta_s_mu.shape == (self.z_dim, self.n_frames)
        self.register_buffer("theta_s_mu", theta_s_mu)

        self.register_buffer("mu_tmp", tr.zeros_like(theta_s_mu))
        self.register_buffer("sig_tmp", tr.ones_like(theta_s_mu))
        # self.register_buffer("z_tmp", tr.normal(self.mu_tmp, self.sig_tmp))

    def get_text_embedding(self, text: str, rand_gen: tr.Generator) -> T:
        # TODO(cm): add rand_gen back and look into p_arpabet
        text_encoded = get_text_embedding(
            text, self.text_cleaners, self.cmudict, self.p_arpabet
        ).to(rand_gen.device)
        return text_encoded

    def get_rand_gen(self, device: str) -> tr.Generator:
        if device == "cpu":
            return self.rand_gen_cpu
        else:
            return self.rand_gen_gpu

    def _forward(self, theta_d: T, theta_s: T, seed: Optional[T] = None) -> T:
        assert theta_d.ndim == theta_s.ndim == 1
        rand_gen = self.get_rand_gen(device=self.speaker_id.device.type)
        # rand_gen = self.get_rand_gen("cpu")
        if seed is not None:
            assert seed.shape == theta_d.shape

        z_s = []
        bs = theta_d.size(0)
        # sigs = []
        for idx in range(bs):
            if seed is not None:
                rand_gen.manual_seed(int(seed[idx].item()))
            # sig = tr.rand((1,), generator=rand_gen) * self.max_sigma
            # sig = tr.tensor([0.5])
            # sigs.append(sig)
            z = tr.normal(self.mu_tmp, self.sig_tmp, generator=rand_gen)
            # z = self.z_tmp
            z_s.append(z)
        z_s = tr.stack(z_s, dim=0)
        # mu_probs = tr.softmax(tr.stack([theta_d, theta_s], dim=1), dim=1)
        # mu_d = theta_d.view(-1, 1, 1) * self.theta_d_mu * mu_probs[:, 0].view(-1, 1, 1)
        # mu_s = theta_s.view(-1, 1, 1) * self.theta_s_mu * mu_probs[:, 1].view(-1, 1, 1)
        # mu = mu_d + mu_s
        theta_d = theta_d.view(-1, 1, 1)
        theta_s = theta_s.view(-1, 1, 1)
        mu_d = theta_d * self.theta_d_mu
        mu_s = theta_s * self.theta_s_mu
        mu = (mu_d + mu_s) * 0.75
        # mu = (mu_d + mu_s) / 2.0
        # sig = theta_d.view(-1, 1, 1) * self.max_sigma
        sig = 0.5
        # sig = tr.cat(sigs, dim=0).view(-1, 1, 1)
        # sig = sig.to(mu.device)
        z_s = z_s * sig + mu

        # Prepare speaker_id
        speaker_id = self.speaker_id.expand(bs)
        # Prepare text embedding
        if self.wordlist:
            if seed is not None:
                # TODO(cm): this is problematic
                rand_gen.manual_seed(int(seed[0].item()))
            words = tr.randint(
                0, len(self.wordlist), (self.n_words,), generator=rand_gen
            )
            words = [self.wordlist[w] for w in words]
            text = " ".join(words)
            # log.info(f"Random text: {text}")
            text_emb = self.get_text_embedding(text, rand_gen).to(z_s.device)
        else:
            text_emb = self.default_text_emb
        text_emb = text_emb.expand(bs, -1)
        # Calc Mel posterior
        mel_posterior, _ = self.model.infer(z_s, speaker_id, text_emb)
        # Calc audio from Mel posterior
        audio = self.vocoder(mel_posterior)
        # audio_2 = self.vocoder(mel_posterior)
        # assert tr.allclose(audio, audio_2)
        # log.info(f"Vocoder is deterministic")
        return audio

    def _forward_cached(self, theta_d: T, theta_s: T, seed: T) -> T:
        assert theta_d.shape == theta_s.shape == seed.shape
        bs = theta_d.size(0)
        idx_to_audio = {}
        uncached_indices = []
        # uncached_indices = list(range(bs))
        for idx, (theta_d, theta_s, s) in enumerate(zip(theta_d, theta_s, seed)):
            cache_name = (
                f"flowtron_{s.item()}_{theta_d.item():.6f}_{theta_s.item():.6f}.wav"
            )
            cache_path = os.path.join(self.cache_dir, cache_name)
            if os.path.isfile(cache_path):
                # log.info(f"Cache hit: {cache_path}")
                audio, sr = torchaudio.load(cache_path)
                assert sr == self.sr
                audio = audio.to(theta_d.device)
                idx_to_audio[idx] = audio
            else:
                uncached_indices.append(idx)
        if uncached_indices:
            uncached_d = theta_d[uncached_indices]
            uncached_s = theta_s[uncached_indices]
            uncached_seed = seed[uncached_indices]
            audio = self._forward(uncached_d, uncached_s, uncached_seed).detach()
            for idx, a, theta_d, theta_s, s in zip(
                uncached_indices, audio, uncached_d, uncached_s, uncached_seed
            ):
                idx_to_audio[idx] = a
                cache_name = (
                    f"flowtron_{s.item()}_{theta_d.item():.6f}_{theta_s.item():.6f}.wav"
                )
                cache_path = os.path.join(self.cache_dir, cache_name)
                assert not os.path.isfile(cache_path)
                torchaudio.save(cache_path, a.cpu(), self.sr)
                # if os.path.isfile(cache_path):
                #     audio_cached, sr = torchaudio.load(cache_path)
                #     assert sr == self.sr
                #     audio_cached = audio_cached.to(theta_sig.device)
                #     assert tr.allclose(a, audio_cached, atol=1e-4)
        assert len(idx_to_audio) == bs
        audio = [idx_to_audio[idx] for idx in range(bs)]
        audio = tr.stack(audio, dim=0)
        return audio

    def forward(
        self,
        theta_d: T,
        theta_s: T,
        seed: Optional[T] = None,
        use_cache: bool = False,
    ) -> T:
        if use_cache:
            assert False  # TODO(cm): tmp
            assert not self.wordlist, "Caching is not supported with wordlist"
            return self._forward_cached(theta_d, theta_s, seed)
        else:
            return self._forward(theta_d, theta_s, seed)

    def make_x_from_theta(
        self, theta_d_0to1: T, theta_s_0to1: T, seed: T, use_cache: bool = False
    ) -> T:
        assert theta_d_0to1.min() >= 0.0
        assert theta_d_0to1.max() <= 1.0
        assert theta_s_0to1.min() >= 0.0
        assert theta_s_0to1.max() <= 1.0
        audio = self.forward(theta_d_0to1, theta_s_0to1, seed, use_cache)
        return audio


class FastSpeech2Synth(nn.Module):
    def __init__(
        self,
        vocoder_path: str,
        default_text: str = "Pigs are flying!",
        wordlist_path: Optional[str] = None,
        n_words: int = 4,
        n_frames: int = 128,
        min_pace: float = 0.8889,
        max_pace: float = 1.125,
        min_pitch_rate: float = 0.25,
        max_pitch_rate: float = 4.0,
        min_energy_rate: float = 0.125,
        max_energy_rate: float = 8.0,
    ):
        super().__init__()
        self.default_text = default_text
        self.n_words = n_words
        self.n_frames = n_frames
        self.register_buffer("min_pace_log", tr.log(tr.tensor(min_pace)))
        self.register_buffer("max_pace_log", tr.log(tr.tensor(max_pace)))
        self.register_buffer("min_pitch_rate_log", tr.log(tr.tensor(min_pitch_rate)))
        self.register_buffer("max_pitch_rate_log", tr.log(tr.tensor(max_pitch_rate)))
        self.register_buffer("min_energy_rate_log", tr.log(tr.tensor(min_energy_rate)))
        self.register_buffer("max_energy_rate_log", tr.log(tr.tensor(max_energy_rate)))

        self.rand_gen_cpu = tr.Generator(device="cpu")
        self.rand_gen_gpu = None
        if tr.cuda.is_available():
            self.rand_gen_gpu = tr.Generator(device="cuda")

        self.hop_len = 256  # TODO(cm): tmp
        self.Q = 24  # TODO(cm): tmp

        log.info(f"Loading FastSpeech2")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if tr.cuda.is_available():
            run_opts = {"device": "cuda"}
        else:
            run_opts = {"device": "cpu"}
        self.model = FastSpeech2.from_hparams(
            source="speechbrain/tts-fastspeech2-ljspeech",
            savedir="pretrained_models/tts-fastspeech2-ljspeech",
            run_opts=run_opts,
        )
        self.model.eval()
        self.sr = self.model.hparams.sample_rate

        log.info(f"Loading Vocoder")
        vocoder_state_dict = tr.load(vocoder_path, map_location="cpu")["generator"]
        vocoder_config_path = os.path.join(MODELS_DIR, "hifigan_t2_v3_config.json")
        with open(vocoder_config_path, "r") as f:
            vocoder_config = json.load(f)
            hyperparams = AttrDict(vocoder_config)
        vocoder = Generator(hyperparams)
        vocoder.load_state_dict(vocoder_state_dict)
        vocoder.eval()
        vocoder.remove_weight_norm()
        self.vocoder = vocoder
        log.info(f"Finished loading models")

        self.wordlist = []
        if wordlist_path is not None:
            with open(wordlist_path, "r") as f:
                self.wordlist = json.load(f)
            log.info(f"Wordlist contains {len(self.wordlist)} words")

    def get_rand_gen(self, device: str) -> tr.Generator:
        if device == "cpu":
            return self.rand_gen_cpu
        else:
            return self.rand_gen_gpu

    def forward(
        self,
        theta_d: T,
        theta_s: T,
        seed: Optional[T] = None,
        seed_words: Optional[T] = None,
    ) -> T:
        assert theta_d.ndim == theta_s.ndim == 1
        if theta_d.device != self.model.g2p.device:
            self.model.g2p.device = theta_d.device
            for mod in self.model.g2p.mods.values():
                if hasattr(mod, "to"):
                    mod.to(theta_d.device)

        # rand_gen = self.get_rand_gen(device=theta_d.device.type)
        rand_gen = self.rand_gen_cpu
        if seed is not None:
            assert seed.shape == theta_d.shape
        if seed_words is not None:
            assert seed_words.shape == theta_d.shape

        bs = theta_d.size(0)
        if self.wordlist:
            batch_text = []
            for idx in range(bs):
                if seed_words is not None:
                    rand_gen.manual_seed(int(seed_words[idx].item()))
                # if seed is not None:
                #     rand_gen.manual_seed(int(seed[idx].item()))
                words = tr.randint(
                    0, len(self.wordlist), (self.n_words,), generator=rand_gen
                )
                words = [self.wordlist[w] for w in words]
                text = " ".join(words)
                # log.info(f"Random text: {text}")
                batch_text.append(text)
        else:
            batch_text = [self.default_text] * bs

        pace_theta_0to1 = []
        for idx in range(bs):
            if seed is not None:
                rand_gen.manual_seed(int(seed[idx].item()))
            pace_0to1 = tr.rand((1,), generator=rand_gen)
            pace_theta_0to1.append(pace_0to1)
        pace_theta_0to1 = tr.cat(pace_theta_0to1, dim=0).to(theta_d.device)
        pace = (
            pace_theta_0to1 * (self.max_pace_log - self.min_pace_log)
            + self.min_pace_log
        )
        pace = tr.exp(pace).view(-1, 1)
        # log.info(f"pace: {pace}")

        # pace_theta = theta_d
        # pace = pace_theta * (self.max_pace_log - self.min_pace_log) + self.min_pace_log
        # pace = tr.exp(pace).view(-1, 1)
        # log.info(f"pace: {pace}")
        # pace = 1.0

        pitch_rate_theta = theta_d
        # pitch_rate = pitch_rate_theta.view(-1, 1, 1)
        pitch_rate = (
            pitch_rate_theta * (self.max_pitch_rate_log - self.min_pitch_rate_log)
            + self.min_pitch_rate_log
        )
        pitch_rate = tr.exp(pitch_rate).view(-1, 1, 1)
        # log.info(f"pitch_rate: {pitch_rate}")
        # pitch_rate = 1.0

        energy_rate_theta = theta_s
        # energy_rate = energy_rate_theta.view(-1, 1, 1)
        energy_rate = (
            energy_rate_theta * (self.max_energy_rate_log - self.min_energy_rate_log)
            + self.min_energy_rate_log
        )
        energy_rate = tr.exp(energy_rate).view(-1, 1, 1)
        # log.info(f"energy_rate: {energy_rate}")
        # energy_rate = 1.0

        mel_posterior, durations, pitch, energy, mel_lens = self.model.encode_text(
            batch_text,
            pace=pace,
            pitch_rate=pitch_rate,
            energy_rate=energy_rate,
        )
        # log.info(f"mel_posterior.shape: {mel_posterior.shape}")
        if tr.all(mel_lens == mel_lens[0]):
            mel_len = mel_lens[0]
            n_repeats = self.n_frames // mel_len + 1
            mel_posterior = mel_posterior.repeat(1, 1, n_repeats)
            mel_posterior = mel_posterior[..., : self.n_frames]
        else:
            mels = []
            for idx, mel_len in enumerate(mel_lens):
                mel = mel_posterior[idx, :, :mel_len]
                n_repeats = self.n_frames // mel_len + 1
                mel = mel.repeat(1, n_repeats)
                mel = mel[..., : self.n_frames]
                mels.append(mel)
            mel_posterior = tr.stack(mels, dim=0)
        audio = self.vocoder(mel_posterior)
        return audio

    def make_x_from_theta(
        self, theta_d_0to1: T, theta_s_0to1: T, seed: T, seed_words: Optional[T] = None
    ) -> T:
        assert theta_d_0to1.min() >= 0.0
        assert theta_d_0to1.max() <= 1.0
        assert theta_s_0to1.min() >= 0.0
        assert theta_s_0to1.max() <= 1.0
        audio = self.forward(theta_d_0to1, theta_s_0to1, seed, seed_words=seed_words)
        return audio


if __name__ == "__main__":
    # # seed = 41  # 0.24
    # # seed = 568  # 0.75
    # # seed = 1040  # 0.55
    # # seed = 391  # 0.99
    # seed = 867  # 0.37
    # generator = tr.Generator(device="cpu")
    # generator.manual_seed(seed)
    # derp = tr.rand((1,), generator=generator)
    # log.info(f"Seed {seed}: {derp}")
    # generator.manual_seed(seed)
    # derp = tr.rand((1,), generator=generator)
    # log.info(f"Seed {seed}: {derp}")
    # exit()

    # config_path = os.path.join(CONFIGS_DIR, "flowtron/flowtron_synth.yml")
    config_path = os.path.join(CONFIGS_DIR, "fastspeech2/fastspeech2_synth.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # synth = FlowtronSynth(**config["init_args"])
    synth = FastSpeech2Synth(**config["init_args"])

    # theta_d = tr.full((1,), 1.0)
    # theta_d = tr.tensor([0.25, 0.5, 1.0, 2.0, 4.0])
    theta_d = tr.tensor([0.0, 0.25, 0.50, 0.75, 1.0])
    # theta_s = tr.tensor([0.0, 0.25, 0.50, 0.75, 1.0])
    # theta_d = tr.tensor([0.5, 0.75, 1.0, 1.25, 1.5])
    # theta_s = tr.tensor([0.01, 0.1, 1.0, 5.0, 10.0])
    # theta_d = tr.tensor([0.1, 1.0, 1.5, 2.0])
    # theta_d = tr.full_like(theta_s, 0.0)
    # theta_d = tr.full_like(theta_s, 0.5)
    theta_s = tr.full_like(theta_d, 0.5)
    seed = tr.full_like(theta_d, 0).long()

    with tr.no_grad():
        audio = synth(theta_d, theta_s, seed)

    for a, curr_d, curr_s, curr_seed in zip(audio, theta_d, theta_s, seed):
        log.info(f"a.max(): {a.max().item()}")
        # save_name = f"flowtron_d{curr_d.item():.2f}_s{curr_s.item():.2f}_{curr_seed.item()}.wav"
        # save_name = f"flowtron_ms_{curr_seed.item()}_d{curr_d.item():.2f}_s{curr_s.item():.2f}.wav"
        # save_name = f"fs2_p_{curr_seed.item()}_d{curr_d.item():.2f}_s{curr_s.item():.2f}_t2.wav"
        save_name = (
            f"fs2_p_{curr_seed.item()}_d{curr_d.item():.2f}_s{curr_s.item():.2f}.wav"
        )
        # save_name = f"fs2_energy_{curr_seed.item()}_d{curr_d.item():.2f}_s{curr_s.item():.2f}.wav"
        save_path = os.path.join(OUT_DIR, save_name)
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
