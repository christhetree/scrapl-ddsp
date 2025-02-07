import logging
import os
from abc import ABC, abstractmethod
from typing import Union, Optional, List

import scipy
import torch as tr
import torch.nn as nn
from msclap import CLAP
from torch import Tensor as T
from torchaudio.transforms import Resample
from transformers import Wav2Vec2Model

from experiments.util import ReadOnlyTensorDict
from kymatio.torch import Scattering1D, TimeFrequencyScattering

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class JTFSTLoss(nn.Module):
    def __init__(
        self,
        shape: int,
        J: int,
        Q1: int,
        Q2: int,
        J_fr: int,
        Q_fr: int,
        T: Optional[Union[str, int]] = None,
        F: Optional[Union[str, int]] = None,
        format_: str = "joint",
        p: int = 2,
    ):
        super().__init__()
        assert format_ in ["time", "joint"]
        self.format = format_
        self.p = p
        self.jtfs = TimeFrequencyScattering(
            shape=(shape,),
            J=J,
            Q=(Q1, Q2),
            Q_fr=Q_fr,
            J_fr=J_fr,
            T=T,
            F=F,
            format=format_,
        )
        jtfs_meta = self.jtfs.meta()
        jtfs_keys = [key for key in jtfs_meta["key"] if len(key) == 2]
        log.info(f"number of JTFS keys = {len(jtfs_keys)}")

    def forward(self, x: T, x_target: T) -> T:
        assert x.ndim == x_target.ndim == 3
        assert x.size(1) == x_target.size(1) == 1
        Sx = self.jtfs(x)
        Sx_target = self.jtfs(x_target)
        if self.format == "time":
            Sx = Sx[:, :, 1:, :]  # Remove the 0th order coefficients
            Sx_target = Sx_target[:, :, 1:, :]  # Remove the 0th order coefficients
            dist = tr.linalg.vector_norm(Sx_target - Sx, ord=self.p, dim=-1)
        else:
            dist = tr.linalg.vector_norm(Sx_target - Sx, ord=self.p, dim=(-2, -1))
        dist = tr.mean(dist)
        return dist


class Scat1DLoss(nn.Module):
    def __init__(
        self,
        shape: int,
        J: int,
        Q1: int,
        Q2: int = 1,
        T: Optional[Union[str, int]] = None,
        max_order: int = 1,
        p: int = 2,
    ):
        super().__init__()
        self.max_order = max_order
        self.p = p
        self.scat_1d = Scattering1D(
            shape=(shape,),
            J=J,
            Q=(Q1, Q2),
            T=T,
            max_order=max_order,
        )

    def forward(self, x: T, x_target: T) -> T:
        assert x.ndim == x_target.ndim == 3
        assert x.size(1) == x_target.size(1) == 1
        Sx = self.scat_1d(x)
        Sx_target = self.scat_1d(x_target)
        Sx = Sx[:, :, 1:, :]  # Remove the 0th order coefficients
        Sx_target = Sx_target[:, :, 1:, :]  # Remove the 0th order coefficients

        if self.max_order == 1:
            dist = tr.linalg.vector_norm(Sx_target - Sx, ord=self.p, dim=(-2, -1))
        else:
            dist = tr.linalg.vector_norm(Sx_target - Sx, ord=self.p, dim=-1)

        dist = tr.mean(dist)
        return dist


class EmbeddingLoss(ABC, nn.Module):
    def __init__(self, use_time_varying: bool = True, in_sr: int = 44100, p: int = 2):
        super().__init__()
        self.use_time_varying = use_time_varying
        self.in_sr = in_sr
        self.p = p
        self.resampler = None
        self.set_resampler(in_sr)

    def set_resampler(self, in_sr: int) -> None:
        self.in_sr = in_sr
        if in_sr != self.get_model_sr():
            self.resampler = Resample(orig_freq=in_sr, new_freq=self.get_model_sr())
        else:
            self.resampler = None

    def preproc_audio(self, x: T) -> T:
        if self.resampler is not None:
            x = self.resampler(x)
        n_samples = x.size(-1)
        model_n_samples = self.get_model_n_samples()
        if model_n_samples == -1:  # Model can handle any number of samples
            return x
        if n_samples < model_n_samples:
            n_repeats = model_n_samples // n_samples + 1
            x = x.repeat(1, n_repeats)
        x = x[:, :model_n_samples]
        return x

    @abstractmethod
    def get_model_sr(self) -> int:
        pass

    @abstractmethod
    def get_model_n_samples(self) -> int:
        pass

    @abstractmethod
    def get_embedding(self, x: T) -> T:
        pass

    def forward(self, x: T, x_target: T) -> T:
        assert x.ndim == x_target.ndim == 3
        assert x.size(1) == x_target.size(1) == 1
        x = x.squeeze(1)
        x_target = x_target.squeeze(1)
        x = self.preproc_audio(x)
        x_target = self.preproc_audio(x_target)
        x_emb = self.get_embedding(x)
        x_target_emb = self.get_embedding(x_target)
        if self.use_time_varying:
            assert x_emb.ndim == x_target_emb.ndim == 3
        elif x_emb.ndim == 3:
            x_emb = x_emb.mean(dim=1)
            x_target_emb = x_target_emb.mean(dim=1)
        diff = x_target_emb - x_emb
        if self.use_time_varying:
            assert diff.ndim == 3
            dist = tr.linalg.vector_norm(diff, ord=self.p, dim=(-2, -1))
        else:
            assert diff.ndim == 2
            dist = tr.linalg.vector_norm(diff, ord=self.p, dim=-1)
        dist = tr.mean(dist)
        return dist


class ClapEmbeddingLoss(EmbeddingLoss):
    def __init__(self, use_cuda: bool, in_sr: int = 44100, p: int = 2):
        self.model = CLAP(version="2023", use_cuda=use_cuda)
        use_time_varying = False  # CLAP is not a time-varying embedding
        super().__init__(use_time_varying, in_sr, p)

    def get_model_sr(self) -> int:
        return self.model.args.sampling_rate

    def get_model_n_samples(self) -> int:
        dur = self.model.args.duration
        n_samples = dur * self.get_model_sr()
        return n_samples

    def get_embedding(self, x: T) -> T:
        x_emb, _ = self.model.clap.audio_encoder(x)
        return x_emb


class Wav2Vec2EmbeddingLoss(EmbeddingLoss):
    def __init__(
        self,
        model_size: str = "base",
        normalize: bool = False,
        eps: float = 1e-8,
        use_time_varying: bool = True,
        in_sr: int = 44100,
        p: int = 2,
    ):
        super().__init__(use_time_varying, in_sr, p)
        self.normalize = normalize
        self.eps = eps
        self.model_size = model_size
        huggingface_id = f"facebook/wav2vec2-{model_size}-960h"
        self.model = Wav2Vec2Model.from_pretrained(huggingface_id)
        # self.processor = AutoProcessor.from_pretrained(huggingface_id)

    def get_model_sr(self) -> int:
        return 16000

    def get_model_n_samples(self) -> int:
        return -1

    def get_embedding(self, x: T) -> T:
        x = x.squeeze(1)
        # x2 = self.processor(x.numpy(), return_tensors="pt").data["input_values"]
        if self.normalize:
            mu = tr.mean(x, dim=-1, keepdim=True)
            std = tr.std(x, dim=-1, keepdim=True)
            x = (x - mu) / (std + self.eps)
        # assert tr.allclose(x, x2, atol=1e-3)
        emb = self.model(x).last_hidden_state
        # TODO(cm): this results in NaN, look into minimum sample length
        log.info(
            f"emb.shape = {emb.shape}, emb.min() = {emb.min().item()}, emb.max() = {emb.max().item()}"
        )
        return emb


class LogMSSLoss(nn.Module):
    def __init__(
        self,
        fft_sizes: Optional[List[int]] = None,
        hop_sizes: Optional[List[int]] = None,
        win_lengths: Optional[List[int]] = None,
        window: str = "flat_top",
        log_mag_eps: float = 1.0,
        gamma: float = 1.0,
        p: int = 2,
    ):
        super().__init__()
        if win_lengths is None:
            win_lengths = [67, 127, 257, 509, 1021, 2053]
            log.info(f"win_lengths = {win_lengths}")
        if fft_sizes is None:
            fft_sizes = win_lengths
            log.info(f"fft_sizes = {fft_sizes}")
        if hop_sizes is None:
            hop_sizes = [w // 2 for w in win_lengths]
            log.info(f"hop_sizes = {hop_sizes}")
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.window = window
        self.log_mag_eps = log_mag_eps
        self.gamma = gamma
        self.p = p
        # Create windows
        windows = {}
        for win_length in win_lengths:
            win = self.make_window(window, win_length)
            windows[win_length] = win
        self.windows = ReadOnlyTensorDict(windows)

    def forward(self, x: T, x_target: T) -> T:
        assert x.ndim == x_target.ndim == 3
        assert x.size(1) == x_target.size(1) == 1
        x = x.squeeze(1)
        x_target = x_target.squeeze(1)
        dists = []
        for fft_size, hop_size, win_length in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths
        ):
            win = self.windows[win_length]
            Sx = tr.stft(
                x,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_length,
                window=win,
                return_complex=True,
            ).abs()
            Sx_target = tr.stft(
                x_target,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_length,
                window=win,
                return_complex=True,
            ).abs()
            if self.log_mag_eps == 1.0:
                log_Sx = tr.log1p(self.gamma * Sx)
                log_Sx_target = tr.log1p(self.gamma * Sx_target)
            else:
                log_Sx = tr.log(self.gamma * Sx + self.log_mag_eps)
                log_Sx_target = tr.log(self.gamma * Sx_target + self.log_mag_eps)
            dist = tr.linalg.vector_norm(
                log_Sx_target - log_Sx, ord=self.p, dim=(-2, -1)
            )
            dists.append(dist)
        dist = tr.stack(dists, dim=1).sum(dim=1)
        dist = dist.mean()  # Aggregate the batch dimension
        return dist

    @staticmethod
    def make_window(window: str, n: int) -> T:
        if window == "rect":
            return tr.ones(n)
        elif window == "hann":
            return tr.hann_window(n)
        elif window == "flat_top":
            window = scipy.signal.windows.flattop(n, sym=False)
            window = tr.from_numpy(window).float()
            return window
        else:
            raise ValueError(f"Unknown window type: {window}")


if __name__ == "__main__":
    audio = tr.randn(3, 1, 4000)
    audio_target = tr.randn(3, 1, 4000)
    mss = LogMSSLoss()
    mss(audio, audio_target)
    exit()

    # w2v2_loss = Wav2Vec2Loss()
    # x = tr.randn(3, 1, 4000) * 3.0
    # w2v2_loss.get_embedding(x)
    # exit()
