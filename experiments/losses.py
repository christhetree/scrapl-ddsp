import logging
import os
from abc import ABC, abstractmethod
from typing import Union, Optional, List, Literal

import scipy
import torch as tr
import torch.nn as nn
from msclap import CLAP
from torch import Tensor as T
from torchaudio.transforms import Resample, MFCC
from transformers import Wav2Vec2Model

from experiments.panns.model_loader import PANNsModel
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
        use_log1p: bool = False,
        log1p_eps: float = 1e-3,  # TODO(cm): what's a good default here?
    ):
        super().__init__()
        assert format_ in ["time", "joint"]
        self.format = format_
        self.p = p
        self.use_log1p = use_log1p
        self.log1p_eps = log1p_eps
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
        bs, n_ch, n_samples = x.size()
        x = x.view(-1, 1, n_samples)
        x_target = x_target.view(-1, 1, n_samples)
        assert x.size(1) == x_target.size(1) == 1
        Sx = self.jtfs(x)
        x_target = x_target.contiguous()
        Sx_target = self.jtfs(x_target)
        if self.use_log1p:
            Sx = tr.log1p(Sx / self.log1p_eps)
            Sx_target = tr.log1p(Sx_target / self.log1p_eps)
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
        bs, n_ch, n_samples = x.size()
        x = x.view(-1, 1, n_samples)
        x_target = x_target.view(-1, 1, n_samples)
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
    def __init__(self, use_time_varying: bool = False, in_sr: int = 44100, p: int = 2):
        super().__init__()
        assert not use_time_varying  # TODO(cm): tmp
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
            # TODO(cm): does this make sense?
            dist = tr.linalg.vector_norm(diff, ord=self.p, dim=(-2, -1))
        else:
            assert diff.ndim == 2
            dist = tr.linalg.vector_norm(diff, ord=self.p, dim=-1)
        dist = tr.mean(dist)
        return dist


class ClapEmbeddingLoss(EmbeddingLoss):
    def __init__(self, use_cuda: bool, in_sr: int = 44100, p: int = 2):
        self.model = CLAP(version="2023", use_cuda=use_cuda)  # Not an nn.Module
        super().__init__(use_time_varying=False, in_sr=in_sr, p=p)
        assert len(list(self.parameters())) == 0

    def get_model_sr(self) -> int:
        return self.model.args.sampling_rate

    def get_model_n_samples(self) -> int:
        dur = self.model.args.duration
        n_samples = dur * self.get_model_sr()
        return n_samples

    def get_embedding(self, x: T) -> T:
        x_emb, _ = self.model.clap.audio_encoder(x)
        return x_emb


class PANNsEmbeddingLoss(EmbeddingLoss):
    def __init__(
        self,
        variant: Literal["cnn14-32k", "cnn14-16k", "wavegram-logmel"],
        in_sr: int = 44100,
        p: int = 2,
    ):
        self.variant = variant
        self.model_sr = 16000 if variant == "cnn14-16k" else 32000
        super().__init__(use_time_varying=False, in_sr=in_sr, p=p)
        # PANNsModel is a nn.Module, hence needs to be added after super()
        self.model = PANNsModel(variant=variant)
        self.model.load_model()
        for param in self.parameters():
            param.requires_grad = False
        log.info(f"Froze {len(list(self.parameters()))} parameter tensors")
        self.eval()

    def get_model_sr(self) -> int:
        return self.model_sr

    def get_model_n_samples(self) -> int:
        return -1

    def get_embedding(self, x: T) -> T:
        x_emb = self.model.get_embedding(x)
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


class MFCCDistance(nn.Module):
    def __init__(
        self,
        sr: int,
        log_mels: bool = True,
        n_fft: int = 2048,
        hop_len: int = 512,
        n_mels: int = 128,
        n_mfcc: int = 40,
        p: int = 1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.p = p

        self.mfcc = MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            log_mels=log_mels,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_len,
                "n_mels": n_mels,
            },
        )
        self.l1 = nn.L1Loss(reduction=reduction)
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, x: T, x_target: T) -> T:
        assert x.ndim == 3
        assert x.shape == x_target.shape
        if self.p == 1:
            return self.l1(self.mfcc(x), self.mfcc(x_target))
        elif self.p == 2:
            return self.mse(self.mfcc(x), self.mfcc(x_target))
        else:
            raise ValueError(f"Unknown p value: {self.p}")


if __name__ == "__main__":
    audio = tr.randn(3, 1, 32768)
    audio_target = tr.randn(3, 1, 32768)
    # panns = PANNsEmbeddingLoss(variant="cnn14-16k", in_sr=8192)
    # panns = PANNsEmbeddingLoss(variant="cnn14-32k", in_sr=8192)
    panns = PANNsEmbeddingLoss(variant="wavegram-logmel", in_sr=8192)
    # emb = panns.get_embedding(audio)
    # log.info(f"emb.shape = {emb.shape}")
    loss = panns.forward(audio, audio_target)
    log.info(f"loss = {loss}")


    # mss = LogMSSLoss()
    # mss(audio, audio_target)
    exit()

    # w2v2_loss = Wav2Vec2Loss()
    # x = tr.randn(3, 1, 4000) * 3.0
    # w2v2_loss.get_embedding(x)
    # exit()
