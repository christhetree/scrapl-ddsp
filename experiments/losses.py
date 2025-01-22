import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union, Any, Optional, Dict, List

import scipy
import torch as tr
import torch.nn as nn

from experiments.util import ReadOnlyTensorDict
from kymatio.torch import Scattering1D, TimeFrequencyScattering
from msclap import CLAP
from torch import Tensor as T
from torch.nn import functional as F
from torchaudio.transforms import Resample
from transformers import Wav2Vec2Model

from experiments import util
from scrapl.torch import TimeFrequencyScrapl

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


class SCRAPLLoss(nn.Module):
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
        p: int = 2,
        sample_all_paths_first: bool = False,
    ):
        super().__init__()
        self.p = p
        self.sample_all_paths_first = sample_all_paths_first

        self.jtfs = TimeFrequencyScrapl(
            shape=(shape,),
            J=J,
            Q=(Q1, Q2),
            Q_fr=Q_fr,
            J_fr=J_fr,
            T=T,
            F=F,
        )
        self.meta = self.jtfs.meta()
        if len(self.meta["key"]) != len(self.meta["order"]):
            self.meta["key"] = self.meta["key"][1:]
        # TODO(cm): check if this is correct
        self.scrapl_keys = [key for key in self.meta["key"] if len(key) == 2]
        self.n_paths = len(self.scrapl_keys)
        # self.n_paths = 8
        log.info(f"number of SCRAPL keys = {self.n_paths}")
        self.path_counts = defaultdict(int)

    def sample_path(self) -> int:
        if self.sample_all_paths_first and len(self.path_counts) < self.n_paths:
            path_idx = len(self.path_counts)
        else:
            path_idx = tr.randint(0, self.n_paths, (1,)).item()
        self.path_counts[path_idx] += 1
        return path_idx

    def calc_dist(self, x: T, x_target: T, path_idx: int) -> (T, T, T):
        n2, n_fr = self.scrapl_keys[path_idx]
        Sx = self.jtfs.scattering_singlepath(x, n2, n_fr)
        Sx = Sx["coef"].squeeze(-1)
        Sx_target = self.jtfs.scattering_singlepath(x_target, n2, n_fr)
        Sx_target = Sx_target["coef"].squeeze(-1)
        # from matplotlib import pyplot as plt
        # plt_Sx = Sx[0, 0, ...].detach().cpu().numpy()
        # plt_Sx_target = Sx_target[0, 0, ...].detach().cpu().numpy()
        # vmin = min(plt_Sx.min(), plt_Sx_target.min())
        # vmax = max(plt_Sx.max(), plt_Sx_target.max())
        # plt.imshow(plt_Sx, aspect="auto", interpolation="none", origin="lower", vmin=vmin, vmax=vmax)
        # plt.title("Sx")
        # plt.show()
        # plt.imshow(plt_Sx_target, aspect="auto", interpolation="none", origin="lower", vmin=vmin, vmax=vmax)
        # plt.title("Sx_target")
        # plt.show()
        diff = Sx_target - Sx
        dist = tr.linalg.vector_norm(diff, ord=self.p, dim=(-2, -1))
        dist = tr.mean(dist)
        return dist, Sx, Sx_target

    def forward(self, x: T, x_target: T, path_idx: Optional[int] = None) -> T:
        assert x.ndim == x_target.ndim == 3
        assert x.size(1) == x_target.size(1) == 1
        if path_idx is None:
            path_idx = self.sample_path()
        else:
            log.info(f"Using specified path_idx = {path_idx}")
            assert 0 <= path_idx < self.n_paths
        dist, _, _ = self.calc_dist(x, x_target, path_idx)
        return dist


class AdaptiveSCRAPLLoss(SCRAPLLoss):
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
        p: int = 2,
        sample_all_paths_first: bool = False,
        min_prob_fac: float = 0.0,
        probs_path: Optional[str] = None,
        get_path_keys_kw_args: Optional[Dict[str, Any]] = None,
        update_prob_is_bin: bool = False,
        eps: float = 1e-12,
    ):
        super().__init__(shape, J, Q1, Q2, J_fr, Q_fr, T, F, p, sample_all_paths_first)
        assert 0 <= min_prob_fac <= 1.0
        self.min_prob_fac = min_prob_fac
        self.get_path_indices_kw_args = get_path_keys_kw_args
        self.update_prob_is_bin = update_prob_is_bin
        self.eps = eps

        self.unif_prob = 1.0 / self.n_paths
        self.min_prob = self.unif_prob * min_prob_fac
        self.log_1m_min_prob_frac = tr.tensor(
            1.0 - self.min_prob_fac, dtype=tr.double
        ).log()
        log.info(f"unif_prob = {self.unif_prob:.8f}, min_prob = {self.min_prob:.8f}")
        self.curr_path_idx = None
        self.updated_indices = set()
        # Use log and double precision to prevent cumulative floating point errors
        # TODO(cm): register buffer, but cpu, for checkpointing
        self.log_vals = tr.log(tr.full((self.n_paths,), self.eps, dtype=tr.double))
        self.log_probs = tr.log(
            tr.full((self.n_paths,), self.unif_prob, dtype=tr.double)
        )
        self.log_eps = tr.log(tr.tensor(self.eps, dtype=tr.double))

        self.enabled_path_indices = []
        if probs_path is not None:
            probs = tr.load(probs_path).double()
            assert probs.shape == (self.n_paths,)
            log.info(f"Loading probs from {probs_path}, min = {probs.min():.6f}, "
                     f"max = {probs.max():.6f}, mean = {probs.mean():.6f}")
            self.probs = probs
        elif get_path_keys_kw_args is not None:
            log.info(f"Disabling a subset of paths")
            path_keys = util.get_path_keys(
                meta=self.meta, Q1=Q1, **get_path_keys_kw_args
            )
            assert path_keys, f"no path keys found for {get_path_keys_kw_args}"
            log.info(f"len(path_keys) = {len(path_keys)}")
            path_indices = []
            probs = tr.zeros((self.n_paths,), dtype=tr.double)
            for k in path_keys:
                assert k in self.scrapl_keys
                path_idx = self.scrapl_keys.index(k)
                path_indices.append(path_idx)
                probs[path_idx] = 1.0
                self.enabled_path_indices.append(path_idx)
            log.info(f"path_indices = {path_indices}")
            probs /= probs.sum()
            self.probs = probs
        else:
            self.probs = tr.full((self.n_paths,), self.unif_prob, dtype=tr.double)

    def sample_path(self) -> int:
        if self.sample_all_paths_first and len(self.path_counts) < self.n_paths:
            path_idx = len(self.path_counts)
        else:
            assert tr.allclose(
                self.probs.sum(), tr.tensor(1.0, dtype=tr.double), atol=self.eps
            ), f"self.probs.sum() = {self.probs.sum()}"
            path_idx = tr.multinomial(self.probs, 1).item()
        self.path_counts[path_idx] += 1
        self.curr_path_idx = path_idx
        return path_idx

    def _get_log_probs(self, log_vals: T) -> T:
        log_probs = log_vals - tr.logsumexp(log_vals, dim=0)
        return log_probs

    def update_prob(self, path_idx: int, val: float) -> None:
        assert 0 <= path_idx < self.n_paths
        assert val >= 0.0
        self.updated_indices.add(path_idx)
        log_val = tr.tensor(val, dtype=tr.double).clamp(min=self.eps).log()
        self.log_vals[path_idx] = log_val
        updated_indices = list(self.updated_indices)
        n_updated = len(updated_indices)
        if n_updated < 2:
            # Only one prob has been updated, so no change required
            log_probs = self.log_probs
        elif n_updated == self.n_paths:
            # All probs have been updated
            log_probs = self._get_log_probs(self.log_vals)
        else:
            # Balance between updated probs and initial uniform probs
            self.log_probs[updated_indices] = self.log_eps
            updated_frac = tr.tensor(n_updated / self.n_paths, dtype=tr.double)
            stale_frac = 1.0 - updated_frac
            updated_probs = self._get_log_probs(self.log_vals) + updated_frac.log()
            stale_probs = self._get_log_probs(self.log_probs) + stale_frac.log()
            log_probs = stale_probs
            log_probs[updated_indices] = updated_probs[updated_indices]
            log_probs = self._get_log_probs(log_probs)

        # Ensure min_prob is enforced
        log_probs += self.log_1m_min_prob_frac
        probs = tr.exp(log_probs) + self.min_prob
        # log.info(f"probs.min() = {probs.min()}, "
        #          f"probs.max() = {probs.max()}, "
        #          f"probs.sum() = {probs.sum()}")

        if self.update_prob_is_bin:
            less_than_unif = probs < self.unif_prob
            n_less_than_unif = less_than_unif.sum().item()
            log.info(f"n_less_than_unif = {n_less_than_unif}")
            new_unif_prob = 1 / (self.n_paths - n_less_than_unif)
            probs[less_than_unif] = 0.0
            probs[~less_than_unif] = new_unif_prob

        self.probs = probs

        # Check that the probabilities are valid
        assert tr.allclose(
            probs.sum(), tr.tensor(1.0, dtype=tr.double), atol=self.eps
        ), f"probs.sum() = {probs.sum()}"

        if not self.update_prob_is_bin:
            assert probs.min() >= self.min_prob - self.eps, f"probs.min() = {probs.min()}"

            # Check ratios
            vals = self.log_vals[updated_indices].exp()
            val_ratios = vals / vals.min()
            raw_probs = self.probs[updated_indices] - self.min_prob
            prob_ratios = raw_probs / raw_probs.min()
            # log.info(f"val_ratios = {val_ratios}, prob_ratios = {prob_ratios}")
            assert tr.allclose(
                val_ratios, prob_ratios, atol=1e-3, rtol=1e-3, equal_nan=True
            ), f"val_ratios = {val_ratios}, prob_ratios = {prob_ratios}"


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
            dist = tr.linalg.vector_norm(log_Sx_target - log_Sx, ord=self.p, dim=(-2, -1))
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

    scrapl = AdaptiveSCRAPLLoss(
        shape=32768,
        J=12,
        Q1=8,
        Q2=2,
        J_fr=3,
        Q_fr=2,
        sample_all_paths_first=False,
        min_prob_fac=0.25,
        # min_prob_fac=0.0,
    )
    # n_iter = 10000
    # for _ in range(n_iter):
    #     path_idx = scrapl.sample_path()
    #     log.info(f"path_idx = {path_idx}")
    #     val = tr.rand(1).item() * 1e-8
    #     scrapl.update_prob(path_idx, val)

    scrapl.update_prob(4, 1.0)
    scrapl.update_prob(5, 3.0)
    scrapl.update_prob(6, 3.0)
    # scrapl.update_prob(6, 1.0)
    scrapl.update_prob(7, 1.0)
    scrapl.update_prob(0, 1.0)
    scrapl.update_prob(1, 1.0)
    scrapl.update_prob(2, 1.0)
    scrapl.update_prob(3, 1.0)
    scrapl.update_prob(4, 1.0)
    scrapl.update_prob(5, 1.0)
    scrapl.update_prob(6, 10.0)

    exit()

    # w2v2_loss = Wav2Vec2Loss()
    # x = tr.randn(3, 1, 4000) * 3.0
    # w2v2_loss.get_embedding(x)
    # exit()

    # tr.manual_seed(0)
    n = 10000
    logits = tr.tensor([0.9, 0.1, 0.0])
    log.info(f"softmax of logits = {F.softmax(logits, dim=0)}")
    results = tr.zeros_like(logits)

    for _ in range(n):
        one_hot = F.gumbel_softmax(logits, hard=True)
        idx = one_hot.argmax().item()
        results[idx] += 1

    results /= n
    log.info(f"results gumbel = {results}")

    # Now sample using multinomial
    results = tr.zeros_like(logits)
    for _ in range(n):
        # idx = tr.multinomial(F.softmax(logits, dim=0), 1).item()
        idx = tr.multinomial(tr.tensor([0.8, 0.1, 0.0]), 1).item()
        results[idx] += 1

    results /= n
    log.info(f"results multinomial = {results}")
