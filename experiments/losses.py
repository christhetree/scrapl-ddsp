import logging
import os
from collections import defaultdict
from typing import Union, Any, Optional

import torch as tr
import torch.nn as nn
from kymatio.torch import Scattering1D, TimeFrequencyScattering
from torch import Tensor as T
from torch.autograd import Function
from torch.nn import functional as F

from dwt import dwt_2d
from experiments import util
from jtfst_implementation.python.jtfst import JTFST2D
from scrapl.torch import TimeFrequencyScrapl
from wavelets import MorletWavelet

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
        format_: str = "time",
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
            dist = tr.linalg.norm(Sx_target - Sx, ord=self.p, dim=-1)
        else:
            dist = tr.linalg.norm(Sx_target - Sx, ord=self.p, dim=(-2, -1))
        dist = tr.mean(dist)
        return dist


class MyJTFST2DLoss(nn.Module):
    def __init__(
        self, sr: float, J: int, Q1: int, Q2: int, J_fr: int, Q_fr: int, T: int, F: int
    ):
        super().__init__()
        should_avg_f = False
        should_avg_t = False
        if F > 1:
            should_avg_f = True
        if T > 1:
            should_avg_t = True

        self.jtfs = JTFST2D(
            sr=sr,
            J_1=J,
            J_2_f=J_fr,
            J_2_t=J,
            Q_1=Q1,
            Q_2_f=Q_fr,
            Q_2_t=Q2,
            should_avg_f=should_avg_f,
            should_avg_t=should_avg_t,
            avg_win_f=F,
            avg_win_t=T,
            reflect_f=True,
        )

    def forward(self, x: T, x_target: T) -> T:
        assert x.ndim == x_target.ndim == 3
        assert x.size(1) == x_target.size(1) == 1
        Sx_o1, _, Sx, _ = self.jtfs(x)
        Sx_o1_target, _, Sx_target, _ = self.jtfs(x_target)
        dist = tr.linalg.vector_norm(Sx_target - Sx, ord=2, dim=(2, 3))
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
        scrapl_meta = self.jtfs.meta()
        # TODO(cm): check if this is correct
        self.scrapl_keys = [key for key in scrapl_meta["key"] if len(key) == 2]
        self.n_paths = len(self.scrapl_keys)
        log.info(f"number of SCRAPL keys = {self.n_paths}")
        self.path_counts = defaultdict(int)
        self.register_buffer("logits", tr.zeros((self.n_paths,)))

    def sample_path(self) -> int:
        if self.sample_all_paths_first and len(self.path_counts) < self.n_paths:
            path_idx = len(self.path_counts)
        else:
            with tr.no_grad():
                probs = F.softmax(self.logits, dim=0)
                path_idx = tr.multinomial(probs, 1).item()
        log.info(f"\npath_idx = {path_idx}")
        self.path_counts[path_idx] += 1
        return path_idx

    def calc_dist(self, x: T, x_target: T, path_idx: int) -> (T, T, T):
        n2, n_fr = self.scrapl_keys[path_idx]
        Sx = self.jtfs.scattering_singlepath(x, n2, n_fr)
        Sx = Sx["coef"].squeeze(-1)
        Sx_target = self.jtfs.scattering_singlepath(x_target, n2, n_fr)
        Sx_target = Sx_target["coef"].squeeze(-1)
        diff = Sx_target - Sx
        dist = tr.linalg.norm(diff, ord=self.p, dim=(-2, -1))
        dist = tr.mean(dist)
        return dist, Sx, Sx_target

    def forward(self, x: T, x_target: T) -> T:
        assert x.ndim == x_target.ndim == 3
        assert x.size(1) == x_target.size(1) == 1
        path_idx = self.sample_path()
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
        is_trainable: bool = True,
    ):
        super().__init__(shape, J, Q1, Q2, J_fr, Q_fr, T, F, p, sample_all_paths_first)
        if is_trainable:
            self.logits = nn.Parameter(tr.zeros((self.n_paths,)))

    def forward(self, x: T, x_target: T) -> T:
        assert x.ndim == x_target.ndim == 3
        assert x.size(1) == x_target.size(1) == 1
        path_idx = self.sample_path()
        dist, Sx, Sx_target = self.calc_dist(x, x_target, path_idx)
        dist = MakeLogitsGradFromEnergy.apply(
            self.logits, path_idx, dist, Sx, Sx_target
        )
        return dist


class MakeLogitsGradFromEnergy(Function):
    @staticmethod
    def forward(ctx: Any, logits: T, path_idx: int, dist: T, Sx: T, Sx_target) -> T:
        ctx.save_for_backward(logits, Sx, Sx_target)
        ctx.path_idx = path_idx
        return dist

    @staticmethod
    def backward(ctx: Any, grad_dist: T) -> (T, None, T, None, None):
        logits, Sx, Sx_target = ctx.saved_tensors
        path_idx = ctx.path_idx
        mean_target_energy = Sx_target.pow(2).sum()
        grad_logits = tr.zeros_like(logits)
        grad_logits[path_idx] = -mean_target_energy
        return grad_logits, None, grad_dist, None, None


class WaveletLoss(nn.Module):
    def __init__(self, sr: float, n_samples: int, J: int, Q1: int):
        super().__init__()
        self.scat_1d = Scattering1D(
            shape=(n_samples,), J=J, Q=(Q1, 1), T=1, max_order=1
        )
        self.wavelets = MorletWavelet(sr, w=None)

    def create_rand_wavelet(self, n_f: int, n_t: int) -> T:
        min_s_f = self.wavelets.min_scale_f
        max_s_f = self.wavelets.n_to_scale(n_f)
        min_s_t = self.wavelets.min_scale_t
        max_s_t = self.wavelets.n_to_scale(n_t)
        # self.wavelets.create_2d_wavelet_from_scale(max_s_f, max_s_t)
        # self.wavelets.create_2d_wavelet_from_scale(min_s_f, min_s_t)
        s_f = util.sample_uniform(min_s_f, max_s_f)
        s_t = util.sample_uniform(min_s_t, max_s_t)
        reflect = util.choice([True, False])
        wavelet = self.wavelets.create_2d_wavelet_from_scale(s_f, s_t, reflect)
        return wavelet

    def forward(self, x: T, x_target: T) -> T:
        # x_target: Tensor,
        # alpha: float = 0.1666,
        # beta: float = 0.0003,
        # reflect: bool = False) -> Tensor:
        assert x.ndim == x_target.ndim == 3
        assert x.size(1) == x_target.size(1) == 1
        u1 = self.scat_1d(x)
        u1 = u1[:, :, 1:, :].squeeze(1)
        u1_target = self.scat_1d(x_target)
        u1_target = u1_target[:, :, 1:, :].squeeze(1)

        # vmin = min(u1.min(), u1_target.min())
        # vmax = max(u1.max(), u1_target.max())
        # log.info(f"vmin = {vmin}, vmax = {vmax}")
        # plt.imshow(u1.detach().squeeze().numpy(), aspect="auto", interpolation=None, origin="upper", cmap="magma", vmin=vmin, vmax=vmax)
        # plt.show()
        # plt.imshow(u1_target.detach().squeeze().numpy(), aspect="auto", interpolation=None, origin="upper", cmap="magma", vmin=vmin, vmax=vmax)
        # plt.show()

        # max_beta_n = u1.size(1)
        # max_alpha_n = 8192
        # beta_n_s = list(range(5, max_beta_n, max_beta_n // 12))
        # betas = [self.wavelets.n_to_scale(n) for n in beta_n_s]
        # betas = list(filter(lambda x: x > self.wavelets.min_scale_f, betas))
        # alpha_n_s = list(range(5, max_alpha_n, max_alpha_n // 12))
        # alphas = [self.wavelets.n_to_scale(n) for n in alpha_n_s]
        # alphas = list(filter(lambda x: x > self.wavelets.min_scale_t, alphas))
        # orientations = [False, True]
        # results = []
        # for beta, alpha, reflect in tqdm(itertools.product(betas, alphas, orientations)):
        #     wavelet = self.wavelets.create_2d_wavelet_from_scale(beta, alpha, reflect)
        #     n_f, n_t = wavelet.shape
        #     Sx = dwt_2d(u1_target, [wavelet])
        #     val = Sx.mean().item() * 1e8
        #     if reflect:
        #         beta = -beta
        #         n_f = -n_f
        #     results.append((n_f, n_t, val))
        #     results.append((beta, alpha, val))
        # results = sorted(results, key=lambda x: x[2])
        # for beta, alpha, val in results:
        #     log.info(f"beta = {beta}, alpha = {alpha}, val = {val}")
        # exit()

        # wavelet = self.wavelets.create_2d_wavelet_from_scale(beta, alpha, reflect)
        wavelet = self.create_rand_wavelet(u1.size(1), 8192)  # TODO(cm): tmp
        Sx = dwt_2d(u1, [wavelet])
        Sx_target = dwt_2d(u1_target, [wavelet])
        # plt.imshow(Sx.detach().squeeze().numpy(), aspect="auto", interpolation=None, origin="upper", cmap="magma")
        # plt.show()
        # plt.imshow(Sx_target.detach().squeeze().numpy(), aspect="auto", interpolation=None, origin="upper", cmap="magma")
        # plt.show()

        # diff = Sx_target - Sx
        # dist = tr.linalg.vector_norm(diff, ord=2, dim=(2, 3))
        # dist = tr.mean(dist)

        Sx = tr.mean(Sx, dim=(2, 3))
        Sx_target = tr.mean(Sx_target, dim=(2, 3))
        diff = Sx_target - Sx
        dist = diff.mean()
        return dist


if __name__ == "__main__":
    # tr.manual_seed(0)
    n = 10000
    logits = tr.tensor([0.5, -0.5, 0.0])
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
        idx = tr.multinomial(F.softmax(logits, dim=0), 1).item()
        results[idx] += 1

    results /= n
    log.info(f"results multinomial = {results}")
