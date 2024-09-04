import functools
import logging
import os
from collections import defaultdict
from typing import Union, Any, Optional, List, Dict

import torch as tr
import torch.nn as nn
from kymatio.torch import TimeFrequencyScattering
from torch import Tensor as T
from torch.autograd import Function
from torch.nn import functional as F

from experiments import util
from jtfst_implementation.python.jtfst import JTFST2D
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
        fixed_path_idx: Optional[int] = None,
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
        self.fixed_path_idx = fixed_path_idx

    def sample_path(self) -> int:
        if self.fixed_path_idx is not None:
            path_idx = self.fixed_path_idx
        elif self.sample_all_paths_first and len(self.path_counts) < self.n_paths:
            path_idx = len(self.path_counts)
        else:
            path_idx = tr.randint(0, self.n_paths, (1,)).item()
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
        tau: float = 1.0,
        max_prob: float = 1.0,
        is_trainable: bool = True,
        probs_path: Optional[str] = None,
    ):
        super().__init__(shape, J, Q1, Q2, J_fr, Q_fr, T, F, p, sample_all_paths_first)
        self.tau = tau
        self.max_prob = max_prob
        self.is_trainable = is_trainable
        self.target_path_energies = defaultdict(list)
        self.path_mean_abs_Sx_grads = defaultdict(list)
        if is_trainable:
            self.logits = nn.Parameter(tr.zeros((self.n_paths,)))
        else:
            self.register_buffer("logits", tr.zeros((self.n_paths,)))
        if probs_path is not None:
            log.info(f"Loading probs from {probs_path}")
            probs = tr.load(probs_path)
            assert probs.shape == (self.n_paths,)
            self.register_buffer("probs", probs)
        else:
            self.probs = None
        self.curr_path_idx = None

    def save_mean_abs_Sx_grad(self, Sx_grad: T, path_idx: int) -> None:
        # grad_np = Sx_grad[0].squeeze().detach().cpu().numpy()
        # plt.imshow(grad_np, cmap="bwr", interpolation="none", aspect="auto", vmin=-0.4, vmax=0.4)
        # plt.imshow(grad_np, cmap="bwr", interpolation="none", aspect="auto")
        # plt.colorbar()
        # plt.show()
        # log.info(f"Sx_grad.min()    = {Sx_grad.min().item()}")
        # log.info(f"Sx_grad.max()    = {Sx_grad.max().item()}")
        # log.info(f"Sx_grad.mean()   = {Sx_grad.mean().item()}")
        mean_abs_Sx_grad = Sx_grad.abs().mean().detach().cpu().item()
        # log.info(f"mean_abs_Sx_grad = {mean_abs_Sx_grad}")
        self.path_mean_abs_Sx_grads[path_idx].append(mean_abs_Sx_grad)

    def sample_path(self) -> int:
        if self.sample_all_paths_first and len(self.path_counts) < self.n_paths:
            path_idx = len(self.path_counts)
            self.path_counts[path_idx] += 1
            self.curr_path_idx = path_idx
            return path_idx

        if self.probs is None:
            with tr.no_grad():
                probs = util.limited_softmax(self.logits, self.tau, self.max_prob)
        else:
            probs = self.probs
        path_idx = tr.multinomial(probs, 1).item()
        log.info(f"\npath_idx = {path_idx}")
        self.path_counts[path_idx] += 1
        self.curr_path_idx = path_idx
        return path_idx

    def forward(self, x: T, x_target: T) -> T:
        assert x.ndim == x_target.ndim == 3
        assert x.size(1) == x_target.size(1) == 1
        path_idx = self.sample_path()
        dist, Sx, Sx_target = self.calc_dist(x, x_target, path_idx)
        # Sx_np = Sx[0].squeeze().detach().cpu().numpy()
        # Sx_target_np = Sx_target[0].squeeze().detach().cpu().numpy()
        # plt.imshow(Sx_target_np - Sx_np, cmap="bwr", interpolation="none", aspect="auto")
        # plt.colorbar()
        # plt.title("Sx_target - Sx")
        # plt.show()
        Sx.register_hook(
            functools.partial(self.save_mean_abs_Sx_grad, path_idx=path_idx)
        )
        dist = MakeLogitsGradFromEnergy.apply(
            self.logits, path_idx, dist, Sx, Sx_target, self.target_path_energies
        )
        return dist


class MakeLogitsGradFromEnergy(Function):
    @staticmethod
    def forward(
        ctx: Any,
        logits: T,
        path_idx: int,
        dist: T,
        Sx: T,
        Sx_target: T,
        target_path_energies: Dict[int, List[float]],
    ) -> T:
        ctx.save_for_backward(logits, Sx, Sx_target)
        ctx.path_idx = path_idx
        ctx.target_path_energies = target_path_energies
        return dist

    @staticmethod
    def backward(ctx: Any, grad_dist: T) -> (T, None, T, None, None, None):
        logits, Sx, Sx_target = ctx.saved_tensors
        path_idx = ctx.path_idx
        target_path_energies = ctx.target_path_energies
        mean_target_energy = Sx_target.pow(2).mean()
        target_path_energies[path_idx].append(mean_target_energy.item())
        # TODO(cm)
        grad_mag = tr.clip(mean_target_energy, min=1e-20)
        grad_mag = tr.log10(grad_mag) + 20
        # log.info(f"grad_mag = {grad_mag}")

        grad_logits = tr.zeros_like(logits)
        grad_logits[path_idx] = -grad_mag
        return grad_logits, None, grad_dist, None, None, None


if __name__ == "__main__":
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
