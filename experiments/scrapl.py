import functools
import logging
import os
from collections import defaultdict
from typing import Union, Any, Optional, Dict, Iterable

import torch as tr
import torch.nn as nn
from sympy.integrals.intpoly import gradient_terms
from torch import Tensor as T

from experiments import util
from experiments.util import ReadOnlyTensorDict
from scrapl.torch import TimeFrequencyScrapl

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


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
        grad_mult: float = 1e8,
        use_pwa: bool = True,
        use_saga: bool = True,
        parameters: Optional[Iterable[T]] = None,
        sample_all_paths_first: bool = False,
        probs_path: Optional[str] = None,
        eps: float = 1e-12,
        pwa_b1: float = 0.9,
        pwa_b2: float = 0.999,
        pwa_eps: float = 1e-8,
    ):
        super().__init__()
        self.p = p
        self.grad_mult = grad_mult
        self.use_pwa = use_pwa
        self.use_saga = use_saga
        self.parameters = parameters
        self.sample_all_paths_first = sample_all_paths_first
        self.eps = eps
        self.pwa_b1 = pwa_b1
        self.pwa_b2 = pwa_b2
        self.pwa_eps = pwa_eps

        # Path related setup
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
        # TODO(cm): add first order only keys too?
        self.scrapl_keys = [key for key in self.meta["key"] if len(key) == 2]
        self.n_paths = len(self.scrapl_keys)
        self.unif_prob = 1.0 / self.n_paths
        log.info(
            f"number of SCRAPL keys = {self.n_paths}, "
            f"unif_prob = {self.unif_prob:.8f}"
        )
        self.curr_path_idx = None
        self.path_counts = defaultdict(int)
        self.rand_gen = tr.Generator(device="cpu")
        self.unsampled_paths = list(range(self.n_paths))

        # Grad related setup
        self.pwa_m = defaultdict(lambda: {})
        self.pwa_v = defaultdict(lambda: {})
        self.pwa_t = defaultdict(lambda: {})
        self.scrapl_t = 0
        self.prev_path_grads = {}
        if parameters is not None:
            self.attach_params(parameters)

        # Sampling probs setup
        if probs_path is not None:
            probs = tr.load(probs_path).double()
            assert probs.shape == (self.n_paths,), \
                f"probs.shape = {probs.shape}, expected {(self.n_paths,)}"
            log.info(f"Loading probs from {probs_path}, min = {probs.min():.6f}, "
                     f"max = {probs.max():.6f}, mean = {probs.mean():.6f}")
            self.probs = probs
        else:
            self.probs = tr.full((self.n_paths,), self.unif_prob, dtype=tr.double)
        assert tr.allclose(
            self.probs.sum(), tr.tensor(1.0, dtype=tr.double), atol=self.eps
        ), f"self.probs.sum() = {self.probs.sum()}"

    def clear(self) -> None:
        # Path related setup
        self.curr_path_idx = None
        self.path_counts.clear()
        self.unsampled_paths = list(range(self.n_paths))
        # Grad related setup
        self.pwa_m = defaultdict(lambda: {})
        self.pwa_v = defaultdict(lambda: {})
        self.pwa_t = defaultdict(lambda: {})
        self.scrapl_t = 0
        for grad in self.prev_path_grads.values():
            grad.fill_(0.0)
        # TODO(cm): clear probs?

    def attach_params(self, parameters: Iterable[T]) -> None:
        prev_path_grads = {}
        for idx, p in enumerate(parameters):
            prev_path_grads[idx] = tr.zeros((self.n_paths, *p.shape))
            p.register_hook(functools.partial(self.grad_hook, param_idx=idx))
        self.prev_path_grads = ReadOnlyTensorDict(prev_path_grads)
        log.info(f"Attached {len(self.prev_path_grads)} parameters")
        self.clear()

    def grad_hook(self, grad: T, param_idx: int) -> T:
        # TODO(cm): check if this works
        if not self.training:
            return grad

        if self.grad_mult != 1.0:
            grad *= self.grad_mult

        assert self.curr_path_idx is not None
        path_idx = self.curr_path_idx
        curr_t = self.scrapl_t + 1

        if self.use_pwa:
            # TODO(cm): preallocate memory
            prev_m_s = self.pwa_m[param_idx]
            prev_v_s = self.pwa_v[param_idx]
            prev_t_s = self.pwa_t[param_idx]
            if path_idx in prev_m_s:
                prev_m = prev_m_s[path_idx]
            else:
                prev_m = tr.zeros_like(grad)
            if path_idx in prev_v_s:
                prev_v = prev_v_s[path_idx]
            else:
                prev_v = tr.zeros_like(grad)
            if path_idx in prev_t_s:
                prev_t = prev_t_s[path_idx]
            else:
                prev_t = 0
            prev_t_norm = prev_t / self.n_paths
            t_norm = curr_t / self.n_paths

            grad, m, v = self.adam_grad_norm_cont(
                grad,
                prev_m,
                prev_v,
                t_norm,
                prev_t_norm,
                b1=self.pwa_b1,
                b2=self.pwa_b2,
                eps=self.pwa_eps,
            )
            prev_m_s[path_idx] = m
            prev_v_s[path_idx] = v
            prev_t_s[path_idx] = curr_t

        if self.use_saga:
            n_paths_seen = len(self.path_counts)
            # Calculate previous average grad
            prev_path_grads = self.prev_path_grads[param_idx]
            prev_avg_grad = prev_path_grads.sum(dim=0) / max(1, n_paths_seen - 1)
            # Get prev path grad
            prev_path_grad = prev_path_grads[path_idx, ...]
            # Calculate SAGA grad
            saga_grad = grad - prev_path_grad + prev_avg_grad
            # Update current path grad
            prev_path_grads[path_idx, ...] = grad
            # Update grad
            grad = saga_grad

        return grad

    def sample_path(self, seed: Optional[int] = None) -> int:
        if seed is not None:
            self.rand_gen.manual_seed(seed)
        if self.sample_all_paths_first and len(self.path_counts) < self.n_paths:
            assert len(self.unsampled_paths) > 0
            unsampled_idx = tr.randint(
                0, len(self.unsampled_paths), (1,), generator=self.rand_gen).item()
            path_idx = self.unsampled_paths.pop(unsampled_idx)
        else:
            path_idx = tr.multinomial(
                self.probs, 1, generator=self.rand_gen).item()
        return path_idx

    def calc_dist(self, x: T, x_target: T, path_idx: int) -> (T, T, T):
        n2, n_fr = self.scrapl_keys[path_idx]
        Sx = self.jtfs.scattering_singlepath(x, n2, n_fr)
        Sx = Sx["coef"].squeeze(-1)
        Sx_target = self.jtfs.scattering_singlepath(x_target, n2, n_fr)
        Sx_target = Sx_target["coef"].squeeze(-1)
        diff = Sx_target - Sx
        dist = tr.linalg.vector_norm(diff, ord=self.p, dim=(-2, -1))
        dist = tr.mean(dist)
        return dist, Sx, Sx_target

    def forward(self, x: T, x_target: T, seed: Optional[int] = None, path_idx: Optional[int] = None) -> T:
        assert x.ndim == x_target.ndim == 3
        assert x.size(1) == x_target.size(1) == 1
        if path_idx is None:
            path_idx = self.sample_path(seed)
        else:
            log.info(f"Using specified path_idx = {path_idx}")
            assert 0 <= path_idx < self.n_paths
        dist, _, _ = self.calc_dist(x, x_target, path_idx)

        self.curr_path_idx = path_idx
        if self.training:  # TODO(cm): check if this works
            self.path_counts[path_idx] += 1
            self.scrapl_t += 1
        return dist

    @staticmethod
    def adam_grad_norm_cont(
        grad: T,
        prev_m: T,
        prev_v: T,
        t: float,
        prev_t: float,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
    ) -> (T, T, T):
        assert t > prev_t >= 0.0
        delta_t = t - prev_t
        eff_b1 = b1**delta_t
        eff_b2 = b2**delta_t
        m = eff_b1 * prev_m + (1 - eff_b1) * grad
        v = eff_b2 * prev_v + (1 - eff_b2) * grad**2
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        grad_hat = m_hat / (T.sqrt(v_hat) + eps)
        return grad_hat, m, v
