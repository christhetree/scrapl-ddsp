import functools
import logging
import os
from collections import defaultdict
from typing import Union, Optional, Iterable, List, Dict, Any, Callable, Literal

import hessian_eigenthings
import torch as tr
import torch.nn as nn
from hessian_eigenthings.operator import LambdaOperator
from torch import Tensor as T
from torch.nn import Parameter

from experiments.util import ReadOnlyTensorDict
from scrapl.torch import TimeFrequencyScrapl

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class SCRAPLLoss(nn.Module):
    STATE_DICT_EXCLUDED_PREFIXES = [
        "jtfs.tensor",
    ]
    STATE_DICT_INCLUDED_ATTRS = [
        # Path related setup
        "path_counts",
        "unsampled_paths",
        # Sampling probs setup
        "probs",
        "loaded_probs_path",
        # Update probs setup
        "updated_path_indices",
        "all_log_vals",
        "all_log_probs",
    ]

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
        sample_all_paths_first: bool = False,
        n_theta: int = 1,
        min_prob_frac: float = 0.0,
        probs_path: Optional[str] = None,
        eps: float = 1e-12,
        pwa_b1: float = 0.9,
        pwa_b2: float = 0.999,
        pwa_eps: float = 1e-8,
    ):
        super().__init__()
        self.p = p
        self.grad_mult = grad_mult
        if use_pwa:
            assert (
                grad_mult > 1.0
            ), "Using PWA requires a grad multiplier to avoid float precision errors"
        self.use_pwa = use_pwa
        self.use_saga = use_saga
        self.sample_all_paths_first = sample_all_paths_first
        assert n_theta >= 1
        self.n_theta = n_theta
        assert 0 <= min_prob_frac < 1.0
        self.min_prob_frac = min_prob_frac
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
            f"SCRAPLLoss:\n"
            f"grad_mult              = {grad_mult:.0e}\n"
            f"use_pwa                = {use_pwa}\n"
            f"use_saga               = {use_saga}\n"
            f"sample_all_paths_first = {sample_all_paths_first}\n"
            f"n_theta                = {n_theta}\n"
            f"min_prob_frac          = {min_prob_frac}\n"
            f"number of SCRAPL keys  = {self.n_paths}\n"
            f"unif_prob              = {self.unif_prob:.8f}\n"
        )
        self.curr_path_idx = None
        self.path_counts = defaultdict(int)
        self.rand_gen = tr.Generator(device="cpu")
        self.unsampled_paths = list(range(self.n_paths))

        # Grad related setup
        self.attached_params = None
        self.pwa_m = {}
        self.pwa_v = {}
        self.pwa_t = defaultdict(lambda: defaultdict(int))
        self.scrapl_t = 0
        self.prev_path_grads = {}
        self.hook_handles = []

        # Sampling probs setup
        # We keep probs on the CPU to avoid GPU and CPU seed discrepancies
        self.loaded_probs_path = None
        if probs_path is None:
            self.probs = tr.full((self.n_paths,), self.unif_prob, dtype=tr.double)
        else:
            self.load_probs(probs_path)

        # Update probs setup
        self.log_eps = tr.tensor(self.eps, dtype=tr.double).log()
        self.log_unif_prob = tr.tensor(self.unif_prob, dtype=tr.double).log()
        self.min_prob = self.unif_prob * self.min_prob_frac
        self.log_1m_min_prob_frac = tr.tensor(
            1.0 - self.min_prob_frac, dtype=tr.double
        ).log()
        self.updated_path_indices = defaultdict(set)
        self.all_log_vals = tr.full(
            (self.n_theta, self.n_paths), self.log_eps, dtype=tr.double
        )
        self.all_log_probs = tr.full(
            (
                self.n_theta,
                self.n_paths,
            ),
            self.log_unif_prob,
            dtype=tr.double,
        )

        # Warmup setup
        theta_eye = tr.eye(self.n_theta)
        self.register_buffer("theta_eye", theta_eye, persistent=False)

        # Check probs
        self._check_probs()

    def load_probs(self, probs_path: str) -> None:
        assert os.path.isfile(probs_path)
        probs = tr.load(probs_path).double()
        assert probs.shape == (
            self.n_paths,
        ), f"probs.shape = {probs.shape}, expected {(self.n_paths,)}"
        log.info(
            f"Loading probs from:\n\t{probs_path}\n\tmin = {probs.min():.6f}, "
            f"max = {probs.max():.6f}, mean = {probs.mean():.6f}"
        )
        self.probs = probs
        self.loaded_probs_path = probs_path
        self._check_probs()

    def _clear_grad_data(self) -> None:
        self.pwa_m = {}
        self.pwa_v = {}
        self.pwa_t = defaultdict(lambda: defaultdict(int))
        self.scrapl_t = 0
        self.prev_path_grads = {}
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.attached_params = None

    def clear(self) -> None:
        # Path related setup
        self.curr_path_idx = None
        self.path_counts.clear()
        self.unsampled_paths = list(range(self.n_paths))
        # Grad related setup
        self._clear_grad_data()
        # Update probs setup
        self.updated_path_indices.clear()
        self.all_log_vals.fill_(self.log_eps)
        self.all_log_probs.fill_(self.log_unif_prob)
        log.info(f"Cleared state")
        self._check_probs()

    def state_dict(self, *args, **kwargs) -> Dict[str, T]:
        # TODO(cm): support resuming training with grad hooks
        state_dict = super().state_dict(*args, **kwargs)
        global_prefix = kwargs.get("prefix", "")
        excluded_keys = []
        for prefix in self.STATE_DICT_EXCLUDED_PREFIXES:
            for k in state_dict:
                if k.startswith(f"{global_prefix}{prefix}"):
                    excluded_keys.append(k)
        for k in excluded_keys:
            del state_dict[k]
        for attr in self.STATE_DICT_INCLUDED_ATTRS:
            assert attr not in state_dict
            state_dict[f"{global_prefix}{attr}"] = getattr(self, attr)
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, T], *args, **kwargs) -> None:
        global_prefix = kwargs.get("prefix", "")
        for attr in self.STATE_DICT_INCLUDED_ATTRS:
            assert f"{global_prefix}{attr}" in state_dict
            setattr(self, attr, state_dict[f"{global_prefix}{attr}"])
        kwargs["strict"] = False  # TODO(cm): is there a better way to do this?
        super().load_state_dict(state_dict, *args, **kwargs)

    def attach_params(self, params: Iterable[T]) -> None:
        self._clear_grad_data()
        self.attached_params = list(params)
        pwa_m = {}
        pwa_v = {}
        prev_path_grads = {}
        for idx, p in enumerate(self.attached_params):
            pwa_m[idx] = tr.zeros((self.n_paths, *p.shape))
            pwa_v[idx] = tr.zeros((self.n_paths, *p.shape))
            prev_path_grads[idx] = tr.zeros((self.n_paths, *p.shape))
            handle = p.register_hook(functools.partial(self.grad_hook, param_idx=idx))
            self.hook_handles.append(handle)
        del self.pwa_m
        del self.pwa_v
        del self.prev_path_grads
        self.register_module("pwa_m", ReadOnlyTensorDict(pwa_m, persistent=False))
        self.register_module("pwa_v", ReadOnlyTensorDict(pwa_v, persistent=False))
        self.register_module(
            "prev_path_grads", ReadOnlyTensorDict(prev_path_grads, persistent=False)
        )
        log.info(f"Attached {len(self.prev_path_grads)} parameters")

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
            prev_m_s = self.pwa_m[param_idx]
            prev_v_s = self.pwa_v[param_idx]
            prev_t_s = self.pwa_t[param_idx]
            prev_m = prev_m_s[path_idx]
            prev_v = prev_v_s[path_idx]
            prev_t = prev_t_s[path_idx]
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
        if seed is None:
            rand_gen = None
        else:
            rand_gen = self.rand_gen
            rand_gen.manual_seed(seed)
        if (
            self.sample_all_paths_first
            and len(self.path_counts) < self.n_paths
            and self.unsampled_paths
        ):
            unsampled_idx = tr.randint(
                0, len(self.unsampled_paths), (1,), generator=rand_gen
            ).item()
            # TODO(cm): will this pop during validation and test?
            path_idx = self.unsampled_paths.pop(unsampled_idx)
        else:
            self.check_probs(self.probs, self.n_paths, self.eps, self.min_prob)
            path_idx = tr.multinomial(
                self.probs, num_samples=1, generator=rand_gen
            ).item()
        return path_idx

    def calc_dist(self, x: T, x_target: T, path_idx: int) -> (T, T, T):
        n2, n_fr = self.scrapl_keys[path_idx]
        Sx = self.jtfs.scattering_singlepath(x, n2, n_fr)
        Sx = Sx["coef"].squeeze(-1)
        Sx_target = self.jtfs.scattering_singlepath(x_target, n2, n_fr)
        Sx_target = Sx_target["coef"].squeeze(-1)
        diff = Sx_target - Sx
        dist = tr.linalg.vector_norm(diff, ord=self.p, dim=(-2, -1))
        # TODO(cm): add reduction option
        dist = tr.mean(dist)
        return dist, Sx, Sx_target

    def forward(
        self,
        x: T,
        x_target: T,
        seed: Optional[int] = None,
        path_idx: Optional[int] = None,
    ) -> T:
        assert x.ndim == x_target.ndim == 3
        bs, n_ch, n_samples = x.size()
        x = x.view(-1, 1, n_samples)
        x_target = x_target.view(-1, 1, n_samples)
        assert x.size(1) == x_target.size(1) == 1
        if path_idx is None:
            path_idx = self.sample_path(seed)
        else:
            log.debug(f"Using specified path_idx = {path_idx}")
            assert 0 <= path_idx < self.n_paths
        dist, _, _ = self.calc_dist(x, x_target, path_idx)

        self.curr_path_idx = path_idx
        if self.training:  # TODO(cm): check if this works
            self.path_counts[path_idx] += 1
            self.scrapl_t += 1
        return dist

    # Update prob methods ==============================================================
    def update_prob(self, path_idx: int, val: float | T, theta_idx: int = 0) -> None:
        assert self.loaded_probs_path is None, (
            f"Cannot update probs if external probs have been loaded "
            f"({self.loaded_probs_path})"
        )
        assert 0 <= path_idx < self.n_paths
        assert 0 <= theta_idx < self.n_theta
        assert val >= 0.0
        if isinstance(val, float):
            val = tr.tensor(val, dtype=tr.double)
        val = val.cpu()
        # Keep track of which and how many paths have been updated
        self.updated_path_indices[theta_idx].add(path_idx)
        updated_indices = list(self.updated_path_indices[theta_idx])
        n_updated = len(updated_indices)
        # Update log vals
        log_vals = self.all_log_vals[theta_idx]
        log_val = val.double().clamp(min=self.eps).log()
        log_vals[path_idx] = log_val
        # Update log probs
        if n_updated < 2:
            # Use init probs since there are not enough updated paths
            return
        elif n_updated == self.n_paths:
            # Use all updated log vals to calculate log probs
            log_probs = self.calc_log_probs(log_vals)
        else:
            # Use a balance between updated log vals and init probs
            log_probs = self.all_log_probs[theta_idx]
            log_probs[updated_indices] = self.log_eps
            updated_frac = tr.tensor(n_updated / self.n_paths, dtype=tr.double)
            stale_frac = 1.0 - updated_frac
            updated_probs = self.calc_log_probs(log_vals) + updated_frac.log()
            stale_probs = self.calc_log_probs(log_probs) + stale_frac.log()
            self.all_log_probs[theta_idx] = stale_probs
            self.all_log_probs[theta_idx][updated_indices] = updated_probs[
                updated_indices
            ]
            log_probs = self.calc_log_probs(self.all_log_probs[theta_idx])

        # Ensure min_prob_frac is enforced
        log_probs += self.log_1m_min_prob_frac
        # Update all log probs
        self.all_log_probs[theta_idx] = log_probs
        # Update probs
        self._recompute_probs()

    def _recompute_probs(self) -> None:
        all_probs = self.all_log_probs.exp() + self.min_prob
        probs = all_probs.mean(dim=0)
        self.probs = probs
        self._check_probs()

    def _check_probs(self) -> None:
        self.check_probs(self.probs, self.n_paths, self.eps, self.min_prob)
        all_probs = self.all_log_probs.exp()
        all_vals = self.all_log_vals.exp()
        for theta_idx in range(self.n_theta):
            probs = all_probs[theta_idx, :]
            vals = all_vals[theta_idx, :]
            updated_indices = list(self.updated_path_indices[theta_idx])
            self.check_probs(
                probs, self.n_paths, self.eps, self.min_prob, vals, updated_indices
            )

    # Warmup methods ===================================================================
    def _calc_batch_theta_param_grad(
        self,
        path_idx: int,
        theta_fn: Callable[..., T],
        synth_fn: Callable[[T, ...], T],
        theta_fn_kwargs: Dict[str, Any],
        params: List[Parameter],
        theta_idx: Optional[int] = None,
        seed: Optional[int] = None,
        synth_fn_kwargs: Optional[Dict[str, Any]] = None,
    ) -> T:
        if synth_fn_kwargs is None:
            synth_fn_kwargs = {}
        assert "x" in theta_fn_kwargs, "x must be provided in theta_fn_kwargs"
        assert params, "params must not be empty"
        assert all(p.grad is None for p in params), "params must have no grad"
        if theta_idx is not None:
            assert 0 <= theta_idx < self.n_theta
        theta_hat = theta_fn(**theta_fn_kwargs)
        assert theta_hat.ndim == 2
        assert theta_hat.size(1) == self.n_theta
        bs = theta_hat.size(0)
        x_hat = synth_fn(theta_hat, **synth_fn_kwargs)
        x = theta_fn_kwargs["x"]
        loss = self.forward(x, x_hat, seed=seed, path_idx=path_idx)

        theta_grad = tr.autograd.grad(
            loss, theta_hat, create_graph=True, materialize_grads=True
        )[0]
        if theta_idx is None:
            # Expand theta_grad to vectorize individual theta grad calculations
            theta_grad_ex = theta_grad.unsqueeze(0).expand(self.n_theta, -1, -1)
            theta_eye_ex = self.theta_eye.unsqueeze(1).expand(-1, bs, -1)
            theta_grad_ex = theta_grad_ex * theta_eye_ex
            is_grads_batched = True
        else:
            # Zero out all but the specified theta grad
            mask = tr.zeros_like(theta_grad)
            mask[:, theta_idx] = 1.0
            theta_grad_ex = theta_grad * mask
            is_grads_batched = False
        # Calculate maybe vectorized param grad
        theta_param_grads = tr.autograd.grad(
            theta_hat,
            params,
            grad_outputs=theta_grad_ex,
            create_graph=True,
            materialize_grads=True,
            is_grads_batched=is_grads_batched,
        )
        if theta_idx is None:
            theta_param_grad = tr.cat(
                [g.view(self.n_theta, -1) for g in theta_param_grads], dim=1
            )
        else:
            theta_param_grad = tr.cat([g.view(-1) for g in theta_param_grads])
        return theta_param_grad

    def _calc_param_hvp(
        self,
        tangent: T,
        param_grad: T,
        params: List[Parameter],
        retain_graph: bool = False,
    ) -> T:
        assert tangent.ndim == 1
        assert param_grad.shape == tangent.shape
        assert all(p.grad is None for p in params), "params must have no grad"
        param_hvps = tr.autograd.grad(
            param_grad,
            params,
            grad_outputs=tangent,
            materialize_grads=True,
            retain_graph=retain_graph,
        )
        param_hvp = tr.cat([g.contiguous().view(-1) for g in param_hvps])
        return param_hvp

    def _calc_largest_eig(
        self,
        param_grad: T,
        params: List[Parameter],
        n_iter: int = 20,
    ) -> T:
        apply_fn = functools.partial(
            self._calc_param_hvp,
            param_grad=param_grad,
            params=params,
            retain_graph=True,
        )
        size = param_grad.size(0)
        hvp_op = LambdaOperator(apply_fn, size)
        use_gpu = param_grad.is_cuda
        eigs, _ = hessian_eigenthings.deflated_power_iteration(
            operator=hvp_op,
            num_eigenthings=1,
            power_iter_steps=n_iter,
            to_numpy=False,
            use_gpu=use_gpu,
        )
        eig1 = tr.from_numpy(eigs.copy()).float()
        return eig1

    def calc_theta_eigs(
        self,
        path_idx: int,
        theta_fn: Callable[..., T],
        synth_fn: Callable[[T, ...], T],
        theta_fn_kwargs: Dict[str, Any],
        params: List[Parameter],
        seed: Optional[int] = None,
        synth_fn_kwargs: Optional[Dict[str, Any]] = None,
        n_iter: int = 20,
    ) -> T:
        theta_param_grad = self._calc_batch_theta_param_grad(
            path_idx,
            theta_fn,
            synth_fn,
            theta_fn_kwargs,
            params,
            seed=seed,
            synth_fn_kwargs=synth_fn_kwargs,
        )
        theta_eigs = []
        for theta_idx in range(self.n_theta):
            param_grad = theta_param_grad[theta_idx, :]
            eig1 = self._calc_largest_eig(param_grad, params, n_iter=n_iter)
            theta_eigs.append(eig1)
        theta_eigs = tr.cat(theta_eigs, dim=0)
        return theta_eigs

    def _calc_param_hvp_multibatch(
        self,
        tangent: T,
        path_idx: int,
        theta_idx: int,
        theta_fn: Callable[..., T],
        synth_fn: Callable[[T, ...], T],
        theta_fn_kwargs: List[Dict[str, Any]],
        params: List[Parameter],
        synth_fn_kwargs: Optional[List[Dict[str, Any]]] = None,
    ) -> T:
        if synth_fn_kwargs is None:
            assert theta_fn_kwargs, "theta_fn_kwargs must not be empty"
            synth_fn_kwargs = [None] * len(theta_fn_kwargs)
        else:
            assert len(synth_fn_kwargs) == len(theta_fn_kwargs), (
                f"len(theta_fn_kwargs) ({len(theta_fn_kwargs)}) != "
                f"len(synth_fn_kwargs) ({len(synth_fn_kwargs)})"
            )
        param_hvp = None
        for curr_theta_fn_kwargs, curr_synth_fn_kwargs in zip(
            theta_fn_kwargs, synth_fn_kwargs
        ):
            curr_param_grad = self._calc_batch_theta_param_grad(
                path_idx,
                theta_fn,
                synth_fn,
                curr_theta_fn_kwargs,
                params,
                theta_idx=theta_idx,
                synth_fn_kwargs=curr_synth_fn_kwargs,
            )
            curr_param_hvp = self._calc_param_hvp(
                tangent, curr_param_grad, params, retain_graph=False
            )
            # TODO(cm): should we average here?
            if param_hvp is None:
                param_hvp = curr_param_hvp
            else:
                param_hvp += curr_param_hvp
        return param_hvp

    def _calc_theta_largest_eig_multibatch(
        self,
        path_idx: int,
        theta_idx: int,
        theta_fn: Callable[..., T],
        synth_fn: Callable[[T, ...], T],
        theta_fn_kwargs: List[Dict[str, Any]],
        params: List[Parameter],
        synth_fn_kwargs: Optional[List[Dict[str, Any]]] = None,
        n_iter: int = 20,
    ) -> T:
        apply_fn = functools.partial(
            self._calc_param_hvp_multibatch,
            path_idx=path_idx,
            theta_idx=theta_idx,
            theta_fn=theta_fn,
            synth_fn=synth_fn,
            theta_fn_kwargs=theta_fn_kwargs,
            params=params,
            synth_fn_kwargs=synth_fn_kwargs,
        )
        size = sum(p.numel() for p in params)
        hvp_op = LambdaOperator(apply_fn, size)
        use_gpu = params[0].is_cuda
        eigs, _ = hessian_eigenthings.deflated_power_iteration(
            operator=hvp_op,
            num_eigenthings=1,
            power_iter_steps=n_iter,
            to_numpy=False,
            use_gpu=use_gpu,
        )
        eig1 = tr.from_numpy(eigs.copy()).float()
        return eig1

    def calc_theta_eigs_multibatch(
        self,
        path_idx: int,
        theta_fn: Callable[..., T],
        synth_fn: Callable[[T, ...], T],
        theta_fn_kwargs: List[Dict[str, Any]],
        params: List[Parameter],
        synth_fn_kwargs: Optional[List[Dict[str, Any]]] = None,
        n_iter: int = 20,
    ) -> T:
        theta_eigs = []
        for theta_idx in range(self.n_theta):
            eig1 = self._calc_theta_largest_eig_multibatch(
                path_idx,
                theta_idx,
                theta_fn,
                synth_fn,
                theta_fn_kwargs,
                params,
                synth_fn_kwargs=synth_fn_kwargs,
                n_iter=n_iter,
            )
            theta_eigs.append(eig1)
        theta_eigs = tr.cat(theta_eigs, dim=0)
        return theta_eigs

    def warmup_lc_hvp(
        self,
        theta_fn: Callable[..., T],
        synth_fn: Callable[[T, ...], T],
        theta_fn_kwargs: List[Dict[str, Any]],
        params: List[Parameter],
        synth_fn_kwargs: Optional[List[Dict[str, Any]]] = None,
        n_iter: int = 20,
        agg: Literal["all", "mean", "max", "med"] = "all",
        force_multibatch: bool = False,
    ) -> None:
        assert params, "params must not be empty"
        assert (
            self.attached_params is None
        ), "Parameters cannot be attached during warmup!"
        assert theta_fn_kwargs, "theta_fn_kwargs must not be empty"
        if synth_fn_kwargs is not None:
            assert len(synth_fn_kwargs) == len(theta_fn_kwargs), (
                f"len(theta_fn_kwargs) ({len(theta_fn_kwargs)}) != "
                f"len(synth_fn_kwargs) ({len(synth_fn_kwargs)})"
            )
        # Check determinism
        theta_fn_kwargs_batch = theta_fn_kwargs[0]
        synth_fn_kwargs_batch = {}
        if synth_fn_kwargs is not None:
            synth_fn_kwargs_batch = synth_fn_kwargs[0]
        if not self.check_is_deterministic(theta_fn, theta_fn_kwargs_batch):
            log.warning("theta_fn is not deterministic")
        if not self.check_is_deterministic(synth_fn, synth_fn_kwargs_batch):
            log.warning("synth_fn is not deterministic")

        # Determine whether to use multibatch or not
        is_multibatch = force_multibatch or len(theta_fn_kwargs) > 1
        log.info(
            f"Starting warmup_lc_hvp with agg = {agg} for {len(params)} "
            f"parameter(s) and {len(theta_fn_kwargs)} batch(es) "
            f"(multibatch = {is_multibatch})"
        )
        if is_multibatch:
            calc_theta_eigs_fn = self.calc_theta_eigs_multibatch
        else:
            calc_theta_eigs_fn = self.calc_theta_eigs
            theta_fn_kwargs = theta_fn_kwargs[0]
            if synth_fn_kwargs is not None:
                synth_fn_kwargs = synth_fn_kwargs[0]
        # Separate the params for separate computations if aggregating LCs across them
        if agg == "all":
            param_groups = [params]
        else:
            param_groups = [[p] for p in params]
        # Compute the theta LCs for each path
        for path_idx in range(self.n_paths):
            vals = []
            for param_group in param_groups:
                curr_vals = calc_theta_eigs_fn(
                    path_idx=path_idx,
                    theta_fn=theta_fn,
                    synth_fn=synth_fn,
                    theta_fn_kwargs=theta_fn_kwargs,
                    params=param_group,
                    synth_fn_kwargs=synth_fn_kwargs,
                    n_iter=n_iter,
                )
                vals.append(curr_vals)
                log.info(f"path_idx = {path_idx}, curr_vals = {curr_vals}")
            # Aggregate the theta LCs across all params
            vals = tr.stack(vals, dim=0)
            if agg == "all":
                vals = vals[0]
            elif agg == "mean":
                vals = vals.mean(dim=0)
            elif agg == "max":
                vals = vals.max(dim=0).values
            elif agg == "med":
                vals = vals.median(dim=0).values
            else:
                raise ValueError(f"Invalid agg = {agg}")
            # Update the probs for each theta
            for theta_idx in range(self.n_theta):
                val = vals[theta_idx]
                self.update_prob(path_idx, val, theta_idx)

    # Static methods ===================================================================
    @staticmethod
    def check_probs(
        probs: T,
        n_paths: int,
        eps: float,
        min_prob: float = 0.0,
        vals: Optional[T] = None,
        updated_indices: Optional[List[int]] = None,
    ) -> None:
        assert probs.shape == (n_paths,)
        assert tr.allclose(
            probs.sum(), tr.tensor(1.0, dtype=tr.double), atol=eps
        ), f"self.probs.sum() = {probs.sum()}"
        assert probs.min() >= min_prob - eps, f"probs.min() = {probs.min()}"

        if vals is None or len(updated_indices) < 2:
            return

        assert vals.shape == (n_paths,)
        assert vals.min() >= eps
        vals = vals[updated_indices]
        val_ratios = vals / vals.min()
        raw_probs = probs[updated_indices] - min_prob
        assert raw_probs.min() > 0.0
        prob_ratios = raw_probs / raw_probs.min()
        assert tr.allclose(
            val_ratios, prob_ratios, atol=1e-3, rtol=1e-3, equal_nan=True
        ), f"val_ratios = {val_ratios}, prob_ratios = {prob_ratios}"

    @staticmethod
    def calc_log_probs(log_vals: T) -> T:
        assert log_vals.ndim == 1
        log_probs = log_vals - tr.logsumexp(log_vals, dim=0)
        return log_probs

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
        delta_t = t - prev_t
        eff_b1 = b1**delta_t
        eff_b2 = b2**delta_t
        m = eff_b1 * prev_m + (1 - eff_b1) * grad
        v = eff_b2 * prev_v + (1 - eff_b2) * grad**2
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        grad_hat = m_hat / (T.sqrt(v_hat) + eps)
        return grad_hat, m, v

    @staticmethod
    def check_is_deterministic(
        fn: Callable[..., T],
        fn_kwargs: Dict[str, Any],
    ) -> bool:
        x_1 = fn(**fn_kwargs)
        x_2 = fn(**fn_kwargs)
        if not tr.allclose(x_1, x_2):
            return False
        return True


if __name__ == "__main__":
    # Setup
    tr.set_printoptions(precision=4, sci_mode=False)
    tr.manual_seed(0)
    bs = 2
    n_samples = 32768
    n_theta = 3

    model = nn.Sequential(
        nn.Linear(n_samples, n_theta),
        nn.PReLU(),
        nn.Linear(n_theta, n_theta),
        nn.Sigmoid(),
    )
    model = model
    synth = nn.Sequential(
        nn.Linear(n_theta, n_theta),
        nn.PReLU(),
        nn.Linear(n_theta, n_samples),
        nn.Tanh(),
    )
    synth = synth
    loss_fn = nn.MSELoss()
    x_1 = tr.rand((bs, n_samples))
    x_2 = tr.rand((bs, n_samples))
    x_3 = tr.rand((bs, n_samples))

    params = [p for p in model.parameters()]
    assert all(not p.grad for p in params)

    scrapl = SCRAPLLoss(
        shape=n_samples,
        J=12,
        Q1=8,
        Q2=2,
        J_fr=3,
        Q_fr=2,
        n_theta=n_theta,
        sample_all_paths_first=False,
    )
    # TODO: Check whether probs are loaded in ckpt or need to be in buffer
    # scrapl.attach_params(params)
    theta_fn = lambda x: model(x.squeeze(1))
    synth_fn = lambda theta: synth(theta).unsqueeze(1)

    theta_fn_kwargs = [
        {"x": x_1.unsqueeze(1)},
        # {"x": x_1.unsqueeze(1)},
        # {"x": x_2.unsqueeze(1)},
        # {"x": x_3.unsqueeze(1)},
    ]
    # scrapl.warmup_lc_hess_multibatch(
    #     theta_fn=theta_fn,
    #     synth_fn=synth_fn,
    #     theta_fn_kwargs=theta_fn_kwargs,
    #     params=params,
    #     n_iter=2,
    # )
    # theta_fn_kwargs = theta_fn_kwargs[0]
    # scrapl.warmup_lc_hess(
    #     theta_fn=theta_fn,
    #     synth_fn=synth_fn,
    #     theta_fn_kwargs=theta_fn_kwargs,
    #     params=params,
    #     n_iter=2,
    # )
    scrapl.warmup_lc_hvp(
        theta_fn=theta_fn,
        synth_fn=synth_fn,
        theta_fn_kwargs=theta_fn_kwargs,
        params=params,
        n_iter=2,
        agg="all",
        # agg="mean",
        force_multibatch=False,
        # force_multibatch=True,
    )

    # INFO:__main__:theta_eigs = tensor([0.0039, 0.0011, 0.0081])
    # INFO:__main__:theta_eigs = tensor([0.0079, 0.0023, 0.0161])

    # save_path = os.path.join(OUT_DIR, "scrapl.pt")
    # tr.save(scrapl.state_dict(), save_path)
    # state_dict = tr.load(save_path)
    # scrapl.load_state_dict(state_dict)
