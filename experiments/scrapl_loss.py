import functools
import logging
import os
from collections import defaultdict
from typing import Union, Optional, Iterable, List

import torch as tr
import torch.nn as nn
from torch import Tensor as T
from tqdm import tqdm

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
        self.use_pwa = use_pwa
        self.use_saga = use_saga
        self.parameters = parameters
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
            f"number of SCRAPL keys = {self.n_paths}, "
            f"unif_prob = {self.unif_prob:.8f}"
        )
        self.curr_path_idx = None
        self.path_counts = defaultdict(int)
        self.rand_gen = tr.Generator(device="cpu")
        self.unsampled_paths = list(range(self.n_paths))

        # Grad related setup
        self.pwa_m = {}
        self.pwa_v = {}
        self.pwa_t = defaultdict(lambda: defaultdict(int))
        self.scrapl_t = 0
        self.prev_path_grads = {}
        self.hook_handles = []
        if parameters is not None:
            self.attach_params(parameters)

        # Sampling probs setup
        if probs_path is not None:
            probs = tr.load(probs_path).double()
            assert probs.shape == (
                self.n_paths,
            ), f"probs.shape = {probs.shape}, expected {(self.n_paths,)}"
            log.info(
                f"Loading probs from {probs_path}, min = {probs.min():.6f}, "
                f"max = {probs.max():.6f}, mean = {probs.mean():.6f}"
            )
            self.probs = probs
        else:
            self.probs = tr.full((self.n_paths,), self.unif_prob, dtype=tr.double)
        self.orig_probs = self.probs.clone()

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

        # Check probs
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
        self.parameters = None

    def clear(self) -> None:
        # Path related setup
        self.curr_path_idx = None
        self.path_counts.clear()
        self.unsampled_paths = list(range(self.n_paths))
        # Grad related setup
        self._clear_grad_data()
        # Sampling probs setup
        self.probs = self.orig_probs.clone()
        # Update probs setup
        self.updated_path_indices.clear()
        self.all_log_vals.fill_(self.log_eps)
        self.all_log_probs.fill_(self.log_unif_prob)
        log.info(f"Cleared state")
        self._check_probs()

    def attach_params(self, parameters: Iterable[T]) -> None:
        self._clear_grad_data()
        self.parameters = list(parameters)
        pwa_m = {}
        pwa_v = {}
        prev_path_grads = {}
        for idx, p in enumerate(self.parameters):
            pwa_m[idx] = tr.zeros((self.n_paths, *p.shape))
            pwa_v[idx] = tr.zeros((self.n_paths, *p.shape))
            prev_path_grads[idx] = tr.zeros((self.n_paths, *p.shape))
            handle = p.register_hook(functools.partial(self.grad_hook, param_idx=idx))
            self.hook_handles.append(handle)
        self.register_module("pwa_m", ReadOnlyTensorDict(pwa_m))
        self.register_module("pwa_v", ReadOnlyTensorDict(pwa_v))
        self.register_module("prev_path_grads", ReadOnlyTensorDict(prev_path_grads))
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
            self.rand_gen.manual_seed(seed)
            rand_gen = self.rand_gen
        if (
            self.sample_all_paths_first
            and len(self.path_counts) < self.n_paths
            and self.unsampled_paths
        ):
            unsampled_idx = tr.randint(
                0, len(self.unsampled_paths), (1,), generator=rand_gen
            ).item()
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

    def update_prob(self, path_idx: int, val: T, theta_idx: int = 0) -> None:
        assert 0 <= path_idx < self.n_paths
        assert 0 <= theta_idx < self.n_theta
        assert val >= 0.0
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


# warmup can be parallelized
# eval should show that the distributions differ to show it's adaptive
# can simulate this by bandlimiting o1 or o2 (already did o2 bands using chirplet synth)

if __name__ == "__main__":
    # Setup
    tr.set_printoptions(precision=4, sci_mode=False)
    tr.manual_seed(0)
    bs = 3
    n_samples = 7
    n_theta = 5

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
    x = tr.rand((bs, n_samples))

    params = [p for p in model.parameters()]
    assert all(not p.grad for p in params)

    theta_hat = model(x)
    x_hat = synth(theta_hat)
    loss = loss_fn(x_hat, x)

    # Calc param grad
    grad_dict = tr.autograd.grad(
        loss, params, create_graph=True, materialize_grads=True
    )
    grad_vec = tr.cat([g.view(-1) for g in grad_dict])
    log.info(f"grad_vec = {grad_vec[:5]}")
    assert all(not p.grad for p in params)
    vec = tr.rand_like(grad_vec)

    # Calc theta param grad and theta param hvp
    def theta_hook(grad: T, theta_idx: int) -> T:
        log.info(f"theta_hook called with theta_idx = {theta_idx}")
        zero_indices = list(range(n_theta))
        zero_indices.remove(theta_idx)
        grad[:, zero_indices] = 0.0
        return grad
        # new_grad = tr.zeros_like(grad)
        # new_grad[:, theta_idx] = grad[:, theta_idx]
        # return new_grad

    all_grad_vecs = []
    all_hvp_vecs = []
    for theta_idx in range(n_theta):
        curr_hook = functools.partial(theta_hook, theta_idx=theta_idx)
        handle = theta_hat.register_hook(curr_hook)
        grad_dict = tr.autograd.grad(
            loss, params, create_graph=True, materialize_grads=True, retain_graph=True
        )
        curr_grad_vec = tr.cat([g.view(-1) for g in grad_dict])
        all_grad_vecs.append(curr_grad_vec)
        handle.remove()

        hvp_dict = tr.autograd.grad(
            curr_grad_vec,
            params,
            grad_outputs=vec,
            materialize_grads=True,
            retain_graph=True,
        )
        curr_hvp_vec = tr.cat([g.view(-1) for g in hvp_dict])
        all_hvp_vecs.append(curr_hvp_vec)

    # Check that the combined theta param grad is the same
    combined_grad_vec = tr.stack(all_grad_vecs, dim=0).sum(dim=0)
    assert tr.allclose(grad_vec, combined_grad_vec)

    # Check that the combined theta param hvp is the same
    hvp_dict = tr.autograd.grad(grad_vec, params, grad_outputs=vec, retain_graph=True)
    hvp_vec = tr.cat([g.view(-1) for g in hvp_dict])
    log.info(f"hvp_vec_1 = {hvp_vec[-8:]}")

    combined_hvp_vec = tr.stack(all_hvp_vecs, dim=0).sum(dim=0)
    assert tr.allclose(hvp_vec, combined_hvp_vec)

    hvp_dict = tr.autograd.grad(
        combined_grad_vec, params, grad_outputs=vec, retain_graph=True
    )
    combined_hvp_vec_2 = tr.cat([g.view(-1) for g in hvp_dict])
    assert tr.allclose(hvp_vec, combined_hvp_vec_2)

    # Vectorized and no hook version ===================================================

    # Calc theta grad
    grad_theta = tr.autograd.grad(
        loss,
        theta_hat,
        create_graph=True,  # Create graphs is super important here
        materialize_grads=True
    )[0]

    # Expand theta grad along batch dime for each theta
    grad_theta_expanded = grad_theta.unsqueeze(0).expand(n_theta, -1, -1)
    # grad_theta_expanded = grad_theta.unsqueeze(0).repeat(n_theta, 1, 1)
    eye = tr.eye(n_theta, device=grad_theta_expanded.device).unsqueeze(1).expand(-1, bs, -1)
    # eye = tr.eye(n_theta, device=grad_theta_expanded.device).unsqueeze(1).repeat(1, bs, 1)
    grad_theta_expanded = grad_theta_expanded * eye

    # Calc vectorized param grad
    grad_params = tr.autograd.grad(
        theta_hat,
        params,
        grad_outputs=grad_theta_expanded,
        create_graph=True,
        materialize_grads=True,
        is_grads_batched=True,
    )
    grad_matrix = tr.cat([g.view(n_theta, -1) for g in grad_params], dim=1)
    for idx, g in enumerate(all_grad_vecs):
        assert tr.allclose(grad_matrix[idx], g)

    # Check that the combined vectorized param grad is the same
    grad_vec_2 = grad_matrix.sum(dim=0)
    assert tr.allclose(grad_vec, grad_vec_2)

    # Calc vectorized param hvp
    all_hvp_vecs_2 = []
    for idx in range(n_theta):
        curr_grad = grad_matrix[idx]
        hvp_dict = tr.autograd.grad(
            curr_grad,
            params,
            grad_outputs=vec,
            materialize_grads=True,
            retain_graph=True,
        )
        curr_hvp_vec = tr.cat([g.view(-1) for g in hvp_dict])
        all_hvp_vecs_2.append(curr_hvp_vec)
        # Check that the individual vectorized param hvp is the same
        other_hvp_vec = all_hvp_vecs[idx]
        assert tr.allclose(curr_hvp_vec, other_hvp_vec)

    # Check that the combined vectorized param hvp is the same
    hvp_vec_2 = tr.stack(all_hvp_vecs_2, dim=0).sum(dim=0)
    log.info(f"hvp_vec_2 = {hvp_vec_2[-8:]}")
    assert tr.allclose(hvp_vec, hvp_vec_2)
    exit()

    n_samples = 32768
    n_theta = 30
    n_iter = 100

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

    for _ in tqdm(range(n_iter)):
        val = tr.rand((1,)) * 1000.0
        path_idx = scrapl.sample_path()
        theta_idx = tr.randint(0, n_theta, (1,)).item()
        log.info(
            f"val = {val.item():.6f}, path_idx = {path_idx}, theta_idx = {theta_idx}"
        )
        scrapl.update_prob(path_idx, val, theta_idx)
