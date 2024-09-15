import functools
import logging
import os
from collections import defaultdict
from datetime import datetime
from functools import partial
from typing import Dict, Optional, List

import pytorch_lightning as pl
import torch as tr
from nnAudio.features import CQT
from torch import Tensor as T
from torch import nn

from experiments.losses import JTFSTLoss, SCRAPLLoss, AdaptiveSCRAPLLoss
from experiments.synth import ChirpTextureSynth, make_x_from_theta
from experiments.util import ReadOnlyTensorDict

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class SCRAPLLightingModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        synth: ChirpTextureSynth,
        loss_func: nn.Module,
        use_sag: bool = False,
        sag_beta: float = 0.99,
        use_p_loss: bool = False,
        use_train_rand_seed: bool = False,
        use_val_rand_seed: bool = False,
        use_rand_seed_hat: bool = False,
        feature_type: str = "cqt",
        J_cqt: int = 5,
        cqt_eps: float = 1e-3,
        log_x: bool = False,
        log_x_hat: bool = False,
        log_val_grads: bool = False,
        run_name: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.synth = synth
        self.loss_func = loss_func
        self.use_sag = use_sag
        self.sag_beta = sag_beta
        self.use_p_loss = use_p_loss
        self.use_train_rand_seed = use_train_rand_seed
        self.use_val_rand_seed = use_val_rand_seed
        self.use_rand_seed_hat = use_rand_seed_hat
        self.J_cqt = J_cqt
        self.feature_type = feature_type
        self.cqt_eps = cqt_eps
        self.log_x = log_x
        self.log_x_hat = log_x_hat
        self.log_val_grads = log_val_grads
        if run_name is None:
            self.run_name = f"run__{datetime.now().strftime('%Y-%m-%d__%H-%M-%S')}"
        else:
            self.run_name = run_name
        log.info(f"Run name: {self.run_name}")

        cqt_params = {
            "sr": synth.sr,
            "bins_per_octave": synth.Q,
            "n_bins": J_cqt * synth.Q,
            "hop_length": synth.hop_len,
            # TODO(cm): check this
            "fmin": (0.4 * synth.sr) / (2**J_cqt),
            "output_format": "Magnitude",
            "verbose": False,
        }
        self.cqt = CQT(**cqt_params)
        self.loss_name = self.loss_func.__class__.__name__
        self.l1 = nn.L1Loss()
        self.global_n = 0

        self.d_grads = defaultdict(list)
        self.s_grads = defaultdict(list)

        if self.use_sag:
            log.info(f"Using SAG with decay {self.sag_beta}")
            n_paths = self.loss_func.n_paths
            self.paths_seen = set()
            prev_path_grads = {}
            for idx, p in enumerate(self.parameters()):
                prev_path_grads[idx] = tr.zeros((n_paths, *p.shape))
                p.register_hook(
                    functools.partial(
                        self.sag_hook, param_idx=idx, scrapl=self.loss_func
                    )
                )
                # break
            self.prev_path_grads = ReadOnlyTensorDict(prev_path_grads)
            self.sag_m = defaultdict(lambda: {})
            self.sag_v = defaultdict(lambda: {})
            self.sag_t = defaultdict(lambda: {})

            paths_beta = self.sag_beta
            # importance_after_n_path_steps = 0.05
            # paths_beta = importance_after_n_path_steps ** (1 / n_paths)
            # log.info(f"paths_beta: {paths_beta}")
            self.register_buffer("path_betas", tr.full((n_paths,), paths_beta))

    @staticmethod
    def adam_grad_norm(
        grad: T,
        prev_m: T,
        prev_v: T,
        t: int,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
    ) -> (T, T, T):
        assert t > 0
        m = b1 * prev_m + (1 - b1) * grad
        v = b2 * prev_v + (1 - b2) * grad**2
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        grad_hat = m_hat / (tr.sqrt(v_hat) + eps)
        return grad_hat, m, v

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
        # grad *= 1e8
        assert t > prev_t >= 0.0
        delta_t = t - prev_t
        eff_b1 = b1**delta_t
        eff_b2 = b2**delta_t
        m = eff_b1 * prev_m + (1 - eff_b1) * grad
        v = eff_b2 * prev_v + (1 - eff_b2) * grad**2
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        grad_hat = m_hat / (T.sqrt(v_hat) + eps)
        # log.info(f"grad_hat.abs().mean() {grad_hat.abs().mean()}")
        # log.info(f"grad_hat.abs().std() {grad_hat.abs().std()}")
        return grad_hat, m, v

    def sag_hook(self, grad: T, param_idx: int, scrapl: AdaptiveSCRAPLLoss) -> T:
        if not self.training:
            log.warning("sag_hook called during eval")
            return grad

        path_idx = scrapl.curr_path_idx
        # path_idx = 250
        self.paths_seen.add(path_idx)
        n_paths = scrapl.n_paths
        curr_t = self.global_step + 1

        # Adam grad normalization
        prev_m_s = self.sag_m[param_idx]
        prev_v_s = self.sag_v[param_idx]
        prev_t_norms = self.sag_t[param_idx]
        if path_idx in prev_m_s:
            prev_m = prev_m_s[path_idx]
        else:
            prev_m = tr.zeros_like(grad)
        if path_idx in prev_v_s:
            prev_v = prev_v_s[path_idx]
        else:
            prev_v = tr.zeros_like(grad)
        if path_idx in prev_t_norms:
            prev_t_norm = prev_t_norms[path_idx]
        else:
            prev_t_norm = 0.0
        t_norm = curr_t / n_paths

        grad, m, v = self.adam_grad_norm_cont(grad, prev_m, prev_v, t_norm, prev_t_norm)
        prev_m_s[path_idx] = m
        prev_v_s[path_idx] = v
        prev_t_norms[path_idx] = t_norm

        # SAG algorithm
        # Get prev path grads
        prev_path_grads = self.prev_path_grads[param_idx]
        # Apply decay
        betas = self.path_betas.view(-1, *([1] * grad.ndim))
        prev_path_grads *= betas
        # Update current path grad
        prev_path_grads[path_idx, ...] = grad
        n_paths_seen = len(self.paths_seen)
        avg_grad = prev_path_grads.sum(dim=0) / n_paths_seen
        return avg_grad

    def on_train_start(self) -> None:
        self.global_n = 0

    def calc_U(self, x: T) -> T:
        if self.feature_type == "cqt":
            return SCRAPLLightingModule.calc_cqt(x, self.cqt, self.cqt_eps)
        else:
            raise NotImplementedError

    # def sag_hook(
    #     self,
    #     grad: T,
    #     batch_indices: T,
    #     path_idx: int,
    #     path_grads: T,
    #     decay_val: float = 1.0,
    # ) -> T:
    #     assert path_idx is not None
    #     if decay_val != 1.0:
    #         path_grads.mul_(decay_val)
    #     path_grads[batch_indices, path_idx, ...] = grad
    #     grad = path_grads[batch_indices, ...]
    #     grad = tr.mean(grad, dim=1)
    #     return grad

    def step(self, batch: (T, T, T), stage: str) -> Dict[str, T]:
        theta_density, theta_slope, seed, batch_indices = batch
        batch_size = theta_density.size(0)
        if stage == "train":
            self.global_n = (
                self.global_step * self.trainer.accumulate_grad_batches * batch_size
            )
        # TODO(cm): check if this works for DDP
        self.log(f"global_n", float(self.global_n), sync_dist=True)

        # TODO(cm): make this cleaner
        seed_range = 9999999
        if stage == "train" and self.use_train_rand_seed:
            seed = tr.randint_like(seed, low=0, high=seed_range)
        elif stage == "val" and self.use_val_rand_seed:
            seed = tr.randint_like(seed, low=0, high=seed_range)
        seed_hat = seed
        if self.use_rand_seed_hat:
            seed_hat = tr.randint_like(seed, low=seed_range, high=2 * seed_range)

        with tr.no_grad():
            x = make_x_from_theta(self.synth, theta_density, theta_slope, seed)
            U = self.calc_U(x)

        U_hat = None
        x_hat = None

        theta_density_hat, theta_slope_hat = self.model(U)
        if stage == "train":
            theta_density_hat.retain_grad()
            theta_slope_hat.retain_grad()

        density_mae = self.l1(theta_density_hat, theta_density)
        slope_mae = self.l1(theta_slope_hat, theta_slope)

        if self.use_p_loss:
            density_loss = self.loss_func(theta_density_hat, theta_density)
            slope_loss = self.loss_func(theta_slope_hat, theta_slope)
            loss = density_loss + slope_loss
            self.log(
                f"{stage}/p_loss_{self.loss_name}", loss, prog_bar=True, sync_dist=True
            )
        else:
            x_hat = make_x_from_theta(
                self.synth, theta_density_hat, theta_slope_hat, seed_hat
            )
            with tr.no_grad():
                U_hat = self.calc_U(x_hat)
            loss = self.loss_func(x_hat, x)
            self.log(f"{stage}/{self.loss_name}", loss, prog_bar=True, sync_dist=True)

        self.log(f"{stage}/l1_d", density_mae, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/l1_s", slope_mae, prog_bar=True, sync_dist=True)
        # Slope has twice the range of density so we normalize it first before averaging
        theta_mae = (density_mae + (slope_mae / 2)) / 2
        self.log(f"{stage}/l1_theta", theta_mae, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/loss", loss, prog_bar=False, sync_dist=True)

        with tr.no_grad():
            if x is None and self.log_x:
                x = make_x_from_theta(self.synth, theta_density, theta_slope, seed)
            if x_hat is None and self.log_x_hat:
                x_hat = make_x_from_theta(
                    self.synth, theta_density_hat, theta_slope_hat, seed_hat
                )
                U_hat = self.calc_U(x_hat)

        # top_n = 8
        # with suppress(Exception):
        # #     logits = self.loss_func.logits
        # #     probs = util.limited_softmax(
        # #         logits, tau=self.loss_func.tau, max_prob=self.loss_func.max_prob
        # #     )
        # #     top = tr.topk(logits, k=top_n, dim=-1)
        # #     logits = [f"{p:.6f}" for p in top.values]
        # #     log.info(f"Top {top_n} logits: {top.indices} {logits}")
        #     probs = self.loss_func.probs
        #     top = tr.topk(probs, k=top_n, dim=-1)
        #     percentages = [f"{p:.6f}" for p in top.values]
        #     log.info(f"Top {top_n} percentages: {top.indices} {percentages}")
        # with suppress(Exception):
        #     path_counts = self.loss_func.path_counts
        #     path_counts = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)
        #     path_counts = list(path_counts)[:top_n]
        #     # log.info(f"sorted path_counts: {path_counts}")

        out_dict = {
            "loss": loss,
            "U": U,
            "U_hat": U_hat,
            "x": x,
            "x_hat": x_hat,
            "theta_density": theta_density,
            "theta_density_hat": theta_density_hat,
            "theta_slope": theta_slope,
            "theta_slope_hat": theta_slope_hat,
            "seed": seed,
            "seed_hat": seed_hat,
        }

        if stage == "train":
            theta_density_hat.register_hook(
                partial(
                    self.save_grad,
                    path_idx=self.loss_func.curr_path_idx,
                    idx_to_grads=self.d_grads,
                )
            )
            theta_slope_hat.register_hook(
                partial(
                    self.save_grad,
                    path_idx=self.loss_func.curr_path_idx,
                    idx_to_grads=self.s_grads,
                )
            )
        return out_dict

    def training_step(self, batch: (T, T, T), batch_idx: int) -> Dict[str, T]:
        if not (
            isinstance(self.loss_func, JTFSTLoss)
            or isinstance(self.loss_func, SCRAPLLoss)
        ):
            assert self.trainer.accumulate_grad_batches == 1
        return self.step(batch, stage="train")

    def validation_step(self, batch: (T, T, T), stage: str) -> Dict[str, T]:
        if self.log_val_grads:
            tr.set_grad_enabled(True)
        return self.step(batch, stage="val")

    def test_step(self, batch: (T, T, T), stage: str) -> Dict[str, T]:
        return self.step(batch, stage="test")

    @staticmethod
    def calc_cqt(x: T, cqt: CQT, cqt_eps: float = 1e-3) -> T:
        U = cqt(x)
        U = tr.log1p(U / cqt_eps)
        return U

    @staticmethod
    def save_grad(
        grad: T, path_idx: int, idx_to_grads: defaultdict[int, List[float]]
    ) -> T:
        assert path_idx is not None
        mean_grad = grad.abs().mean().detach().cpu().item()
        idx_to_grads[path_idx].append(mean_grad)
        return grad
