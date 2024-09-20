import functools
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional

import pytorch_lightning as pl
import torch as tr
from nnAudio.features import CQT
from torch import Tensor as T
from torch import nn

from experiments.losses import JTFSTLoss, SCRAPLLoss, AdaptiveSCRAPLLoss
from experiments.synth import ChirpTextureSynth
from experiments.util import ReadOnlyTensorDict

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class SCRAPLLightingModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        synth: nn.Module,
        loss_func: nn.Module,
        grad_multiplier: Optional[float] = 1e8,
        use_pathwise_adam: bool = True,
        vr_algo: Optional[str] = None,
        vr_beta: float = 0.99,
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
        self.grad_multiplier = grad_multiplier
        self.use_pathwise_adam = use_pathwise_adam
        if vr_algo is not None:
            assert vr_algo in ["sag", "saga", "none"]
            if vr_algo == "none":
                vr_algo = None
            assert isinstance(loss_func, AdaptiveSCRAPLLoss)
        self.vr_algo = vr_algo
        self.vr_beta = vr_beta
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

        if type(self.loss_func) in {SCRAPLLoss, AdaptiveSCRAPLLoss}:
            assert self.grad_multiplier is not None, "Grad multiplier is required"
        else:
            # TODO(cm): check JTFS
            assert self.grad_multiplier is None, "Grad multiplier is only for SCRAPL"
            assert not use_pathwise_adam, "Pathwise ADAM is only for SCRAPL"
            assert vr_algo is None, "VR is only for SCRAPL"

        if hasattr(self.loss_func, "set_resampler"):
            self.loss_func.set_resampler(self.synth.sr)

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
        self.val_maes = defaultdict(list)

        if use_pathwise_adam or vr_algo:
            log.info(
                f"Pathwise ADAM: {use_pathwise_adam}, "
                f"grad multiplier: {grad_multiplier:.0e}"
            )
            if vr_algo:
                log.info(f"Using {self.vr_algo.upper()} with decay {self.vr_beta}")
            n_paths = self.loss_func.n_paths
            self.paths_seen = set()
            prev_path_grads = {}
            for idx, p in enumerate(self.model.parameters()):
                prev_path_grads[idx] = tr.zeros((n_paths, *p.shape))
                p.register_hook(
                    functools.partial(
                        self.vr_hook, param_idx=idx, scrapl=self.loss_func
                    )
                )
                # break
            self.prev_path_grads = ReadOnlyTensorDict(prev_path_grads)
            self.sag_m = defaultdict(lambda: {})
            self.sag_v = defaultdict(lambda: {})
            self.sag_t = defaultdict(lambda: {})

            paths_beta = self.vr_beta
            # importance_after_n_path_steps = 0.05
            # paths_beta = importance_after_n_path_steps ** (1 / n_paths)
            # log.info(f"paths_beta: {paths_beta}")
            self.register_buffer("path_betas", tr.full((n_paths,), paths_beta))
        elif grad_multiplier is not None:
            log.info("Not using VR or pathwise ADAM, adding grad multiplier hook")
            for p in self.model.parameters():
                p.register_hook(self.grad_multiplier_hook)

        for p in self.synth.parameters():
            p.requires_grad = False

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

    def grad_multiplier_hook(self, grad: T) -> T:
        if not self.training:
            log.warning("grad_multiplier_hook called during eval")
            return grad
        grad *= self.grad_multiplier
        return grad

    def vr_hook(self, grad: T, param_idx: int, scrapl: AdaptiveSCRAPLLoss) -> T:
        if not self.training:
            log.warning("vr_hook called during eval")
            return grad

        path_idx = scrapl.curr_path_idx
        # path_idx = 250
        self.paths_seen.add(path_idx)
        n_paths = scrapl.n_paths
        curr_t = self.global_step + 1

        # if param_idx == 15:
        #     save_path = os.path.join(
        #         OUT_DIR, f"{self.run_name}_weight_{param_idx}_{curr_t}_{path_idx}.pt"
        #     )
        #     weight = list(self.parameters())[param_idx]
        #     assert weight.shape == grad.shape
        #     tr.save(weight.detach().cpu(), save_path)
        #
        #     save_path = os.path.join(
        #         OUT_DIR, f"{self.run_name}_grad_{param_idx}_{curr_t}_{path_idx}.pt"
        #     )
        #     tr.save(grad.detach().cpu(), save_path)

        if self.grad_multiplier is not None:
            grad *= self.grad_multiplier

        if self.use_pathwise_adam:
            # Adam grad continuous normalization
            prev_m_s = self.sag_m[param_idx]
            prev_v_s = self.sag_v[param_idx]
            prev_t_s = self.sag_t[param_idx]
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
            prev_t_norm = prev_t / n_paths
            t_norm = curr_t / n_paths

            grad, m, v = self.adam_grad_norm_cont(
                grad, prev_m, prev_v, t_norm, prev_t_norm
            )
            prev_m_s[path_idx] = m
            prev_v_s[path_idx] = v
            prev_t_s[path_idx] = curr_t

        # if param_idx == 15:
        #     save_path = os.path.join(
        #         OUT_DIR, f"{self.run_name}_grad_norm_{param_idx}_{curr_t}_{path_idx}.pt"
        #     )
        #     tr.save(grad.detach().cpu(), save_path)

        if self.vr_algo is None:
            return grad

        # VR algorithms
        # Get prev path grads
        prev_path_grads = self.prev_path_grads[param_idx]
        # Apply decay
        if self.vr_beta != 1.0:
            betas = self.path_betas.view(-1, *([1] * grad.ndim))
            prev_path_grads *= betas
        # Get number of paths seen
        n_paths_seen = len(self.paths_seen)

        if self.vr_algo == "sag":
            # Update current path grad
            prev_path_grads[path_idx, ...] = grad
            # Calculate SAG grad
            sag_grad = prev_path_grads.sum(dim=0) / n_paths_seen
            return sag_grad
        elif self.vr_algo == "saga":
            # Calculate previous average grad
            prev_avg_grad = prev_path_grads.sum(dim=0) / max(1, n_paths_seen - 1)
            # Get prev path grad
            prev_path_grad = prev_path_grads[path_idx, ...]
            if self.vr_beta != 1.0 and prev_t > 0:
                # Undo decay
                assert False  # TODO(cm): tmp, this can cause NaN
                delta_t = curr_t - prev_t
                prev_path_grad /= self.vr_beta**delta_t
            # Calculate SAGA grad
            # alpha = 0.9
            # saga_grad = alpha * (grad - prev_path_grad) + (1.0 - alpha) * prev_avg_grad
            saga_grad = grad - prev_path_grad + prev_avg_grad
            # Update current path grad
            prev_path_grads[path_idx, ...] = grad
            return saga_grad
        else:
            raise ValueError(f"Unknown VR algorithm: {self.vr_algo}")

    def on_train_start(self) -> None:
        self.global_n = 0

    def calc_U(self, x: T) -> T:
        if self.feature_type == "cqt":
            return SCRAPLLightingModule.calc_cqt(x, self.cqt, self.cqt_eps)
        else:
            raise NotImplementedError

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
            x = self.synth.make_x_from_theta(theta_density, theta_slope, seed)
            U = self.calc_U(x)

        U_hat = None
        x_hat = None

        theta_density_hat, theta_slope_hat = self.model(U)
        if stage == "train":
            theta_density_hat.retain_grad()
            theta_slope_hat.retain_grad()

        density_mae = self.l1(theta_density_hat, theta_density)
        slope_mae = self.l1(theta_slope_hat, theta_slope)
        if stage == "val":
            self.val_maes["l1_d"].append(density_mae.detach().cpu())
            self.val_maes["l1_s"].append(slope_mae.detach().cpu() / 2.0)  # Normalize

        if self.use_p_loss:
            density_loss = self.loss_func(theta_density_hat, theta_density)
            slope_loss = self.loss_func(theta_slope_hat, theta_slope)
            loss = density_loss + slope_loss
            self.log(
                f"{stage}/p_loss_{self.loss_name}", loss, prog_bar=True, sync_dist=True
            )
        else:
            x_hat = self.synth.make_x_from_theta(
                theta_density_hat, theta_slope_hat, seed_hat
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
                x = self.synth.make_x_from_theta(theta_density, theta_slope, seed)
            if x_hat is None and self.log_x_hat:
                x_hat = self.synth.make_x_from_theta(
                    theta_density_hat, theta_slope_hat, seed_hat
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

    def on_validation_epoch_end(self) -> None:
        for name, maes in self.val_maes.items():
            if len(maes) > 1:
                l1_var = tr.stack(maes, dim=0).var(dim=0)
                self.log(f"val/{name}_var", l1_var, prog_bar=False)
        self.val_maes.clear()

    @staticmethod
    def calc_cqt(x: T, cqt: CQT, cqt_eps: float = 1e-3) -> T:
        U = cqt(x)
        U = tr.log1p(U / cqt_eps)
        return U
