import functools
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional, List, Any

import hessian_eigenthings
import numpy as np
import pytorch_lightning as pl
import torch as tr
from nnAudio.features import CQT
from torch import Tensor as T
from torch import nn
from tqdm import tqdm

from experiments.hessian import HVPAutograd
from experiments.losses import JTFSTLoss, SCRAPLLoss, AdaptiveSCRAPLLoss, Scat1DLoss
from experiments.paths import OUT_DIR
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
        vr_beta: float = 1.0,
        use_p_loss: bool = False,
        use_train_rand_seed: bool = False,
        use_val_rand_seed: bool = False,
        use_rand_seed_hat: bool = False,
        feature_type: str = "cqt",
        cqt_eps: float = 1e-3,
        log_x: bool = False,
        log_x_hat: bool = False,
        log_val_grads: bool = False,
        update_paths: bool = False,
        run_name: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.synth = synth
        self.loss_func = loss_func
        self.grad_multiplier = grad_multiplier
        # if use_pathwise_adam:
        #     assert self.trainer.accumulate_grad_batches == 1, \
        #         "Pathwise ADAM does not support gradient accumulation"
        self.use_pathwise_adam = use_pathwise_adam
        if vr_algo is not None:
            assert vr_algo in ["sag", "saga", "none"]
            if vr_algo == "none":
                vr_algo = None
            assert isinstance(loss_func, AdaptiveSCRAPLLoss)
        self.vr_algo = vr_algo
        self.vr_beta = vr_beta
        self.use_p_loss = use_p_loss
        if use_train_rand_seed:
            log.info("Using a random seed for training data samples")
        self.use_train_rand_seed = use_train_rand_seed
        if use_val_rand_seed:
            log.info("Using a random seed for validation data samples")
        self.use_val_rand_seed = use_val_rand_seed
        if use_rand_seed_hat:
            log.info("============== MESOSCALE ============== ")
        else:
            log.info("============== MICROSCALE ============== ")
        self.use_rand_seed_hat = use_rand_seed_hat
        self.feature_type = feature_type
        self.cqt_eps = cqt_eps
        self.log_x = log_x
        self.log_x_hat = log_x_hat
        self.log_val_grads = log_val_grads
        self.update_paths = update_paths
        if update_paths:
            log.info("Updating path probabilities")
            assert vr_algo or use_pathwise_adam, \
                "VR or PWA is required for path updates"
        if run_name is None:
            self.run_name = f"run__{datetime.now().strftime('%Y-%m-%d__%H-%M-%S')}"
        else:
            self.run_name = run_name
        log.info(f"Run name: {self.run_name}")

        if type(self.loss_func) in {
            SCRAPLLoss,
            AdaptiveSCRAPLLoss,
            JTFSTLoss,
            Scat1DLoss,
        }:
            assert self.grad_multiplier is not None, "Grad multiplier is required"
        else:
            assert self.grad_multiplier is None, "Grad multiplier is only for JTFS"
        if type(self.loss_func) not in {SCRAPLLoss, AdaptiveSCRAPLLoss}:
            assert not use_pathwise_adam, "Pathwise ADAM is only for SCRAPL"
            assert vr_algo is None, "VR is only for SCRAPL"

        if hasattr(self.loss_func, "set_resampler"):
            self.loss_func.set_resampler(self.synth.sr)
        if hasattr(self.loss_func, "in_sr"):
            assert self.loss_func.in_sr == self.synth.sr

        cqt_params = {
            "sr": synth.sr,
            "bins_per_octave": synth.Q,
            "n_bins": synth.J_cqt * synth.Q,
            "hop_length": synth.hop_len,
            # TODO(cm): check this
            "fmin": (0.4 * synth.sr) / (2**synth.J_cqt),
            "output_format": "Magnitude",
            "verbose": False,
        }
        self.cqt = CQT(**cqt_params)
        self.loss_name = self.loss_func.__class__.__name__
        self.l1 = nn.L1Loss()
        self.global_n = 0
        self.val_l1_s = defaultdict(list)
        self.param_idx_to_lc = {}
        self.strict_loading = False

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

        self.model_param_grads = None
        self.first_batch = None

        # TSV logging
        tsv_cols = [
            "seed",
            "stage",
            "step",
            "global_n",
            "time_epoch",
            "loss",
            "l1_theta",
            "l1_d",
            "l1_s",
        ]
        if run_name:
            self.tsv_path = os.path.join(OUT_DIR, f"{self.run_name}.tsv")
            if not os.path.exists(self.tsv_path):
                with open(self.tsv_path, "w") as f:
                    f.write("\t".join(tsv_cols) + "\n")
        else:
            self.tsv_path = None

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

    def functional_loss_from_theta_hat(
        self,
        theta_hat: (T, T),
        synth_p: (Dict[str, T], Dict[str, T]),
        loss_p: (Dict[str, T], Dict[str, T]),
        x: T,
        seed_hat: T,
        curr_path_idx: Optional[int] = None,
    ) -> (T, T):
        theta_d_0to1_hat, theta_s_0to1_hat = theta_hat
        x_hat = tr.func.functional_call(
            self.synth,
            synth_p,
            args=(theta_d_0to1_hat, theta_s_0to1_hat, seed_hat),
            strict=True,
        )
        if curr_path_idx is None:
            loss_args = (x_hat, x)
        else:
            loss_args = (x_hat, x, curr_path_idx)
        loss = tr.func.functional_call(
            self.loss_func, loss_p, args=loss_args, strict=True
        )
        return loss, x_hat

    def specific_theta_hat_functional_loss(
        self,
        specific_theta_hat: T,
        theta_name: str,
        other_theta_hat: T,
        synth_p: (Dict[str, T], Dict[str, T]),
        loss_p: (Dict[str, T], Dict[str, T]),
        x: T,
        seed_hat: T,
        curr_path_idx: Optional[int] = None,
    ) -> T:
        if theta_name == "d":
            theta_hat = (specific_theta_hat, other_theta_hat)
        else:
            theta_hat = (other_theta_hat, specific_theta_hat)
        loss, _ = self.functional_loss_from_theta_hat(
            theta_hat, synth_p, loss_p, x, seed_hat, curr_path_idx
        )
        return loss

    def functional_loss(
        self,
        model_p: (Dict[str, T], Dict[str, T]),
        synth_p: (Dict[str, T], Dict[str, T]),
        loss_p: (Dict[str, T], Dict[str, T]),
        x: T,
        U: T,
        seed_hat: T,
        curr_path_idx: Optional[int] = None,
    ) -> (T, (T, T, T)):
        theta_d_0to1_hat, theta_s_0to1_hat = tr.func.functional_call(
            self.model, model_p, U, strict=True
        )
        loss, x_hat = self.functional_loss_from_theta_hat(
            (theta_d_0to1_hat, theta_s_0to1_hat),
            synth_p,
            loss_p,
            x,
            seed_hat,
            curr_path_idx,
        )
        return loss, (theta_d_0to1_hat, theta_s_0to1_hat, x_hat)

    def specific_model_p_functional_loss(
        self,
        p: T,
        p_name: str,
        model_p: (Dict[str, T], Dict[str, T]),
        synth_p: (Dict[str, T], Dict[str, T]),
        loss_p: (Dict[str, T], Dict[str, T]),
        x: T,
        U: T,
        seed_hat: T,
        curr_path_idx: Optional[int] = None,
    ) -> T:
        model_params, _ = model_p
        orig_param = model_params[p_name]
        p = p.view(orig_param.shape)
        # assert tr.allclose(p, orig_param)
        model_params[p_name] = p
        loss, _ = self.functional_loss(
            model_p, synth_p, loss_p, x, U, seed_hat, curr_path_idx
        )
        return loss

    def calc_model_param_grads(
        self,
        x: T,
        U: T,
        seed_hat: T,
    ) -> (Dict[str, T], T, T, T, T):
        model_params = {k: v.detach() for k, v in self.model.named_parameters()}
        model_buffers = {k: v.detach() for k, v in self.model.named_buffers()}
        synth_params = {k: v.detach() for k, v in self.synth.named_parameters()}
        synth_buffers = {k: v.detach() for k, v in self.synth.named_buffers()}
        loss_params = {k: v.detach() for k, v in self.loss_func.named_parameters()}
        loss_buffers = {k: v.detach() for k, v in self.loss_func.named_buffers()}
        model_p = (model_params, model_buffers)
        synth_p = (synth_params, synth_buffers)
        loss_p = (loss_params, loss_buffers)
        try:
            curr_path_idx = self.loss_func.curr_path_idx
        except Exception:
            curr_path_idx = None

        out = tr.func.grad_and_value(self.functional_loss, has_aux=True)(
            model_p,
            synth_p,
            loss_p,
            x,
            U,
            seed_hat,
            curr_path_idx,
        )
        (model_param_grads, _), (loss, aux) = out
        theta_d_0to1_hat, theta_s_0to1_hat, x_hat = aux
        return model_param_grads, loss, theta_d_0to1_hat, theta_s_0to1_hat, x_hat

    def grad_multiplier_hook(self, grad: T) -> T:
        # log.info(f"grad.abs().max() = {grad.abs().max()}")
        if not self.training:
            log.warning("grad_multiplier_hook called during eval")
            return grad
        grad *= self.grad_multiplier
        return grad

    def save_grad_hook(self, grad: T, name: str, curr_t: Optional[int] = None) -> T:
        if not self.training:
            log.warning("save_grad_hook called during eval")
            return grad

        if curr_t is None:
            curr_t = self.global_step + 1
        try:
            path_idx = self.loss_func.curr_path_idx
        except AttributeError:
            log.warning("save_grad_hook: path_idx not found")
            path_idx = -1
        save_dir = os.path.join(OUT_DIR, f"{self.run_name}")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir, f"{self.run_name}__{name}_{curr_t}_{path_idx}.pt"
        )
        tr.save(grad.detach().cpu(), save_path)
        return grad

    def vr_hook(self, grad: T, param_idx: int, scrapl: AdaptiveSCRAPLLoss) -> T:
        if not self.training:
            log.warning("vr_hook called during eval")
            return grad

        if self.update_paths:
            # For adaptive SCRAPL, update path probabilities
            grad_norm = grad.detach().cpu().view(-1).norm(p=2)
            self.param_idx_to_lc[param_idx] = grad_norm

        path_idx = scrapl.curr_path_idx
        assert path_idx is not None
        # path_idx = 250
        self.paths_seen.add(path_idx)
        n_paths = scrapl.n_paths
        curr_t = self.global_step + 1

        # curr_t = 1
        # self.save_grad_hook(grad, f"g_raw_{param_idx}", curr_t)
        # return grad

        if self.grad_multiplier is not None:
            grad *= self.grad_multiplier

        # grad_norm = grad.detach().cpu().view(-1).norm(p=2)
        # self.param_idx_to_lc[param_idx] = grad_norm

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

            # self.save_grad_hook(grad, f"g_adam_{param_idx}", curr_t)

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

            # self.save_grad_hook(saga_grad, f"g_saga_{param_idx}", curr_t)
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
        if self.update_paths:
            # For adaptive SCRAPL, update path probabilities
            if stage == "train" and isinstance(self.loss_func, AdaptiveSCRAPLLoss):
                if self.loss_func.curr_path_idx is not None:
                    prev_path_idx = self.loss_func.curr_path_idx
                    if self.param_idx_to_lc:
                        # assert len(self.param_idx_to_lc) == 28, f"len = {len(self.param_idx_to_lc)}"
                        lc_s = [lc for lc in self.param_idx_to_lc.values()]
                        lc = tr.stack(lc_s, dim=0).mean().item()
                        # log.info(f"Path {prev_path_idx} LC: {lc}")
                        self.loss_func.update_prob(prev_path_idx, lc)
            self.param_idx_to_lc.clear()

        theta_d_0to1, theta_s_0to1, seed, batch_indices = batch

        # # Grad collection setup ======================================================
        # if stage != "train":
        #     log.info(f"Skipping validation/test step")
        #     return {}
        # log.info(f"loss_func.curr_path_idx: {self.loss_func.curr_path_idx}")
        # if self.loss_func.curr_path_idx == self.loss_func.n_paths - 1:
        #     exit()
        # if self.first_batch is None:
        #     self.first_batch = batch
        # else:
        #     theta_d_0to1, theta_s_0to1, seed, batch_indices = self.first_batch
        # log.info(
        #     f"theta_d_0to1.mean(): {theta_d_0to1.mean()}, "
        #     f"theta_s_0to1.mean(): {theta_s_0to1.mean()}, "
        #     f"seed.max(): {seed.max()}"
        # )
        # p1 = list(self.parameters())[0]
        # assert p1.requires_grad
        # log.info(f"p1.max() = {p1.max()}")
        # # Grad collection setup ======================================================

        batch_size = theta_d_0to1.size(0)
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
            x = self.synth(theta_d_0to1, theta_s_0to1, seed)
            U = self.calc_U(x)

        U_hat = None
        x_hat = None

        theta_d_0to1_hat, theta_s_0to1_hat = self.model(U)
        if stage == "train":
            theta_d_0to1_hat.retain_grad()
            theta_s_0to1_hat.retain_grad()
            # theta_d_0to1_hat.register_hook(
            #     functools.partial(self.save_grad_hook, name="theta_d_0to1_hat")
            # )
            # theta_s_0to1_hat.register_hook(
            #     functools.partial(self.save_grad_hook, name="theta_s_0to1_hat")
            # )

        l1_d = self.l1(theta_d_0to1_hat, theta_d_0to1)
        l1_s = self.l1(theta_s_0to1_hat, theta_s_0to1)
        if stage == "val":
            self.val_l1_s["l1_d"].append(l1_d.detach().cpu())
            self.val_l1_s["l1_s"].append(l1_s.detach().cpu())

        if self.use_p_loss:
            loss_d = self.loss_func(theta_d_0to1_hat, theta_d_0to1)
            loss_s = self.loss_func(theta_s_0to1_hat, theta_s_0to1)
            loss = loss_d + loss_s
            self.log(
                f"{stage}/p_loss_{self.loss_name}", loss, prog_bar=True, sync_dist=True
            )
        else:
            x_hat = self.synth(theta_d_0to1_hat, theta_s_0to1_hat, seed_hat)
            with tr.no_grad():
                U_hat = self.calc_U(x_hat)
            loss = self.loss_func(x_hat, x)
            self.log(f"{stage}/{self.loss_name}", loss, prog_bar=True, sync_dist=True)

        self.log(f"{stage}/l1_d", l1_d, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/l1_s", l1_s, prog_bar=True, sync_dist=True)
        # Slope has twice the range of density so we normalize it first before averaging
        theta_mae = (l1_d + l1_s) / 2
        self.log(f"{stage}/l1_theta", theta_mae, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/loss", loss, prog_bar=False, sync_dist=True)

        with tr.no_grad():
            if x is None and self.log_x:
                x = self.synth(theta_d_0to1, theta_s_0to1, seed)
            if x_hat is None and self.log_x_hat:
                x_hat = self.synth(theta_d_0to1_hat, theta_s_0to1_hat, seed_hat)
                U_hat = self.calc_U(x_hat)

        # # HVP calculation ============================================================
        # model_params = {k: v.detach() for k, v in self.model.named_parameters()}
        # model_buffers = {k: v.detach() for k, v in self.model.named_buffers()}
        # synth_params = {k: v.detach() for k, v in self.synth.named_parameters()}
        # synth_buffers = {k: v.detach() for k, v in self.synth.named_buffers()}
        # loss_params = {k: v.detach() for k, v in self.loss_func.named_parameters()}
        # loss_buffers = {k: v.detach() for k, v in self.loss_func.named_buffers()}
        # model_p = (model_params, model_buffers)
        # synth_p = (synth_params, synth_buffers)
        # loss_p = (loss_params, loss_buffers)
        # try:
        #     curr_path_idx = self.loss_func.curr_path_idx
        # except Exception:
        #     curr_path_idx = None
        #
        # name_to_idx = {name: idx for idx, name in enumerate(model_params)}
        # max_numel = np.inf
        # hess_param_names = [
        #     name for name, p in model_params.items() if p.numel() <= max_numel
        # ]
        # # hess_param_names = ["cnn.1.weight"]
        # # hess_param_names = ["cnn.3.weight"]
        # log.info(f"max_numel = {max_numel}, hess_param_names = {hess_param_names}")
        # for p_name in tqdm(hess_param_names):
        #     loss_fn = functools.partial(
        #         self.specific_model_p_functional_loss,
        #         p_name=p_name,
        #         model_p=model_p,
        #         synth_p=synth_p,
        #         loss_p=loss_p,
        #         x=x,
        #         U=U,
        #         seed_hat=seed_hat,
        #         curr_path_idx=curr_path_idx,
        #     )
        #     param = model_params[p_name]
        #     log.info(f"param_name = {p_name}, param.numel() = {param.numel()}")
        #     primal = param.view(-1)
        #     hvp_op = HVPAutograd(loss_fn, primal)
        #
        #     # tangent = tr.randn_like(primal)
        #     # log.info(f"starting hvp")
        #     # hvp = hvp_op.apply(tangent)
        #     # assert not hvp.isnan().any()
        #
        #     n_iter = 20
        #     n_eig = 1
        #     # try:
        #     #     eigenvals, _ = hessian_eigenthings.lanczos(
        #     #         operator=hvp_op,
        #     #         num_eigenthings=1,
        #     #         which="LM",
        #     #         max_steps=n_iter,
        #     #         use_gpu=True,
        #     #     )
        #     #     log.info(f"Eigenvalues lanczos: {eigenvals}")
        #     # except Exception as e:
        #     #     log.info(f"Lanczos failed: {e}")
        #     use_gpu = tr.cuda.is_available()
        #     eigenvals_power, _ = hessian_eigenthings.deflated_power_iteration(
        #         operator=hvp_op,
        #         num_eigenthings=n_eig,
        #         power_iter_steps=n_iter,
        #         to_numpy=False,
        #         use_gpu=use_gpu,
        #     )
        #     eigenvals_power = tr.from_numpy(eigenvals_power.copy()).float()
        #     log.info(f"Eigenvalues power:   {eigenvals_power}")
        #     assert not eigenvals_power.isnan().any()
        #
        #     param_idx = name_to_idx[p_name]
        #     curr_t = 1
        #     save_dir = os.path.join(OUT_DIR, f"{self.run_name}")
        #     os.makedirs(save_dir, exist_ok=True)
        #     save_path = os.path.join(
        #         save_dir,
        #         f"{self.run_name}__eig{n_eig}_{param_idx}_{curr_t}_{curr_path_idx}.pt",
        #     )
        #     tr.save(eigenvals_power.detach().cpu(), save_path)
        #
        #     # exit()
        # # HVP calculation ============================================================

        # TSV logging
        if self.tsv_path:
            # TODO(cm): check if there is a better way to do this
            seed_everything = tr.initial_seed() % (2**32)
            time_epoch = time.time()
            with open(self.tsv_path, "a") as f:
                f.write(
                    f"{seed_everything}\t{stage}\t{self.global_step}\t"
                    f"{self.global_n}\t{time_epoch}\t{loss.item()}\t"
                    f"{theta_mae.item()}\t{l1_d.item()}\t{l1_s.item()}\n"
                )

        out_dict = {
            "loss": loss,
            "U": U,
            "U_hat": U_hat,
            "x": x,
            "x_hat": x_hat,
            "theta_d": theta_d_0to1,
            "theta_d_hat": theta_d_0to1_hat,
            "theta_s": theta_s_0to1,
            "theta_s_hat": theta_s_0to1_hat,
            "seed": seed,
            "seed_hat": seed_hat,
        }
        return out_dict

    def training_step(self, batch: (T, T, T), batch_idx: int) -> Dict[str, T]:
        return self.step(batch, stage="train")

    def validation_step(self, batch: (T, T, T), stage: str) -> Dict[str, T]:
        if self.log_val_grads:
            tr.set_grad_enabled(True)
        return self.step(batch, stage="val")

    def test_step(self, batch: (T, T, T), stage: str) -> Dict[str, T]:
        return self.step(batch, stage="test")

    def on_validation_epoch_end(self) -> None:
        l1_tv_all = []
        for name, maes in self.val_l1_s.items():
            if len(maes) > 1:
                l1_tv = self.calc_total_variation(maes, norm_by_len=True)
                self.log(f"val/{name}_tv", l1_tv, prog_bar=False)
                l1_tv_all.append(l1_tv)
        if l1_tv_all:
            l1_theta_tv = tr.stack(l1_tv_all, dim=0).mean(dim=0)
            self.log(f"val/l1_theta_tv", l1_theta_tv, prog_bar=False)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        state_dict = checkpoint["state_dict"]
        excluded_keys = [k for k in state_dict
                         if k.startswith("synth")
                         or k.startswith("loss_func.jtfs")
                         or k.startswith("prev_path_grads")  # Tmp to reduce chkpt size
                         or k.startswith("cqt")]
        for k in excluded_keys:
            del state_dict[k]

    @staticmethod
    def calc_total_variation(x: List[T], norm_by_len: bool = True) -> T:
        diffs = tr.stack(
            [tr.abs(x[idx + 1] - x[idx]) for idx in range(len(x) - 1)],
            dim=0,
        )
        assert diffs.ndim == 1
        if norm_by_len:
            return diffs.mean()
        else:
            return diffs.sum()

    @staticmethod
    def calc_cqt(x: T, cqt: CQT, cqt_eps: float = 1e-3) -> T:
        U = cqt(x)
        U = tr.log1p(U / cqt_eps)
        return U
