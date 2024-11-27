import functools
import logging
import os
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime
from typing import Dict, Optional, List, Any

import pytorch_lightning as pl
import torch as tr
from nnAudio.features import CQT
from torch import Tensor as T
from torch import nn
from torch.autograd.profiler import record_function

from experiments.losses import JTFSTLoss, SCRAPLLoss, AdaptiveSCRAPLLoss, Scat1DLoss
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

        self.param_idx_hooks_in_use = set()
        self.model_param_grads = None
        self.first_batch = None

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

    @staticmethod
    def calc_mag_entropy(x: T, eps: float = 1e-8) -> T:
        x = x.abs()
        x = x / (x.sum() + eps)
        entropy = -x * tr.log(x + eps)
        entropy = entropy.sum()
        entropy = entropy / (tr.log(tr.tensor(x.numel())) + eps)
        return entropy

    def functional_loss(
        self,
        model_p: ([str, T], Dict[str, T]),
        synth_p: ([str, T], Dict[str, T]),
        loss_p: ([str, T], Dict[str, T]),
        x: T,
        U: T,
        seed_hat: T,
        curr_path_idx: Optional[int] = None,
    ) -> (T, (T, T, T)):
        theta_d_0to1_hat, theta_s_0to1_hat = tr.func.functional_call(
            self.model, model_p, U, strict=True
        )
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
        return loss, (theta_d_0to1_hat, theta_s_0to1_hat, x_hat)

    def specific_model_p_functional_loss(
        self,
        p: T,
        p_name: str,
        model_p: ([str, T], Dict[str, T]),
        synth_p: ([str, T], Dict[str, T]),
        loss_p: ([str, T], Dict[str, T]),
        x: T,
        U: T,
        seed_hat: T,
        curr_path_idx: Optional[int] = None,
    ) -> T:
        model_params, _ = model_p
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

    def calc_model_param_hessian(
        self,
        param_name: str,
        x: T,
        U: T,
        seed_hat: T,
        use_autograd: bool = False,
        use_profiler: bool = False,
    ) -> T:
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

        param = model_params[param_name]
        log.debug(f"param_name = {param_name}, param.shape = {param.shape}")
        ag_fn = functools.partial(
            self.specific_model_p_functional_loss,
            p_name=param_name,
            model_p=model_p,
            synth_p=synth_p,
            loss_p=loss_p,
            x=x,
            U=U,
            seed_hat=seed_hat,
            curr_path_idx=curr_path_idx,
        )

        with (
            tr.profiler.profile(
                activities=[tr.profiler.ProfilerActivity.CPU],
                with_stack=True,
                profile_memory=True,
                record_shapes=False,
            )
            if use_profiler
            else nullcontext()
        ) as prof:
            if use_autograd:
                with record_function("hessian") if use_profiler else nullcontext():
                    hess = tr.autograd.functional.hessian(
                        ag_fn,
                        param,
                        create_graph=False,
                        strict=True,
                        vectorize=False,
                        outer_jacobian_strategy="reverse-mode",
                    )
            else:
                with record_function("jacrev") if use_profiler else nullcontext():
                    hess = tr.func.jacrev(
                        tr.func.jacrev(self.specific_model_p_functional_loss)
                    )(
                        param,
                        param_name,
                        model_p,
                        synth_p,
                        loss_p,
                        x,
                        U,
                        seed_hat,
                        curr_path_idx,
                    )
        if use_profiler:
            log.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1))
        return hess

    # def backward(self, loss: T, **kwargs: Dict[str, Any]) -> None:
    #     loss.backward(create_graph=True, **kwargs)

    def grad_multiplier_hook(self, grad: T) -> T:
        # log.info(f"grad.abs().max() = {grad.abs().max()}")
        if not self.training:
            log.warning("grad_multiplier_hook called during eval")
            return grad
        grad *= self.grad_multiplier
        return grad

    def vr_hook(self, grad: T, param_idx: int, scrapl: AdaptiveSCRAPLLoss) -> T:
        if not self.training:
            log.warning("vr_hook called during eval")
            return grad

        # # Hessian ====================================================================
        # if param_idx in self.param_idx_hooks_in_use:
        #     log.debug(f"param_idx {param_idx} hook is currently in use")
        #     return grad
        #
        # self.param_idx_hooks_in_use.add(param_idx)
        # assert not grad.isnan().any()
        # p = list(self.model.parameters())[param_idx]
        # p_name = list(self.model.named_parameters())[param_idx][0]
        # grad_2 = self.model_param_grads[p_name]
        # assert tr.allclose(grad, grad_2)
        # assert util.is_connected_via_ad_graph(grad, p)
        #
        # def calc_hess_row(g: T, p: T) -> T:
        #     assert g.numel() == 1
        #     hess_row = tr.autograd.grad(
        #         g,
        #         [p],
        #         retain_graph=True,
        #         create_graph=False,
        #     )[0]
        #     hess_row = hess_row.view(-1)
        #     return hess_row
        #
        # grad_flat = grad.view(-1)
        # log.info(f"calc hess, p.shape = {p.shape}")
        # h_rows = [calc_hess_row(g, p) for g in tqdm(grad_flat)]
        # hess = tr.stack(h_rows, dim=0)
        # log.info(f"hess.shape = {hess.shape}, hess[0, 0] = {hess[0, 0]}")
        # assert tr.allclose(hess, hess.t())
        # self.param_idx_hooks_in_use.remove(param_idx)
        # # Hessian ====================================================================

        path_idx = scrapl.curr_path_idx
        assert path_idx is not None
        # path_idx = 250
        self.paths_seen.add(path_idx)
        n_paths = scrapl.n_paths
        curr_t = self.global_step + 1
        # curr_t = 1

        # save_param_idx = None
        # # save_param_idx = 15
        # if save_param_idx is None or param_idx == save_param_idx:
        #     save_path = os.path.join(
        #         OUT_DIR, f"{self.run_name}__w_{param_idx}_{curr_t}_{path_idx}.pt"
        #     )
        #     weight = list(self.parameters())[param_idx]
        #     assert weight.shape == grad.shape
        #     tr.save(weight.detach().cpu(), save_path)
        #
        #     save_path = os.path.join(
        #         OUT_DIR, f"{self.run_name}__g_raw_{param_idx}_{curr_t}_{path_idx}.pt"
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

        # if save_param_idx is None or param_idx == save_param_idx:
        #     save_path = os.path.join(
        #         OUT_DIR, f"{self.run_name}__g_adam_{param_idx}_{curr_t}_{path_idx}.pt"
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

            # if save_param_idx is None or param_idx == save_param_idx:
            #     save_path = os.path.join(
            #         OUT_DIR,
            #         f"{self.run_name}__g_saga_{param_idx}_{curr_t}_{path_idx}.pt"
            #     )
            #     tr.save(saga_grad.detach().cpu(), save_path)

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

        # # Func grad calculation ======================================================
        # model_params = {k: v for k, v in self.model.named_parameters()}
        # name_to_idx = {k: idx for idx, k in enumerate(model_params)}
        #
        # curr_t = self.global_step + 1
        # path_idx = self.loss_func.curr_path_idx
        # save_dir = os.path.join(OUT_DIR, f"{self.run_name}")
        # os.makedirs(save_dir, exist_ok=True)
        #
        # model_grads, loss_2, theta_d_2, theta_s_2, x_hat_2 = (
        #     self.calc_model_param_grads(x, U, seed_hat)
        # )
        # for name, grad in model_grads.items():
        #     p = model_params[name]
        #     assert p.shape == grad.shape
        #     p_idx = name_to_idx[name]
        #     save_path = os.path.join(
        #         save_dir, f"{self.run_name}__w_{p_idx}_{curr_t}_{path_idx}.pt"
        #     )
        #     tr.save(p.detach().cpu(), save_path)
        #     save_path = save_path.replace("__w_", "__g_raw_")
        #     tr.save(grad.detach().cpu(), save_path)
        #
        # assert tr.allclose(loss_2, loss)
        # assert tr.allclose(theta_d_2, theta_d_0to1_hat)
        # assert tr.allclose(theta_s_2, theta_s_0to1_hat)
        # if x_hat is not None:
        #     assert tr.allclose(x_hat_2, x_hat)
        # # Func grad calculation ======================================================
        #
        # # Hessian calculation ========================================================
        # # param_name = "cnn.3.weight"  # numel = 1
        # # param_name = "fc_d.bias"     # numel = 32
        # # param_name = "fc.bias"       # numel = 64
        # # param_name = "cnn.17.bias"   # numel = 128
        # max_numel = 1
        # # max_numel = 32
        # hess_param_names = [name for name, p in model_params.items()
        #                     if p.numel() <= max_numel]
        # log.info(f"max_numel = {max_numel}, hess_param_names = {hess_param_names}")
        # for p_name in tqdm(hess_param_names):
        #     p_idx = name_to_idx[p_name]
        #     hess = self.calc_model_param_hessian(
        #         p_name, x, U, seed_hat, use_autograd=False, use_profiler=False
        #     )
        #     # log.info(f"hess.shape = {hess.shape}, hess = {hess}")
        #     save_path = os.path.join(
        #         save_dir, f"{self.run_name}__h_{p_idx}_{curr_t}_{path_idx}.pt"
        #     )
        #     tr.save(hess.detach().cpu(), save_path)
        # # Hessian calculation ========================================================

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
        excluded_keys = [k for k in state_dict if k.startswith("synth")]
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
