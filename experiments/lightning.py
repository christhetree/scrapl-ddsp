import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional, List, Any

import pytorch_lightning as pl
import torch as tr
from nnAudio.features import CQT
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor as T
from torch import nn

from experiments.losses import JTFSTLoss, Scat1DLoss
from experiments.paths import OUT_DIR
from experiments.scrapl_loss import SCRAPLLoss

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class SCRAPLLightingModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        synth: nn.Module,
        loss_func: nn.Module,
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
        grad_mult: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.synth = synth
        self.loss_func = loss_func
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
        self.grad_mult = grad_mult
        if type(self.loss_func) in {
            JTFSTLoss,
            Scat1DLoss,
        }:
            assert self.grad_mult != 1.0
        else:
            assert self.grad_mult == 1.0

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

        if grad_mult != 1.0:
            assert not isinstance(self.loss_func, SCRAPLLoss)
            log.info(f"Adding grad multiplier hook of {self.grad_mult}")
            for p in self.model.parameters():
                p.register_hook(self.grad_multiplier_hook)

        for p in self.synth.parameters():
            p.requires_grad = False

        if isinstance(self.loss_func, SCRAPLLoss):
            params = list(self.model.parameters())
            self.loss_func.attach_params(params)

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

    def on_train_start(self) -> None:
        self.global_n = 0

    def on_train_epoch_start(self) -> None:
        try:
            if self.loss_func.use_pwa:
                assert (
                    self.trainer.accumulate_grad_batches == 1
                ), "Pathwise ADAM does not support gradient accumulation"
            if self.loss_func.use_saga:
                assert (
                    self.trainer.accumulate_grad_batches == 1
                ), "SAGA does not support gradient accumulation"
        except AttributeError:
            pass

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        # if self.update_paths:
        #     assert isinstance(self.loss_func, AdaptiveSCRAPLLoss)
        #     assert self.loss_func.curr_path_idx is not None
        pass

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
        excluded_keys = [
            k
            for k in state_dict
            if k.startswith("synth")
            or k.startswith("loss_func.jtfs")
            or k.startswith("loss_func.prev_path_grads")  # Tmp to reduce chkpt size
            or k.startswith("cqt")
        ]
        for k in excluded_keys:
            del state_dict[k]

    def grad_multiplier_hook(self, grad: T) -> T:
        # log.info(f"grad.abs().max() = {grad.abs().max()}")
        if not self.training:
            log.warning("grad_multiplier_hook called during eval")
            return grad
        grad *= self.grad_mult
        return grad

    def save_grad_hook(self, grad: T, name: str, curr_t: Optional[int] = None) -> T:
        if not self.training:
            log.warning("save_grad_hook called during eval")
            return grad

        if curr_t is None:
            curr_t = self.global_step
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

    def calc_U(self, x: T) -> T:
        if self.feature_type == "cqt":
            return SCRAPLLightingModule.calc_cqt(x, self.cqt, self.cqt_eps)
        else:
            raise NotImplementedError

    def step(self, batch: (T, T, T), stage: str) -> Dict[str, T]:
        theta_d_0to1, theta_s_0to1, seed, batch_indices = batch
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
            # x_hat = self.synth(theta_d_0to1_hat, theta_s_0to1_hat, seed_hat, seed_target=seed)
            with tr.no_grad():
                U_hat = self.calc_U(x_hat)
            loss = self.loss_func(x_hat, x)
            self.log(f"{stage}/{self.loss_name}", loss, prog_bar=True, sync_dist=True)

        self.log(f"{stage}/l1_d", l1_d, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/l1_s", l1_s, prog_bar=True, sync_dist=True)
        theta_mae = (l1_d + l1_s) / 2
        self.log(f"{stage}/l1_theta", theta_mae, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/loss", loss, prog_bar=False, sync_dist=True)

        with tr.no_grad():
            if x is None and self.log_x:
                x = self.synth(theta_d_0to1, theta_s_0to1, seed)
            if x_hat is None and self.log_x_hat:
                x_hat = self.synth(theta_d_0to1_hat, theta_s_0to1_hat, seed_hat)
                # x_hat = self.synth(theta_d_0to1_hat, theta_s_0to1_hat, seed_hat, seed_target=seed)
                U_hat = self.calc_U(x_hat)

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
