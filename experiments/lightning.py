import logging
import os
from contextlib import suppress
from typing import Dict

import pytorch_lightning as pl
import torch as tr
from nnAudio.features import CQT
from torch import Tensor as T
from torch import nn

from experiments.losses import JTFSTLoss, SCRAPLLoss
from experiments.synth import ChirpTextureSynth, make_x_from_theta

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class SCRAPLLightingModule(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 synth: ChirpTextureSynth,
                 loss_func: nn.Module,
                 use_p_loss: bool = False,
                 use_rand_seed: bool = False,
                 use_rand_seed_hat: bool = False,
                 feature_type: str = "cqt",
                 J_cqt: int = 5,
                 cqt_eps: float = 1e-3,
                 log_x: bool = False,
                 log_x_hat: bool = False,
                 log_val_grads: bool = False):
        super().__init__()
        self.model = model
        self.synth = synth
        self.loss_func = loss_func
        self.use_p_loss = use_p_loss
        self.use_rand_seed = use_rand_seed
        self.use_rand_seed_hat = use_rand_seed_hat
        self.J_cqt = J_cqt
        self.feature_type = feature_type
        self.cqt_eps = cqt_eps
        self.log_x = log_x
        self.log_x_hat = log_x_hat
        self.log_val_grads = log_val_grads

        cqt_params = {
            "sr": synth.sr,
            "bins_per_octave": synth.Q,
            "n_bins": J_cqt * synth.Q,
            "hop_length": synth.hop_len,
            # TODO(cm): check this
            "fmin": (0.4 * synth.sr) / (2 ** J_cqt),
            "output_format": "Magnitude",
            "verbose": False,
        }
        self.cqt = CQT(**cqt_params)
        self.loss_name = self.loss_func.__class__.__name__
        self.l1 = nn.L1Loss()
        self.global_n = 0
        # self.register_buffer("path_grads_density", tr.zeros(3072, self.loss_func.n_paths))
        # self.register_buffer("path_grads_slope", tr.zeros(3072, self.loss_func.n_paths))

    def on_train_start(self) -> None:
        self.global_n = 0

    def calc_U(self, x: T) -> T:
        if self.feature_type == "cqt":
            return SCRAPLLightingModule.calc_cqt(x, self.cqt, self.cqt_eps)
        else:
            raise NotImplementedError

    def sag_hook(self, grad: T, batch_indices: T, path_idx: int, path_grads: T,
                 decay_val: float = 1.0) -> T:
        assert path_idx is not None
        if decay_val != 1.0:
            path_grads.mul_(decay_val)
        path_grads[batch_indices, path_idx, ...] = grad
        grad = path_grads[batch_indices, ...]
        grad = tr.mean(grad, dim=1)
        return grad

    def step(self, batch: (T, T, T), stage: str) -> Dict[str, T]:
        theta_density, theta_slope, seed, batch_indices = batch
        batch_size = theta_density.size(0)
        if stage == "train":
            self.global_n = (self.global_step *
                             self.trainer.accumulate_grad_batches *
                             batch_size)
        # TODO(cm): check if this works for DDP
        self.log(f"global_n", float(self.global_n), sync_dist=True)

        seed_range = 9999999
        if self.use_rand_seed:
            seed = tr.randint_like(seed, low=0, high=seed_range)
        seed_hat = seed
        if self.use_rand_seed_hat:
            max_seed = max(seed_range, seed.max())
            seed_hat = tr.randint_like(seed,
                                       low=max_seed + 1,
                                       high=max_seed + seed_range)

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
            self.log(f"{stage}/p_loss_{self.loss_name}",
                     loss,
                     prog_bar=True,
                     sync_dist=True)
        else:
            x_hat = make_x_from_theta(self.synth,
                                      theta_density_hat,
                                      theta_slope_hat,
                                      seed_hat)
            with tr.no_grad():
                U_hat = self.calc_U(x_hat)
            loss = self.loss_func(x_hat, x)
            self.log(f"{stage}/{self.loss_name}",
                     loss,
                     prog_bar=True,
                     sync_dist=True)

        self.log(f"{stage}/l1_d", density_mae, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/l1_s", slope_mae, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/loss", loss, prog_bar=False, sync_dist=True)

        with tr.no_grad():
            if x is None and self.log_x:
                x = make_x_from_theta(self.synth, theta_density, theta_slope, seed)
            if x_hat is None and self.log_x_hat:
                x_hat = make_x_from_theta(self.synth,
                                          theta_density_hat,
                                          theta_slope_hat,
                                          seed_hat)
                U_hat = self.calc_U(x_hat)

        with suppress(Exception):
            logits = self.loss_func.logits
            probs = tr.softmax(logits, dim=-1)
            top_n = 8
            top = tr.topk(probs, k=top_n, dim=-1)
            percentages = [f"{p:.6f}" for p in top.values]
            log.info(f"Top {top_n} logits: {top.indices} {percentages}")
            path_counts = self.loss_func.path_counts
            path_counts = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)
            path_counts = list(path_counts)[:top_n]
            log.info(f"sorted path_counts: {path_counts}")

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

        # if stage == "train":
        #     theta_density_hat.register_hook(
        #         partial(self.sag_hook,
        #                 batch_indices=batch_indices,
        #                 path_idx=self.loss_func.curr_path_idx,
        #                 path_grads=self.path_grads_density))
        #     theta_slope_hat.register_hook(
        #         partial(self.sag_hook,
        #                 batch_indices=batch_indices,
        #                 path_idx=self.loss_func.curr_path_idx,
        #                 path_grads=self.path_grads_slope))
        return out_dict

    def training_step(self, batch: (T, T, T), batch_idx: int) -> Dict[str, T]:
        if not (isinstance(self.loss_func, JTFSTLoss)
                or isinstance(self.loss_func, SCRAPLLoss)):
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
