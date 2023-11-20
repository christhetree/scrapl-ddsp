import logging
import os
from typing import Optional

import pytorch_lightning as pl
import torch as tr
from nnAudio.features import CQT
from torch import Tensor as T
from torch import nn

from experiments.datasets import ChirpTextureDataset
from experiments.synth import ChirpTextureSynth

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class SCRAPLLightingModule(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 synth: ChirpTextureSynth,
                 loss_func: nn.Module,
                 use_p_loss: bool = False,
                 feature_type: str = "cqt",
                 J_cqt: int = 5,
                 cqt_eps: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        log.info(f"\n{self.hparams}")
        self.model = model
        self.synth = synth
        self.loss_func = loss_func
        self.use_p_loss = use_p_loss
        self.J_cqt = J_cqt
        self.feature_type = feature_type
        self.cqt_eps = cqt_eps

        cqt_params = {
            "sr": synth.sr,
            "bins_per_octave": synth.Q,
            "n_bins": J_cqt * synth.Q,
            "hop_length": synth.hop_len,
            "fmin": (0.4 * synth.sr) / (2 ** J_cqt),
        }
        self.cqt = CQT(**cqt_params)
        self.loss_name = self.loss_func.__class__.__name__
        self.l1 = nn.L1Loss()

    def step(self, batch: (T, T, T, T), stage: str) -> (T, T, Optional[T], ...):
        U, theta_density, theta_slope, seed = batch
        U_hat = None
        x = None,
        x_hat = None

        theta_density_hat, theta_slope_hat = self.model(U)
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
            # TODO(cm): add batch support to synth
            x = []
            x_hat = []
            for idx in range(len(U)):
                curr_x = self.synth(theta_density[idx], theta_slope[idx], seed[idx])
                curr_x_hat = self.synth(theta_density_hat[idx],
                                        theta_slope_hat[idx],
                                        seed[idx])
                x.append(curr_x)
                x_hat.append(curr_x_hat)
            x = tr.stack(x, dim=0)
            x_hat = tr.stack(x_hat, dim=0)

            if self.feature_type == "cqt":
                U_hat = ChirpTextureDataset.calc_cqt(x_hat, self.cqt, self.cqt_eps)
                U_tmp = ChirpTextureDataset.calc_cqt(x, self.cqt, self.cqt_eps)
                assert tr.allclose(U, U_tmp, atol=1e-5)
            loss = self.loss_func(x_hat, x)
            self.log(f"{stage}/{self.loss_name}",
                     loss,
                     prog_bar=True,
                     sync_dist=True)

        self.log(f"{stage}/l1_d", density_mae, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/l1_s", slope_mae, prog_bar=True, sync_dist=True)

        return loss, U, U_hat, x, x_hat

    def training_step(self, batch: (T, T, T, T), batch_idx: int) -> T:
        loss, _, _, _, _ = self.step(batch, stage="train")
        return loss

    def validation_step(self, batch: (T, T, T, T), stage: str) -> (T, T, Optional[T], ...):
        out = self.step(batch, stage="val")
        return out

    def test_step(self, batch: (T, T, T, T), stage: str) -> (T, T, Optional[T], ...):
        out = self.step(batch, stage="test")
        return out
