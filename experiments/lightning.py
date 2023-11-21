import logging
import os
from typing import Dict

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
                 cqt_eps: float = 1e-3,
                 log_x: bool = False,
                 log_x_hat: bool = False):
        super().__init__()
        self.save_hyperparameters(ignore=["loss_func", "model", "synth"])
        log.info(f"\n{self.hparams}")
        self.model = model
        self.synth = synth
        self.loss_func = loss_func
        self.use_p_loss = use_p_loss
        self.J_cqt = J_cqt
        self.feature_type = feature_type
        self.cqt_eps = cqt_eps
        self.log_x = log_x
        self.log_x_hat = log_x_hat

        cqt_params = {
            "sr": synth.sr,
            "bins_per_octave": synth.Q,
            "n_bins": J_cqt * synth.Q,
            "hop_length": synth.hop_len,
            "fmin": (0.4 * synth.sr) / (2 ** J_cqt),
            "output_format": "Magnitude",
            "verbose": False,
        }
        self.cqt = CQT(**cqt_params)
        self.loss_name = self.loss_func.__class__.__name__
        self.l1 = nn.L1Loss()

    def make_x_from_theta(self, theta_density: T, theta_slope: T, seed: T) -> T:
        # TODO(cm): add batch support to synth
        x = []
        for idx in range(theta_density.size(0)):
            curr_x = self.synth(theta_density[idx], theta_slope[idx], seed[idx])
            x.append(curr_x)
        x = tr.stack(x, dim=0).unsqueeze(1)  # Unsqueeze channel dim
        return x

    def step(self, batch: (T, T, T, T), stage: str) -> (T, Dict[str, T]):
        U, theta_density, theta_slope, seed = batch
        U_hat = None
        x = None
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
            x = self.make_x_from_theta(theta_density, theta_slope, seed)
            x_hat = self.make_x_from_theta(theta_density_hat, theta_slope_hat, seed)
            if self.feature_type == "cqt":
                U_hat = ChirpTextureDataset.calc_cqt(x_hat, self.cqt, self.cqt_eps)
                # TODO(cm): this fails on GPU since CPU generates random differently
                # U_tmp = ChirpTextureDataset.calc_cqt(x, self.cqt, self.cqt_eps)
                # assert tr.allclose(U, U_tmp, atol=1e-5)
            loss = self.loss_func(x_hat, x)
            self.log(f"{stage}/{self.loss_name}",
                     loss,
                     prog_bar=True,
                     sync_dist=True)

        self.log(f"{stage}/l1_d", density_mae, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/l1_s", slope_mae, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/loss", loss, prog_bar=False, sync_dist=True)

        if x is None and self.log_x:
            x = self.make_x_from_theta(theta_density, theta_slope, seed)
        if x_hat is None and self.log_x_hat:
            x_hat = self.make_x_from_theta(theta_density_hat, theta_slope_hat, seed)
            U_hat = ChirpTextureDataset.calc_cqt(x_hat, self.cqt, self.cqt_eps)

        data_dict = {
            "U": U,
            "U_hat": U_hat,
            "x": x,
            "x_hat": x_hat,
            "theta_density": theta_density,
            "theta_density_hat": theta_density_hat,
            "theta_slope": theta_slope,
            "theta_slope_hat": theta_slope_hat,
        }
        data_dict = {k: v.detach().float().cpu()
                     for k, v in data_dict.items() if v is not None}
        return loss, data_dict

    def training_step(self, batch: (T, T, T, T), batch_idx: int) -> T:
        loss, _ = self.step(batch, stage="train")
        return loss

    def validation_step(self, batch: (T, T, T, T), stage: str) -> (T, Dict[str, T]):
        out = self.step(batch, stage="val")
        return out

    def test_step(self, batch: (T, T, T, T), stage: str) -> (T, Dict[str, T]):
        out = self.step(batch, stage="test")
        return out