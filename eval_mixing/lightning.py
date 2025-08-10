import logging
import os
from typing import Dict

import auraloss
import pytorch_lightning as pl
import torch as tr
from torch import Tensor as T
from torch import nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class MixingLightingModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_func: nn.Module,
    ):
        super().__init__()
        self.model = model
        self.loss_func = loss_func
        self.metrics = nn.ModuleDict(
            {
                "mss": auraloss.freq.MultiResolutionSTFTLoss(),
                "sisdr": auraloss.time.SISDRLoss(),
                "mss_mixing": auraloss.freq.MultiResolutionSTFTLoss(
                    fft_sizes=[512, 2048, 8192],
                    hop_sizes=[256, 1024, 4096],
                    win_lengths=[512, 2048, 8192],
                    w_sc=0.0,
                    w_phs=0.0,
                    w_lin_mag=1.0,
                    w_log_mag=1.0,
                ),
            }
        )

    def step(self, batch: (T, T, T), stage: str) -> Dict[str, T]:
        x, y, mask = batch  # tracks, mix, mask
        y_hat, params = self.model(x, mask)
        loss = self.loss_func(y_hat, y)
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
        out = {
            "loss": loss,
        }
        with tr.no_grad():
            for name, metric in self.metrics.items():
                metric_value = metric(y_hat, y)
                self.log(
                    f"{stage}/{name}", metric_value, prog_bar=False, sync_dist=True
                )
                assert (
                    name not in out
                ), f"Metric {name} already exists in output dictionary."
                out[name] = metric_value
        return out

    def training_step(self, batch: (T, T, T), batch_idx: int) -> Dict[str, T]:
        return self.step(batch, stage="train")

    def validation_step(self, batch: (T, T, T), batch_idx: int) -> Dict[str, T]:
        return self.step(batch, stage="val")

    def test_step(self, batch: (T, T, T), batch_idx: int) -> Dict[str, T]:
        return self.step(batch, stage="test")
