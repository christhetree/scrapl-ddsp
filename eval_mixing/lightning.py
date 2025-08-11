import logging
import os
from typing import Dict

import auraloss
import pytorch_lightning as pl
import torch as tr
from torch import Tensor as T
from torch import nn

from automix.models.dmc import Mixer

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class MixingLightingModule(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        emb_dim: int,
        mixer: nn.Module,
        loss_func: nn.Module,
        pp_hidden_dim: int = 256,
        pp_dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = encoder
        self.emb_dim = emb_dim
        self.mixer = mixer
        self.loss_func = loss_func
        self.pp_hidden_dim = pp_hidden_dim
        self.pp_dropout = pp_dropout

        self.post_processor = nn.Sequential(
            tr.nn.Linear(2 * emb_dim, pp_hidden_dim),
            tr.nn.Dropout(pp_dropout),
            tr.nn.PReLU(),
            tr.nn.Linear(pp_hidden_dim, pp_hidden_dim),
            tr.nn.Dropout(pp_dropout),
            tr.nn.PReLU(),
            tr.nn.Linear(pp_hidden_dim, self.mixer.num_params),
            tr.nn.Sigmoid(),
        )
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
        x, y, track_mask = batch  # tracks, mix, mask

        bs, num_tracks, n_samples = x.shape
        # If no track_mask supplied assume all tracks active
        if track_mask is None:
            track_mask = tr.zeros(bs, num_tracks).type_as(x).bool()

        # Move tracks to the batch dimension to fully parallelize embedding computation
        x = x.view(-1, n_samples)

        # Generate a single embedding for each track
        track_emb = self.encoder(x)
        track_emb = track_emb.view(bs, num_tracks, -1)  # (bs, num_tracks, d_embed)

        # Generate the "context" embedding
        context_emb = []
        for bidx in range(bs):
            c_n = track_emb[bidx, ~track_mask[bidx, :], :].mean(
                dim=0, keepdim=True
            )  # (bs, 1, d_embed)
            c_n = c_n.repeat(num_tracks, 1)  # (bs, num_tracks, d_embed)
            context_emb.append(c_n)
        context_emb = tr.stack(context_emb, dim=0)

        # ==============================================================================
        # # Compute sum over non-masked tracks
        # sum_emb = (track_emb * ~track_mask.unsqueeze(-1)).sum(dim=1)  # (bs, d_embed)
        # count = (~track_mask).sum(dim=1, keepdim=True)  # (bs, 1)
        #
        # # Mean embedding per batch
        # mean_emb = sum_emb / count  # (bs, d_embed)
        #
        # # Expand to (bs, num_tracks, d_embed)
        # context_emb2 = mean_emb.unsqueeze(1).expand(-1, num_tracks, -1)
        # assert tr.allclose(context_emb, context_emb2)
        # ==============================================================================

        # Fuse the track embs and context embs
        emb = tr.cat((track_emb, context_emb), dim=-1)  # (bs, num_tracks, 2 * d_embed)

        # Estimate mixing parameters for each track (in parallel)
        p_hat_0to1 = self.post_processor(emb)  # (bs, num_tracks, num_params)

        # Generate the stereo mix
        x = x.view(bs, num_tracks, -1)  # move tracks back from batch dim
        y_hat, p_hat = self.mixer(x, p_hat_0to1)  # (bs, 2, n_samples) # and denormalized params

        # Compute loss
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
