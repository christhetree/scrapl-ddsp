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
        encoder: nn.Module,
        emb_dim: int,
        mixer: nn.Module,
        n_track_params: int,
        n_fx_bus_params: int,
        n_master_bus_params: int,
        loss_func: nn.Module,
        pp_hidden_dim: int = 256,
        pp_dropout: float = 0.2,
    ):
        super().__init__()
        assert n_track_params > 0, "n_track_params must be greater than 0"
        self.encoder = encoder
        self.emb_dim = emb_dim
        self.mixer = mixer
        self.n_track_params = n_track_params
        self.n_fx_bus_params = n_fx_bus_params
        self.n_master_bus_params = n_master_bus_params
        self.loss_func = loss_func
        self.pp_hidden_dim = pp_hidden_dim
        self.pp_dropout = pp_dropout

        self.n_global_params = n_fx_bus_params + n_master_bus_params
        self.track_post_processor = nn.Sequential(
            tr.nn.Linear(2 * emb_dim, pp_hidden_dim),
            tr.nn.Dropout(pp_dropout),
            tr.nn.PReLU(),
            tr.nn.Linear(pp_hidden_dim, pp_hidden_dim),
            tr.nn.Dropout(pp_dropout),
            tr.nn.PReLU(),
            tr.nn.Linear(pp_hidden_dim, n_track_params),
            tr.nn.Sigmoid(),
        )
        self.global_post_processor = None
        if self.n_global_params > 0:
            self.global_post_processor = nn.Sequential(
                tr.nn.Linear(emb_dim, pp_hidden_dim),
                tr.nn.Dropout(pp_dropout),
                tr.nn.PReLU(),
                tr.nn.Linear(pp_hidden_dim, pp_hidden_dim),
                tr.nn.Dropout(pp_dropout),
                tr.nn.PReLU(),
                tr.nn.Linear(pp_hidden_dim, self.n_global_params),
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

        # Move tracks back from batch dim
        x = x.view(bs, num_tracks, -1)

        # # Generate the "context" embedding
        # context_emb = []
        # for bidx in range(bs):
        #     c_n = track_emb[bidx, ~track_mask[bidx, :], :].mean(
        #         dim=0, keepdim=True
        #     )  # (bs, 1, d_embed)
        #     c_n = c_n.repeat(num_tracks, 1)  # (bs, num_tracks, d_embed)
        #     context_emb.append(c_n)
        # context_emb = tr.stack(context_emb, dim=0)

        # ==============================================================================
        # Compute sum over non-masked tracks
        sum_emb = (track_emb * ~track_mask.unsqueeze(-1)).sum(dim=1)  # (bs, d_embed)
        count = (~track_mask).sum(dim=1, keepdim=True)  # (bs, 1)

        # Mean embedding per batch
        mean_emb = sum_emb / count  # (bs, d_embed)

        # Expand to (bs, num_tracks, d_embed)
        context_emb = mean_emb
        # assert tr.allclose(context_emb, context_emb2)
        # ==============================================================================

        # Fuse the track embs and context embs
        context_emb_expanded = context_emb.unsqueeze(1).expand(-1, num_tracks, -1)
        fused_emb = tr.cat(
            (track_emb, context_emb_expanded), dim=-1
        )  # (bs, num_tracks, 2 * d_embed)

        # Estimate mixing parameters for each track (in parallel)
        p_track_hat_0to1 = self.track_post_processor(
            fused_emb
        )  # (bs, num_tracks, num_params)
        if self.n_global_params > 0:
            p_global_hat_0to1 = self.global_post_processor(context_emb)
            p_fx_bus_hat_0to1 = p_global_hat_0to1[:, : self.n_fx_bus_params]
            p_master_bus_hat_0to1 = p_global_hat_0to1[:, self.n_fx_bus_params :]
            y_track_hat, y_hat, p_track_hat, p_fx_hat, p_master_hat = self.mixer(
                tracks=x,
                track_params=p_track_hat_0to1,
                fx_bus_params=p_fx_bus_hat_0to1,
                master_bus_params=p_master_bus_hat_0to1,
            )
        else:
            y_hat, p_hat = self.mixer(
                x, p_track_hat_0to1
            )  # (bs, 2, n_samples) # and denormalized params

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
