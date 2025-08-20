import logging
import os
from collections import defaultdict
from typing import Dict, List

import auraloss
import pytorch_lightning as pl
import torch as tr
from torch import Tensor as T
from torch import nn

from automix.evaluation.utils_evaluation import get_features
from experiments.losses import MFCCDistance, Scat1DLoss
from experiments.scrapl_loss import SCRAPLLoss

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
        grad_mult: float = 1.0,
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
        self.grad_mult = grad_mult

        if grad_mult != 1.0:
            assert not isinstance(self.loss_func, SCRAPLLoss)
            log.info(f"Adding grad multiplier hook of {self.grad_mult}")
            for p in self.encoder.parameters():
                p.register_hook(self.grad_multiplier_hook)

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

        if tr.cuda.is_available():
            self.encoder = tr.compile(self.encoder)
            self.track_post_processor = tr.compile(self.track_post_processor)
            if self.global_post_processor is not None:
                self.global_post_processor = tr.compile(self.global_post_processor)

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
                "mfcc_l1": MFCCDistance(sr=mixer.sample_rate),
                "scat1d_l1": Scat1DLoss(
                    shape=65536, J=12, Q1=8, Q2=2, max_order=1, p=1
                ),
            }
        )

    def grad_multiplier_hook(self, grad: T) -> T:
        # log.info(f"grad.abs().max() = {grad.abs().max()}")
        if not self.training:
            log.warning("grad_multiplier_hook called during eval")
            return grad
        grad *= self.grad_mult
        return grad

    def step(self, batch: List[T], stage: str) -> Dict[str, T]:
        x, y, track_mask = batch  # tracks, mix, mask
        y = y.contiguous()

        bs, num_tracks, n_samples = x.shape
        # If no track_mask supplied assume all tracks active
        if track_mask is None:
            track_mask = tr.zeros(bs, num_tracks).type_as(x).bool()

        # Move tracks to the batch dimension to fully parallelize embedding computation
        x = x.view(-1, n_samples)

        # for idx in range(x.size(0)):
        #     track = x[idx, :]
        #     shift_samples = tr.randint(low=-8192, high=8192, size=(1,)).item()
        #     track = tr.roll(track, shifts=shift_samples, dims=0)
        #     x[idx, :] = track

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
                use_master_bus=False,
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
        if stage != "train":
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

        if stage != "train":
            eval_features = defaultdict(list)
            bs = y.size(0)
            y_numpy = y.detach().cpu().numpy()
            y_hat_numpy = y_hat.detach().cpu().numpy()
            for idx in range(bs):
                try:
                    curr_eval_features = get_features(
                        y_numpy[idx, :, :],
                        y_hat_numpy[idx, :, :],
                        sr=self.mixer.sample_rate,
                    )
                    for k, v in curr_eval_features.items():
                        eval_features[k].append(v)
                except Exception as e:
                    log.warning(f"Exception during get_features: {e}")
            for name, vals in eval_features.items():
                if len(vals) == 0:
                    log.warning(f"No values for {name} in {stage} stage.")
                    continue
                mean_val = tr.tensor(vals, dtype=tr.float).mean()
                self.log(
                    f"{stage}/mape__{name}",
                    mean_val,
                    prog_bar=False,
                    sync_dist=True,
                )

        return out

    def training_step(self, batch: (T, T, T), batch_idx: int) -> Dict[str, T]:
        return self.step(batch, stage="train")

    def validation_step(self, batch: (T, T, T), batch_idx: int) -> Dict[str, T]:
        return self.step(batch, stage="val")

    def test_step(self, batch: (T, T, T), batch_idx: int) -> Dict[str, T]:
        return self.step(batch, stage="test")
