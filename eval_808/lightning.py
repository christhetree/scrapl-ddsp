import logging
import os
import time
from datetime import datetime
from typing import Dict, Optional, Literal, List, Tuple, Any

import auraloss
import pytorch_lightning as pl
import torch as tr
import torchaudio
from nnAudio.features import CQT
from torch import Tensor as T
from torch import nn

from eval_808.features import FeatureCollection, CascadingFrameExtactor
from experiments import util
from experiments.lightning import SCRAPLLightingModule
from experiments.losses import JTFSTLoss, Scat1DLoss, MFCCDistance
from experiments.paths import OUT_DIR, CONFIGS_DIR
from experiments.scrapl_loss import SCRAPLLoss

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class DDSP808LightingModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        synth: nn.Module,
        fe: FeatureCollection,
        loss_func: nn.Module,
        use_p_loss: bool = False,
        feature_type: str = "cqt",
        cqt_eps: float = 1e-3,
        run_name: Optional[str] = None,
        grad_mult: float = 1.0,
        scrapl_probs_path: Optional[str] = None,
        use_warmup: bool = False,
        warmup_n_batches: int = 1,
        warmup_n_iter: int = 20,
        warmup_param_agg: Literal["none", "mean", "max", "med"] = "none",
    ):
        super().__init__()
        self.model = model
        self.synth = synth
        self.fe = fe
        self.loss_func = loss_func
        self.use_p_loss = use_p_loss
        self.feature_type = feature_type
        self.cqt_eps = cqt_eps
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
        self.scrapl_probs_path = scrapl_probs_path
        self.use_warmup = use_warmup
        self.warmup_n_batches = warmup_n_batches
        self.warmup_n_iter = warmup_n_iter
        self.warmup_param_agg = warmup_param_agg

        # ckpt_path = os.path.join(OUT_DIR, f"ploss_724k_adamw_1e-5__theta14_10k_b16__epoch_42_step_8041.ckpt")
        # ckpt_path = None
        # if ckpt_path is not None and os.path.exists(ckpt_path):
        #     log.info(f"Loading pretrained model from {ckpt_path}")
        #     state_dict = tr.load(ckpt_path, map_location="cpu")["state_dict"]
        #     model_state_dict = {
        #         k.replace("model._orig_mod.", ""): v
        #         for k, v in state_dict.items()
        #         if k.startswith("model._orig_mod.")
        #     }
        #     msg = self.model.load_state_dict(model_state_dict, strict=True)
        #     log.info(f"Loaded model with msg: {msg}")
        # else:
        #     log.info(f"No pretrained model found at {ckpt_path}, training from scratch")

        self.n_params = synth.n_params
        fe_names = []
        for feature in fe.features:
            assert isinstance(feature, CascadingFrameExtactor)
            names = feature.flattened_features
            names = ["_".join(n) for n in names]
            fe_names.extend(names)
        self.fe_names = fe_names
        self.n_features = len(fe_names)

        if hasattr(self.loss_func, "set_resampler"):
            self.loss_func.set_resampler(self.synth.sr)
        if hasattr(self.loss_func, "in_sr"):
            assert self.loss_func.in_sr == self.synth.sr

        cqt_params = {
            "sr": synth.sr,
            "bins_per_octave": synth.Q_cqt,
            "n_bins": synth.J_cqt * synth.Q_cqt,
            "hop_length": synth.hop_len,
            # TODO(cm): check this
            "fmin": (0.4 * synth.sr) / (2**synth.J_cqt),
            "output_format": "Magnitude",
            "verbose": False,
        }
        self.cqt = CQT(**cqt_params)
        self.loss_name = self.loss_func.__class__.__name__
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.global_n = 0

        self.audio_dists = nn.ModuleDict()
        win_len = 2048
        hop_len = 512
        self.audio_dists["mss_meso_log"] = util.load_class_from_yaml(
            os.path.join(CONFIGS_DIR, "losses/mss_meso_log.yml")
        )
        # self.audio_dists["scat_1d_o2"] = util.load_class_from_yaml(
        #     os.path.join(CONFIGS_DIR, "eval_808/scat_1d.yml")
        # )
        self.audio_dists["mel_stft"] = auraloss.freq.MelSTFTLoss(
            sample_rate=synth.sr,
            fft_size=win_len,
            hop_size=hop_len,
            win_length=win_len,
            n_mels=128,
        )
        self.audio_dists["mfcc"] = MFCCDistance(
            sr=synth.sr,
            log_mels=True,
            n_fft=win_len,
            hop_len=hop_len,
            n_mels=128,
        )

        if grad_mult != 1.0:
            assert not isinstance(self.loss_func, SCRAPLLoss)
            log.info(f"Adding grad multiplier hook of {self.grad_mult}")
            for p in self.model.parameters():
                p.register_hook(self.grad_multiplier_hook)

        for p in self.synth.parameters():
            p.requires_grad = False

        if not use_warmup and isinstance(self.loss_func, SCRAPLLoss):
            params = list(self.model.parameters())
            self.loss_func.attach_params(params)
            if scrapl_probs_path is not None:
                self.loss_func.load_probs(scrapl_probs_path)

        # # TSV logging
        # tsv_cols = [
        #     "seed",
        #     "stage",
        #     "step",
        #     "global_n",
        #     "time_epoch",
        #     "loss",
        #     # "l1_theta",
        #     # "l2_theta",
        #     # "rmse_theta",
        # ]
        # # for idx in range(self.n_params):
        # #     param_name = synth.param_names[idx]
        # #     tsv_cols.append(f"l1_{param_name}")
        # #     tsv_cols.append(f"l2_{param_name}")
        # #     tsv_cols.append(f"rmse_{param_name}")
        # tsv_cols.extend(
        #     [
        #         "l1_fe",
        #         "l2_fe",
        #         "rmse_fe",
        #     ]
        # )
        # for fe_name in fe_names:
        #     tsv_cols.append(f"l1_{fe_name}")
        #     tsv_cols.append(f"l2_{fe_name}")
        #     tsv_cols.append(f"rmse_{fe_name}")
        # if run_name and not use_warmup:
        #     self.tsv_path = os.path.join(OUT_DIR, f"{self.run_name}.tsv")
        #     if not os.path.exists(self.tsv_path):
        #         with open(self.tsv_path, "w") as f:
        #             f.write("\t".join(tsv_cols) + "\n")
        # else:
        #     self.tsv_path = None

        # Compile
        if tr.cuda.is_available() and not use_warmup:
            self.model = tr.compile(self.model)

        self.samples_dir = os.path.join(OUT_DIR, "eval_808_samples")
        self.max_n_samples = 16
        os.makedirs(self.samples_dir, exist_ok=True)

        self.val_batches = []
        self.test_batches = []

    def on_fit_start(self) -> None:
        if self.use_warmup:
            self.warmup()

    def on_train_start(self) -> None:
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

        self.global_n = 0

    def state_dict(self, *args, **kwargs) -> Dict[str, T]:
        # TODO(cm): support resuming training with grad hooks
        state_dict = super().state_dict(*args, **kwargs)
        excluded_keys = [
            k for k in state_dict if k.startswith("synth") or k.startswith("cqt")
        ]
        for k in excluded_keys:
            del state_dict[k]
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, T], *args, **kwargs) -> None:
        kwargs["strict"] = False
        super().load_state_dict(state_dict, *args, **kwargs)

    def grad_multiplier_hook(self, grad: T) -> T:
        if not self.training:
            log.warning("grad_multiplier_hook called during eval")
            return grad
        grad *= self.grad_mult
        return grad

    def calc_U(self, x: T) -> T:
        if self.feature_type == "cqt":
            return SCRAPLLightingModule.calc_cqt(x, self.cqt, self.cqt_eps)
        else:
            raise NotImplementedError

    def warmup(self) -> None:
        # Make model and synth as deterministic as possible
        self.model.eval()
        self.synth.eval()

        def theta_fn(x: T) -> T:
            with tr.no_grad():
                U = self.calc_U(x)
            theta_0to1_hat = self.model(U)
            theta_0to1_hat = tr.stack(theta_0to1_hat, dim=1)
            return theta_0to1_hat

        def synth_fn(theta_0to1_hat: T) -> T:
            x_hat = self.synth(theta_0to1_hat)
            return x_hat

        assert isinstance(self.loss_func, SCRAPLLoss)
        train_dl_iter = iter(self.trainer.datamodule.train_dataloader())
        theta_fn_kwargs = []
        for batch_idx in range(self.warmup_n_batches):
            # theta = next(train_dl_iter)
            # TODO(cm): why is this needed here and not in the other lightning.py?
            # theta = theta.to(self.device)
            # with tr.no_grad():
            #     x = self.synth(theta)
            x = next(train_dl_iter)
            x = x.to(self.device)
            t_kwargs = {"x": x}
            theta_fn_kwargs.append(t_kwargs)

        params = list(self.model.parameters())

        suffix = (
            f"n_theta_{self.loss_func.n_theta}"
            f"__n_params_{len(params)}"
            f"__n_batches_{self.warmup_n_batches}"
            f"__n_iter_{self.warmup_n_iter}"
            f"__min_prob_frac_{self.loss_func.min_prob_frac}"
            f"__param_agg_{self.warmup_param_agg}"
            f"__seed_{tr.random.initial_seed()}.pt"
        )
        log.info(f"Running warmup with suffix: {suffix}")

        self.loss_func.warmup_lc_hvp(
            theta_fn=theta_fn,
            synth_fn=synth_fn,
            theta_fn_kwargs=theta_fn_kwargs,
            params=params,
            n_iter=self.warmup_n_iter,
            agg=self.warmup_param_agg,
        )

        log_vals_save_path = os.path.join(
            OUT_DIR, f"{self.run_name}__log_vals__{suffix}"
        )
        tr.save(self.loss_func.all_log_vals, log_vals_save_path)
        log_probs_save_path = os.path.join(
            OUT_DIR, f"{self.run_name}__log_probs__{suffix}"
        )
        tr.save(self.loss_func.all_log_probs, log_probs_save_path)
        probs_save_path = os.path.join(OUT_DIR, f"{self.run_name}__probs__{suffix}")
        tr.save(self.loss_func.probs, probs_save_path)
        log.info(f"Completed warmup, saved probs to {probs_save_path}")
        exit()

    def step(self, batch: Tuple[T, List[str]], stage: str) -> Dict[str, T]:
        # theta_0to1 = batch
        # batch_size = theta_0to1.size(0)
        x, drum_types = batch
        batch_size = x.size(0)
        if stage == "train":
            self.global_n = (
                self.global_step * self.trainer.accumulate_grad_batches * batch_size
            )
        self.log(f"global_n", float(self.global_n))

        with tr.no_grad():
            # x = self.synth(theta_0to1)
            U = self.calc_U(x)

        U_hat = None

        theta_0to1_hat = self.model(U)
        theta_0to1_hat = tr.stack(theta_0to1_hat, dim=1)

        # # Loss
        # if self.use_p_loss:
        #     loss = self.loss_func(theta_0to1_hat, theta_0to1)
        #     with tr.no_grad():
        #         x_hat = self.synth(theta_0to1_hat)
        #         U_hat = self.calc_U(x_hat)
        # else:
        x_hat = self.synth(theta_0to1_hat)
        with tr.no_grad():
            U_hat = self.calc_U(x_hat)
        loss = self.loss_func(x_hat, x)

        # Theta metrics
        # l1_theta = self.l1(theta_0to1_hat, theta_0to1)
        # l2_theta = self.mse(theta_0to1_hat, theta_0to1)
        # rmse_theta = l2_theta.sqrt()
        # self.log(f"{stage}/l1_theta", l1_theta, prog_bar=True)
        # self.log(f"{stage}/l2_theta", l2_theta, prog_bar=False)
        # self.log(f"{stage}/rmse_theta", rmse_theta, prog_bar=False)
        # l1_theta_vals = []
        # l2_theta_vals = []
        # rmse_theta_vals = []
        # for idx in range(self.n_params):
        #     l1 = self.l1(theta_0to1_hat[:, idx], theta_0to1[:, idx])
        #     l2 = self.mse(theta_0to1_hat[:, idx], theta_0to1[:, idx])
        #     rmse = l2.sqrt()
        #     l1_theta_vals.append(l1)
        #     l2_theta_vals.append(l2)
        #     rmse_theta_vals.append(rmse)
        #     param_name = self.synth.param_names[idx]
        #     self.log(f"{stage}/l1_theta_{param_name}", l1, prog_bar=False)
        #     self.log(f"{stage}/l2_theta_{param_name}", l2, prog_bar=False)
        #     self.log(f"{stage}/rmse_theta_{param_name}", rmse, prog_bar=False)

        self.log(f"{stage}/loss", loss, prog_bar=True)

        # TSV logging
        # if stage != "train" and self.tsv_path:
        #     seed_everything = tr.random.initial_seed()
        #     time_epoch = time.time()
        #     row_elems = [
        #         f"{seed_everything}",
        #         stage,
        #         f"{self.global_step}",
        #         f"{self.global_n}",
        #         f"{time_epoch}",
        #         f"{loss.item()}",
        #         # f"{l1_theta.item()}",
        #         # f"{l2_theta.item()}",
        #         # f"{rmse_theta.item()}",
        #     ]
        #     # for idx in range(self.n_params):
        #     #     row_elems.append(f"{l1_theta_vals[idx].item()}")
        #     #     row_elems.append(f"{l2_theta_vals[idx].item()}")
        #     #     row_elems.append(f"{rmse_theta_vals[idx].item()}")
        #     row_elems.extend(
        #         [
        #             f"{l1_feat.item()}",
        #             f"{l2_feat.item()}",
        #             f"{rmse_feat.item()}",
        #         ]
        #     )
        #     for idx in range(self.n_features):
        #         row_elems.append(f"{l1_fe_vals[idx].item()}")
        #         row_elems.append(f"{l2_fe_vals[idx].item()}")
        #         row_elems.append(f"{rmse_fe_vals[idx].item()}")
        #     row = "\t".join(row_elems) + "\n"
        #     with open(self.tsv_path, "a") as f:
        #         f.write(row)

        if stage == "val":
            self.val_batches.append((drum_types, x, x_hat, U, U_hat))
        if stage == "test":
            self.test_batches.append((drum_types, x, x_hat, U, U_hat))

        out_dict = {
            "loss": loss,
            # "U": U,
            # "U_hat": U_hat,
            # "x": x,
            # "x_hat": x_hat,
            # "theta_0to1": theta_0to1,
            # "theta_0to1_hat": theta_0to1_hat,
        }
        return out_dict

    def training_step(self, batch: T, batch_idx: int) -> Dict[str, T]:
        return self.step(batch, stage="train")

    def validation_step(self, batch: T, stage: str) -> Dict[str, T]:
        return self.step(batch, stage="val")

    def test_step(self, batch: T, stage: str) -> Dict[str, T]:
        return self.step(batch, stage="test")

    def log_fe_metrics(self, x: T, x_hat: T, stage: str, prefix: str = "") -> None:
        bs = x.size(0)
        feat = self.fe(x.squeeze(1))
        feat_hat = self.fe(x_hat.squeeze(1))
        assert feat_hat.shape == (bs, self.n_features)
        l1_fe_vals = []
        l2_fe_vals = []
        rmse_fe_vals = []
        for idx in range(self.n_features):
            l1 = self.l1(feat_hat[:, idx], feat[:, idx])
            l2 = self.mse(feat_hat[:, idx], feat[:, idx])
            rmse = l2.sqrt()
            l1_fe_vals.append(l1)
            l2_fe_vals.append(l2)
            rmse_fe_vals.append(rmse)
            feat_name = self.fe_names[idx]
            self.log(f"{stage}/{prefix}l1_fe_{feat_name}", l1, prog_bar=False)
            self.log(f"{stage}/{prefix}l2_fe_{feat_name}", l2, prog_bar=False)
            self.log(f"{stage}/{prefix}rmse_fe_{feat_name}", rmse, prog_bar=False)

    def log_audio_metrics(
        self, x: T, x_hat: T, U: T, U_hat: T, stage: str, prefix: str = ""
    ) -> None:
        for name, dist in self.audio_dists.items():
            with tr.no_grad():
                dist_val = dist(x_hat, x)
            self.log(f"{stage}/{prefix}audio_{name}", dist_val, prog_bar=False)
            with tr.no_grad():
                l1_U = self.l1(U_hat, U)
                l2_U = self.mse(U_hat, U)
                rmse_U = l2_U.sqrt()
            self.log(f"{stage}/{prefix}audio_U_l1", l1_U, prog_bar=False)
            self.log(f"{stage}/{prefix}audio_U_l2", l2_U, prog_bar=False)
            self.log(f"{stage}/{prefix}audio_U_rmse", rmse_U, prog_bar=False)

    def log_results(self, batches: List[Tuple[Any]], stage: str) -> None:
        drum_types, x, x_hat, U, U_hat = zip(*batches)
        drum_types = [dt for sublist in drum_types for dt in sublist]
        x = tr.cat(x, dim=0)
        x_hat = tr.cat(x_hat, dim=0)
        U = tr.cat(U, dim=0)
        U_hat = tr.cat(U_hat, dim=0)

        self.log_fe_metrics(x, x_hat, stage)
        self.log_audio_metrics(x, x_hat, U, U_hat, stage)
        unique_drum_types = sorted(list(set(drum_types)))
        for drum_type in unique_drum_types:
            idxs = [i for i, dt in enumerate(drum_types) if dt == drum_type]
            assert len(idxs) > 0
            curr_x = x[idxs, :, :]
            curr_x_hat = x_hat[idxs, :, :]
            curr_U = U[idxs, :, :]
            curr_U_hat = U_hat[idxs, :, :]
            self.log_fe_metrics(curr_x, curr_x_hat, stage, prefix=f"{drum_type}__")
            self.log_audio_metrics(
                curr_x, curr_x_hat, curr_U, curr_U_hat, stage, prefix=f"{drum_type}__"
            )

        for idx in range(x.size(0)):
            if idx >= self.max_n_samples:
                break
            curr_x = x[idx, :, :].detach().cpu()
            curr_x_hat = x_hat[idx, :, :].detach().cpu()
            save_path = os.path.join(self.samples_dir, f"{self.run_name}__{stage}__{idx}.wav")
            torchaudio.save(save_path, curr_x, sample_rate=self.synth.sr)
            save_path = os.path.join(self.samples_dir, f"{self.run_name}__{stage}__{idx}__hat.wav")
            torchaudio.save(save_path, curr_x_hat, sample_rate=self.synth.sr)

    def on_validation_epoch_end(self) -> None:
        assert self.val_batches
        self.log_results(self.val_batches, stage="val")
        self.val_batches = []

    def on_test_epoch_end(self) -> None:
        assert self.test_batches
        self.log_results(self.test_batches, stage="test")
        self.test_batches = []
