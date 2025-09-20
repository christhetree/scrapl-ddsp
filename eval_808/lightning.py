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
from tqdm import tqdm

from eval_808.features import FeatureCollection, CascadingFrameExtactor
from experiments import util
from experiments.lightning import SCRAPLLightingModule
from experiments.losses import JTFSTLoss, Scat1DLoss, MFCCDistance
from experiments.paths import OUT_DIR, CONFIGS_DIR, AUDIO_SAVE_DIR, TSV_SAVE_DIR
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
        # if type(self.loss_func) in {
        #     JTFSTLoss,
        #     Scat1DLoss,
        # }:
        #     assert self.grad_mult != 1.0
        # else:
        #     assert self.grad_mult == 1.0
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
        self.jtfs = util.load_class_from_yaml(
            os.path.join(CONFIGS_DIR, "eval_808/jtfst.yml")
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

        # TSV logging
        self.drum_types = ["BD", "SD", "Tom", "HH"]
        tsv_cols = [
            "seed",
            "stage",
            "step",
            "global_n",
            "time_epoch",
        ]
        for fe_name in fe_names:
            tsv_cols.append(f"fe__{fe_name}__l1")
            tsv_cols.append(f"fe__{fe_name}__mse")
            tsv_cols.append(f"fe__{fe_name}__rmse")
        for audio_name in self.audio_dists.keys():
            tsv_cols.append(f"audio__{audio_name}")
        tsv_cols.append("audio__U__l1")
        tsv_cols.append("audio__U__mse")
        tsv_cols.append("audio__U__rmse")
        for drum_type in self.drum_types:
            for fe_name in fe_names:
                tsv_cols.append(f"{drum_type}__fe__{fe_name}__l1")
                tsv_cols.append(f"{drum_type}__fe__{fe_name}__mse")
                tsv_cols.append(f"{drum_type}__fe__{fe_name}__rmse")
            for audio_name in self.audio_dists.keys():
                tsv_cols.append(f"{drum_type}__audio__{audio_name}")
            tsv_cols.append(f"{drum_type}__audio__U__l1")
            tsv_cols.append(f"{drum_type}__audio__U__mse")
            tsv_cols.append(f"{drum_type}__audio__U__rmse")
        tsv_cols.append("audio__jtfs")
        for drum_type in self.drum_types:
            tsv_cols.append(f"{drum_type}__audio__jtfs")
        self.tsv_cols = tsv_cols

        if run_name and not use_warmup:
            self.tsv_path = os.path.join(TSV_SAVE_DIR, f"{self.run_name}.tsv")
            if not os.path.exists(self.tsv_path):
                with open(self.tsv_path, "w") as f:
                    f.write("\t".join(tsv_cols) + "\n")
        else:
            self.tsv_path = None

        # Compile
        if tr.cuda.is_available() and not use_warmup:
            self.model = tr.compile(self.model)

        self.samples_dir = AUDIO_SAVE_DIR
        self.max_n_samples = 128
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
        synth_fn_kwargs = []
        for batch_idx in range(self.warmup_n_batches):
            # theta = next(train_dl_iter)
            # TODO(cm): why is this needed here and not in the other lightning.py?
            # theta = theta.to(self.device)
            # with tr.no_grad():
            #     x = self.synth(theta)
            x, drum_types, delta = next(train_dl_iter)
            x = x.to(self.device)
            delta = delta.to(self.device)
            t_kwargs = {"x": x}
            theta_fn_kwargs.append(t_kwargs)
            s_kwargs = {"delta": delta}
            synth_fn_kwargs.append(s_kwargs)

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

    def step(self, batch: Tuple[T, List[str], T], stage: str) -> Dict[str, T]:
        # theta_0to1 = batch
        # batch_size = theta_0to1.size(0)
        x, drum_types, delta = batch
        # Add padding introduced by synth for delta_min < 0
        if self.synth.delta_min < 0:
            x = tr.nn.functional.pad(
                x, (-self.synth.delta_min, 0), mode="constant", value=0.0
            )
            x = x[:, :, : self.synth.num_samples]

        batch_size = x.size(0)
        if stage == "train":
            self.global_n = (
                self.global_step * self.trainer.accumulate_grad_batches * batch_size
            )
        # self.log(f"global_n", float(self.global_n))

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

        if stage != "train":
            delta = None
        x_hat = self.synth(params=theta_0to1_hat, delta=delta)
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

    def log_fe_metrics(
        self, x: T, x_hat: T, stage: str, prefix: str = ""
    ) -> List[float]:
        tsv_vals = []
        # Remove padding introduced by synth for delta_min < 0
        n_padding = -self.synth.delta_min
        if n_padding > 0:
            x = x[:, :, n_padding:]
            x_hat = x_hat[:, :, n_padding:]
        with tr.no_grad():
            feat = self.fe(x.squeeze(1))
            feat_hat = self.fe(x_hat.squeeze(1))
        assert len(feat) == self.n_features
        for idx in range(self.n_features):
            feat_name = self.fe_names[idx]
            curr_feat = feat[idx]
            curr_feat_hat = feat_hat[idx]
            assert curr_feat.size() == curr_feat_hat.size()
            # TODO(cm): this is nasty
            # Scale by loudness for spectral features
            if "SpectralCentroid" in feat_name or "SpectralFlatness" in feat_name:
                loudness_min_val = self.fe.features[0].extractors[0].min_val
                loudness_name = feat_name.replace(
                    "SpectralCentroid", "Loudness"
                ).replace("SpectralFlatness", "Loudness")
                loudness_idx = self.fe_names.index(loudness_name)
                loudness = feat[loudness_idx] - loudness_min_val
                loudness_hat = feat_hat[loudness_idx] - loudness_min_val
                assert loudness.min() >= 0.0
                assert loudness_hat.min() >= 0.0
                assert curr_feat.size() == loudness.size()
                assert curr_feat_hat.size() == loudness_hat.size()
                curr_feat = curr_feat * loudness
                curr_feat_hat = curr_feat_hat * loudness_hat
            with tr.no_grad():
                l1 = self.l1(curr_feat_hat, curr_feat)
                l2 = self.mse(curr_feat_hat, curr_feat)
                rmse = l2.sqrt()
            # self.log(f"{stage}/{prefix}l1_fe_{feat_name}", l1, prog_bar=False)
            # self.log(f"{stage}/{prefix}l2_fe_{feat_name}", l2, prog_bar=False)
            # self.log(f"{stage}/{prefix}rmse_fe_{feat_name}", rmse, prog_bar=False)
            tsv_vals.extend([l1.cpu().item(), l2.cpu().item(), rmse.cpu().item()])
        return tsv_vals

    def log_audio_metrics(
        self, x: T, x_hat: T, U: T, U_hat: T, stage: str, prefix: str = ""
    ) -> List[float]:
        tsv_vals = []
        for name, dist in self.audio_dists.items():
            with tr.no_grad():
                dist_val = dist(x_hat, x)
            # self.log(f"{stage}/{prefix}audio_{name}", dist_val, prog_bar=False)
            tsv_vals.append(dist_val.cpu().item())
        with tr.no_grad():
            l1_U = self.l1(U_hat, U)
            l2_U = self.mse(U_hat, U)
            rmse_U = l2_U.sqrt()
        # self.log(f"{stage}/{prefix}audio_U_l1", l1_U, prog_bar=False)
        # self.log(f"{stage}/{prefix}audio_U_l2", l2_U, prog_bar=False)
        # self.log(f"{stage}/{prefix}audio_U_rmse", rmse_U, prog_bar=False)
        tsv_vals.extend([l1_U.cpu().item(), l2_U.cpu().item(), rmse_U.cpu().item()])
        return tsv_vals

    def log_results(self, batches: List[Tuple[Any]], stage: str) -> None:
        drum_types, x, x_hat, U, U_hat = zip(*batches)
        drum_types = [dt for sublist in drum_types for dt in sublist]
        x = tr.cat(x, dim=0)
        x_hat = tr.cat(x_hat, dim=0)
        U = tr.cat(U, dim=0)
        U_hat = tr.cat(U_hat, dim=0)
        tsv_vals = [
            f"{tr.random.initial_seed()}",
            stage,
            f"{self.global_step}",
            f"{self.global_n}",
            f"{time.time()}",
        ]

        tsv_vals.extend(self.log_fe_metrics(x, x_hat, stage))
        tsv_vals.extend(self.log_audio_metrics(x, x_hat, U, U_hat, stage))
        for drum_type in self.drum_types:
            idxs = [i for i, dt in enumerate(drum_types) if dt == drum_type]
            assert len(idxs) > 0
            curr_x = x[idxs, :, :]
            curr_x_hat = x_hat[idxs, :, :]
            curr_U = U[idxs, :, :]
            curr_U_hat = U_hat[idxs, :, :]
            tsv_vals.extend(
                self.log_fe_metrics(curr_x, curr_x_hat, stage, prefix=f"{drum_type}__")
            )
            tsv_vals.extend(
                self.log_audio_metrics(
                    curr_x,
                    curr_x_hat,
                    curr_U,
                    curr_U_hat,
                    stage,
                    prefix=f"{drum_type}__",
                )
            )

        if stage == "test":
            jtfs_vals = []
            for idx in tqdm(range(x.size(0))):
                curr_x = x[idx, :, :].unsqueeze(0)
                curr_x_hat = x_hat[idx, :, :].unsqueeze(0)
                with tr.no_grad():
                    jtfs_dist = self.jtfs(curr_x_hat, curr_x)
                jtfs_vals.append(jtfs_dist)
            jtfs_dists = tr.stack(jtfs_vals, dim=0)
            jtfs_dist = jtfs_dists.mean()
            # self.log(f"{stage}/audio_jtfs", jtfs_dist, prog_bar=False)
            tsv_vals.append(jtfs_dist.cpu().item())
            for drum_type in self.drum_types:
                idxs = [i for i, dt in enumerate(drum_types) if dt == drum_type]
                assert len(idxs) > 0
                dt_jtfs_dists = jtfs_dists[idxs]
                dt_jtfs_dist = dt_jtfs_dists.mean()
                # self.log(
                #     f"{stage}/{drum_type}__audio_jtfs", dt_jtfs_dist, prog_bar=False
                # )
                tsv_vals.append(dt_jtfs_dist.cpu().item())
        else:
            tsv_vals.append(None)
            for _ in self.drum_types:
                tsv_vals.append(None)

        assert len(tsv_vals) == len(self.tsv_cols), f"{len(tsv_vals)} vs {len(self.tsv_cols)}"
        if self.tsv_path:
            tsv_vals = [str(v) for v in tsv_vals]
            row = "\t".join(tsv_vals) + "\n"
            with open(self.tsv_path, "a") as f:
                f.write(row)

        if stage == "test":
            seed = tr.random.initial_seed()
            for idx in range(x.size(0)):
                if idx >= self.max_n_samples:
                    break
                curr_x = x[idx, :, :].detach().cpu()
                curr_x_hat = x_hat[idx, :, :].detach().cpu()
                save_path = os.path.join(
                    self.samples_dir, f"{self.run_name}__seed_{seed}__{stage}__{idx}.wav"
                )
                torchaudio.save(save_path, curr_x, sample_rate=self.synth.sr)
                save_path = os.path.join(
                    self.samples_dir,
                    f"{self.run_name}__seed_{seed}__{stage}__{idx}__hat.wav",
                )
                torchaudio.save(save_path, curr_x_hat, sample_rate=self.synth.sr)

    def on_validation_epoch_end(self) -> None:
        assert self.val_batches
        self.log_results(self.val_batches, stage="val")
        self.val_batches = []

    def on_test_epoch_end(self) -> None:
        assert self.test_batches
        self.log_results(self.test_batches, stage="test")
        self.test_batches = []
