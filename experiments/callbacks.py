import logging
import os
from collections import defaultdict
from typing import Any, Dict

import wandb
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer, Callback, LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor as T

from experiments.lightning import SCRAPLLightingModule
from experiments.plotting import fig2img, plot_waveforms_stacked, plot_scalogram, \
    plot_xy_points_and_grads

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ConsoleLRMonitor(LearningRateMonitor):
    # TODO(cm): enable every n steps
    def on_train_epoch_start(self,
                             trainer: Trainer,
                             *args: Any,
                             **kwargs: Any) -> None:
        super().on_train_epoch_start(trainer, *args, **kwargs)
        if self.logging_interval != "step":
            interval = "epoch" if self.logging_interval is None else "any"
            latest_stat = self._extract_stats(trainer, interval)
            latest_stat_str = {k: f"{v:.8f}" for k, v in latest_stat.items()}
            if latest_stat:
                log.info(f"\nCurrent LR: {latest_stat_str}")


class LogScalogramCallback(Callback):
    def __init__(self, n_examples: int = 5) -> None:
        super().__init__()
        self.n_examples = n_examples
        self.images = []

    def on_validation_batch_end(self,
                                trainer: Trainer,
                                pl_module: LightningModule,
                                out_dict: Dict[str, T],
                                batch: (T, T, T, T),
                                batch_idx: int,
                                dataloader_idx: int = 0) -> None:
        out_dict = {k: v.detach().cpu() for k, v in out_dict.items() if v is not None}

        U = out_dict.get("U")
        U_hat = out_dict.get("U_hat")
        if U is None and U_hat is None:
            log.warning(f"U and U_hat are both None, cannot log spectrograms")
            return

        theta_density = out_dict["theta_density"]
        theta_slope = out_dict["theta_slope"]
        theta_density_hat = out_dict["theta_density_hat"]
        theta_slope_hat = out_dict["theta_slope_hat"]
        seed = out_dict["seed"]
        seed_hat = out_dict["seed_hat"]

        n_batches = theta_density.size(0)
        if batch_idx == 0:
            self.images = []
            for idx in range(self.n_examples):
                if idx < n_batches:
                    title = (f"batch_idx_{idx}, "
                             f"θd: {theta_density[idx]:.2f} -> "
                             f"{theta_density_hat[idx]:.2f}, "
                             f"θs: {theta_slope[idx]:.2f} -> "
                             f"{theta_slope_hat[idx]:.2f}")

                    fig, ax = plt.subplots(nrows=2,
                                           figsize=(6, 12),
                                           sharex="all",
                                           squeeze=True)
                    fig.suptitle(title, fontsize=14)
                    curr_U = U[idx]
                    y_coords = pl_module.cqt.frequencies
                    hop_len = pl_module.cqt.hop_length
                    sr = pl_module.synth.sr
                    vmax = None
                    if U_hat is not None:
                        curr_U_hat = U_hat[idx]
                        vmax = max(curr_U.max(), curr_U_hat.max())
                        plot_scalogram(ax[1],
                                       curr_U_hat,
                                       sr,
                                       y_coords,
                                       title=f"U_hat, seed: {int(seed_hat[idx])}",
                                       hop_len=hop_len,
                                       vmax=vmax)
                    plot_scalogram(ax[0],
                                   curr_U,
                                   sr,
                                   y_coords,
                                   title=f"U, seed: {int(seed[idx])}",
                                   hop_len=hop_len,
                                   vmax=vmax)

                    fig.tight_layout()
                    img = fig2img(fig)
                    self.images.append(img)

    def on_validation_epoch_end(self,
                                trainer: Trainer,
                                pl_module: LightningModule) -> None:
        if self.images:
            for logger in trainer.loggers:
                # TODO(cm): enable for tensorboard as well
                if isinstance(logger, WandbLogger):
                    logger.log_image(key="spectrograms",
                                     images=self.images,
                                     step=trainer.global_step)


class LogAudioCallback(Callback):
    def __init__(self, n_examples: int = 5) -> None:
        super().__init__()
        self.n_examples = n_examples
        self.x_audio = []
        self.x_hat_audio = []
        self.images = []

    def on_validation_batch_end(self,
                                trainer: Trainer,
                                pl_module: SCRAPLLightingModule,
                                out_dict: Dict[str, T],
                                batch: (T, T, T, T),
                                batch_idx: int,
                                dataloader_idx: int = 0) -> None:
        out_dict = {k: v.detach().cpu() for k, v in out_dict.items() if v is not None}

        x = out_dict.get("x")
        x_hat = out_dict.get("x_hat")
        if x is None and x_hat is None:
            log.debug(f"x and x_hat are both None, cannot log audio")
            return

        theta_density = out_dict["theta_density"]
        theta_slope = out_dict["theta_slope"]
        theta_density_hat = out_dict["theta_density_hat"]
        theta_slope_hat = out_dict["theta_slope_hat"]

        n_batches = theta_density.size(0)
        if batch_idx == 0:
            self.images = []
            self.x_audio = []
            self.x_hat_audio = []
            for idx in range(self.n_examples):
                if idx < n_batches:
                    waveforms = []
                    labels = []
                    if x is not None:
                        curr_x = x[idx]
                        waveforms.append(curr_x)
                        labels.append("x")
                        self.x_audio.append(curr_x.swapaxes(0, 1).numpy())
                    if x_hat is not None:
                        curr_x_hat = x_hat[idx]
                        waveforms.append(curr_x_hat)
                        labels.append("x_hat")
                        self.x_hat_audio.append(curr_x_hat.swapaxes(0, 1).numpy())

                    title = (f"batch_idx_{idx}, "
                             f"θd: {theta_density[idx]:.2f} -> "
                             f"{theta_density_hat[idx]:.2f}, "
                             f"θs: {theta_slope[idx]:.2f} -> "
                             f"{theta_slope_hat[idx]:.2f}")

                    fig = plot_waveforms_stacked(waveforms,
                                                 pl_module.synth.sr,
                                                 title,
                                                 labels)
                    img = fig2img(fig)
                    self.images.append(img)

    def on_validation_epoch_end(self,
                                trainer: Trainer,
                                pl_module: SCRAPLLightingModule) -> None:
        for logger in trainer.loggers:
            # TODO(cm): enable for tensorboard as well
            if isinstance(logger, WandbLogger):
                logger.log_image(key="waveforms",
                                 images=self.images,
                                 step=trainer.global_step)
                data = defaultdict(list)
                columns = [f"idx_{idx}" for idx in range(len(self.images))]
                for idx, curr_x_audio in enumerate(self.x_audio):
                    data["x_audio"].append(
                        wandb.Audio(curr_x_audio,
                                    caption=f"x_{idx}",
                                    sample_rate=int(pl_module.synth.sr))
                    )
                for idx, curr_x_hat_audio in enumerate(self.x_hat_audio):
                    data["x_hat_audio"].append(
                        wandb.Audio(curr_x_hat_audio,
                                    caption=f"x_hat_{idx}",
                                    sample_rate=int(pl_module.synth.sr))
                    )
                data = list(data.values())
                logger.log_table(key="audio",
                                 columns=columns,
                                 data=data,
                                 step=trainer.global_step)


class LogGradientCallback(Callback):
    out_dict_keys = [
        "theta_density", "theta_slope", "theta_density_hat", "theta_slope_hat"]

    def __init__(self, n_examples: int = 5, max_n_points: int = 16) -> None:
        super().__init__()
        self.n_examples = n_examples
        self.max_n_points = max_n_points
        self.images = []
        self.density_grads = {}
        self.slope_grads = {}
        self.train_out_dicts = {}

    def on_train_batch_end(self,
                           trainer: Trainer,
                           pl_module: SCRAPLLightingModule,
                           out_dict: Dict[str, T],
                           batch: (T, T, T, T),
                           batch_idx: int,
                           dataloader_idx: int = 0) -> None:
        density_grad = out_dict["theta_density_hat"].grad.detach().cpu()
        slope_grad = out_dict["theta_slope_hat"].grad.detach().cpu()

        self.density_grads[batch_idx] = density_grad
        self.slope_grads[batch_idx] = slope_grad
        if batch_idx < self.n_examples:
            out_dict = {k: out_dict[k] for k in self.out_dict_keys if k in out_dict}
            self.train_out_dicts[batch_idx] = out_dict

    def on_validation_batch_end(self,
                                trainer: Trainer,
                                pl_module: SCRAPLLightingModule,
                                val_out_dict: Dict[str, T],
                                batch: (T, T, T, T),
                                batch_idx: int,
                                dataloader_idx: int = 0) -> None:
        if not self.train_out_dicts:
            log.warning("train_out_dicts is empty, cannot log gradients")

        if batch_idx == 0:
            self.images = []

        if batch_idx < self.n_examples:
            fig, ax = plt.subplots(nrows=2, figsize=(4, 8), squeeze=True)
            title_suffix = "meso" if pl_module.use_rand_seed_hat else "micro"

            train_out_dict = self.train_out_dicts.get(batch_idx)
            if train_out_dict is not None:
                out_dict = {k: v.detach().cpu()[:self.max_n_points]
                            for k, v in train_out_dict.items() if v is not None}
                density_grad = self.density_grads[batch_idx][:self.max_n_points]
                slope_grad = self.slope_grads[batch_idx][:self.max_n_points]
                max_grad = max(density_grad.abs().max(), slope_grad.abs().max())
                density_grad /= max_grad
                slope_grad /= max_grad
                plot_xy_points_and_grads(ax[0],
                                         out_dict["theta_slope"],
                                         out_dict["theta_density"],
                                         out_dict["theta_slope_hat"],
                                         out_dict["theta_density_hat"],
                                         slope_grad,
                                         density_grad,
                                         title=f"train_{batch_idx}_{title_suffix}")

            if val_out_dict is not None:
                out_dict = {k: val_out_dict[k]
                            for k in self.out_dict_keys if k in val_out_dict}
                out_dict = {k: v.detach().cpu()[:self.max_n_points]
                            for k, v in out_dict.items() if v is not None}
                plot_xy_points_and_grads(ax[1],
                                         out_dict["theta_slope"],
                                         out_dict["theta_density"],
                                         out_dict["theta_slope_hat"],
                                         out_dict["theta_density_hat"],
                                         title=f"val_{batch_idx}_{title_suffix}")
            fig.tight_layout()
            img = fig2img(fig)
            self.images.append(img)

    def on_validation_epoch_end(self,
                                trainer: Trainer,
                                pl_module: LightningModule) -> None:
        if self.images:
            for logger in trainer.loggers:
                # TODO(cm): enable for tensorboard as well
                if isinstance(logger, WandbLogger):
                    logger.log_image(key="xy_points_and_grads",
                                     images=self.images,
                                     step=trainer.global_step)
