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
from experiments.plotting import fig2img, plot_waveforms_stacked, plot_scalogram

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
                                outputs: (T, Dict[str, T]),
                                batch: (T, T, T, T),
                                batch_idx: int,
                                dataloader_idx: int = 0) -> None:
        if outputs is None:
            return
        _, data_dict = outputs
        U = data_dict.get("U")
        U_hat = data_dict.get("U_hat")
        if U is None and U_hat is None:
            log.warning(f"U and U_hat are both None, cannot log spectrograms")
            return

        theta_density = data_dict["theta_density"]
        theta_slope = data_dict["theta_slope"]
        theta_density_hat = data_dict["theta_density_hat"]
        theta_slope_hat = data_dict["theta_slope_hat"]

        n_batches = theta_density.size(0)
        if batch_idx == 0:
            self.images = []
            for idx in range(self.n_examples):
                if idx < n_batches:
                    title = (f"idx_{idx}, "
                             f"θd: {theta_density[idx]:.2f}, "
                             f"θd_hat: {theta_density_hat[idx]:.2f}, "
                             f"θs: {theta_slope[idx]:.2f}, "
                             f"θs_hat: {theta_slope_hat[idx]:.2f}")

                    fig, ax = plt.subplots(nrows=2,
                                           figsize=(6, 10),
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
                                       title="U_hat",
                                       hop_len=hop_len,
                                       vmax=vmax)
                    plot_scalogram(ax[0],
                                   curr_U,
                                   sr,
                                   y_coords,
                                   title="U",
                                   hop_len=hop_len,
                                   vmax=vmax)

                    fig.tight_layout()
                    img = fig2img(fig)
                    self.images.append(img)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
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
                                outputs: (T, Dict[str, T]),
                                batch: (T, T, T, T),
                                batch_idx: int,
                                dataloader_idx: int = 0) -> None:
        if outputs is None:
            return
        _, data_dict = outputs
        x = data_dict.get("x")
        x_hat = data_dict.get("x_hat")
        if x is None and x_hat is None:
            log.debug(f"x and x_hat are both None, cannot log audio")
            return

        theta_density = data_dict["theta_density"]
        theta_slope = data_dict["theta_slope"]
        theta_density_hat = data_dict["theta_density_hat"]
        theta_slope_hat = data_dict["theta_slope_hat"]

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

                    title = (f"idx_{idx}, "
                             f"θd: {theta_density[idx]:.2f}, "
                             f"θd_hat: {theta_density_hat[idx]:.2f}, "
                             f"θs: {theta_slope[idx]:.2f}, "
                             f"θs_hat: {theta_slope_hat[idx]:.2f}")

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
                columns = []
                for idx, curr_x_audio in enumerate(self.x_audio):
                    columns.append(f"idx_{idx}")  # TODO(cm)
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
