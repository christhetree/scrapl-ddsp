import logging
import os
from collections import defaultdict
from typing import Any, Dict

import torch as tr
import wandb
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer, Callback, LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor as T

from experiments.lightning import SCRAPLLightingModule
from experiments.plotting import (
    fig2img,
    plot_waveforms_stacked,
    plot_scalogram,
    plot_xy_points_and_grads,
)

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ConsoleLRMonitor(LearningRateMonitor):
    # TODO(cm): enable every n steps
    def on_train_epoch_start(self, trainer: Trainer, *args: Any, **kwargs: Any) -> None:
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
        self.out_dicts = {}

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        out_dict: Dict[str, T],
        batch: (T, T, T),
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        example_idx = batch_idx // trainer.accumulate_grad_batches
        if example_idx < self.n_examples:
            if example_idx not in self.out_dicts:
                out_dict = {
                    k: v.detach().cpu() for k, v in out_dict.items() if v is not None
                }
                self.out_dicts[example_idx] = out_dict

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        images = []
        for example_idx in range(self.n_examples):
            if example_idx not in self.out_dicts:
                log.warning(f"example_idx={example_idx} not in out_dicts")
                continue

            out_dict = self.out_dicts[example_idx]
            U = out_dict.get("U")
            U_hat = out_dict.get("U_hat")

            if U is None and U_hat is None:
                log.warning(f"U and U_hat are both None for example_idx={example_idx}")
                continue

            U = U[0]
            theta_d = out_dict["theta_d"][0]
            theta_s = out_dict["theta_s"][0]
            theta_d_hat = out_dict["theta_d_hat"][0]
            theta_s_hat = out_dict["theta_s_hat"][0]
            seed = out_dict["seed"][0]
            seed_hat = out_dict["seed_hat"][0]

            title = (
                f"batch_idx_{example_idx}, "
                f"θd: {theta_d:.2f} -> {theta_d_hat:.2f}, "
                f"θs: {theta_s:.2f} -> {theta_s_hat:.2f}"
            )

            fig, ax = plt.subplots(nrows=2, figsize=(6, 12), sharex="all", squeeze=True)
            fig.suptitle(title, fontsize=14)
            y_coords = pl_module.cqt.frequencies
            hop_len = pl_module.cqt.hop_length
            sr = pl_module.synth.sr
            vmax = None
            if U_hat is not None:
                U_hat = U_hat[0]
                vmax = max(U.max(), U_hat.max())
                plot_scalogram(
                    ax[1],
                    U_hat,
                    sr,
                    y_coords,
                    title=f"U_hat, seed: {int(seed_hat)}",
                    hop_len=hop_len,
                    vmax=vmax,
                )
            plot_scalogram(
                ax[0],
                U,
                sr,
                y_coords,
                title=f"U, seed: {int(seed)}",
                hop_len=hop_len,
                vmax=vmax,
            )

            fig.tight_layout()
            img = fig2img(fig)
            images.append(img)

        if images:
            for logger in trainer.loggers:
                # TODO(cm): enable for tensorboard as well
                if isinstance(logger, WandbLogger):
                    logger.log_image(
                        key="spectrograms", images=images, step=trainer.global_step
                    )

        self.out_dicts.clear()


class LogAudioCallback(Callback):
    def __init__(self, n_examples: int = 5) -> None:
        super().__init__()
        self.n_examples = n_examples
        self.out_dicts = {}
        self.columns = ["row_id"] + [f"idx_{idx}" for idx in range(n_examples)]
        self.rows = []

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: SCRAPLLightingModule,
        out_dict: Dict[str, T],
        batch: Dict[str, T],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        example_idx = batch_idx // trainer.accumulate_grad_batches
        if example_idx < self.n_examples:
            if example_idx not in self.out_dicts:
                out_dict = {
                    k: v.detach().cpu() for k, v in out_dict.items() if v is not None
                }
                self.out_dicts[example_idx] = out_dict

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: SCRAPLLightingModule
    ) -> None:
        images = []
        x_waveforms = []
        x_hat_waveforms = []
        suffixes = []
        sr = pl_module.synth.sr

        for example_idx in range(self.n_examples):
            if example_idx not in self.out_dicts:
                log.warning(f"example_idx={example_idx} not in out_dicts")
                continue

            out_dict = self.out_dicts[example_idx]
            x = out_dict.get("x")
            x_hat = out_dict.get("x_hat")
            if x is None and x_hat is None:
                log.debug(f"x and x_hat are both None, cannot log audio")
                return

            theta_d = out_dict["theta_d"]
            theta_s = out_dict["theta_s"]
            theta_d_hat = out_dict["theta_d_hat"]
            theta_s_hat = out_dict["theta_s_hat"]
            suffix = (
                f"θd: {theta_d[0]:.2f} -> "
                f"{theta_d_hat[0]:.2f}, "
                f"θs: {theta_s[0]:.2f} -> "
                f"{theta_s_hat[0]:.2f}"
            )
            suffixes.append(suffix)
            title = f"{trainer.global_step}_idx_{example_idx}, {suffix}"
            waveforms = []
            labels = []

            if x is not None:
                x = x[0]
                waveforms.append(x)
                labels.append(f"x")
                x_waveforms.append(x.swapaxes(0, 1).numpy())
            if x_hat is not None:
                x_hat = x_hat[0]
                waveforms.append(x_hat)
                labels.append(f"x_hat")
                x_hat_waveforms.append(x_hat.swapaxes(0, 1).numpy())

            fig = plot_waveforms_stacked(waveforms, sr, title, labels)
            img = fig2img(fig)
            images.append(img)

        if images and wandb.run:
            wandb.log(
                {
                    "waveforms": [wandb.Image(i) for i in images],
                    "global_step": trainer.global_step,
                }
            )

        data = defaultdict(list)
        if x_waveforms:
            suffix = "\n".join(suffixes)
            data["x"].append(f"{trainer.global_step}_x\n{suffix}")
        for idx, curr_x in enumerate(x_waveforms):
            data["x"].append(
                wandb.Audio(
                    curr_x,
                    caption=f"{trainer.global_step}_x_{idx}",
                    sample_rate=int(sr),
                )
            )
        if x_hat_waveforms:
            suffix = "\n".join(suffixes)
            data["x_hat"].append(f"{trainer.global_step}_x_hat\n{suffix}")
        for idx, curr_x_hat in enumerate(x_hat_waveforms):
            data["x_hat"].append(
                wandb.Audio(
                    curr_x_hat,
                    caption=f"{trainer.global_step}_x_hat_{idx}",
                    sample_rate=int(sr),
                )
            )
        data = list(data.values())
        for row in data:
            self.rows.append(row)
        if wandb.run:
            wandb.log(
                {
                    "audio": wandb.Table(columns=self.columns, data=self.rows),
                }
            )
            self.out_dicts.clear()


class LogGradientCallback(Callback):
    REQUIRED_OUT_DICT_KEYS = {
        "theta_d",
        "theta_s",
        "theta_d_hat",
        "theta_s_hat",
    }

    def __init__(self, n_examples: int = 5, max_n_points: int = 16) -> None:
        super().__init__()
        self.n_examples = n_examples
        self.max_n_points = max_n_points
        self.train_d_grads = defaultdict(list)
        self.train_s_grads = defaultdict(list)
        self.train_out_dicts = defaultdict(lambda: defaultdict(list))
        self.val_d_grads = defaultdict(list)
        self.val_s_grads = defaultdict(list)
        self.val_out_dicts = defaultdict(lambda: defaultdict(list))

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: SCRAPLLightingModule,
        out_dict: Dict[str, T],
        batch: (T, T, T),
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        example_idx = batch_idx // trainer.accumulate_grad_batches
        batch_size = batch[0].size(0)

        if example_idx < self.n_examples:
            d_grad = out_dict["theta_d_hat"].grad.detach().cpu()
            s_grad = out_dict["theta_s_hat"].grad.detach().cpu()
            self.train_d_grads[example_idx].append(d_grad)
            self.train_s_grads[example_idx].append(s_grad)

            train_out_dict = self.train_out_dicts[example_idx]
            for k, v in out_dict.items():
                if k in self.REQUIRED_OUT_DICT_KEYS and v is not None:
                    if len(train_out_dict[k]) * batch_size < self.max_n_points:
                        train_out_dict[k].append(v.detach().cpu())

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: SCRAPLLightingModule,
        out_dict: Dict[str, T],
        batch: (T, T, T),
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        example_idx = batch_idx // trainer.accumulate_grad_batches
        batch_size = batch[0].size(0)

        if example_idx < self.n_examples:
            if pl_module.log_val_grads:
                theta_d_hat = out_dict["theta_d_hat"]
                theta_s_hat = out_dict["theta_s_hat"]
                dist = out_dict["loss"].clone()
                d_grad, s_grad = tr.autograd.grad(dist, [theta_d_hat, theta_s_hat])
                d_grad = d_grad.detach().cpu()
                s_grad = s_grad.detach().cpu()
                d_grad /= trainer.accumulate_grad_batches
                s_grad /= trainer.accumulate_grad_batches
                self.val_d_grads[example_idx].append(d_grad)
                self.val_s_grads[example_idx].append(s_grad)

            val_out_dict = self.val_out_dicts[example_idx]
            for k, v in out_dict.items():
                if k in self.REQUIRED_OUT_DICT_KEYS and v is not None:
                    if len(val_out_dict[k]) * batch_size < self.max_n_points:
                        val_out_dict[k].append(v.detach().cpu())

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        images = []
        for example_idx in range(self.n_examples):
            fig, ax = plt.subplots(nrows=2, figsize=(4, 8), squeeze=True)
            title_suffix = "meso" if pl_module.use_rand_seed_hat else "micro"

            train_out_dict = self.train_out_dicts[example_idx]
            train_out_dict = {
                k: tr.cat(v, dim=0)[: self.max_n_points]
                for k, v in train_out_dict.items()
            }
            if train_out_dict:
                # TODO(cm): remove duplicate code
                d_grad = self.train_d_grads[example_idx]
                s_grad = self.train_s_grads[example_idx]
                d_grad = tr.cat(d_grad, dim=0)[: self.max_n_points]
                s_grad = tr.cat(s_grad, dim=0)[: self.max_n_points]
                max_d_grad = d_grad.abs().max()
                max_s_grad = s_grad.abs().max()
                avg_d_grad = d_grad.abs().mean()
                avg_s_grad = s_grad.abs().mean()
                max_grad = max(max_d_grad, max_s_grad)
                d_grad /= max_grad
                s_grad /= max_grad

                plot_xy_points_and_grads(
                    ax[0],
                    train_out_dict["theta_s"],
                    train_out_dict["theta_d"],
                    train_out_dict["theta_s_hat"],
                    train_out_dict["theta_d_hat"],
                    s_grad,
                    d_grad,
                    title=f"train_{example_idx}_{title_suffix}"
                    f"\nmax_d∇: {max_d_grad:.4f}"
                    f" max_s∇: {max_s_grad:.4f}"
                    f"\navg_d∇: {avg_d_grad:.4f}"
                    f" avg_s∇: {avg_s_grad:.4f}",
                )
            else:
                log.warning(f"train_out_dict for example_idx={example_idx} is empty")

            val_out_dict = self.val_out_dicts[example_idx]
            val_out_dict = {
                k: tr.cat(v, dim=0)[: self.max_n_points]
                for k, v in val_out_dict.items()
            }
            if val_out_dict:
                d_grad = None
                s_grad = None
                if pl_module.log_val_grads:
                    d_grad = self.val_d_grads[example_idx]
                    s_grad = self.val_s_grads[example_idx]
                    d_grad = tr.cat(d_grad, dim=0)[: self.max_n_points]
                    s_grad = tr.cat(s_grad, dim=0)[: self.max_n_points]
                    max_d_grad = d_grad.abs().max()
                    max_s_grad = s_grad.abs().max()
                    avg_d_grad = d_grad.abs().mean()
                    avg_s_grad = s_grad.abs().mean()
                    max_grad = max(max_d_grad, max_s_grad)
                    d_grad /= max_grad
                    s_grad /= max_grad
                    title = (
                        f"val_{example_idx}_{title_suffix}"
                        f"\nmax_d∇: {max_d_grad:.4f}"
                        f" max_s∇: {max_s_grad:.4f}"
                        f"\navg_d∇: {avg_d_grad:.4f}"
                        f" avg_s∇: {avg_s_grad:.4f}"
                    )
                else:
                    title = f"val_{example_idx}_{title_suffix}"

                plot_xy_points_and_grads(
                    ax[1],
                    val_out_dict["theta_s"],
                    val_out_dict["theta_d"],
                    val_out_dict["theta_s_hat"],
                    val_out_dict["theta_d_hat"],
                    s_grad,
                    d_grad,
                    title=title,
                )
            else:
                log.warning(f"val_out_dict for example_idx={example_idx} is empty")

            fig.tight_layout()
            img = fig2img(fig)
            images.append(img)

        if images:
            for logger in trainer.loggers:
                # TODO(cm): enable for tensorboard as well
                if isinstance(logger, WandbLogger):
                    logger.log_image(
                        key="xy_points_and_grads",
                        images=images,
                        step=trainer.global_step,
                    )

        self.train_d_grads.clear()
        self.train_s_grads.clear()
        self.train_out_dicts.clear()
        self.val_d_grads.clear()
        self.val_s_grads.clear()
        self.val_out_dicts.clear()
