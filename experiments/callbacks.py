import logging
import os
from collections import defaultdict
from typing import Any, Dict

import torch as tr
import wandb
import yaml
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer, Callback, LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor as T

from experiments import util
from experiments.lightning import SCRAPLLightingModule
from experiments.paths import OUT_DIR
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
            theta_density = out_dict["theta_density"][0]
            theta_slope = out_dict["theta_slope"][0]
            theta_density_hat = out_dict["theta_density_hat"][0]
            theta_slope_hat = out_dict["theta_slope_hat"][0]
            seed = out_dict["seed"][0]
            seed_hat = out_dict["seed_hat"][0]

            title = (
                f"batch_idx_{example_idx}, "
                f"θd: {theta_density:.2f} -> {theta_density_hat:.2f}, "
                f"θs: {theta_slope:.2f} -> {theta_slope_hat:.2f}"
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
        self.x_audio = []
        self.x_hat_audio = []
        self.images = []

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: SCRAPLLightingModule,
        out_dict: Dict[str, T],
        batch: (T, T, T),
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
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

                    title = (
                        f"batch_idx_{idx}, "
                        f"θd: {theta_density[idx]:.2f} -> "
                        f"{theta_density_hat[idx]:.2f}, "
                        f"θs: {theta_slope[idx]:.2f} -> "
                        f"{theta_slope_hat[idx]:.2f}"
                    )

                    fig = plot_waveforms_stacked(
                        waveforms, pl_module.synth.sr, title, labels
                    )
                    img = fig2img(fig)
                    self.images.append(img)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: SCRAPLLightingModule
    ) -> None:
        for logger in trainer.loggers:
            # TODO(cm): enable for tensorboard as well
            if isinstance(logger, WandbLogger):
                logger.log_image(
                    key="waveforms", images=self.images, step=trainer.global_step
                )
                data = defaultdict(list)
                columns = [f"idx_{idx}" for idx in range(len(self.images))]
                for idx, curr_x_audio in enumerate(self.x_audio):
                    data["x_audio"].append(
                        wandb.Audio(
                            curr_x_audio,
                            caption=f"x_{idx}",
                            sample_rate=int(pl_module.synth.sr),
                        )
                    )
                for idx, curr_x_hat_audio in enumerate(self.x_hat_audio):
                    data["x_hat_audio"].append(
                        wandb.Audio(
                            curr_x_hat_audio,
                            caption=f"x_hat_{idx}",
                            sample_rate=int(pl_module.synth.sr),
                        )
                    )
                data = list(data.values())
                logger.log_table(
                    key="audio", columns=columns, data=data, step=trainer.global_step
                )


class LogGradientCallback(Callback):
    REQUIRED_OUT_DICT_KEYS = {
        "theta_density",
        "theta_slope",
        "theta_density_hat",
        "theta_slope_hat",
    }

    def __init__(self, n_examples: int = 5, max_n_points: int = 16) -> None:
        super().__init__()
        self.n_examples = n_examples
        self.max_n_points = max_n_points
        self.train_density_grads = defaultdict(list)
        self.train_slope_grads = defaultdict(list)
        self.train_out_dicts = defaultdict(lambda: defaultdict(list))
        self.val_density_grads = defaultdict(list)
        self.val_slope_grads = defaultdict(list)
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
            density_grad = out_dict["theta_density_hat"].grad.detach().cpu()
            slope_grad = out_dict["theta_slope_hat"].grad.detach().cpu()
            self.train_density_grads[example_idx].append(density_grad)
            self.train_slope_grads[example_idx].append(slope_grad)

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
                theta_density_hat = out_dict["theta_density_hat"]
                theta_slope_hat = out_dict["theta_slope_hat"]
                dist = out_dict["loss"].clone()
                density_grad, slope_grad = tr.autograd.grad(
                    dist, [theta_density_hat, theta_slope_hat]
                )
                density_grad = density_grad.detach().cpu()
                slope_grad = slope_grad.detach().cpu()
                density_grad /= trainer.accumulate_grad_batches
                slope_grad /= trainer.accumulate_grad_batches
                self.val_density_grads[example_idx].append(density_grad)
                self.val_slope_grads[example_idx].append(slope_grad)

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
                density_grad = self.train_density_grads[example_idx]
                slope_grad = self.train_slope_grads[example_idx]
                density_grad = tr.cat(density_grad, dim=0)[: self.max_n_points]
                slope_grad = tr.cat(slope_grad, dim=0)[: self.max_n_points]
                max_density_grad = density_grad.abs().max()
                max_slope_grad = slope_grad.abs().max()
                avg_density_grad = density_grad.abs().mean()
                avg_slope_grad = slope_grad.abs().mean()
                max_grad = max(max_density_grad, max_slope_grad)
                density_grad /= max_grad
                slope_grad /= max_grad

                plot_xy_points_and_grads(
                    ax[0],
                    train_out_dict["theta_slope"],
                    train_out_dict["theta_density"],
                    train_out_dict["theta_slope_hat"],
                    train_out_dict["theta_density_hat"],
                    slope_grad,
                    density_grad,
                    title=f"train_{example_idx}_{title_suffix}"
                    f"\nmax_d∇: {max_density_grad:.4f}"
                    f" max_s∇: {max_slope_grad:.4f}"
                    f"\navg_d∇: {avg_density_grad:.4f}"
                    f" avg_s∇: {avg_slope_grad:.4f}",
                )
            else:
                log.warning(f"train_out_dict for example_idx={example_idx} is empty")

            val_out_dict = self.val_out_dicts[example_idx]
            val_out_dict = {
                k: tr.cat(v, dim=0)[: self.max_n_points]
                for k, v in val_out_dict.items()
            }
            if val_out_dict:
                density_grad = None
                slope_grad = None
                if pl_module.log_val_grads:
                    density_grad = self.val_density_grads[example_idx]
                    slope_grad = self.val_slope_grads[example_idx]
                    density_grad = tr.cat(density_grad, dim=0)[: self.max_n_points]
                    slope_grad = tr.cat(slope_grad, dim=0)[: self.max_n_points]
                    max_density_grad = density_grad.abs().max()
                    max_slope_grad = slope_grad.abs().max()
                    avg_density_grad = density_grad.abs().mean()
                    avg_slope_grad = slope_grad.abs().mean()
                    max_grad = max(max_density_grad, max_slope_grad)
                    density_grad /= max_grad
                    slope_grad /= max_grad
                    title = (
                        f"val_{example_idx}_{title_suffix}"
                        f"\nmax_d∇: {max_density_grad:.4f}"
                        f" max_s∇: {max_slope_grad:.4f}"
                        f"\navg_d∇: {avg_density_grad:.4f}"
                        f" avg_s∇: {avg_slope_grad:.4f}"
                    )
                else:
                    title = f"val_{example_idx}_{title_suffix}"

                plot_xy_points_and_grads(
                    ax[1],
                    val_out_dict["theta_slope"],
                    val_out_dict["theta_density"],
                    val_out_dict["theta_slope_hat"],
                    val_out_dict["theta_density_hat"],
                    slope_grad,
                    density_grad,
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

        self.train_density_grads.clear()
        self.train_slope_grads.clear()
        self.train_out_dicts.clear()
        self.val_density_grads.clear()
        self.val_slope_grads.clear()
        self.val_out_dicts.clear()


class SaveSCRAPLLogitsCallback(Callback):
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        try:
            logits = pl_module.loss_func.logits.detach().cpu()
        except Exception as e:
            return
        log.info("Saving logits and probs")
        out_path = os.path.join(OUT_DIR, f"{pl_module.run_name}__logits.pt")
        tr.save(logits, out_path)
        probs = util.limited_softmax(
            logits, tau=pl_module.loss_func.tau, max_prob=pl_module.loss_func.max_prob
        )
        out_path = os.path.join(OUT_DIR, f"{pl_module.run_name}__probs.pt")
        tr.save(probs, out_path)


class SavePathCountsCallback(Callback):
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        try:
            path_counts = dict(pl_module.loss_func.path_counts)
        except Exception as e:
            return
        log.info("Saving path counts")
        out_path = os.path.join(OUT_DIR, f"{pl_module.run_name}__path_counts.yml")
        data = yaml.dump(path_counts)
        with open(out_path, "w") as f:
            f.write(data)


class SaveTargetPathEnergiesCallback(Callback):
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        try:
            target_path_energies = dict(pl_module.loss_func.target_path_energies)
        except Exception as e:
            return
        log.info("Saving target path energies")
        out_path = os.path.join(OUT_DIR, f"{pl_module.run_name}__target_energies.yml")
        data = yaml.dump(target_path_energies)
        with open(out_path, "w") as f:
            f.write(data)


class SaveMeanAbsSxGradsCallback(Callback):
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        try:
            mean_abs_Sx_grads = dict(pl_module.loss_func.path_mean_abs_Sx_grads)
        except Exception as e:
            return
        log.info("Saving mean abs Sx grads")
        out_path = os.path.join(OUT_DIR, f"{pl_module.run_name}__mean_abs_Sx_grads.yml")
        data = yaml.dump(mean_abs_Sx_grads)
        with open(out_path, "w") as f:
            f.write(data)


class SaveThetaGradsCallback(Callback):
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        try:
            d_grads = dict(pl_module.d_grads)
            s_grads = dict(pl_module.s_grads)
        except Exception as e:
            return
        log.info("Saving D and S grads")
        out_path = os.path.join(OUT_DIR, f"{pl_module.run_name}__d_grads.yml")
        data = yaml.dump(d_grads)
        with open(out_path, "w") as f:
            f.write(data)
        out_path = os.path.join(OUT_DIR, f"{pl_module.run_name}__s_grads.yml")
        data = yaml.dump(s_grads)
        with open(out_path, "w") as f:
            f.write(data)
