import logging
import os

import torch

from experiments.cli import CustomLightningCLI
from experiments.paths import CONFIGS_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    # config_name = "train.yml"
    # config_name = "train/am_fm/train.yml"

    config_name = "train/texture/train_jtfst.yml"
    # config_name = "train/texture/train_scrapl_adam.yml"
    # config_name = "train/texture/train_scrapl_saga_adam.yml"
    # config_name = "train/texture/train_scrapl_pwa.yml"
    # config_name = "train/texture/train_scrapl_saga_pwa.yml"

    # config_name = "train/chirplet/train_scrapl_saga_pwa.yml"
    # config_name = "train/chirplet/train_scrapl_saga_pwa__probs__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_None.yml"
    # config_name = "train/chirplet/train_scrapl_saga_pwa__probs__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_mean.yml"
    # config_name = "train/chirplet/train_scrapl_saga_pwa__probs__n_batches_20__n_iter_20__min_prob_frac_0.0__param_agg_None.yml"
    # config_name = "train/chirplet/train_scrapl_saga_pwa_ds_update.yml"

    # config_name = "train/chirplet/train_scrapl_adam.yml"
    # config_name = "train/chirplet/train_scrapl_pwa.yml"
    # config_name = "train/chirplet/train_scrapl_saga.yml"
    # config_name = "train/chirplet/train_scrapl_saga_am_or_fm.yml"
    # config_name = "train/chirplet/train_scrapl_saga_bin.yml"
    # config_name = "train/chirplet/train_scrapl_saga_ds_w0.yml"
    # config_name = "train/chirplet/train_scrapl_saga_d_w0.yml"
    # config_name = "train/chirplet/train_scrapl_saga_s_w0.yml"

    # config_name = "train/chirplet/am/train_scrapl_adam.yml"
    # config_name = "train/chirplet/am/train_scrapl_pwa.yml["
    # config_name = "train/chirplet/am/train_scrapl_saga.yml"
    # config_name = "train/chirplet/am/train_scrapl_saga_am.yml"
    # config_name = "train/chirplet/am/train_scrapl_saga_bin.yml"
    # config_name = "train/chirplet/am/train_scrapl_saga_w0.yml"

    # config_name = "train/chirplet/fm/train_scrapl_adam.yml"
    # config_name = "train/chirplet/fm/train_scrapl_pwa.yml"
    # config_name = "train/chirplet/fm/train_scrapl_saga.yml"
    # config_name = "train/chirplet/fm/train_scrapl_saga_fm.yml"
    # config_name = "train/chirplet/fm/train_scrapl_saga_bin.yml"
    # config_name = "train/chirplet/fm/train_scrapl_saga_w0.yml"

    # ==================================================================================
    # config_name = "eval_mixing/train.yml"

    log.info(f"Running with config: {config_name}")
    # seeds = None
    # seeds = [0, 1, 2]
    seeds = list(range(20))

    config_path = os.path.join(CONFIGS_DIR, config_name)

    if seeds is None:
        cli = CustomLightningCLI(
            args=["fit", "-c", config_path],
            trainer_defaults=CustomLightningCLI.make_trainer_defaults()
        )
        trainer = cli.trainer
        trainer.test(model=cli.model, datamodule=cli.datamodule, ckpt_path="best")
    else:
        log.info(f"Running with seeds: {seeds}")
        for seed in seeds:
            log.info(f"Current seed_everything value: {seed}")
            cli = CustomLightningCLI(
                args=["fit", "-c", config_path, "--seed_everything", str(seed)],
                trainer_defaults=CustomLightningCLI.make_trainer_defaults()
            )
            trainer = cli.trainer
            trainer.test(model=cli.model, datamodule=cli.datamodule, ckpt_path="best")
