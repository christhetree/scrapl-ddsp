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
    config_name = "train.yml"
    seeds = None
    # seeds = [0, 1, 43]
    # seeds = list(range(20))

    config_path = os.path.join(CONFIGS_DIR, config_name)

    if seeds is None:
        cli = CustomLightningCLI(
            args=["fit", "-c", config_path],
            trainer_defaults=CustomLightningCLI.trainer_defaults,
        )
        trainer = cli.trainer
        trainer.test(model=cli.model, datamodule=cli.datamodule, ckpt_path="best")
    else:
        log.info(f"Running with seeds: {seeds}")
        for seed in seeds:
            log.info(f"Current seed_everything value: {seed}")
            cli = CustomLightningCLI(
                args=["fit", "-c", config_path, "--seed_everything", str(seed)],
                trainer_defaults=CustomLightningCLI.trainer_defaults,
            )
            trainer = cli.trainer
            trainer.test(model=cli.model, datamodule=cli.datamodule, ckpt_path="best")
