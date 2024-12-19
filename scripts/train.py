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
    # seeds = [0, 1]

    config_path = os.path.join(CONFIGS_DIR, config_name)

    if seeds is None:
        cli = CustomLightningCLI(
            args=["fit", "-c", config_path],
            trainer_defaults=CustomLightningCLI.trainer_defaults,
        )
    else:
        for seed in seeds:
            cli = CustomLightningCLI(
                args=["fit", "-c", config_path, "--seed_everything", seed],
                trainer_defaults=CustomLightningCLI.trainer_defaults,
            )
