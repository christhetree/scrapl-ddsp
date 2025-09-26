import logging
import os
import torch

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

ROOT_DIR = os.path.abspath(os.path.join(__file__, "../../"))

CONFIGS_DIR = os.path.join(ROOT_DIR, "configs")
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUT_DIR = os.path.join(ROOT_DIR, "out")

assert os.path.isdir(DATA_DIR)
assert os.path.isdir(OUT_DIR)

WANDB_LOGS_DIR = os.path.join(OUT_DIR, "wandb_logs")
LIGHTNING_LOGS_DIR = os.path.join(OUT_DIR, "lightning_logs")
AUDIO_SAVE_DIR = os.path.join(OUT_DIR, "eval_808_samples")
TSV_SAVE_DIR = OUT_DIR

# if torch.cuda.is_available():
#     WANDB_LOGS_DIR = "/import/anonymous/out/wandb_logs"
#     LIGHTNING_LOGS_DIR = "/import/anonymous/out/lightning_logs"
#     AUDIO_SAVE_DIR = "/import/anonymous/out/eval_808_samples"
#     TSV_SAVE_DIR = "/import/anonymous/out"
