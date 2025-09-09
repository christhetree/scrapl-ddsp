import logging
import os

import yaml
import torch as tr
from tqdm import tqdm

from experiments.paths import OUT_DIR, CONFIGS_DIR
from experiments.scrapl_loss import SCRAPLLoss

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    scrapl_config_path = os.path.join(CONFIGS_DIR, "losses/scrapl_chirplet.yml")
    with open(scrapl_config_path, "r") as f:
        scrapl_config = yaml.safe_load(f)
    scrapl_loss = SCRAPLLoss(**scrapl_config["init_args"])

    paths_dir = os.path.join(OUT_DIR, "iclr_2026_raw/paths/")
    agg = "none"
    vals = []
    for path_idx in tqdm(range(scrapl_loss.n_paths)):
        vals_path = os.path.join(paths_dir, f"vals_{path_idx}.pt")
        vals = tr.load(vals_path)

        if agg == "none":
            assert vals.size(0) == 1
            vals = vals[0]
        elif agg == "mean":
            vals = vals.mean(dim=0)
        elif agg == "max":
            vals = vals.max(dim=0).values
        elif agg == "med":
            vals = vals.median(dim=0).values
        else:
            raise ValueError(f"Invalid agg = {agg}")
        # Update the probs for each theta
        for theta_idx in range(scrapl_loss.n_theta):
            val = vals[theta_idx]
            scrapl_loss.update_prob(path_idx, val, theta_idx)

    run_name = "scrapl_saga_pwa_1p2M_1e-4__theta14_10k_b16"
    suffix = "n_theta_14__n_params_91__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0__combined.pt"

    probs_save_path = os.path.join(OUT_DIR, f"{run_name}__probs__{suffix}")
    tr.save(scrapl_loss.probs, probs_save_path)
