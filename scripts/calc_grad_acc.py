import importlib
import logging
import os
import random
from collections import defaultdict
from typing import Optional, Dict

import auraloss
import numpy as np
import torch as tr
import yaml
from torch import nn
from tqdm import tqdm

from experiments.losses import AdaptiveSCRAPLLoss, JTFSTLoss, ClapEmbeddingLoss
from experiments.paths import CONFIGS_DIR, OUT_DIR
from experiments.synths import ChirpTextureSynth

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def calc_grad_stats(
    loss_fn: nn.Module,
    synth: nn.Module,
    n_batches: int,
    bs: int,
    is_meso: bool,
    fixed_d_val: Optional[float] = None,
    fixed_s_val: Optional[float] = None,
    grad_mult: float = 1.0,
    path_idx: Optional[int] = None,
    device: tr.device = tr.device("cpu"),
) -> Dict[str, float]:
    n_trials = n_batches * bs
    grads = defaultdict(list)
    grad_signs = defaultdict(list)
    accs = defaultdict(list)

    for idx in tqdm(range(n_batches)):
        # Prepare seeds
        start_idx = idx * bs
        end_idx = start_idx + bs
        seeds = tr.tensor(list(range(start_idx, end_idx)), dtype=tr.long, device=device)
        if is_meso:
            seeds_hat = (
                tr.randint(0, 9999999, (bs,), dtype=tr.long, device=device) + n_trials
            )
        else:
            seeds_hat = seeds

        # Prepare theta
        theta_d = tr.rand(bs, device=device)
        theta_d_hat = tr.rand(bs, device=device)
        if fixed_d_val is not None:
            # theta_d.fill_(fixed_d_val)
            theta_d_hat.fill_(fixed_d_val)
        theta_d_hat.requires_grad = True

        theta_s = tr.rand(bs, device=device)
        theta_s_hat = tr.rand(bs, device=device)
        if fixed_s_val is not None:
            # theta_s.fill_(fixed_s_val)
            theta_s_hat.fill_(fixed_s_val)
        theta_s_hat.requires_grad = True

        # Make audio
        with tr.no_grad():
            x = synth(theta_d, theta_s, seeds)
        x_hat = synth(theta_d_hat, theta_s_hat, seeds_hat)

        # Calc loss and grad
        if path_idx is None:
            loss = loss_fn(x, x_hat)
        else:
            loss = loss_fn(x, x_hat, path_idx=path_idx)
        loss.backward()
        grad_d = theta_d_hat.grad.detach()
        grad_s = theta_s_hat.grad.detach()
        grad_d *= grad_mult
        grad_s *= grad_mult

        # Store grad
        grads["theta_d"].append(grad_d.cpu())
        grads["theta_s"].append(grad_s.cpu())

        # Check whether the sign of the gradient is correct
        theta_d_sign = tr.sign(theta_d_hat - theta_d)
        theta_s_sign = tr.sign(theta_s_hat - theta_s)
        grad_signs["theta_d"].append(theta_d_sign.cpu())
        grad_signs["theta_s"].append(theta_s_sign.cpu())

        # Calc and store accuracy metrics
        acc_d = theta_d_sign == tr.sign(grad_d)
        acc_s = theta_s_sign == tr.sign(grad_s)
        accs["theta_d"].append(acc_d.int().detach().cpu())
        accs["theta_s"].append(acc_s.int().detach().cpu())
        acc_both = acc_d & acc_s
        acc_only_d = acc_d & ~acc_s
        acc_only_s = ~acc_d & acc_s
        acc_neither = ~acc_d & ~acc_s
        acc_either = acc_d | acc_s
        accs["both"].append(acc_both.int().detach().cpu())
        accs["only_d"].append(acc_only_d.int().detach().cpu())
        accs["only_s"].append(acc_only_s.int().detach().cpu())
        accs["neither"].append(acc_neither.int().detach().cpu())
        accs["either"].append(acc_either.int().detach().cpu())

    results = {}

    # Calc grad magnitude metric
    for theta_name in ["theta_d", "theta_s"]:
        grad_vals = tr.cat(grads[theta_name], dim=0)
        grad_mag_std = grad_vals.abs().std()
        results[f"{theta_name}_mag_std"] = grad_mag_std
        correct_grad_signs = tr.cat(grad_signs[theta_name], dim=0)
        # grad_mean = grad_vals.mean()  # This should be close to 0
        # grad_std = grad_vals.std()
        # log.info(f"{theta_name} grad mean: {grad_mean:.4f}, std: {grad_std:.4f}")
        # grad_vals_normalized = grad_vals / grad_std
        # grads_pos_neg = correct_grad_signs * grad_vals_normalized
        # grad_mag_metric = grads_pos_neg.mean()
        # log.info(f"{theta_name} grad_mag_metric: {grad_mag_metric:.4f}")
        # results[f"{theta_name}_metric"] = grad_mag_metric.item()
        total_mag = grad_vals.abs().sum()
        correct_grads = tr.sign(grad_vals) == correct_grad_signs
        correct_mag = grad_vals[correct_grads].abs().sum()
        grad_mag_metric = correct_mag / total_mag
        results[f"{theta_name}_mag"] = grad_mag_metric.item()

    for k, v in accs.items():
        means = tr.stack([batch_v.float().mean() for batch_v in v], dim=0)
        mean_acc = means.mean().item()
        # mean_acc_2 = tr.cat(v, dim=0).float().mean().item()
        # assert mean_acc == mean_acc_2
        results[f"{k}"] = mean_acc
        # if k in ["theta_d", "theta_s"]:
        #     mean_batch_std = means.std().item()
        #     results[f"{k}_batch_std"] = mean_batch_std

    return results


if __name__ == "__main__":
    seed = 42
    tr.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    gpu_idx = 7
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_idx}"
    if tr.cuda.is_available():
        log.info(f"Using GPU {gpu_idx}")
        device = tr.device("cuda")
    else:
        log.info("Using CPU")
        device = tr.device("cpu")

    config_path = os.path.join(CONFIGS_DIR, "synths/chirp_texture_8khz.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    synth = ChirpTextureSynth(**config["init_args"]).to(device)

    config_path = os.path.join(CONFIGS_DIR, "losses/scrapl_adaptive.yml")
    # config_path = os.path.join(CONFIGS_DIR, "losses/jtfst_dtfa.yml")
    # config_path = os.path.join(CONFIGS_DIR, "losses/clap.yml")
    # config_path = os.path.join(CONFIGS_DIR, "losses/mss.yml")
    # config_path = os.path.join(CONFIGS_DIR, "losses/rand_mss.yml")
    # config_path = os.path.join(CONFIGS_DIR, "losses/mss_revisited.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    class_path = config["class_path"]
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    loss_fn_cls = getattr(module, class_name)
    loss_fn = loss_fn_cls(**config["init_args"])
    loss_fn = loss_fn.to(device)
    # loss_fn = nn.L1Loss()

    n_batches = 100
    # n_batches = 10000
    bs = 32
    # n_batches = 400
    # bs = 8
    is_meso = True
    # is_meso = False
    grad_mult = 1.0
    # grad_mult = 1e8
    # fixed_d_val = None
    # fixed_s_val = None
    fixed_d_val = 0.5
    fixed_s_val = 0.5

    if type(loss_fn) == AdaptiveSCRAPLLoss:
    # if False:
        all_stats = defaultdict(list)
        for path_idx in tqdm(range(loss_fn.n_paths)):
            stats = calc_grad_stats(
                loss_fn,
                synth,
                n_batches,
                bs,
                is_meso,
                fixed_d_val=fixed_d_val,
                fixed_s_val=fixed_s_val,
                grad_mult=grad_mult,
                path_idx=path_idx,
                device=device,
            )
            for k, v in stats.items():
                log.info(f"{k:>16}: {v:.4f}")
                all_stats[k].append(v)
        save_path = os.path.join(OUT_DIR, "path_grad_accs.pt")
        tr.save(all_stats, save_path)
        stats = {k: np.mean(v) for k, v in all_stats.items()}
    else:
        stats = calc_grad_stats(
            loss_fn,
            synth,
            n_batches,
            bs,
            is_meso,
            fixed_d_val=fixed_d_val,
            fixed_s_val=fixed_s_val,
            grad_mult=grad_mult,
            device=device,
        )

    log.info(
        f"loss_fn: {loss_fn.__class__.__name__}, " f"synth: {synth.__class__.__name__}"
    )
    log.info(f"n_batches = {n_batches}, bs = {bs}, n_trials = {n_batches * bs}")
    log.info(
        f"is_meso = {is_meso}, "
        f"fixed_d_val = {fixed_d_val}, fixed_s_val = {fixed_s_val}"
    )
    for k, v in stats.items():
        log.info(f"{k:>16}: {v:.4f}")


# INFO:__main__:loss_fn: L1Loss, synth: ChirpTextureSynth
# INFO:__main__:n_batches = 100, bs = 32, n_trials = 3200
# INFO:__main__:is_meso = True, fixed_d_val = 0.5, fixed_s_val = 0.5
# INFO:__main__: theta_d_mag_std: 0.0000
# INFO:__main__:     theta_d_mag: 0.6212
# INFO:__main__: theta_s_mag_std: 0.0001
# INFO:__main__:     theta_s_mag: 0.5332
# INFO:__main__:         theta_d: 0.5169
# INFO:__main__:         theta_s: 0.5144
# INFO:__main__:            both: 0.2728
# INFO:__main__:          only_d: 0.2441
# INFO:__main__:          only_s: 0.2416
# INFO:__main__:         neither: 0.2416
# INFO:__main__:          either: 0.7584

# INFO:__main__:loss_fn: MultiResolutionSTFTLoss, synth: ChirpTextureSynth
# INFO:__main__:n_batches = 100, bs = 32, n_trials = 3200
# INFO:__main__:is_meso = True, fixed_d_val = 0.5, fixed_s_val = 0.5
# INFO:__main__: theta_d_mag_std: 0.0105
# INFO:__main__:     theta_d_mag: 0.7954
# INFO:__main__: theta_s_mag_std: 0.0014
# INFO:__main__:     theta_s_mag: 0.5138
# INFO:__main__:         theta_d: 0.6900
# INFO:__main__:         theta_s: 0.4934
# INFO:__main__:            both: 0.3341
# INFO:__main__:          only_d: 0.3559
# INFO:__main__:          only_s: 0.1594
# INFO:__main__:         neither: 0.1506
# INFO:__main__:          either: 0.8494

# INFO:__main__:loss_fn: RandomResolutionSTFTLoss, synth: ChirpTextureSynth
# INFO:__main__:n_batches = 100, bs = 32, n_trials = 3200
# INFO:__main__:is_meso = True, fixed_d_val = 0.5, fixed_s_val = 0.5
# INFO:__main__: theta_d_mag_std: 0.0208
# INFO:__main__:     theta_d_mag: 0.8293
# INFO:__main__: theta_s_mag_std: 0.1018
# INFO:__main__:     theta_s_mag: 0.5065
# INFO:__main__:         theta_d: 0.7506
# INFO:__main__:         theta_s: 0.4941
# INFO:__main__:            both: 0.3672
# INFO:__main__:          only_d: 0.3834
# INFO:__main__:          only_s: 0.1269
# INFO:__main__:         neither: 0.1225
# INFO:__main__:          either: 0.8775

# INFO:__main__:loss_fn: LogMSSLoss, synth: ChirpTextureSynth
# INFO:__main__:n_batches = 100, bs = 32, n_trials = 3200
# INFO:__main__:is_meso = True, fixed_d_val = 0.5, fixed_s_val = 0.5
# INFO:__main__: theta_d_mag_std: 0.2084
# INFO:__main__:     theta_d_mag: 0.6880
# INFO:__main__: theta_s_mag_std: 0.0618
# INFO:__main__:     theta_s_mag: 0.5205
# INFO:__main__:         theta_d: 0.6378
# INFO:__main__:         theta_s: 0.5150
# INFO:__main__:            both: 0.3291
# INFO:__main__:          only_d: 0.3088
# INFO:__main__:          only_s: 0.1859
# INFO:__main__:         neither: 0.1762
# INFO:__main__:          either: 0.8238

# INFO:__main__:loss_fn: ClapEmbeddingLoss, synth: ChirpTextureSynth
# INFO:__main__:n_batches = 100, bs = 32, n_trials = 3200
# INFO:__main__:is_meso = True, fixed_d_val = 0.5, fixed_s_val = 0.5
# INFO:__main__: theta_d_mag_std: 0.3162
# INFO:__main__:     theta_d_mag: 0.5762
# INFO:__main__: theta_s_mag_std: 1.5415
# INFO:__main__:     theta_s_mag: 0.6023
# INFO:__main__:         theta_d: 0.5500
# INFO:__main__:         theta_s: 0.5497
# INFO:__main__:            both: 0.3075
# INFO:__main__:          only_d: 0.2425
# INFO:__main__:          only_s: 0.2422
# INFO:__main__:         neither: 0.2078
# INFO:__main__:          either: 0.7922

# INFO:__main__:loss_fn: JTFSTLoss, synth: ChirpTextureSynth
# INFO:__main__:n_batches = 400, bs = 8, n_trials = 3200
# INFO:__main__:is_meso = True, fixed_d_val = 0.5, fixed_s_val = 0.5
# INFO:__main__: theta_d_mag_std: 0.0000
# INFO:__main__:     theta_d_mag: 0.8120
# INFO:__main__: theta_s_mag_std: 0.0000
# INFO:__main__:     theta_s_mag: 0.9664
# INFO:__main__:         theta_d: 0.7422
# INFO:__main__:         theta_s: 0.9203
# INFO:__main__:            both: 0.6850
# INFO:__main__:          only_d: 0.0572
# INFO:__main__:          only_s: 0.2353
# INFO:__main__:         neither: 0.0225
# INFO:__main__:          either: 0.9775

# Averaged across paths (each path has 3200 trials)
# INFO:__main__:loss_fn: AdaptiveSCRAPLLoss, synth: ChirpTextureSynth
# INFO:__main__:n_batches = 100, bs = 32, n_trials = 3200
# INFO:__main__:is_meso = True, fixed_d_val = 0.5, fixed_s_val = 0.5
# INFO:__main__: theta_d_mag_std: 0.0000
# INFO:__main__:     theta_d_mag: 0.6730
# INFO:__main__: theta_s_mag_std: 0.0000
# INFO:__main__:     theta_s_mag: 0.5808
# INFO:__main__:         theta_d: 0.5955
# INFO:__main__:         theta_s: 0.5739
# INFO:__main__:            both: 0.3361
# INFO:__main__:          only_d: 0.2594
# INFO:__main__:          only_s: 0.2378
# INFO:__main__:         neither: 0.1667
# INFO:__main__:          either: 0.8333

# INFO:__main__:loss_fn: AdaptiveSCRAPLLoss, synth: ChirpTextureSynth
# INFO:__main__:n_batches = 10000, bs = 32, n_trials = 320000
# INFO:__main__:is_meso = True, fixed_d_val = 0.5, fixed_s_val = 0.5
# INFO:__main__: theta_d_mag_std: 0.0000
# INFO:__main__:     theta_d_mag: 0.5621
# INFO:__main__: theta_s_mag_std: 0.0000
# INFO:__main__:     theta_s_mag: 0.7033
# INFO:__main__:         theta_d: 0.5938
# INFO:__main__:         theta_s: 0.5745
# INFO:__main__:            both: 0.3350
# INFO:__main__:          only_d: 0.2589
# INFO:__main__:          only_s: 0.2395
# INFO:__main__:         neither: 0.1666
# INFO:__main__:          either: 0.8334
