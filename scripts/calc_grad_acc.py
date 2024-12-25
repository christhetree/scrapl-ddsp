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
from experiments.paths import CONFIGS_DIR
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
            theta_d.fill_(fixed_d_val)
            theta_d_hat.fill_(fixed_d_val)
        theta_d_hat.requires_grad = True

        theta_s = tr.rand(bs, device=device)
        theta_s_hat = tr.rand(bs, device=device)
        if fixed_s_val is not None:
            theta_s.fill_(fixed_s_val)
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
        v = tr.cat(v, dim=0)
        mean_acc = v.float().mean().item()
        results[f"{k}"] = mean_acc

    return results


if __name__ == "__main__":
    seed = 42
    tr.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    if tr.cuda.is_available():
        log.info("Using GPU")
        device = tr.device("cuda")
    else:
        log.info("Using CPU")
        device = tr.device("cpu")

    config_path = os.path.join(CONFIGS_DIR, "synths/chirp_texture_8khz.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    synth = ChirpTextureSynth(**config["init_args"]).to(device)

    # config_path = os.path.join(CONFIGS_DIR, "losses/scrapl_adaptive.yml")
    # config_path = os.path.join(CONFIGS_DIR, "losses/jtfst_dtfa.yml")
    # config_path = os.path.join(CONFIGS_DIR, "losses/clap.yml")
    config_path = os.path.join(CONFIGS_DIR, "losses/mss.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # loss_fn = AdaptiveSCRAPLLoss(**config["init_args"]).to(device)
    # loss_fn = JTFSTLoss(**config["init_args"]).to(device)
    # loss_fn = ClapEmbeddingLoss(**config["init_args"]).to(device)
    loss_fn = auraloss.freq.MultiResolutionSTFTLoss(**config["init_args"])
    # loss_fn = nn.L1Loss()

    # n_batches = 1000
    # bs = 32
    n_batches = 1000
    bs = 1
    is_meso = True
    # is_meso = False
    grad_mult = 1.0
    # grad_mult = 1e8
    fixed_d_val = None
    fixed_s_val = None

    if type(loss_fn) == AdaptiveSCRAPLLoss:
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


# INFO:__main__:loss_fn: MultiResolutionSTFTLoss, synth: ChirpTextureSynth
# INFO:__main__:n_batches = 1000, bs = 32, n_trials = 32000
# INFO:__main__:is_meso = True, fixed_d_val = None, fixed_s_val = None
# INFO:__main__:theta_d sign accuracy: 0.6367
# INFO:__main__:theta_s sign accuracy: 0.7256
# INFO:__main__:both sign accuracy: 0.4581
# INFO:__main__:only_d sign accuracy: 0.1787
# INFO:__main__:only_s sign accuracy: 0.2675
# INFO:__main__:neither sign accuracy: 0.0958
# INFO:__main__:either sign accuracy: 0.9042

# INFO:__main__:loss_fn: AdaptiveSCRAPLLoss, synth: ChirpTextureSynth
# INFO:__main__:n_batches = 1000, bs = 32, n_trials = 32000
# INFO:__main__:is_meso = True, fixed_d_val = None, fixed_s_val = None
# INFO:__main__:theta_d sign accuracy: 0.6353
# INFO:__main__:theta_s sign accuracy: 0.5704
# INFO:__main__:both sign accuracy: 0.3543
# INFO:__main__:only_d sign accuracy: 0.2810
# INFO:__main__:only_s sign accuracy: 0.2161
# INFO:__main__:neither sign accuracy: 0.1486
# INFO:__main__:either sign accuracy: 0.8514

# INFO:__main__:loss_fn: ClapEmbeddingLoss, synth: ChirpTextureSynth
# INFO:__main__:n_batches = 1000, bs = 32, n_trials = 32000
# INFO:__main__:is_meso = True, fixed_d_val = None, fixed_s_val = None
# INFO:__main__:theta_d sign accuracy: 0.6359
# INFO:__main__:theta_s sign accuracy: 0.6693
# INFO:__main__:both sign accuracy: 0.4282
# INFO:__main__:only_d sign accuracy: 0.2077
# INFO:__main__:only_s sign accuracy: 0.2412
# INFO:__main__:neither sign accuracy: 0.1230
# INFO:__main__:either sign accuracy: 0.8770

# INFO:__main__:loss_fn: JTFSTLoss, synth: ChirpTextureSynth
# INFO:__main__:n_batches = 1000, bs = 8, n_trials = 8000
# INFO:__main__:is_meso = True, fixed_d_val = None, fixed_s_val = None
# INFO:__main__:theta_d sign accuracy: 0.7165
# INFO:__main__:theta_s sign accuracy: 0.9034
# INFO:__main__:both sign accuracy: 0.6472
# INFO:__main__:only_d sign accuracy: 0.0693
# INFO:__main__:only_s sign accuracy: 0.2561
# INFO:__main__:neither sign accuracy: 0.0274
# INFO:__main__:either sign accuracy: 0.9726

# INFO:__main__:loss_fn: JTFSTLoss, synth: ChirpTextureSynth
# INFO:__main__:n_batches = 1000, bs = 1, n_trials = 1000
# INFO:__main__:is_meso = True, fixed_d_val = None, fixed_s_val = None
# INFO:__main__:theta_d sign accuracy: 0.7170
# INFO:__main__:theta_s sign accuracy: 0.9110
# INFO:__main__:both sign accuracy: 0.6490
# INFO:__main__:only_d sign accuracy: 0.0680
# INFO:__main__:only_s sign accuracy: 0.2620
# INFO:__main__:neither sign accuracy: 0.0210
# INFO:__main__:either sign accuracy: 0.9790
# INFO:__main__:theta_d grad mean: -248.9628, std: 3050.1624
# INFO:__main__:theta_s grad mean: 345.2253, std: 24780.7773
# INFO:__main__:theta_d grad z-score mean: 0.4559, var: 0.8664
# INFO:__main__:theta_s grad z-score mean: 0.2656, var: 0.9220

# INFO:__main__:loss_fn: MultiResolutionSTFTLoss, synth: ChirpTextureSynth
# INFO:__main__:n_batches = 1000, bs = 1, n_trials = 1000
# INFO:__main__:is_meso = True, fixed_d_val = None, fixed_s_val = None
# INFO:__main__:theta_d sign accuracy: 0.6320
# INFO:__main__:theta_s sign accuracy: 0.7240
# INFO:__main__:both sign accuracy: 0.4600
# INFO:__main__:only_d sign accuracy: 0.1720
# INFO:__main__:only_s sign accuracy: 0.2640
# INFO:__main__:neither sign accuracy: 0.1040
# INFO:__main__:either sign accuracy: 0.8960
# INFO:__main__:theta_d grad mean: 0.4560, std: 0.6196
# INFO:__main__:theta_s grad mean: 0.0304, std: 10.3713
# INFO:__main__:theta_d grad z-score mean: -0.3146, var: 1.3645
# INFO:__main__:theta_s grad z-score mean: 0.1409, var: 0.9793

# INFO:__main__:loss_fn: AdaptiveSCRAPLLoss, synth: ChirpTextureSynth
# INFO:__main__:n_batches = 1000, bs = 1, n_trials = 1000
# INFO:__main__:is_meso = True, fixed_d_val = None, fixed_s_val = None
# INFO:__main__:theta_d sign accuracy: 0.6000
# INFO:__main__:theta_s sign accuracy: 0.5780
# INFO:__main__:both sign accuracy: 0.3400
# INFO:__main__:only_d sign accuracy: 0.2600
# INFO:__main__:only_s sign accuracy: 0.2380
# INFO:__main__:neither sign accuracy: 0.1620
# INFO:__main__:either sign accuracy: 0.8380
# INFO:__main__:theta_d grad mean: -221.2339, std: 16201.7988
# INFO:__main__:theta_s grad mean: -23151.0312, std: 831001.5000
# INFO:__main__:theta_d grad z-score mean: 0.0892, var: 0.9945
# INFO:__main__:theta_s grad z-score mean: 0.0677, var: 0.9992





# INFO:__main__:loss_fn: MultiResolutionSTFTLoss, synth: ChirpTextureSynth
# INFO:__main__:n_batches = 100, bs = 32, n_trials = 3200
# INFO:__main__:is_meso = True, fixed_d_val = None, fixed_s_val = None
# INFO:__main__:     theta_d_mag: 0.7247
# INFO:__main__:     theta_s_mag: 0.8987
# INFO:__main__:         theta_d: 0.6350
# INFO:__main__:         theta_s: 0.7309
# INFO:__main__:            both: 0.4591
# INFO:__main__:          only_d: 0.1759
# INFO:__main__:          only_s: 0.2719
# INFO:__main__:         neither: 0.0931
# INFO:__main__:          either: 0.9069


# INFO:__main__:loss_fn: JTFSTLoss, synth: ChirpTextureSynth
# INFO:__main__:n_batches = 800, bs = 4, n_trials = 3200
# INFO:__main__:is_meso = True, fixed_d_val = None, fixed_s_val = None
# INFO:__main__:     theta_d_mag: 0.7700
# INFO:__main__:     theta_s_mag: 0.8708
# INFO:__main__:         theta_d: 0.7050
# INFO:__main__:         theta_s: 0.9013
# INFO:__main__:            both: 0.6344
# INFO:__main__:          only_d: 0.0706
# INFO:__main__:          only_s: 0.2669
# INFO:__main__:         neither: 0.0281
# INFO:__main__:          either: 0.9719

# INFO:__main__:loss_fn: AdaptiveSCRAPLLoss, synth: ChirpTextureSynth
# INFO:__main__:n_batches = 10, bs = 32, n_trials = 320
# INFO:__main__:is_meso = True, fixed_d_val = None, fixed_s_val = None
# INFO:__main__:     theta_d_mag: 0.6679
# INFO:__main__:     theta_s_mag: 0.5713
# INFO:__main__:         theta_d: 0.6331
# INFO:__main__:         theta_s: 0.5704
# INFO:__main__:            both: 0.3522
# INFO:__main__:          only_d: 0.2808
# INFO:__main__:          only_s: 0.2181
# INFO:__main__:         neither: 0.1488
# INFO:__main__:          either: 0.8512
