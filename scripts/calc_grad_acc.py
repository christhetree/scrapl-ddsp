import logging
import os
import random
from collections import defaultdict

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

    config_path = os.path.join(CONFIGS_DIR, "losses/scrapl_adaptive.yml")
    # config_path = os.path.join(CONFIGS_DIR, "losses/jtfst_dtfa.yml")
    # config_path = os.path.join(CONFIGS_DIR, "losses/clap.yml")
    # config_path = os.path.join(CONFIGS_DIR, "losses/mss.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    loss_fn = AdaptiveSCRAPLLoss(**config["init_args"]).to(device)
    # loss_fn = JTFSTLoss(**config["init_args"]).to(device)
    # loss_fn = ClapEmbeddingLoss(**config["init_args"]).to(device)
    # loss_fn = auraloss.freq.MultiResolutionSTFTLoss(**config["init_args"])
    # loss_fn = nn.L1Loss()

    n_batches = 1000
    bs = 32
    # n_batches = 10
    # bs = 32
    is_meso = True
    # is_meso = False

    fixed_d_val = None
    # fixed_d_val = 0.5
    fixed_s_val = None
    # fixed_s_val = 0.5

    n_trials = n_batches * bs
    losses = []
    grads = defaultdict(list)
    accs = defaultdict(list)

    # theta_d = tr.rand(bs, device=device)
    # theta_d_hat = tr.rand(bs, device=device)
    # if fixed_d_val is not None:
    #     theta_d.fill_(fixed_d_val)
    #     theta_d_hat.fill_(fixed_d_val)
    # theta_d_hat.requires_grad = True
    #
    # theta_s = tr.rand(bs, device=device)
    # theta_s_hat = tr.rand(bs, device=device)
    # if fixed_s_val is not None:
    #     theta_s.fill_(fixed_s_val)
    #     theta_s_hat.fill_(fixed_s_val)
    # theta_s_hat.requires_grad = True

    for idx in tqdm(range(n_batches)):
        start_idx = idx * bs
        end_idx = start_idx + bs
        seeds = tr.tensor(list(range(start_idx, end_idx)), dtype=tr.long, device=device)
        if is_meso:
            seeds_hat = (
                tr.randint(0, 9999999, (bs,), dtype=tr.long, device=device) + n_trials
            )
        else:
            seeds_hat = seeds

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

        with tr.no_grad():
            x = synth(theta_d, theta_s, seeds)
        x_hat = synth(theta_d_hat, theta_s_hat, seeds_hat)

        loss = loss_fn(x, x_hat)
        loss.backward()

        losses.append(loss.detach().cpu())
        grads["theta_d"].append(theta_d_hat.grad.detach().cpu())
        grads["theta_s"].append(theta_s_hat.grad.detach().cpu())

        # Check whether the sign of the gradient is correct
        theta_d_sign = theta_d_hat - theta_d
        theta_s_sign = theta_s_hat - theta_s
        acc_d = tr.sign(theta_d_sign) == tr.sign(theta_d_hat.grad)
        acc_s = tr.sign(theta_s_sign) == tr.sign(theta_s_hat.grad)
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

        # theta_d_hat.grad.zero_()
        # theta_s_hat.grad.zero_()

    log.info(f"loss_fn: {loss_fn.__class__.__name__}, "
             f"synth: {synth.__class__.__name__}")
    log.info(f"n_batches = {n_batches}, bs = {bs}, n_trials = {n_trials}")
    log.info(f"is_meso = {is_meso}, "
             f"fixed_d_val = {fixed_d_val}, fixed_s_val = {fixed_s_val}")
    losses = tr.stack(losses)
    for k, v in accs.items():
        v = tr.cat(v, dim=0)
        mean_acc = v.float().mean().item()
        log.info(f"{k} sign accuracy: {mean_acc:.4f}")


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
