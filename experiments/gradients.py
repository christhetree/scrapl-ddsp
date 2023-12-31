import logging
import os
import random
from typing import Optional

import auraloss
import numpy as np
import torch as tr
import yaml
from matplotlib import pyplot as plt
from torch import Tensor as T
from torch import nn
from tqdm import tqdm

from experiments.losses import SCRAPLLoss, JTFSTLoss
from experiments.paths import CONFIGS_DIR, OUT_DIR
from experiments.synth import ChirpTextureSynth

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def calc_distance_grad_matrix(dist_func: nn.Module,
                              synth: ChirpTextureSynth,
                              theta_density: Optional[T] = None,
                              theta_slope: Optional[T] = None,
                              n_density: int = 9,
                              n_slope: int = 9,
                              use_rand_seeds: bool = False,
                              save_path: Optional[str] = None,
                              seed: int = 42) -> None:
    # TODO(cm): control seed, separate plotting

    if theta_density is None:
        theta_density = tr.tensor(0.5)
    if theta_slope is None:
        theta_slope = tr.tensor(0.0)
    seed = tr.tensor(seed)
    x = synth(theta_density, theta_slope, seed)
    theta_density_indices = list(range(n_density))
    theta_slope_indices = list(range(n_slope))
    theta_density_hats = tr.linspace(0.0, 1.0, n_density + 2, requires_grad=True)[1:-1]
    theta_slope_hats = tr.linspace(-1.0, 1.0, n_slope + 2, requires_grad=True)[1:-1]
    dist_rows = []
    density_grad_rows = []
    slope_grad_rows = []
    for theta_density_hat in tqdm(theta_density_hats):
        dist_row = []
        density_grad_row = []
        slope_grad_row = []
        for theta_slope_hat in tqdm(theta_slope_hats):
            if use_rand_seeds:
                seed = tr.randint(seed.item(), seed.item() + 999999, (1,))
            x_hat = synth(theta_density_hat, theta_slope_hat, seed)
            dist = dist_func(x_hat.view(1, 1, -1), x.view(1, 1, -1))
            dist = dist.squeeze()
            dist_row.append(dist.item())

            density_grad, slope_grad = tr.autograd.grad(dist,
                                                        [theta_density_hat, theta_slope_hat],
                                                        allow_unused=True)  # TODO(cm)
            density_grad_row.append(density_grad.item())
            if slope_grad is None:
                slope_grad_row.append(0.0)
            else:
                slope_grad_row.append(slope_grad.item())
        dist_rows.append(dist_row)
        density_grad_rows.append(density_grad_row)
        slope_grad_rows.append(slope_grad_row)

    dist_matrix = tr.tensor(dist_rows)
    dgm = tr.tensor(density_grad_rows)
    dgm_mu = dgm.mean()
    dgm_std = dgm.std()
    log.info(f"dgm_mu={dgm_mu:.4f}, dgm_std={dgm_std:.4f}, max dgm={dgm.abs().max():.4f}")
    dgm = (dgm - dgm_mu) / dgm_std
    dgm = tr.clip(dgm, -3, 3)
    sgm = tr.tensor(slope_grad_rows)
    sgm_mu = sgm.mean()
    sgm_std = sgm.std()
    log.info(f"sgm_mu={sgm_mu:.4f}, sgm_std={sgm_std:.4f}, max sgm={sgm.abs().max():.4f}")
    sgm = (sgm - sgm_mu) / sgm_std
    sgm = tr.clip(sgm, -3, 3)
    # log.info(f"dgm=\n{dgm}")
    # log.info(f"sgm=\n{sgm}")
    dist_matrix = dist_matrix / dist_matrix.abs().max()
    max_grad = max(dgm.abs().max(), sgm.abs().max())
    log.info(f"max_grad={max_grad:.4f}")
    dgm /= max_grad
    sgm /= max_grad

    fontsize = 14
    ax = plt.gca()
    ax.imshow(dist_matrix.numpy(), cmap='gray_r')
    # ax.imshow(tr.log1p(dist_matrix).numpy(), cmap='gray_r')
    x_labels = [f"{theta_slope_hat:.2f}" for theta_slope_hat in theta_slope_hats]
    ax.set_xticks(theta_slope_indices)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(r"θ slope hat", fontsize=fontsize - 2)
    y_labels = [f"{theta_density_hat:.2f}" for theta_density_hat in theta_density_hats]
    ax.set_yticks(theta_density_indices)
    ax.set_yticklabels(y_labels)
    ax.set_ylabel(r"θ density hat", fontsize=fontsize - 2)

    theta_slope_idx = tr.argmin(tr.abs(theta_slope_hats - theta_slope)).item()
    theta_density_idx = tr.argmin(tr.abs(theta_density_hats - theta_density)).item()
    ax.scatter([theta_slope_idx], [theta_density_idx], color='blue', marker='o', s=100)
    ax.quiver(theta_slope_indices,
              theta_density_indices,
              -sgm.numpy(),
              -dgm.numpy(),
              color='red',
              angles='xy',
              scale=8.0,
              scale_units="width")
    ax.set_title(f"{dist_func.__class__.__name__}\n"
                 f"θ density={theta_density:.2f}, "
                 f"θ slope={theta_slope:.2f}, "
                 f"{'meso' if use_rand_seeds else 'micro'}",
                 fontsize=fontsize)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


if __name__ == "__main__":
    seed = 42
    tr.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    if tr.cuda.is_available():
        log.info("Using GPU")
        device = tr.device("cuda")
    else:
        log.info("Using CPU")
        device = tr.device("cpu")

    config_path = os.path.join(CONFIGS_DIR, "synths/chirp_texture.yml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    synth = ChirpTextureSynth(**config["init_args"])

    config_path = os.path.join(CONFIGS_DIR, "losses/scrapl.yml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    scrapl_loss = SCRAPLLoss(**config["init_args"])

    config_path = os.path.join(CONFIGS_DIR, "losses/jtfst.yml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    jtfst_loss = JTFSTLoss(**config["init_args"])

    # dist_func = nn.L1Loss()
    # dist_func = nn.MSELoss()
    # dist_func = auraloss.freq.RandomResolutionSTFTLoss(max_fft_size=16384 * 2 - 1)
    # dist_func = auraloss.freq.MultiResolutionSTFTLoss()
    # dist_func = scrapl_loss
    dist_func = jtfst_loss

    synth = synth.to(device)
    dist_func = dist_func.to(device)

    use_rand_seeds = False
    # use_rand_seeds = True

    suffix = "global_euclidean"

    if use_rand_seeds:
        save_name = f"dist__{dist_func.__class__.__name__}__meso__{suffix}.png"
    else:
        save_name = f"dist__{dist_func.__class__.__name__}__micro__{suffix}.png"

    save_path = os.path.join(OUT_DIR, save_name)
    calc_distance_grad_matrix(dist_func,
                              synth,
                              theta_density=tr.tensor(0.6),
                              theta_slope=tr.tensor(0.4),
                              n_density=9,
                              n_slope=9,
                              use_rand_seeds=use_rand_seeds,
                              save_path=save_path,
                              seed=seed)
