import logging
import os
import random
from typing import Optional

import numpy as np
import torch as tr
import yaml
from matplotlib import pyplot as plt
from torch import Tensor as T
from torch import nn
from tqdm import tqdm

from experiments.losses import SCRAPLLoss, JTFSTLoss, WaveletLoss
from experiments.paths import CONFIGS_DIR, OUT_DIR
from experiments.synth import ChirpTextureSynth

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def calc_distance_grad_matrices(dist_func: nn.Module,
                                synth: ChirpTextureSynth,
                                theta_density: T,
                                theta_slope: T,
                                n_density: int = 9,
                                n_slope: int = 9,
                                use_rand_seeds: bool = False,
                                grad_clip_val: Optional[float] = None,
                                seed: int = 42,
                                pbar: Optional[tqdm] = None) -> (T, T, T, T, T):
    # TODO(cm): control meso seed
    seed = tr.tensor(seed)
    x = synth(theta_density, theta_slope, seed)
    # J_cqt = 5
    # cqt_params = {
    #     "sr": synth.sr,
    #     "bins_per_octave": synth.Q,
    #     "n_bins": J_cqt * synth.Q,
    #     "hop_length": synth.hop_len,
    #     # TODO(cm): check this
    #     "fmin": (0.4 * synth.sr) / (2 ** J_cqt),
    #     "output_format": "Magnitude",
    #     "verbose": False,
    # }
    # cqt = CQT(**cqt_params)
    # y_coords = cqt.frequencies.tolist()
    # with tr.no_grad():
    #     U = SCRAPLLightingModule.calc_cqt(x, cqt).squeeze()
    #     fig, ax = plt.subplots(1, 1)
    #     plot_scalogram(ax, U, synth.sr, y_coords, hop_len=synth.hop_len, title="U")
    #     fig.show()
    # exit()

    # TODO(cm)
    theta_density_hats = tr.linspace(0.4, 0.6, n_density + 2, requires_grad=True)[1:-1]
    theta_slope_hats = tr.linspace(-1.0, 1.0, n_slope + 2, requires_grad=True)[1:-1]
    dist_rows = []
    density_grad_rows = []
    slope_grad_rows = []

    for theta_density_hat in theta_density_hats:
        dist_row = []
        density_grad_row = []
        slope_grad_row = []
        for theta_slope_hat in theta_slope_hats:
            seed_hat = seed
            if use_rand_seeds:
                # TODO(cm): make cleaner
                seed_hat = tr.randint(seed.item(), seed.item() + 999999, (1,))
            x_hat = synth(theta_density_hat, theta_slope_hat, seed_hat)

            # with tr.no_grad():
            #     U = SCRAPLLightingModule.calc_cqt(x_hat, cqt).squeeze()
            #     fig, ax = plt.subplots(1, 1)
            #     plot_scalogram(ax, U, synth.sr, y_coords, hop_len=synth.hop_len, title=f"U S={theta_slope_hat:.2f}")
            #     fig.show()

            dist = dist_func(x_hat.view(1, 1, -1), x.view(1, 1, -1))
            dist = dist.squeeze()
            dist_row.append(dist.item())

            density_grad, slope_grad = tr.autograd.grad(
                dist, [theta_density_hat, theta_slope_hat])
            density_grad_row.append(density_grad.item())
            if slope_grad is None:
                log.warning(f"slope_grad is None")
                slope_grad_row.append(0.0)
            else:
                slope_grad_row.append(slope_grad.item())

            if pbar is not None:
                pbar.update(1)

        dist_rows.append(dist_row)
        density_grad_rows.append(density_grad_row)
        slope_grad_rows.append(slope_grad_row)

    dist_matrix = tr.tensor(dist_rows)
    dgm = tr.tensor(density_grad_rows)
    sgm = tr.tensor(slope_grad_rows)

    if grad_clip_val is not None:
        assert False  # Disable for now
        # TODO(cm): this is different from default clipping which acts on the norm
        log.info(f"grad_clip_val={grad_clip_val:.2f}")
        dgm = tr.clip(dgm, -grad_clip_val, grad_clip_val)
        sgm = tr.clip(sgm, -grad_clip_val, grad_clip_val)

    return theta_density_hats, theta_slope_hats, dist_matrix, dgm, sgm


def plot_gradients(theta_density: T,
                   theta_slope: T,
                   dist_matrix: T,
                   theta_density_hats: T,
                   theta_slope_hats: T,
                   dgm: T,
                   sgm: T,
                   title: Optional[str] = None,
                   save_path: Optional[str] = None) -> None:
    if title is None:
        title = f"gradients\nθ density={theta_density:.2f}, θ slope={theta_slope:.2f}"

    n_density = dist_matrix.size(0)
    n_slope = dist_matrix.size(1)
    theta_density_indices = list(range(n_density))
    theta_slope_indices = list(range(n_slope))

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
    ax.set_title(title, fontsize=fontsize)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


if __name__ == "__main__":
    seed = 43
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

    config_path = os.path.join(CONFIGS_DIR, "synths/chirp_8khz.yml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    synth = ChirpTextureSynth(**config["init_args"])
    synth = synth.to(device)

    config_path = os.path.join(CONFIGS_DIR, "losses/scrapl.yml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    scrapl_loss = SCRAPLLoss(**config["init_args"])

    config_path = os.path.join(CONFIGS_DIR, "losses/jtfst2.yml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    jtfst_loss = JTFSTLoss(**config["init_args"])

    # config_path = os.path.join(CONFIGS_DIR, "losses/jtfst2_mine_2d.yml")
    # with open(config_path, 'r') as f:
    #     config = yaml.safe_load(f)
    # jtfst_mine_2d_loss = MyJTFST2DLoss(**config["init_args"])

    # dist_func = nn.L1Loss()
    # dist_func = nn.MSELoss()
    # dist_func = auraloss.freq.RandomResolutionSTFTLoss(max_fft_size=16384 * 2 - 1)
    # dist_func = auraloss.freq.MultiResolutionSTFTLoss()
    # dist_func = scrapl_loss
    # dist_func = jtfst_loss
    # dist_func = jtfst_mine_2d_loss
    dist_func = WaveletLoss(sr=synth.sr,
                            n_samples=synth.n_samples,
                            J=6,
                            Q1=24)

    dist_func = dist_func.to(device)

    use_rand_seeds = False  # Micro
    # use_rand_seeds = True  # Meso
    theta_density = tr.tensor(0.5)
    theta_slope = tr.tensor(0.4)
    n_density = 1
    n_slope = 9
    n_trials = 1

    if dist_func == jtfst_loss or dist_func == scrapl_loss:
        psi1_freqs = [f["xi"] * synth.sr for f in dist_func.jtfs.psi1_f]
        psi1_max_freq = max(psi1_freqs)
        psi1_min_freq = min(psi1_freqs)

        log.info(f"f0_min_hz={synth.f0_min_hz:.2f}, f0_max_hz={synth.f0_max_hz:.2f}")
        log.info(f"n_psi1 = {len(psi1_freqs)}, "
                 f"psi1_min_freq={psi1_min_freq:.2f}, "
                 f"psi1_max_freq={psi1_max_freq:.2f}")
        log.info(f"psi1_freqs={[round(f) for f in psi1_freqs]}")
        assert psi1_min_freq < synth.f0_min_hz

        psi2_freqs = [f["xi"] * synth.sr for f in dist_func.jtfs.psi2_f]
        psi2_max_freq = max(psi2_freqs)
        psi2_min_freq = min(psi2_freqs)
        log.info(f"n_psi2 = {len(psi2_freqs)}, "
                 f"psi2_min_freq={psi2_min_freq:.2f}, "
                 f"psi2_max_freq={psi2_max_freq:.2f}")
        log.info(f"psi2_freqs={[round(f) for f in psi2_freqs]}")

        log.info(f"n_psi_fr = {len(dist_func.jtfs.filters_fr[1])}")

        suffix = (f"d{n_density}s{n_slope}t{n_trials}__J{dist_func.jtfs.J}"
                  f"_Q{dist_func.jtfs.Q[0]}"
                  f"_QQ{dist_func.jtfs.Q[1]}"
                  f"_Jfr{dist_func.jtfs.J_fr}"
                  f"_Qfr{dist_func.jtfs.Q_fr[0]}"
                  f"_T{dist_func.jtfs.T}"
                  f"_F{dist_func.jtfs.F}")
    # elif dist_func == jtfst_mine_2d_loss:
    #     suffix = (f"d{n_density}s{n_slope}t{n_trials}__J{dist_func.jtfs.J_1}"
    #               f"_Q{dist_func.jtfs.Q_1}"
    #               f"_QQ{dist_func.jtfs.Q_2_t}"
    #               f"_Jfr{dist_func.jtfs.J_2_f}"
    #               f"_Qfr{dist_func.jtfs.Q_2_f}"
    #               f"_T{dist_func.jtfs.avg_win_t}"
    #               f"_F{dist_func.jtfs.avg_win_f}")
    else:
        suffix = f"d{n_density}s{n_slope}t{n_trials}"
    # exit()

    if use_rand_seeds:
        save_name = f"dist__{dist_func.__class__.__name__}__meso__{suffix}.png"
    else:
        save_name = f"dist__{dist_func.__class__.__name__}__micro__{suffix}.png"

    log.info(f"save_name={save_name}")

    save_path = os.path.join(OUT_DIR, save_name)

    title = (f"{dist_func.__class__.__name__}\n"
             f"θ density={theta_density:.2f}, "
             f"θ slope={theta_slope:.2f}, "
             f"{'meso' if use_rand_seeds else 'micro'}, "
             f"t{n_trials}")

    theta_density_hats = None
    theta_slope_hats = None
    dist_matrices = []
    dgm_all = []
    sgm_all = []

    with tqdm(total=n_trials * n_density * n_slope) as pbar:
        for idx in range(n_trials):
            theta_density_hats, theta_slope_hats, dist_matrix, dgm, sgm = (
                calc_distance_grad_matrices(dist_func,
                                            synth,
                                            theta_density=theta_density,
                                            theta_slope=theta_slope,
                                            n_density=n_density,
                                            n_slope=n_slope,
                                            use_rand_seeds=use_rand_seeds,
                                            seed=seed + idx,
                                            pbar=pbar)
            )
            dist_matrices.append(dist_matrix)
            dgm_all.append(dgm)
            sgm_all.append(sgm)

    dist_matrix = tr.stack(dist_matrices)
    dgm = tr.stack(dgm_all)
    sgm = tr.stack(sgm_all)
    if n_trials > 1:
        # TODO(cm): variance measurements are currently not comparable across losses?
        dist_avg_var = dist_matrix.var(dim=0).mean()
        dgm_avg_var = dgm.var(dim=0).mean()
        sgm_avg_var = sgm.var(dim=0).mean()
        log.info(f"dist_avg_var={dist_avg_var:.4f}, "
                 f" dgm_avg_var={dgm_avg_var:.4f}, "
                 f" sgm_avg_var={sgm_avg_var:.4f}")
    dist_matrix = dist_matrix.mean(dim=0)
    dgm = dgm.mean(dim=0)
    sgm = sgm.mean(dim=0)

    dist_matrix = dist_matrix / dist_matrix.abs().max()
    max_grad = max(dgm.abs().max(), sgm.abs().max())
    log.info(f"max_grad={max_grad:.4f}")
    dgm /= max_grad
    sgm /= max_grad

    plot_gradients(theta_density,
                   theta_slope,
                   dist_matrix,
                   theta_density_hats,
                   theta_slope_hats,
                   dgm,
                   sgm,
                   title=title,
                   save_path=save_path)
