import logging
import math
import os

import torch as tr
import yaml
from matplotlib import cm, patches
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import (
    FuncFormatter,
    FixedLocator,
    MaxNLocator,
)

from experiments import util
from experiments.paths import CONFIGS_DIR, DATA_DIR
from experiments.scrapl_loss import SCRAPLLoss

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

if __name__ == "__main__":
    # probs_path = os.path.join(
    #     DATA_DIR,
    #     "adaptive_scrapl/scrapl_saga_pwa_1e-5__texture_32_32_5_meso_b32__probs__n_theta_2__n_params_28__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt",
    # )
    probs_path = os.path.join(
        # OUT_DIR,
        # "iclr_2026_raw/scrapl_saga_pwa_1e-5__chirplet_32_32_5_meso_b16_am_lo_fm_lo__2__probs__n_theta_2__n_params_28__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt",
        # "iclr_2026_raw/scrapl_saga_pwa_1e-5__chirplet_32_32_5_meso_b16_am_hi_fm_lo__probs__n_theta_2__n_params_28__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt",
        # "iclr_2026_raw/scrapl_saga_pwa_1e-5__chirplet_32_32_5_meso_b16_am_lo_fm_hi__probs__n_theta_2__n_params_28__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt",
        # "iclr_2026_raw/scrapl_saga_pwa_1e-5__chirplet_32_32_5_meso_b16_am_hi_fm_hi__probs__n_theta_2__n_params_28__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt",
        # "iclr_2026_raw/scrapl_saga_pwa_1e-5__chirplet_32_32_5_meso_b16_am_lo_fm_lo__log_probs__n_theta_2__n_params_28__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt",
        # "iclr_2026_raw/scrapl_saga_pwa_1e-5__chirplet_32_32_5_meso_b16_am_hi_fm_hi__log_probs__n_theta_2__n_params_28__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt",
        # "iclr_2026_raw/scrapl_saga_pwa_1e-5__chirplet_32_32_5_meso_b16_am_hi_fm_lo__log_probs__n_theta_2__n_params_28__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt",
        # "iclr_2026_raw/scrapl_saga_pwa_1e-5__chirplet_32_32_5_meso_b16_am_lo_fm_hi__log_probs__n_theta_2__n_params_28__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt",
        # "scrapl_saga_pwa_1e-5__chirplet_32_32_5_meso_b16_am_lo_fm_lo__probs__n_theta_2__n_params_28__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt",
        # "scrapl_saga_pwa_1e-5__chirplet_32_32_5_meso_b28_am_lo_fm_lo__probs__n_theta_2__n_params_28__n_batches_10__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt",
        DATA_DIR,
        # "adaptive_scrapl/scrapl_saga_pwa_1e-5__chirplet_32_32_5_meso_b28_am_lo_fm_lo__probs__n_theta_2__n_params_28__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt",
        # "adaptive_scrapl/scrapl_saga_pwa_1e-5__chirplet_32_32_5_meso_b28_am_hi_fm_hi__probs__n_theta_2__n_params_28__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt",
        # "adaptive_scrapl/scrapl_saga_pwa_1e-5__chirplet_32_32_5_meso_b28_am_lo_fm_hi__probs__n_theta_2__n_params_28__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt",
        # "adaptive_scrapl/scrapl_saga_pwa_1e-5__chirplet_32_32_5_meso_b28_am_hi_fm_lo__probs__n_theta_2__n_params_28__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt",

        # "adaptive_scrapl/scrapl_saga_pwa_1e-5__chirplet_32_32_5_meso_b16_am_lo_fm_lo__probs__n_theta_2__n_params_28__n_batches_10__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt",
        # "adaptive_scrapl/scrapl_saga_pwa_1e-5__chirplet_32_32_5_meso_b16_am_hi_fm_hi__probs__n_theta_2__n_params_28__n_batches_10__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt",
        # "adaptive_scrapl/scrapl_saga_pwa_1e-5__chirplet_32_32_5_meso_b16_am_lo_fm_hi__probs__n_theta_2__n_params_28__n_batches_10__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt",
        # "adaptive_scrapl/scrapl_saga_pwa_1e-5__chirplet_32_32_5_meso_b16_am_hi_fm_lo__probs__n_theta_2__n_params_28__n_batches_10__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt",

        # "adaptive_scrapl/scrapl_saga_pwa_1e-4__chirplet2_32_32_5_meso_b32_am_lo_fm_lo__probs__n_theta_2__n_params_28__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt",
        # "adaptive_scrapl/scrapl_saga_pwa_1e-4__chirplet2_32_32_5_meso_b32_am_lo_fm_med__probs__n_theta_2__n_params_28__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt",
        # "adaptive_scrapl/scrapl_saga_pwa_1e-4__chirplet2_32_32_5_meso_b32_am_hi_fm_med__probs__n_theta_2__n_params_28__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt",
        "adaptive_scrapl/scrapl_saga_pwa_1e-4__chirplet2_32_32_5_meso_b32_am_hi_fm_hi__probs__n_theta_2__n_params_28__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt",
    )
    probs = tr.load(probs_path)
    # probs = probs.exp()
    # probs = probs[0, :]
    # probs = probs[1, :]
    n_paths = len(probs)

    if "am_lo_fm_lo" in probs_path:
        # coords = [(0.9333, 1.3333), (2.8, 1.3333), (2.8, 4.0), (0.9333, 4.0)]
        coords = [(0.9899, 0.5), (1.9799, 0.5), (1.9799, 1.0), (0.9899, 1.0)]
        title = f"Adaptive SCRAPL Path Sampling Probabilities\n(Chirplet Synth {n_paths} paths: slow AM, slow FM)"
    elif "am_hi_fm_hi" in probs_path:
        # coords = [(2.8, 4.0), (8.4, 4.0), (8.4, 12.0), (2.8, 12.0)]
        coords = [(3.9598, 8.0), (7.9196, 8.0), (7.9196, 16.0), (3.9598, 16.0)]
        title = f"Adaptive SCRAPL Path Sampling Probabilities\n(Chirplet Synth {n_paths} paths: fast AM, fast FM)"
    elif "am_lo_fm_hi" in probs_path:
        coords = [(0.9333, 4.0), (2.8, 4.0), (2.8, 12.0), (0.9333, 12.0)]
        title = f"Adaptive SCRAPL Path Sampling Probabilities\n(Chirplet Synth {n_paths} paths: slow AM, fast FM)"
    elif "am_hi_fm_lo" in probs_path:
        coords = [(2.8, 1.3333), (8.4, 1.3333), (8.4, 4.0), (2.8, 4.0)]
        title = f"Adaptive SCRAPL Path Sampling Probabilities\n(Chirplet Synth {n_paths} paths: fast AM, slow FM)"
    elif "am_lo_fm_med" in probs_path:
        coords = [(0.9899, 2.0), (1.9799, 2.0), (1.9799, 4.0), (0.9899, 4.0)]
        title = f"Adaptive SCRAPL Path Sampling Probabilities\n(Chirplet Synth {n_paths} paths: slow AM, moderate FM)"
    elif "am_hi_fm_med" in probs_path:
        coords = [(3.9598, 2.0), (7.9196, 2.0), (7.9196, 4.0), (3.9598, 4.0)]
        title = f"Adaptive SCRAPL Path Sampling Probabilities\n(Chirplet Synth {n_paths} paths: fast AM, moderate FM)"
    else:
        raise ValueError(f"Unrecognized probs_path: {probs_path}")

    scrapl_config_path = os.path.join(CONFIGS_DIR, "losses/scrapl_chirplet.yml")
    with open(scrapl_config_path, "r") as f:
        scrapl_config = yaml.safe_load(f)
    scrapl_loss = SCRAPLLoss(**scrapl_config["init_args"])
    Q1 = scrapl_loss.jtfs.Q[0]

    sr = 8192.0
    _, key_data = util.get_path_keys(scrapl_loss.meta, sr=sr, Q1=Q1)
    assert len(key_data) == scrapl_loss.n_paths
    assert scrapl_loss.n_paths == probs.size(0)

    x = []
    y = []
    for k in key_data:
        spin = k["spin"]
        am_hz = k["am_cf_hz"]
        fm_hz = k["fm_cf_oct_hz"]
        if spin == -1:
            fm_hz = -fm_hz
        x.append(round(am_hz, 2))
        y.append(round(fm_hz, 2))

    unique_x = sorted(set(x))
    x_to_idx = {v: i for i, v in enumerate(unique_x)}
    unique_y = sorted(set(y))
    unique_y = unique_y[:-1]
    y_to_idx = {v: i for i, v in enumerate(unique_y)}
    z = tr.zeros((len(unique_y), len(unique_x)))

    for idx, (curr_x, curr_y) in enumerate(zip(x, y)):
        if curr_y == tr.inf:
            continue
        x_idx = x_to_idx[curr_x]
        y_idx = y_to_idx[curr_y]
        assert z[y_idx, x_idx] == 0.0, f"Collision at x={curr_x}, y={curr_y}"
        z[y_idx, x_idx] = probs[idx].item()

    unif_prob = 1.0 / len(probs)
    z = z / unif_prob
    z_max = z.max().item()
    log.info(f"z.max() = {z.max():.4f}, z.min() = {z.min():.4f}")
    # max_z_val = tr.tensor(6.2285).log1p().item()
    # max_z_val = tr.tensor(7.0231).log1p().item()
    max_z_val = tr.tensor(7.2690).log1p().item()
    mid_z_val = tr.tensor(1.0).log1p().item()
    z = z.log1p()

    plt.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=120)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=mid_z_val, vmax=max_z_val)
    cf = ax.contourf(unique_x, unique_y, z, levels=32, cmap=cm.coolwarm, norm=norm)
    ax.scatter(x, y, color="black", s=15.0)
    ax.set_xlabel("AM rate (Hz)")
    ax.set_xscale("log")
    # ax.set_xlim(0.5, 16.0)
    # ax.set_xticks([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    ax.set_xlim(0.8, 9.6)
    ax.set_xticks([1.0, 2.0, 4.0, 8.0])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_ylabel("FM rate (octaves / s)")
    ax.set_yscale("log")
    # ax.set_ylim(1.0, 16.0)
    # ax.set_yticks([1.0, 2.0, 4.0, 8.0, 16.0])
    ax.set_ylim(0.4, 19.2)
    ax.set_yticks([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    # ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}"))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    # ax.set_yscale("symlog", linthresh=0.5)
    # plt.axhline(y=1.333, color="r", linestyle="--")
    # plt.axhline(y=4.0, color="r", linestyle="--")
    # plt.axhline(y=12.0, color="r", linestyle="--")
    # plt.axvline(x=0.9333, color="r", linestyle="--")
    # plt.axvline(x=2.8, color="r", linestyle="--")
    # plt.axvline(x=8.4, color="r", linestyle="--")
    # plt.title(probs_path)
    plt.title(title)

    rect = patches.Polygon(
        coords, closed=True, fill=False, edgecolor="black", linewidth=2, linestyle="--"
    )
    ax.add_patch(rect)
    cbar = plt.colorbar(cf, label="Uniform Probability Ratio")
    # ticks = tr.tensor([0.0, 0.5, 1.0, 2.0, 4.0, math.floor(z_max * 10) / 10.0]).log1p().tolist()
    ticks = tr.tensor([0.0, 0.5, 1.0, 2.0, 4.0]).log1p().tolist()
    cbar.set_ticks(ticks)
    # cbar.locator = MaxNLocator(nbins=6)
    # cbar.update_ticks()
    cbar.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{tr.tensor(x).expm1().item():.1f}  ")
    )
    plt.tight_layout()
    plt.show()
