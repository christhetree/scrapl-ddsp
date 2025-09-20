import logging
import os
from collections import defaultdict
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Subplot
from pandas import DataFrame

from experiments.paths import OUT_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "WARNING"))


def calc_tv(df: DataFrame, x_col: str, y_col: str) -> (float, float):
    # Check that x_col is monotonically increasing and unique
    assert df[x_col].is_monotonic_increasing
    assert df[x_col].is_unique
    n = len(df)
    y_vals = df[y_col].values
    tv = 0.0
    for i in range(1, n):
        tv += np.abs(y_vals[i] - y_vals[i - 1])
    tv_x_normed = tv / n
    y_min = df[y_col].min()
    y_max = df[y_col].max()
    y_range = y_max - y_min
    y_vals_0to1 = (y_vals - y_min) / y_range
    tv = 0.0
    for i in range(1, n):
        tv += np.abs(y_vals_0to1[i] - y_vals_0to1[i - 1])
    tv_xy_normed = tv / n
    return tv_x_normed, tv_xy_normed


def prepare_tsv_data(
    tsv_path: str,
    stage: str,
    x_col: str,
    y_col: str,
    y_converge_val: float = 0.1,
    trial_col: str = "seed",
    time_col: str = "time_epoch",
    allow_var_n: bool = False,
) -> Dict[str, np.ndarray]:
    tsv_col_names = ["stage", "x_col", "y_col"]
    print_tsv_vals = [stage, x_col, y_col]
    df = pd.read_csv(tsv_path, sep="\t", index_col=False)

    # Filter out stage
    df = df[df["stage"] == stage]
    log.debug(f"Number of rows before removing warmup steps: {len(df)}")
    # Remove sanity check rows
    df = df[~((df["step"] == 0) & (df["stage"] == "val"))]
    log.debug(f"Number of rows after  removing warmup steps: {len(df)}")

    # if "jtfs" in tsv_path:
    #     df["step"] = df["step"] * 18.5
    #     df["step"] = df["step"] - df["step"].min() + 96.0

    data = defaultdict(list)
    grouped = df.groupby(trial_col)
    n = len(grouped)
    log.info(f"Number of trials: {n}")
    tsv_col_names.append("n_trials")
    print_tsv_vals.append(n)

    # if "warmup" in tsv_path:
    #     # subtract 315 from step column
    #     df["step"] = df["step"] - 314

    x_val_mins = []
    x_val_maxs = []
    x_val_ranges = []
    durs = []
    tvs_x_normed = []
    tvs_xy_normed = []
    converged = []
    converged_x_vals = []

    for _, group in grouped:
        # Calc ranges and duration per step
        x_val_min = group[x_col].min()
        x_val_min_ts = group[group[x_col] == x_val_min][time_col].values[0]
        x_val_max = group[x_col].max()
        x_val_max_ts = group[group[x_col] == x_val_max][time_col].values[0]
        x_val_mins.append(x_val_min)
        x_val_maxs.append(x_val_max)
        x_val_ranges.append(x_val_max - x_val_min)
        dur = x_val_max_ts - x_val_min_ts
        durs.append(dur)
        # Take mean of y values if there are multiple for each x value (e.g. val / test or grad accumulation)
        grouped_x = group.groupby(x_col).agg({y_col: "mean"})
        for x_val, y_val in grouped_x.itertuples():
            data[x_val].append(y_val)
        if stage != "test":
            # Calc TV
            grouped_x.reset_index(drop=False, inplace=True)
            tv_x_normed, tv_xy_normed = calc_tv(grouped_x, x_col, y_col)
            tvs_x_normed.append(tv_x_normed)
            tvs_xy_normed.append(tv_xy_normed)
        # Check for convergence
        y_val_min = grouped_x[y_col].min()
        if y_val_min <= y_converge_val:
            converged.append(1)
            if stage != "test":
                # Find first y value less than y_converge_val and corresponding x value
                assert grouped_x[x_col].is_monotonic_increasing
                con_x_val = grouped_x[grouped_x[y_col] <= y_converge_val][x_col].values[
                    0
                ]
                converged_x_vals.append(con_x_val)
        else:
            converged.append(0)

    if not allow_var_n:
        assert len(set(x_val_mins)) == 1, f"Found var min x val {x_val_mins}"
        assert len(set(x_val_maxs)) == 1, f"Found var max x val {x_val_maxs}"
        assert len(set(x_val_ranges)) == 1, "Found var range x val"

    x_vals = []
    y_means = []
    y_vars = []
    y_stds = []
    y_mins = []
    y_maxs = []
    y_95cis = []
    y_ns = []
    # We use a for loop to handle jagged data
    for x_val in sorted(data):
        x_vals.append(x_val)
        y_vals = data[x_val]
        n = len(y_vals)
        y_mean = np.mean(y_vals)
        y_var = np.var(y_vals)
        y_std = np.std(y_vals)
        y_sem = y_std / np.sqrt(n)
        y_95ci = 1.96 * y_sem
        y_min = np.min(y_vals)
        y_max = np.max(y_vals)
        y_means.append(y_mean)
        y_vars.append(y_var)
        y_stds.append(y_std)
        y_mins.append(y_min)
        y_maxs.append(y_max)
        y_95cis.append(y_95ci)
        y_ns.append(n)
    if not allow_var_n:
        assert len(set(y_ns)) == 1, "Found var no. of trials across different x vals"
    x_vals = np.array(x_vals)
    y_means = np.array(y_means)
    y_95cis = np.array(y_95cis)
    y_mins = np.array(y_mins)
    y_maxs = np.array(y_maxs)

    # Display mean, 95% CI, and range for test stage
    if stage == "test":
        assert len(data) == 1
        # Calc mean, 95% CI, and range
        y_vals = list(data.values())[0]
        y_mean = np.mean(y_vals)
        y_std = np.std(y_vals)
        y_sem = y_std / np.sqrt(len(y_vals))
        y_95ci = 1.96 * y_sem
        y_min = np.min(y_vals)
        y_max = np.max(y_vals)
        log.info(
            f"{y_col} mean: {y_mean:.4f}, 95% CI: {y_95ci:.4f} "
            f"({y_mean - y_95ci:.4f}, {y_mean + y_95ci:.4f}), "
            f"range: ({y_min:.4f}, {y_max:.4f}), n: {len(y_vals)}"
        )
        print_tsv_vals.extend(
            [y_mean, y_95ci, y_mean - y_95ci, y_mean + y_95ci, y_min, y_max]
        )
    else:
        print_tsv_vals.extend(["n/a"] * 6)
    tsv_col_names.extend(["y_mean", "y_95ci", "y_mean-", "y_mean+", "y_min", "y_max"])

    # Display variance info
    y_std = np.mean(y_stds)
    y_var = np.mean(y_vars)
    log.info(f"{y_col} mean std: {y_std:.4f}, mean var: {y_var:.8f}")
    print_tsv_vals.extend([y_std, y_var])
    tsv_col_names.extend(["y_std", "y_var"])

    # Display TV information
    if stage != "test":
        tv_x_normed = np.mean(tvs_x_normed)
        tv_x_normed_95ci = 1.96 * (np.std(tvs_x_normed) / np.sqrt(len(tvs_x_normed)))
        tv_xy_normed = np.mean(tvs_xy_normed)
        tv_xy_normed_95ci = 1.96 * (np.std(tvs_xy_normed) / np.sqrt(len(tvs_xy_normed)))
        print_tsv_vals.extend([
            tv_x_normed,
            tv_x_normed_95ci,
            tv_x_normed - tv_x_normed_95ci,
            tv_x_normed + tv_x_normed_95ci,
            tv_xy_normed,
            tv_xy_normed_95ci,
            tv_xy_normed - tv_xy_normed_95ci,
            tv_xy_normed + tv_xy_normed_95ci
        ])
    else:
        print_tsv_vals.extend(["n/a"] * 8)
    tsv_col_names.extend(["tv_x_normed", "tv_x_normed_95ci",
                          "tv_x_normed-", "tv_x_normed+",
                          "tv_xy_normed", "tv_xy_normed_95ci",
                          "tv_xy_normed-", "tv_xy_normed+"])

    # Display convergence information
    con_rate = np.mean(converged)
    log.info(f"Converged rate: {con_rate:.4f}, y converge val: {y_converge_val}")
    print_tsv_vals.extend([y_converge_val, con_rate])
    tsv_col_names.extend(["y_converge_val", "con_rate"])
    if stage != "test" and con_rate > 0:
        con_x_val = np.mean(converged_x_vals)
        con_x_std = np.std(converged_x_vals)
        con_x_min = np.min(converged_x_vals)
        con_x_max = np.max(converged_x_vals)
        con_x_sem = con_x_std / np.sqrt(len(converged_x_vals))
        con_x_95ci = 1.96 * con_x_sem
        log.info(
            f"Converged {x_col}: {con_x_val:.0f}, 95% CI: {con_x_95ci:.0f} "
            f"({con_x_val - con_x_95ci:.0f}, {con_x_val + con_x_95ci:.0f}), "
            f"range: ({con_x_min:.0f}, {con_x_max:.0f}), n: {len(converged_x_vals)}"
        )
        print_tsv_vals.extend(
            [
                con_x_val,
                con_x_95ci,
                con_x_val - con_x_95ci,
                con_x_val + con_x_95ci,
                con_x_min,
                con_x_max,
            ]
        )
    else:
        print_tsv_vals.extend(["n/a"] * 6)
    tsv_col_names.extend(
        [
            "con_x_val",
            "con_x_95ci",
            "con_x_val-",
            "con_x_val+",
            "con_x_min",
            "con_x_max",
        ]
    )

    # Display duration information
    x_val_min = x_val_mins[0]
    x_val_max = x_val_maxs[0]
    if x_val_max != x_val_min:
        durs_per_step = [
            dur / x_val_range for dur, x_val_range in zip(durs, x_val_ranges)
        ]
        avg_dur_per_step = np.mean(durs_per_step)
    else:
        avg_dur_per_step = 0.0
    log.info(
        f"Min {x_col}: {x_val_min}, Max {x_col}: {x_val_max}, "
        f"Avg dur per {x_col}: {avg_dur_per_step:.4f} sec"
    )
    print_tsv_vals.extend([x_val_min, x_val_max, avg_dur_per_step])
    tsv_col_names.extend(["x_val_min", "x_val_max", "avg_dur_per_step"])

    return {
        "x_vals": x_vals,
        "y_means": y_means,
        "y_95cis": y_95cis,
        "y_mins": y_mins,
        "y_maxs": y_maxs,
        "tsv_col_names": tsv_col_names,
        "tsv_vals": print_tsv_vals,
    }


def plot_xy_vals(
    ax: Subplot,
    data: Dict[str, np.ndarray],
    title: Optional[str] = None,
    plot_95ci: bool = True,
    plot_range: bool = True,
    use_log_x: bool = False,
    use_log_y: bool = False,
    color: Optional[str] = None,
) -> None:
    x_vals = data["x_vals"]
    y_means = data["y_means"]
    y_95cis = data["y_95cis"]
    y_mins = data["y_mins"]
    y_maxs = data["y_maxs"]

    y_vals = y_means
    y_95n_vals = y_means - y_95cis
    y_95p_vals = y_means + y_95cis
    if use_log_x:
        x_vals = np.log10(x_vals)
    if use_log_y:
        y_vals = np.log10(y_vals)
        y_95n_vals = np.log10(y_95n_vals)
        y_95p_vals = np.log10(y_95p_vals)
        y_mins = np.log10(y_mins)
        y_maxs = np.log10(y_maxs)

    mean_label = "mean"
    if title is not None:
        mean_label = title
    ax.plot(x_vals, y_vals, label=mean_label, lw=2, color=color)
    if plot_95ci:
        ax.fill_between(
            x_vals,
            y_95n_vals,
            y_95p_vals,
            alpha=0.4,
            # color=color,
        )
    if plot_range:
        ax.fill_between(x_vals, y_mins, y_maxs, color="gray", alpha=0.4)

    # Labels and legend
    # ax.set_xlim(0, 19200)
    # ax.set_xlim(0, 80000)
    # ax.set_xlim(0, 4800)
    ax.set_xlabel(f"{x_col}")
    # ax.set_xlabel(f"Steps")
    # ax.set_xticklabels([])
    # ax.set_xlabel(f"Wall Clock Time")
    # ax.set_ylim(bottom=0.0)
    # ax.set_ylim(bottom=0.0, top=0.12)
    ax.set_ylabel(f"{y_col}")
    # ax.set_ylabel("$\\theta_{synth} \\; L_2$")
    ax.legend()
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0)
    ax.grid(True)


if __name__ == "__main__":
    tsv_names_and_paths = [
        # ("adam_prev", os.path.join(OUT_DIR, f"results/texture/scrapl_adamw_1e-5_b32__texture_32_32_5_meso.tsv")),  # TODO(cm): something is off here
        # ("pwa_prev", os.path.join(OUT_DIR, f"results/texture/scrapl_pwa_sgd_1e-5_b32__texture_32_32_5_meso.tsv")),
        # ("saga_adam_prev", os.path.join(OUT_DIR, f"results/texture/scrapl_just_saga_adam_1e-5_b32__texture_32_32_5_meso.tsv")),
        # ("saga_pwa_prev", os.path.join(OUT_DIR, f"results/texture/scrapl_saga_sgd_1e-5_b32__texture_32_32_5_meso.tsv")),
        # ("saga_pwa_prev2", os.path.join(OUT_DIR, f"results_scrapl_loss/texture/scrapl_saga_pwa_1e-5_b32__texture_32_32_5_meso.tsv")),
        # ("jtfs_prev", os.path.join(OUT_DIR, f"results/texture/jtfs_adamw_1e-5_b32__texture_32_32_5_meso.tsv")),
        # ("clap_prev", os.path.join(OUT_DIR, f"results/texture/clap_adamw_1e-5_b32__texture_32_32_5_meso.tsv")),
        # ("mss_prev", os.path.join(OUT_DIR, f"results/texture/mss_adamw_1e-5_b32__texture_32_32_5_meso.tsv")),
        # ("rand_mss_prev", os.path.join(OUT_DIR, f"results/texture/rand_mss_adamw_1e-5_b32__texture_32_32_5_meso.tsv")),
        # ("mss_rev_prev", os.path.join(OUT_DIR, f"results/texture/mss_revisited_adamw_1e-5_b32__texture_32_32_5_meso.tsv")),
        # ("saga_ds_w0", os.path.join(OUT_DIR, f"results/texture/scrapl_saga_ds_w0_sgd_1e-5_b32__texture_32_32_5_meso.tsv")),

        # ("saga_pwa", os.path.join(OUT_DIR, f"iclr_2026_done/scrapl_saga_pwa_1e-5__texture_32_32_5_meso_b32.tsv")),
        # ("Adaptive\nSCRAPL", os.path.join(OUT_DIR, f"iclr_2026_done/scrapl_saga_pwa_1e-5__adaptive_n_batches_10_n_iter_20_param_agg_none__texture_32_32_5_meso_b32.tsv")),
        # ("jtfs", os.path.join(OUT_DIR, f"iclr_2026_done/jtfst_adam_1e-5__texture_32_32_5_meso_b32.tsv")),
        # ("mss", os.path.join(OUT_DIR, f"iclr_2026_done/mss_1e-5__texture_32_32_5_meso_b32.tsv")),
        # ("mss_meso", os.path.join(OUT_DIR, f"mss_meso_1e-5__texture_32_32_5_meso_b32.tsv")),
        # ("MSS", os.path.join(OUT_DIR, f"mss_meso_log_1e-5__texture_32_32_5_meso_b32.tsv")),
        # ("MSS Rev.", os.path.join(OUT_DIR, f"iclr_2026_done/mss_revisited_1e-5__texture_32_32_5_meso_b32.tsv")),
        # ("Rand. MSS", os.path.join(OUT_DIR, f"iclr_2026_done/rand_mss_1e-5__texture_32_32_5_meso_b32.tsv")),
        # ("MS-CLAP", os.path.join(OUT_DIR, f"iclr_2026_done/clap_adam_1e-5__texture_32_32_5_meso_b32.tsv")),
        # ("PANNs", os.path.join(OUT_DIR, f"iclr_2026_done/panns_wglm_adam_1e-5__texture_32_32_5_meso_b32.tsv")),
        # ("ploss", os.path.join(OUT_DIR, f"iclr_2026_done/ploss_adam_1e-5__texture_32_32_5_meso_b32.tsv")),

        # ("Adaptive\nSCRAPL", os.path.join(OUT_DIR, f"iclr_2026_done/scrapl_saga_pwa_1e-5__adaptive_n_batches_10_n_iter_20_param_agg_none__texture_32_32_5_meso_b32.tsv")),
        # ("MSS", os.path.join(OUT_DIR, f"mss_meso_log_1e-5__texture_32_32_5_meso_b32.tsv")),
        # ("MSS Rev.", os.path.join(OUT_DIR, f"iclr_2026_done/mss_revisited_1e-5__texture_32_32_5_meso_b32.tsv")),
        # ("Rand. MSS", os.path.join(OUT_DIR, f"iclr_2026_done/rand_mss_1e-5__texture_32_32_5_meso_b32.tsv")),
        # ("MS-CLAP", os.path.join(OUT_DIR, f"iclr_2026_done/clap_adam_1e-5__texture_32_32_5_meso_b32.tsv")),
        # ("PANNs", os.path.join(OUT_DIR, f"iclr_2026_done/panns_wglm_adam_1e-5__texture_32_32_5_meso_b32.tsv")),

        # ("SCRAPL (no PWA, no SAGA)", os.path.join(OUT_DIR, f"iclr_2026_done/scrapl_adam_1e-5__texture_32_32_5_meso_b32.tsv")),
        # # ("saga_adam", os.path.join(OUT_DIR, f"iclr_2026_done/scrapl_saga_adam_1e-5__texture_32_32_5_meso_b32.tsv")),
        # ("SCRAPL (no SAGA)", os.path.join(OUT_DIR, f"iclr_2026_done/scrapl_pwa_1e-5__texture_32_32_5_meso_b32.tsv")),
        # ("SCRAPL", os.path.join(OUT_DIR, f"iclr_2026_done/scrapl_saga_pwa_1e-5__texture_32_32_5_meso_b32.tsv")),
        # ("Adaptive SCRAPL", os.path.join(OUT_DIR, f"iclr_2026_done/scrapl_saga_pwa_1e-5__adaptive_n_batches_10_n_iter_20_param_agg_none__texture_32_32_5_meso_b32.tsv")),
        # ("JTFS", os.path.join(OUT_DIR, f"iclr_2026_done/jtfst_adam_1e-5__texture_32_32_5_meso_b32.tsv")),
        # ("ploss", os.path.join(OUT_DIR, f"iclr_2026_done/ploss_adam_1e-5__texture_32_32_5_meso_b32.tsv")),

        # ("adam_old", os.path.join(OUT_DIR, f"iclr_2026/scrapl_adam_1e-5__texture_32_32_5_meso_b32__old.tsv")),
        # ("saga_pwa_b1_none", os.path.join(OUT_DIR, f"iclr_2026_done/scrapl_saga_pwa_1e-5__adaptive_n_batches_1_n_iter_20_param_agg_none__texture_32_32_5_meso_b32.tsv")),
        # ("saga_pwa_b1_mean", os.path.join(OUT_DIR, f"iclr_2026_done/scrapl_saga_pwa_1e-5__adaptive_n_batches_1_n_iter_20_param_agg_mean__texture_32_32_5_meso_b32.tsv")),
        # ("saga_pwa_b1_max", os.path.join(OUT_DIR, f"iclr_2026_done/scrapl_saga_pwa_1e-5__adaptive_n_batches_1_n_iter_20_param_agg_max__texture_32_32_5_meso_b32.tsv")),
        # ("clap_2", os.path.join(OUT_DIR, f"iclr_2026/clap_adam_1e-5__texture_32_32_5_meso_b32.tsv")),

        # ("saga_pwa", os.path.join(OUT_DIR, f"iclr_2026_done/scrapl_saga_pwa_1e-4__chirplet_32_32_5_meso_b16_am_lo_fm_lo.tsv")),
        # ("saga_pwa_b1_none", os.path.join(OUT_DIR, f"iclr_2026_raw/scrapl_saga_pwa_1e-4__adaptive_n_batches_1_n_iter_20_param_agg_none__chirplet_32_32_5_meso_b16_am_lo_fm_lo.tsv")),

        # ("lo_lo", os.path.join(OUT_DIR, f"iclr_2026_done/chirplet/scrapl_saga_pwa_1e-4__chirplet2_32_32_5_meso_b32_am_lo_fm_lo.tsv")),
        # ("lo_lo_b1_none", os.path.join(OUT_DIR, f"iclr_2026_done/chirplet/scrapl_saga_pwa_1e-4__chirplet2_32_32_5_meso_b32_am_lo_fm_lo__probs_n_batches_1.tsv")),
        # ("lo_med", os.path.join(OUT_DIR, f"iclr_2026_done/chirplet/scrapl_saga_pwa_1e-4__chirplet2_32_32_5_meso_b32_am_lo_fm_med.tsv")),
        # ("lo_med_b1_none", os.path.join(OUT_DIR, f"iclr_2026_done/chirplet/scrapl_saga_pwa_1e-4__chirplet2_32_32_5_meso_b32_am_lo_fm_med__probs_n_batches_1.tsv")),
        # ("hi_med", os.path.join(OUT_DIR, f"iclr_2026_done/chirplet/scrapl_saga_pwa_1e-4__chirplet2_32_32_5_meso_b32_am_hi_fm_med.tsv")),
        # ("hi_med_b1_none", os.path.join(OUT_DIR, f"iclr_2026_done/chirplet/scrapl_saga_pwa_1e-4__chirplet2_32_32_5_meso_b32_am_hi_fm_med__probs_n_batches_1.tsv")),
        # ("hi_hi", os.path.join(OUT_DIR, f"iclr_2026_done/chirplet/scrapl_saga_pwa_1e-4__chirplet2_32_32_5_meso_b32_am_hi_fm_hi.tsv")),
        # ("hi_hi_b1_none", os.path.join(OUT_DIR, f"iclr_2026_done/chirplet/scrapl_saga_pwa_1e-4__chirplet2_32_32_5_meso_b32_am_hi_fm_hi__probs_n_batches_1.tsv")),

        # DDSP 808
        # ("mss_mi", os.path.join(OUT_DIR, f"iclr_2026_done/eval_808/mss_724k_adamw_1e-4t5__mars_808_n681_b8_micro.tsv")),
        # ("mss_log_mi", os.path.join(OUT_DIR, f"iclr_2026_done/eval_808/mss_meso_log_724k_adamw_1e-4t5__mars_808_n681_b8_micro.tsv")),
        # ("mss_rev_mi", os.path.join(OUT_DIR, f"iclr_2026_done/eval_808/mss_rev_724k_adamw_1e-4t5__mars_808_n681_b8_micro.tsv")),
        # ("rand_mss_mi", os.path.join(OUT_DIR, f"iclr_2026_done/eval_808/rand_mss_724k_adamw_1e-4t5__mars_808_n681_b8_micro.tsv")),
        # ("scrapl_mi", os.path.join(OUT_DIR, f"iclr_2026_done/eval_808/scrapl_Jfr5_T2048_F1_saga_pwa_log1p_nogm_724k_1e-4t5__mars_808_n681_b8_micro.tsv")),
        # ("scrapl_no_log1p_mi", os.path.join(OUT_DIR, f"iclr_2026_done/eval_808/scrapl_Jfr5_T2048_F1_saga_pwa_724k_1e-4t5__mars_808_n681_b8_micro.tsv")),
        # ("scrapl_og_mi", os.path.join(OUT_DIR, f"iclr_2026_done/eval_808/scrapl_saga_pwa_724k_1e-4t5__mars_808_n681_b8_micro.tsv")),
        # ("jtfs_mi", os.path.join(OUT_DIR, f"iclr_2026_raw/eval_808/jtfs_Jfr5_T2048_F1_log1p_nogm_724k_adamw_1e-4t5__mars_808_n681_b8_micro.tsv")),

        ("mss_me", os.path.join(OUT_DIR, f"iclr_2026_done/eval_808/mss_724k_adamw_1e-4t5__mars_808_n681_b8_meso2048.tsv")),
        ("mss_log_me", os.path.join(OUT_DIR, f"iclr_2026_done/eval_808/mss_meso_log_724k_adamw_1e-4t5__mars_808_n681_b8_meso2048.tsv")),
        ("mss_rev_me", os.path.join(OUT_DIR, f"iclr_2026_done/eval_808/mss_rev_724k_adamw_1e-4t5__mars_808_n681_b8_meso2048.tsv")),
        ("rand_mss_me", os.path.join(OUT_DIR, f"iclr_2026_done/eval_808/rand_mss_724k_adamw_1e-4t5__mars_808_n681_b8_meso2048.tsv")),
        ("scrapl_me", os.path.join(OUT_DIR, f"iclr_2026_done/eval_808/scrapl_Jfr5_T2048_F1_saga_pwa_log1p_nogm_724k_1e-4t5__mars_808_n681_b8_meso2048.tsv")),
        ("scrapl_no_log1p_me", os.path.join(OUT_DIR, f"iclr_2026_done/eval_808/scrapl_Jfr5_T2048_F1_saga_pwa_724k_1e-4t5__mars_808_n681_b8_meso2048.tsv")),
        ("scrapl_og_me", os.path.join(OUT_DIR, f"iclr_2026_done/eval_808/scrapl_saga_pwa_724k_1e-4t5__mars_808_n681_b8_meso2048.tsv")),
        ("jtfs_me", os.path.join(OUT_DIR, f"iclr_2026_raw/eval_808/jtfs_Jfr5_T2048_F1_log1p_nogm_724k_adamw_1e-4t5__mars_808_n681_b8_meso2048.tsv")),

        # Old

        # ("pwa_saga", os.path.join(OUT_DIR, f"results/chirplet/scrapl_saga_sgd_1e-4_b32__chirplet_32_32_5_meso.tsv")),
        # ("saga_a0.25", os.path.join(OUT_DIR, f"results/chirplet/scrapl_saga_a0.25_sgd_1e-4_b32__chirplet_32_32_5_meso.tsv")),
        # ("saga_a0.125", os.path.join(OUT_DIR, f"results/chirplet/scrapl_saga_a0.125_sgd_1e-4_b32__chirplet_32_32_5_meso.tsv")),
        # ("saga_am_or_fm", os.path.join(OUT_DIR, f"results/chirplet/scrapl_saga_am_or_fm_sgd_1e-4_b32__chirplet_32_32_5_meso.tsv")),
        # ("saga_bin", os.path.join(OUT_DIR, f"results/chirplet/scrapl_saga_bin_sgd_1e-4_b32__chirplet_32_32_5_meso.tsv")),
        # ("saga_ds_w0", os.path.join(OUT_DIR, f"results/chirplet/scrapl_saga_ds_w0_sgd_1e-4_b32__chirplet_32_32_5_meso.tsv")),
        # ("saga_d_w0", os.path.join(OUT_DIR, f"results/chirplet/scrapl_saga_d_w0_sgd_1e-4_b32__chirplet_32_32_5_meso.tsv")),
        # ("saga_s_w0", os.path.join(OUT_DIR, f"results/chirplet/scrapl_saga_s_w0_sgd_1e-4_b32__chirplet_32_32_5_meso.tsv")),

        # ("saga_pwa", os.path.join(OUT_DIR, f"results_scrapl_loss/chirplet/scrapl_saga_pwa_1e-4_b32__chirplet_32_32_5_meso.tsv")),
        # ("saga_pwa_ds_up", os.path.join(OUT_DIR, f"results_scrapl_loss/chirplet/scrapl_saga_pwa_ds_update_1e-4_b32__chirplet_32_32_5_meso.tsv")),
        # ("saga_pwa_ds", os.path.join(OUT_DIR, f"results_scrapl_loss/chirplet/scrapl_saga_pwa_1e-4_b32__chirplet_32_32_5_meso__probs__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_None.tsv")),
        # ("saga_pwa_ds_mean", os.path.join(OUT_DIR, f"results_scrapl_loss/chirplet/scrapl_saga_pwa_1e-4_b32__chirplet_32_32_5_meso__probs__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_mean.tsv")),
        # ("saga_pwa_ds_20", os.path.join(OUT_DIR, f"results_scrapl_loss/chirplet/scrapl_saga_pwa_1e-4_b32__chirplet_32_32_5_meso__probs__n_batches_20__n_iter_20__min_prob_frac_0.0__param_agg_None.tsv")),

        # ("saga", os.path.join(OUT_DIR, f"results/chirplet/am/scrapl_saga_sgd_1e-4_b32__chirplet_am_32_32_5_meso.tsv")),
        # ("saga_a0.25", os.path.join(OUT_DIR, f"results/chirplet/am/scrapl_saga_a0.25_sgd_1e-4_b32__chirplet_am_32_32_5_meso.tsv")),
        # ("saga_a0.125", os.path.join(OUT_DIR, f"results/chirplet/am/scrapl_saga_a0.125_sgd_1e-4_b32__chirplet_am_32_32_5_meso.tsv")),
        # ("saga_am", os.path.join(OUT_DIR, f"results/chirplet/am/scrapl_saga_am_sgd_1e-4_b32__chirplet_am_32_32_5_meso.tsv")),
        # ("saga_bin", os.path.join(OUT_DIR, f"results/chirplet/am/scrapl_saga_am_bin_sgd_1e-4_b32__chirplet_am_32_32_5_meso.tsv")),
        # ("saga_w0", os.path.join(OUT_DIR, f"results/chirplet/am/scrapl_saga_w0_sgd_1e-4_b32__chirplet_am_32_32_5_meso.tsv")),

        # ("saga", os.path.join(OUT_DIR, f"results/chirplet/fm/scrapl_saga_sgd_1e-4_b32__chirplet_fm_32_32_5_meso.tsv")),
        # ("saga_a0.25", os.path.join(OUT_DIR, f"chirplet/fm/scrapl_saga_a0.25_sgd_1e-4_b32__chirplet_fm_32_32_5_meso.tsv")),
        # ("saga_a0.125", os.path.join(OUT_DIR, f"chirplet/fm/scrapl_saga_a0.125_sgd_1e-4_b32__chirplet_fm_32_32_5_meso.tsv")),
        # ("saga_fm", os.path.join(OUT_DIR, f"results/chirplet/fm/scrapl_saga_fm_sgd_1e-4_b32__chirplet_fm_32_32_5_meso.tsv")),
        # ("saga_bin", os.path.join(OUT_DIR, f"chirplet/fm/scrapl_saga_fm_bin_sgd_1e-4_b32__chirplet_fm_32_32_5_meso.tsv")),
        # ("saga_w0", os.path.join(OUT_DIR, f"results/chirplet/fm/scrapl_saga_w0_sgd_1e-4_b32__chirplet_fm_32_32_5_meso.tsv")),
    ]

    # stage = "train"
    # stage = "val"
    stage = "test"
    x_col = "step"

    y_col_prefix = ""
    # y_col_prefix = "BD__"
    # y_col_prefix = "SD__"
    # y_col_prefix = "Tom__"
    # y_col_prefix = "HH__"
    # y_col_prefix = "l1"
    # y_col_prefix = "l2"
    # y_col_prefix = "rmse"

    y_col_dist = "l1"
    # y_col_dist = "rmse"

    use_log_y = False
    # use_log_y = True

    # colors = ["green", "orange", "black", "blue", "red"]
    # colors = ["blue", "red", "orange", "magenta", "green", "black"]
    # colors = ["black", "red"]
    colors = ["black", "blue"]

    # for y_col_suffix in ["_theta", "_d", "_s"]:
    # for y_col_suffix in ["_theta"]:
    for y_col_suffix in [
        f"audio__mss_meso_log",
        f"audio__mel_stft",
        f"audio__mfcc",
        f"audio__U__{y_col_dist}",
        f"audio__jtfs",
        f"fe__Loudness_0_2__{y_col_dist}",
        f"fe__Loudness_2_64__{y_col_dist}",
        f"fe__SpectralCentroid_0_2__{y_col_dist}",
        f"fe__SpectralCentroid_2_64__{y_col_dist}",
        f"fe__SpectralFlatness_0_2__{y_col_dist}",
        f"fe__SpectralFlatness_2_64__{y_col_dist}",
        f"fe__TemporalCentroid_0_1__{y_col_dist}",
    ]:
        y_col = f"{y_col_prefix}{y_col_suffix}"
        # Plot
        plt.rcParams.update({"font.size": 12})
        # plt.rcParams.update({"font.size": 14})
        fig, ax = plt.subplots(figsize=(6, 4), dpi=240)
        # fig, ax = plt.subplots(figsize=(8.5, 4), dpi=240)
        ax.set_title(f"{stage} {y_col}")
        # ax.set_title("Chirplet Synth 315 paths:\nslow AM, slow FM")
        # ax.set_title("Chirplet Synth 315 paths:\nslow AM, moderate FM")
        # ax.set_title("Chirplet Synth 315 paths:\nfast AM, moderate FM")
        # ax.set_title("Chirplet Synth 315 paths:\nfast AM, fast FM")
        df_rows = []
        df_cols = []
        for idx, (name, tsv_path) in enumerate(tsv_names_and_paths):
            log.info(f"Plotting {name}, stage: {stage} ===================================")
            data = prepare_tsv_data(
                tsv_path, stage, x_col, y_col, y_converge_val=0.1, allow_var_n=False
            )
            # color = colors[idx]
            color = None
            plot_xy_vals(ax, data, title=name, plot_95ci=True, plot_range=False, use_log_y=use_log_y, color=color)
            # plot_xy_vals(ax, data, title=name, plot_95ci=False, plot_range=False, use_log_y=use_log_y, color=color)
            df_cols = ["name"] + data["tsv_col_names"]
            df_row = [name] + data["tsv_vals"]
            assert len(df_cols) == len(df_row)
            df_rows.append(df_row)

        if stage != "test":
            fig.tight_layout()
            plt.show()

        df = pd.DataFrame(df_rows, columns=df_cols)
        # if stage != "test":
        #     df["tv_x_normed"] = df["tv_x_normed"] * 100
        #     df["tv_x_normed_95ci"] = df["tv_x_normed_95ci"] * 100
        # df["con_rate"] = df["con_rate"] * 100
        # pd.set_option("display.float_format", lambda x: "%.2f" % x)
        # pd.set_option("display.float_format", lambda x: "%.3f" % x)
        # pd.set_option("display.float_format", lambda x: "%.0f" % x)
        print(df.to_string(index=False))
        print()
