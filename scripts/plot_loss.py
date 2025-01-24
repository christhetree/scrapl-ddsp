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
    print_tsv_vals = [stage, x_col, y_col]
    df = pd.read_csv(tsv_path, sep="\t", index_col=False)

    # Filter out stage
    df = df[df["stage"] == stage]
    log.debug(f"Number of rows before removing warmup steps: {len(df)}")
    # Remove sanity check rows
    df = df[~((df["step"] == 0) & (df["stage"] == "val"))]
    log.debug(f"Number of rows after  removing warmup steps: {len(df)}")

    data = defaultdict(list)
    grouped = df.groupby(trial_col)
    n = len(grouped)
    log.info(f"Number of trials: {n}")
    print_tsv_vals.append(n)

    if "warmup" in tsv_path:
        # subtract 315 from step column
        df["step"] = df["step"] - 314

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
                con_x_val = grouped_x[grouped_x[y_col] <= y_converge_val][x_col].values[0]
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
            [y_mean, y_95ci, y_mean - y_95ci, y_mean + y_95ci, y_min, y_max])
    else:
        print_tsv_vals.extend(["n/a"] * 6)

    # Display variance info
    y_std = np.mean(y_stds)
    y_var = np.mean(y_vars)
    log.info(f"{y_col} mean std: {y_std:.4f}, mean var: {y_var:.8f}")
    print_tsv_vals.extend([y_std, y_var])

    # Display TV information
    if stage != "test":
        tv_x_normed = np.mean(tvs_x_normed)
        tv_xy_normed = np.mean(tvs_xy_normed)
        log.info(f"TV {y_col} (x normed): {tv_x_normed:.4f}, "
                 f"TV {y_col} (xy normed): {tv_xy_normed:.4f}")
        print_tsv_vals.extend([tv_x_normed, tv_xy_normed])
    else:
        print_tsv_vals.extend(["n/a"] * 2)

    # Display convergence information
    con_rate = np.mean(converged)
    log.info(f"Converged rate: {con_rate:.4f}, y converge val: {y_converge_val}")
    print_tsv_vals.extend([y_converge_val, con_rate])
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
        print_tsv_vals.extend([con_x_val, con_x_95ci, con_x_val - con_x_95ci, con_x_val + con_x_95ci, con_x_min, con_x_max])
    else:
        print_tsv_vals.extend(["n/a"] * 6)

    # Display duration information
    x_val_min = x_val_mins[0]
    x_val_max = x_val_maxs[0]
    if x_val_max != x_val_min:
        durs_per_step = [dur / x_val_range for dur, x_val_range in
                         zip(durs, x_val_ranges)]
        avg_dur_per_step = np.mean(durs_per_step)
    else:
        avg_dur_per_step = 0.0
    log.info(f"Min {x_col}: {x_val_min}, Max {x_col}: {x_val_max}, "
             f"Avg dur per {x_col}: {avg_dur_per_step:.4f} sec")
    print_tsv_vals.extend([x_val_min, x_val_max, avg_dur_per_step])

    return {
        "x_vals": x_vals,
        "y_means": y_means,
        "y_95cis": y_95cis,
        "y_mins": y_mins,
        "y_maxs": y_maxs,
        "tsv_vals": print_tsv_vals,
    }


def plot_xy_vals(
    ax: Subplot,
    data: Dict[str, np.ndarray],
    title: Optional[str] = None,
    plot_95ci: bool = True,
    plot_range: bool = True,
) -> None:
    x_vals = data["x_vals"]
    y_means = data["y_means"]
    y_95cis = data["y_95cis"]
    y_mins = data["y_mins"]
    y_maxs = data["y_maxs"]

    mean_label = "mean"
    if title is not None:
        mean_label = title
    ax.plot(x_vals, y_means, label=mean_label, lw=2)
    if plot_95ci:
        ax.fill_between(
            x_vals,
            y_means - y_95cis,
            y_means + y_95cis,
            alpha=0.4,
        )
    if plot_range:
        ax.fill_between(x_vals, y_mins, y_maxs, color="gray", alpha=0.4)

    # Labels and legend
    ax.set_xlabel(f"{x_col}")
    ax.set_ylabel(f"{y_col}")
    ax.legend()
    ax.grid(True)


if __name__ == "__main__":
    tsv_names_and_paths = [
        # ("saga_adam_prev", os.path.join(OUT_DIR, f"scrapl_adamw_1e-5_b32__texture_32_32_5_meso.tsv")),
        # ("saga_pwa_prev", os.path.join(OUT_DIR, f"scrapl_pwa_sgd_1e-5_b32__texture_32_32_5_meso.tsv")),
        # ("saga_prev", os.path.join(OUT_DIR, f"scrapl_saga_sgd_1e-5_b32__texture_32_32_5_meso.tsv")),
        # ("saga_a0.25_prev", os.path.join(OUT_DIR, f"scrapl_saga_sgd_1e-5_b32_a0.25__texture_32_32_5_meso.tsv")),
        # ("jtfs_prev", os.path.join(OUT_DIR, f"jtfs_adamw_1e-5_b32__texture_32_32_5_meso.tsv")),
        # ("clap_prev", os.path.join(OUT_DIR, f"clap_adamw_1e-5_b32__texture_32_32_5_meso.tsv")),
        ("adam", os.path.join(OUT_DIR, f"texture/scrapl_adamw_1e-5_b32__texture_32_32_5_meso.tsv")),
        ("pwa", os.path.join(OUT_DIR, f"texture/scrapl_pwa_sgd_1e-5_b32__texture_32_32_5_meso.tsv")),
        ("saga_adam", os.path.join(OUT_DIR, f"texture/scrapl_just_saga_adam_1e-5_b32__texture_32_32_5_meso.tsv")),
        ("saga", os.path.join(OUT_DIR, f"texture/scrapl_saga_sgd_1e-5_b32__texture_32_32_5_meso.tsv")),
        # ("saga_a0.25", os.path.join(OUT_DIR, f"texture/scrapl_saga_a0.25_sgd_1e-5_b32__texture_32_32_5_meso.tsv")),
        # ("saga_a0.125", os.path.join(OUT_DIR, f"texture/scrapl_saga_a0.125_sgd_1e-5_b32__texture_32_32_5_meso.tsv")),
        # ("saga_bin", os.path.join(OUT_DIR, f"texture/scrapl_saga_bin_sgd_1e-5_b32__texture_32_32_5_meso.tsv")),
        ("saga_ds_w0", os.path.join(OUT_DIR, f"out/scrapl_saga_ds_w0_sgd_1e-5_b32__texture_32_32_5_meso.tsv")),
        ("jtfs", os.path.join(OUT_DIR, f"texture/jtfs_adamw_1e-5_b32__texture_32_32_5_meso.tsv")),
        # ("clap", os.path.join(OUT_DIR, f"texture/clap_adamw_1e-5_b32__texture_32_32_5_meso.tsv")),
        # ("mss", os.path.join(OUT_DIR, f"texture/mss_adamw_1e-5_b32__texture_32_32_5_meso.tsv")),
        # ("rand_mss", os.path.join(OUT_DIR, f"texture/rand_mss_adamw_1e-5_b32__texture_32_32_5_meso.tsv")),
        # ("mss_rev", os.path.join(OUT_DIR, f"texture/mss_revisited_adamw_1e-5_b32__texture_32_32_5_meso.tsv")),

        # ("saga", os.path.join(OUT_DIR, f"chirplet/scrapl_saga_sgd_1e-4_b32__chirplet_32_32_5_meso.tsv")),
        # ("saga_a0.25", os.path.join(OUT_DIR, f"chirplet/scrapl_saga_a0.25_sgd_1e-4_b32__chirplet_32_32_5_meso.tsv")),
        # ("saga_a0.125", os.path.join(OUT_DIR, f"chirplet/scrapl_saga_a0.125_sgd_1e-4_b32__chirplet_32_32_5_meso.tsv")),
        # ("saga_am_or_fm", os.path.join(OUT_DIR, f"chirplet/scrapl_saga_am_or_fm_sgd_1e-4_b32__chirplet_32_32_5_meso.tsv")),
        # ("saga_bin", os.path.join(OUT_DIR, f"chirplet/scrapl_saga_bin_sgd_1e-4_b32__chirplet_32_32_5_meso.tsv")),
        # ("saga_ds_w0", os.path.join(OUT_DIR, f"out/scrapl_saga_ds_w0_sgd_1e-4_b32__chirplet_32_32_5_meso.tsv")),
        # ("saga_d_w0", os.path.join(OUT_DIR, f"out/scrapl_saga_d_w0_sgd_1e-4_b32__chirplet_32_32_5_meso.tsv")),
        # ("saga_s_w0", os.path.join(OUT_DIR, f"out/scrapl_saga_s_w0_sgd_1e-4_b32__chirplet_32_32_5_meso.tsv")),

        # ("saga", os.path.join(OUT_DIR, f"chirplet/am/scrapl_saga_sgd_1e-4_b32__chirplet_am_32_32_5_meso.tsv")),
        # ("saga_a0.25", os.path.join(OUT_DIR, f"chirplet/am/scrapl_saga_a0.25_sgd_1e-4_b32__chirplet_am_32_32_5_meso.tsv")),
        # ("saga_a0.125", os.path.join(OUT_DIR, f"chirplet/am/scrapl_saga_a0.125_sgd_1e-4_b32__chirplet_am_32_32_5_meso.tsv")),
        # ("saga_am", os.path.join(OUT_DIR, f"chirplet/am/scrapl_saga_am_sgd_1e-4_b32__chirplet_am_32_32_5_meso.tsv")),
        # ("saga_bin", os.path.join(OUT_DIR, f"chirplet/am/scrapl_saga_am_bin_sgd_1e-4_b32__chirplet_am_32_32_5_meso.tsv")),
        # ("saga_w0", os.path.join(OUT_DIR, f"chirplet/am/scrapl_saga_w0_sgd_1e-4_b32__chirplet_am_32_32_5_meso.tsv")),

        # ("saga", os.path.join(OUT_DIR, f"chirplet/fm/scrapl_saga_sgd_1e-4_b32__chirplet_fm_32_32_5_meso.tsv")),
        # ("saga_a0.25", os.path.join(OUT_DIR, f"chirplet/fm/scrapl_saga_a0.25_sgd_1e-4_b32__chirplet_fm_32_32_5_meso.tsv")),
        # ("saga_a0.125", os.path.join(OUT_DIR, f"chirplet/fm/scrapl_saga_a0.125_sgd_1e-4_b32__chirplet_fm_32_32_5_meso.tsv")),
        # ("saga_fm", os.path.join(OUT_DIR, f"chirplet/fm/scrapl_saga_fm_sgd_1e-4_b32__chirplet_fm_32_32_5_meso.tsv")),
        # ("saga_bin", os.path.join(OUT_DIR, f"chirplet/fm/scrapl_saga_fm_bin_sgd_1e-4_b32__chirplet_fm_32_32_5_meso.tsv")),
        # ("saga_w0", os.path.join(OUT_DIR, f"chirplet/fm/scrapl_saga_w0_sgd_1e-4_b32__chirplet_fm_32_32_5_meso.tsv")),
    ]
    # stage = "train"
    stage = "val"
    # stage = "test"
    x_col = "step"
    # x_col = "global_n"
    # y_col = "l1_theta"
    y_col = "l1_d"
    # y_col = "l1_s"

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"{stage} {y_col}")
    for name, tsv_path in tsv_names_and_paths:
        log.info(f"Plotting {name}, stage: {stage} ===================================")
        data = prepare_tsv_data(tsv_path, stage, x_col, y_col, y_converge_val=0.1, allow_var_n=True)
        plot_xy_vals(ax, data, title=name, plot_95ci=True, plot_range=False)
        tsv_vals = [name] + data["tsv_vals"]
        tsv_string = "\t".join(str(val) for val in tsv_vals)
        print(f"{tsv_string}")

    if stage != "test":
        plt.show()
