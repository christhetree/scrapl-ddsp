import itertools
import logging
import math
import os
from collections import defaultdict
from typing import Dict, List, Callable, Optional, Any, Tuple

import numpy as np
import torch as tr
import yaml
from matplotlib import pyplot as plt
from torch import Tensor as T

from experiments.losses import AdaptiveSCRAPLLoss
from experiments.paths import OUT_DIR, CONFIGS_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def collect_indices(
    meta: Dict[str, Any],
    J1: Optional[List[int]] = None,
    J2: Optional[List[int]] = None,
    J_fr: Optional[List[int]] = None,
    spin: Optional[List[int]] = None,
    orders: Optional[List[int]] = None,
) -> List[int]:
    use_all_J1 = False
    meta_j = meta["j"]
    meta_j_fr = meta["j_fr"]
    meta_spin = meta["spin"]
    meta_order = meta["order"]
    if not J1:
        use_all_J1 = True
        J1 = np.unique(meta_j[:, 0])
        J1 = J1[~np.isnan(J1)].astype(int).tolist()
        log.debug(f"Setting J1 to {J1}")
    else:
        log.debug(f"J1 = {J1}")
    if not J2:
        J2 = np.unique(meta_j[:, 1])
        J2 = J2[~np.isnan(J2)].astype(int).tolist()
        log.debug(f"Setting J2 to {J2}")
    else:
        log.debug(f"J2 = {J2}")
    if not J_fr:
        J_fr = np.unique(meta_j_fr)
        J_fr = J_fr[~np.isnan(J_fr)].astype(int).tolist()
        log.debug(f"Setting J_fr to {J_fr}")
    else:
        log.debug(f"J_fr = {J_fr}")
    if not spin:
        spin = np.unique(meta_spin)
        spin = spin[~np.isnan(spin)].astype(int).tolist()
        log.debug(f"Setting spin to {spin}")
    else:
        log.debug(f"spin = {spin}")
    if not orders:
        orders = [0, 1, 2]
        log.debug(f"Setting orders to {orders}")
    else:
        log.debug(f"orders = {orders}")

    indices = []

    n_paths = len(meta_order)
    for idx in range(n_paths):
        curr_order = int(meta_order[idx])
        if curr_order == 0:
            if 0 in orders:
                indices.append(idx)
            continue
        curr_J1 = meta_j[idx, 0]
        curr_J2 = meta_j[idx, 1]
        curr_J_fr = meta_j_fr[idx]
        assert not np.isnan(curr_J_fr)
        curr_J_fr = int(curr_J_fr)
        curr_spin = meta_spin[idx]
        assert not np.isnan(curr_spin)
        curr_spin = int(curr_spin)
        if curr_order == 1 and 1 in orders:
            assert np.isnan(curr_J2)
            # format is joint so J1 is mushed together for each path
            if np.isnan(curr_J1) and use_all_J1:
                if curr_J_fr in J_fr and curr_spin in spin:
                    indices.append(idx)
                    continue
            if (
                not np.isnan(curr_J1)
                and int(curr_J1) in J1
                and curr_J_fr in J_fr
                and curr_spin in spin
            ):
                indices.append(idx)
                continue
        if curr_order == 2 and 2 in orders:
            assert not np.isnan(curr_J2)
            if np.isnan(curr_J1) and use_all_J1:
                if int(curr_J2) in J2 and curr_J_fr in J_fr and curr_spin in spin:
                    indices.append(idx)
                    continue
            if (
                not np.isnan(curr_J1)
                and int(curr_J1) in J1
                and int(curr_J2) in J2
                and curr_J_fr in J_fr
                and curr_spin in spin
            ):
                indices.append(idx)
                continue

    for idx in indices:
        log.debug(
            f"idx = {idx:3}, order = {meta_order[idx]}, "
            f"J1 = {meta_j[idx, 0]:3.0f}, "
            f"J2 = {meta_j[idx, 1]:3.0f}, "
            f"J_fr = {meta_j_fr[idx]:2.0f}, "
            f"spin = {meta_spin[idx]:2.0f}"
        )
    log.debug(f"n_indices / n_paths = {len(indices)} / {n_paths}")
    return indices


def make_segmentation_indices(
    seg_axis: str, meta: Dict[str, Any], name: str
) -> List[Tuple[str, List[int]]]:
    orders = None  # TODO(cm): caution when removing this
    if seg_axis == "J1":
        J1 = np.unique(meta["j"][:, 0])
        J1 = J1[~np.isnan(J1)].astype(int).tolist()
        segments = J1
    elif seg_axis == "J2":
        J2 = np.unique(meta["j"][:, 1])
        J2 = J2[~np.isnan(J2)].astype(int).tolist()
        segments = J2
        orders = [2]
    elif seg_axis == "J_fr":
        J_fr = np.unique(meta["j_fr"])
        J_fr = J_fr[~np.isnan(J_fr)].astype(int).tolist()
        segments = J_fr
        orders = [2]
    elif seg_axis == "spin":
        spin = np.unique(meta["spin"])
        spin = spin[~np.isnan(spin)].astype(int).tolist()
        segments = spin
        orders = [2]
    elif seg_axis == "orders":
        orders = np.unique(meta["order"])
        orders = orders[~np.isnan(orders)].astype(int).tolist()
        segments = orders
    else:
        raise ValueError(f"Unknown segmentation axis: {seg_axis}")
    log.debug(f"{seg_axis} segments = {segments}")
    seg_indices = []
    for seg in segments:
        indices = collect_indices(meta, **{seg_axis: [seg], "orders": orders})
        if indices:
            seg_indices.append((f"{name}__{seg_axis}_{seg}", indices))
    return seg_indices


def process_path_dict(
    data: Dict[int, List[float]],
    name: str,
    n_paths: int = 315,
    n_bins: int = 100,
    min_val: Optional[float] = None,
    agg_func: Callable[[List[float]], float] = np.mean,
    plt_path_counts: bool = False,
    plt_vals: bool = False,
    plt_dist: bool = False,
    plt_probs: bool = True,
    plt_log: bool = False,
) -> T:
    # vals = list(itertools.chain(*data.values()))
    # m_s = []
    # for idx, vals in data.items():
    #     plt.plot(vals)
    #     # Plot line of best fit
    #     x = np.arange(len(vals))
    #     y = np.array(vals)
    #     m, b = np.polyfit(x, y, 1)
    #     log.info(f"{idx} m = {m}")
    #     m_s.append(m)
    #     plt.plot(x, m * x + b)
    #     plt.title(f"{name} path {idx}")
    #     plt.show()
    #     derp = 1
    # log.info(f"mean m = {np.mean(m_s)}")
    # log.info(f"std m = {np.std(m_s)}")
    # exit()

    if plt_path_counts:
        bar_heights = [len(v) for v in data.values()]
        plt.bar(list(data.keys()), bar_heights)
        plt.title(f"{name} path counts")
        plt.show()

    if plt_vals:
        vals = list(itertools.chain(*data.values()))
        plt.hist(vals, bins=n_bins)
        plt.title(f"{name} vals")
        plt.show()
        if plt_log:
            log_vals = [math.log10(v) for v in vals]
            plt.hist(log_vals, bins=n_bins)
            plt.title(f"{name} log10 vals")
            plt.show()

    dist = tr.zeros((n_paths,))
    for path_idx, vals in data.items():
        val = agg_func(vals)
        dist[path_idx] = val
    log_dist = tr.log10(dist)
    log.info(f"dist.min() = {dist.min()}")
    log.info(f"dist.max() = {dist.max()}")

    probs = dist / dist.sum()
    log.info(f"probs.min() = {probs.min():.6f}")
    log.info(f"probs.max() = {probs.max():.6f}")

    if min_val is None:
        log_probs = log_dist - log_dist.min()
    else:
        log_probs = tr.clip(log_dist - tr.log10(tr.tensor(min_val)), min=0.0)
    log_probs = log_probs / log_probs.sum()
    log.info(f"log_probs.min() = {log_probs.min():.6f}")
    log.info(f"log_probs.max() = {log_probs.max():.6f}")

    if plt_dist:
        plt.plot(dist.numpy())
        plt.title(f"{name} dist")
        plt.show()
        if plt_log:
            plt.plot(log_dist.numpy())
            plt.title(f"{name} log10 dist")
            plt.show()

    if plt_probs:
        # plt.plot(probs.numpy())
        plt.bar(range(probs.size(0)), probs.numpy())
        plt.title(f"{name} probs")
        plt.show()
        if plt_log:
            # plt.plot(log_probs.numpy())
            plt.bar(range(log_probs.size(0)), log_probs.numpy())
            plt.title(f"{name} log10 probs")
            plt.show()

    return probs


if __name__ == "__main__":
    config_path = os.path.join(CONFIGS_DIR, "losses/scrapl_adaptive.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    scrapl = AdaptiveSCRAPLLoss(**config["init_args"])
    meta = scrapl.jtfs.meta()

    dir_path = OUT_DIR
    names = [
        "data_meso_t50"
    ]
    # seg_axes = ["J2", "J_fr", "spin", "orders"]
    seg_axes = []
    # seg_axes = ["J2"]
    # seg_axes = ["J_fr"]
    # seg_axes = ["spin"]
    # seg_axes = ["orders"]
    probs_all = []

    for name in names:
        data_path = os.path.join(dir_path, name)

        # dir = data_path
        # paths = [
        #     os.path.join(dir, f)
        #     for f in os.listdir(dir)
        #     if f.endswith(".pt") and f.startswith("dgm")
        # ]
        # data = defaultdict(list)
        # for path in paths:
        #     path_idx = int(path.split("__")[-1].split(".")[0][1:])
        #     grads = tr.load(path).abs().view(-1).tolist()
        #     data[path_idx].extend(grads)

        data = yaml.safe_load(open(data_path, "r"))
        probs = process_path_dict(data, name)
        probs_all.append(probs)

        for seg_axis in seg_axes:
            seg_indices = make_segmentation_indices(seg_axis, meta, name)
            seg_name = f"{name}__{seg_axis}"
            seg_data = defaultdict(list)
            seg_n_paths = len(seg_indices)
            seg_idx_to_label = {}
            for seg_idx, (label, path_indices) in enumerate(seg_indices):
                new_vals = []
                for path_idx in path_indices:
                    if path_idx in data:
                        new_vals.extend(data[path_idx])
                seg_data[seg_idx] = new_vals
                seg_idx_to_label[seg_idx] = label

            process_path_dict(seg_data, seg_name, seg_n_paths)
            log.info(f"============ {seg_axis} ==============")
            for k, v in seg_idx_to_label.items():
                log.info(f"{k}: {v}")

    # for name, probs in zip(names, probs_all):
    #     out_path = os.path.join(OUT_DIR, f"{name}.pt")
    #     tr.save(probs, out_path)
    #
    # mean_probs = tr.mean(tr.stack(probs_all, dim=0), dim=0)
    # log.info(f"mean_probs.sum() = {mean_probs.sum()}")
    # log.info(f"mean_probs.min() = {mean_probs.min()}")
    # log.info(f"mean_probs.max() = {mean_probs.max()}")
    # out_path = os.path.join(OUT_DIR, f"mean_probs.pt")
    # tr.save(mean_probs, out_path)
    #
    # plt.bar(range(mean_probs.size(0)), mean_probs.numpy())
    # plt.title("mean_probs")
    # plt.show()
