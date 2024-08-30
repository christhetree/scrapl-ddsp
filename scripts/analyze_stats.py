import itertools
import logging
import math
import os
from typing import Dict, List, Callable, Optional

import numpy as np
import torch as tr
import yaml
from matplotlib import pyplot as plt

from experiments.paths import OUT_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


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
) -> None:
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
        plt.plot(log_dist.numpy())
        plt.title(f"{name} log10 dist")
        plt.show()

    if plt_probs:
        plt.plot(probs.numpy())
        # plt.bar(range(probs.size(0)), probs.numpy())
        plt.title(f"{name} probs")
        plt.show()
        plt.plot(log_probs.numpy())
        # plt.bar(log_probs.numpy(), log_probs.numpy())
        plt.title(f"{name} log10 probs")
        plt.show()


if __name__ == "__main__":
    dir_path = os.path.join(OUT_DIR, "out")
    names = [
        # "micro_p2__d_grads.yml",
        # "micro_p2__s_grads.yml",
        # "meso_p2__d_grads.yml",
        "meso_p2__s_grads.yml",
    ]
    for name in names:
        data_path = os.path.join(dir_path, name)
        data = yaml.safe_load(open(data_path, "r"))
        process_path_dict(data, name)
