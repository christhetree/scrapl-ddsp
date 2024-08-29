import itertools
import logging
import math
import os
import torch as tr

import numpy as np
import yaml
from matplotlib import pyplot as plt

from experiments import util
from experiments.paths import OUT_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    dir = os.path.join(OUT_DIR, "out")
    # prefix = "uniform_warmup_micro_p2"
    # prefix = "uniform_warmup_meso_p2"
    prefix = "meso_p2_energy_probs"

    path_counts = yaml.safe_load(
        open(os.path.join(dir, f"{prefix}__path_counts.yml"), "r")
    )
    path_count_vals = []
    for path_idx, n in path_counts.items():
        path_count_vals.extend([path_idx] * n)
    log.info(f"number of path counts = {len(path_count_vals)}")
    n_bins = max(path_counts.keys()) + 1
    plt.hist(path_count_vals, bins=n_bins)
    plt.title(f"Histogram of path counts for {prefix}")
    plt.show()

    n_bins = 100
    energies = yaml.safe_load(
        open(os.path.join(dir, f"{prefix}__target_energies.yml"), "r")
    )
    energy_vals = list(itertools.chain(*energies.values()))
    log.info(f"number of energy values = {len(energy_vals)}")
    log_energy_vals = [math.log10(x) for x in energy_vals]

    # plt.hist(energy_vals, bins=n_bins)
    # plt.title(f"Histogram of energy values for {prefix}")
    # plt.show()

    # plt.hist(log_energy_vals, bins=n_bins)
    # plt.title(f"Histogram of log energy values for {prefix}")
    # plt.show()

    mean_energies = [np.mean(x) for x in energies.values()]
    log_mean_energies = [math.log10(x) for x in mean_energies]
    # Plot bar graph of mean energy values
    # plt.bar(energies.keys(), mean_energies)
    # plt.plot(energies.keys(), mean_energies)
    # plt.title(f"Mean energy values for {prefix}")
    # plt.show()

    # Plot bar graph of log mean energy values
    # plt.bar(energies.keys(), log_mean_energies)
    # plt.plot(energies.keys(), log_mean_energies)
    # plt.title(f"Log mean energy values for {prefix}")
    # plt.show()

    grads = yaml.safe_load(
        open(os.path.join(dir, f"{prefix}__mean_abs_Sx_grads.yml"), "r")
    )
    grad_vals = list(itertools.chain(*grads.values()))
    log.info(f"number of grad values = {len(grad_vals)}")

    # plt.hist(grad_vals, bins=n_bins)
    # plt.title(f"Histogram of grad values for {prefix}")
    # plt.show()

    mean_grads = [np.mean(x) for x in grads.values()]
    log_mean_grads = [math.log10(x) for x in mean_grads]
    # Plot bar graph of mean grad values
    # plt.bar(grads.keys(), mean_grads)
    # plt.plot(grads.keys(), mean_grads)
    # plt.title(f"Mean grad values for {prefix}")
    # plt.show()

    n_paths = 315
    # n_paths = len(energies)

    mean_energy_probs = tr.zeros((n_paths,))
    for path_idx, mean_energy in zip(energies.keys(), mean_energies):
        mean_energy_probs[path_idx] = mean_energy
    mean_energy_probs -= mean_energy_probs.min()
    mean_energy_probs /= mean_energy_probs.sum()
    log.info(f"mean_energy_probs.sum() = {mean_energy_probs.sum()}")
    log.info(f"mean_energy_probs.min() = {mean_energy_probs.min()}")
    log.info(f"mean_energy_probs.max() = {mean_energy_probs.max()}")
    # plt.plot(mean_energy_probs.numpy())
    # plt.title(f"Mean energy probs for {prefix}")
    # plt.show()

    log_mean_energy_probs = tr.zeros((n_paths,))
    for path_idx, log_mean_energy in zip(energies.keys(), log_mean_energies):
        log_mean_energy_probs[path_idx] = log_mean_energy
    log_mean_energy_probs -= log_mean_energy_probs.min()
    log_mean_energy_probs /= log_mean_energy_probs.sum()
    log.info(f"log_mean_energy_probs.sum() = {log_mean_energy_probs.sum()}")
    log.info(f"log_mean_energy_probs.min() = {log_mean_energy_probs.min()}")
    log.info(f"log_mean_energy_probs.max() = {log_mean_energy_probs.max()}")
    # plt.plot(log_mean_energy_probs.numpy())
    # plt.title(f"Log mean energy probs for {prefix}")
    # plt.show()

    mean_grad_probs = tr.zeros((n_paths,))
    for path_idx, mean_grad in zip(grads.keys(), mean_grads):
        mean_grad_probs[path_idx] = mean_grad
    mean_grad_probs -= mean_grad_probs.min()
    mean_grad_probs /= mean_grad_probs.sum()
    log.info(f"mean_grad_probs.sum() = {mean_grad_probs.sum()}")
    log.info(f"mean_grad_probs.min() = {mean_grad_probs.min()}")
    log.info(f"mean_grad_probs.max() = {mean_grad_probs.max()}")
    plt.plot(mean_grad_probs.numpy())
    plt.title(f"Mean grad probs for {prefix}")
    plt.ylim(0, 0.01)
    plt.show()

    out_path = os.path.join(OUT_DIR, f"{prefix}__energy_probs.pt")
    tr.save(mean_energy_probs, out_path)
    out_path = os.path.join(OUT_DIR, f"{prefix}__log_energy_probs.pt")
    tr.save(log_mean_energy_probs, out_path)
    out_path = os.path.join(OUT_DIR, f"{prefix}__grad_probs.pt")
    tr.save(mean_grad_probs, out_path)

    tau = 0.0001
    taued_mean_grad_probs = util.stable_softmax(mean_grad_probs, tau=tau)
    log.info(f"taued_mean_grad_probs.sum() = {taued_mean_grad_probs.sum()}")
    log.info(f"taued_mean_grad_probs.min() = {taued_mean_grad_probs.min()}")
    log.info(f"taued_mean_grad_probs.max() = {taued_mean_grad_probs.max()}")
    plt.plot(taued_mean_grad_probs.numpy())
    plt.title(f"Tau={tau} mean grad probs for {prefix}")
    plt.ylim(0, 0.01)
    plt.show()

    out_path = os.path.join(OUT_DIR, f"{prefix}__grad_probs__tau{tau}.pt")
    tr.save(taued_mean_grad_probs, out_path)
