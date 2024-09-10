import logging
import os

import numpy as np
import torch as tr
from torch import Tensor as T
from tqdm import tqdm

from experiments.gradients import plot_gradients
from experiments.paths import OUT_DIR, DATA_DIR
from experiments.util import target_softmax

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def measure_convexity(grads: T) -> float:
    points = []
    if grads.ndim == 3:
        for i in range(grads.size(0)):
            for j in range(grads.size(1)):
                point = (tr.tensor([i, j]).float(), grads[i, j, :])
                points.append(point)
    elif grads.ndim == 2:
        for i in range(grads.size(0)):
            point = (tr.tensor([i]).float(), grads[i, :])
            points.append(point)
    else:
        raise ValueError(f"grads.ndim must be 2 or 3, but is {grads.ndim}")

    # Compare every pair of points
    convexities = []
    n_zero = 0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p1, g1 = points[i]
            p2, g2 = points[j]
            g_diff = g2 - g1
            p_diff = p2 - p1
            convexity = tr.dot(g_diff, p_diff)
            if convexity > 0:
                convexities.append(1.0)
            elif convexity < 0:
                convexities.append(-1.0)
            else:
                n_zero += 1
                continue
                # convexities.append(0.0)
    # assert n_zero == 324
    overall_convexity = np.mean(convexities)
    return overall_convexity


def estimate_lipschitz_constant(grads: T) -> float:
    points = []
    if grads.ndim == 3:
        for i in range(grads.size(0)):
            for j in range(grads.size(1)):
                point = (tr.tensor([i, j]).float(), grads[i, j, :])
                points.append(point)
    elif grads.ndim == 2:
        for i in range(grads.size(0)):
            point = (tr.tensor([i]).float(), grads[i, :])
            points.append(point)
    else:
        raise ValueError(f"grads.ndim must be 2 or 3, but is {grads.ndim}")

    # Compare every pair of points
    lipschitz_constants = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p1, g1 = points[i]
            p2, g2 = points[j]
            g_diff = (g2 - g1).norm()
            p_diff = (p2 - p1).norm()
            assert p_diff.item() > 0
            lipschitz_constant = g_diff / p_diff
            lipschitz_constants.append(lipschitz_constant)
    lipschitz_constant = tr.stack(lipschitz_constants).max().item()
    # lipschitz_constant = tr.stack(lipschitz_constants).mean().item()
    return lipschitz_constant


if __name__ == "__main__":
    # probs_d = tr.load(os.path.join(DATA_DIR, "meso_t50__convex_min0_d.pt"))
    # probs_s = tr.load(os.path.join(DATA_DIR, "meso_t50__convex_min0_s.pt"))
    # probs = tr.stack([probs_d, probs_s], dim=0).mean(dim=0)
    # log.info(f"probs.sum() = {probs.sum()}")
    # log.info(f"probs.max() = {probs.max()}")
    # log.info(f"probs_d.max() = {probs_d.max()}")
    # log.info(f"probs_s.max() = {probs_s.max()}")
    # log.info(f"probs.min() = {probs.min()}")
    # tr.save(probs, os.path.join(DATA_DIR, "meso_t50__convex_min0_ds.pt"))
    # exit()

    # dir = os.path.join(OUT_DIR, "data_micro")
    dir = os.path.join(OUT_DIR, "data_meso_t50")
    d_paths = [
        os.path.join(dir, f)
        for f in os.listdir(dir)
        if f.endswith(".pt") and f.startswith("dgm")
    ]

    d_target_idx = 4
    s_target_idx = 5
    d_indices = tr.arange(0, 9).float().view(-1, 1).repeat(1, 9)
    s_indices = tr.arange(0, 9).float().view(1, -1).repeat(9, 1)
    grad_indices = tr.stack([d_indices, s_indices], dim=2)

    d_target_indices = tr.full_like(d_indices, d_target_idx)
    s_target_indices = tr.full_like(s_indices, s_target_idx)
    target_indices = tr.stack([d_target_indices, s_target_indices], dim=2)

    target_angles = grad_indices - target_indices
    target_angles = target_angles / target_angles.norm(dim=2, keepdim=True)
    target_angles[target_angles.isnan()] = 0

    # plot_gradients(
    #     theta_density=tr.tensor(d_target_idx),
    #     theta_slope=tr.tensor(s_target_idx),
    #     dist_matrix=target_angles[:, :, 0],
    #     theta_density_hats=tr.arange(0, 9),
    #     theta_slope_hats=tr.arange(0, 9),
    #     dgm=target_angles[:, :, 0],
    #     sgm=target_angles[:, :, 1],
    #     title="Target Gradient",
    # )

    d_grads = []
    s_grads = []
    for d_path in tqdm(d_paths):
        path_idx = int(d_path.split("__")[-1].split(".")[0][1:])
        s_path = d_path.replace("dgm", "sgm")
        d_grad = tr.load(d_path)
        s_grad = tr.load(s_path)
        d_grads.append(d_grad)
        s_grads.append(s_grad)

    d_grads = tr.stack(d_grads, dim=0)
    s_grads = tr.stack(s_grads, dim=0)
    d_mag_mean = d_grads.abs().mean()
    s_mag_mean = s_grads.abs().mean()
    d_mag_std = d_grads.abs().std()
    s_mag_std = s_grads.abs().std()

    all_grads = []
    grads = []
    for d_path in tqdm(d_paths):
        path_idx = int(d_path.split("__")[-1].split(".")[0][1:])
        s_path = d_path.replace("dgm", "sgm")
        d_grad = tr.load(d_path)
        s_grad = tr.load(s_path)

        # mean_mag_d = d_grad.abs().mean()
        # d_normed = d_grad / mean_mag_d
        # sim_d = d_normed.abs().mean().item()
        # mean_mag_s = s_grad.abs().mean()
        # s_normed = s_grad / mean_mag_s
        # sim_s = s_normed.abs().mean().item()
        # sim = min(sim_d, sim_s)

        # z_score_d = (d_grad - d_grad.mean()) / d_grad.std()
        # z_score_s = (s_grad - s_grad.mean()) / s_grad.std()
        # sim_d = z_score_d.abs().mean().item()
        # sim_s = z_score_s.abs().mean().item()
        # sim = min(sim_d, sim_s)

        # d_grad_mag = d_grad.abs()
        # s_grad_mag = s_grad.abs()
        # z_score_d = (d_grad_mag - d_grad_mag.mean()) / d_grad_mag.std()
        # z_score_s = (s_grad_mag - s_grad_mag.mean()) / s_grad_mag.std()
        # # z_score_d = (d_grad_mag - d_mag_mean) / d_mag_std
        # # z_score_s = (s_grad_mag - s_mag_mean) / s_mag_std
        # sim_d = z_score_d.mean().item()
        # sim_s = z_score_s.mean().item()
        # sim = min(sim_d, sim_s)
        # sim = max(sim_d, sim_s)

        # grad = tr.stack([d_grad, s_grad], dim=3)
        # grad_abs = grad.abs()
        # grad_sums = grad_abs.sum(dim=3, keepdim=True)
        # ayy = grad_sums.min()
        # grad_abs = grad_abs / grad_sums
        # ayy2 = grad_abs.sum(dim=3, keepdim=True)
        # entropy = -grad_abs * tr.log(grad_abs)
        # entropy = entropy.sum(dim=3)
        # entropy = entropy / tr.log(tr.tensor(2.0))
        # sim = entropy.mean().item()

        # grad = d_grad.flatten().abs()
        # grad = d_grad.mean(dim=0).flatten().abs()
        # grad = grad / grad.sum()
        # entropy = -grad * tr.log(grad)
        # n = grad.numel()
        # entropy = entropy.sum()
        # entropy = entropy / tr.log(tr.tensor(n))
        # sim = entropy.item()

        d_grad = d_grad.mean(dim=0)
        s_grad = s_grad.mean(dim=0)

        # sim_d = measure_convexity(d_grad)
        # sim_s = measure_convexity(s_grad)
        # sim = min(sim_d, sim_s)

        grad = tr.stack([d_grad, s_grad], dim=2)
        grad_mags = grad.norm(dim=2)

        # grad = grad / grad.norm(dim=-1).max()
        # sim = estimate_lipschitz_constant(grad)

        sim = measure_convexity(grad)

        # grad_d = grad.clone()
        # grad_d[:, :, 1] = 0
        # sim_d = measure_convexity(grad_d)
        # grad_s = grad.clone()
        # grad_s[:, :, 0] = 0
        # sim_s = measure_convexity(grad_s)
        # sim = min(sim_d, sim_s)
        # sim = max(sim_d, sim_s)

        grads.append((path_idx, sim, grad))

    # mean_grad = tr.stack(all_grads, dim=0).mean(dim=0)
    # mean_grad = mean_grad / mean_grad.abs().max()
    # # mean_grad = mean_grad / mean_grad.norm(dim=2, keepdim=True)
    # log.info(f"mean_grad.abs().max() = {mean_grad.abs().max()}")
    # plot_gradients(
    #     theta_density=tr.tensor(d_target_idx),
    #     theta_slope=tr.tensor(s_target_idx),
    #     dist_matrix=mean_grad.norm(dim=2),
    #     theta_density_hats=tr.arange(0, 9),
    #     theta_slope_hats=tr.arange(0, 9),
    #     dgm=mean_grad[:, :, 0],
    #     sgm=mean_grad[:, :, 1],
    #     title="Mean Gradient",
    # )
    # exit()

    grads = sorted(grads, key=lambda x: x[1])
    for idx, (path_idx, sim, _) in enumerate(grads):
        log.info(f"{idx:3} Path {path_idx} has similarity {sim:.6f}")

    for path_idx, sim, grad in grads[1:1]:
        # for path_idx, sim, grad in grads[:10]:
        # for path_idx, sim, grad in grads[-10:]:
        # for path_idx, sim, grad in grads[140:155]:
        # if path_idx in {262, 198, 192, 313, 105}:  # data_meso_t50 min
        # if path_idx in {229, 190}:  # data_micro
        # if path_idx in {107, 68, 190, 193, 291}:  # data_micro density
        # if path_idx in {244, 196, 119, 51, 128}:  # data_micro slope
        # if path_idx in {267, 233}:  # data_meso_t50
        # if path_idx in {143, 94, 218, 249, 235}:  # data_meso_t50 density

        # grad = grad / grad.norm(dim=2, keepdim=True) / 2.0
        # grad[:, :, 1] = 0
        grad /= grad.abs().max()
        grad_mags = grad.norm(dim=2)
        plot_gradients(
            theta_density=tr.tensor(d_target_idx),
            theta_slope=tr.tensor(s_target_idx),
            dist_matrix=grad_mags,
            theta_density_hats=tr.arange(0, 9),
            theta_slope_hats=tr.arange(0, 9),
            dgm=grad[:, :, 0],
            sgm=grad[:, :, 1],
            title=f"Path {path_idx}",
        )
        # exit()

    prob = tr.zeros((315,))
    for path_idx, sim, _ in grads:
        prob[path_idx] = sim
    prob_2 = prob.clone()
    # prob += 1.0
    prob -= prob.min()
    prob /= prob.sum()

    # max_prob = prob.max().item()
    sampling_factor = 4
    uniform_prob = 1 / 315
    target_min_prob = uniform_prob / sampling_factor
    target_max_prob = uniform_prob * sampling_factor
    log.info(
        f"uniform_prob = {uniform_prob:.6f}, "
        f"target_min_prob = {target_min_prob:.6f}, "
        f"target_max_prob = {target_max_prob:.6f}"
    )
    target_range = target_max_prob - target_min_prob
    prob_2 = target_softmax(prob_2, max_prob=target_range)
    prob_2 -= prob_2.min()
    prob_2 += target_min_prob

    pairs = []
    pairs_2 = []
    for idx, p in enumerate(prob):
        pairs.append((idx, p.item()))
        pairs_2.append((idx, prob_2[idx].item()))

    pairs = sorted(pairs, key=lambda x: x[1])
    pairs_2 = sorted(pairs_2, key=lambda x: x[1])
    for (idx, p), (idx_2, p_2) in zip(pairs, pairs_2):
        assert idx == idx_2
    # plot both on same plot
    import matplotlib.pyplot as plt

    plt.plot([x[1] for x in pairs], label="prob")
    plt.plot([x[1] for x in pairs_2], label="prob_2")
    plt.plot([1 / 315] * 315, linestyle="--", label="uniform")
    plt.legend()
    plt.show()

    log.info(f"max  prob: {prob.max():.6f}, min  prob: {prob.min():.6f}")
    log.info(f"max prob2: {prob_2.max():.6f}, min prob2: {prob_2.min():.6f}")

    # out_path = os.path.join(OUT_DIR, "micro__convex_min0.pt")  # 0.007319
    # out_path = os.path.join(OUT_DIR, "micro__convex_min0_d.pt")  # 0.005064
    # out_path = os.path.join(OUT_DIR, "meso_t50__convex_min0.pt")  # 0.005341
    # out_path = os.path.join(OUT_DIR, "meso_t50__convex_min0_d.pt")  # 0.004509
    # out_path = os.path.join(OUT_DIR, "meso_t50__convex_min0_s.pt")  # 0.007498
    # out_path = os.path.join(OUT_DIR, "meso_t50__convex_min0_sd_min.pt")  # 0.007015
    # out_path = os.path.join(OUT_DIR, "meso_t50__z_min_ds_min0.pt")  # 0.007102
    # out_path = os.path.join(OUT_DIR, "meso_t50__entropy_min0.pt")  # 0.005244
    # out_path = os.path.join(OUT_DIR, "meso_t50__entropy_4x.pt")  # 0.012699
    # out_path = os.path.join(OUT_DIR, "meso_t50__lipschitz_norm_min0.pt")  # 0.006999
    # out_path = os.path.join(OUT_DIR, "meso_t50__entropy_4x_v2.pt")
    # out_path = os.path.join(OUT_DIR, "meso_t50__lc_4x_v2.pt")
    # out_path = os.path.join(OUT_DIR, "meso_t50__lc_mean_4x_v2.pt")
    # out_path = os.path.join(OUT_DIR, "meso_t50__convex_4x_v2.pt")
    # tr.save(prob_2, out_path)
