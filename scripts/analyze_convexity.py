import logging
import os

import numpy as np
import torch as tr
from torch import Tensor as T
from tqdm import tqdm

from experiments.gradients import plot_gradients
from experiments.paths import OUT_DIR, DATA_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def measure_convexity(grads: T) -> float:
    assert grads.ndim == 3
    points = []
    for i in range(grads.size(0)):
        for j in range(grads.size(1)):
            point = (tr.tensor([i, j]).float(), grads[i, j, :])
            points.append(point)

    # Compare every pair of points
    convexities = []
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
                convexities.append(0.0)
    overall_convexity = np.mean(convexities)
    return overall_convexity


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

        d_grad_mag = d_grad.abs()
        s_grad_mag = s_grad.abs()
        z_score_d = (d_grad_mag - d_grad_mag.mean()) / d_grad_mag.std()
        z_score_s = (s_grad_mag - s_grad_mag.mean()) / s_grad_mag.std()
        sim_d = z_score_d.mean().item()
        sim_s = z_score_s.mean().item()
        # sim = min(sim_d, sim_s)
        sim = max(sim_d, sim_s)

        d_grad = d_grad.mean(dim=0)
        s_grad = s_grad.mean(dim=0)

        grad = tr.stack([d_grad, s_grad], dim=2)
        # all_grads.append(grad / grad.norm(dim=2, keepdim=True))

        # max_vals = tr.amax(grad.abs(), dim=(0, 1), keepdim=True)
        # grad = grad / max_vals
        grad_mags = grad.norm(dim=2)
        # grad = grad / grad.norm(dim=2, keepdim=True)

        # sim = F.cosine_similarity(grad, target_angles, dim=2)
        # sim[d_target_idx, s_target_idx] = -1.0  # Penalize large gradient at target

        # sim = sim * grad_mags
        # sim = sim.mean().item()

        # grad_d = grad.clone()
        # grad_d[:, :, 1] = 0
        # sim_d = measure_convexity(grad_d)
        # grad_s = grad.clone()
        # grad_s[:, :, 0] = 0
        # sim_s = measure_convexity(grad_s)
        # sim = (sim_d + sim_s) / 2.0
        # sim = min(sim_d, sim_s)

        # sim = measure_convexity(grad)
        # sim = 0
        # sim = sim_d
        # sim = grad_mags.mean().item()
        # grad_2 = grad / grad_mags.max()
        # grad_mags_2 = grad_2.norm(dim=2)
        # sim = grad_mags_2.mean().item()
        grads.append((path_idx, sim, grad))
        # grads.append((path_idx, sim_d, sim_s))

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
    for path_idx, sim, _ in grads:
        log.info(f"Path {path_idx} has similarity {sim}")
    # for path_idx, sim_d, sim_s in grads:
    #     log.info(f"Path {path_idx} has similarity {sim_d}, {sim_s}")

    # for path_idx, sim, grad in grads[:10]:
    for path_idx, sim, grad in grads[-10:]:
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
    # prob += 1.0
    prob -= prob.min()
    prob /= prob.sum()
    log.info(f"max prob: {prob.max():.6f}, min prob: {prob.min():.6f}")

    # out_path = os.path.join(OUT_DIR, "micro__convex_min0.pt")  # 0.007319
    # out_path = os.path.join(OUT_DIR, "micro__convex_min0_d.pt")  # 0.005064
    # out_path = os.path.join(OUT_DIR, "meso_t50__convex_min0.pt")  # 0.005341
    # out_path = os.path.join(OUT_DIR, "meso_t50__convex_min0_d.pt")  # 0.004509
    # out_path = os.path.join(OUT_DIR, "meso_t50__convex_min0_s.pt")  # 0.007498
    # out_path = os.path.join(OUT_DIR, "meso_t50__convex_min0_sd_min.pt")  # 0.007015
    # out_path = os.path.join(OUT_DIR, "meso_t50__z_min_ds_min0.pt")  # 0.007102
    # out_path = os.path.join(OUT_DIR, "meso_t50__z_abs_min_ds_min0.pt")  # 0.005446
    # tr.save(prob, out_path)
