import logging
import os

import torch as tr
import torch.nn.functional as F

from experiments.paths import OUT_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    dir = os.path.join(OUT_DIR, "data_micro")
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
    # grad_indices = tr.stack([d_indices], dim=2)

    d_target_indices = tr.full_like(d_indices, d_target_idx)
    s_target_indices = tr.full_like(s_indices, s_target_idx)
    target_indices = tr.stack([d_target_indices, s_target_indices], dim=2)
    # target_indices = tr.stack([d_target_indices], dim=2)

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

    grads = []
    for d_path in d_paths:
        path_idx = int(d_path.split("__")[-1].split(".")[0][1:])
        s_path = d_path.replace("dgm", "sgm")
        d_grad = tr.load(d_path).squeeze(0)
        s_grad = tr.load(s_path).squeeze(0)

        grad = tr.stack([d_grad, s_grad], dim=2)
        # grad = tr.stack([d_grad], dim=2)

        # max_vals = tr.amax(grad.abs(), dim=(0, 1), keepdim=True)
        # grad = grad / max_vals
        grad_mags = grad.norm(dim=2)
        # grad = grad / grad.norm(dim=2, keepdim=True)

        sim = F.cosine_similarity(grad, target_angles, dim=2)
        sim[d_target_idx, s_target_idx] = -1.0  # Penalize large gradient at target

        # sim = sim * grad_mags
        sim = sim.mean().item()
        grads.append((path_idx, sim))

        # if path_idx == -1:
        #     # grad = grad / grad.norm(dim=2, keepdim=True)
        #     grad /= grad.abs().max()
        #     plot_gradients(
        #         theta_density=tr.tensor(d_target_idx),
        #         theta_slope=tr.tensor(s_target_idx),
        #         dist_matrix=grad_mags,
        #         theta_density_hats=tr.arange(0, 9),
        #         theta_slope_hats=tr.arange(0, 9),
        #         dgm=grad[:, :, 0],
        #         sgm=grad[:, :, 1],
        #         title="Path Gradient",
        #     )

    grads = sorted(grads, key=lambda x: x[1])
    for path_idx, sim in grads:
        log.info(f"Path {path_idx} has similarity {sim}")

    prob = tr.zeros((315,))
    for path_idx, sim in grads:
        prob[path_idx] = sim
    prob += 1.0
    # prob -= prob.min()
    prob /= prob.sum()
    log.info(f"max prob: {prob.max()}, min prob: {prob.min()}")

    # out_path = os.path.join(OUT_DIR, "micro_p2__cos_sim.pt")  # max prob: 0.004277152009308338, min prob: 0.0015152201522141695
    # out_path = os.path.join(OUT_DIR, "micro_p2__cos_sim_weighted.pt")  # max prob: 0.012973041273653507, min prob: 0.0
    # out_path = os.path.join(OUT_DIR, "micro_p2__d_cos_sim.pt")  # max prob: 0.003954584710299969, min prob: 0.0011049575405195355
    # out_path = os.path.join(OUT_DIR, "micro_p2__d_cos_sim_weighted.pt")  # max prob: 0.011630410328507423, min prob: 0.0
    # out_path = os.path.join(OUT_DIR, "micro_p2__cos_sim_theta_norm.pt")  # max prob: 0.0038949870504438877, min prob: 0.0014727504458278418
    # out_path = os.path.join(OUT_DIR, "micro_p2__cos_sim_theta_norm_weighted.pt")  # max prob: 0.007348468992859125, min prob: 0.0
    # tr.save(prob, out_path)
