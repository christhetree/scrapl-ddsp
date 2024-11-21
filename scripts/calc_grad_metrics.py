import logging
import os
from collections import defaultdict
from typing import List, Callable

import torch as tr
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import Tensor as T

from experiments.losses import AdaptiveSCRAPLLoss
from experiments.paths import OUT_DIR, CONFIGS_DIR
from experiments.util import target_range_softmax

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def reduce(x: T, reduction: str = "mean") -> T:
    if reduction == "mean":
        return x.mean()
    elif reduction == "max":
        return x.max()
    elif reduction == "median":
        return x.median()
    else:
        raise ValueError(f"Unknown reduction {reduction}")


def calc_lc(
    a: T, b: T, grad_a: T, grad_b: T, elementwise: bool = False, eps: float = 1e-8
) -> T:
    assert a.shape == b.shape == grad_a.shape == grad_b.shape
    delta_coord = tr.abs(a - b)
    delta_grad = tr.abs(grad_a - grad_b)
    # assert (delta_coord > eps).all()
    # assert (delta_grad > eps).all()
    if elementwise:
        lc = delta_grad / (delta_coord + eps)
    else:
        a = a.flatten()
        b = b.flatten()
        grad_a = grad_a.flatten()
        grad_b = grad_b.flatten()
        delta_coord = (a - b).norm(p=2)
        delta_grad = (grad_a - grad_b).norm(p=2)
        lc = delta_grad / (delta_coord + eps)
    return lc


def calc_convexity(a: T, b: T, grad_a: T, grad_b: T, elementwise: bool = False) -> T:
    assert a.shape == b.shape == grad_a.shape == grad_b.shape
    if not elementwise:
        a = a.flatten()
        b = b.flatten()
        grad_a = grad_a.flatten()
        grad_b = grad_b.flatten()
    delta_coord = b - a
    delta_grad = grad_b - grad_a
    if elementwise:
        convexity = delta_grad * delta_coord
    else:
        convexity = tr.dot(delta_grad, delta_coord)
    convexity[convexity > 0] = 1.0
    convexity[convexity < 0] = -1.0
    convexity[convexity == 0] = 0.0
    return convexity


def calc_pairwise_metric(
    coords: List[T],
    grads: List[T],
    metric_fn: Callable[[T, T, T, T, bool], T],
    reduction: str = "max",
    elementwise: bool = False,
    compare_adj_only: bool = False,
) -> float:
    assert len(coords) == len(grads)
    assert reduction in {"mean", "max", "median"}
    metrics = []
    if compare_adj_only:
        for i in range(1, len(coords)):
            a = coords[i - 1]
            b = coords[i]
            grad_a = grads[i - 1]
            grad_b = grads[i]
            metric = metric_fn(a, b, grad_a, grad_b, elementwise)
            if elementwise:
                metric = reduce(metric, reduction)
            metrics.append(metric)
    else:
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                a = coords[i]
                b = coords[j]
                grad_a = grads[i]
                grad_b = grads[j]
                metric = metric_fn(a, b, grad_a, grad_b, elementwise)
                if elementwise:
                    metric = reduce(metric, reduction)
                metrics.append(metric)

    # log.info(f"Number of metrics: {len(metrics)}")
    metric = tr.stack(metrics)
    metric = reduce(metric, reduction)
    return metric.item()


def calc_mag_entropy(x: T, eps: float = 1e-12) -> T:
    x = x.abs() + eps
    x /= x.sum()
    entropy = -x * tr.log(x)
    entropy = entropy.sum()
    entropy /= tr.log(tr.tensor(x.numel()))
    assert not tr.isnan(entropy).any()
    return entropy


if __name__ == "__main__":
    scrapl_config_path = os.path.join(CONFIGS_DIR, "losses/scrapl_adaptive.yml")
    with open(scrapl_config_path, "r") as f:
        scrapl_config = yaml.safe_load(f)
    scrapl = AdaptiveSCRAPLLoss(**scrapl_config["init_args"])
    subset_indices = scrapl.enabled_path_indices
    subset_indices_compl = [
        idx for idx in range(scrapl.n_paths) if idx not in subset_indices
    ]

    n_paths = scrapl.n_paths
    sampling_factor = 5
    # metric_name = "norm"
    metric_name = "lc"
    # metric_name = "conv"
    # metric_name = "ent"

    # reduction = "mean"
    reduction = "median"
    # reduction = "max"

    # elementwise = True
    elementwise = False
    # compare_adj_only = True
    compare_adj_only = False

    max_t = None
    # max_t = 500

    dir_path = OUT_DIR
    name = "scrapl_saga_sgd_1e-4_b16__chirplet_32_32_5_only_fm_meso"
    data_path = os.path.join(dir_path, name)
    # grad_id = "__g_raw_"
    grad_id = "__g_adam_"
    # grad_id = "__g_saga_"

    dir = data_path
    paths = [
        os.path.join(dir, f)
        for f in os.listdir(dir)
        if f.endswith(".pt") and grad_id in f
    ]
    log.info(f"Found {len(paths)} files")
    data = defaultdict(lambda: defaultdict(list))
    for path in tqdm(paths):
        weight_path = path.replace(grad_id, "__w_")
        param_idx = int(path.split("_")[-3])
        t = int(path.split("_")[-2])
        path_idx = int(path.split("_")[-1].split(".")[0])
        grad = tr.load(path, map_location=tr.device("cpu")).detach()
        weight = tr.load(weight_path, map_location=tr.device("cpu")).detach()
        if max_t is not None and t > max_t:
            continue
        else:
            data[path_idx][param_idx].append((t, weight, grad))

    metrics = defaultdict(lambda: {})
    for path_idx, param_data in tqdm(data.items()):
        for param_idx, t_data in param_data.items():
            if len(t_data) < 1:
                log.warning(f"Not enough data for path_idx = {path_idx}, "
                            f"param_idx = {param_idx}, len(t_data) = {len(t_data)}")
            # assert len(t_data) > 1, f"path_idx = {path_idx}, param_idx = {param_idx}"
            if len(t_data) > 1:
                t_data = sorted(t_data, key=lambda x: x[0])
                weights = [x[1] for x in t_data]
                grads = [x[2] for x in t_data]

                if metric_name == "norm":
                    vals = [g.flatten().norm(p=2) for g in grads]
                    metric = tr.stack(vals).mean()
                elif metric_name == "ent":
                    vals = [calc_mag_entropy(g) for g in grads]
                    metric = tr.stack(vals).mean()
                else:
                    if metric_name == "lc":
                        metric_fn = calc_lc
                    else:
                        metric_fn = calc_convexity
                    metric = calc_pairwise_metric(
                        weights,
                        grads,
                        metric_fn=metric_fn,
                        reduction=reduction,
                        elementwise=elementwise,
                        compare_adj_only=compare_adj_only,
                    )
                metrics[path_idx][param_idx] = metric

    del data
    param_indices = {k for path_idx in metrics for k in metrics[path_idx]}
    logits_all = []
    for param_idx in param_indices:
        logits = tr.zeros((n_paths,))
        for path_idx in range(n_paths):
            # assert param_idx in metrics[path_idx]
            if param_idx in metrics[path_idx]:
                logits[path_idx] = metrics[path_idx][param_idx]
        logits_all.append(logits)

        # plt.bar(range(logits.size(0)), logits.numpy())
        # plt.title(
        #     f"{metric_name} p{param_idx} ({reduction}, elem {elementwise}, adj {compare_adj_only})"
        # )
        # plt.show()

    assert len(logits_all) == 1
    logits = tr.stack(logits_all, dim=0).mean(dim=0)

    # plt.bar(range(logits.size(0)), logits.numpy())
    # plt.title(
    #     f"{grad_id} logits {metric_name} ({reduction}, elem {elementwise}, adj {compare_adj_only})"
    # )
    # plt.show()

    uniform_prob = 1 / n_paths
    target_min_prob = uniform_prob / sampling_factor
    target_max_prob = uniform_prob * sampling_factor
    target_range = target_max_prob - target_min_prob
    prob = target_range_softmax(logits, target_range=target_range)
    prob -= prob.min()
    prob += target_min_prob

    # plt.plot(range(prob.size(0)), prob.numpy())
    # plt.ylim(0, (sampling_factor + 0.5) * uniform_prob)
    # plt.title(f"prob {metric_name} ({reduction}, elem {elementwise}, adj {compare_adj_only})")
    # plt.show()
    colors = ["r" if idx in subset_indices else "b" for idx in range(n_paths)]
    plt.bar(range(prob.size(0)), prob.numpy(), color=colors)
    plt.ylim(0, (sampling_factor + 0.5) * uniform_prob)
    plt.title(
        f"{grad_id} prob {metric_name} ({reduction}, elem {elementwise}, adj {compare_adj_only})"
    )
    plt.show()

    vals = [(idx, p.item()) for idx, p in enumerate(prob)]
    vals = sorted(vals, key=lambda x: x[1])
    sorted_indices, vals = zip(*vals)

    plt.plot(range(len(vals)), vals, label="prob", color="b")
    plt.plot(
        range(len(vals)),
        [uniform_prob] * len(vals),
        linestyle="--",
        label="uniform",
        color="orange",
    )
    # Add red dot for subset indices
    for idx in subset_indices:
        sorted_idx = sorted_indices.index(idx)
        plt.plot(sorted_idx, prob[idx].item(), "rd")
    plt.yscale("log")
    plt.title(
        f"{grad_id} sorted {metric_name} ({reduction}, elem {elementwise}, adj {compare_adj_only})"
    )
    plt.legend()
    plt.show()

    log.info(
        f"uniform_prob = {uniform_prob:.6f}, "
        f"target_min_prob = {target_min_prob:.6f}, "
        f"target_max_prob = {target_max_prob:.6f}"
    )
    log.info(
        f"Min prob: {prob.min().item():.6f}, "
        f"Max prob: {prob.max().item():.6f}, "
        f"Mean prob: {prob.mean().item():.6f}"
    )
    out_path = os.path.join(
        OUT_DIR,
        f"{name}_{metric_name}_{reduction}"
        f"_elem_{str(elementwise)[0]}"
        f"_adj_{str(compare_adj_only)[0]}_{sampling_factor}x.pt",
    )

    # log.info(f"Saving to {out_path}")
    # tr.save(prob, out_path)
