import logging
import os
from collections import defaultdict
from typing import List, Callable

import torch as tr
import yaml
from matplotlib import pyplot as plt
from torch import Tensor as T
from tqdm import tqdm

from experiments import util
from experiments.paths import OUT_DIR, CONFIGS_DIR, DATA_DIR
from experiments.scrapl_loss import SCRAPLLoss
from experiments.util import stable_softmax

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def reduce(x: T, reduction: str = "mean") -> T:
    if reduction == "mean":
        return x.mean()
    elif reduction == "max":
        return x.max()
    elif reduction == "min":
        return x.min()
    elif reduction == "median":
        return x.median()
    else:
        raise ValueError(f"Unknown reduction {reduction}")


def estimate_lc(
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


def estimate_convexity(
    a: T, b: T, grad_a: T, grad_b: T, elementwise: bool = False
) -> T:
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


def calc_abs_val(g: T) -> T:
    # assert g.numel() == 1
    assert g.shape == (32,)
    val = g.squeeze().abs()
    return val


def calc_norm(g: T, p: int = 2) -> T:
    norm = g.flatten().norm(p=p)
    return norm


def calc_spec_norm(g: T) -> T:
    assert g.ndim <= 2
    if g.ndim < 2:
        spec_norm = g.norm(p=2)
    else:
        S = tr.linalg.svdvals(g)
        spec_norm = S.max()

    spec_norm_2 = tr.linalg.norm(g, ord=2)
    assert spec_norm == spec_norm_2

    norm = calc_norm(g)
    assert spec_norm <= norm
    # log.info(f"norm - spec_norm = {norm - spec_norm}")
    return spec_norm


if __name__ == "__main__":
    # prob_am = tr.load(os.path.join(OUT_DIR, "out/scrapl_saga_w0_sgd_1e-5_b32__texture_32_32_5_meso__d.pt"))
    # prob_fm = tr.load(os.path.join(OUT_DIR, "out/scrapl_saga_w0_sgd_1e-5_b32__texture_32_32_5_meso__s.pt"))
    # prob_am_fm = (prob_am + prob_fm) / 2.0
    # tr.save(prob_am_fm, os.path.join(OUT_DIR, "out/scrapl_saga_w0_sgd_1e-5_b32__texture_32_32_5_meso__ds.pt"))
    # exit()

    scrapl_config_path = os.path.join(CONFIGS_DIR, "losses/chirplet/scrapl_am.yml")
    # scrapl_config_path = os.path.join(CONFIGS_DIR, "losses/chirplet/scrapl_fm.yml")
    # scrapl_config_path = os.path.join(CONFIGS_DIR, "losses/chirplet/scrapl_am_or_fm.yml")

    with open(scrapl_config_path, "r") as f:
        scrapl_config = yaml.safe_load(f)
    init_args = scrapl_config["init_args"]
    Q1 = init_args["Q1"]
    get_path_keys_kw_args = init_args["get_path_keys_kw_args"]
    del init_args["get_path_keys_kw_args"]
    scrapl = SCRAPLLoss(**scrapl_config["init_args"])
    path_keys, _ = util.get_path_keys(meta=scrapl.meta, Q1=Q1, **get_path_keys_kw_args)
    subset_indices = []
    for k in path_keys:
        assert k in scrapl.scrapl_keys
        path_idx = scrapl.scrapl_keys.index(k)
        subset_indices.append(path_idx)
    subset_indices_compl = [
        idx for idx in range(scrapl.n_paths) if idx not in subset_indices
    ]
    log.info(f"len(subset_indices) = {len(subset_indices)}, subset_indices = {subset_indices}")
    exit()

    n_paths = scrapl.n_paths
    # sampling_factor = 0.25
    sampling_factor = 0.0
    uniform_prob = 1 / n_paths
    target_min_prob = uniform_prob * sampling_factor

    # prob = None
    # prob = tr.load(os.path.join(DATA_DIR, "scrapl_saga_w0_sgd_1e-4_b32__chirplet_32_32_5_meso__d.pt"))
    # prob = tr.load(os.path.join(DATA_DIR, "scrapl_saga_w0_sgd_1e-4_b32__chirplet_32_32_5_meso__s.pt"))
    # prob = tr.load(os.path.join(DATA_DIR, "scrapl_saga_w0_sgd_1e-4_b32__chirplet_32_32_5_meso__ds.pt"))
    # prob = tr.load(os.path.join(DATA_DIR, "probs/scrapl_saga_pwa_1e-4_b32__chirplet_32_32_5_meso__probs__n_batches_1__n_iter_20__min_prob_frac_0.0.pt"))
    # prob = tr.load(os.path.join(DATA_DIR, "probs/scrapl_saga_pwa_1e-4_b32__chirplet_32_32_5_meso__probs__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_None.pt"))
    # prob = tr.load(os.path.join(DATA_DIR, "probs/scrapl_saga_pwa_1e-4_b32__chirplet_32_32_5_meso__probs__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_None__multibatch.pt"))
    # prob = tr.load(os.path.join(DATA_DIR, "probs/scrapl_saga_pwa_1e-4_b32__chirplet_32_32_5_meso__probs__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_mean.pt"))
    # prob = tr.load(os.path.join(DATA_DIR, "probs/scrapl_saga_pwa_1e-4_b32__chirplet_32_32_5_meso__probs__n_batches_20__n_iter_20__min_prob_frac_0.0__param_agg_None.pt"))

    prob = tr.load(os.path.join(DATA_DIR, "adaptive_scrapl/scrapl_saga_pwa_1e-5__chirplet_32_32_5_meso_b28_am_lo_fm_lo__probs__n_theta_2__n_params_28__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt"))
    # prob = tr.load(os.path.join(DATA_DIR, "adaptive_scrapl/scrapl_saga_pwa_1e-5__chirplet_32_32_5_meso_b28_am_hi_fm_hi__probs__n_theta_2__n_params_28__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt"))
    # prob = tr.load(os.path.join(DATA_DIR, "adaptive_scrapl/scrapl_saga_pwa_1e-5__chirplet_32_32_5_meso_b28_am_lo_fm_hi__probs__n_theta_2__n_params_28__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt"))
    # prob = tr.load(os.path.join(DATA_DIR, "adaptive_scrapl/scrapl_saga_pwa_1e-5__chirplet_32_32_5_meso_b28_am_hi_fm_lo__probs__n_theta_2__n_params_28__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt"))

    # prob = tr.load(os.path.join(DATA_DIR, "probs/before_seed_and_eval_fix/scrapl_saga_pwa_1e-4_b32__chirplet_32_32_5_meso__log_probs__n_batches_1__n_iter_20__min_prob_frac_0.0.pt"))
    # prob = tr.load(os.path.join(DATA_DIR, "probs/before_seed_and_eval_fix/scrapl_saga_pwa_1e-4_b32__chirplet_32_32_5_meso__log_probs__n_batches_1__n_iter_20__min_prob_frac_0.0__multibatch.pt"))
    # prob = tr.load(os.path.join(DATA_DIR, "probs/scrapl_saga_pwa_1e-4_b32__chirplet_32_32_5_meso__log_probs__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_None__fixed_seed_hat.pt"))
    # prob = tr.load(os.path.join(DATA_DIR, "probs/scrapl_saga_pwa_1e-4_b32__chirplet_32_32_5_meso__log_probs__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_None.pt"))
    # prob = tr.load(os.path.join(DATA_DIR, "probs/scrapl_saga_pwa_1e-4_b32__chirplet_32_32_5_meso__log_probs__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_None__multibatch.pt"))
    # prob = tr.load(os.path.join(DATA_DIR, "probs/scrapl_saga_pwa_1e-4_b32__chirplet_32_32_5_meso__log_probs__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_mean.pt"))
    # prob = tr.load(os.path.join(DATA_DIR, "probs/scrapl_saga_pwa_1e-4_b32__chirplet_32_32_5_meso__log_probs__n_batches_20__n_iter_20__min_prob_frac_0.0__param_agg_None.pt"))
    # prob = prob.exp()
    # prob = prob[0, :]
    # prob = prob[1, :]

    grad_id = "0to1_hat"
    # grad_id = "__g_raw_"
    # grad_id = "__eig1_"
    # grad_id = "__h_"
    # grad_id = "__g_adam_"
    # grad_id = "__g_saga_"

    # metric_name = "norm"
    metric_name = "abs"
    # metric_name = "spec_norm"
    # metric_name = "ent"
    # metric_name = "est_lc"
    # metric_name = "est_conv"

    param_reduction = "mean"
    # param_reduction = "max"
    # param_reduction = "rescaled_mean"
    # param_reduction = "sqrt_numel_scaled"

    # allowed_param_indices = None
    # allowed_param_indices = {15}
    # allowed_param_indices = {"theta_d_0to1_hat"}
    allowed_param_indices = {"theta_s_0to1_hat"}
    # allowed_param_indices = {"theta_d_0to1_hat", "theta_s_0to1_hat"}
    param_indices = allowed_param_indices

    max_t = None
    # max_t = 400

    dir_path = OUT_DIR
    # name = "grads/scrapl_b32_no_do_power__chirplet_32_32_5_meso_am"
    # name = "grads/scrapl_b32_no_do_power__chirplet_32_32_5_meso_fm"
    # name = "grads/scrapl_b32_no_do_power__chirplet_32_32_5_meso"
    name = "grads/scrapl_saga_w0_sgd_1e-5_b32__texture_32_32_5_meso"

    if prob is None:
        assert False
        data_path = os.path.join(dir_path, name)
        param_idx_to_numel = {}
        dir = data_path
        paths = [
            os.path.join(dir, f)
            for f in os.listdir(dir)
            if f.endswith(".pt") and grad_id in f
        ]
        log.info(f"Found {len(paths)} files")
        data = defaultdict(lambda: defaultdict(list))
        for path in tqdm(paths):
            try:
                param_idx = int(path.split("_")[-3])
            except ValueError:
                if allowed_param_indices is None:
                    continue
                if "theta_s_0to1_hat" in path:
                    param_idx = "theta_s_0to1_hat"
                else:
                    assert "theta_d_0to1_hat" in path
                    param_idx = "theta_d_0to1_hat"
            if allowed_param_indices is not None and param_idx not in allowed_param_indices:
                continue
            t = int(path.split("_")[-2])
            if max_t is not None and t > max_t:
                continue
            path_idx = int(path.split("_")[-1].split(".")[0])
            grad = tr.load(path, map_location=tr.device("cpu")).detach()
            if param_idx not in param_idx_to_numel:
                param_idx_to_numel[param_idx] = grad.numel()

            # prev_m = tr.zeros_like(grad)
            # prev_v = tr.zeros_like(grad)
            # # grad *= 1e8
            # grad, _, _ = SCRAPLLightingModule.adam_grad_norm_cont(
            #     grad, prev_m, prev_v, t=1.0, prev_t=0.0
            # )

            # weight_path = path.replace(grad_id, "__w_")
            # weight = tr.load(weight_path, map_location=tr.device("cpu")).detach()
            weight = None
            data[path_idx][param_idx].append((t, weight, grad))

        metrics = defaultdict(lambda: {})
        for path_idx, param_data in tqdm(data.items()):
            for param_idx, t_data in param_data.items():
                # assert len(t_data) > 1, f"path_idx = {path_idx}, param_idx = {param_idx}"
                if len(t_data):
                    t_data = sorted(t_data, key=lambda x: x[0])
                    weights = [x[1] for x in t_data]
                    grads = [x[2] for x in t_data]

                    if metric_name == "abs":
                        vals = [calc_abs_val(g) for g in grads]
                        assert len(vals) == 1
                        metric = tr.stack(vals).mean()
                    elif metric_name == "norm":
                        vals = [calc_norm(g) for g in grads]
                        metric = tr.stack(vals).mean()
                    elif metric_name == "spec_norm":
                        vals = [calc_spec_norm(g) for g in grads]
                        metric = tr.stack(vals).mean()
                    elif metric_name == "ent":
                        vals = [calc_mag_entropy(g) for g in grads]
                        metric = tr.stack(vals).mean()
                    else:
                        raise ValueError(f"Unknown metric {metric_name}")
                    metrics[path_idx][param_idx] = metric
                else:
                    log.warning(
                        f"Not enough data for path_idx = {path_idx}, "
                        f"param_idx = {param_idx}, len(t_data) = {len(t_data)}"
                    )

        del data
        param_indices = {k for path_idx in metrics for k in metrics[path_idx]}
        log.info(f"Param indices: {param_indices}")
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

        # TODO(cm): look into different aggregation techniques
        logits = tr.stack(logits_all, dim=0)
        assert logits.size(0) == 1

        if param_reduction == "mean":
            logits = logits.mean(dim=0)
        elif param_reduction == "max":
            logits, _ = logits.max(dim=0, keepdim=False)
        elif param_reduction == "rescaled_mean":
            logit_sums = logits.sum(dim=1, keepdim=True)
            logit_sums[logit_sums == 0.0] = 1.0
            logits = logits / logit_sums
            logits = logits.mean(dim=0)
        elif param_reduction == "sqrt_numel_scaled":
            for param_idx, numel in param_idx_to_numel.items():
                fac = tr.sqrt(tr.tensor(numel))
                log.info(f"factors[{param_idx}] = {fac}")
                logits[param_idx, :] *= fac
            logits = logits.mean(dim=0)
        assert logits.ndim == 1

        log.info(
            f"logits.min() = {logits.min():.6f}, logits.max() = {logits.max():.6f}, "
            f"logits.median() = {logits.median():.6f} "
            f"logits.mean() = {logits.mean():.6f}, logits.std() = {logits.std():.6f}"
        )

        # plt.bar(range(logits.size(0)), logits.numpy())
        # plt.title(
        #     f"{grad_id} logits {metric_name} ({reduction}, elem {elementwise}, adj {compare_adj_only})"
        # )
        # plt.show()

        scaling_factor = 1.0 - (n_paths * target_min_prob)
        prob = logits / logits.sum() * scaling_factor + target_min_prob

    prob = prob.float()
    assert tr.allclose(prob.sum(), tr.tensor(1.0)), f"prob.sum() = {prob.sum()}"

    # target_range = target_max_prob - target_min_prob
    # prob = target_range_softmax(logits, target_range=target_range)
    # prob -= prob.min()
    # prob += target_min_prob

    subset_prob = prob[subset_indices].sum()
    subset_unif_prob = uniform_prob * len(subset_indices)
    ratio = subset_prob / subset_unif_prob
    greater_than_uniform = len(
        [idx for idx in subset_indices if prob[idx] >= uniform_prob]
    ) / len(subset_indices)

    title = (
        f"{grad_id}, {metric_name}, {param_reduction} "
        f"(r = {ratio:.2f}, SF = {sampling_factor:.2f}, "
        f"> unif = {greater_than_uniform:.2f})"
        f"\n{param_indices}"
    )
    log.info(title)

    # plt.plot(range(prob.size(0)), prob.numpy())
    # plt.ylim(0, (sampling_factor + 0.5) * uniform_prob)
    # plt.title(f"prob {metric_name} ({reduction}, elem {elementwise}, adj {compare_adj_only})")
    # plt.show()

    # colors = ["r" if idx in subset_indices else "b" for idx in range(n_paths)]
    # plt.bar(range(prob.size(0)), prob.numpy(), color=colors)
    # plt.ylim(0, 8 * uniform_prob)
    # plt.title(title)
    # plt.show()

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
    # plt.yscale("log")
    plt.title(title)
    plt.legend()
    plt.show()

    log.info(
        f"uniform_prob = {uniform_prob:.6f}, "
        f"target_min_prob = {target_min_prob:.6f}, "
    )
    log.info(
        f"Min prob: {prob.min().item():.6f}, "
        f"Max prob: {prob.max().item():.6f}, "
        f"Mean prob: {prob.mean().item():.6f}"
    )

    # out_path = os.path.join(
    #     OUT_DIR,
    #     f"{name}_{metric_name}_{param_reduction}_a{sampling_factor}.pt",
    # )
    # log.info(f"Saving to {out_path}")
    # tr.save(prob, out_path)
