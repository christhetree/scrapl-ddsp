import logging
import os
from typing import List, Any, Union

import torch as tr
import torch.nn.functional as F
from scipy.stats import loguniform
from torch import Tensor as T

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def linear_interpolate_last_dim(x: T, n: int, align_corners: bool = True) -> T:
    n_dim = x.ndim
    assert 1 <= n_dim <= 3
    if x.size(-1) == n:
        return x
    if n_dim == 1:
        x = x.view(1, 1, -1)
    elif n_dim == 2:
        x = x.unsqueeze(1)
    x = F.interpolate(x, n, mode="linear", align_corners=align_corners)
    if n_dim == 1:
        x = x.view(-1)
    elif n_dim == 2:
        x = x.squeeze(1)
    return x


def choice(items: List[Any]) -> Any:
    assert len(items) > 0
    idx = randint(0, len(items))
    return items[idx]


def randint(low: int, high: int, n: int = 1) -> Union[int, T]:
    x = tr.randint(low=low, high=high, size=(n,))
    if n == 1:
        return x.item()
    return x


def sample_uniform(low: float, high: float, n: int = 1) -> Union[float, T]:
    x = (tr.rand(n) * (high - low)) + low
    if n == 1:
        return x.item()
    return x


def sample_log_uniform(low: float, high: float, n: int = 1) -> Union[float, T]:
    # TODO(cm): replace with torch
    if low == high:
        if n == 1:
            return low
        else:
            return tr.full(size=(n,), fill_value=low)
    x = loguniform.rvs(low, high, size=n)
    if n == 1:
        return float(x)
    return tr.from_numpy(x)


def clip_norm(x: T, max_norm: float, p: int = 2, eps: float = 1e-8) -> T:
    total_norm = tr.linalg.vector_norm(x.flatten(), ord=p)
    clip_coef = max_norm / (total_norm + eps)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = tr.clamp(clip_coef, max=1.0)
    x_clipped = x * clip_coef_clamped
    return x_clipped


def stable_softmax(logits: T, tau: float = 1.0) -> T:
    assert tau > 0, f"Invalid temperature: {tau}, must be > 0"
    # Subtract the max logit for numerical stability
    max_logit = tr.max(logits, dim=-1, keepdim=True)[0]
    logits = logits - max_logit
    # Apply temperature scaling
    scaled_logits = logits / tau
    # Compute the exponential
    exp_logits = tr.exp(scaled_logits)
    # Normalize the probabilities
    sum_exp_logits = tr.sum(exp_logits, dim=-1, keepdim=True)
    softmax_probs = exp_logits / sum_exp_logits
    return softmax_probs


def limited_softmax(logits: T, tau: float = 1.0, max_prob: float = 1.0) -> T:
    """
    Compute a softmax with a maximum probability for each class.
    If a class has a probability greater than the maximum, the excess probability is
    distributed uniformly among the other classes.

    Args:
        logits: The input logits.
        tau: The temperature scaling factor.
        max_prob: The maximum probability for each class.
    """
    n_classes = logits.size(-1)
    min_max_prob = 1.0 / n_classes
    assert min_max_prob < max_prob <= 1.0
    softmax_probs = stable_softmax(logits, tau)
    if max_prob == 1.0:
        return softmax_probs
    clipped_probs = tr.clip(softmax_probs, max=max_prob)
    excess_probs = tr.clip(softmax_probs - clipped_probs, min=0.0)
    n_excess_probs = (excess_probs > 0.0).sum(dim=-1, keepdim=True)
    excess_probs = excess_probs.sum(dim=-1, keepdim=True)
    excess_probs = excess_probs / (n_classes - n_excess_probs)
    lim_probs = tr.clip(clipped_probs + excess_probs, max=max_prob)
    # lim_prob_sums = lim_probs.sum(dim=-1, keepdim=True)
    # assert tr.allclose(lim_prob_sums, tr.ones_like(lim_prob_sums))
    return lim_probs


def target_softmax(
    logits: T, max_prob: float = 1.0, eps: float = 1e-6, max_iter: int = 10000
) -> T:
    assert logits.ndim == 1
    n_classes = logits.size(-1)
    min_max_prob = 1.0 / n_classes
    assert min_max_prob < max_prob <= 1.0
    curr_tau = 1.0
    probs = stable_softmax(logits, curr_tau)
    idx = 0
    for idx in range(max_iter):
        curr_min_prob = probs.min().item()
        curr_max_prob = probs.max().item()
        # delta = curr_max_prob - max_prob
        curr_range = curr_max_prob - curr_min_prob
        delta = curr_range - max_prob
        if abs(delta) < eps:
            break
        elif delta < 0:
            curr_tau *= 0.9
        else:
            curr_tau *= 1.1
        probs = stable_softmax(logits, curr_tau)
    log.info(f"idx = {idx}")
    if idx == max_iter - 1:
        log.warning(f"target_softmax: max_iter reached: {max_iter}")
    return probs


if __name__ == "__main__":
    print("Hello, world!")
    print("This is a util module.")

    # Print torch tensors as 2 decimal places not in scientific notation
    tr.set_printoptions(precision=2, sci_mode=False)

    logits = tr.tensor([[0.5, 0.4, 0.1], [2.0, 3.0, 4.0]])
    tau = 0.25
    softmax_probs = stable_softmax(logits, tau)
    print(softmax_probs)
    softmax_probs = limited_softmax(logits, tau, max_prob=0.60)
    print(softmax_probs)
    softmax_probs = tr.softmax(logits, dim=-1)
    print(softmax_probs)
