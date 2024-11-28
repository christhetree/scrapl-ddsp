import logging
import os
from typing import List, Any, Union, Dict, Tuple, Iterator, Set, Optional

import torch as tr
import torch.nn.functional as F
from scipy.stats import loguniform
from torch import Tensor as T, nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ReadOnlyTensorDict(nn.Module):
    def __init__(self, data: Dict[str | int, T]):
        super().__init__()
        self.keys = set(data.keys())
        for k, v in data.items():
            self.register_buffer(f"tensor_{k}", v)

    def __getitem__(self, key: str | int) -> T:
        return self.get_buffer(f"tensor_{key}")

    def __contains__(self, key: str | int) -> bool:
        return key in self.keys

    def __len__(self) -> int:
        return len(self.keys)

    def __iter__(self) -> Iterator[str | int]:
        return iter(self.keys)

    def keys(self) -> Iterator[str | int]:
        return iter(self.keys)

    def values(self) -> Iterator[T]:
        for k in self.keys:
            yield self[k]

    def items(self) -> Iterator[Tuple[str | int, T]]:
        for k in self.keys:
            yield k, self[k]


def get_path_keys(
    meta: Dict[str, Any],
    sr: float,
    Q1: int,
    am_hz_min: Optional[float] = None,
    am_hz_max: Optional[float] = None,
    fm_oct_hz_min: Optional[float] = None,
    fm_oct_hz_max: Optional[float] = None,
    spins: Optional[Set[int]] = None,
    ignored_spins: Optional[Set[int]] = None,
    use_complement: bool = False,
    use_am_fm_union: bool = False,  # Use the AM and FM intersection by default
) -> List[Tuple[int, int]]:
    assert len(meta["key"]) == len(
        meta["order"]
    ), f"len(meta['key']) != len(meta['order'])"
    if spins is None:
        spins = {-1, 0, 1}
    all_keys = []
    keys = []
    key_infos = {}
    for idx, order in enumerate(meta["order"]):
        if order != 2:
            continue
        spin = meta["spin"][idx]
        # Ignored spins are specified separately to not include them in the complement
        if ignored_spins is not None and spin in ignored_spins:
            continue

        k = meta["key"][idx]
        am_cf_cycles_p_sec = meta["xi"][idx][1] * sr
        fm_cf_cycles_p_oct = abs(meta["xi_fr"][idx] * Q1)
        if fm_cf_cycles_p_oct == 0.0:
            fm_cf_oct_hz = tr.inf
        else:
            fm_cf_oct_hz = am_cf_cycles_p_sec / fm_cf_cycles_p_oct
        key_info = (
            f"{idx}: key = {k}, spin = {spin}, "
            f"am_cf_cycles_p_sec = {am_cf_cycles_p_sec:.2f}, "
            f"fm_cf_oct_hz = {fm_cf_oct_hz:.2f}"
        )
        key_infos[k] = key_info
        all_keys.append(k)

        if spin not in spins:
            continue
        satisfies_am = True
        if am_hz_min is not None and am_cf_cycles_p_sec < am_hz_min:
            satisfies_am = False
        if am_hz_max is not None and am_cf_cycles_p_sec > am_hz_max:
            satisfies_am = False
        satisfies_fm = True
        if fm_oct_hz_min is not None and fm_cf_oct_hz < fm_oct_hz_min:
            satisfies_fm = False
        if fm_oct_hz_max is not None and fm_cf_oct_hz > fm_oct_hz_max:
            satisfies_fm = False
        if (use_am_fm_union and (satisfies_am or satisfies_fm)) or (
            not use_am_fm_union and (satisfies_am and satisfies_fm)
        ):
            keys.append(k)

    if use_complement:
        complement_keys = [k for k in all_keys if k not in keys]
        keys = complement_keys
    for k in keys:
        log.info(key_infos[k])
    return keys


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


def target_range_softmax(
    logits: T, target_range: float, eps: float = 1e-6, max_iter: int = 10000
) -> T:
    assert logits.ndim == 1
    assert 0.0 < target_range < 1.0
    n_classes = logits.size(-1)
    min_max_prob = 1.0 / n_classes
    assert min_max_prob < target_range <= 1.0
    curr_tau = 1.0
    probs = stable_softmax(logits, curr_tau)
    idx = 0
    for idx in range(max_iter):
        curr_min_prob = probs.min().item()
        curr_max_prob = probs.max().item()
        curr_range = curr_max_prob - curr_min_prob
        delta = curr_range - target_range
        if abs(delta) < eps:
            break
        elif delta < 0:
            curr_tau *= 0.9
        else:
            curr_tau *= 1.1
        probs = stable_softmax(logits, curr_tau)
    # log.info(f"idx = {idx}")
    if idx == max_iter - 1:
        log.warning(f"target_softmax: max_iter reached: {max_iter}")
    return probs


def is_connected_via_ad_graph(output: T, input_: T) -> bool:
    visited = set()
    stack = [output.grad_fn]

    while stack:
        fn = stack.pop()
        if fn is None or fn in visited:
            continue
        visited.add(fn)

        # Check if the source tensor is an input to this function
        if hasattr(fn, 'variable') and fn.variable is input_:
            return True

        # Add next functions to the stack
        stack.extend(next_fn[0] for next_fn in fn.next_functions)

    log.info(f"Looked at {len(visited)} functions, input_ not found")
    return False


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
