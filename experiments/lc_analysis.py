import functools
import logging
import os
from typing import Optional, List, Dict, Any, Callable, Literal

import hessian_eigenthings
import torch as tr
import torch.nn as nn
from hessian_eigenthings.operator import LambdaOperator
from torch import Tensor as T
from torch.nn import Parameter

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def _calc_batch_theta_param_grad(
    theta_fn: Callable[..., T],
    loss_fn: Callable[[T, ...], T],
    theta_fn_kwargs: Dict[str, Any],
    params: List[Parameter],
    n_theta: int,
    theta_idx: Optional[int] = None,
    loss_fn_kwargs: Optional[Dict[str, Any]] = None,
) -> T:
    if loss_fn_kwargs is None:
        loss_fn_kwargs = {}
    assert params, "params must not be empty"
    assert all(p.grad is None for p in params), "params must have no grad"
    if theta_idx is not None:
        assert 0 <= theta_idx < n_theta
    theta_hat = theta_fn(**theta_fn_kwargs)
    assert theta_hat.ndim == 2
    assert theta_hat.size(1) == n_theta
    bs = theta_hat.size(0)
    loss = loss_fn(theta_hat, **loss_fn_kwargs)
    assert loss.numel() == 1

    theta_grad = tr.autograd.grad(
        loss, theta_hat, create_graph=True, materialize_grads=True
    )[0]
    if theta_idx is None:
        # Expand theta_grad to vectorize individual theta grad calculations
        theta_grad_ex = theta_grad.unsqueeze(0).expand(n_theta, -1, -1)
        theta_eye = tr.eye(
            n_theta, dtype=theta_grad_ex.dtype, device=theta_grad_ex.device
        )
        theta_eye_ex = theta_eye.unsqueeze(1).expand(-1, bs, -1)
        theta_grad_ex = theta_grad_ex * theta_eye_ex
        is_grads_batched = True
    else:
        # Zero out all but the specified theta grad
        mask = tr.zeros_like(theta_grad)
        mask[:, theta_idx] = 1.0
        theta_grad_ex = theta_grad * mask
        is_grads_batched = False
    # Calculate maybe vectorized param grad
    theta_param_grads = tr.autograd.grad(
        theta_hat,
        params,
        grad_outputs=theta_grad_ex,
        create_graph=True,
        materialize_grads=True,
        is_grads_batched=is_grads_batched,
    )
    if theta_idx is None:
        theta_param_grad = tr.cat(
            [g.view(n_theta, -1) for g in theta_param_grads], dim=1
        )
    else:
        theta_param_grad = tr.cat([g.view(-1) for g in theta_param_grads])
    return theta_param_grad


def _calc_param_hvp(
    tangent: T,
    param_grad: T,
    params: List[Parameter],
    retain_graph: bool = False,
) -> T:
    assert tangent.ndim == 1
    assert param_grad.shape == tangent.shape
    assert all(p.grad is None for p in params), "params must have no grad"
    param_hvps = tr.autograd.grad(
        param_grad,
        params,
        grad_outputs=tangent,
        materialize_grads=True,
        retain_graph=retain_graph,
    )
    param_hvp = tr.cat([g.contiguous().view(-1) for g in param_hvps])
    return param_hvp


def _calc_largest_eig(
    param_grad: T,
    params: List[Parameter],
    n_iter: int = 20,
) -> T:
    apply_fn = functools.partial(
        _calc_param_hvp,
        param_grad=param_grad,
        params=params,
        retain_graph=True,
    )
    size = param_grad.size(0)
    hvp_op = LambdaOperator(apply_fn, size)
    use_gpu = param_grad.is_cuda
    eigs, _ = hessian_eigenthings.deflated_power_iteration(
        operator=hvp_op,
        num_eigenthings=1,
        power_iter_steps=n_iter,
        to_numpy=False,
        use_gpu=use_gpu,
    )
    eig1 = tr.from_numpy(eigs.copy()).float()
    return eig1


def calc_theta_eigs(
    theta_fn: Callable[..., T],
    loss_fn: Callable[[T, ...], T],
    theta_fn_kwargs: Dict[str, Any],
    params: List[Parameter],
    n_theta: int,
    loss_fn_kwargs: Optional[Dict[str, Any]] = None,
    n_iter: int = 20,
) -> T:
    theta_param_grad = _calc_batch_theta_param_grad(
        theta_fn,
        loss_fn,
        theta_fn_kwargs,
        params,
        n_theta,
        loss_fn_kwargs=loss_fn_kwargs,
    )
    theta_eigs = []
    for theta_idx in range(n_theta):
        param_grad = theta_param_grad[theta_idx, :]
        eig1 = _calc_largest_eig(param_grad, params, n_iter=n_iter)
        theta_eigs.append(eig1)
    theta_eigs = tr.cat(theta_eigs, dim=0)
    return theta_eigs


def _calc_param_hvp_multibatch(
    tangent: T,
    n_theta: int,
    theta_idx: int,
    theta_fn: Callable[..., T],
    loss_fn: Callable[[T, ...], T],
    theta_fn_kwargs: List[Dict[str, Any]],
    params: List[Parameter],
    loss_fn_kwargs: Optional[List[Dict[str, Any]]] = None,
) -> T:
    if loss_fn_kwargs is None:
        assert theta_fn_kwargs, "theta_fn_kwargs must not be empty"
        loss_fn_kwargs = [None] * len(theta_fn_kwargs)
    else:
        assert len(loss_fn_kwargs) == len(theta_fn_kwargs), (
            f"len(theta_fn_kwargs) ({len(theta_fn_kwargs)}) != "
            f"len(loss_fn_kwargs) ({len(loss_fn_kwargs)})"
        )
    param_hvp = None
    for curr_theta_fn_kwargs, curr_loss_fn_kwargs in zip(
        theta_fn_kwargs, loss_fn_kwargs
    ):
        curr_param_grad = _calc_batch_theta_param_grad(
            theta_fn,
            loss_fn,
            curr_theta_fn_kwargs,
            params,
            n_theta,
            theta_idx=theta_idx,
            loss_fn_kwargs=curr_loss_fn_kwargs,
        )
        curr_param_hvp = _calc_param_hvp(
            tangent, curr_param_grad, params, retain_graph=False
        )
        # TODO(cm): should we average here?
        if param_hvp is None:
            param_hvp = curr_param_hvp
        else:
            param_hvp += curr_param_hvp
    return param_hvp


def _calc_theta_largest_eig_multibatch(
    n_theta: int,
    theta_idx: int,
    theta_fn: Callable[..., T],
    loss_fn: Callable[[T, ...], T],
    theta_fn_kwargs: List[Dict[str, Any]],
    params: List[Parameter],
    loss_fn_kwargs: Optional[List[Dict[str, Any]]] = None,
    n_iter: int = 20,
) -> T:
    apply_fn = functools.partial(
        _calc_param_hvp_multibatch,
        n_theta=n_theta,
        theta_idx=theta_idx,
        theta_fn=theta_fn,
        loss_fn=loss_fn,
        theta_fn_kwargs=theta_fn_kwargs,
        params=params,
        loss_fn_kwargs=loss_fn_kwargs,
    )
    size = sum(p.numel() for p in params)
    hvp_op = LambdaOperator(apply_fn, size)
    use_gpu = params[0].is_cuda
    eigs, _ = hessian_eigenthings.deflated_power_iteration(
        operator=hvp_op,
        num_eigenthings=1,
        power_iter_steps=n_iter,
        to_numpy=False,
        use_gpu=use_gpu,
    )
    eig1 = tr.from_numpy(eigs.copy()).float()
    return eig1


def calc_theta_eigs_multibatch(
    theta_fn: Callable[..., T],
    loss_fn: Callable[[T, ...], T],
    theta_fn_kwargs: List[Dict[str, Any]],
    params: List[Parameter],
    n_theta: int,
    loss_fn_kwargs: Optional[List[Dict[str, Any]]] = None,
    n_iter: int = 20,
) -> T:
    theta_eigs = []
    for theta_idx in range(n_theta):
        eig1 = _calc_theta_largest_eig_multibatch(
            n_theta,
            theta_idx,
            theta_fn,
            loss_fn,
            theta_fn_kwargs,
            params,
            loss_fn_kwargs=loss_fn_kwargs,
            n_iter=n_iter,
        )
        theta_eigs.append(eig1)
    theta_eigs = tr.cat(theta_eigs, dim=0)
    return theta_eigs


def check_is_deterministic(
    fn: Callable[..., T],
    fn_kwargs: Dict[str, Any],
) -> bool:
    x_1 = fn(**fn_kwargs)
    x_2 = fn(**fn_kwargs)
    if not tr.allclose(x_1, x_2):
        return False
    return True


def analyze_lc_hvp(
    theta_fn: Callable[..., T],
    loss_fn: Callable[[T, ...], T],
    theta_fn_kwargs: List[Dict[str, Any]],
    params: List[Parameter],
    n_theta: int,
    loss_fn_kwargs: Optional[List[Dict[str, Any]]] = None,
    n_iter: int = 20,
    agg: Literal["all", "mean", "max", "med"] = "all",
    force_multibatch: bool = False,
) -> T:
    assert params, "params must not be empty"
    assert theta_fn_kwargs, "theta_fn_kwargs must not be empty"
    if loss_fn_kwargs is not None:
        assert len(loss_fn_kwargs) == len(theta_fn_kwargs), (
            f"len(theta_fn_kwargs) ({len(theta_fn_kwargs)}) != "
            f"len(loss_fn_kwargs) ({len(loss_fn_kwargs)})"
        )
    is_multibatch = force_multibatch or len(theta_fn_kwargs) > 1
    log.info(
        f"Starting warmup_lc_hvp with agg = {agg} for {len(params)} "
        f"parameter(s) and {len(theta_fn_kwargs)} batch(es) "
        f"(multibatch = {is_multibatch})"
    )
    # Check determinism
    if not check_is_deterministic(theta_fn, theta_fn_kwargs[0]):
        log.warning("theta_fn is not deterministic")
    # Determine whether to use multibatch or not
    if is_multibatch:
        calc_theta_eigs_fn = calc_theta_eigs_multibatch
    else:
        calc_theta_eigs_fn = calc_theta_eigs
        theta_fn_kwargs = theta_fn_kwargs[0]
        if loss_fn_kwargs is not None:
            loss_fn_kwargs = loss_fn_kwargs[0]
    # Separate the params for separate computations if aggregating LCs across them
    if agg == "all":
        param_groups = [params]
    else:
        param_groups = [[p] for p in params]
    # Compute the theta LCs
    vals = []
    for param_group in param_groups:
        curr_vals = calc_theta_eigs_fn(
            theta_fn=theta_fn,
            loss_fn=loss_fn,
            theta_fn_kwargs=theta_fn_kwargs,
            params=param_group,
            n_theta=n_theta,
            loss_fn_kwargs=loss_fn_kwargs,
            n_iter=n_iter,
        )
        vals.append(curr_vals)
    # Aggregate the theta LCs across all params
    vals = tr.stack(vals, dim=0)
    if agg == "all":
        vals = vals[0]
    elif agg == "mean":
        vals = vals.mean(dim=0)
    elif agg == "max":
        vals = vals.max(dim=0).values
    elif agg == "med":
        vals = vals.median(dim=0).values
    else:
        raise ValueError(f"Invalid agg = {agg}")
    return vals


if __name__ == "__main__":
    # Setup
    tr.set_printoptions(precision=4, sci_mode=False)
    tr.manual_seed(0)
    bs = 2
    n_samples = 32768
    n_theta = 3

    model = nn.Sequential(
        nn.Linear(n_samples, n_theta),
        nn.PReLU(),
        nn.Linear(n_theta, n_theta),
        nn.Sigmoid(),
    )
    model = model
    synth = nn.Sequential(
        nn.Linear(n_theta, n_theta),
        nn.PReLU(),
        nn.Linear(n_theta, n_samples),
        nn.Tanh(),
    )
    synth = synth
    loss_func = nn.MSELoss()
    x_1 = tr.rand((bs, n_samples))
    x_2 = tr.rand((bs, n_samples))
    x_3 = tr.rand((bs, n_samples))

    params = [p for p in model.parameters()]
    assert all(not p.grad for p in params)

    theta_fn = lambda x: model(x.squeeze(1))

    def loss_fn(theta: T, x: T) -> T:
        x_hat = synth(theta).unsqueeze(1)
        loss = loss_func(x_hat, x)
        return loss

    theta_fn_kwargs = [
        {"x": x_1.unsqueeze(1)},
        # {"x": x_1.unsqueeze(1)},
        # {"x": x_2.unsqueeze(1)},
        # {"x": x_3.unsqueeze(1)},
    ]
    loss_fn_kwargs = theta_fn_kwargs
    lc_vals = analyze_lc_hvp(
        theta_fn=theta_fn,
        loss_fn=loss_fn,
        theta_fn_kwargs=theta_fn_kwargs,
        loss_fn_kwargs=loss_fn_kwargs,
        params=params,
        n_theta=n_theta,
        n_iter=20,
        agg="all",
        # agg="max",
        force_multibatch=False,
        # force_multibatch=True,
    )
    log.info(f"lc_vals = {lc_vals}")
