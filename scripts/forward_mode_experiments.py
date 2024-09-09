import logging
import os
from typing import Any

from torch.autograd import Function

import torch as tr
from torch import Tensor as T

from experiments.losses import SCRAPLLoss

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


scrapl = SCRAPLLoss(shape=32768, J=12, Q1=8, Q2=2, J_fr=3, Q_fr=2, fixed_path_idx=0)


def fn(x: T) -> T:
    assert x.ndim == 2
    y = x.repeat(1, 2)
    y *= 3
    return y


def scrapl_fn(x: T) -> T:
    x = x.view(1, 1, -1)
    y = scrapl.jtfs(x)
    y = y.squeeze()
    return y


class FnWrapper(Function):
    @staticmethod
    def forward(ctx: Any, x: T) -> T:
        y = fn(x)
        ctx.save_for_backward(x, y)
        return y

    @staticmethod
    def vjp(ctx: Any, grad_output: T) -> T:
        (x, y) = ctx.saved_tensors
        (jacobian,) = tr.autograd.functional.jacobian(
            fn, (x,), vectorize=True, strategy="reverse-mode"
        )
        # y = y.unsqueeze(0).unsqueeze(0)
        y = y.unsqueeze(-1).unsqueeze(-1)
        grad_input = (jacobian * y).sum(dim=1).sum(dim=1)
        return grad_input


if __name__ == "__main__":
    # x = tr.rand((3, 2))
    x = tr.rand((32768,))

    # y1, jac1 = tr.autograd.functional.jacobian(scrapl_fn, (x,), vectorize=True, strategy="reverse-mode")
    y1, jac1 = tr.autograd.functional.jacobian(scrapl_fn, (x,), vectorize=True, strategy="forward-mode")
    log.info(f"y1.shape: {y1.shape}")
    log.info(f"jac1.shape: {jac1.shape}")
    exit()

    x_b = x.clone()
    x_b2 = x_b.clone()
    x_f = x_b.clone()
    x_f2 = x_b.clone()
    # func = fn
    func = scrapl_fn

    # Calc input gradients
    x_b.requires_grad = True
    y_b = func(x_b)
    (grad_b,) = tr.autograd.grad(y_b, [x_b], grad_outputs=tr.ones_like(y_b))
    log.info(f"grad_b.shape: {grad_b.shape}")
    log.info(f"grad_b: {grad_b}")
    log.info(f"y_b = {y_b}")

    # Calc input gradients using vjp
    (y_b2, (grad_b2,)) = tr.autograd.functional.vjp(func, (x_b2,), v=tr.ones_like(y_b))
    log.info(f"grad_b2.shape: {grad_b2.shape}")
    log.info(f"grad_b2: {grad_b2}")
    log.info(f"y_b2 = {y_b2}")
    assert grad_b.shape == x_b.shape == grad_b2.shape == x_b2.shape
    assert tr.allclose(grad_b, grad_b2)

    # Calc output gradients
    tangent = tr.ones_like(x_f)
    y_f, grad_f = tr.func.jvp(func, (x_f,), (tangent,))
    log.info(f"grad_f.shape: {grad_f.shape}")
    log.info(f"grad_f: {grad_f}")
    log.info(f"y_f = {y_f}")

    # Slow forward gradient using backward backward trick
    (y_f2, grad_f2) = tr.autograd.functional.jvp(func, (x_f2,), v=tr.ones_like(x_f2))
    log.info(f"grad_f2.shape: {grad_f2.shape}")
    log.info(f"grad_f2: {grad_f2}")
    log.info(f"y_f2 = {y_f2}")

    assert grad_f.shape == y_f.shape == grad_f2.shape == y_f2.shape
    assert tr.allclose(grad_f, grad_f2)

    assert tr.allclose(y_f, y_f2)
    assert tr.allclose(y_b, y_b2)

    x.requires_grad = True
    y = FnWrapper.apply(x)
    y = y.sum()
    y.backward()
    grad = x.grad
    log.info(f"grad.shape: {grad.shape}")
    log.info(f"grad: {grad}")
    log.info(f"y: {y}")
