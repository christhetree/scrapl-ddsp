import logging
import os
from contextlib import nullcontext
from typing import Callable, Literal

import hessian_eigenthings
import torch as tr
from hessian_eigenthings.operator import Operator
from torch import Tensor as T
from torch.autograd.profiler import record_function

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class HVPFunc(Operator):
    def __init__(
        self,
        loss_fn: Callable[[T], T],
        primal: T,
        use_profiler: bool = False,
        strategy: Literal["revrev", "fwdrev"] = "revrev",
    ):
        assert primal.ndim == 1
        size = primal.size(0)
        super().__init__(size)
        self.loss_fn = loss_fn
        self.primal = primal
        self.use_profiler = use_profiler
        assert strategy in ["revrev", "fwdrev"]
        self.strategy = strategy

    def _hvp_revrev(self, tangent: T) -> (T, T):
        loss, vjp_fn = tr.func.vjp(tr.func.grad(self.loss_fn), self.primal)
        hvp = vjp_fn(tangent)[0]
        return loss, hvp

    def _hvp_fwdrev(self, tangent: T) -> (T, T):
        loss, hvp = tr.func.jvp(tr.func.grad(self.loss_fn), (self.primal,), (tangent,))
        return loss, hvp

    def _hvp(self, tangent: T) -> T:
        if self.strategy == "revrev":
            loss, hvp = self._hvp_revrev(tangent)
        else:
            loss, hvp = self._hvp_fwdrev(tangent)

        # l1, h1 = self._hvp_revrev(tangent)
        # l2, h2 = self._hvp_fwdrev(tangent)
        # assert tr.allclose(l1, l2)
        # assert tr.allclose(h1, h2)
        return hvp

    def apply(self, tangent: T) -> T:
        assert tangent.shape == self.primal.shape
        with (
            tr.profiler.profile(
                activities=[tr.profiler.ProfilerActivity.CPU],
                with_stack=True,
                profile_memory=True,
                record_shapes=False,
            )
            if self.use_profiler
            else nullcontext()
        ) as prof:
            with record_function("_hvp") if self.use_profiler else nullcontext():
                hvp = self._hvp(tangent)
        if self.use_profiler:
            log.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1))
        return hvp


class HVPAutograd(HVPFunc):
    def _hvp_revrev(self, tangent: T) -> (T, T):
        loss, vhp = tr.autograd.functional.vhp(
            self.loss_fn, self.primal, tangent, create_graph=False, strict=False
        )
        hvp = vhp.t()
        return loss, hvp

    def _hvp_fwdrev(self, tangent: T) -> (T, T):
        log.warning(
            "Fwdrev strategy is slower then revrev in tr.autograd.functional "
            "because it isn't using forward mode AD under the hood."
        )
        loss, hvp = tr.autograd.functional.hvp(
            self.loss_fn, self.primal, tangent, create_graph=False, strict=False
        )

        # self.primal.requires_grad_()
        # loss_2 = self.loss_fn(self.primal)
        # grad = tr.autograd.grad(
        #     loss_2, self.primal, create_graph=True, allow_unused=False
        # )[0]
        # hvp_2 = tr.autograd.grad(
        #     grad,
        #     self.primal,
        #     grad_outputs=tangent,
        #     create_graph=False,
        #     allow_unused=False,
        # )[0]
        # assert tr.allclose(loss, loss_2)
        # assert tr.allclose(hvp, hvp_2)

        return loss, hvp


if __name__ == "__main__":

    def eps_fn(x: T, eps: float = 1e-12) -> T:
        # x = tr.where(x < 0, x - eps, x + eps)
        # x += eps
        # x[x < 0].sub_(2 * eps)
        # x.add_(eps)
        x.apply_(lambda x: x + eps if x > 0 else x - eps)
        # x[x > 0].add_(2 * eps)
        # x.sub_(eps)
        # real = x[..., 0]
        # imag = x[..., 1]
        # x[..., 0] = tr.sqrt(real**2 + imag**2)
        return x

    x = tr.randn((1000000, 2))
    with tr.profiler.profile(
        activities=[tr.profiler.ProfilerActivity.CPU],
        with_stack=True,
        profile_memory=True,
        record_shapes=False,
    ) as prof:
        with record_function("eps_fn"):
            x = eps_fn(x)
    log.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=5))
    exit()

    def loss_fn(x: T) -> T:
        return tr.sum(x**3)

    size = 1000000
    # primal = tr.randn(size).requires_grad_()
    primal = tr.randn(size)
    tangent = tr.randn(size)

    hvp_func = HVPFunc(loss_fn, primal, use_profiler=False)
    hvp_autograd = HVPAutograd(loss_fn, primal, use_profiler=False)

    hvp_1 = hvp_func.apply(tangent)
    hvp_2 = hvp_autograd.apply(tangent)

    assert tr.allclose(hvp_1, hvp_2)
    log.info("All tests passed")
    # exit()

    n_iter = 100000
    n_eigenthings = 1
    eigenvals, _ = hessian_eigenthings.lanczos(
        # operator=hvp_func,
        operator=hvp_autograd,
        num_eigenthings=n_eigenthings,
        tol=1e-3,
        which="LM",
        max_steps=n_iter,
        use_gpu=False,
    )
    log.info(f"Eigenvalues1: {eigenvals}")

    eigenvals_2, _ = hessian_eigenthings.deflated_power_iteration(
        operator=hvp_func,
        num_eigenthings=n_eigenthings,
        power_iter_steps=n_iter,
        power_iter_err_threshold=1e-3,
        use_gpu=False,
    )
    log.info(f"Eigenvalues2: {eigenvals_2}")
