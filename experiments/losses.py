import logging
import os
from typing import Union, List, Any

import torch as tr
import torch.nn as nn
from kymatio.torch import TimeFrequencyScattering
from torch import Tensor as Tensor

from jtfst_implementation.python.jtfst import JTFST2D
from scrapl.torch import TimeFrequencyScrapl

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class JTFSTLoss(nn.Module):
    def __init__(self,
                 shape: int,
                 J: int,
                 Q1: int,
                 Q2: int,
                 J_fr: int,
                 Q_fr: int,
                 T: Union[str, int],
                 F: Union[str, int],
                 format: str = "time"):
        super().__init__()
        # TODO(cm): try with joint format and just mae distance
        self.jtfs = TimeFrequencyScattering(
            shape=(shape,),
            J=J,
            Q=(Q1, Q2),
            Q_fr=Q_fr,
            J_fr=J_fr,
            T=T,
            F=F,
            format="time",
        )

    def forward(self, x: Tensor, x_target: Tensor) -> Tensor:
        # a = tr.tensor([1, 2, 3], dtype=tr.float32).unsqueeze(0)
        # b = tr.tensor([4, 5, 6], dtype=tr.float32).unsqueeze(0)
        # norm_manual = tr.sqrt(tr.sum((a - b) ** 2))
        # norm_linalg = tr.linalg.vector_norm(a - b, ord=2, dim=1)
        # norm_nmf = torchnmf.metrics.euclidean(a, b)

        assert x.ndim == x_target.ndim == 3
        assert x.size(1) == x_target.size(1) == 1
        Sx = self.jtfs(x)
        Sx = Sx[:, :, 1:, :]
        Sx_target = self.jtfs(x_target)
        Sx_target = Sx_target[:, :, 1:, :]
        # log_Sx = tr.log(Sx)
        # log_Sx_target = tr.log(Sx_target)
        # dist_l1 = tr.linalg.vector_norm(Sx_target - Sx, ord=1, dim=(2, 3))
        # dist_l1_log = tr.linalg.vector_norm(log_Sx_target - log_Sx, ord=1, dim=(2, 3))
        # dist = dist_l1 + dist_l1_log
        # dist = tr.mean(dist)
        # Sx = Sx / Sx.sum()
        # Sx_target = Sx_target / Sx_target.sum()
        # dist = torchnmf.metrics.kl_div(Sx, Sx_target)
        # dist = torchnmf.metrics.euclidean(Sx, Sx_target)
        dist = tr.linalg.vector_norm(Sx_target - Sx, ord=2, dim=(2, 3))
        dist = tr.mean(dist)
        return dist


class MyJTFST2DLoss(nn.Module):
    def __init__(self,
                 sr: float,
                 J: int,
                 Q1: int,
                 Q2: int,
                 J_fr: int,
                 Q_fr: int,
                 T: int,
                 F: int):
        super().__init__()
        should_avg_f = False
        should_avg_t = False
        if F > 1:
            should_avg_f = True
        if T > 1:
            should_avg_t = True

        self.jtfs = JTFST2D(sr=sr,
                            J_1=J,
                            J_2_f=J_fr,
                            J_2_t=J,
                            Q_1=Q1,
                            Q_2_f=Q_fr,
                            Q_2_t=Q2,
                            should_avg_f=should_avg_f,
                            should_avg_t=should_avg_t,
                            avg_win_f=F,
                            avg_win_t=T,
                            reflect_f=True)

    def forward(self, x: Tensor, x_target: Tensor) -> Tensor:
        assert x.ndim == x_target.ndim == 3
        assert x.size(1) == x_target.size(1) == 1
        Sx_o1, _, Sx, _ = self.jtfs(x)
        Sx_o1_target, _, Sx_target, _ = self.jtfs(x_target)
        dist = tr.linalg.vector_norm(Sx_target - Sx, ord=2, dim=(2, 3))
        dist = tr.mean(dist)
        return dist


class SCRAPLLoss(nn.Module):
    def __init__(self,
                 shape: int,
                 J: int,
                 Q1: int,
                 Q2: int,
                 J_fr: int,
                 Q_fr: int,
                 T: Union[str, int],
                 F: Union[str, int]):
        super().__init__()
        self.jtfs = TimeFrequencyScrapl(
            shape=(shape,),
            J=J,
            Q=(Q1, Q2),
            Q_fr=Q_fr,
            J_fr=J_fr,
            T=T,
            F=F,
        )
        scrapl_meta = self.jtfs.meta()
        self.scrapl_keys = [key for key in scrapl_meta["key"] if len(key) == 2]
        log.info(f"number of SCRAPL keys = {len(self.scrapl_keys)}")

    def forward(self, x: Tensor, x_target: Tensor) -> Tensor:
        assert x.ndim == x_target.ndim == 3
        assert x.size(1) == x_target.size(1) == 1
        n2, n_fr = SCRAPLLoss.choice(self.scrapl_keys)
        Sx = self.jtfs.scattering_singlepath(x, n2, n_fr)
        Sx = Sx["coef"].squeeze(-1)
        Sx_target = self.jtfs.scattering_singlepath(x_target, n2, n_fr)
        Sx_target = Sx_target["coef"].squeeze(-1)
        diff = Sx_target - Sx
        dist = tr.linalg.vector_norm(diff, ord=2, dim=(2, 3))
        dist = tr.mean(dist)
        return dist

    @staticmethod
    def randint(low: int, high: int, n: int = 1) -> Union[int, Tensor]:
        x = tr.randint(low=low, high=high, size=(n,))
        if n == 1:
            return x.item()
        return x

    @staticmethod
    def choice(items: List[Any]) -> Any:
        assert len(items) > 0
        idx = SCRAPLLoss.randint(low=0, high=len(items))
        return items[idx]
