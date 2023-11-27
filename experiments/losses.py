import logging
import os
from typing import Union, List, Any

import torch as tr
import torch.nn as nn
from kymatio.torch import TimeFrequencyScattering
from torch import Tensor as Tensor

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
                 T: Union[str, int] = "global",
                 F: Union[str, int] = "global",
                 format: str = "time"):
        super().__init__()
        self.jtfs = TimeFrequencyScattering(
            shape=(shape,),
            J=J,
            Q=(Q1, Q2),
            Q_fr=Q_fr,
            J_fr=J_fr,
            T=T,
            F=F,
            format=format,
        )

    def forward(self, x: Tensor, x_target: Tensor) -> Tensor:
        assert x.ndim == x_target.ndim == 3
        assert x.size(1) == x_target.size(1) == 1
        Sx = self.jtfs(x)
        Sx = Sx[:, :, 1:, :]
        Sx_target = self.jtfs(x_target)
        Sx_target = Sx_target[:, :, 1:, :]
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
                 T: Union[str, int] = "global",
                 F: Union[str, int] = "global"):
        super().__init__()
        self.scrapl = TimeFrequencyScrapl(
            shape=(shape,),
            J=J,
            Q=(Q1, Q2),
            Q_fr=Q_fr,
            J_fr=J_fr,
            T=T,
            F=F,
        )
        scrapl_meta = self.scrapl.meta()
        self.scrapl_keys = [key for key in scrapl_meta["key"] if len(key) == 2]

    def forward(self, x: Tensor, x_target: Tensor) -> Tensor:
        assert x.ndim == x_target.ndim == 3
        assert x.size(1) == x_target.size(1) == 1
        n2, n_fr = SCRAPLLoss.choice(self.scrapl_keys)
        Sx = self.scrapl.scattering_singlepath(x, n2, n_fr)
        Sx = Sx["coef"].squeeze(-1)
        Sx_target = self.scrapl.scattering_singlepath(x_target, n2, n_fr)
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
