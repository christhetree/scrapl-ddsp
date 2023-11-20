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
                 Q: Tuple[int, int],
                 J_fr: int,
                 Q_fr: int,
                 T: str = "global",
                 F: str = "global",
                 format_: str = "time"):
        super().__init__()
        self.jtfs = TimeFrequencyScattering(
            shape=(shape,),
            J=J,
            Q=Q,
            Q_fr=Q_fr,
            J_fr=J_fr,
            T=T,
            F=F,
            format=format_,
        )

    def forward(self, x: Tensor, x_target: Tensor) -> Tensor:
        Sx = self.jtfs(x)[1:]
        Sx_target = self.jtfs(x_target)[1:]
        dist = tr.linalg.vector_norm(Sx_target - Sx, p=2, dim=1)
        return dist


class SCRAPLLoss(nn.Module):
    def __init__(self,
                 shape: int,
                 J: int,
                 Q: Tuple[int, int],
                 J_fr: int,
                 Q_fr: int,
                 T: str = "global",
                 F: str = "global"):
        super().__init__()
        self.scrapl = TimeFrequencyScrapl(
            shape=(shape,),
            J=J,
            Q=Q,
            Q_fr=Q_fr,
            J_fr=J_fr,
            T=T,
            F=F,
        )
        scrapl_meta = self.scrapl.meta()
        self.scrapl_keys = [key for key in scrapl_meta["key"] if len(key) == 2]

    def forward(self, x: Tensor, x_target: Tensor) -> Tensor:
        n2, n_fr = SCRAPLLoss.choice(self.scrapl_keys)
        Sx = self.scrapl.scattering_singlepath(x, n2, n_fr)
        Sx_target = self.scrapl.scattering_singlepath(x_target, n2, n_fr)
        dist = tr.linalg.vector_norm(Sx_target - Sx, p=2, dim=1)
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
