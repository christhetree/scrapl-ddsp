import logging
import os

import torch as tr
from nnAudio.features import CQT
from pandas import DataFrame
from torch import Tensor as T
from torch.utils.data import Dataset

from experiments.synth import ChirpTextureSynth

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class ChirpTextureDataset(Dataset):
    def __init__(self,
                 df: DataFrame,
                 synth: ChirpTextureSynth,
                 J: int,
                 feature_type: str = "cqt",
                 cqt_eps: float = 1e-3):
        super().__init__()
        self.df = df
        self.synth = synth
        self.J = J
        self.feature_type = feature_type
        self.cqt_eps = cqt_eps
        cqt_params = {
            "sr": synth.sr,
            "bins_per_octave": synth.Q,
            "n_bins": self.J * synth.Q,
            "hop_length": synth.hop_len,
            "fmin": (0.4 * synth.sr) / (2 ** J),
        }
        self.cqt = CQT(**cqt_params)

    # TODO(cm): move to data module if we want GPU acceleration
    def calc_u(self, x: T) -> T:
        if self.feature_type == "cqt":
            return self.calc_cqt(x, self.cqt, self.cqt_eps)
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> (T, T, T, int):
        theta_density = tr.tensor(self.df.iloc[idx]["density"], dtype=tr.float32)
        theta_slope = tr.tensor(self.df.iloc[idx]["slope"], dtype=tr.float32)
        seed = self.df.iloc[idx]["seed"]
        x = self.synth(theta_density, theta_slope, seed)
        u = self.calc_u(x)
        return u, theta_density, theta_slope, seed

    @staticmethod
    def calc_cqt(x: T, cqt: CQT, cqt_eps: float = 1e-3) -> T:
        u = cqt(x)
        u = u.abs()
        u = tr.log1p(u / cqt_eps)
        return u