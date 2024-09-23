import logging
import os

import torch as tr
from pandas import DataFrame
from torch import Tensor as T
from torch.utils.data import Dataset

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ChirpTextureDataset(Dataset):
    def __init__(self, df: DataFrame):
        super().__init__()
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> (T, T, T, int, int):
        theta_d_0to1 = tr.tensor(self.df.iloc[idx]["d"], dtype=tr.float32)
        theta_s_0to1 = tr.tensor(self.df.iloc[idx]["s"], dtype=tr.float32)
        seed = tr.tensor(self.df.iloc[idx]["seed"], dtype=tr.long)
        return theta_d_0to1, theta_s_0to1, seed, idx
