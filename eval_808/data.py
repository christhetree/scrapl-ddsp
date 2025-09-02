import logging
import os
from typing import List

import pytorch_lightning as pl
import torch as tr
from torch import Tensor as T
from torch.utils.data import DataLoader, Dataset

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class SeedDataset(Dataset):
    def __init__(
        self,
        seeds: List[int],
        n_params: int,
        randomize_seed: bool = False,
    ):
        super().__init__()
        self.seeds = seeds
        self.n_params = n_params
        self.randomize_seed = randomize_seed
        self.rand_gen = tr.Generator(device="cpu")

    def __len__(self):
        return len(self.seeds)

    def __getitem__(self, idx: int) -> T:
        seed = self.seeds[idx]
        if self.randomize_seed:
            seed = tr.randint(0, 99999999, (1,)).item()
        self.rand_gen.manual_seed(seed)
        params = tr.rand((self.n_params,), generator=self.rand_gen)
        return params


class SeedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        n_seeds: int,
        n_params: int,
        val_split: float = 0.2,
        test_split: float = 0.2,
        randomize_train_seed: bool = False,
        num_workers: int = 0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.n_seeds = n_seeds
        self.n_params = n_params
        self.val_split = val_split
        self.test_split = test_split
        self.randomize_train_seed = randomize_train_seed
        self.num_workers = num_workers

        seeds = list(range(n_seeds))
        train_end_idx = int(n_seeds * (1.0 - val_split - test_split))
        val_end_idx = int(n_seeds * (1.0 - test_split))
        train_seeds = seeds[:train_end_idx]
        val_seeds = seeds[train_end_idx:val_end_idx]
        test_seeds = seeds[val_end_idx:]

        self.train_dataset = SeedDataset(
            seeds=train_seeds,
            n_params=n_params,
            randomize_seed=randomize_train_seed,
        )
        self.val_dataset = SeedDataset(
            seeds=val_seeds,
            n_params=n_params,
        )
        self.test_dataset = SeedDataset(
            seeds=test_seeds,
            n_params=n_params,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )
