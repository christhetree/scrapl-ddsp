import itertools
import logging
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch as tr
from torch.utils.data import DataLoader

from experiments.datasets import ChirpTextureDataset

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ThetaDSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        n_d: int,
        n_s: int,
        n_seeds_per_theta: int = 1,
        n_folds: int = 5,
        use_unique_seeds: bool = True,
        use_unique_theta_in_val_test: bool = True,
        num_workers: int = 0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.n_d = n_d
        self.n_s = n_s
        self.n_seeds_per_theta = n_seeds_per_theta
        assert n_folds >= 3, "n_folds must be at least 3."
        self.n_folds = n_folds
        if use_unique_seeds:
            log.info(f"Using a unique seed for each theta combination.")
        self.use_unique_seeds = use_unique_seeds
        if use_unique_theta_in_val_test:
            log.info(f"Theta combinations are unique in validation and test sets.")
        self.use_unique_theta_in_val_test = use_unique_theta_in_val_test
        self.num_workers = num_workers

        d_idx = np.arange(n_d)
        s_idx = np.arange(n_s)
        if use_unique_theta_in_val_test:
            seeds = np.arange(n_seeds_per_theta)
        else:
            seeds = np.arange(n_seeds_per_theta * n_folds)

        theta_idx = list(itertools.product(d_idx, s_idx, seeds))
        df_idx = pd.DataFrame(theta_idx, columns=["d_idx", "s_idx", "seed"])
        d_s = np.linspace(0, 1, n_d + 2)[1:-1]
        s_s = np.linspace(0, 1, n_s + 2)[1:-1]
        thetas = list(itertools.product(d_s, s_s, seeds))
        df = pd.DataFrame(thetas, columns=["d", "s", "seeds_tmp"])
        del df["seeds_tmp"]
        df = df_idx.merge(df, left_index=True, right_index=True)

        folds = df["seed"] % n_folds
        df["fold"] = folds

        # Replace seeds with unique values for each theta combination if requested
        if use_unique_seeds:
            seeds = np.arange(len(df))
            df["seed"] = seeds

        # Ensure unseen theta combinations in validation and test sets if requested
        if use_unique_theta_in_val_test:
            df["unique_theta_idx"] = df["d_idx"] * n_s + df["s_idx"]
            folds = df["unique_theta_idx"] % n_folds
            df["fold"] = folds
            del df["unique_theta_idx"]

        # Shuffle such that batches in validation and test contain a variety of
        # different theta values. This makes the visualization callbacks more diverse.
        df = df.sample(frac=1, random_state=tr.random.initial_seed()).reset_index(
            drop=True
        )

        self.df = df
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: str) -> None:
        train_df = self.df[self.df["fold"] < (self.n_folds - 2)]
        self.train_ds = ChirpTextureDataset(train_df)
        val_df = self.df[self.df["fold"] == (self.n_folds - 2)]
        self.val_ds = ChirpTextureDataset(val_df)
        test_df = self.df[self.df["fold"] == (self.n_folds - 1)]
        self.test_ds = ChirpTextureDataset(test_df)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=True,  # For more diversity in visualization callbacks
            num_workers=self.num_workers,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )
