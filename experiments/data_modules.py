import itertools
import logging
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from experiments.datasets import ChirpTextureDataset
from experiments.synth import ChirpTextureSynth

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class ChirpTextureDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int,
                 n_densities: int,
                 n_slopes: int,
                 n_seeds_per_fold: int,
                 n_folds: int,
                 synth: ChirpTextureSynth,
                 feature_type: str = "cqt",
                 J_cqt: int = 5,
                 cqt_eps: float = 1e-3,
                 num_workers: int = 0):
        super().__init__()
        self.save_hyperparameters(ignore=["synth"])
        log.info(f"\n{self.hparams}")

        self.batch_size = batch_size
        self.n_densities = n_densities
        self.n_slopes = n_slopes
        self.n_seeds_per_fold = n_seeds_per_fold
        self.n_folds = n_folds
        self.synth = synth
        self.feature_type = feature_type
        self.J_cqt = J_cqt
        self.cqt_eps = cqt_eps
        self.num_workers = num_workers

        slope_idx = np.arange(n_slopes)
        density_idx = np.arange(n_densities)
        seeds = np.arange(n_seeds_per_fold * n_folds)

        theta_idx = list(itertools.product(density_idx, slope_idx, seeds))
        df_idx = pd.DataFrame(theta_idx, columns=["density_idx", "slope_idx", "seed"])
        densities = np.linspace(0, 1, n_densities + 2)[1:-1]
        slopes = np.linspace(-1, 1, n_slopes + 2)[1:-1]
        thetas = list(itertools.product(densities, slopes, seeds))
        df = pd.DataFrame(thetas, columns=["density", "slope", "seeds_tmp"])
        del df["seeds_tmp"]
        df = df_idx.merge(df, left_index=True, right_index=True)

        folds = df["seed"] % n_folds
        df["fold"] = folds

        self.df = df
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: str) -> None:
        train_df = self.df[self.df["fold"] < (self.n_folds - 2)]
        self.train_ds = ChirpTextureDataset(train_df,
                                            self.synth,
                                            self.feature_type,
                                            self.J_cqt,
                                            self.cqt_eps)
        val_df = self.df[self.df["fold"] == (self.n_folds - 2)]
        self.val_ds = ChirpTextureDataset(val_df,
                                          self.synth,
                                          self.feature_type,
                                          self.J_cqt,
                                          self.cqt_eps)
        test_df = self.df[self.df["fold"] == (self.n_folds - 1)]
        self.test_ds = ChirpTextureDataset(test_df,
                                           self.synth,
                                           self.feature_type,
                                           self.J_cqt,
                                           self.cqt_eps)

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
            shuffle=False,
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