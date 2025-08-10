import logging
import os
from typing import List

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from automix.data import DSD100Dataset, MedleyDBDataset

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class DSD100DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        dataset_dir: str,
        sr: int,
        train_n_samples: int,
        val_n_samples: int,
        train_indices: List[int],
        val_indices: List[int],
        test_indices: List[int],
        num_examples_per_epoch: int,
        num_workers: int = 0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_dir = dataset_dir
        self.sr = sr
        self.train_n_samples = train_n_samples
        self.val_n_samples = val_n_samples
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        self.num_examples_per_epoch = num_examples_per_epoch
        self.num_workers = num_workers

        self.train_ds = DSD100Dataset(
            root_dir=dataset_dir,
            length=train_n_samples,
            sample_rate=sr,
            indices=train_indices,
            num_examples_per_epoch=num_examples_per_epoch,
        )
        self.val_ds = DSD100Dataset(
            root_dir=dataset_dir,
            length=val_n_samples,
            sample_rate=sr,
            indices=val_indices,
            num_examples_per_epoch=num_examples_per_epoch,
        )
        self.test_ds = DSD100Dataset(
            root_dir=dataset_dir,
            length=val_n_samples,
            sample_rate=sr,
            indices=test_indices,
            num_examples_per_epoch=num_examples_per_epoch,
        )

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


class MedleyDBDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        dataset_dirs: List[str],
        sr: int,
        train_n_samples: int,
        val_n_samples: int,
        train_indices: List[int],
        val_indices: List[int],
        test_indices: List[int],
        max_n_tracks: int,
        num_examples_per_epoch: int,
        buffer_size_gb: float = 2.0,
        buffer_reload_rate: int = 4000,
        normalization: str = "peak",
        num_workers: int = 0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_dirs = dataset_dirs
        self.sr = sr
        self.train_n_samples = train_n_samples
        self.val_n_samples = val_n_samples
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        self.max_n_tracks = max_n_tracks
        self.num_examples_per_epoch = num_examples_per_epoch
        self.buffer_size_gb = buffer_size_gb
        self.buffer_reload_rate = buffer_reload_rate
        self.normalization = normalization
        self.num_workers = num_workers

        self.train_ds = MedleyDBDataset(
            root_dirs=dataset_dirs,
            length=train_n_samples,
            sample_rate=sr,
            indices=train_indices,
            max_num_tracks=max_n_tracks,
            num_examples_per_epoch=num_examples_per_epoch,
            buffer_size_gb=buffer_size_gb,
            buffer_reload_rate=buffer_reload_rate,
            buffer_audio_length=train_n_samples,
            normalization=normalization,
        )
        self.val_ds = MedleyDBDataset(
            root_dirs=dataset_dirs,
            length=val_n_samples,
            sample_rate=sr,
            indices=val_indices,
            max_num_tracks=max_n_tracks,
            num_examples_per_epoch=num_examples_per_epoch,
            buffer_size_gb=buffer_size_gb,
            buffer_reload_rate=buffer_reload_rate,
            buffer_audio_length=val_n_samples,
            normalization=normalization,
        )
        self.test_ds = MedleyDBDataset(
            root_dirs=dataset_dirs,
            length=val_n_samples,
            sample_rate=sr,
            indices=test_indices,
            max_num_tracks=max_n_tracks,
            num_examples_per_epoch=num_examples_per_epoch,
            buffer_size_gb=buffer_size_gb,
            buffer_reload_rate=buffer_reload_rate,
            buffer_audio_length=val_n_samples,
            normalization=normalization,
        )

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
