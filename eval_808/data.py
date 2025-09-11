import logging
import os
from typing import List
import numpy as np
import pytorch_lightning as pl
import torch as tr
from torch import Tensor as T
from torch.utils.data import DataLoader, Dataset
import torchaudio

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


class WavDataset(Dataset):
    def __init__(
        self,
        samples: T,
    ):
        super().__init__()
        self.samples = samples

    def __len__(self) -> int:
        return self.samples.size(0)

    def __getitem__(self, idx: int) -> T:
        return self.samples[idx]


class WavDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        root_dir: str,
        sr: int,
        n_samples: int,
        n_train: int,
        n_val: int,
        n_test: int,
        shuffle_seed: int = 42,
        num_workers: int = 0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.sr = sr
        self.n_samples = n_samples
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.num_workers = num_workers

        sample_paths = [p for p in os.listdir(root_dir) if p.endswith(".wav")]
        log.info(f"Found {len(sample_paths)} .wav files in {root_dir}")
        assert len(sample_paths) >= (n_train + n_val + n_test), (
            f"Not enough .wav files in {root_dir}"
        )
        sample_paths = sorted(sample_paths)
        np.random.seed(shuffle_seed)
        np.random.shuffle(sample_paths)

        samples = []
        for p in sample_paths:
            sample, sample_sr = torchaudio.load(os.path.join(root_dir, p))
            assert sample_sr == sr
            if sample.size(0) > 1:
                sample = tr.mean(sample, dim=0, keepdim=True)
                log.warning(f"File {p} has more than 1 channel")
            if sample.size(1) >= n_samples:
                sample = sample[:, :n_samples]
            else:
                sample = tr.nn.functional.pad(
                    sample, (0, n_samples - sample.size(1)), mode="constant", value=0.0
                )
            samples.append(sample)
        samples = tr.stack(samples, dim=0)

        train_samples = samples[:n_train]
        val_samples = samples[n_train : n_train + n_val]
        test_samples = samples[n_train + n_val : n_train + n_val + n_test]

        self.train_dataset = WavDataset(train_samples)
        self.val_dataset = WavDataset(val_samples)
        self.test_dataset = WavDataset(test_samples)

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
