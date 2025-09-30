import logging
import os
from typing import List, Tuple
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
        drum_types: List[str],
        n_delta_per_item: int = 1,
        delta_min: int = -2048,
        delta_max: int = 2048,
    ):
        super().__init__()
        assert samples.size(0) == len(drum_types)
        self.samples = samples
        assert set(drum_types) == {"BD", "SD", "HH", "Tom"}
        self.drum_types = drum_types
        self.n_delta_per_item = n_delta_per_item
        self.delta_min = delta_min
        self.delta_max = delta_max

        self.deltas = tr.randint(
            delta_min,
            delta_max + 1,
            (self.samples.size(0) * n_delta_per_item,),
        )
        # deltas = tr.randint(
        #     delta_max // 2,
        #     delta_max + 1,
        #     (self.samples.size(0) * n_delta_per_item,),
        # )
        # signs = tr.randint(0, 2, (self.samples.size(0) * n_delta_per_item,)) * 2 - 1
        # self.deltas = deltas * signs

    def __len__(self) -> int:
        return self.samples.size(0) * self.n_delta_per_item

    def __getitem__(self, idx: int) -> Tuple[T, str, T]:
        sample_idx = idx // self.n_delta_per_item
        sample = self.samples[sample_idx]
        drum_type = self.drum_types[sample_idx]
        delta = self.deltas[idx]
        return sample, drum_type, delta


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
        n_delta_per_item: int = 1,
        delta_min: int = 0,
        delta_max: int = 0,
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
        self.n_delta_per_item = n_delta_per_item
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.shuffle_seed = shuffle_seed
        self.num_workers = num_workers

        log.info(
            f"shuffle_seed: {shuffle_seed}, "
            f"delta_min: {delta_min}, "
            f"delta_max: {delta_max}, "
            f"n_delta_per_item: {n_delta_per_item}"
        )
        sample_paths = [p for p in os.listdir(root_dir) if p.endswith(".wav")]
        log.info(f"Found {len(sample_paths)} .wav files in {root_dir}")
        assert len(sample_paths) >= (
            n_train + n_val + n_test
        ), f"Not enough .wav files in {root_dir}"
        sample_paths = sorted(sample_paths)
        np.random.seed(shuffle_seed)
        np.random.shuffle(sample_paths)

        samples = []
        drum_types = []
        for p in sample_paths:
            f_name = os.path.basename(p)
            if f_name.startswith("BD "):
                drum_types.append("BD")
            elif f_name.startswith("SD "):
                drum_types.append("SD")
            elif f_name.startswith("OH "):
                drum_types.append("HH")
            elif f_name.startswith("CH "):
                drum_types.append("HH")
            elif f_name.startswith("Tom "):
                drum_types.append("Tom")
            else:
                raise ValueError(f"Unknown drum type in file name: {f_name}")

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
        # Print drum type stats
        unique, counts = np.unique(drum_types, return_counts=True)
        drum_type_counts = dict(zip(unique, counts))
        log.info(f"Drum type counts: {drum_type_counts}")

        train_samples = samples[:n_train]
        train_drum_types = drum_types[:n_train]
        unique, counts = np.unique(train_drum_types, return_counts=True)
        drum_type_counts = dict(zip(unique, counts))
        log.info(f"Train drum type counts: {drum_type_counts}")

        val_samples = samples[n_train : n_train + n_val]
        val_drum_types = drum_types[n_train : n_train + n_val]
        unique, counts = np.unique(val_drum_types, return_counts=True)
        drum_type_counts = dict(zip(unique, counts))
        log.info(f"Validation drum type counts: {drum_type_counts}")

        test_samples = samples[n_train + n_val : n_train + n_val + n_test]
        test_drum_types = drum_types[n_train + n_val : n_train + n_val + n_test]
        unique, counts = np.unique(test_drum_types, return_counts=True)
        drum_type_counts = dict(zip(unique, counts))
        log.info(f"Test drum type counts: {drum_type_counts}")

        # for idx, dt in enumerate(test_drum_types):
        #     print(f'{idx}: "{dt}",')
        # exit()

        self.train_dataset = WavDataset(
            train_samples, train_drum_types, n_delta_per_item, delta_min, delta_max
        )
        self.val_dataset = WavDataset(
            val_samples, val_drum_types, n_delta_per_item, delta_min, delta_max
        )
        self.test_dataset = WavDataset(
            test_samples, test_drum_types, n_delta_per_item, delta_min, delta_max
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
