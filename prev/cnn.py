import auraloss
import itertools
from kymatio.torch import TimeFrequencyScattering
from nnAudio.features import CQT
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from scrapl.torch import TimeFrequencyScrapl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision

import loss
import synth


class EffNet(pl.LightningModule):
    def __init__(self, loss_type, save_path, steps_per_epoch):
        self.seed = None
        self.J = 5
        self.Q = 24
        self.sr = 2**13
        self.hop_length = 2**6
        self.cqt_epsilon = 1e-3
        self.duration = 4
        self.event_duration = 2 ** (-2)
        self.fmin = 2**8
        self.fmax = 2**11
        self.n_events = 2**6
        self.sr = 2**13

        super().__init__()
        self.batchnorm1 = nn.BatchNorm2d(
            1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )

        # adapt to EfficientNet's mandatory 3 input channels
        n_hidden = 128
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1, 1))
        self.effnet = torchvision.models.efficientnet_b0(num_classes=n_hidden)
        self.batchnorm2 = nn.BatchNorm1d(n_hidden)
        self.dense = nn.Linear(n_hidden, 2, bias=False)
        self.n_batches_train = steps_per_epoch

        self.loss_type = loss_type
        if self.loss_type == "mss":
            self.spectral_distance = auraloss.freq.MultiResolutionSTFTLoss()
        elif self.loss_type == "jtfs":
            self.jtfs = TimeFrequencyScattering(
                shape=(2**15),
                J=6,
                Q=(24, 2),
                Q_fr=2,
                J_fr=5,
                T="global",
                F="global",
                format="time",
            )

            def jtfs_distance(x, x_pred):
                Sx = self.jtfs(x)[1:]
                Sx_pred = self.jtfs(x_pred)[1:]
                return torch.linalg.vector_norm(Sx - Sx_pred, p=2, dim=1)

            self.spectral_distance = jtfs_distance
        elif self.loss_type == "scrapl":
            self.scrapl = TimeFrequencyScrapl(
                shape=(2**15),
                J=6,
                Q=(24, 2),
                Q_fr=2,
                J_fr=5,
                T="global",
                F="global",
            )
            scrapl_meta = self.scrapl.meta()
            self.scrapl_keys = [key for key in scrapl_meta["key"] if len(key) == 2]

            def scrapl_distance(x, x_pred):
                n2, n_fr = np.random.choice(self.scrapl_keys)
                Sx = self.scrapl.scattering_singlepath(x, n2, n_fr)
                Sx_pred = self.scrapl.scattering_singlepath(x_pred, n2, n_fr)
                return torch.linalg.vector_norm(Sx - Sx_pred, p=2, dim=1)

            self.spectral_distance = scrapl_distance

        # TODO: Open-L3 loss

        self.save_path = save_path
        self.val_loss = None
        # self.metric_macro = metrics.JTFSloss()
        # self.metric_mss = metrics.MSSloss()
        self.monitor_valloss = torch.inf
        self.current_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.best_params = self.parameters
        self.epoch = 0

        self.test_preds = []
        self.test_gts = []
        self.train_outputs = []
        self.test_outputs = []
        self.val_outputs = []

        self.optimizer = self.configure_optimizers()

    def forward(self, input_tensor):
        input_tensor = input_tensor.unsqueeze(1)
        x = self.batchnorm1(input_tensor)
        x = self.conv2d(x)
        x = self.effnet(x)
        # apply batch normalization onto x
        x = self.batchnorm2(x)
        x = self.dense(x)

        density = torch.sigmoid(x[:, 0])
        slope = torch.tanh(x[:, 1])
        return {"density": density, "slope": slope}

    def step(self, batch, subset):
        U = batch["feature"].to(self.current_device)
        density = batch["density"].to(self.current_device)
        slope = batch["slope"].to(self.current_device)

        theta_pred = self(U)
        density_pred = theta_pred["density"]
        slope_pred = theta_pred["slope"]
        if self.loss_type == "ploss":
            density_loss = F.mse_loss(density_pred, density)
            slope_loss = F.mse_loss(slope_pred, slope)
            loss = density_loss + slope_loss
        else:  # spectral loss
            x, x_pred = [], []
            for i in range(U.shape[0]):
                x.append(self.x_from_theta(density[i], slope[i]))
                x_pred.append(self.x_from_theta(density_pred[i], slope_pred[i]))
            x = torch.stack(x)
            x_pred = torch.stack(x_pred)
            if self.loss_type == "mss":
                loss = self.spectral_distance(x.unsqueeze(1), x_pred.unsqueeze(1))
            elif self.loss_type == "jtfs":
                loss = self.spectral_distance(x, x_pred)

        if subset == "train":
            self.train_outputs.append(loss)
            self.log("train_loss", loss, prog_bar=True)
        elif subset == "test":
            self.test_outputs.append(loss)
            self.test_preds.append(theta_pred)
            theta_ref = {"density": density, "slope": slope}
            self.test_gts.append(theta_ref)
        elif subset == "val":
            self.val_outputs.append(loss)
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def on_train_epoch_start(self):
        self.train_outputs = []
        self.test_outputs = []
        self.val_outputs = []
        self.log("lr", self.optimizer.param_groups[-1]["lr"])

    def on_train_epoch_end(self):
        avg_loss = torch.tensor(self.train_outputs).mean()
        self.log("train_loss", avg_loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def x_from_theta(self, theta_density, theta_slope):
        return synth.generate_chirp_texture(
            theta_density,
            theta_slope,
            duration=self.duration,
            event_duration=self.event_duration,
            sr=self.sr,
            fmin=self.fmin,
            fmax=self.fmax,
            n_events=self.n_events,
            Q=self.Q,
            hop_length=self.hop_length,
            seed=self.seed,
        )


class ChirpTextureData(Dataset):
    def __init__(self, df):
        super().__init__()

        self.df = df
        self.J = 5
        self.Q = 24
        self.sr = 2**13
        self.hop_length = 2**6

        self.cqt_epsilon = 1e-3
        self.duration = 4
        self.event_duration = 2 ** (-2)
        self.fmin = 2**8
        self.fmax = 2**11
        self.n_events = 2**6
        self.sr = 2**13

        # define CQT closure
        cqt_params = {
            "sr": self.sr,
            "bins_per_octave": self.Q,
            "n_bins": self.J * self.Q,
            "hop_length": self.hop_length,
            "fmin": (0.4 * self.sr) / (2**self.J),
        }
        # TODO(cm)
        # if torch.cuda.is_available():
        #     self.cqt_function = CQT(**cqt_params).cuda()
        # else:
        #     self.cqt_function = CQT(**cqt_params)

    def __getitem__(self, idx):
        theta_density = torch.tensor(self.df.iloc[idx]["density"], dtype=torch.float32)
        theta_slope = torch.tensor(self.df.iloc[idx]["slope"], dtype=torch.float32)
        seed = self.df.iloc[idx]["seed"]

        x = synth.generate_chirp_texture(
            theta_density,
            theta_slope,
            duration=self.duration,
            event_duration=self.event_duration,
            sr=self.sr,
            fmin=self.fmin,
            fmax=self.fmax,
            n_events=self.n_events,
            Q=self.Q,
            hop_length=self.hop_length,
            seed=seed,
        )
        U = self.cqt_from_x(x)
        return {"feature": U, "density": theta_density, "slope": theta_slope}

    def __len__(self):
        return len(self.df)

    def cqt_from_x(self, x):
        CQT_x = self.cqt_function(x).abs()
        n_columns = CQT_x.shape[2]
        CQT_x = CQT_x[0, :, (n_columns // 4) : (3 * n_columns // 4)]
        # return CQT_x
        return torch.log1p(CQT_x / self.cqt_epsilon)


class ChirpTextureDataModule(pl.LightningDataModule):
    def __init__(self, n_densities, n_slopes, n_seeds_per_fold, n_folds, batch_size):
        super().__init__()

        self.n_densities = n_densities
        self.n_slopes = n_slopes
        self.n_seeds_per_fold = n_seeds_per_fold
        self.n_folds = n_folds
        self.batch_size = batch_size

        slope_idx = np.arange(n_slopes)
        density_idx = np.arange(n_densities)
        seeds = np.arange(n_seeds_per_fold * n_folds)

        theta_idx = list(itertools.product(density_idx, slope_idx, seeds))
        df_idx = pd.DataFrame(theta_idx, columns=["density_idx", "slope_idx", "seed"])
        slopes = np.linspace(-1, 1, n_slopes + 2)[1:-1]
        densities = np.linspace(0, 1, n_densities + 2)[1:-1]
        thetas = list(itertools.product(densities, slopes, seeds))
        df = pd.DataFrame(thetas, columns=["density", "slope", "seeds_bis"])
        del df["seeds_bis"]
        df = df_idx.merge(df, left_index=True, right_index=True)

        folds = df["seed"] % n_folds
        df["fold"] = folds
        self.df = df

    def setup(self, stage=None):
        train_df = self.df[self.df["fold"] < (self.n_folds - 2)]
        self.train_ds = ChirpTextureData(train_df)

        val_df = self.df[self.df["fold"] == (self.n_folds - 2)]
        self.val_ds = ChirpTextureData(val_df)

        test_df = self.df[self.df["fold"] == (self.n_folds - 1)]
        self.test_ds = ChirpTextureData(test_df)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)
