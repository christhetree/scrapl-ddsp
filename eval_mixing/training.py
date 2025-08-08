import glob
import os
from argparse import Namespace

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio
from torch.utils.data import DataLoader

from automix.data import DSD100Dataset
from automix.system import System


def train(args: Namespace, system: System) -> None:
    # setup callbacks
    callbacks = [
        # LogAudioCallback(),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.ModelCheckpoint(
            filename=f"{args.dataset_name}-{args.automix_model}"
            + "_epoch-{epoch}-step-{step}",
            monitor="val/loss_epoch",
            mode="min",
            save_last=True,
            auto_insert_metric_name=False,
        ),
    ]

    # we not will use weights and biases
    # wandb_logger = WandbLogger(save_dir=log_dir, project="automix-notebook")

    # create PyTorch Lightning trainer
    # trainer = pl.Trainer(args, callbacks=callbacks)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=callbacks,
        # Add other trainer arguments here if needed
    )

    train_dataset = DSD100Dataset(
        args.dataset_dir,
        args.train_length,
        44100,
        indices=[0, 4],
        num_examples_per_epoch=100,
    )
    val_dataset = DSD100Dataset(
        args.dataset_dir,
        args.val_length,
        44100,
        indices=[0, 4],
        num_examples_per_epoch=100,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        # persistent_workers=true,
        persistent_workers=False,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=False,
    )
    trainer.fit(system, train_dataloader, val_dataloader)


# Test =================================================================================
def test(system: System) -> None:
    start_sample = 262144 * 2
    end_sample = 262144 * 3

    # load the input tracks
    track_dir = "DSD100subset/Sources/Dev/081 - Patrick Talbot - Set Me Free/"
    track_ext = "wav"

    track_filepaths = glob.glob(os.path.join(track_dir, f"*.{track_ext}"))
    track_filepaths = sorted(track_filepaths)
    track_names = []
    tracks = []
    for idx, track_filepath in enumerate(track_filepaths):
        x, sr = torchaudio.load(track_filepath)
        x = x[:, start_sample:end_sample]

        for n in range(x.shape[0]):
            x_sub = x[n : n + 1, :]

            gain_dB = np.random.rand() * 12
            gain_dB *= np.random.choice([1.0, -1.0])
            gain_ln = 10 ** (gain_dB / 20.0)
            x_sub *= gain_ln

            tracks.append(x_sub)
            track_names.append(os.path.basename(track_filepath))
            # IPython.display.display(
            #     ipd.Audio(x[n, :].view(1, -1).numpy(), rate=sr, normalize=True)
            # )
            print(idx + 1, os.path.basename(track_filepath))

    # add dummy tracks of silence if needed
    if system.hparams.automix_model == "mixwaveunet" and len(tracks) < 8:
        tracks.append(torch.zeros(x.shape))

    # stack tracks into a tensor
    tracks = torch.stack(tracks, dim=0)
    tracks = tracks.permute(1, 0, 2)
    # tracks have shape (1, num_tracks, seq_len)
    print(tracks.shape)

    # listen to the input (mono) before mixing
    input_mix = tracks.sum(dim=1, keepdim=True)
    input_mix /= input_mix.abs().max()
    print(input_mix.shape)
    plt.figure(figsize=(10, 2))
    librosa.display.waveshow(input_mix.view(2, -1).numpy(), sr=sr, zorder=3)
    plt.ylim([-1, 1])
    plt.grid(c="lightgray")
    plt.show()
    # IPython.display.display(
    #     ipd.Audio(input_mix.view(1, -1).numpy(), rate=sr, normalize=False)
    # )

    tracks = tracks.view(1, 8, -1)

    with torch.no_grad():
        y_hat, p = system(tracks)

    # view the mix
    print(y_hat.shape)
    y_hat /= y_hat.abs().max()
    plt.figure(figsize=(10, 2))
    librosa.display.waveshow(y_hat.view(2, -1).cpu().numpy(), sr=sr, zorder=3)
    plt.ylim([-1, 1])
    plt.grid(c="lightgray")
    plt.show()
    # IPython.display.display(
    #     ipd.Audio(y_hat.view(2, -1).cpu().numpy(), rate=sr, normalize=True)
    # )

    # print the parameters
    if system.hparams.automix_model == "dmc":
        for track_fp, param in zip(track_names, p.squeeze()):
            print(os.path.basename(track_fp), param)


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)

    os.makedirs("checkpoints/", exist_ok=True)
    # !wget https://huggingface.co/csteinmetz1/automix-toolkit/resolve/main/encoder.ckpt
    # !mv encoder.ckpt checkpoints/encoder.ckpt

    # !wget https://huggingface.co/csteinmetz1/automix-toolkit/resolve/main/DSD100subset.zip
    # !unzip -o DSD100subset.zip

    args = {
        "dataset_dir": "./DSD100subset",
        "dataset_name": "DSD100",
        "automix_model": "dmc",
        "pretrained_encoder": True,
        "train_length": 65536,
        "val_length": 65536,
        "accelerator": "cpu",  # you can select "cpu" or "gpu"
        "devices": 1,
        "batch_size": 4,
        "lr": 3e-4,
        "max_epochs": 10,
        "schedule": "none",
        "recon_losses": ["sd"],
        # "recon_losses": ["mss"],
        # "recon_losses": ["scrapl"],
        "recon_loss_weights": [1.0],
        "sample_rate": 44100,
        "num_workers": 0,
    }
    args = Namespace(**args)

    # create the System
    system = System(**vars(args))

    train(args, system)
    # test(system)
