"""
This script computes the Jacobian of (Phi \circ g)
for one random seed of the chirp texture dataset.
"""
import datetime
import fire
import kymatio
from kymatio.torch import TimeFrequencyScattering
import os
import sys
import time
import torch

import cnn
import synth


def run(density_idx, slope_idx, seed, sav_dir, job_id):
    # Print header
    start_time = int(time.time())
    print("Job ID: " + str(job_id))
    print(str(datetime.datetime.now()) + " Start.")
    print(__doc__ + "\n")
    print("density_idx: " + str(density_idx))
    print("slope_idx: " + str(slope_idx))
    print("seed: " + str(seed))
    print("\n")

    # Print version numbers
    for module in [kymatio, torch]:
        print("{} version: {:s}".format(module.__name__, module.__version__))
    print("\n")
    sys.stdout.flush()

    dataset = cnn.ChirpTextureDataModule(
        n_densities=7, n_slopes=7, n_folds=1, batch_size=1
    )
    dataset.setup()

    sub_df = dataset.df[
        (dataset.df["density_idx"] == density_idx)
        & (dataset.df["slope_idx"] == slope_idx)
    ]
    if len(sub_df) == 1:
        row = sub_df.iloc[0]
        theta = torch.tensor([row["density"], row["slope"]], requires_grad=True)
    else:
        raise ValueError("Expected one row, got:\n {}".format(sub_df))

    sc = TimeFrequencyScattering(
        shape=(2**15),
        J=6,
        Q=(24, 2),
        Q_fr=2,
        J_fr=5,
        T="global",
        F="global",
        format="time",
    )

    def S_from_theta(theta, sc, dataset, seed):
        x = synth.generate_chirp_texture(
            theta_density=theta[0],
            theta_slope=theta[1],
            duration=dataset.train_ds.duration,
            event_duration=dataset.train_ds.event_duration,
            sr=dataset.train_ds.sr,
            fmin=dataset.train_ds.fmin,
            fmax=dataset.train_ds.fmax,
            n_events=dataset.train_ds.n_events,
            Q=dataset.train_ds.Q,
            hop_length=dataset.train_ds.hop_length,
            seed=seed,
        )
        return sc(x)

    S_closure = lambda theta: S_from_theta(theta, sc, dataset, seed)
    J = torch.autograd.functional.jacobian(S_closure, theta)

    # save the coefficients to sav_dir with formatted name
    jtfs_name = "jtfs_density-{}_slope-{}_seed-{}.pt".format(
        density_idx, slope_idx, seed
    )
    jtfs_dir = os.path.join(sav_dir, "jtfs_7x7")
    os.makedirs(jtfs_dir, exist_ok=True)
    jtfs_path = os.path.join(jtfs_dir, jtfs_name)
    torch.save(S_closure(theta), jtfs_path)

    # save the Jacobian to sav_dir with formatted name
    jac_name = "jacobian_density-{}_slope-{}_seed-{}.pt".format(
        density_idx, slope_idx, seed
    )
    jac_dir = os.path.join(sav_dir, "jacobians_7x7")
    os.makedirs(jac_dir, exist_ok=True)
    jac_path = os.path.join(jac_dir, jac_name)
    torch.save(J.squeeze(), jac_path)

    print("Coefficients saved at: {}".format(jtfs_path))
    print("Jacobian saved at: {}".format(jac_path))
    print("\n")

    # Print elapsed time.
    print(str(datetime.datetime.now()) + " Success.")
    elapsed_time = time.time() - int(start_time)
    elapsed_hours = int(elapsed_time / (60 * 60))
    elapsed_minutes = int((elapsed_time % (60 * 60)) / 60)
    elapsed_seconds = elapsed_time % 60.0
    elapsed_str = "{:>02}:{:>02}:{:>05.2f}".format(
        elapsed_hours, elapsed_minutes, elapsed_seconds
    )
    print("Total elapsed time: " + elapsed_str + ".")


if __name__ == "__main__":
    fire.Fire(run)
