import cnn
import datetime
from datetime import timedelta
import fire
import kymatio
import os
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import time
import torch
import sys


def run(loss_type, sav_dir, job_id):
    # Print header
    start_time = int(time.time())
    print(str(datetime.datetime.now()) + " Start.")
    print(__doc__ + "\n")
    print("Loss type: " + loss_type)
    print("Save directory: " + sav_dir)
    print("Job ID: " + str(job_id))
    print("\n")

    # Print version numbers
    for module in [kymatio, torch, pl]:
        print("{} version: {:s}".format(module.__name__, module.__version__))
    print("\n")
    sys.stdout.flush()

    n_densities = 32
    n_slopes = 32
    n_folds = 8
    batch_size = 64

    dataset = cnn.ChirpTextureDataModule(
        n_densities=n_densities,
        n_slopes=n_slopes,
        n_folds=n_folds,
        batch_size=batch_size,
    )

    dataset.setup()

    samples_per_epoch = 768
    steps_per_epoch = samples_per_epoch / dataset.batch_size

    steps_per_epoch = 1
    models_dir = os.path.join(sav_dir, "models_{}".format(loss_type))
    logs_dir = os.path.join(sav_dir, "logs_{}".format(loss_type))
    model = cnn.EffNet(loss_type, models_dir, steps_per_epoch)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=logs_dir)

    trainer = pl.Trainer(
        max_epochs=-1,
        limit_train_batches=steps_per_epoch,
        callbacks=[],
        logger=tb_logger,
        max_time=timedelta(hours=12),
    )

    trainer.fit(model, dataset)

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
