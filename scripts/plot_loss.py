import logging
import os
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from experiments.paths import OUT_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    # tsv_path = os.path.join(OUT_DIR, "loss.tsv")
    tsv_path = "/Users/puntland/local_christhetree/aim/scrapl-ddsp/out/out/scrapl_adamw_5e-5_b32__texture_32_32_5_meso.tsv"
    stage = "train"
    x_col = "step"
    y_col = "l1_theta"

    # Load using pandas
    df = pd.read_csv(tsv_path, sep="\t", index_col=False)
    # exit()

    # Filter out stage
    df = df[df["stage"] == stage]

    grouped = df.groupby("seed")
    # Create list of y_col values for each x_col for each seed
    data = {name: (group[x_col].values, group[y_col].values) for name, group in grouped}
    data_dict = defaultdict(list)
    for seed, (x, y) in data.items():
        assert len(x) == len(y)
        for curr_x, curr_y in zip(x, y):
            data_dict[curr_x].append(curr_y)
    data = data_dict

    # data = {
    #     0.1: [0.2, 0.25, 0.22, 0.21],
    #     0.2: [0.18, 0.20, 0.19, 0.21],
    #     0.3: [0.15, 0.14, 0.16, 0.13],
    #     # Add more timestamps and loss values
    # }

    # Extract timestamps and loss values
    timestamps = np.array(list(data.keys()))
    loss_values = np.array(list(data.values()))

    # Calculate statistics
    means = np.mean(loss_values, axis=1)
    std_devs = np.std(loss_values, axis=1)
    sem = std_devs / np.sqrt(loss_values.shape[1])  # Standard error of the mean
    conf_interval = 1.96 * sem  # 95% confidence interval
    loss_min = np.min(loss_values, axis=1)
    loss_max = np.max(loss_values, axis=1)

    # Plot
    plt.figure(figsize=(10, 6))

    # Main line plot for mean loss
    plt.plot(timestamps, means, label="Mean Loss", color="blue", lw=2)

    # Shaded region for confidence interval
    plt.fill_between(
        timestamps,
        means - conf_interval,
        means + conf_interval,
        color="blue",
        alpha=0.2,
        label="95% CI",
    )

    # Shaded region for range
    plt.fill_between(
        timestamps, loss_min, loss_max, color="gray", alpha=0.3, label="Range"
    )

    # Labels and legend
    plt.xlabel("Timestamp")
    plt.ylabel("Loss")
    plt.title("Loss Over Time with 95% CI and Range")
    plt.legend()
    plt.grid(True)
    plt.show()
