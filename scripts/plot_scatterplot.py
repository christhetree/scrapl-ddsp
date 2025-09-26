import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from experiments.paths import OUT_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "WARNING"))


if __name__ == "__main__":
    model_step_speed = 54.727
    p_loss_speed = 0.5158
    p_loss_l1 = 20.505

    orig_coords = [
        ("JTFS", 42.377, 0, 1731),
        # ("SCRAPL (no $\\theta$-IS)", 73.809, 13.376, 89.82),
        ("SCRAPL", 65.68, 4.207, 89.82),
        ("MSS Linear", 370.14, 0.52049, 26.26),
        ("MSS Log + Linear", 259.1, 1.7071, 19.06),
        ("MSS Revisited", 311.06, 19.407, 16.96),
        ("MSS Random", 195.47, 4.1916, 24.71),
        ("MS-CLAP", 165.85, 8.2092, 75.55),
        ("PANNs", 158.94, 4.3815, 29.29),
    ]

    coords = []
    for name, y_mean, y_ci, t in orig_coords:
        t += model_step_speed

        y_ci_p = (y_mean + y_ci) / p_loss_l1
        y_ci_n = (y_mean - y_ci) / p_loss_l1
        y_mean /= p_loss_l1
        y_ci_p = y_ci_p - y_mean
        y_ci_n = y_mean - y_ci_n
        t /= p_loss_speed + model_step_speed
        coords.append((name, y_mean, y_ci_p, y_ci_n, t))

    names, y, y_ci_p, y_ci_n, x = zip(*coords)

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    y_ci_p = np.array(y_ci_p, dtype=float)  # positive CI
    y_ci_n = np.array(y_ci_n, dtype=float)  # negative CI
    names = np.array(names)

    plt.figure(figsize=(11, 4), dpi=300)
    plt.rcParams.update({"font.size": 14})

    # Error bars with asymmetric CIs
    plt.errorbar(
        x,
        y,
        yerr=[y_ci_n, y_ci_p],  # [lower, upper]
        fmt="o",
        markersize=6,
        color="black",
        ecolor="black",  # error bar colora
        elinewidth=1.5,  # error bar thickness
        capsize=3,  # cap size in points
        capthick=1.5,  # cap thickness
        alpha=1.0,
        barsabove=False,
    )

    # Annotate names next to points
    texts = []
    fontsize = 12.5
    for xi, yi, label in zip(x, y, names):
        if label == "JTFS":
            xi *= 0.83
        elif label == "SCRAPL":
            fontsize = 14
            xi *= 1.05
        else:
            xi *= 1.05
        texts.append(
            plt.text(xi, yi, label, fontsize=fontsize, ha="left", va="center")
        )  # Increased font size

    # Plt a single point at 1, 1,
    plt.plot(1.0, 1.0, "ro", markersize=6)  # Red point
    plt.text(
        1.05, 1.0, "Supervised", fontsize=fontsize, ha="left", va="center", color="r"
    )

    # Log-log scales
    plt.xscale("log")
    plt.yscale("log")

    plt.xticks(
        [1, 2, 4, 8, 16, 32],
        ["1", "2", "4", "8", "16", "32"],
    )
    plt.xlim(0.9, 34)
    plt.yticks(
        # [2, 4, 8, 16],
        # ["2", "4", "8", "16"],
        [1, 2, 4, 8, 16],
        ["1", "2", "4", "8", "16"],
    )
    # set ylim
    plt.ylim(0.85, 24)
    # Disable minor ticks
    plt.gca().xaxis.set_minor_locator(ticker.NullLocator())
    plt.gca().yaxis.set_minor_locator(ticker.NullLocator())

    # Invert y-axis
    plt.gca().invert_yaxis()

    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel("Relative Computational Cost")
    plt.ylabel("Relative Error")
    plt.tight_layout()

    # Save as .PDF
    out_path = os.path.join(OUT_DIR, "fig_scatterplot.pdf")
    plt.savefig(out_path, format="pdf", dpi=300)
    plt.show()
