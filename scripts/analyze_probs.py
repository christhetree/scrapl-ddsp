import logging
import os

import matplotlib.pyplot as plt
import torch as tr

from experiments.paths import DATA_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "WARNING"))


if __name__ == "__main__":
    prob_names_and_paths = [
        ("none_1_bs", os.path.join(DATA_DIR, "adaptive_scrapl/scrapl_saga_pwa_1e-5__texture_32_32_5_meso_b32__log_probs__n_theta_2__n_params_28__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt")),
        ("none_10_bs", os.path.join(DATA_DIR, "adaptive_scrapl/scrapl_saga_pwa_1e-5__texture_32_32_5_meso_b32__log_probs__n_theta_2__n_params_28__n_batches_10__n_iter_20__min_prob_frac_0.0__param_agg_none__seed_0.pt")),
        # ("mean_1_bs", os.path.join(DATA_DIR, "adaptive_scrapl/scrapl_saga_pwa_1e-5__texture_32_32_5_meso_b32__log_probs__n_theta_2__n_params_28__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_mean__seed_0.pt")),
        # ("max_1_bs", os.path.join(DATA_DIR, "adaptive_scrapl/scrapl_saga_pwa_1e-5__texture_32_32_5_meso_b32__log_probs__n_theta_2__n_params_28__n_batches_1__n_iter_20__min_prob_frac_0.0__param_agg_max__seed_0.pt")),
    ]
    n_theta, n_classes = tr.load(prob_names_and_paths[0][1]).shape
    unif_prob = 1 / n_classes

    for theta_idx in range(n_theta):
        sorted_indices = None
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, probs_path in prob_names_and_paths:
            log_probs = tr.load(probs_path)
            assert log_probs.shape == (n_theta, n_classes)
            probs = log_probs[theta_idx, :].exp()
            assert tr.allclose(
                probs.sum(), tr.tensor(1.0, dtype=tr.double), atol=1e-12
            ), f"self.probs.sum() = {probs.sum()}"
            if sorted_indices is None:
                sorted_indices = tr.argsort(probs, descending=True)
            probs = probs[sorted_indices]

            ax.plot(
                probs.numpy(),
                label=name,
                # marker="s",
                # markersize=5,
                linewidth=2,
                # linewidth=0,
            )
            # n = 30
            # indices = list(range(n_classes))[:n]
            # ax.bar(x=indices, height=probs.numpy()[:n], label=name, alpha=0.5)

            ax.set_xlabel("Path (sorted by probability)")
            ax.set_ylabel("Probability")
            ax.set_ylim(bottom=0.0)
            ax.set_title(f"Probabilities for Theta {theta_idx}")
            ax.legend()
        ax.axhline(
            unif_prob,
            color="orange",
            linestyle="--",
            linewidth=2,
            label="unif",
        )
        plt.show()
