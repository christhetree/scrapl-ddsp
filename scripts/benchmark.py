import logging
import os

from nnAudio.features import CQT

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from typing import Callable, Any, Dict, Optional, Tuple
import pandas as pd
from torch.utils.benchmark import Measurement
from tqdm import tqdm

from experiments import util
from experiments.paths import CONFIGS_DIR, OUT_DIR

from torch import Tensor as T, nn
import torch as tr
from torch.utils import benchmark

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

batch_size = 4
n_samples = 32768
# n_samples = 2
# n_samples = 10
num_threads = 1
min_run_time = 15.0
# device = "cpu"
device = "cuda"


def benchmark_loss_func(
    x: T, x_hat: T, loss_func: Callable[..., T], path_idx: Optional[int] = None
) -> None:
    if path_idx is None:
        loss = loss_func(x_hat, x)
    else:
        loss = loss_func(x_hat, x, path_idx=path_idx)
    loss.backward()


def calc_benchmark_stats(
    globals: Dict[str, Any],
    device: str,
    num_threads: int = 1,
    min_run_time: float = 1.0,
) -> Tuple[float, float, int, float]:
    if device.startswith("cuda"):
        tr.cuda.reset_peak_memory_stats(device)
    result: Measurement = benchmark.Timer(
        stmt="fn(x, x_hat, loss_func, path_idx)",
        globals=globals,
        num_threads=num_threads,
    ).blocked_autorange(min_run_time=min_run_time)
    max_mem = 0.0
    if device.startswith("cuda"):
        max_mem = tr.cuda.max_memory_allocated(device) / (1024**2)
    n_runs = len(result.times)
    return result.median, result.iqr, n_runs, max_mem


def main() -> None:
    log.info(
        f"Benchmarking with device={device}, num_threads={num_threads}, "
        f"min_run_time={min_run_time}, batch_size={batch_size}, "
        f"n_samples={n_samples}"
    )
    tr.manual_seed(42)

    config_path = os.path.join(CONFIGS_DIR, "losses/mss_meso.yml")
    mss_lin_loss = util.load_class_from_yaml(config_path)
    config_path = os.path.join(CONFIGS_DIR, "losses/mss_meso_log.yml")
    mss_log_loss = util.load_class_from_yaml(config_path)
    config_path = os.path.join(CONFIGS_DIR, "losses/mss_revisited.yml")
    mss_rev_loss = util.load_class_from_yaml(config_path)
    config_path = os.path.join(CONFIGS_DIR, "losses/rand_mss.yml")
    rand_mss_loss = util.load_class_from_yaml(config_path)
    config_path = os.path.join(CONFIGS_DIR, "losses/clap.yml")
    clap_loss = util.load_class_from_yaml(config_path)
    config_path = os.path.join(CONFIGS_DIR, "losses/panns_wglm.yml")
    panns_wglm_loss = util.load_class_from_yaml(config_path)
    config_path = os.path.join(CONFIGS_DIR, "losses/jtfs.yml")
    jtfs_loss = util.load_class_from_yaml(config_path)
    config_path = os.path.join(CONFIGS_DIR, "losses/scrapl.yml")
    scrapl_loss = util.load_class_from_yaml(config_path)
    n_paths = scrapl_loss.n_paths

    x = tr.rand((batch_size, 1, n_samples))
    x_hat = tr.rand((batch_size, 1, n_samples))
    x_hat.requires_grad_(True)
    x = x.to(device)
    x_hat = x_hat.to(device)

    df_cols = ["name", "median_time", "iqr", "n_runs", "max_mem_MB"]
    df_rows = []
    globals = {
        "x": x,
        "x_hat": x_hat,
        "fn": benchmark_loss_func,
        "path_idx": None,
    }
    loss_funcs = [
        ("mse", nn.MSELoss()),
        ("jtfs", jtfs_loss),
        ("mss_lin", mss_lin_loss),
        ("mss_log", mss_log_loss),
        ("mss_rev", mss_rev_loss),
        ("mss_rand", rand_mss_loss),
        ("clap", clap_loss),
        ("panns", panns_wglm_loss),
    ]

    for name, loss_func in loss_funcs:
        log.info(f"Benchmarking {name}...")
        loss_func = loss_func.to(device)
        globals["loss_func"] = loss_func
        row = calc_benchmark_stats(
            globals=globals,
            device=device,
            num_threads=num_threads,
            min_run_time=min_run_time,
        )
        df_rows.append((name,) + row)

    scrapl_rows = []
    scrapl_loss = scrapl_loss.to(device)
    globals["loss_func"] = scrapl_loss
    for path_idx in tqdm(range(n_paths)):
        log.info(f"Benchmarking scrapl path {path_idx} / {n_paths}...")
        globals["path_idx"] = path_idx
        row = calc_benchmark_stats(
            globals=globals,
            device=device,
            num_threads=num_threads,
            min_run_time=min_run_time,
        )
        scrapl_rows.append((f"scrapl_path_{path_idx}",) + row)

    scrapl_df = pd.DataFrame(scrapl_rows, columns=df_cols)
    print(scrapl_df.to_string(index=False))
    save_path = os.path.join(OUT_DIR, "scrapl_benchmark.csv")
    scrapl_df.to_csv(save_path, index=False, sep="\t")

    df_rows.append(
        (
            "scrapl",
            scrapl_df["median_time"].median(),
            scrapl_df["iqr"].median(),
            scrapl_df["n_runs"].median(),
            scrapl_df["max_mem_MB"].max(),
        )
    )
    df = pd.DataFrame(df_rows, columns=df_cols)
    print(df.to_string(index=False))
    save_path = os.path.join(OUT_DIR, "benchmark.tsv")
    df.to_csv(save_path, index=False, sep="\t")


if __name__ == "__main__":
    main()

    # TODO: cleanup
    # Benchmark model step
    model_config = os.path.join(CONFIGS_DIR, "models/spectral_2dcnn.yml")
    model = util.load_class_from_yaml(model_config)
    for p in model.parameters():
        p.requires_grad_(True)
        log.info(f"p.requires_grad: {p.requires_grad}")
    model = model.to(device)
    synth_config = os.path.join(CONFIGS_DIR, "synths/chirp_texture_8khz.yml")
    synth = util.load_class_from_yaml(synth_config)
    synth = synth.to(device)

    cqt_params = {
            "sr": synth.sr,
            "bins_per_octave": synth.Q,
            "n_bins": synth.J_cqt * synth.Q,
            "hop_length": synth.hop_len,
            # TODO: check this
            "fmin": (0.4 * synth.sr) / (2**synth.J_cqt),
            "output_format": "Magnitude",
            "verbose": False,
    }
    cqt = CQT(**cqt_params)
    cqt = cqt.to(device)

    x = tr.rand((batch_size, 1, n_samples))
    x = x.to(device)
    seed = tr.randint(0, 999999, (batch_size,))
    loss_func = nn.MSELoss()

    def model_step(x: T, seed: T, loss_func: Callable[..., T]) -> None:
        with tr.no_grad():
            U = cqt(x)
            U = tr.log1p(U / 1e-3)
        theta_hat_d, theta_hat_s = model(U)
        x_hat = synth(theta_hat_d, theta_hat_s, seed)
        loss = loss_func(x_hat, x)
        loss.backward()

    globals = {
        "x": x,
        "seed": seed,
        "loss_func": loss_func,
        "fn": model_step,
    }

    if device.startswith("cuda"):
        tr.cuda.reset_peak_memory_stats(device)
    result: Measurement = benchmark.Timer(
        stmt="fn(x, seed, loss_func)",
        globals=globals,
        num_threads=num_threads,
    ).blocked_autorange(min_run_time=min_run_time)
    max_mem = 0.0
    if device.startswith("cuda"):
        max_mem = tr.cuda.max_memory_allocated(device) / (1024**2)
    n_runs = len(result.times)

    print(
        f"model_step: median_time={result.median}, iqr={result.iqr}, "
        f"n_runs={n_runs}, max_mem_MB={max_mem}"
    )
