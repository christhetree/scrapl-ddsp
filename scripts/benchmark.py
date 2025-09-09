import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from typing import Callable, Any, Dict, Optional, Tuple
import pandas as pd
from torch.utils.benchmark import Measurement
from tqdm import tqdm

from experiments import util
from experiments.paths import CONFIGS_DIR, OUT_DIR

from torch import Tensor as T
import torch as tr
from torch.utils import benchmark

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

batch_size = 4
n_samples = 32768
num_threads = 1
min_run_time = 5.0
device = "cpu"
# device = "cuda"


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

    mss_config_path = os.path.join(CONFIGS_DIR, "losses/mss.yml")
    mss_loss = util.load_class_from_yaml(mss_config_path)
    mss_rev_config_path = os.path.join(CONFIGS_DIR, "losses/mss_revisited.yml")
    mss_rev_loss = util.load_class_from_yaml(mss_rev_config_path)
    rand_mss_config_path = os.path.join(CONFIGS_DIR, "losses/rand_mss.yml")
    rand_mss_loss = util.load_class_from_yaml(rand_mss_config_path)
    clap_config_path = os.path.join(CONFIGS_DIR, "losses/clap.yml")
    clap_loss = util.load_class_from_yaml(clap_config_path)
    panns_wglm_config_path = os.path.join(CONFIGS_DIR, "losses/panns_wglm.yml")
    panns_wglm_loss = util.load_class_from_yaml(panns_wglm_config_path)
    jtfs_config_path = os.path.join(CONFIGS_DIR, "losses/jtfst.yml")
    jtfs_loss = util.load_class_from_yaml(jtfs_config_path)
    scrapl_config_path = os.path.join(CONFIGS_DIR, "losses/scrapl.yml")
    scrapl_loss = util.load_class_from_yaml(scrapl_config_path)
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
        ("jtfs", jtfs_loss),
        ("mss", mss_loss),
        ("mss_rev", mss_rev_loss),
        ("rand_mss", rand_mss_loss),
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
    save_path = os.path.join(OUT_DIR, "benchmark.csv")
    df.to_csv(save_path, index=False, sep="\t")


if __name__ == "__main__":
    main()
