import logging
import os

import pandas as pd

from experiments.paths import DATA_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "WARNING"))


if __name__ == "__main__":
    tsv_path = os.path.join(
        DATA_DIR, "benchmarks/benchmark__bs4_ns32768__min_run_time_15.tsv"
    )
    df = pd.read_csv(tsv_path, sep="\t")
    formatters = {
        "median_time": lambda x: f"{x * 1000:.4g}",  # 3 significant figures
        "iqr": lambda x: f"{x * 1000:.4g}",  # 3 significant figures
        "max_mem_MB": lambda x: f"{x:.0f}",  # 3 significant figures
    }
    # Apply formatting when displaying
    print(df.to_string(formatters=formatters, index=False))
