import logging
import os
import pandas as pd
from experiments.paths import OUT_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def remove_training(tsv_path: str) -> None:
    df = pd.read_csv(tsv_path, sep="\t", index_col=False)
    log.info(f"Before filtering, n_rows={len(df)}")
    df = df[df["stage"] != "train"]
    log.info(f"After filtering, n_rows={len(df)}")
    df = df[~((df["step"] == 0) & (df["stage"] == "val"))]
    log.info(f"After removing step 0 val, n_rows={len(df)}")
    df.to_csv(tsv_path, sep="\t", index=False)


if __name__ == "__main__":
    root_dir = os.path.join(OUT_DIR, "iclr_2026_done/roland_tr_808")

    # Find all .tsv files in root_dir
    tsv_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".tsv"):
                tsv_paths.append(os.path.join(dirpath, filename))

    for tsv_path in tsv_paths:
        log.info(f"Processing {tsv_path}")
        remove_training(tsv_path)

    # # Read all lines in each file
    # for tsv_path in tsv_paths:
    #     log.info(f"Processing {tsv_path}")
    #     with open(tsv_path, "r") as f:
    #         lines = f.readlines()
    #
    #     log.info(f"Found {len(lines)} lines")
    #     if (len(lines) - 1) % 3 != 0:
    #         log.info(f"File {tsv_path} has {(len(lines) - 1)} lines, which is not a multiple of 3 + 1")
    #         continue
    #
    #     new_lines = [lines[0]]
    #     lines = lines[1:]  # Skip header
    #
    #     if len(lines[0].strip().split("\t")) != 9:
    #         log.info(f"File {tsv_path} does not have 9 columns in the first data line")
    #         continue
    #
    #     for idx in range(0, len(lines), 3):
    #         start = lines[idx].strip()
    #         assert len(start.split("\t")) == 9
    #         mid = lines[idx + 1].strip()
    #         assert len(mid.split("\t")) == 3
    #         end = lines[idx + 2].strip()
    #         assert len(end.split("\t")) == 3
    #         new_lines.append(f"{start}\t{mid}\t{end}\n")
    #
    #     # Overwrite
    #     with open(tsv_path, "w") as f:
    #         f.writelines(new_lines)
