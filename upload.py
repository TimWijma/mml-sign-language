import gc
import pickle
import logging
import argparse
from pathlib import Path
from datasets import Dataset

DST_DATASET = "TimWijma/how2sign-3s-mosaic"
CHUNK_SIZE  = 100
PARQUET_DIR = Path("./parquet_out")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl",   required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()

    PARQUET_DIR.mkdir(exist_ok=True)

    log.info("Loading pickle …")
    with open(args.pkl, "rb") as f:
        rows = pickle.load(f)
    total = len(rows)
    log.info("Loaded %d rows, writing parquet chunks …", total)

    parquet_files = []
    for start in range(0, total, CHUNK_SIZE):
        chunk = rows[start : start + CHUNK_SIZE]
        ds = Dataset.from_list(chunk)
        del chunk
        gc.collect()
        out_path = str(PARQUET_DIR / f"{args.split}_{start:06d}.parquet")
        ds.to_parquet(out_path)
        del ds
        gc.collect()
        parquet_files.append(out_path)
        log.info("Wrote %s", out_path)

    del rows
    gc.collect()

    log.info("All parquet files written. Uploading …")
    from datasets import load_dataset
    ds_full = load_dataset("parquet", data_files={args.split: [str(PARQUET_DIR / f"{args.split}_*.parquet")]})
    ds_full[args.split].push_to_hub(DST_DATASET, split=args.split, token=args.token, max_shard_size="500MB")
    log.info("Done.")