"""
Quick summary statistics for a chunked dataset.

Usage:
    modal run summary.py
"""
from modal import App, Image, Volume

from config import get_dataset, chunked_dataset_name, shard_files

ds = get_dataset()

DATASET_DIR = "/data"
DATASET_SAVE_CHUNKED = chunked_dataset_name()
files = shard_files(ext="parquet")

volume = Volume.from_name(ds.volume, create_if_missing=True)
image = Image.debian_slim(python_version="3.10").pip_install("pandas", "pyarrow")
app = App(image=image)


@app.function(volumes={DATASET_DIR: volume}, timeout=3000)
def process_dataset(file):
    import pandas as pd

    df = pd.read_parquet(f"{DATASET_DIR}/{DATASET_SAVE_CHUNKED}/train/{file}")
    return {
        "file": file,
        "num_rows": len(df),
        "tokens": int(df["chunk_token_count"].sum()),
        "less2": int((df["chunk_token_count"] < 2).sum()),
        "less10": int((df["chunk_token_count"] < 10).sum()),
        "less50": int((df["chunk_token_count"] < 50).sum()),
    }


@app.local_entrypoint()
def main():
    totals = {"rows": 0, "tokens": 0, "less2": 0, "less10": 0, "less50": 0}
    for resp in process_dataset.map(files, order_outputs=False, return_exceptions=True):
        if isinstance(resp, Exception):
            print(f"EXCEPTION: {resp}")
            continue
        print(resp)
        totals["rows"] += resp["num_rows"]
        totals["tokens"] += resp["tokens"]
        totals["less2"] += resp["less2"]
        totals["less10"] += resp["less10"]
        totals["less50"] += resp["less50"]

    print(f"\nTotal rows: {totals['rows']:,}")
    print(f"Total tokens: {totals['tokens']:,}")
    print(f"Chunks < 2 tokens: {totals['less2']:,}")
    print(f"Chunks < 10 tokens: {totals['less10']:,}")
    print(f"Chunks < 50 tokens: {totals['less50']:,}")
