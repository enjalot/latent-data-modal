"""
Filter out small chunks (< min_tokens) from chunked dataset files.

With the fixed chunker, this shouldn't be needed for new runs, but useful
for cleaning up previously chunked data.

Usage:
    modal run filter.py
"""
from modal import App, Image, Volume

from config import get_dataset, chunked_dataset_name, shard_files

ds = get_dataset()

DATASET_DIR = "/data"
DIRECTORY = f"{DATASET_DIR}/{chunked_dataset_name()}/train"
MIN_TOKENS = 50
files = shard_files(ext="parquet")

volume = Volume.from_name(ds.volume, create_if_missing=True)
image = Image.debian_slim(python_version="3.10").pip_install("pandas", "pyarrow")
app = App(image=image)


@app.function(volumes={DATASET_DIR: volume}, timeout=60000)
def filter_file(file):
    import pandas as pd

    path = f"{DIRECTORY}/{file}"
    df = pd.read_parquet(path)
    before = len(df)
    filtered = df[df["chunk_token_count"] >= MIN_TOKENS]
    filtered.to_parquet(path)
    volume.commit()
    return f"{file}: {before} → {len(filtered)} rows"


@app.local_entrypoint()
def main():
    for resp in filter_file.map(files, order_outputs=False, return_exceptions=True):
        if isinstance(resp, Exception):
            print(f"EXCEPTION: {resp}")
        else:
            print(resp)
