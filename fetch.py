"""
Fetch a single file from a Modal volume for local inspection.

Usage:
    modal run fetch.py
"""
from modal import App, Image, Volume

from config import get_dataset, embedding_dataset_name

ds = get_dataset()

# Point at whatever volume/directory you want to fetch from
VOLUME = "embeddings"
DATASET_DIR = "/embeddings"
DIRECTORY = f"{DATASET_DIR}/{embedding_dataset_name()}/train"

volume = Volume.from_name(VOLUME, create_if_missing=True)
image = Image.debian_slim(python_version="3.10").pip_install("pandas", "pyarrow")
app = App(image=image)


@app.function(volumes={DATASET_DIR: volume}, timeout=3000)
def fetch_file(file_path):
    import pandas as pd

    print(f"Loading {file_path}")
    if file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)
    else:
        from datasets import load_dataset
        dataset = load_dataset("arrow", data_files=file_path)
        return pd.DataFrame(dataset["train"])


@app.local_entrypoint()
def main():
    file = "data-00000-of-00041.parquet"
    file_path = f"{DIRECTORY}/{file}"
    resp = fetch_file.remote(file_path)
    if isinstance(resp, Exception):
        print(f"EXCEPTION: {resp}")
    else:
        print(resp)
        resp.to_parquet(f"./notebooks/{file}")
