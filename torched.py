"""
Convert embedding parquet files to .pt (PyTorch tensor) files for faster loading.

This is a one-off utility — only needed if downstream code expects .pt files
instead of the default .npy memmap format.

Usage:
    modal run torched.py
"""
from modal import App, Image, Volume

from config import get_dataset, embedding_dataset_name, shard_files

ds = get_dataset()

DATASET_DIR = "/embeddings"
VOLUME = "embeddings"
DIRECTORY = f"{DATASET_DIR}/{embedding_dataset_name()}"
SAVE_DIRECTORY = f"{DIRECTORY}-torched"

files = shard_files(ext="parquet")

volume = Volume.from_name(VOLUME, create_if_missing=True)
image = Image.debian_slim(python_version="3.10").pip_install(
    "pandas", "pyarrow", "torch", "numpy"
)
app = App(image=image)


@app.function(volumes={DATASET_DIR: volume}, timeout=60000)
def torch_dataset_shard(file):
    import os
    import pandas as pd
    import numpy as np
    import torch

    print(f"Loading {file}")
    df = pd.read_parquet(f"{DIRECTORY}/train/{file}")
    embeddings = np.array([np.array(e, dtype=np.float32) for e in df["embedding"]])

    shard = file.split(".")[0]
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)
    torch.save(torch.tensor(embeddings), f"{SAVE_DIRECTORY}/{shard}.pt")

    volume.commit()
    return shard


@app.local_entrypoint()
def main():
    for resp in torch_dataset_shard.map(files, order_outputs=False, return_exceptions=True):
        if isinstance(resp, Exception):
            print(f"EXCEPTION: {resp}")
        else:
            print(resp)
