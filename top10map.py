"""
For each SAE feature, find the top-N highest-activation samples per shard file.

This is the "map" phase — each file is processed independently. The "reduce"
phase (top10reduce.py) merges across shards.

Uses shared config.py and cleaned up worker logic.

Usage:
    modal run top10map.py
"""
import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures

from modal import App, Image, Volume

from config import (
    get_dataset, get_model, get_sae,
    features_dataset_name, shard_files,
)

ds = get_dataset()
model = get_model()
sae = get_sae()

NUM_CPU = 4
N = 10  # top-N samples to keep per feature per shard

DATASET_DIR = "/embeddings"
VOLUME = "embeddings"

DIRECTORY = f"{DATASET_DIR}/{features_dataset_name()}"
SAVE_DIRECTORY = f"{DIRECTORY}-top{N}"

files = shard_files(ext="parquet")

volume = Volume.from_name(VOLUME, create_if_missing=True)
image = Image.debian_slim(python_version="3.10").pip_install(
    "pandas", "pyarrow", "numpy", "tqdm"
)
app = App(image=image)


def get_top_n_rows_by_activation(file, top_indices, top_acts, feature, n):
    """Find the top-n rows with highest activation for a given SAE feature."""
    rows_with_feature = np.any(top_indices == feature, axis=1)

    if not np.any(rows_with_feature):
        return pd.DataFrame(columns=["shard", "index", "feature", "activation"])

    filtered_indices = top_indices[rows_with_feature]
    filtered_acts = top_acts[rows_with_feature]

    positions = np.argwhere(filtered_indices == feature)
    row_indices = positions[:, 0]
    col_indices = positions[:, 1]
    act_values = filtered_acts[row_indices, col_indices]

    original_indices = np.where(rows_with_feature)[0][row_indices]

    if len(act_values) > n:
        top_n_pos = np.argpartition(act_values, -n)[-n:]
        top_n_pos = top_n_pos[np.argsort(act_values[top_n_pos])[::-1]]
    else:
        top_n_pos = np.argsort(act_values)[::-1]

    return pd.DataFrame({
        "shard": file,
        "index": original_indices[top_n_pos],
        "feature": feature,
        "activation": act_values[top_n_pos],
    })


def process_feature_chunk(file, feature_ids, chunk_index):
    """Worker: load the shard and find top-N for a subset of features."""
    start = time.perf_counter()
    df = pd.read_parquet(f"{DIRECTORY}/train/{file}", columns=["top_indices", "top_acts"])
    print(f"Worker {chunk_index}: loaded {file} in {time.perf_counter() - start:.1f}s")

    top_indices = np.array(df["top_indices"].tolist())
    top_acts = np.array(df["top_acts"].tolist())
    del df

    results = []
    for feature in tqdm(feature_ids, desc=f"Worker {chunk_index}", position=chunk_index):
        top = get_top_n_rows_by_activation(file, top_indices, top_acts, feature, N)
        if len(top) > 0:
            results.append(top)

    if not results:
        return None

    combined_df = pd.concat(results, ignore_index=True)

    os.makedirs(SAVE_DIRECTORY, exist_ok=True)
    temp_file = f"{SAVE_DIRECTORY}/temp_{file}_{chunk_index}.parquet"
    combined_df.to_parquet(temp_file)

    del top_indices, top_acts, results, combined_df
    return temp_file


@app.function(cpu=NUM_CPU, volumes={DATASET_DIR: volume}, timeout=6000)
def process_dataset(file):
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)

    num_features = model.dim * sae.expansion
    features_per_worker = num_features // NUM_CPU
    feature_batches = [
        list(range(i, min(i + features_per_worker, num_features)))
        for i in range(0, num_features, features_per_worker)
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CPU) as executor:
        futures = [
            executor.submit(process_feature_chunk, file, batch, i)
            for i, batch in enumerate(feature_batches)
        ]
        temp_files = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                temp_files.append(result)

    # Merge worker outputs
    dfs = []
    for temp_file in temp_files:
        dfs.append(pd.read_parquet(temp_file))
        os.remove(temp_file)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_parquet(f"{SAVE_DIRECTORY}/{file}")
    volume.commit()

    return f"Done: {file} — {len(combined_df)} top-{N} samples"


@app.local_entrypoint()
def main():
    for resp in process_dataset.map(files, order_outputs=False, return_exceptions=True):
        if isinstance(resp, Exception):
            print(f"EXCEPTION: {resp}")
        else:
            print(resp)
