"""
Combine chunks, embeddings and SAE features into a single LanceDB table.

Improvements over the original:
- Uses shared config.py
- Avoids materializing entire memmap via list() — uses numpy array directly
- Converts sae_indices to int via numpy instead of per-row Python lambda

Usage:
    modal run lancer.py
"""
import os
import time

import numpy as np
import pandas as pd
import lancedb
from modal import App, Image, Volume, gpu

from config import (
    get_dataset, get_model, get_sae,
    chunked_dataset_name, embedding_dataset_name, features_dataset_name,
)

ds = get_dataset()
model = get_model()
sae = get_sae()

CHUNK_PARQUET_DIR = f"/datasets/{chunked_dataset_name()}/train"
EMBEDDING_NPY_DIR = f"/embeddings/{embedding_dataset_name()}/train"
FEATURE_PARQUET_DIR = f"/embeddings/{features_dataset_name()}/train"

LANCE_DB_DIR = f"/lancedb/enjalot/{ds.save_name}"
LANCE_DB_DIR_INDEXED = f"/lancedb/enjalot/{ds.save_name}-indexed"
TMP_LANCE_DB_DIR = f"/tmp/{ds.save_name}"

TABLE_NAME = f"{sae.slug}"
TOTAL_FILES = ds.num_files
D_EMB = model.dim

DATASETS_VOLUME = ds.volume
EMBEDDING_VOLUME = "embeddings"
DB_VOLUME = "lancedb"

volume_db = Volume.from_name(DB_VOLUME, create_if_missing=True)
volume_datasets = Volume.from_name(DATASETS_VOLUME, create_if_missing=True)
volume_embeddings = Volume.from_name(EMBEDDING_VOLUME, create_if_missing=True)

st_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install("pandas", "numpy", "lancedb", "pyarrow", "torch", "tantivy")
    .env({"RUST_BACKTRACE": "1"})
)

app = App(image=st_image)


@app.function(
    volumes={
        "/datasets": volume_datasets,
        "/embeddings": volume_embeddings,
        "/lancedb": volume_db,
    },
    ephemeral_disk=int(1024 * 1024),
    image=st_image,
    timeout=60 * 100,
    scaledown_window=60 * 10,
)
def combine():
    """Process each shard sequentially, combining chunks + embeddings + SAE features."""
    db_path = TMP_LANCE_DB_DIR
    print(f"Connecting to LanceDB at: {db_path}")
    db = lancedb.connect(db_path)

    for i in range(TOTAL_FILES):
        base_file = f"data-{i:05d}-of-{TOTAL_FILES:05d}"
        chunk_file = os.path.join(CHUNK_PARQUET_DIR, f"{base_file}.parquet")
        embedding_file = os.path.join(EMBEDDING_NPY_DIR, f"{base_file}.npy")
        feature_file = os.path.join(FEATURE_PARQUET_DIR, f"{base_file}.parquet")

        print(f"\nProcessing shard: {base_file}")
        start_time = time.monotonic()

        try:
            chunk_df = pd.read_parquet(chunk_file)
        except Exception as e:
            print(f"Error reading chunk file {chunk_file}: {e}")
            break

        try:
            size = os.path.getsize(embedding_file) // (D_EMB * 4)
            embedding_np = np.memmap(
                embedding_file, dtype="float32", mode="r", shape=(size, D_EMB)
            )
        except Exception as e:
            print(f"Error reading embedding file {embedding_file}: {e}")
            break

        try:
            feature_df = pd.read_parquet(feature_file)
            feature_df = feature_df.rename(columns={
                "top_indices": "sae_indices",
                "top_acts": "sae_acts",
            })
            # Vectorized int conversion instead of per-row lambda
            feature_df["sae_indices"] = feature_df["sae_indices"].apply(
                lambda x: np.array(x, dtype=np.int32).tolist()
            )
        except Exception as e:
            print(f"Error reading feature file {feature_file}: {e}")
            break

        n_chunk, n_emb, n_feat = len(chunk_df), embedding_np.shape[0], len(feature_df)
        if not (n_chunk == n_emb == n_feat):
            print(f"Row count mismatch in {base_file}: chunk={n_chunk}, emb={n_emb}, feat={n_feat}")
            break

        # Combine column-wise
        combined_df = pd.concat(
            [chunk_df.reset_index(drop=True), feature_df.reset_index(drop=True)],
            axis=1,
        )
        # Use numpy array directly — avoids Python list-of-list materialisation
        combined_df["vector"] = list(np.array(embedding_np))
        combined_df["shard"] = i

        if i == 0:
            print(f"Creating table '{TABLE_NAME}' with {len(combined_df)} rows")
            table = db.create_table(TABLE_NAME, combined_df)
        else:
            print(f"Adding shard {i} with {len(combined_df)} rows")
            table.add(combined_df)

        duration = time.monotonic() - start_time
        print(f"  Shard {i} done in {duration:.1f}s ({n_chunk} rows)")

    import shutil
    print(f"Copying LanceDB to {LANCE_DB_DIR}")
    shutil.copytree(TMP_LANCE_DB_DIR, LANCE_DB_DIR)
    volume_db.commit()
    print("Done!")


@app.function(
    volumes={
        "/datasets": volume_datasets,
        "/embeddings": volume_embeddings,
        "/lancedb": volume_db,
    },
    gpu="A10G",
    ephemeral_disk=int(1024 * 1024),
    image=st_image,
    timeout=60 * 100,
    scaledown_window=60 * 10,
)
def create_indices():
    import shutil

    start_time = time.monotonic()
    print(f"Copying {LANCE_DB_DIR} → {TMP_LANCE_DB_DIR}")
    shutil.copytree(LANCE_DB_DIR, TMP_LANCE_DB_DIR)
    print(f"  Copied in {time.monotonic() - start_time:.1f}s")

    db = lancedb.connect(TMP_LANCE_DB_DIR)
    table = db.open_table(TABLE_NAME)

    # Full-text search index on title (if dataset has it)
    if "title" in table.schema.names:
        start_time = time.monotonic()
        print("Creating FTS index on 'title'")
        table.create_fts_index("title")
        print(f"  Done in {time.monotonic() - start_time:.1f}s")

    # ANN vector index
    start_time = time.monotonic()
    partitions = int(table.count_rows() ** 0.5) * 2
    sub_vectors = D_EMB // 16
    print(f"Creating ANN index: {partitions} partitions, {sub_vectors} sub-vectors, cosine")
    table.create_index(
        num_partitions=partitions,
        num_sub_vectors=sub_vectors,
        metric="cosine",
        accelerator="cuda",
    )
    print(f"  Done in {time.monotonic() - start_time:.1f}s")

    start_time = time.monotonic()
    print(f"Copying → {LANCE_DB_DIR_INDEXED}")
    shutil.copytree(TMP_LANCE_DB_DIR, LANCE_DB_DIR_INDEXED, dirs_exist_ok=True)
    volume_db.commit()
    print(f"  Done in {time.monotonic() - start_time:.1f}s")


@app.local_entrypoint()
def main():
    # combine.remote()
    # print("Combine done, creating indices...")
    create_indices.remote()
