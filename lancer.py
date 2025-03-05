"""
Combine chunks, embeddings and features into a single LanceDB table.

This script loops over each corresponding file:
  - The chunk parquet produced by chunker.py (e.g. "/data/medrag-pubmed-500/train/data-00000-of-00138.parquet")
  - The embedding npy file produced by features.py (e.g. "/embeddings/medrag-pubmed-500-nomic-embed-text-v1.5/train/data-00000-of-00138.npy")
  - The features parquet file produced by features.py (e.g. "/embeddings/medrag-pubmed-500-nomic-embed-text-v1.5-64_32/train/data-00000-of-00138.parquet")
  
They are then concatenated (column-wise) row‐by‐row in the natural order and written to a lancedb table.
  
Usage (from Modal CLI):
    modal run combine.py
"""

import os
import time
import numpy as np
import pandas as pd
import lancedb
from modal import App, Image, Volume, enter, method, gpu

# ============================================================================
# Configuration variables – adjust these to your environment/path names!
# ============================================================================

# Directories for the input files:
CHUNK_PARQUET_DIR = "/datasets/medrag-pubmed-500/train"  
EMBEDDING_NPY_DIR = "/embeddings/medrag-pubmed-500-nomic-embed-text-v1.5/train"
FEATURE_PARQUET_DIR = "/embeddings/medrag-pubmed-500-nomic-embed-text-v1.5-64_32/train"

# Directory (volume) where the LanceDB table will be stored.
LANCE_DB_DIR = "/lancedb/enjalot/medrag-pubmed"  
LANCE_DB_DIR_INDEXED = "/lancedb/enjalot/medrag-pubmed-indexed"  
TMP_LANCE_DB_DIR = "/tmp/medrag-pubmed"  
TABLE_NAME = "500-64_32"

TOTAL_FILES = 138    # total number of shards (files)
D_EMB = 768          # embedding dimension

# Volume for the lancedb storage
DATASETS_VOLUME = "datasets"
EMBEDDING_VOLUME = "embeddings"
DB_VOLUME = "lancedb"

# ============================================================================
# Modal Resources
# ============================================================================

volume_db = Volume.from_name(DB_VOLUME, create_if_missing=True)
volume_datasets = Volume.from_name(DATASETS_VOLUME, create_if_missing=True)
volume_embeddings = Volume.from_name(EMBEDDING_VOLUME, create_if_missing=True)

st_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "pandas", "numpy", "lancedb", "pyarrow", "torch", "tantivy"
    )
    .env({"RUST_BACKTRACE": "1"})
)


app = App(image=st_image)

# ============================================================================
# Class to combine and write data into a lancedb table
# ============================================================================

@app.function(volumes={
    "/datasets": volume_datasets,
    "/embeddings": volume_embeddings,
    "/lancedb": volume_db
    }, 
    ephemeral_disk=int(1024*1024), # in MiB
    image=st_image, 
    timeout=60*100,
    scaledown_window=60*10
    )
def combine():
    """
    Sequentially process each shard by reading the corresponding chunk parquet,
    embedding npy, and features parquet files. The data are combined (column-wise)
    and then appended to a single lancedb table.
    """
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

        # Load the chunk parquet file.
        try:
            chunk_df = pd.read_parquet(chunk_file)
        except Exception as e:
            print(f"Error reading chunk file {chunk_file}: {e}")
            break
        
        # Load the embeddings npy file.
        try:
            size = os.path.getsize(embedding_file) // (D_EMB * 4)
            embedding_np = np.memmap(embedding_file, 
                  dtype='float32', 
                  mode='r', 
                  shape=(size, D_EMB))
        except Exception as e:
            print(f"Error reading embedding file {embedding_file}: {e}")
            break 
        
        # Load the features parquet file.
        try:
            feature_df = pd.read_parquet(feature_file)
            feature_df = feature_df.rename(columns={
                'top_indices': 'sae_indices',
                'top_acts': 'sae_acts'
            })
            # Convert sae_indices from float to int for each row
            feature_df['sae_indices'] = feature_df['sae_indices'].apply(lambda x: [int(i) for i in x])
        except Exception as e:
            print(f"Error reading feature file {feature_file}: {e}")
            break 

        # Validate that the three sources have the same number of rows.
        n_chunk = len(chunk_df)
        n_embedding = embedding_np.shape[0]
        n_feature = len(feature_df)
        if not (n_chunk == n_embedding == n_feature):
            print(f"Row count mismatch in {base_file}: chunk {n_chunk}, embedding {n_embedding}, feature {n_feature}")
            break

        # Store the embedding data as a list column. (Alternatively, you could split the embedding vector into columns.)

        vector_column = list(embedding_np)

        # Combine the dataframes (reseting indices to ensure correct alignment).
        combined_df = pd.concat(
            [chunk_df.reset_index(drop=True),
              feature_df.reset_index(drop=True)],
            axis=1,
        )
        combined_df["vector"] = vector_column
        combined_df["shard"] = i
        
        if i == 0:
            msg = f"Creating LanceDB table '{TABLE_NAME}' at {db_path} with {len(combined_df)} rows."
            print(msg)
            table = db.create_table(TABLE_NAME, combined_df)
        else:
            msg = f"Adding shard {base_file} to LanceDB table '{TABLE_NAME}' at {db_path} with {len(combined_df)} rows."
            print(msg)
            table.add(combined_df)
        # if i == 2:
        #     break

        duration = time.monotonic() - start_time
        print(f"Shard {base_file} processed in {duration:.2f} seconds; {n_chunk} rows")


    print(f"Copying LanceDB to {LANCE_DB_DIR}")
    # copy the tmp lancedb directory to the volume
    import shutil
    shutil.copytree(TMP_LANCE_DB_DIR, LANCE_DB_DIR)
    print(f"Done!")


@app.function(volumes={
    "/datasets": volume_datasets,
    "/embeddings": volume_embeddings,
    "/lancedb": volume_db
    }, 
    gpu="A10G",
    ephemeral_disk=int(1024*1024), # in MiB
    image=st_image, 
    timeout=60*100,
    scaledown_window=60*10
    )
def create_indices():
    import lancedb
    import shutil
    start_time = time.monotonic()
    print(f"Copying table {LANCE_DB_DIR} to {TMP_LANCE_DB_DIR}")
    shutil.copytree(LANCE_DB_DIR, TMP_LANCE_DB_DIR)
    duration = time.monotonic() - start_time
    print(f"Copying table {LANCE_DB_DIR} to {TMP_LANCE_DB_DIR} took {duration:.2f} seconds")

    db = lancedb.connect(TMP_LANCE_DB_DIR)
    table = db.open_table(TABLE_NAME)

    # start_time = time.monotonic()
    # print(f"Creating index for sae_indices on table '{TABLE_NAME}'")
    # table.create_scalar_index("sae_indices", index_type="LABEL_LIST")
    # duration = time.monotonic() - start_time
    # print(f"Creating index for sae_indices on table '{TABLE_NAME}' took {duration:.2f} seconds")

    start_time = time.monotonic()
    print(f"Creating FTS index for title on table '{TABLE_NAME}'")
    table.create_fts_index("title")
    duration = time.monotonic() - start_time
    print(f"Creating FTS index for title on table '{TABLE_NAME}' took {duration:.2f} seconds")

    start_time = time.monotonic()
    print(f"Creating ANN index for embeddings on table '{TABLE_NAME}'")
    partitions = int(table.count_rows() ** 0.5) * 2
    sub_vectors = D_EMB // 16
    metric = "cosine"
    print(f"Partitioning into {partitions} partitions, {sub_vectors} sub-vectors")
    table.create_index(
        num_partitions=partitions, 
        num_sub_vectors=sub_vectors, 
        metric=metric,
        accelerator="cuda"
    )
    duration = time.monotonic() - start_time
    print(f"Creating ANN index for embeddings on table '{TABLE_NAME}' took {duration:.2f} seconds")

    # print(f"Deleting existing {LANCE_DB_DIR}")
    # shutil.rmtree(LANCE_DB_DIR, ignore_errors=True)
    start_time = time.monotonic() 
    print(f"Copying table {TABLE_NAME} to {LANCE_DB_DIR_INDEXED}")
    shutil.copytree(TMP_LANCE_DB_DIR, LANCE_DB_DIR_INDEXED, dirs_exist_ok=True)
    duration = time.monotonic() - start_time
    print(f"Copying table {TMP_LANCE_DB_DIR} to {LANCE_DB_DIR_INDEXED} took {duration:.2f} seconds")

# ============================================================================
# Modal Local Entrypoint
# ============================================================================

@app.local_entrypoint()
def main():
    # Combine all shards and write to LanceDB.
    # combine.remote()
    # print("done with combine, creating indices")
    create_indices.remote()