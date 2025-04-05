"""
For each of the parquet files with activations, find the top 10 and write to an intermediate file
modal run top10map.py
"""
from modal import App, Image, Volume
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures
from functools import partial

NUM_CPU=4

N=5 # the number of samples to keep per feature

DATASET_DIR="/embeddings"
VOLUME = "embeddings"

D_IN = 768 # the dimensions from the embedding models
K=64
# EXPANSION = 128
EXPANSION = 32
SAE = f"{K}_{EXPANSION}"
# DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500-HF4-{SAE}-3" 
# SAVE_DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500-HF4-{SAE}-3-top10"
DIRECTORY = f"{DATASET_DIR}/wikipedia-en-chunked-500-nomic-embed-text-v1.5-{SAE}" 
SAVE_DIRECTORY = f"{DATASET_DIR}/wikipedia-en-chunked-500-nomic-embed-text-v1.5-{SAE}-top{N}"


files = [f"data-{i:05d}-of-00041.parquet" for i in range(41)]

# We define our Modal Resources that we'll need
volume = Volume.from_name(VOLUME, create_if_missing=True)
image = Image.debian_slim(python_version="3.9").pip_install(
    "datasets==2.16.1", "apache_beam==2.53.0", "transformers", "pandas", "tqdm"
)
app = App(image=image)  # Note: prior to April 2024, "app" was called "stub"

# def get_top_n_rows_by_top_act(file, top_indices, top_acts, feature):
#     # feature_positions = np.where(np.any(top_indices == feature, axis=1),
#     #                        np.argmax(top_indices == feature, axis=1),
#     #                        -1)
#     # act_values = np.where(feature_positions != -1, 
#     #                   top_acts[np.arange(len(top_acts)), feature_positions], 
#     #                   0)
#     # top_n_indices = np.argsort(act_values)[-N:][::-1]

#     # Find positions where feature appears (returns a boolean mask)
#     feature_mask = top_indices == feature
    
#     # Get the activation values where the feature appears (all others will be 0)
#     act_values = np.where(feature_mask.any(axis=1),
#                          top_acts[feature_mask].reshape(-1),
#                          0)
    
#     # Use partition to get top N indices efficiently
#     top_n_indices = np.argpartition(act_values, -N)[-N:]
#     # Sort just the top N indices
#     top_n_indices = top_n_indices[np.argsort(act_values[top_n_indices])[::-1]]

#     filtered_df = pd.DataFrame({
#         "shard": file,
#         "index": top_n_indices,
#         "feature": feature,
#         "activation": act_values[top_n_indices]
#     })
#     return filtered_df

def get_top_n_rows_by_top_act(file, top_indices, top_acts, feature):
    # Use memory-efficient approach to find rows with this feature
    rows_with_feature = np.any(top_indices == feature, axis=1)
    
    # Only process rows that have this feature
    filtered_indices = top_indices[rows_with_feature]
    filtered_acts = top_acts[rows_with_feature]
    
    # Get positions of the feature in each row
    positions = np.argwhere(filtered_indices == feature)
    
    # Create array of activation values (sparse approach)
    row_indices = positions[:, 0]
    col_indices = positions[:, 1]
    act_values = filtered_acts[row_indices, col_indices]
    
    # Map back to original indices
    original_indices = np.where(rows_with_feature)[0][row_indices]
    
    # Get top N
    if len(act_values) > N:
        top_n_pos = np.argpartition(act_values, -N)[-N:]
        top_n_pos = top_n_pos[np.argsort(act_values[top_n_pos])[::-1]]
    else:
        # If we have fewer than N matches, take all of them
        top_n_pos = np.argsort(act_values)[::-1]
    
    filtered_df = pd.DataFrame({
        "shard": file,
        "index": original_indices[top_n_pos],
        "feature": feature,
        "activation": act_values[top_n_pos]
    })
    return filtered_df


def process_feature_chunk(file, feature_ids, chunk_index):
    start = time.perf_counter()
    print(f"Loading dataset from {DIRECTORY}/train/{file}", chunk_index)
    
    # Only read the columns we need
    df = pd.read_parquet(f"{DIRECTORY}/train/{file}", columns=['top_indices', 'top_acts'])
    print(f"Dataset loaded in {time.perf_counter()-start:.2f} seconds for {file}", chunk_index) 

    top_indices = np.array(df['top_indices'].tolist())
    top_acts = np.array(df['top_acts'].tolist())
    
    # Free up memory by deleting the DataFrame after conversion to numpy
    del df
    
    print(f"top_indices shape: {top_indices.shape}")
    print(f"top_acts shape: {top_acts.shape}")
    print("got numpy arrays", chunk_index)
    
    results = []
    
    # Process each feature in this worker's batch
    for feature in tqdm(feature_ids, desc=f"Processing features (worker {chunk_index})", position=chunk_index):
        # Get the true top N rows for this feature across the entire chunk
        top = get_top_n_rows_by_top_act(file, top_indices, top_acts, feature)
        results.append(top)
    
    # Combine results for all features in this worker
    combined_df = pd.concat(results, ignore_index=True)
    
    # Write to a temporary file to save memory
    temp_file = f"{SAVE_DIRECTORY}/temp_{file}_{chunk_index}.parquet"
    combined_df.to_parquet(temp_file)
    
    # Free memory
    del top_indices, top_acts, results, combined_df
    
    return temp_file

@app.function(cpu=NUM_CPU, volumes={DATASET_DIR: volume}, timeout=6000)
def process_dataset(file):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Ensure directory exists
    if not os.path.exists(f"{SAVE_DIRECTORY}"):
        os.makedirs(f"{SAVE_DIRECTORY}")

    num_features = D_IN * EXPANSION

    # Split the features among workers - each worker handles a subset of features
    # but processes the ENTIRE dataset for those features
    features_per_worker = num_features // NUM_CPU
    feature_batches = [list(range(i, min(i + features_per_worker, num_features))) 
                      for i in range(0, num_features, features_per_worker)]

    with ProcessPoolExecutor(max_workers=NUM_CPU) as executor:
        futures = [executor.submit(process_feature_chunk, file, feature_batch, i) 
                  for i, feature_batch in enumerate(feature_batches)]
        
        temp_files = []
        for future in as_completed(futures):
            temp_file = future.result()
            temp_files.append(temp_file)
    
    # Combine temporary files
    print("Combining temporary files")
    dfs = []
    for temp_file in temp_files:
        dfs.append(pd.read_parquet(temp_file))
        # Remove temp file after reading
        os.remove(temp_file)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_parquet(f"{SAVE_DIRECTORY}/{file}")
    volume.commit()
    
    return f"All done with {file}", len(combined_df)


@app.local_entrypoint()
def main():
    for resp in process_dataset.map(files, order_outputs=False, return_exceptions=True):
        if isinstance(resp, Exception):
            print(f"Exception: {resp}")
            continue
        print(resp)


