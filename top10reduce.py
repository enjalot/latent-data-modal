from modal import App, Image, Volume, Secret

EMBEDDINGS_DIR="/embeddings"
EMBEDDINGS_VOLUME = "embeddings"
DATASETS_DIR="/datasets"
DATASETS_VOLUME = "datasets"

SAE = "64_32"

# SAMPLE_DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500-HF4-{SAE}-3/train"
# DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500-HF4-{SAE}-3-top10"
# SAVE_DIRECTORY = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500-HF4-{SAE}-3-top10/combined"

SAMPLE_DIRECTORY = f"{DATASETS_DIR}/wikipedia-en-chunked-500/train"
SAE_DIRECTORY = f"{EMBEDDINGS_DIR}/wikipedia-en-chunked-500-nomic-embed-text-v1.5-{SAE}/train"
DIRECTORY = f"{EMBEDDINGS_DIR}/wikipedia-en-chunked-500-nomic-embed-text-v1.5-{SAE}-top5"
SAVE_DIRECTORY = f"{EMBEDDINGS_DIR}/wikipedia-en-chunked-500-nomic-embed-text-v1.5-{SAE}-top5/combined"





# We define our Modal Resources that we'll need
embeddings_volume = Volume.from_name(EMBEDDINGS_VOLUME, create_if_missing=True)
datasets_volume = Volume.from_name(DATASETS_VOLUME, create_if_missing=True)
image = Image.debian_slim(python_version="3.9").pip_install(
    "pandas", "datasets==2.16.1", "apache_beam==2.53.0"
)
app = App(image=image) 

@app.function(
    volumes={DATASETS_DIR: datasets_volume, EMBEDDINGS_DIR: embeddings_volume}, 
    timeout=60000,
    # ephemeral_disk=2145728, # in MiB
)
def populate_indices(samples):
    import pandas as pd

    shard = samples.iloc[0]['shard']
    indices = samples['index'].tolist()

    print("reading shard", shard, len(indices))
    sample_df = pd.read_parquet(f"{SAMPLE_DIRECTORY}/{shard}")
    sample_df = sample_df.iloc[indices].copy()
    sample_df['feature'] = samples['feature'].tolist()
    sample_df['activation'] = samples['activation'].tolist()
    sample_df['top_indices'] = samples['top_indices'].tolist()
    sample_df['top_acts'] = samples['top_acts'].tolist()
    print("returning samples for", shard)

    return sample_df

@app.function(
    volumes={
        DATASETS_DIR: datasets_volume, 
        EMBEDDINGS_DIR: embeddings_volume
    }, 
    timeout=60000,
    # ephemeral_disk=2145728, # in MiB
)
def reduce_top10_indices(directory, save_directory, sae_directory, N):
    import os
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    files = [f for f in os.listdir(directory) if f.endswith('.parquet')]
    print("len files", len(files))

    import pandas as pd

    combined_indices_path = f"{save_directory}/combined_indices.parquet"
    if not os.path.exists(combined_indices_path):
        print("creating combined_indices")
        all_dataframes = []
        for file in files:
            print(f"Reading {file}")
            # Read from top directory
            df = pd.read_parquet(f"{directory}/{file}")
            
            # Read corresponding file from SAE directory to get top_indices and top_acts
            if os.path.exists(f"{sae_directory}/{file}"):
                sae_df = pd.read_parquet(f"{sae_directory}/{file}")
                # Ensure we have the right columns
                if 'top_indices' in sae_df.columns and 'top_acts' in sae_df.columns:
                    # Match records based on feature (assuming they're in the same order)
                    df['top_indices'] = sae_df['top_indices']
                    df['top_acts'] = sae_df['top_acts']
                    print(f"Added top_indices and top_acts columns from {file}")
                else:
                    print(f"Warning: top_indices or top_acts not found in {file} from SAE directory")
            else:
                print(f"Warning: file {file} not found in SAE directory")
                
            all_dataframes.append(df)

        # Concatenate all DataFrames into a single DataFrame
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print("combined")
        combined_df.to_parquet(combined_indices_path)
    else:
        print(f"{combined_indices_path} already exists. Loading it.")
        combined_df = pd.read_parquet(combined_indices_path)

    combined_df = combined_df.sort_values(by=['feature', 'activation'], ascending=[True, False])
    combined_df = combined_df.groupby('feature').head(N).reset_index(drop=True)
    print(f"writing top{N}")
    combined_df.to_parquet(f"{save_directory}/combined_indices_top{N}.parquet")
    embeddings_volume.commit()

    shard_counts = combined_df.groupby('shard').size().reset_index(name='count')
    print("shard_counts", shard_counts.head())

    print("Number of shards:", len(shard_counts))
    rows_by_shard = [combined_df[combined_df['shard'] == shard] for shard in combined_df['shard'].unique()]

    results = []
    for resp in populate_indices.map(rows_by_shard, order_outputs=False, return_exceptions=True):
        if isinstance(resp, Exception):
            print(f"Exception: {resp}")
            continue
        results.append(resp)

    print("concatenating final results")
    final_df = pd.concat(results, ignore_index=True)
    final_df = final_df.drop(columns=['index', '__index_level_0__'], errors='ignore')
    print("sorting final results")
    final_df = final_df.sort_values(by=['feature', 'activation'], ascending=[True, False])
    print("writing final results")
    final_df.to_parquet(f"{save_directory}/samples_top{N}.parquet")
    embeddings_volume.commit()
    return "done"


    # for resp in reduce_top10.map(pairs, order_outputs=False, return_exceptions=True):
    #     if isinstance(resp, Exception):
    #         print(f"Exception: {resp}")
    #         continue
    #     print(resp)



@app.local_entrypoint()
def main():
    reduce_top10_indices.remote(DIRECTORY, SAVE_DIRECTORY, SAE_DIRECTORY, 10)
    

