"""
Reduce phase: merge per-shard top-N samples across all shards, then join
back to the original chunk text to produce the final samples file.

Usage:
    modal run top10reduce.py
"""
from modal import App, Image, Volume

from config import (
    get_dataset, get_model, get_sae,
    chunked_dataset_name, features_dataset_name,
)

ds = get_dataset()
model = get_model()
sae = get_sae()

N = 10  # keep top-N per feature after global merge (must match top10map.py)

EMBEDDINGS_DIR = "/embeddings"
DATASETS_DIR = "/datasets"

SAMPLE_DIRECTORY = f"{DATASETS_DIR}/{chunked_dataset_name()}/train"
SAE_DIRECTORY = f"{EMBEDDINGS_DIR}/{features_dataset_name()}/train"
DIRECTORY = f"{EMBEDDINGS_DIR}/{features_dataset_name()}-top{N}"
SAVE_DIRECTORY = f"{DIRECTORY}/combined"

embeddings_volume = Volume.from_name("embeddings", create_if_missing=True)
datasets_volume = Volume.from_name(ds.volume, create_if_missing=True)

image = (
    Image.debian_slim(python_version="3.10")
    .pip_install("pandas", "pyarrow")
    .add_local_file("config.py", "/root/config.py", copy=True)
)
app = App(image=image)


@app.function(
    volumes={DATASETS_DIR: datasets_volume, EMBEDDINGS_DIR: embeddings_volume},
    timeout=60000,
)
def populate_indices(samples):
    """Join top-N indices back to original chunk data for one shard."""
    import pandas as pd

    shard = samples.iloc[0]["shard"]
    indices = samples["index"].tolist()

    print(f"Reading shard {shard} ({len(indices)} samples)")
    sample_df = pd.read_parquet(f"{SAMPLE_DIRECTORY}/{shard}")
    sample_df = sample_df.iloc[indices].copy()
    sample_df["feature"] = samples["feature"].tolist()
    sample_df["activation"] = samples["activation"].tolist()
    sample_df["top_indices"] = samples["top_indices"].tolist()
    sample_df["top_acts"] = samples["top_acts"].tolist()

    return sample_df


@app.function(
    volumes={DATASETS_DIR: datasets_volume, EMBEDDINGS_DIR: embeddings_volume},
    timeout=60000,
)
def reduce_top_samples(directory, save_directory, sae_directory, n):
    """Global reduce: merge per-shard top-N → global top-N, then hydrate."""
    import os
    import pandas as pd

    os.makedirs(save_directory, exist_ok=True)

    files = [f for f in os.listdir(directory) if f.endswith(".parquet")]
    print(f"{len(files)} shard files to merge")

    combined_indices_path = f"{save_directory}/combined_indices.parquet"

    if not os.path.exists(combined_indices_path):
        all_dfs = []
        for file in files:
            df = pd.read_parquet(f"{directory}/{file}")
            sae_path = f"{sae_directory}/{file}"
            if os.path.exists(sae_path):
                sae_df = pd.read_parquet(sae_path)
                if "top_indices" in sae_df.columns and "top_acts" in sae_df.columns:
                    df["top_indices"] = sae_df["top_indices"]
                    df["top_acts"] = sae_df["top_acts"]
            all_dfs.append(df)

        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_parquet(combined_indices_path)
    else:
        print(f"Loading cached {combined_indices_path}")
        combined_df = pd.read_parquet(combined_indices_path)

    # Keep global top-N per feature
    combined_df = combined_df.sort_values(
        by=["feature", "activation"], ascending=[True, False]
    )
    combined_df = combined_df.groupby("feature").head(n).reset_index(drop=True)
    combined_df.to_parquet(f"{save_directory}/combined_indices_top{n}.parquet")
    embeddings_volume.commit()

    # Hydrate: join back to original chunk text
    rows_by_shard = [
        combined_df[combined_df["shard"] == shard]
        for shard in combined_df["shard"].unique()
    ]

    results = []
    for resp in populate_indices.map(rows_by_shard, order_outputs=False, return_exceptions=True):
        if isinstance(resp, Exception):
            print(f"EXCEPTION: {resp}")
        else:
            results.append(resp)

    final_df = pd.concat(results, ignore_index=True)
    final_df = final_df.drop(columns=["index", "__index_level_0__"], errors="ignore")
    final_df = final_df.sort_values(by=["feature", "activation"], ascending=[True, False])
    final_df.to_parquet(f"{save_directory}/samples_top{n}.parquet")
    embeddings_volume.commit()
    print(f"Done — {len(final_df)} samples written")
    return "done"


@app.local_entrypoint()
def main():
    reduce_top_samples.remote(DIRECTORY, SAVE_DIRECTORY, SAE_DIRECTORY, 10)
