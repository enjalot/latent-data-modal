"""
Convert a directory of parquet files into a HuggingFace Dataset on a Modal volume.

Usage:
    modal run todataset.py
"""
from modal import App, Image, Volume, Secret

from config import get_dataset, embedding_dataset_name

ds = get_dataset()

DATASET_DIR = "/embeddings"
VOLUME = "embeddings"
DIRECTORY = f"{DATASET_DIR}/{embedding_dataset_name()}"
SAVE_DIRECTORY = f"{DIRECTORY}-HF"

volume = Volume.from_name(VOLUME, create_if_missing=True)
image = Image.debian_slim(python_version="3.10").pip_install("datasets", "pyarrow")
app = App(image=image)


@app.function(
    volumes={DATASET_DIR: volume},
    timeout=6000,
    secrets=[Secret.from_name("huggingface-secret")],
)
def convert_dataset():
    from datasets import load_dataset

    print(f"Loading parquets from {DIRECTORY}/train/*.parquet")
    dataset = load_dataset("parquet", data_files=f"{DIRECTORY}/train/*.parquet")
    print(f"Saving to {SAVE_DIRECTORY}")
    dataset.save_to_disk(SAVE_DIRECTORY, num_shards={"train": ds.num_files})
    volume.commit()
    print("Done!")


@app.local_entrypoint()
def main():
    convert_dataset.remote()
