"""
Download a dataset from HuggingFace to a Modal volume.

Usage:
    modal run download.py
"""
from modal import App, Image, Volume, Secret

from config import get_dataset

ds = get_dataset()

DATASET_DIR = "/data"
HF_CACHE_DIR = f"{DATASET_DIR}/cache"

volume = Volume.from_name(ds.volume, create_if_missing=True)
image = Image.debian_slim(python_version="3.10").pip_install("datasets")
app = App(image=image)


@app.function(
    volumes={DATASET_DIR: volume},
    timeout=60000,
    ephemeral_disk=int(3145728),
    secrets=[Secret.from_name("huggingface-secret")],
)
def download_dataset():
    import os
    import time
    from datasets import load_dataset, DownloadConfig, logging

    os.environ["HF_HOME"] = HF_CACHE_DIR
    logging.set_verbosity_debug()

    start = time.time()
    kwargs = dict(
        num_proc=6,
        trust_remote_code=True,
        download_config=DownloadConfig(resume_download=True, cache_dir=HF_CACHE_DIR),
    )
    if ds.hf_data_files:
        dataset = load_dataset(ds.name, data_files=ds.hf_data_files, **kwargs)
    elif ds.hf_subset:
        dataset = load_dataset(ds.name, ds.hf_subset, **kwargs)
    else:
        dataset = load_dataset(ds.name, **kwargs)

    print(f"Download complete in {time.time() - start:.0f}s")
    dataset.save_to_disk(f"{DATASET_DIR}/{ds.save_name}")
    volume.commit()


@app.local_entrypoint()
def main():
    download_dataset.remote()
