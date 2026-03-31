"""
Upload a dataset from a Modal volume to HuggingFace Hub.

Usage:
    modal run upload.py
"""
from modal import App, Image, Volume, Secret

from config import get_dataset, chunked_dataset_name

ds = get_dataset()

DATASET_DIR = "/data"
VOLUME = ds.volume
# Set this to the HF repo you want to push to
HF_REPO = f"enjalot/{chunked_dataset_name()}"
DIRECTORY = f"{DATASET_DIR}/{chunked_dataset_name()}-HF"

volume = Volume.from_name(VOLUME, create_if_missing=True)
image = Image.debian_slim(python_version="3.10").pip_install(
    "datasets", "huggingface_hub"
)
app = App(image=image)


@app.function(
    volumes={DATASET_DIR: volume},
    timeout=60000,
    secrets=[Secret.from_name("huggingface-secret")],
)
def upload_dataset(directory, repo):
    import os
    import time
    from huggingface_hub import HfApi
    from datasets import load_from_disk

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo, private=False, repo_type="dataset", exist_ok=True)

    print(f"Loading from {directory}")
    dataset = load_from_disk(directory)

    print(f"Pushing to {repo}")
    start = time.perf_counter()
    max_retries = 10
    for attempt in range(max_retries):
        try:
            dataset.push_to_hub(repo, num_shards={"train": ds.num_files})
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(5 * (attempt + 1))
            else:
                raise
    print(f"Uploaded in {time.perf_counter() - start:.0f}s")


@app.local_entrypoint()
def main():
    upload_dataset.remote(DIRECTORY, HF_REPO)
