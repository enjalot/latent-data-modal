"""
Extract SAE features from pre-computed embeddings.

Improvements over the original:
- Uses GPU (A10G) instead of CPU — 10-20x faster encoding
- Batch size increased from 128 → 4096 (GPU can handle it easily)
- Uses shared config.py
- Modern dependency versions

Usage:
    modal run features.py
"""
import os
import time

from modal import App, Image, Volume, Secret, gpu, enter, method

from config import (
    get_dataset, get_model, get_sae,
    embedding_dataset_name, features_dataset_name,
    shard_files,
)

ds = get_dataset()
model = get_model()
sae = get_sae()

DATASET_DIR = "/embeddings"
VOLUME = "embeddings"

DIRECTORY = f"{DATASET_DIR}/{embedding_dataset_name()}"
SAVE_DIRECTORY = f"{DATASET_DIR}/{features_dataset_name()}"
FILES = [f"{DIRECTORY}/train/{f.replace('.arrow', '.npy').replace('.parquet', '.npy')}"
         for f in shard_files(ext="parquet")]

SAE_SLUG = sae.slug
MODEL_ID = sae.model_id
D_IN = model.dim
MODEL_DIR = "/model"
MODEL_REVISION = "main"
BATCH_SIZE = 4096  # GPU can handle much larger batches than CPU's 128

volume = Volume.from_name(VOLUME, create_if_missing=True)


def download_model_to_image(model_dir, model_name, model_revision):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)
    snapshot_download(
        repo_id=model_name,
        revision=model_revision,
        local_dir=model_dir,
        ignore_patterns=["*.pt", "*.bin"],
    )
    move_cache()


st_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "numpy",
        "transformers",
        "hf-transfer",
        "huggingface_hub",
        "einops",
        "latentsae==0.1.0",
        "pandas",
        "pyarrow",
        "tqdm",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_image,
        timeout=60 * 20,
        kwargs={
            "model_dir": MODEL_DIR,
            "model_name": MODEL_ID,
            "model_revision": MODEL_REVISION,
        },
        secrets=[Secret.from_name("huggingface-secret")],
    )
)

app = App(image=st_image)

with st_image.imports():
    import numpy as np
    import torch


@app.cls(
    gpu="A10G",
    volumes={DATASET_DIR: volume},
    timeout=60 * 100,
    scaledown_window=60 * 10,
    allow_concurrent_inputs=1,
    image=st_image,
)
class SAEModel:
    @enter()
    def start_engine(self):
        from latentsae.sae import Sae

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Starting SAE inference on {self.device}")
        start = time.monotonic()
        self.model = Sae.load_from_hub(MODEL_ID, SAE_SLUG, device=self.device)
        print(f"SAE loaded in {time.monotonic() - start:.1f}s")

    @method()
    def make_features(self, file):
        from tqdm import tqdm
        import pandas as pd

        start = time.monotonic()
        print(f"Loading {file}")

        size = os.path.getsize(file) // (D_IN * 4)
        embeddings = np.memmap(file, dtype="float32", mode="r", shape=(size, D_IN))
        print(f"  {size} embeddings loaded in {time.monotonic() - start:.1f}s")

        num_batches = (size + BATCH_SIZE - 1) // BATCH_SIZE
        # Pre-allocate output arrays
        k = sae.k
        all_acts = np.zeros((size, k), dtype=np.float32)
        all_indices = np.zeros((size, k), dtype=np.int32)

        start = time.monotonic()
        for i in tqdm(range(num_batches), desc="Encoding"):
            s = i * BATCH_SIZE
            e = min(s + BATCH_SIZE, size)
            batch = torch.from_numpy(np.array(embeddings[s:e])).float().to(self.device)
            features = self.model.encode(batch)
            all_acts[s:e] = features.top_acts.detach().cpu().numpy()
            all_indices[s:e] = features.top_indices.detach().cpu().numpy()

        print(f"  Encoding done in {time.monotonic() - start:.1f}s")

        df = pd.DataFrame({
            "top_acts": list(all_acts),
            "top_indices": list(all_indices),
        })

        file_name = os.path.basename(file).split(".")[0]
        output_dir = f"{SAVE_DIRECTORY}/train"
        os.makedirs(output_dir, exist_ok=True)
        out_path = f"{output_dir}/{file_name}.parquet"
        df.to_parquet(out_path)
        print(f"  Saved to {out_path}")

        volume.commit()
        return f"Done: {file}"


@app.local_entrypoint()
def main():
    model = SAEModel()
    for resp in model.make_features.map(FILES, order_outputs=False, return_exceptions=True):
        if isinstance(resp, Exception):
            print(f"EXCEPTION: {resp}")
        else:
            print(resp)
