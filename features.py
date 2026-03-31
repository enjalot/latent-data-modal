"""
Extract SAE features from pre-computed embeddings.

Supports both GPU (for large SAEs) and CPU (for small SAEs like MiniLM 8x).
The compute mode is configured in config.py per SAE model.

Usage:
    modal run features.py                          # process all shards
    modal run features.py --start-shard 60         # process shards 60+
    modal run features.py --start-shard 60 --end-shard 80  # process shards 60-79
    modal run features.py --dry-run                # just list what would run
"""
import os
import time

from modal import App, Image, Volume, Secret, enter, method

from config import (
    get_dataset, get_model, get_sae,
    embedding_dataset_name, features_dataset_name,
    shard_files,
)

# ---------------------------------------------------------------------------
# Read config
# ---------------------------------------------------------------------------

ds = get_dataset()
model_cfg = get_model()
sae_config = get_sae()

USE_GPU = sae_config.compute == "gpu"
BATCH_SIZE = sae_config.batch_size

DATASET_DIR = "/embeddings"
VOLUME = "embeddings"

DIRECTORY = f"{DATASET_DIR}/{embedding_dataset_name()}"
SAVE_DIRECTORY = f"{DATASET_DIR}/{features_dataset_name()}"
FILES = [f"{DIRECTORY}/train/{f.replace('.arrow', '.npy').replace('.parquet', '.npy')}"
         for f in shard_files(ext="parquet")]

SAE_SLUG = sae_config.slug
MODEL_ID = sae_config.model_id
D_IN = model_cfg.dim
K = sae_config.k
MODEL_DIR = "/model"
MODEL_REVISION = "main"

# Cost rates ($/hr)
COST_CPU_8CORE = 0.56
COST_A10G = 1.10

volume = Volume.from_name(VOLUME, create_if_missing=True)


# ---------------------------------------------------------------------------
# Image build helpers
# ---------------------------------------------------------------------------

def download_model_to_image(model_dir, model_name, model_revision):
    """Download embedding model weights during image build."""
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


def download_sae_to_image(model_dir, model_id, sae_slug):
    """Download SAE weights during image build so they're baked in."""
    from latentsae.sae import Sae
    import os

    sae_path = os.path.join(model_dir, sae_slug)
    if not os.path.exists(sae_path):
        sae = Sae.load_from_hub(model_id, sae_slug)
        sae.save_to_disk(sae_path)


# ---------------------------------------------------------------------------
# Image definition
# ---------------------------------------------------------------------------

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
        "pyarrow",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_commands([
        # Download embedding model (for transformers compatibility check)
        f"python -c \""
        f"from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id='{model_cfg.model_id}', revision='main', "
        f"local_dir='{MODEL_DIR}', ignore_patterns=['*.pt', '*.bin'])"
        f"\"",
        # Download SAE weights
        f"python -c \""
        f"from latentsae.sae import Sae; "
        f"sae = Sae.load_from_hub('{MODEL_ID}', '{SAE_SLUG}'); "
        f"sae.save_to_disk('{MODEL_DIR}/{SAE_SLUG}')"
        f"\"",
    ], secrets=[Secret.from_name("huggingface-secret")])
)

app = App(image=st_image)

with st_image.imports():
    import numpy as np
    import torch


# ---------------------------------------------------------------------------
# Common encode logic
# ---------------------------------------------------------------------------

def encode_shard(model, file, device, batch_size, k, d_in):
    """Encode one shard of embeddings. Returns (output_path, stats dict)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    stats = {}

    # Load embeddings
    t0 = time.perf_counter()
    size = os.path.getsize(file) // (d_in * 4)
    embeddings = np.memmap(file, dtype="float32", mode="r", shape=(size, d_in))
    stats["load_time"] = time.perf_counter() - t0
    stats["n_samples"] = size

    # Pre-allocate output arrays
    num_batches = (size + batch_size - 1) // batch_size
    all_acts = np.zeros((size, k), dtype=np.float32)
    all_indices = np.zeros((size, k), dtype=np.int32)

    # Encode
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(num_batches):
            s = i * batch_size
            e = min(s + batch_size, size)
            batch = torch.from_numpy(np.array(embeddings[s:e])).float().to(device)
            features = model.encode(batch)
            all_acts[s:e] = features.top_acts.cpu().numpy()
            all_indices[s:e] = features.top_indices.cpu().numpy()
    stats["encode_time"] = time.perf_counter() - t0
    stats["throughput"] = size / stats["encode_time"] if stats["encode_time"] > 0 else 0

    # Save with pyarrow (much faster than pandas list-of-arrays)
    t0 = time.perf_counter()
    table = pa.table({
        "top_acts": pa.FixedSizeListArray.from_arrays(
            pa.array(all_acts.ravel(), type=pa.float32()),
            list_size=k,
        ),
        "top_indices": pa.FixedSizeListArray.from_arrays(
            pa.array(all_indices.ravel(), type=pa.int32()),
            list_size=k,
        ),
    })

    file_name = os.path.basename(file).split(".")[0]
    output_dir = f"{SAVE_DIRECTORY}/train"
    os.makedirs(output_dir, exist_ok=True)
    out_path = f"{output_dir}/{file_name}.parquet"
    pq.write_table(table, out_path)
    stats["save_time"] = time.perf_counter() - t0

    return out_path, stats


def format_shard_summary(file, stats):
    """Format a one-line summary for a completed shard."""
    name = os.path.basename(file).split(".")[0]
    n = stats["n_samples"]
    t_enc = stats["encode_time"]
    tp = stats["throughput"]
    t_load = stats["load_time"]
    t_save = stats["save_time"]
    total = t_load + t_enc + t_save
    return (
        f"Shard {name}: {n:,} samples, {total:.1f}s total "
        f"(load {t_load:.1f}s, encode {t_enc:.1f}s, save {t_save:.1f}s), "
        f"{tp:,.0f} samples/sec"
    )


# ---------------------------------------------------------------------------
# GPU class (used when sae.compute == "gpu")
# ---------------------------------------------------------------------------

@app.cls(
    gpu="A10G",
    volumes={DATASET_DIR: volume},
    timeout=60 * 100,
    scaledown_window=60 * 10,
    image=st_image,
)
class SAEModelGPU:
    @enter()
    def start(self):
        from latentsae.sae import Sae

        print("Starting SAE inference on GPU (A10G)")
        t0 = time.perf_counter()
        self.model = Sae.load_from_disk(f"{MODEL_DIR}/{SAE_SLUG}", device="cuda")
        self.device = torch.device("cuda")
        print(f"SAE loaded in {time.perf_counter() - t0:.1f}s")

    @method()
    def make_features(self, file):
        print(f"Processing {file}")
        path, stats = encode_shard(self.model, file, self.device, BATCH_SIZE, K, D_IN)
        summary = format_shard_summary(file, stats)
        print(f"  {summary}")
        print(f"  Saved to {path}")
        volume.commit()
        return summary


# ---------------------------------------------------------------------------
# CPU class (used when sae.compute == "cpu")
# ---------------------------------------------------------------------------

@app.cls(
    cpu=8,
    memory=16384,
    volumes={DATASET_DIR: volume},
    timeout=60 * 100,
    scaledown_window=60 * 10,
    image=st_image,
)
class SAEModelCPU:
    @enter()
    def start(self):
        from latentsae.sae import Sae

        print("Starting SAE inference on CPU (8-core)")
        t0 = time.perf_counter()
        self.model = Sae.load_from_disk(f"{MODEL_DIR}/{SAE_SLUG}", device="cpu")
        self.device = torch.device("cpu")
        print(f"SAE loaded in {time.perf_counter() - t0:.1f}s")

    @method()
    def make_features(self, file):
        print(f"Processing {file}")
        path, stats = encode_shard(self.model, file, self.device, BATCH_SIZE, K, D_IN)
        summary = format_shard_summary(file, stats)
        print(f"  {summary}")
        print(f"  Saved to {path}")
        volume.commit()
        return summary


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(start_shard: int = 0, end_shard: int = -1, dry_run: bool = False):
    files_to_process = FILES[start_shard:end_shard if end_shard > 0 else None]

    compute_label = "GPU (A10G)" if USE_GPU else "CPU (8-core)"
    cost_per_hour = COST_A10G if USE_GPU else COST_CPU_8CORE

    print(f"SAE: {MODEL_ID} ({SAE_SLUG})")
    print(f"Compute: {compute_label}, batch_size={BATCH_SIZE}")
    print(f"Input: {DIRECTORY}")
    print(f"Output: {SAVE_DIRECTORY}")
    print(f"Shards: {len(files_to_process)} (range [{start_shard}:{end_shard if end_shard > 0 else 'end'}])")
    print()

    if dry_run:
        print(f"Would process {len(files_to_process)} shards:")
        for f in files_to_process:
            print(f"  {f}")
        return

    Model = SAEModelGPU if USE_GPU else SAEModelCPU
    model = Model()

    wall_start = time.perf_counter()
    total_samples = 0
    completed = 0
    errors = 0

    for resp in model.make_features.map(
        files_to_process, order_outputs=False, return_exceptions=True
    ):
        if isinstance(resp, Exception):
            print(f"EXCEPTION: {resp}")
            errors += 1
        else:
            print(resp)
            # Parse sample count from summary (format: "Shard X: N samples, ...")
            try:
                parts = resp.split(": ")[1]
                n_str = parts.split(" samples")[0].replace(",", "")
                total_samples += int(n_str)
            except (IndexError, ValueError):
                pass
            completed += 1

    wall_time = time.perf_counter() - wall_start
    cost_estimate = (wall_time / 3600) * cost_per_hour

    print()
    print("=" * 60)
    print(f"DONE: {completed} shards completed, {errors} errors")
    print(f"Total samples: {total_samples:,}")
    print(f"Wall-clock time: {wall_time:.1f}s ({wall_time/60:.1f} min)")
    if wall_time > 0:
        print(f"Overall throughput: {total_samples / wall_time:,.0f} samples/sec")
    print(f"Estimated cost: ${cost_estimate:.2f} ({compute_label} @ ${cost_per_hour:.2f}/hr)")
    print("=" * 60)
