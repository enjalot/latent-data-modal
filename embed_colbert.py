"""
ColBERT late-interaction embedding via pylate on Modal GPU.
Outputs per-token embeddings (not pooled single vectors).

Usage:
    modal run embed_colbert.py::app_mxbai --chunk-size 120
    modal run embed_colbert.py::app_answerai --chunk-size 120
"""
import json, os, time
from modal import App, Image, Volume

DATASET_DIR = "/datasets"
EMBEDDING_DIR = "/embeddings"
N_CHUNKS = 2000

DATASET_VOLUME = Volume.from_name("datasets", create_if_missing=True)
EMBEDDING_VOLUME = Volume.from_name("embeddings", create_if_missing=True)

MODELS = {
    "mxbai-17m": {"id": "mixedbread-ai/mxbai-edge-colbert-v0-17m", "dim": 48, "params": "17M"},
    "mxbai-32m": {"id": "mixedbread-ai/mxbai-edge-colbert-v0-32m", "dim": 64, "params": "32M"},
    "colbertv2": {"id": "colbert-ir/colbertv2.0", "dim": 128, "params": "110M"},
    "answerai": {"id": "answerdotai/answerai-colbert-small-v1", "dim": 96, "params": "33M"},
}

pylate_img = (
    Image.debian_slim(python_version="3.11")
    .pip_install("pylate", "numpy", "pandas", "pyarrow", "huggingface_hub")
    .run_commands(
        'python3 -c "from huggingface_hub import snapshot_download; snapshot_download(\'mixedbread-ai/mxbai-edge-colbert-v0-17m\')"',
        'python3 -c "from huggingface_hub import snapshot_download; snapshot_download(\'mixedbread-ai/mxbai-edge-colbert-v0-32m\')"',
        'python3 -c "from huggingface_hub import snapshot_download; snapshot_download(\'colbert-ir/colbertv2.0\')"',
        'python3 -c "from huggingface_hub import snapshot_download; snapshot_download(\'answerdotai/answerai-colbert-small-v1\')"',
    )
)


def _bench(gpu_type, model_key, model_id, chunk_size):
    import numpy as np, pandas as pd
    from pylate import models

    cost_hr = {"T4": 0.60, "A10G": 1.00, "L40S": 1.95, "A100-40": 2.00}.get(gpu_type, 1.00)
    fp = f"{DATASET_DIR}/wikipedia-en-chunked-{chunk_size}/train/data-00000-of-00041.parquet"
    df = pd.read_parquet(fp).head(N_CHUNKS).copy()
    n = len(df)
    texts = [row["chunk_text"] if row["chunk_text"].strip() else " " for _, row in df.iterrows()]
    total_tokens = int(df["chunk_token_count"].sum())

    print(f"\npylate ColBERT: {model_key} | {gpu_type} | {chunk_size}-tok | {n} chunks")

    t_load = time.monotonic()
    model = models.ColBERT(model_name_or_path=model_id, device="cuda")
    load_s = time.monotonic() - t_load
    print(f"  Model loaded in {load_s:.1f}s")

    # Warmup
    _ = model.encode(texts[:4], batch_size=4, show_progress_bar=False, is_query=False)

    t0 = time.monotonic()
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, is_query=False)
    wall = time.monotonic() - t0

    # emb is list of numpy arrays, each (n_tokens, dim)
    dim = emb[0].shape[-1]
    tokens_per_doc = [e.shape[0] for e in emb]
    total_vectors = sum(tokens_per_doc)
    avg_vectors = np.mean(tokens_per_doc)
    cps = n / wall
    tps = total_tokens / wall

    storage_fp16_per_M = total_vectors / n * dim * 2 * 1e6 / 1e9  # GB per million chunks
    cost_per_M = cost_hr * (1_000_000 / cps) / 3600

    r = {
        "engine": "pylate", "model": model_key, "gpu": gpu_type,
        "dim": dim, "chunks": chunk_size, "n": n, "tokens": total_tokens,
        "avg_vectors_per_chunk": round(float(avg_vectors), 1),
        "total_vectors": int(total_vectors),
        "load_s": round(load_s, 2), "wall_s": round(wall, 2),
        "cps": round(cps, 1), "tps": round(tps, 0),
        "cost_hr": cost_hr, "cost_per_M": round(cost_per_M, 2),
        "storage_fp16_gb_per_M": round(storage_fp16_per_M, 2),
    }

    out = f"{EMBEDDING_DIR}/benchmark_colbert"
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/pylate_{model_key}_{gpu_type}_{chunk_size}.json", "w") as f:
        json.dump(r, f, indent=2)
    # Save first 10 doc embeddings for verification
    np.savez_compressed(f"{out}/pylate_{model_key}_{gpu_type}_{chunk_size}_sample.npz",
                        *[emb[i] for i in range(min(10, len(emb)))])
    EMBEDDING_VOLUME.commit()

    print(f"  {cps:.1f} chunks/s | dim={dim} | {avg_vectors:.0f} vecs/chunk | ${cost_per_M:.2f}/M")
    print(json.dumps(r, indent=2))
    return r


# ── Generic app: embed any ColBERT model on any GPU ──
app = App("colbert-embed")

@app.function(gpu="A10G", image=pylate_img,
    volumes={DATASET_DIR: DATASET_VOLUME, EMBEDDING_DIR: EMBEDDING_VOLUME}, timeout=600)
def embed_a10g(model_key: str, chunk_size: int = 120):
    m = MODELS[model_key]
    return _bench("A10G", model_key, m["id"], chunk_size)

@app.function(gpu="L40S", image=pylate_img,
    volumes={DATASET_DIR: DATASET_VOLUME, EMBEDDING_DIR: EMBEDDING_VOLUME}, timeout=600)
def embed_l40s(model_key: str, chunk_size: int = 120):
    m = MODELS[model_key]
    return _bench("L40S", model_key, m["id"], chunk_size)

@app.local_entrypoint()
def main(model: str = "mxbai-32m", gpu: str = "A10G", chunk_size: int = 120):
    if model not in MODELS:
        print(f"Available models: {list(MODELS.keys())}")
        return
    fn = embed_a10g if gpu == "A10G" else embed_l40s
    r = fn.remote(model, chunk_size)
    print(json.dumps(r, indent=2))
