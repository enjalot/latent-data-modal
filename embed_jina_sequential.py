"""
Embed chunked FineWeb-2 data with jina-nano — sequential, one language at a time.
Streams embeddings to disk in batches to avoid RAM accumulation.

Usage:
    modal run embed_jina_sequential.py --lang swe_Latn --gpu A10G
"""
import json
import os
import time

from modal import App, Image, Volume

DATASET_DIR = "/datasets"
EMBEDDING_DIR = "/embeddings"
CHUNK_SIZE = 500
BATCH_SIZE = 512
WRITE_EVERY = 50_000  # write to disk every 50K embeddings

DATASET_VOLUME = Volume.from_name("datasets", create_if_missing=True)
EMBEDDING_VOLUME = Volume.from_name("embeddings", create_if_missing=True)

vllm_img = (
    Image.debian_slim(python_version="3.11")
    .pip_install("vllm>=0.18", "numpy", "pandas", "pyarrow", "huggingface_hub")
    .run_commands("pip install 'transformers>=5.0,<5.1'")
    .run_commands(
        'python3 -c "from huggingface_hub import snapshot_download; snapshot_download(\'jinaai/jina-embeddings-v5-text-nano-retrieval\')"',
    )
)

app = App("embed-jina-seq")


@app.function(
    gpu="A10G", image=vllm_img,
    volumes={DATASET_DIR: DATASET_VOLUME, EMBEDDING_DIR: EMBEDDING_VOLUME},
    timeout=7200,
)
def embed_a10g(lang_code: str):
    return _embed(lang_code, "A10G")


@app.function(
    gpu="L40S", image=vllm_img,
    volumes={DATASET_DIR: DATASET_VOLUME, EMBEDDING_DIR: EMBEDDING_VOLUME},
    timeout=7200,
)
def embed_l40s(lang_code: str):
    return _embed(lang_code, "L40S")


def _embed(lang_code, gpu_type):
    import numpy as np
    import pandas as pd
    from vllm import LLM

    cost_hr = 1.00 if gpu_type == "A10G" else 1.95
    chunk_path = f"{DATASET_DIR}/fineweb2-{lang_code}-chunked-{CHUNK_SIZE}/train/000_00000.parquet"
    emb_dir = f"{EMBEDDING_DIR}/fineweb2-{lang_code}-chunked-{CHUNK_SIZE}-jina-nano/train"
    emb_path = f"{emb_dir}/000_00000.npy"

    # Check if already done
    if os.path.exists(emb_path):
        size_mb = os.path.getsize(emb_path) / 1e6
        print(f"{lang_code}: already embedded ({size_mb:.0f} MB)")
        return {"lang": lang_code, "status": "cached", "size_mb": round(size_mb)}

    # Load chunks
    print(f"{lang_code}: loading chunks from {chunk_path}")
    df = pd.read_parquet(chunk_path, columns=["chunk_text"])
    texts = df["chunk_text"].tolist()
    n = len(texts)
    del df
    print(f"{lang_code}: {n:,} chunks loaded")

    # Load model
    t_load = time.monotonic()
    model = LLM(
        model="jinaai/jina-embeddings-v5-text-nano-retrieval",
        runner="pooling", dtype="float16",
        max_model_len=1024,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
    )
    print(f"{lang_code}: model loaded in {time.monotonic()-t_load:.0f}s")

    # Embed — stream to memmap to avoid RAM explosion
    os.makedirs(emb_dir, exist_ok=True)
    dim = 768  # jina-nano output dim

    # Use memmap for streaming writes
    mmap = np.memmap(emb_path, dtype="float16", mode="w+", shape=(n, dim))

    t0 = time.monotonic()
    idx = 0
    for i in range(0, n, BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        # Truncate overlong chunks
        batch = [t[:3000] if len(t) > 3000 else t for t in batch]
        try:
            outputs = model.encode(batch, pooling_task="embed")
            for o in outputs:
                mmap[idx] = np.array(o.outputs.data, dtype="float16")
                idx += 1
        except Exception:
            for t in batch:
                try:
                    out = model.encode([t[:2000]], pooling_task="embed")
                    mmap[idx] = np.array(out[0].outputs.data, dtype="float16")
                except Exception:
                    pass  # leave as zeros
                idx += 1

        # Flush periodically
        if idx % WRITE_EVERY < BATCH_SIZE:
            mmap.flush()

        # Progress
        if (i // BATCH_SIZE) % 100 == 0:
            elapsed = time.monotonic() - t0
            cps = idx / elapsed if elapsed > 0 else 0
            eta_min = (n - idx) / cps / 60 if cps > 0 else 0
            cost_so_far = elapsed * cost_hr / 3600
            print(f"  {lang_code}: {idx:,}/{n:,} ({cps:.0f}/s, ~{eta_min:.0f}m left, ${cost_so_far:.2f} so far)")

    mmap.flush()
    del mmap, texts, model

    embed_s = time.monotonic() - t0
    cps = n / embed_s
    cost = embed_s * cost_hr / 3600
    size_mb = os.path.getsize(emb_path) / 1e6

    EMBEDDING_VOLUME.commit()

    print(f"{lang_code}: DONE — {n:,} chunks, {cps:.0f}/s, {embed_s:.0f}s, ${cost:.2f}, {size_mb:.0f} MB")
    return {
        "lang": lang_code, "n": n, "cps": round(cps),
        "seconds": round(embed_s), "cost": round(cost, 2),
        "size_mb": round(size_mb), "gpu": gpu_type, "status": "embedded",
    }


@app.local_entrypoint()
def main(lang: str = "swe_Latn", gpu: str = "A10G"):
    print(f"Embedding {lang} on {gpu}")
    fn = embed_a10g if gpu == "A10G" else embed_l40s
    r = fn.remote(lang)
    print(json.dumps(r, indent=2))
