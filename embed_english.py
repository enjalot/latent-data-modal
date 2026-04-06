"""
Embed 2M English chunks from fineweb-edu-10BT with jina-nano on A10G.
Reads from pre-chunked parquet on embedding-fineweb-edu volume.
Matches the multilingual setup: 2M chunks, 500-tok, jina-nano, fp16.

Usage:
    modal run embed_english.py
"""
import json
import os
import time

from modal import App, Image, Volume

DATASET_DIR = "/data"
EMBEDDING_DIR = "/embeddings"
MAX_CHUNKS = 2_000_000
BATCH_SIZE = 512

FINEWEB_VOLUME = Volume.from_name("embedding-fineweb-edu", create_if_missing=True)
EMBEDDING_VOLUME = Volume.from_name("embeddings", create_if_missing=True)

vllm_img = (
    Image.debian_slim(python_version="3.11")
    .pip_install("vllm>=0.18", "numpy", "pandas", "pyarrow", "huggingface_hub")
    .run_commands("pip install 'transformers>=5.0,<5.1'")
    .run_commands(
        'python3 -c "from huggingface_hub import snapshot_download; snapshot_download(\'jinaai/jina-embeddings-v5-text-nano-retrieval\')"',
    )
)

app = App("embed-english-jina")


@app.function(
    gpu="A10G", image=vllm_img,
    volumes={DATASET_DIR: FINEWEB_VOLUME, EMBEDDING_DIR: EMBEDDING_VOLUME},
    timeout=7200,
)
def embed_english():
    import numpy as np
    import pandas as pd
    from vllm import LLM

    emb_dir = f"{EMBEDDING_DIR}/fineweb-edu-10BT-chunked-500-jina-nano/train"
    emb_path = f"{emb_dir}/000_00000.npy"

    if os.path.exists(emb_path):
        size_mb = os.path.getsize(emb_path) / 1e6
        print(f"Already embedded ({size_mb:.0f} MB)")
        return {"status": "cached", "size_mb": round(size_mb)}

    # Load chunks from pre-chunked shards until we have 2M
    chunk_dir = f"{DATASET_DIR}/fineweb-edu-sample-10BT-chunked-500/train"
    shards = sorted([f for f in os.listdir(chunk_dir) if f.endswith(".parquet")])
    print(f"Found {len(shards)} shards in {chunk_dir}")

    texts = []
    for shard in shards:
        df = pd.read_parquet(f"{chunk_dir}/{shard}", columns=["chunk_text"])
        texts.extend(df["chunk_text"].tolist())
        print(f"  Loaded {shard}: {len(df):,} chunks (total: {len(texts):,})")
        del df
        if len(texts) >= MAX_CHUNKS:
            break

    texts = texts[:MAX_CHUNKS]
    n = len(texts)
    print(f"Using {n:,} English chunks")

    # Load model
    t_load = time.monotonic()
    model = LLM(
        model="jinaai/jina-embeddings-v5-text-nano-retrieval",
        runner="pooling", dtype="float16",
        max_model_len=1024,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
    )
    print(f"Model loaded in {time.monotonic()-t_load:.0f}s")

    # Embed with memmap streaming
    dim = 768
    os.makedirs(emb_dir, exist_ok=True)
    mmap = np.memmap(emb_path, dtype="float16", mode="w+", shape=(n, dim))

    t0 = time.monotonic()
    idx = 0
    for i in range(0, n, BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
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
                    pass
                idx += 1

        if idx % 50_000 < BATCH_SIZE:
            mmap.flush()

        if (i // BATCH_SIZE) % 100 == 0:
            elapsed = time.monotonic() - t0
            cps = idx / elapsed if elapsed > 0 else 0
            eta = (n - idx) / cps / 60 if cps > 0 else 0
            cost = elapsed / 3600
            print(f"  {idx:,}/{n:,} ({cps:.0f}/s, ~{eta:.0f}m left, ${cost:.2f})")

    mmap.flush()
    del mmap, texts, model

    embed_s = time.monotonic() - t0
    cps = n / embed_s
    cost = embed_s / 3600
    size_mb = os.path.getsize(emb_path) / 1e6
    EMBEDDING_VOLUME.commit()

    print(f"DONE — {n:,} chunks, {cps:.0f}/s, {embed_s:.0f}s, ${cost:.2f}, {size_mb:.0f} MB")
    return {"n": n, "cps": round(cps), "seconds": round(embed_s),
            "cost": round(cost, 2), "size_mb": round(size_mb), "status": "embedded"}


@app.local_entrypoint()
def main():
    r = embed_english.remote()
    print(json.dumps(r, indent=2))
