#!/usr/bin/env python3
"""Embed the 41.97M capedia-en 120-token chunks with all-MiniLM-L6-v2 on the 5090.

Prepared 2026-08-20 for the owner GPU window (see
latent-labs/guides/plan-gpu-window-2026-08-21.md §1). Follows the local embed
pattern of /data/raw/common-corpus/embed_probes.py, scaled to the full corpus:

  input : /data/chunks/communityarchive-tweets/train/NNN_NNNNNNNNN.parquet
  output: /data/embeddings/communityarchive-tweets-all-MiniLM-L6-v2/train/
            NNN_NNNNNNNNN.npy   # (rows, 384) float16, row-parallel with
                                      # the input parquet (title sidecar joins
                                      # by shard+row, no ids file needed)
            manifest.json             # written last, after all shards verify

fp16 on disk (~32 GB total); consumers memmap lazily and cast per-batch
(feedback_large_shard_memory rule). Resumable per shard: a finished shard is
skipped, a partial shard is rewritten from scratch (atomic .tmp rename).

Run (only when the GPU is free — check nvidia-smi first):
  cd ~/code/latent-basemap && systemd-run --user --unit=ca-embed \
    .venv/bin/python /home/enjalot/code/latent-data-modal/embed_capedia_local.py

Env: BATCH (default 2048), SHARD_LIMIT (debug: only N shards), SMOKE (embed
10k rows of shard 0 to /tmp and exit — throughput check without touching
/data/embeddings).
"""
import glob
import json
import os
import sys
import time

os.environ.setdefault("HF_HOME", "/data/hf")

import numpy as np

CHUNKS = "/data/chunks/communityarchive-tweets/train"
OUT = "/data/embeddings/communityarchive-tweets-all-MiniLM-L6-v2/train"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
BATCH = int(os.environ.get("BATCH", "2048"))
SHARD_LIMIT = int(os.environ.get("SHARD_LIMIT", "0"))
SMOKE = bool(int(os.environ.get("SMOKE", "0")))
DIM = 384


def embed_texts(model, texts: list[str]) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=BATCH,
        convert_to_numpy=True,
        normalize_embeddings=False,  # substrate pipelines L2-normalize at use
        show_progress_bar=False,
    ).astype(np.float16)


def main() -> int:
    import pandas as pd
    import torch
    from sentence_transformers import SentenceTransformer

    shards = sorted(glob.glob(f"{CHUNKS}/*.parquet"))
    if SHARD_LIMIT:
        shards = shards[:SHARD_LIMIT]
    if not shards:
        print(f"no chunk shards at {CHUNKS}", file=sys.stderr)
        return 1

    model = SentenceTransformer(MODEL_ID, device="cuda")
    model = model.half()
    print(f"model {MODEL_ID} fp16, batch {BATCH}, {len(shards)} shards", flush=True)

    if SMOKE:
        df = pd.read_parquet(shards[0], columns=["chunk_text"]).head(10_000)
        t0 = time.time()
        vecs = embed_texts(model, df["chunk_text"].tolist())
        dt = time.time() - t0
        print(f"SMOKE: 10k rows in {dt:.1f}s = {10_000/dt:,.0f} chunks/s "
              f"-> 42M in ~{10_000_000/(10_000/dt)/3600:.1f} h; shape {vecs.shape}")
        return 0

    os.makedirs(OUT, exist_ok=True)
    total_rows = 0
    t_start = time.time()
    for path in shards:
        name = os.path.basename(path).replace(".parquet", ".npy")
        out = os.path.join(OUT, name)
        if os.path.exists(out):
            total_rows += np.load(out, mmap_mode="r").shape[0]
            print(f"skip {name} (done)", flush=True)
            continue
        df = pd.read_parquet(path, columns=["chunk_text"])
        texts = df["chunk_text"].tolist()
        del df
        t0 = time.time()
        vecs = embed_texts(model, texts)
        assert vecs.shape == (len(texts), DIM), vecs.shape
        tmp = out + ".tmp.npy"
        np.save(tmp, vecs)
        os.rename(tmp, out)
        dt = time.time() - t0
        total_rows += len(texts)
        rate = len(texts) / dt
        done_frac = total_rows / 10_000_000
        print(f"{name}: {len(texts):,} rows in {dt/60:.1f} min ({rate:,.0f}/s) "
              f"| total {total_rows:,} ({done_frac:.0%})", flush=True)
        del vecs, texts
        torch.cuda.empty_cache()

    manifest = {
        "model": MODEL_ID,
        "dim": DIM,
        "dtype": "float16",
        "rows": total_rows,
        "shards": len(shards),
        "row_parallel_with": CHUNKS,
        "normalized": False,
        "wall_s": time.time() - t_start,
        "note": "rows are row-parallel with the input parquet shards; join "
                "title/url by shard+row for sidecars.",
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"DONE {total_rows:,} rows in {(time.time()-t_start)/3600:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
