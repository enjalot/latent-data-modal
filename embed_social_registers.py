#!/usr/bin/env python3
"""Embed the first 2,000,000 120-token chunks of the Twitter100M + Bluesky-5M
social register corpora with all-MiniLM-L6-v2 (384-d), in the SAME on-disk
format reddit-2m / communityarchive-2m used, so the substrate builder can read
them identically.

On-disk format (verified against reddit/CA outputs 2026-08-25):
    real .npy files (magic \\x93NUMPY, 128-byte header), dtype float16,
    shape (rows, 384), NOT L2-normalized at write time
    (all-MiniLM-L6-v2's Normalize layer already yields ~unit norm; matches
    reddit/CA where normalize_embeddings=False). One .npy per input parquet
    shard, row-parallel with it, + manifest.json written last.

Embed logic copied from embed_reddit_local.py / embed_ca_local.py, plus:
  - DEVICE env: "cpu" (default) or "cuda". On cuda we .half() the model like the
    originals; on cpu we keep fp32 compute and cast the OUTPUT to fp16 for storage
    (fp16 matmul is unsupported/slow on CPU), so the on-disk dtype is identical.
  - MAX_ROWS cap (default 2,000,000): stop after that many chunks per corpus
    (truncating the shard that crosses the cap).

Write-once & resumable per shard (finished shard skipped; partial rewritten via
atomic .tmp rename).

Validate (dry run, embeds 100 rows, prints shape/dtype/norm; no write):
  HF_HOME=/data/hf .venv/bin/python embed_social_registers.py twitter --dry-run
  HF_HOME=/data/hf .venv/bin/python embed_social_registers.py bluesky --dry-run

Full run (launch via systemd, see report). DEVICE=cuda only when GPU is free:
  HF_HOME=/data/hf DEVICE=cuda .venv/bin/python embed_social_registers.py twitter
  HF_HOME=/data/hf DEVICE=cuda .venv/bin/python embed_social_registers.py bluesky
"""
import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault("HF_HOME", "/data/hf")

import numpy as np

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384
BATCH = int(os.environ.get("BATCH", "2048"))
DEVICE = os.environ.get("DEVICE", "cpu")
MAX_ROWS = int(os.environ.get("MAX_ROWS", "2000000"))

CORPORA = {
    "twitter": {
        "chunks": "/data/chunks/twitter100m-chunked-120/train",
        "out": "/data/embeddings/twitter100m-chunked-120-all-MiniLM-L6-v2/train",
    },
    "bluesky": {
        "chunks": "/data/chunks/bluesky-5m-chunked-120/train",
        "out": "/data/embeddings/bluesky-5m-chunked-120-all-MiniLM-L6-v2/train",
    },
}


def make_model():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_ID, device=DEVICE)
    if DEVICE == "cuda":
        model = model.half()
    return model


def embed_texts(model, texts):
    return model.encode(
        texts,
        batch_size=BATCH,
        convert_to_numpy=True,
        normalize_embeddings=False,   # substrate pipelines L2-normalize at use
        show_progress_bar=False,
    ).astype(np.float16)              # fp16 on disk, matching reddit/CA


def dry_run(corpus):
    import pandas as pd
    cfg = CORPORA[corpus]
    shards = sorted(glob.glob(f"{cfg['chunks']}/*.parquet"))
    src = shards[0] if shards else None
    fallback = os.environ.get("DRY_PARQUET")
    if src is None:
        if not fallback:
            print(f"[dry-run] no chunk shards at {cfg['chunks']} yet; set "
                  f"DRY_PARQUET=<a parquet with a chunk_text column> to validate "
                  f"the embed path before the full chunk run.", file=sys.stderr)
            return 1
        src = fallback
    print(f"[dry-run] corpus={corpus} device={DEVICE} reading 100 rows from {src}")
    texts = pd.read_parquet(src, columns=["chunk_text"]).head(100)["chunk_text"].tolist()
    model = make_model()
    t0 = time.time()
    vecs = embed_texts(model, texts)
    dt = time.time() - t0
    norms = np.linalg.norm(vecs.astype(np.float32), axis=1)
    print(f"[dry-run] embedded {len(texts)} rows in {dt:.2f}s")
    print(f"[dry-run] shape={vecs.shape} dtype={vecs.dtype} "
          f"norm mean={norms.mean():.4f} min={norms.min():.4f} max={norms.max():.4f}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", choices=list(CORPORA))
    ap.add_argument("--dry-run", action="store_true",
                    help="embed 100 rows, print shape/dtype/norm; no write")
    args = ap.parse_args()

    if args.dry_run:
        return dry_run(args.corpus)

    import pandas as pd
    import torch

    cfg = CORPORA[args.corpus]
    chunks, out = cfg["chunks"], cfg["out"]
    shards = sorted(glob.glob(f"{chunks}/*.parquet"))
    if not shards:
        print(f"no chunk shards at {chunks}", file=sys.stderr)
        return 1

    model = make_model()
    print(f"model {MODEL_ID} device={DEVICE} batch={BATCH} shards={len(shards)} "
          f"cap={MAX_ROWS:,}", flush=True)

    os.makedirs(out, exist_ok=True)
    total_rows = 0
    t_start = time.time()
    for path in shards:
        if total_rows >= MAX_ROWS:
            break
        name = os.path.basename(path).replace(".parquet", ".npy")
        outp = os.path.join(out, name)
        if os.path.exists(outp):
            total_rows += np.load(outp, mmap_mode="r").shape[0]
            print(f"skip {name} (done)", flush=True)
            continue
        df = pd.read_parquet(path, columns=["chunk_text"])
        texts = df["chunk_text"].tolist()
        del df
        remaining = MAX_ROWS - total_rows
        if len(texts) > remaining:
            texts = texts[:remaining]     # truncate the shard that crosses the cap
        t0 = time.time()
        vecs = embed_texts(model, texts)
        assert vecs.shape == (len(texts), DIM), vecs.shape
        tmp = outp + ".tmp.npy"
        np.save(tmp, vecs)
        os.rename(tmp, outp)
        dt = time.time() - t0
        total_rows += len(texts)
        print(f"{name}: {len(texts):,} rows in {dt/60:.1f} min "
              f"({len(texts)/dt:,.0f}/s) | total {total_rows:,}/{MAX_ROWS:,}",
              flush=True)
        del vecs, texts
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    manifest = {
        "model": MODEL_ID,
        "dim": DIM,
        "dtype": "float16",
        "rows": total_rows,
        "device": DEVICE,
        "max_rows": MAX_ROWS,
        "row_parallel_with": chunks,
        "normalized": False,
        "wall_s": time.time() - t_start,
        "note": "rows are row-parallel with the input parquet shards (first "
                "MAX_ROWS chunks); real .npy fp16, matches reddit/CA format.",
    }
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"DONE {total_rows:,} rows in {(time.time()-t_start)/3600:.2f} h",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
