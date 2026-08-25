#!/usr/bin/env python3
"""Chunk the Twitter100M + Bluesky-5M social register corpora into 120-token
wordpiece windows, matching the canonical latent-data-modal chunker EXACTLY
(bert-base-uncased wordpiece, CHUNK_MAX_TOKENS=120, CHUNK_OVERLAP=0.1). CPU ONLY.

Chunker logic + params are copied verbatim from
/data/chunks/reddit-tldr17-chunked-120/chunk_reddit.py so these corpora are
comparable to reddit-2m / communityarchive-2m.

Key differences from chunk_reddit.py:
  - Source is parquet (downloaded by download_social_registers.py), not a JSON zip.
  - The Twitter source is GROUPED BY USER (see download script SORT-CHECK note),
    so rows are sampled RANDOMLY with a seeded RNG (seed 42) across ALL downloaded
    shards before chunking, so the STOP_AFTER_CHUNKS cap does not re-concentrate
    on a narrow user range. Bluesky is a single file, sampled randomly the same way.

Output:
  /data/chunks/twitter100m-chunked-120/train/{shard:03d}_{cumulative:09d}.parquet
  /data/chunks/bluesky-5m-chunked-120/train/{shard:03d}_{cumulative:09d}.parquet
with a `chunk_text` column (+ chunk_index, chunk_token_count, id + provenance).

Write-once: refuses to run if the target train/ dir already has *.parquet
(pass --force to override).

Validate (dry run, no full chunking):
  HF_HOME=/data/hf .venv/bin/python chunk_social_registers.py twitter --dry-run
  HF_HOME=/data/hf .venv/bin/python chunk_social_registers.py bluesky --dry-run

Full run (long — launch via systemd, see README/report):
  HF_HOME=/data/hf CUDA_VISIBLE_DEVICES= .venv/bin/python chunk_social_registers.py twitter
  HF_HOME=/data/hf CUDA_VISIBLE_DEVICES= .venv/bin/python chunk_social_registers.py bluesky
"""
import argparse
import glob
import os
import sys
import time

os.environ["HF_HOME"] = "/data/hf"
os.environ["CUDA_VISIBLE_DEVICES"] = ""            # hard CPU guard
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import transformers
transformers.logging.set_verbosity_error()
from transformers import AutoTokenizer

# ---- canonical chunking params (match chunk_reddit.py / config.py) ----
CHUNK_MAX_TOKENS = 120
CHUNK_OVERLAP = 0.1

SHARD_SIZE = 500_000
STOP_AFTER_CHUNKS = 2_500_000          # headroom over the 2M embed sample
SEED = 42

TW_DIR = ("/data/hf/hub/datasets--enryu43--twitter100m_tweets/snapshots/"
          "5d742bec6f777adf262006017cd3b67e985b0874/data")
BS_DIR = ("/data/hf/hub/datasets--Roronotalt--bluesky-five-million/snapshots/"
          "608a2e5fa2968efba57133c50eb09c20a1fba268/data")

CORPORA = {
    "twitter": {
        "shards": [
            f"{TW_DIR}/train-00000-of-00041-3f49db2da17edd5a.parquet",
            f"{TW_DIR}/train-00010-of-00041-39a461f69a92fa95.parquet",
            f"{TW_DIR}/train-00020-of-00041-cb49f7a05f4c5137.parquet",
            f"{TW_DIR}/train-00030-of-00041-9712df7d5c3d07f0.parquet",
            f"{TW_DIR}/train-00040-of-00041-9a723429a2a70e30.parquet",
        ],
        "text_key": "tweet",
        "keep_keys": ["id", "user", "date"],
        "out_dir": "/data/chunks/twitter100m-chunked-120/train",
    },
    "bluesky": {
        "shards": [f"{BS_DIR}/train-0000.parquet"],
        "text_key": "text",
        "keep_keys": ["uri", "author", "created_at", "langs"],
        "out_dir": "/data/chunks/bluesky-5m-chunked-120/train",
    },
}

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased", model_max_length=CHUNK_MAX_TOKENS
)


def chunk_row(row, text_key, keep_keys):
    """Verbatim from chunk_reddit.py / latent-data-modal chunker.py."""
    text = row[text_key]
    tokens = tokenizer.encode(text)
    token_count = len(tokens)

    if token_count <= CHUNK_MAX_TOKENS:
        return [{
            "chunk_index": 0,
            "chunk_text": text,
            "chunk_token_count": token_count,
            **{key: row.get(key) for key in keep_keys},
        }]

    overlap = int(CHUNK_MAX_TOKENS * CHUNK_OVERLAP)
    stride = CHUNK_MAX_TOKENS - overlap
    chunks = []
    ci = 0
    start = 0

    while start < len(tokens):
        end = min(start + CHUNK_MAX_TOKENS, len(tokens))
        chunk_tokens = tokens[start:end]

        if len(chunk_tokens) < overlap and ci > 0:
            break

        chunks.append({
            "chunk_index": ci,
            "chunk_text": tokenizer.decode(chunk_tokens),
            "chunk_token_count": len(chunk_tokens),
            **{key: row.get(key) for key in keep_keys},
        })
        start += stride
        ci += 1

    return chunks


def build_schema(keep_keys):
    fields = [
        ("chunk_index", pa.int64()),
        ("chunk_text", pa.large_string()),
        ("chunk_token_count", pa.int64()),
    ]
    # All provenance stored as string for a stable schema across shards.
    fields += [(k, pa.large_string()) for k in keep_keys]
    return pa.schema(fields)


def write_shard(buf, shard_idx, cumulative, keep_keys, schema, out_dir):
    cols = ["chunk_index", "chunk_text", "chunk_token_count", *keep_keys]
    df = pd.DataFrame(buf, columns=cols)
    # Coerce provenance to string to match the large_string schema.
    for k in keep_keys:
        df[k] = df[k].map(lambda v: "" if v is None else str(v))
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    path = f"{out_dir}/{shard_idx:03d}_{cumulative:09d}.parquet"
    pq.write_table(table, path)
    return path


def iter_rows_shuffled(shards, text_key, keep_keys, rng, dry_rows=0):
    """Yield row dicts in a globally-shuffled (seed 42) order across all shards.

    Reads only the needed columns into memory (text + provenance), concatenates,
    then applies a single seeded permutation so the STOP_AFTER_CHUNKS cap draws
    from the whole downloaded sample rather than a contiguous user range.
    """
    cols = list(dict.fromkeys([text_key, *keep_keys]))
    frames = []
    for sp in shards:
        avail = pq.ParquetFile(sp).schema_arrow.names
        use = [c for c in cols if c in avail]
        if dry_rows:
            df = pq.ParquetFile(sp).read_row_group(0, columns=use).to_pandas()
            df = df.head(dry_rows)
        else:
            df = pd.read_parquet(sp, columns=use)
        for c in cols:
            if c not in df.columns:
                df[c] = None
        frames.append(df)
    big = pd.concat(frames, ignore_index=True)
    del frames
    if dry_rows:
        big = big.head(dry_rows)          # dry run: first rows, no shuffle needed
        order = np.arange(len(big))
    else:
        order = rng.permutation(len(big))
    for i in order:
        yield big.iloc[i].to_dict()


def run(corpus, dry_run=False, force=False):
    cfg = CORPORA[corpus]
    text_key, keep_keys, out_dir = cfg["text_key"], cfg["keep_keys"], cfg["out_dir"]
    schema = build_schema(keep_keys)
    rng = np.random.default_rng(SEED)

    if dry_run:
        print(f"[dry-run] corpus={corpus} text_key={text_key!r} "
              f"keep_keys={keep_keys}")
        gen = iter_rows_shuffled(cfg["shards"], text_key, keep_keys, rng,
                                 dry_rows=1000)
        n_rows, n_chunks, samples = 0, 0, []
        for row in gen:
            if not row.get(text_key):
                continue
            n_rows += 1
            ch = chunk_row(row, text_key, keep_keys)
            n_chunks += len(ch)
            if len(samples) < 3:
                samples.append(ch[0])
        print(f"[dry-run] read {n_rows} non-empty rows -> {n_chunks} chunks "
              f"({n_chunks / max(n_rows, 1):.2f} chunks/row)")
        for i, s in enumerate(samples):
            print(f"\n--- sample chunk {i} (tok={s['chunk_token_count']}) ---")
            print(repr(s["chunk_text"])[:300])
        return 0

    os.makedirs(out_dir, exist_ok=True)
    existing = glob.glob(f"{out_dir}/*.parquet")
    if existing and not force:
        print(f"ABORT: {out_dir} already has {len(existing)} parquet(s). "
              f"Write-once; pass --force to overwrite.", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    buf, shard_idx, cumulative, rows_read, capped = [], 0, 0, 0, False
    gen = iter_rows_shuffled(cfg["shards"], text_key, keep_keys, rng)
    for row in gen:
        if not row.get(text_key):
            continue
        rows_read += 1
        buf.extend(chunk_row(row, text_key, keep_keys))
        while len(buf) >= SHARD_SIZE:
            shard, buf = buf[:SHARD_SIZE], buf[SHARD_SIZE:]
            cumulative += len(shard)
            write_shard(shard, shard_idx, cumulative, keep_keys, schema, out_dir)
            el = time.perf_counter() - t0
            print(f"[{el:8.1f}s] wrote shard {shard_idx:03d} -> {cumulative:,} "
                  f"chunks | rows read {rows_read:,} | {rows_read/el:,.0f} rows/s",
                  flush=True)
            shard_idx += 1
            if cumulative + len(buf) >= STOP_AFTER_CHUNKS:
                capped = True
                break
        if capped:
            break

    if not capped and buf:
        if cumulative + len(buf) > STOP_AFTER_CHUNKS:
            buf = buf[: STOP_AFTER_CHUNKS - cumulative]
            capped = True
        cumulative += len(buf)
        write_shard(buf, shard_idx, cumulative, keep_keys, schema, out_dir)
        shard_idx += 1
        print(f"wrote final shard {shard_idx-1:03d} -> {cumulative:,} chunks",
              flush=True)

    el = time.perf_counter() - t0
    print(f"DONE in {el:.1f}s | shards={shard_idx} | total_chunks={cumulative:,} "
          f"| rows_read={rows_read:,} | capped={capped}", flush=True)

    notes = (
        f"# {corpus} social-register chunk build notes\n\n"
        f"- Source shards: {cfg['shards']}\n"
        f"- Field chunked = `{text_key}`; keep-keys = {keep_keys}.\n"
        f"- Chunker: canonical latent-data-modal logic (bert-base-uncased "
        f"wordpiece, CHUNK_MAX_TOKENS=120, CHUNK_OVERLAP=0.1; overlap=12, stride=108).\n"
        f"- Rows sampled RANDOMLY across all downloaded shards (numpy seed {SEED}) "
        f"before chunking, to avoid re-concentrating on the user-grouped source.\n"
        f"- Shards: {shard_idx} parquet x up to {SHARD_SIZE:,} chunks "
        f"(naming {{shard:03d}}_{{cumulative:09d}}.parquet).\n"
        f"- Total chunks: {cumulative:,}. STOP_AFTER_CHUNKS={STOP_AFTER_CHUNKS:,}. "
        f"capped={capped}. rows_read={rows_read:,}.\n"
    )
    with open(os.path.join(os.path.dirname(out_dir), "NOTES.md"), "w") as f:
        f.write(notes)
    print("wrote NOTES.md", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", choices=list(CORPORA))
    ap.add_argument("--dry-run", action="store_true",
                    help="chunk first ~1000 rows, print 3 samples + counts; no write")
    ap.add_argument("--force", action="store_true", help="overwrite existing output")
    args = ap.parse_args()
    return run(args.corpus, dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
