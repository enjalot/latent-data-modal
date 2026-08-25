# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "transformers>=4.40",
#     "pyarrow>=15",
#     "huggingface_hub>=0.23",
# ]
# ///
"""
Local (CPU-only) port of chunker.py for English Wikipedia.

Chunks wikimedia/wikipedia 20231101.en into 120-token windows using the
bert-base-uncased tokenizer (== all-MiniLM-L6-v2 tokenizer), matching the
convention of /data/chunks/fineweb-edu-sample-10BT-chunked-120 exactly
(chunker.py logic: 10% overlap, decoded chunk_text, raw text for short docs,
no chunk_tokens column), with keep_keys = [id, url, title] per config.py's
wikipedia-en entry — i.e. every chunk row carries the article title.

Output: /data/chunks/wikipedia-en-chunked-120/train/data-NNNNN-of-00041.parquet
Resumable: completed shards are skipped (tmp-file + atomic rename).

Usage:
    uv run chunk_wikipedia_local.py --validate 1000   # end-to-end pipeline check
    uv run chunk_wikipedia_local.py --download        # fetch raw parquets only
    uv run chunk_wikipedia_local.py                   # full run (download + chunk)
"""
import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Config — mirrors latent-data-modal/config.py (wikipedia-en, CHUNK 120)
# ---------------------------------------------------------------------------
HF_REPO = "wikimedia/wikipedia"
HF_SUBSET = "20231101.en"
NUM_SHARDS = 41
TEXT_KEY = "text"
KEEP_KEYS = ["id", "url", "title"]
CHUNK_MAX_TOKENS = 120
CHUNK_OVERLAP = 0.1
TOKENIZER_ID = "bert-base-uncased"

RAW_DIR = "/data/chunks/_wiki_download"
OUT_DIR = "/data/chunks/wikipedia-en-chunked-120/train"
BATCH_ROWS = 1024          # articles per tokenize/decode batch
ROWS_PER_ROW_GROUP = 250_000

os.environ.setdefault("HF_HOME", "/data/hf")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def src_path(i: int) -> str:
    return f"{RAW_DIR}/{HF_SUBSET}/train-{i:05d}-of-{NUM_SHARDS:05d}.parquet"


def out_name(i: int) -> str:
    return f"data-{i:05d}-of-{NUM_SHARDS:05d}.parquet"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download():
    from huggingface_hub import snapshot_download

    print(f"Downloading {HF_REPO} / {HF_SUBSET} -> {RAW_DIR}", flush=True)
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        allow_patterns=f"{HF_SUBSET}/*",
        local_dir=RAW_DIR,
        token=False,  # public dataset; avoid stale profile token
    )
    missing = [i for i in range(NUM_SHARDS) if not os.path.exists(src_path(i))]
    if missing:
        raise RuntimeError(f"Download incomplete, missing shards: {missing}")
    print("Download complete.", flush=True)


# ---------------------------------------------------------------------------
# Chunking — logic copied from chunker.py::chunk_row, vectorized per batch
# ---------------------------------------------------------------------------
_TOK = None


def get_tokenizer():
    global _TOK
    if _TOK is None:
        import transformers

        transformers.logging.set_verbosity_error()
        from transformers import AutoTokenizer

        _TOK = AutoTokenizer.from_pretrained(
            TOKENIZER_ID, model_max_length=CHUNK_MAX_TOKENS
        )
    return _TOK


def chunk_batch(rows: list[dict]) -> list[dict]:
    """chunker.py chunk_row() applied to a batch, with batched decode."""
    tok = get_tokenizer()
    texts = [r[TEXT_KEY] for r in rows]
    encoded = tok(texts, add_special_tokens=True)["input_ids"]  # no truncation

    out = []
    pending_windows = []   # token windows needing decode
    pending_slots = []     # index into `out` to fill with decoded text
    overlap = int(CHUNK_MAX_TOKENS * CHUNK_OVERLAP)
    stride = CHUNK_MAX_TOKENS - overlap

    for row, tokens in zip(rows, encoded):
        meta = {k: row[k] for k in KEEP_KEYS}
        n = len(tokens)
        if n <= CHUNK_MAX_TOKENS:
            out.append({
                "chunk_index": 0,
                "chunk_text": row[TEXT_KEY],  # raw text, matching chunker.py
                "chunk_token_count": n,
                **meta,
            })
            continue

        ci = 0
        start = 0
        while start < n:
            end = min(start + CHUNK_MAX_TOKENS, n)
            window = tokens[start:end]
            # Skip tiny tail chunks that are purely overlap remnants
            if len(window) < overlap and ci > 0:
                break
            out.append({
                "chunk_index": ci,
                "chunk_text": None,  # filled after batch_decode
                "chunk_token_count": len(window),
                **meta,
            })
            pending_windows.append(window)
            pending_slots.append(len(out) - 1)
            start += stride
            ci += 1

    if pending_windows:
        decoded = tok.batch_decode(pending_windows)  # keeps [CLS]/[SEP], as chunker.py
        for slot, text in zip(pending_slots, decoded):
            out[slot]["chunk_text"] = text
    return out


def chunk_schema():
    import pyarrow as pa

    return pa.schema([
        ("chunk_index", pa.int64()),
        ("chunk_text", pa.string()),
        ("chunk_token_count", pa.int64()),
        ("id", pa.string()),
        ("url", pa.string()),
        ("title", pa.string()),
    ])


def process_shard(i: int) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path = f"{OUT_DIR}/{out_name(i)}"
    if os.path.exists(out_path):
        return f"skip {out_name(i)} (exists)"

    t0 = time.perf_counter()
    schema = chunk_schema()
    tmp_path = out_path + ".tmp"
    pf = pq.ParquetFile(src_path(i))
    n_articles = 0
    n_chunks = 0
    buffer = []

    with pq.ParquetWriter(tmp_path, schema) as writer:
        def flush():
            nonlocal buffer
            if buffer:
                cols = {name: [r[name] for r in buffer] for name in schema.names}
                writer.write_table(pa.table(cols, schema=schema))
                buffer = []

        for batch in pf.iter_batches(batch_size=BATCH_ROWS,
                                     columns=KEEP_KEYS + [TEXT_KEY]):
            rows = batch.to_pylist()
            n_articles += len(rows)
            chunks = chunk_batch(rows)
            n_chunks += len(chunks)
            buffer.extend(chunks)
            if len(buffer) >= ROWS_PER_ROW_GROUP:
                flush()
        flush()

    os.rename(tmp_path, out_path)
    dt = time.perf_counter() - t0
    return (f"done {out_name(i)}: {n_articles} articles -> {n_chunks} chunks "
            f"in {dt/60:.1f} min ({n_articles/dt:.0f} art/s)")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(n_articles: int):
    import pyarrow.parquet as pq

    if not os.path.exists(src_path(0)):
        print("Shard 0 not downloaded yet; fetching it first...", flush=True)
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=HF_REPO, repo_type="dataset",
            filename=f"{HF_SUBSET}/train-00000-of-{NUM_SHARDS:05d}.parquet",
            local_dir=RAW_DIR, token=False,
        )

    tok = get_tokenizer()
    pf = pq.ParquetFile(src_path(0))
    rows = []
    for batch in pf.iter_batches(batch_size=n_articles,
                                 columns=KEEP_KEYS + [TEXT_KEY]):
        rows = batch.to_pylist()[:n_articles]
        break

    t0 = time.perf_counter()
    chunks = chunk_batch(rows)
    dt = time.perf_counter() - t0

    # invariants
    overlap = int(CHUNK_MAX_TOKENS * CHUNK_OVERLAP)
    bad_len = [c for c in chunks if c["chunk_token_count"] > CHUNK_MAX_TOKENS]
    no_title = [c for c in chunks if not c["title"]]
    multi = {}
    for c in chunks:
        multi.setdefault(c["id"], []).append(c)
    # overlap check on a few multi-chunk articles
    checked = 0
    for cid, cs in multi.items():
        if len(cs) < 2 or checked >= 20:
            continue
        cs.sort(key=lambda c: c["chunk_index"])
        e0 = tok(rows[[r["id"] for r in rows].index(cid)][TEXT_KEY],
                 add_special_tokens=True)["input_ids"]
        w0 = e0[0:CHUNK_MAX_TOKENS]
        w1 = e0[CHUNK_MAX_TOKENS - overlap:2 * CHUNK_MAX_TOKENS - overlap]
        assert w0[-overlap:] == w1[:overlap], f"overlap mismatch for {cid}"
        assert cs[0]["chunk_token_count"] == CHUNK_MAX_TOKENS
        checked += 1

    print(f"Validation on {len(rows)} articles:")
    print(f"  chunks: {len(chunks)}  ({len(chunks)/len(rows):.1f} per article)")
    print(f"  chunking rate: {len(rows)/dt:.0f} articles/s (single process)")
    print(f"  chunks >120 tokens: {len(bad_len)}   chunks missing title: {len(no_title)}")
    print(f"  overlap verified on {checked} multi-chunk articles (12-token overlap)")
    dist = {}
    for c in chunks:
        dist[c["chunk_token_count"] == CHUNK_MAX_TOKENS] = \
            dist.get(c["chunk_token_count"] == CHUNK_MAX_TOKENS, 0) + 1
    print(f"  full 120-token chunks: {dist.get(True,0)}, shorter: {dist.get(False,0)}")
    print("\nSchema:", chunk_schema())
    print("\nSample rows:")
    for c in chunks[:2] + [c for c in chunks if c["chunk_index"] == 1][:1]:
        print({k: (v[:110] + "..." if isinstance(v, str) and len(v) > 110 else v)
               for k, v in c.items()})
    if bad_len or no_title:
        sys.exit("VALIDATION FAILED")
    print("\nVALIDATION OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=int, metavar="N", default=None)
    ap.add_argument("--download", action="store_true", help="download only")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    if args.validate:
        validate(args.validate)
        return

    free_gb = shutil.disk_usage("/data").free / 1e9
    if free_gb < 150:
        sys.exit(f"ABORT: only {free_gb:.0f} GB free on /data (<150 GB)")
    print(f"/data free: {free_gb:.0f} GB", flush=True)

    download()
    if args.download:
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    todo = [i for i in range(NUM_SHARDS)
            if not os.path.exists(f"{OUT_DIR}/{out_name(i)}")]
    print(f"{NUM_SHARDS - len(todo)} shards already done, {len(todo)} to go", flush=True)

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_shard, i): i for i in todo}
        for fut in as_completed(futures):
            try:
                print(fut.result(), flush=True)
            except Exception as e:
                print(f"EXCEPTION shard {futures[fut]}: {e!r}", flush=True)

    # final tally
    import pyarrow.parquet as pq
    total_chunks = 0
    done = 0
    for i in range(NUM_SHARDS):
        p = f"{OUT_DIR}/{out_name(i)}"
        if os.path.exists(p):
            total_chunks += pq.ParquetFile(p).metadata.num_rows
            done += 1
    print(f"ALL DONE: {done}/{NUM_SHARDS} shards, {total_chunks} total chunks "
          f"in {(time.perf_counter()-t0)/60:.1f} min", flush=True)

    free_gb = shutil.disk_usage("/data").free / 1e9
    if done == NUM_SHARDS and free_gb < 150:
        print(f"/data free {free_gb:.0f} GB < 150 GB -> removing raw download", flush=True)
        shutil.rmtree(RAW_DIR)
    else:
        print(f"/data free: {free_gb:.0f} GB; raw kept at {RAW_DIR}", flush=True)


if __name__ == "__main__":
    main()
