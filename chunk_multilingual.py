"""
Chunk FineWeb-2 multilingual data into 500-token windows.
CPU-only — no GPU needed. Caps at 2M chunks per language.

Uses bert-base-multilingual-cased tokenizer for proper multilingual support.

Usage:
    modal run chunk_multilingual.py            # all 20 languages
    modal run chunk_multilingual.py --lang fra_Latn  # single language
"""
from modal import App, Image, Volume

DATASET_DIR = "/data"
CHUNK_MAX_TOKENS = 500
CHUNK_OVERLAP = 0.1
MAX_CHUNKS = 2_000_000
NUM_CPU = 8
BATCH_SIZE = 1000  # rows per thread batch

LANGUAGES = [
    "fra_Latn", "deu_Latn", "spa_Latn", "ita_Latn", "por_Latn",
    "nld_Latn", "pol_Latn", "rus_Cyrl", "cmn_Hani", "jpn_Jpan",
    "kor_Hang", "arb_Arab", "hin_Deva", "tur_Latn", "vie_Latn",
    "tha_Thai", "ind_Latn", "swe_Latn", "ces_Latn", "ell_Grek",
]

volume = Volume.from_name("datasets", create_if_missing=True)
image = Image.debian_slim(python_version="3.11").pip_install(
    "transformers", "pandas", "pyarrow", "tqdm"
)
app = App(image=image)


def chunk_row(text, tokenizer, tokenizer_encode):
    """Split one document into overlapping token-window chunks.
    Returns list of (chunk_text, chunk_token_count) tuples.
    """
    if not text or not text.strip():
        return []

    tokens = tokenizer_encode(text)
    n = len(tokens)

    if n <= CHUNK_MAX_TOKENS:
        return [(text, n)]

    overlap = int(CHUNK_MAX_TOKENS * CHUNK_OVERLAP)
    stride = CHUNK_MAX_TOKENS - overlap
    chunks = []
    start = 0

    while start < n:
        end = min(start + CHUNK_MAX_TOKENS, n)
        chunk_tokens = tokens[start:end]
        if len(chunk_tokens) < overlap and start > 0:
            break
        chunks.append((tokenizer.decode(chunk_tokens), len(chunk_tokens)))
        start += stride

    return chunks


@app.function(cpu=NUM_CPU, memory=32768, volumes={DATASET_DIR: volume}, timeout=7200)
def chunk_language(lang_code: str):
    """Chunk one language's parquet shard. Returns stats dict."""
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import pandas as pd
    import transformers
    from tqdm import tqdm

    transformers.logging.set_verbosity_error()
    from transformers import AutoTokenizer

    save_name = f"fineweb2-{lang_code}"
    parquet_path = f"{DATASET_DIR}/{save_name}/train/000_00000.parquet"
    out_dir = f"{DATASET_DIR}/{save_name}-chunked-{CHUNK_MAX_TOKENS}/train"

    # Check if already chunked
    out_path = f"{out_dir}/000_00000.parquet"
    if os.path.exists(out_path):
        existing = pd.read_parquet(out_path)
        n = len(existing)
        print(f"  {lang_code}: already chunked ({n:,} chunks)")
        return {"lang": lang_code, "n_chunks": n, "status": "cached"}

    print(f"\n{'='*60}")
    print(f"  Chunking {lang_code} @ {CHUNK_MAX_TOKENS} tokens (cap {MAX_CHUNKS:,})")
    print(f"{'='*60}")

    # Load multilingual tokenizer (use_fast=True for Rust-backed speed)
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-multilingual-cased", model_max_length=CHUNK_MAX_TOKENS, use_fast=True
    )
    # Disable special tokens for raw encode (faster)
    tokenizer_encode = lambda text: tokenizer.encode(text, add_special_tokens=False)

    # Load raw data
    t0 = time.perf_counter()
    df = pd.read_parquet(parquet_path, columns=["text"])
    n_docs = len(df)
    print(f"  {n_docs:,} documents loaded in {time.perf_counter()-t0:.1f}s")

    # Chunk with thread pool
    all_chunks = []
    done = False

    def process_batch(batch_texts):
        batch_chunks = []
        for text in batch_texts:
            batch_chunks.extend(chunk_row(text, tokenizer, tokenizer_encode))
        return batch_chunks

    texts = df["text"].tolist()
    del df

    batches = [texts[i:i+BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    del texts

    pbar = tqdm(total=len(batches), desc=f"Chunking {lang_code}")
    with ThreadPoolExecutor(max_workers=NUM_CPU) as executor:
        futures = {executor.submit(process_batch, b): i for i, b in enumerate(batches)}
        for future in as_completed(futures):
            result = future.result()
            all_chunks.extend(result)
            pbar.update(1)
            if len(all_chunks) >= MAX_CHUNKS:
                # Cancel remaining futures
                for f in futures:
                    f.cancel()
                break
    pbar.close()

    # Cap
    all_chunks = all_chunks[:MAX_CHUNKS]
    n_chunks = len(all_chunks)

    # Build dataframe
    chunk_df = pd.DataFrame({
        "chunk_text": [c[0] for c in all_chunks],
        "chunk_token_count": [c[1] for c in all_chunks],
        "chunk_index": range(n_chunks),
    })
    del all_chunks

    os.makedirs(out_dir, exist_ok=True)
    chunk_df.to_parquet(out_path)
    volume.commit()

    size_mb = os.path.getsize(out_path) / 1e6
    elapsed = time.perf_counter() - t0
    print(f"  {n_chunks:,} chunks → {out_path} ({size_mb:.0f} MB, {elapsed:.0f}s)")

    return {
        "lang": lang_code, "n_docs": n_docs, "n_chunks": n_chunks,
        "size_mb": round(size_mb), "seconds": round(elapsed), "status": "chunked",
    }


@app.local_entrypoint()
def main(lang: str = "all"):
    import json

    targets = LANGUAGES if lang == "all" else [lang]
    print(f"Chunking {len(targets)} languages at {CHUNK_MAX_TOKENS} tokens (cap {MAX_CHUNKS:,})")

    results = []
    for r in chunk_language.map(targets, order_outputs=False, return_exceptions=True):
        if isinstance(r, Exception):
            print(f"  EXCEPTION: {r}")
            results.append({"status": "failed", "error": str(r)})
        else:
            results.append(r)
            print(f"  {r['lang']}: {r.get('n_chunks', '?'):,} chunks ({r.get('status')})")

    ok = [r for r in results if r.get("status") in ("chunked", "cached")]
    total_chunks = sum(r.get("n_chunks", 0) for r in ok)
    total_mb = sum(r.get("size_mb", 0) for r in ok)
    print(f"\n  {len(ok)}/{len(results)} done, {total_chunks:,} total chunks, {total_mb:,} MB")
