"""
Chunk a downloaded HuggingFace dataset into fixed-size token windows.

Fixes from the original:
- No longer creates tiny (<overlap) tail chunks that are pure overlap noise
- Drops the `chunk_tokens` column (only text + count needed; saves ~60% parquet size)
- Uses shared config.py for all dataset parameters

Usage:
    modal run chunker.py
"""
from modal import App, Image, Volume

from config import (
    get_dataset, shard_files,
    CHUNK_MAX_TOKENS, CHUNK_OVERLAP,
    chunked_dataset_name,
)

ds = get_dataset()

NUM_CPU = 4
BATCH_SIZE = 200  # rows per thread batch

DATASET_DIR = "/data"
DATASET_SAVE = ds.save_name
DATASET_SAVE_CHUNKED = chunked_dataset_name()
TEXT_KEY = ds.text_key
KEEP_KEYS = ds.keep_keys
files = shard_files(ext="arrow")

volume = Volume.from_name(ds.volume, create_if_missing=True)
image = Image.debian_slim(python_version="3.10").pip_install(
    "datasets", "transformers", "pandas", "tqdm"
)
app = App(image=image)


def chunk_row(row, tokenizer):
    """Split one document into overlapping token-window chunks.

    Returns a list of dicts, each with chunk_text, chunk_token_count,
    chunk_index, and the KEEP_KEYS metadata columns.
    """
    text = row[TEXT_KEY]
    tokens = tokenizer.encode(text)
    token_count = len(tokens)

    if token_count <= CHUNK_MAX_TOKENS:
        return [{
            "chunk_index": 0,
            "chunk_text": text,
            "chunk_token_count": token_count,
            **{key: row[key] for key in KEEP_KEYS},
        }]

    overlap = int(CHUNK_MAX_TOKENS * CHUNK_OVERLAP)
    stride = CHUNK_MAX_TOKENS - overlap
    chunks = []
    ci = 0
    start = 0

    while start < len(tokens):
        end = min(start + CHUNK_MAX_TOKENS, len(tokens))
        chunk_tokens = tokens[start:end]

        # Skip tiny tail chunks that are purely overlap remnants
        if len(chunk_tokens) < overlap and ci > 0:
            break

        chunks.append({
            "chunk_index": ci,
            "chunk_text": tokenizer.decode(chunk_tokens),
            "chunk_token_count": len(chunk_tokens),
            **{key: row[key] for key in KEEP_KEYS},
        })
        start += stride
        ci += 1

    return chunks


@app.function(cpu=NUM_CPU, volumes={DATASET_DIR: volume}, timeout=3000)
def process_dataset(file):
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import pandas as pd
    import transformers
    from datasets import load_dataset
    from tqdm import tqdm

    transformers.logging.set_verbosity_error()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased", model_max_length=CHUNK_MAX_TOKENS
    )

    start = time.perf_counter()
    print(f"Loading {DATASET_DIR}/{DATASET_SAVE}/train/{file}")
    dataset = load_dataset("arrow", data_files=f"{DATASET_DIR}/{DATASET_SAVE}/train/{file}")
    df = pd.DataFrame(dataset["train"])
    print(f"  {len(df)} rows loaded in {time.perf_counter() - start:.1f}s")

    chunks_list = []

    def process_batch(batch):
        batch_chunks = []
        for row in batch:
            batch_chunks.extend(chunk_row(row, tokenizer))
        return batch_chunks

    batches = [
        df.iloc[i : i + BATCH_SIZE].to_dict(orient="records")
        for i in range(0, len(df), BATCH_SIZE)
    ]

    pbar = tqdm(total=len(batches), desc=f"Chunking {file}")
    with ThreadPoolExecutor(max_workers=NUM_CPU) as executor:
        futures = [executor.submit(process_batch, batch) for batch in batches]
        for future in as_completed(futures):
            chunks_list.extend(future.result())
            pbar.update(1)
    pbar.close()

    chunked_df = pd.DataFrame(chunks_list)
    output_dir = f"{DATASET_DIR}/{DATASET_SAVE_CHUNKED}/train"
    os.makedirs(output_dir, exist_ok=True)

    file_name = file.split(".")[0]
    out_path = f"{output_dir}/{file_name}.parquet"
    chunked_df.to_parquet(out_path)
    print(f"  {len(chunks_list)} chunks → {out_path}")

    volume.commit()
    return f"Done: {file} — {len(chunks_list)} chunks"


@app.local_entrypoint()
def main():
    for resp in process_dataset.map(files, order_outputs=False, return_exceptions=True):
        if isinstance(resp, Exception):
            print(f"EXCEPTION: {resp}")
        else:
            print(resp)
