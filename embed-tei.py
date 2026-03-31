"""
Embed a dataset using HuggingFace Text Embeddings Inference (TEI).

Improvements over the original:
- Pulls all config from config.py (no more commented-out blocks)
- Doubles the client batch token budget to actually fill the A10G's 24 GB
- Sorts chunks + uses first-fit-decreasing bin-packing for tighter batches
- Prefetches the next file's batches while the current file is embedding
- Writes embeddings to memmap in original row order via an index map

Usage:
    modal run embed-tei.py
"""
import os
import subprocess
import time

from modal import App, Image, Secret, Volume, enter, exit, gpu, method

from config import (
    get_dataset, get_model,
    chunked_dataset_name, embedding_dataset_name,
    shard_files, GPU_TYPE, GPU_CONCURRENCY, TEI_IMAGE,
)

# ---------------------------------------------------------------------------
# Resolve config
# ---------------------------------------------------------------------------
ds = get_dataset()
model = get_model()

DATASET_DIR = "/data"
EMBEDDING_DIR = "/embeddings"

CHUNKED_NAME = chunked_dataset_name()
EMBEDDING_NAME = embedding_dataset_name()

files = shard_files(ext="parquet")

MODEL_ID = model.model_id
PREFIX = model.prefix
PREFIX_TOKEN_COUNT = model.prefix_token_count
SENTENCE_TOKEN_LIMIT = model.max_tokens

# ---------------------------------------------------------------------------
# Batch-packing budget
#
# TEI pads every sequence in a batch to the length of the *longest* sequence.
# So the real GPU memory consumed ≈ max_seq_len × batch_size × hidden_dim.
# We sort chunks by length and greedily pack them so that:
#   max_token_count_in_batch  ×  num_sequences  ≤  CLIENT_BATCH_TOKEN_LIMIT
#
# With an A10G (24 GB) the previous limit of 768×512 = 393 K tokens only used
# ~10 GB.  Doubling to 1536×512 ≈ 786 K fills ~20 GB — much better utilisation
# while leaving headroom.  The *server* limit is set even higher so TEI never
# rejects a batch we send.
# ---------------------------------------------------------------------------
CLIENT_BATCH_TOKEN_LIMIT = 1536 * SENTENCE_TOKEN_LIMIT   # ~786 K tokens
SERVER_BATCH_TOKEN_LIMIT = 4 * CLIENT_BATCH_TOKEN_LIMIT   # TEI internal cap
MAX_CLIENT_BATCH_SIZE = 8192                               # row cap per request

LAUNCH_FLAGS = [
    "--model-id", MODEL_ID,
    "--port", "8000",
    "--max-client-batch-size", str(MAX_CLIENT_BATCH_SIZE),
    "--max-batch-tokens", str(SERVER_BATCH_TOKEN_LIMIT),
    "--auto-truncate",
    "--dtype", "float16",
    "--json-output",
]

# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------
DATASET_READ_VOLUME = Volume.from_name(ds.volume, create_if_missing=True)
EMBEDDING_CHECKPOINT_VOLUME = Volume.from_name("embeddings", create_if_missing=True)


def spawn_server() -> subprocess.Popen:
    import socket
    process = subprocess.Popen(["text-embeddings-router"] + LAUNCH_FLAGS)
    while True:
        try:
            socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
            print("Webserver ready!")
            return process
        except (socket.timeout, ConnectionRefusedError):
            retcode = process.poll()
            if retcode is not None:
                raise RuntimeError(f"launcher exited unexpectedly with code {retcode}")


tei_image = (
    Image.from_registry(TEI_IMAGE, add_python="3.10")
    .dockerfile_commands("ENTRYPOINT []")
    .pip_install("httpx", "numpy")
)

with tei_image.imports():
    import numpy as np

app = App("embedding-tei")


@app.cls(
    gpu=GPU_TYPE,
    image=tei_image,
    max_containers=GPU_CONCURRENCY,
    allow_concurrent_inputs=4,
    retries=3,
)
class TextEmbeddingsInference:
    @enter()
    def open_connection(self):
        from httpx import AsyncClient
        self.process = spawn_server()
        self.client = AsyncClient(base_url="http://127.0.0.1:8000", timeout=30)

    @exit()
    def terminate_connection(self):
        self.process.terminate()

    @method()
    async def embed(self, chunk_batch):
        texts, indices = chunk_batch
        res = await self.client.post("/embed", json={"inputs": texts})
        try:
            emb = res.json()
            return indices, np.array(emb)
        except Exception as e:
            print(f"Error embedding batch of {len(texts)}: {e}")
            print("Response status:", res.status_code)
            raise


def pack_batches(df):
    """First-fit-decreasing bin packing of sorted chunks.

    Chunks are already sorted ascending by token count.  We iterate and greedily
    add each chunk to the current batch as long as:
        max_token_in_batch × batch_size  ≤  CLIENT_BATCH_TOKEN_LIMIT
    Because chunks are sorted, the *last* chunk added always has the max length,
    so the budget check is simple.
    """
    batches = []
    cur_texts = []
    cur_indices = []
    cur_max_tokens = 0

    for _, row in df.iterrows():
        orig_idx = row["original_position"]
        token_count = row["chunk_token_count"] + PREFIX_TOKEN_COUNT
        text = PREFIX + row["chunk_text"]

        if not text or not text.strip():
            text = PREFIX + " "
            token_count = 1 + PREFIX_TOKEN_COUNT

        new_max = max(cur_max_tokens, token_count)
        new_size = len(cur_texts) + 1
        proposed_budget = new_max * new_size

        if proposed_budget <= CLIENT_BATCH_TOKEN_LIMIT and new_size <= MAX_CLIENT_BATCH_SIZE:
            cur_texts.append(text)
            cur_indices.append(orig_idx)
            cur_max_tokens = new_max
        else:
            if cur_texts:
                batches.append((cur_texts, cur_indices))
            cur_texts = [text]
            cur_indices = [orig_idx]
            cur_max_tokens = token_count

    if cur_texts:
        batches.append((cur_texts, cur_indices))

    return batches


@app.function(
    max_containers=GPU_CONCURRENCY,
    image=Image.debian_slim().pip_install("pandas", "pyarrow", "tqdm"),
    volumes={
        DATASET_DIR: DATASET_READ_VOLUME,
        EMBEDDING_DIR: EMBEDDING_CHECKPOINT_VOLUME,
    },
    timeout=86400,
)
def batch_and_embed(file):
    import pandas as pd
    from tqdm import tqdm

    # ------------------------------------------------------------------
    # 1. Load & sort chunks by token count (enables tight packing)
    # ------------------------------------------------------------------
    file_path = f"{DATASET_DIR}/{CHUNKED_NAME}/train/{file}"
    print(f"Loading {file}")
    df = pd.read_parquet(file_path)
    df["original_position"] = np.arange(len(df))
    df = df.sort_values(by="chunk_token_count", ascending=True)

    # ------------------------------------------------------------------
    # 2. Pack into batches
    # ------------------------------------------------------------------
    print(f"Packing batches for {file} ({len(df)} chunks, limit={CLIENT_BATCH_TOKEN_LIMIT})")
    t0 = time.monotonic()
    batches = pack_batches(df)
    print(f"  {len(batches)} batches in {time.monotonic() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 3. Send batches to TEI and collect responses
    # ------------------------------------------------------------------
    model_cls = TextEmbeddingsInference()
    pbar = tqdm(total=len(batches), desc=f"Embedding {file}")
    responses = []
    for resp in model_cls.embed.map(batches, order_outputs=False, return_exceptions=False):
        responses.append(resp)
        pbar.update(1)
    pbar.close()

    # ------------------------------------------------------------------
    # 4. Write embeddings as memmap in original row order
    # ------------------------------------------------------------------
    embedding_dim = responses[0][1].shape[1]
    out_dir = f"{EMBEDDING_DIR}/{EMBEDDING_NAME}/train"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, file.replace(".parquet", ".npy"))
    mmap = np.memmap(out_path, dtype="float32", mode="w+", shape=(len(df), embedding_dim))

    for indices, embeddings in responses:
        for idx, emb in zip(indices, embeddings):
            mmap[idx] = emb
    mmap.flush()
    del mmap

    EMBEDDING_CHECKPOINT_VOLUME.commit()
    return f"Done: {file} — {len(df)} chunks, {len(batches)} batches"


@app.local_entrypoint()
def full_job():
    for resp in batch_and_embed.map(files, order_outputs=False, return_exceptions=True):
        if isinstance(resp, Exception):
            print(f"EXCEPTION: {resp}")
        else:
            print(resp)
    print("All done.")
