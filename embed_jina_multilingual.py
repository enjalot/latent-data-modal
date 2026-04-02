"""
Embed chunked FineWeb-2 multilingual data with jina-nano on L40S.
Reads from chunked parquet, writes embeddings as fp16 npy.

Usage:
    modal run embed_jina_multilingual.py          # all 20 languages
    modal run embed_jina_multilingual.py --lang fra_Latn
"""
import json
import os
import time

from modal import App, Image, Volume

DATASET_DIR = "/datasets"
EMBEDDING_DIR = "/embeddings"
CHUNK_SIZE = 500
BATCH_SIZE = 512

DATASET_VOLUME = Volume.from_name("datasets", create_if_missing=True)
EMBEDDING_VOLUME = Volume.from_name("embeddings", create_if_missing=True)

LANGUAGES = [
    "fra_Latn", "deu_Latn", "spa_Latn", "ita_Latn", "por_Latn",
    "nld_Latn", "pol_Latn", "rus_Cyrl", "cmn_Hani", "jpn_Jpan",
    "kor_Hang", "arb_Arab", "hin_Deva", "tur_Latn", "vie_Latn",
    "tha_Thai", "ind_Latn", "swe_Latn", "ces_Latn", "ell_Grek",
]

vllm_img = (
    Image.debian_slim(python_version="3.11")
    .pip_install("vllm>=0.18", "numpy", "pandas", "pyarrow", "huggingface_hub")
    .run_commands("pip install 'transformers>=5.0,<5.1'")
    .run_commands(
        'python3 -c "from huggingface_hub import snapshot_download; snapshot_download(\'jinaai/jina-embeddings-v5-text-nano-retrieval\')"',
    )
)

app = App("embed-jina-multilingual")


@app.function(
    gpu="L40S", image=vllm_img,
    volumes={DATASET_DIR: DATASET_VOLUME, EMBEDDING_DIR: EMBEDDING_VOLUME},
    timeout=3600,
)
def embed_language(lang_code: str):
    """Embed one language's chunked data with jina-nano."""
    import numpy as np
    import pandas as pd
    from vllm import LLM

    chunk_path = f"{DATASET_DIR}/fineweb2-{lang_code}-chunked-{CHUNK_SIZE}/train/000_00000.parquet"
    emb_dir = f"{EMBEDDING_DIR}/fineweb2-{lang_code}-chunked-{CHUNK_SIZE}-jina-nano/train"
    emb_path = f"{emb_dir}/000_00000.npy"

    # Check if already embedded
    if os.path.exists(emb_path):
        size_mb = os.path.getsize(emb_path) / 1e6
        print(f"  {lang_code}: already embedded ({size_mb:.0f} MB)")
        return {"lang": lang_code, "status": "cached", "size_mb": round(size_mb)}

    print(f"\n  {lang_code}: loading chunks...")
    df = pd.read_parquet(chunk_path, columns=["chunk_text"])
    texts = df["chunk_text"].tolist()
    n = len(texts)
    del df
    print(f"  {lang_code}: {n:,} chunks loaded")

    # Load model
    t_load = time.monotonic()
    model = LLM(
        model="jinaai/jina-embeddings-v5-text-nano-retrieval",
        runner="pooling", dtype="float16",
        max_model_len=1024,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
    )
    print(f"  {lang_code}: model loaded in {time.monotonic()-t_load:.0f}s")

    # Embed — truncate overlong chunks to avoid VLLMValidationError
    # jina-nano tokenizer may produce >1024 tokens from 500 bert-multilingual tokens
    all_embs = []
    t0 = time.monotonic()
    for i in range(0, n, BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        # Truncate very long texts (>3000 chars ≈ safe for 1024 jina tokens)
        batch = [t[:3000] if len(t) > 3000 else t for t in batch]
        try:
            outputs = model.encode(batch, pooling_task="embed")
            all_embs.extend([o.outputs.data.tolist() for o in outputs])
        except Exception as e:
            # If batch fails, try one-by-one with aggressive truncation
            for t in batch:
                try:
                    out = model.encode([t[:2000]], pooling_task="embed")
                    all_embs.extend([out[0].outputs.data.tolist()])
                except Exception:
                    # Last resort: embed a placeholder
                    out = model.encode([" "], pooling_task="embed")
                    all_embs.extend([out[0].outputs.data.tolist()])
        done = min(i + BATCH_SIZE, n)
        if (i // BATCH_SIZE) % 50 == 0:
            elapsed = time.monotonic() - t0
            cps = done / elapsed if elapsed > 0 else 0
            eta = (n - done) / cps / 60 if cps > 0 else 0
            print(f"    {lang_code}: {done:,}/{n:,} ({cps:.0f}/s, ~{eta:.0f}m remaining)")

    embed_s = time.monotonic() - t0
    cps = n / embed_s

    # Save as fp16
    embeddings = np.array(all_embs, dtype="float16")
    del all_embs, texts, model
    dim = embeddings.shape[1]

    os.makedirs(emb_dir, exist_ok=True)
    np.save(emb_path, embeddings)
    EMBEDDING_VOLUME.commit()

    size_mb = os.path.getsize(emb_path) / 1e6
    print(f"  {lang_code}: done — {n:,} chunks, {cps:.0f}/s, {embed_s:.0f}s, {size_mb:.0f} MB")

    return {
        "lang": lang_code, "n_chunks": n, "dim": dim,
        "embed_s": round(embed_s), "cps": round(cps),
        "size_mb": round(size_mb), "status": "embedded",
    }


@app.local_entrypoint()
def main(lang: str = "all"):
    targets = LANGUAGES if lang == "all" else [lang]
    print(f"Embedding {len(targets)} languages with jina-nano on L40S")
    print(f"  Chunk size: {CHUNK_SIZE}, batch: {BATCH_SIZE}")
    print()

    results = []
    # Use .map() for parallel embedding across languages
    for r in embed_language.map(targets, order_outputs=False, return_exceptions=True):
        if isinstance(r, Exception):
            print(f"  EXCEPTION: {r}")
            results.append({"status": "failed", "error": str(r)})
        else:
            results.append(r)
            print(f"  {r['lang']}: {r.get('n_chunks', '?'):,} chunks @ {r.get('cps', '?')}/s — {r.get('status')}")

    ok = [r for r in results if r.get("status") in ("embedded", "cached")]
    total_chunks = sum(r.get("n_chunks", 0) for r in ok)
    total_mb = sum(r.get("size_mb", 0) for r in ok)
    print(f"\n  {len(ok)}/{len(results)} done")
    print(f"  Total: {total_chunks:,} chunks, {total_mb:,} MB ({total_mb/1024:.1f} GB)")
