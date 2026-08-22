#!/usr/bin/env python3
"""Embed the jina-map training subsets WITH the document prompt (owner order
2026-08-22).

The existing jina-v5-nano corpora were embedded RAW (pooling_task="embed", no
prompt) — the known July problem: "Document: " shifts cosine to 0.73–0.94, so
raw-trained maps can't receive normally-embedded datasets. This embeds ONLY
the subset rows the 2M jina maps train on, with the model's own document-side
prompt, on the 5090:

  en-2m:    first 666,667 / 666,667 / 666,666 chunks of the sorted
            fineweb-edu / RedPajama / pile chunked-500 parquets
  multi-1m: first 50,000 chunks of each of the 20 fineweb2-<lang>-chunked-500

Output (fp16, row order = the definition above):
  /data/latent-basemap/substrates/jina-prompted/en-2m.f16.npy
  /data/latent-basemap/substrates/jina-prompted/multi-1m.f16.npy
  + rows manifests (corpus/lang spans) + prompt provenance.

Resumable per corpus-block via .done markers. Batch/throughput logged.
"""
from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/hf")

import numpy as np

OUT = Path("/data/latent-basemap/substrates/jina-prompted")
MODEL_ID = "jinaai/jina-embeddings-v5-text-nano-retrieval"
BATCH = int(os.environ.get("BATCH", "256"))
DIM = 768

EN = [
    ("fineweb-edu", "fineweb-edu-sample-10BT-chunked-500", 666_667),
    ("redpajama", "RedPajama-Data-V2-sample-10B-chunked-500", 666_667),
    ("pile", "pile-uncopyrighted-chunked-500", 666_666),
]
LANGS = ("arb_Arab", "ces_Latn", "cmn_Hani", "deu_Latn", "ell_Grek",
         "fra_Latn", "hin_Deva", "ind_Latn", "ita_Latn", "jpn_Jpan",
         "kor_Hang", "nld_Latn", "pol_Latn", "por_Latn", "rus_Cyrl",
         "spa_Latn", "swe_Latn", "tha_Thai", "tur_Latn", "vie_Latn")


def read_texts(chunk_dirname: str, n: int) -> list[str]:
    import pyarrow.parquet as pq
    texts: list[str] = []
    for f in sorted(glob.glob(f"/data/chunks/{chunk_dirname}/train/*.parquet")):
        t = pq.read_table(f, columns=["chunk_text"])["chunk_text"].to_pylist()
        texts.extend(t[:n - len(texts)])
        if len(texts) >= n:
            break
    assert len(texts) == n, (chunk_dirname, len(texts), n)
    return [t[:3000] for t in texts]  # same overlong guard as the raw pipeline


def pick_doc_prompt(model) -> tuple[str, str]:
    prompts = getattr(model, "prompts", None) or {}
    for key in ("passage", "document", "retrieval.passage", "doc"):
        if key in prompts:
            return key, prompts[key]
    return "__manual__", "Document: "


def embed_block(model, prompt_key, prompt_text, texts, out_path: Path,
                label: str) -> None:
    done = out_path.with_suffix(".done")
    if done.exists():
        print(f"{label}: done, skip", flush=True)
        return
    t0 = time.time()
    if prompt_key == "__manual__":
        vecs = model.encode([prompt_text + t for t in texts],
                            batch_size=BATCH, convert_to_numpy=True,
                            show_progress_bar=False)
    else:
        vecs = model.encode(texts, prompt_name=prompt_key, batch_size=BATCH,
                            convert_to_numpy=True, show_progress_bar=False)
    assert vecs.shape == (len(texts), DIM), vecs.shape
    np.save(out_path.with_suffix(".tmp.npy"), vecs.astype(np.float16))
    os.rename(out_path.with_suffix(".tmp.npy"), out_path)
    done.write_text("ok")
    dt = time.time() - t0
    print(f"{label}: {len(texts):,} in {dt/60:.1f} min ({len(texts)/dt:,.0f}/s)",
          flush=True)


def main() -> int:
    import torch
    from sentence_transformers import SentenceTransformer

    OUT.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL_ID, device="cuda",
                                trust_remote_code=True)
    model = model.half()
    prompt_key, prompt_text = pick_doc_prompt(model)
    print(f"document prompt: {prompt_key!r} -> {prompt_text!r}", flush=True)

    for name, dirname, n in EN:
        embed_block(model, prompt_key, prompt_text, read_texts(dirname, n),
                    OUT / f"en-{name}.f16.npy", f"en/{name}")
        torch.cuda.empty_cache()
    for lang in LANGS:
        embed_block(model, prompt_key, prompt_text,
                    read_texts(f"fineweb2-{lang}-chunked-500", 50_000),
                    OUT / f"ml-{lang}.f16.npy", f"ml/{lang}")
        torch.cuda.empty_cache()

    # assemble the two substrates in the pipeline's row order
    en = np.concatenate([np.load(OUT / f"en-{n}.f16.npy", mmap_mode="r")
                         for n, _, _ in EN])
    np.save(OUT / "en-2m.f16.npy", en)
    ml = np.concatenate([np.load(OUT / f"ml-{lang}.f16.npy", mmap_mode="r")
                         for lang in LANGS])
    np.save(OUT / "multi-1m.f16.npy", ml)
    (OUT / "manifest.json").write_text(json.dumps({
        "model": MODEL_ID,
        "prompt_key": prompt_key,
        "prompt_text": prompt_text,
        "en_spans": [{"corpus": n, "rows": c} for n, _, c in EN],
        "ml_spans": [{"lang": l, "rows": 50_000} for l in LANGS],
        "note": "document-side prompt applied (unlike the raw corpora); maps "
                "trained on these accept normally-embedded datasets.",
    }, indent=1))
    print(f"assembled en-2m {en.shape} + multi-1m {ml.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
