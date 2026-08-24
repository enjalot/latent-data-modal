#!/usr/bin/env python3
"""30M multilingual jina corpus — CPU background embed (owner order 2026-08-23).

Composition mirrors jina-multi-2m scaled 15x: 50% EN mix (fineweb-edu / RPJ /
pile chunked-500, 5M each) + 50% multilingual (20 fineweb2 langs x 750K) =
30M chunks, jina-v5-nano with the DOCUMENT prompt (the prompted-convention
ruling). Adjusting ratios is cheap while early — the manifest records spans.

Runs for days on CPU while the GPU does experiments; designed for that:
  * streams parquet slices -> encode -> append to a per-unit npy; nothing
    large ever resident (OOM-careful: unit = 100K rows ≈ 150 MB fp16).
  * resumable per unit (.done markers); --device cuda finishes the remainder
    ~20x faster whenever the GPU frees up (same layout, either device).
  * intended launch: systemd-run -p MemoryMax=48G -p CPUWeight=40 + nice,
    torch threads capped, so GPU jobs' data loading is never starved.

Output: /data/embeddings/jina30m-multilingual-v1/train/<block>-NNN.npy
(fp16, row order = manifest spans). ~46 GB total.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/hf")
os.environ.setdefault("OMP_NUM_THREADS", "24")

import numpy as np

MODEL_ID = "jinaai/jina-embeddings-v5-text-nano-retrieval"
OUT = Path("/data/embeddings/jina30m-multilingual-v1/train")
UNIT = 100_000
STOP_AFTER_UNITS = int(os.environ.get("STOP_AFTER_UNITS", "0"))  # 0 = run all
BATCH = int(os.environ.get("BATCH", "64"))
DEVICE = os.environ.get("DEVICE", "cpu")

EN = [("en-fineweb-edu", "fineweb-edu-sample-10BT-chunked-500", 5_000_000),
      ("en-redpajama", "RedPajama-Data-V2-sample-10B-chunked-500", 5_000_000),
      ("en-pile", "pile-uncopyrighted-chunked-500", 5_000_000)]
LANGS = ("arb_Arab", "ces_Latn", "cmn_Hani", "deu_Latn", "ell_Grek",
         "fra_Latn", "hin_Deva", "ind_Latn", "ita_Latn", "jpn_Jpan",
         "kor_Hang", "nld_Latn", "pol_Latn", "por_Latn", "rus_Cyrl",
         "spa_Latn", "swe_Latn", "tha_Thai", "tur_Latn", "vie_Latn")
BLOCKS = EN + [(f"ml-{l}", f"fineweb2-{l}-chunked-500", 750_000) for l in LANGS]


def iter_unit_texts(chunk_dirname: str, start: int, count: int):
    """Rows [start, start+count) of the sorted chunk parquets, streamed."""
    import pyarrow.parquet as pq
    import glob
    seen = 0
    out = []
    for f in sorted(glob.glob(f"/data/chunks/{chunk_dirname}/train/*.parquet")):
        pf = pq.ParquetFile(f)
        n = pf.metadata.num_rows
        if seen + n <= start:
            seen += n
            continue
        for b in pf.iter_batches(batch_size=32_768, columns=["chunk_text"]):
            texts = b.column(0).to_pylist()
            lo = max(0, start - seen)
            take = texts[lo:lo + (count - len(out))]
            out.extend(t[:3000] for t in take)
            seen_batch_end = seen + len(texts)
            if len(out) >= count:
                return out
            seen = seen_batch_end
        # continue next file with `seen` already advanced by full batches
    return out


def main() -> int:
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "24")))
    OUT.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL_ID, device=DEVICE, trust_remote_code=True)
    if DEVICE == "cuda":
        model = model.half()
    prompts = getattr(model, "prompts", {}) or {}
    pkey = next((k for k in ("passage", "document") if k in prompts), None)
    print(f"device {DEVICE}, prompt {pkey or 'manual Document:'}", flush=True)

    total_done = 0
    t_start = time.time()
    for name, dirname, rows in BLOCKS:
        for u, start in enumerate(range(0, rows, UNIT)):
            out = OUT / f"{name}-{u:03d}.npy"
            done = out.with_suffix(".done")
            if done.exists():
                total_done += min(UNIT, rows - start)
                continue
            texts = iter_unit_texts(dirname, start, min(UNIT, rows - start))
            t0 = time.time()
            if pkey:
                vecs = model.encode(texts, prompt_name=pkey, batch_size=BATCH,
                                    convert_to_numpy=True,
                                    show_progress_bar=False)
            else:
                vecs = model.encode(["Document: " + t for t in texts],
                                    batch_size=BATCH, convert_to_numpy=True,
                                    show_progress_bar=False)
            np.save(out.with_suffix(".tmp.npy"), vecs.astype(np.float16))
            os.rename(out.with_suffix(".tmp.npy"), out)
            done.write_text("ok")
            if STOP_AFTER_UNITS and sum(
                    1 for _ in OUT.glob("*.done")) % STOP_AFTER_UNITS == 0:
                pass  # marker counted below
            dt = time.time() - t0
            total_done += len(texts)
            rate = len(texts) / dt
            remain = 30_000_000 - total_done
            print(f"{name}-{u:03d}: {len(texts):,} in {dt/60:.1f} min "
                  f"({rate:,.0f}/s) | total {total_done:,} | "
                  f"ETA {remain/rate/86400:.1f} days", flush=True)
            del vecs, texts
            if STOP_AFTER_UNITS:
                STOP_AFTER_UNITS -= 1
                if STOP_AFTER_UNITS == 0:
                    print("STOP_AFTER_UNITS reached; yielding", flush=True)
                    return 0
    (OUT / "manifest.json").write_text(json.dumps({
        "model": MODEL_ID, "prompt": pkey or "Document: ",
        "blocks": [{"name": n, "chunks_dir": d, "rows": r} for n, d, r in BLOCKS],
        "rows": total_done, "dtype": "float16", "dim": 768,
    }, indent=1))
    print(f"DONE {total_done:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
