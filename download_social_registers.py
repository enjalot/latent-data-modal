#!/usr/bin/env python3
"""Download the two social-media register corpora (Twitter100M + Bluesky-5M) for
the OOD-coverage experiment. CPU/network only; small footprint (~1.6 GB).

Feeds the identical reddit-2m / communityarchive-2m register pipeline:
    download (this script) -> chunk_social_registers.py -> embed_social_registers.py

Targets (HF datasets):
  - enryu43/twitter100m_tweets  — 41 parquet shards of ~2.15M tweets each.
    SORT-CHECK (2026-08-25): the corpus is GROUPED BY USER — each shard covers a
    distinct contiguous user range (shard 0 head=MarketOne_Intl, tail=GlobalDubbing;
    shard 20 head=dauber246). Dates are mixed within a shard. Two adjacent shards
    would therefore be a narrow, concentrated user/register sample, so we pull
    EVERY ~10th shard (0,10,20,30,40) and the chunk step samples rows RANDOMLY
    (seed 42) across them.
  - Roronotalt/bluesky-five-million — one 5M-row parquet (MIT). Single file, so
    the chunk step just samples rows randomly (seed 42).

Downloads ONLY the specific files below via hf_hub_download (never load_dataset
the full 88M-row Twitter corpus). Idempotent: re-run resumes from HF cache.

Run:  HF_HOME=/data/hf .venv/bin/python download_social_registers.py
"""
import os
import shutil
import sys

os.environ.setdefault("HF_HOME", "/data/hf")

from huggingface_hub import hf_hub_download

# --- disk guard: abort if /data has < MIN_FREE_GB free -----------------------
MIN_FREE_GB = 40

# --- exact files to pull (see SORT-CHECK note above) -------------------------
TWITTER_REPO = "enryu43/twitter100m_tweets"
TWITTER_FILES = [
    "data/train-00000-of-00041-3f49db2da17edd5a.parquet",
    "data/train-00010-of-00041-39a461f69a92fa95.parquet",
    "data/train-00020-of-00041-cb49f7a05f4c5137.parquet",
    "data/train-00030-of-00041-9712df7d5c3d07f0.parquet",
    "data/train-00040-of-00041-9a723429a2a70e30.parquet",
]
BLUESKY_REPO = "Roronotalt/bluesky-five-million"
BLUESKY_FILES = ["data/train-0000.parquet"]

# Detected during column inspection (2026-08-25).
TWITTER_TEXT_COL, TWITTER_ID_COL = "tweet", "id"
BLUESKY_TEXT_COL, BLUESKY_ID_COL = "text", "uri"


def main() -> int:
    free_gb = shutil.disk_usage("/data").free / 1e9
    if free_gb < MIN_FREE_GB:
        print(f"ABORT: /data has only {free_gb:.1f} GB free (< {MIN_FREE_GB} GB "
              f"floor). Free space before downloading.", file=sys.stderr)
        return 1
    print(f"/data free: {free_gb:.1f} GB (>= {MIN_FREE_GB} GB floor) — proceeding.\n")

    paths = {"twitter100m": [], "bluesky-5m": []}
    for fn in TWITTER_FILES:
        p = hf_hub_download(TWITTER_REPO, fn, repo_type="dataset")
        paths["twitter100m"].append(p)
        print(f"twitter  {fn}\n         -> {p}")
    for fn in BLUESKY_FILES:
        p = hf_hub_download(BLUESKY_REPO, fn, repo_type="dataset")
        paths["bluesky-5m"].append(p)
        print(f"bluesky  {fn}\n         -> {p}")

    print("\n=== detected columns ===")
    print(f"twitter100m : text={TWITTER_TEXT_COL!r}  id={TWITTER_ID_COL!r}  "
          f"(keep: user, date)")
    print(f"bluesky-5m  : text={BLUESKY_TEXT_COL!r}  id={BLUESKY_ID_COL!r}  "
          f"(keep: author, created_at, langs)")
    print(f"\nfree after: {shutil.disk_usage('/data').free / 1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
