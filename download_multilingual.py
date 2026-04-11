"""
Download FineWeb-2 multilingual data to Modal volume.
Downloads one shard per language (~1B tokens each) with robust error handling.

Each shard is ~4.8 GB parquet. Downloads individually to avoid zombie processes.

Usage:
    modal run download_multilingual.py  # downloads all 20 languages
    modal run download_multilingual.py --lang fra_Latn  # single language
"""
import json
import os
import time

from modal import App, Image, Volume, Secret

DATASET_DIR = "/data"
DATASET_NAME = "HuggingFaceFW/fineweb-2"
SAVE_PREFIX = "fineweb2"

# Target: 1 train shard per language (~1B tokens each)
LANGUAGES = {
    "fra_Latn": "French",
    "deu_Latn": "German",
    "spa_Latn": "Spanish",
    "ita_Latn": "Italian",
    "por_Latn": "Portuguese",
    "nld_Latn": "Dutch",
    "pol_Latn": "Polish",
    "rus_Cyrl": "Russian",
    "cmn_Hani": "Chinese",
    "jpn_Jpan": "Japanese",
    "kor_Hang": "Korean",
    "arb_Arab": "Arabic",
    "hin_Deva": "Hindi",
    "tur_Latn": "Turkish",
    "vie_Latn": "Vietnamese",
    "tha_Thai": "Thai",
    "ind_Latn": "Indonesian",
    "swe_Latn": "Swedish",
    "ces_Latn": "Czech",
    "ell_Grek": "Greek",
}

volume = Volume.from_name("datasets", create_if_missing=True)
image = Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub[hf_transfer]", "pyarrow", "pandas"
)
app = App(image=image)


@app.function(
    volumes={DATASET_DIR: volume},
    timeout=3600,  # 1 hour max per shard (plenty for ~5GB)
    secrets=[Secret.from_name("huggingface-secret")],
)
def download_shard(lang_code: str, shard_file: str):
    """Download a single parquet shard directly via huggingface_hub.

    No datasets library — direct file download to avoid zombie processes.
    Downloads to volume, commits, returns metadata.
    """
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    lang_name = LANGUAGES.get(lang_code, lang_code)
    save_dir = f"{DATASET_DIR}/{SAVE_PREFIX}-{lang_code}"
    os.makedirs(f"{save_dir}/train", exist_ok=True)

    local_path = f"{save_dir}/train/{shard_file.split('/')[-1]}"

    # Check if already downloaded
    if os.path.exists(local_path):
        pf = pq.ParquetFile(local_path)
        n_rows = pf.metadata.num_rows
        print(f"  {lang_code} ({lang_name}): already exists ({n_rows:,} rows)")
        return {"lang": lang_code, "name": lang_name, "file": local_path,
                "rows": n_rows, "status": "cached"}

    print(f"  Downloading {lang_code} ({lang_name}): {shard_file}")
    t0 = time.monotonic()

    try:
        # Direct file download — no datasets library, no multi-process
        downloaded = hf_hub_download(
            repo_id=DATASET_NAME,
            filename=shard_file,
            repo_type="dataset",
            local_dir=f"{save_dir}/_cache",
        )

        # Copy to final location
        import shutil
        shutil.copy2(downloaded, local_path)

        # Verify
        pf = pq.ParquetFile(local_path)
        n_rows = pf.metadata.num_rows
        size_mb = os.path.getsize(local_path) / 1e6
        elapsed = time.monotonic() - t0

        volume.commit()

        print(f"  {lang_code}: {n_rows:,} rows, {size_mb:.0f} MB, {elapsed:.0f}s")
        return {"lang": lang_code, "name": lang_name, "file": local_path,
                "rows": n_rows, "size_mb": round(size_mb), "seconds": round(elapsed),
                "status": "downloaded"}

    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"  {lang_code} FAILED after {elapsed:.0f}s: {e}")
        return {"lang": lang_code, "name": lang_name, "status": "failed",
                "error": str(e), "seconds": round(elapsed)}


@app.local_entrypoint()
def main(lang: str = "all"):
    """Download one shard per language from FineWeb-2."""
    if lang == "all":
        targets = list(LANGUAGES.keys())
    else:
        targets = [lang]

    # Build download list: first train shard for each language
    jobs = []
    for lang_code in targets:
        shard = f"data/{lang_code}/train/000_00000.parquet"
        jobs.append((lang_code, shard))

    print(f"Downloading {len(jobs)} shards ({len(targets)} languages)")
    print(f"Each shard is ~4-5 GB")
    print()

    results = []
    # Download sequentially to avoid overloading — each shard is large
    for lang_code, shard in jobs:
        try:
            r = download_shard.remote(lang_code, shard)
            results.append(r)
            status = r.get("status", "unknown")
            rows = r.get("rows", "?")
            print(f"  {r['lang']} ({r['name']}): {status} — {rows} rows")
        except Exception as e:
            print(f"  {lang_code}: EXCEPTION — {e}")
            results.append({"lang": lang_code, "status": "exception", "error": str(e)})

    # Summary
    print(f"\n{'='*60}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    ok = [r for r in results if r.get("status") in ("downloaded", "cached")]
    failed = [r for r in results if r.get("status") not in ("downloaded", "cached")]
    print(f"  Success: {len(ok)}/{len(results)}")
    if failed:
        print(f"  Failed: {[r['lang'] for r in failed]}")
    total_rows = sum(r.get("rows", 0) for r in ok)
    total_mb = sum(r.get("size_mb", 0) for r in ok)
    print(f"  Total rows: {total_rows:,}")
    print(f"  Total size: {total_mb:,} MB")
