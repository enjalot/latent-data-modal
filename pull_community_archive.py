#!/usr/bin/env python3
"""Pull the Community Archive tweet corpus via its public REST API.

Sanctioned access path (2026-08-22): the project's bulk parquet export is
paused over consent enforcement; the live Supabase REST API is the public
interface and respects CURRENT consent (opted-out users vanish). We pull
full_text of original tweets (retweets filtered server-side), keyset-
paginated, at a polite rate.

Output: /data/chunks/communityarchive-tweets/train/NNN.parquet shards of
500K rows (tweet_id, chunk_text, created_at, account_id), resumable via
state.json. ~8.2M tweets ≈ 17 shards. License: archive mission is an open
public-domain dataset (apache-2.0 on the HF org); volunteered archives.
Internal OOD-probe use; re-check terms before any public artifact.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://fabxmporizzqflnftavs.supabase.co/rest/v1/tweets"
KEY = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6"
       "ImZhYnhtcG9yaXp6cWZsbmZ0YXZzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjIyNDQ5"
       "MTIsImV4cCI6MjAzNzgyMDkxMn0.UIEJiUNkLsW28tBHmG-RQDW-I5JNlJLt62CSk9D_qG8")
OUT = Path("/data/chunks/communityarchive-tweets/train")
STATE = OUT.parent / "state.json"
PAGE = 1000
SHARD_ROWS = 500_000
SLEEP = 0.25  # ~4 req/s ≈ 4K rows/s — ~35 min for 8.2M


def fetch(after_id: str) -> list[dict]:
    params = {
        "select": "tweet_id,full_text,created_at,account_id",
        "order": "tweet_id.asc",
        "limit": str(PAGE),
        "full_text": "not.like.RT @*",
    }
    if after_id:
        params["tweet_id"] = f"gt.{after_id}"
    url = BASE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"apikey": KEY})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001 — transient network/API blips
            wait = 2 ** attempt
            print(f"retry {attempt} after {wait}s: {e}", flush=True)
            time.sleep(wait)
    raise RuntimeError("6 consecutive fetch failures")


def main() -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    OUT.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {
        "after_id": "", "shard": 0, "total": 0}
    buf: list[dict] = []
    t0 = time.time()

    def flush():
        nonlocal buf
        if not buf:
            return
        table = pa.table({
            "tweet_id": [r["tweet_id"] for r in buf],
            "chunk_text": [r["full_text"] or "" for r in buf],
            "created_at": [r["created_at"] for r in buf],
            "account_id": [str(r.get("account_id") or "") for r in buf],
        })
        pq.write_table(table, OUT / f"{state['shard']:03d}.parquet")
        print(f"shard {state['shard']:03d}: {len(buf):,} rows "
              f"(total {state['total']:,}, {state['total']/max(time.time()-t0,1):,.0f}/s)",
              flush=True)
        state["shard"] += 1
        buf = []
        STATE.write_text(json.dumps(state))

    while True:
        rows = fetch(state["after_id"])
        if not rows:
            break
        buf.extend(rows)
        state["after_id"] = rows[-1]["tweet_id"]
        state["total"] += len(rows)
        if len(buf) >= SHARD_ROWS:
            flush()
        time.sleep(SLEEP)
    flush()
    (OUT.parent / "manifest.json").write_text(json.dumps({
        "source": BASE, "rows": state["total"],
        "filter": "retweets excluded (full_text not like 'RT @%')",
        "access": "public REST API (bulk export paused for consent policy; "
                  "live API respects current consent)",
        "pulled_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=1))
    print(f"DONE {state['total']:,} tweets in {(time.time()-t0)/60:.0f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
