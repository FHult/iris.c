#!/usr/bin/env python
"""
style_neighbors.py — per-record top-k style-neighbor lists (SREF-1 pairing substrate).

Input: a style-embedding cache built by style_precompute.py (per-shard npz bundles of
[768] f16 CSD embeddings). Output: neighbors.sqlite mapping every record to its top-k
style neighbors — the table dataset.py's cross-ref will sample SAME-STYLE/different-
content references from.

Two exclusions matter (measured on JourneyDB):
  - SELF, obviously.
  - NEAR-DUPLICATES (cos > --dedup-cos, default 0.95): JourneyDB ships ~4 variants per
    job with near-identical style AND content; pairing those would re-introduce the
    content leak that style pairing exists to avoid.

Also reports the acceptance metric: mean top-k NN distance / mean random-pair distance
(gate <= 0.7 from BACKLOG SREF-1).

Usage:
  style_neighbors.py --style-cache DIR --out neighbors.sqlite [--k 10]
                     [--dedup-cos 0.95] [--chunk 2048]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np


def load_cache(cache_dir: Path) -> tuple[list[str], np.ndarray]:
    ids: list[str] = []
    mats: list[np.ndarray] = []
    for f in sorted(cache_dir.glob("*.npz")):
        d = np.load(f)
        for rid in d.files:
            ids.append(rid)
            mats.append(d[rid])
    if not ids:
        raise SystemExit(f"no embeddings under {cache_dir}")
    E = np.stack(mats).astype(np.float32)
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)
    return ids, E


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--style-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--dedup-cos", type=float, default=0.95)
    ap.add_argument("--chunk", type=int, default=2048)
    args = ap.parse_args()

    ids, E = load_cache(Path(args.style_cache))
    n = len(ids)
    print(f"{n:,} embeddings loaded ({E.nbytes/1e6:.0f} MB f32)", flush=True)

    k = args.k
    top_idx = np.empty((n, k), dtype=np.int64)
    top_cos = np.empty((n, k), dtype=np.float32)
    n_deduped = 0
    t0 = time.time()
    for s in range(0, n, args.chunk):
        e = min(s + args.chunk, n)
        cos = E[s:e] @ E.T                       # [chunk, n]
        for r in range(e - s):
            cos[r, s + r] = -2.0                 # exclude self
        dup = cos > args.dedup_cos               # exclude near-duplicates
        n_deduped += int(dup.sum())
        cos[dup] = -2.0
        part = np.argpartition(-cos, k, axis=1)[:, :k]
        pc = np.take_along_axis(cos, part, axis=1)
        order = np.argsort(-pc, axis=1)
        top_idx[s:e] = np.take_along_axis(part, order, axis=1)
        top_cos[s:e] = np.take_along_axis(pc, order, axis=1)
        if (s // args.chunk) % 20 == 0:
            done = e / n
            print(f"  {e:,}/{n:,} ({100*done:.0f}%)  ETA {((time.time()-t0)/max(done,1e-9))*(1-done):.0f}s",
                  flush=True)

    # acceptance metric: NN-pair vs random-pair distance (euclidean on unit sphere)
    rng = np.random.RandomState(0)
    i, j = rng.randint(0, n, 8000), rng.randint(0, n, 8000)
    m = i != j
    rand_d = np.linalg.norm(E[i[m]] - E[j[m]], axis=1).mean()
    nn_d = np.sqrt(np.maximum(2 - 2 * top_cos[:, :5], 0)).mean()
    ratio = float(nn_d / rand_d)
    print(f"top-5 NN/random distance ratio: {ratio:.3f}  (gate <= 0.7)", flush=True)
    print(f"near-duplicate exclusions: {n_deduped:,}", flush=True)

    out = Path(args.out)
    db = sqlite3.connect(out)
    db.execute("CREATE TABLE IF NOT EXISTS neighbors "
               "(rec_id TEXT PRIMARY KEY, neighbor_ids TEXT, neighbor_cos TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    rows = ((ids[r],
             json.dumps([ids[c] for c in top_idx[r]]),
             json.dumps([round(float(c), 4) for c in top_cos[r]]))
            for r in range(n))
    db.executemany("INSERT OR REPLACE INTO neighbors VALUES (?,?,?)", rows)
    for kk, vv in {"k": k, "dedup_cos": args.dedup_cos, "n_records": n,
                   "nn5_random_ratio": round(ratio, 4),
                   "style_cache": str(args.style_cache)}.items():
        db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (kk, json.dumps(vv)))
    db.commit()
    db.close()
    print(f"wrote {n:,} neighbor lists -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
