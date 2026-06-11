#!/usr/bin/env python
"""
style_shard_report.py — per-shard style-signal metrics from CSD bundles (SREF-1).

Consumes a style-embedding cache (per-shard npz bundles from style_precompute.py) and
ranks shards by how much STYLE TRAINING SIGNAL they carry:

  diversity     mean intra-shard pairwise style distance — how many distinct styles a
                shard contributes to the pool (higher = broader coverage).
  pair_rich     fraction of the shard's records with >= MIN_NBRS strong style neighbors
                (cos >= --tau) anywhere in the encoded pool — isolated styles can't form
                same-style/different-content training pairs (higher = better cross-ref).
  connectivity  number of OTHER shards this shard's records have strong neighbors in —
                staging style-kin shards together is what makes iteration-local
                neighbor lists rich.

Writes a JSON report (sorted by pair_rich) + a connectivity edge list that shard
selection can use to co-stage style neighborhoods.

Usage:
  style_shard_report.py --style-cache DIR --out report.json
                        [--tau 0.6] [--min-nbrs 3] [--sample-per-shard 400]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--style-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.6,
                    help="cosine threshold for a 'strong' style neighbor")
    ap.add_argument("--min-nbrs", type=int, default=3)
    ap.add_argument("--dedup-cos", type=float, default=0.95,
                    help="cosines above this are near-duplicates, not style "
                         "neighbors (match style_neighbors --dedup-cos)")
    ap.add_argument("--sample-per-shard", type=int, default=400,
                    help="cap per-shard rows in the global matrix (keeps 1M-pool RAM sane)")
    ap.add_argument("--chunk", type=int, default=2048)
    args = ap.parse_args()

    cache = Path(args.style_cache)
    rng = np.random.RandomState(42)

    # Load (sampled) embeddings, tracking shard membership.
    shard_names: list[str] = []
    shard_of: list[int] = []
    mats: list[np.ndarray] = []
    for f in sorted(cache.glob("*.npz")):
        if ".tmp" in f.name:        # crash-orphaned temp from style_precompute
            continue
        d = np.load(f)
        keys = list(d.files)
        if len(keys) > args.sample_per_shard:
            keys = list(rng.choice(keys, args.sample_per_shard, replace=False))
        si = len(shard_names)
        shard_names.append(f.stem)
        for k in keys:
            mats.append(d[k])
            shard_of.append(si)
    if not mats:
        raise SystemExit(f"no bundles under {cache}")
    E = np.stack(mats).astype(np.float32)
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)
    shard_of_a = np.array(shard_of)
    n, S = len(E), len(shard_names)
    print(f"{n:,} sampled embeddings across {S} shards", flush=True)

    # Per-record strong-neighbor counts + cross-shard connectivity (chunked exact).
    nbr_count = np.zeros(n, dtype=np.int32)
    conn = [set() for _ in range(S)]
    t0 = time.time()
    for s in range(0, n, args.chunk):
        e = min(s + args.chunk, n)
        cos = E[s:e] @ E.T
        for r in range(e - s):
            cos[r, s + r] = -2.0
        strong = cos >= args.tau
        # drop near-duplicates from the neighbor counts (same-job variants)
        strong &= cos <= args.dedup_cos
        nbr_count[s:e] = strong.sum(axis=1)
        rows, cols = np.nonzero(strong)
        for r, c in zip(shard_of_a[s + rows], shard_of_a[cols]):
            if r != c:
                conn[r].add(int(c))
        if (s // args.chunk) % 20 == 0:
            done = e / n
            print(f"  {e:,}/{n:,}  ETA {((time.time()-t0)/max(done,1e-9))*(1-done):.0f}s",
                  flush=True)

    report = []
    for si, name in enumerate(shard_names):
        idx = np.where(shard_of_a == si)[0]
        Es = E[idx]
        if len(idx) >= 2:
            sub = Es[rng.choice(len(Es), min(150, len(Es)), replace=False)]
            d = sub @ sub.T
            iu = np.triu_indices(len(sub), 1)
            diversity = float(np.sqrt(np.maximum(2 - 2 * d[iu], 0)).mean())
        else:
            diversity = 0.0
        pair_rich = float((nbr_count[idx] >= args.min_nbrs).mean())
        report.append({"shard": name, "n_sampled": int(len(idx)),
                       "diversity": round(diversity, 4),
                       "pair_rich": round(pair_rich, 4),
                       "connectivity": len(conn[si])})

    report.sort(key=lambda r: -r["pair_rich"])
    out = {"tau": args.tau, "min_nbrs": args.min_nbrs,
           "n_shards": S, "n_sampled": n,
           "pool_pair_rich": round(float((nbr_count >= args.min_nbrs).mean()), 4),
           "shards": report,
           "connectivity_edges": [[shard_names[a], shard_names[b]]
                                  for a in range(S) for b in sorted(conn[a]) if a < b]}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"pool pair_rich: {out['pool_pair_rich']:.2%} of records have >= "
          f"{args.min_nbrs} strong style neighbors")
    print("top shards by pair_rich:")
    for r in report[:8]:
        print(f"  {r['shard']}: pair_rich={r['pair_rich']:.2f} "
              f"diversity={r['diversity']:.3f} connectivity={r['connectivity']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
