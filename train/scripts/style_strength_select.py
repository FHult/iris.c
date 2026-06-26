#!/usr/bin/env python
"""
style_strength_select.py — rank records by STYLE-SIGNAL strength and cut an ultra-signal subset.

The SREF adapter learns style from cross-ref pairs (same-style/different-content CSD neighbors).
A record with a DENSE, high-cosine style neighborhood has a strong, repeatable style AND clean
pairs available — exactly the high-signal data for style learning. A record with weak/sparse
neighbors (e.g. a generic photo) teaches little style. This reads the existing neighbors.sqlite
(built by style_neighbors.py from CSD embeddings — no GPU here) and:
  - scores each record's style_strength = mean of its top-N neighbor cosines,
  - ranks + selects the top --top-pct as the "ultra-signal" subset,
  - reports the per-shard breakdown (which shards are signal-rich → candidates to cut into
    ultra-signal shards) and writes a manifest of selected rec_ids.

This is the cheap-score-first step: select on style strength BEFORE the expensive VAE/Qwen3/SigLIP
precompute, then precompute only the ultra-signal subset. CPU-only.

Usage:
  train/.venv/bin/python train/scripts/style_strength_select.py NEIGHBORS.sqlite \
      [--top-pct 25] [--topn 5] [--min-neighbors 3] [--out manifest.json]
"""
from __future__ import annotations
import argparse, json, sqlite3
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("neighbors", help="neighbors.sqlite (style_neighbors.py output)")
    ap.add_argument("--top-pct", type=float, default=25.0, help="keep this %% of records (by strength)")
    ap.add_argument("--topn", type=int, default=5, help="avg the top-N neighbor cosines for strength")
    ap.add_argument("--min-neighbors", type=int, default=3, help="drop records with fewer neighbors")
    ap.add_argument("--out", default=None, help="write selected rec_ids manifest JSON here")
    args = ap.parse_args()

    db = sqlite3.connect(f"file:{args.neighbors}?mode=ro", uri=True)
    recs = []   # (strength, rec_id, shard, n_neigh)
    for rec_id, ncos in db.execute("SELECT rec_id, neighbor_cos FROM neighbors"):
        cos = json.loads(ncos)
        if len(cos) < args.min_neighbors:
            continue
        topn = sorted(cos, reverse=True)[: args.topn]
        strength = sum(topn) / len(topn)
        shard = rec_id.rsplit("_", 1)[0]
        recs.append((strength, rec_id, shard, len(cos)))
    db.close()

    if not recs:
        print("no records with enough neighbors", file=sys.stderr)
        return 1
    recs.sort(reverse=True)
    n = len(recs)
    keep = max(1, int(n * args.top_pct / 100.0))
    sel = recs[:keep]
    thr = sel[-1][0]

    # strength distribution
    import statistics as st
    vals = [r[0] for r in recs]
    pct = lambda p: sorted(vals)[min(n - 1, int(n * p / 100))]
    print(f"records with >= {args.min_neighbors} neighbors: {n:,}  (of the full neighbor set)")
    print(f"style_strength (mean top-{args.topn} neighbor cos): "
          f"min={min(vals):.3f}  p25={pct(25):.3f}  med={st.median(vals):.3f}  "
          f"p75={pct(75):.3f}  p90={pct(90):.3f}  max={max(vals):.3f}")
    print(f"SELECT top {args.top_pct:.0f}% → {keep:,} records, strength >= {thr:.3f}")

    # per-shard: how concentrated is the ultra-signal? (→ which shards to cut)
    tot_by_shard = defaultdict(int); sel_by_shard = defaultdict(int)
    for _, _, sh, _ in recs:
        tot_by_shard[sh] += 1
    for _, _, sh, _ in sel:
        sel_by_shard[sh] += 1
    print(f"\nper-shard ultra-signal concentration (selected/total, % — high % = signal-rich shard):")
    rows = sorted(tot_by_shard, key=lambda s: -(sel_by_shard[s] / max(tot_by_shard[s], 1)))
    for sh in rows:
        t, s = tot_by_shard[sh], sel_by_shard[sh]
        print(f"  {sh:<12} {s:>5}/{t:<5} {100*s/max(t,1):5.1f}%")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "top_pct": args.top_pct, "topn": args.topn, "strength_threshold": round(thr, 4),
            "n_selected": keep, "rec_ids": [r[1] for r in sel],
        }, indent=2))
        print(f"\nwrote manifest ({keep:,} rec_ids) -> {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
