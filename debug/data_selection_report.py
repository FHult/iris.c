#!/usr/bin/env python
"""
data_selection_report.py — curation view for a from-scratch data-selection flywheel
(read-only on shard_scores.db).

When the flywheel runs `from_scratch_each_iter` (warmup-run4+), each iteration's cond_gap
measures its selected shard mix and the bandit + contrastive attribution decide WHICH data
to keep. The campaign cond_gap trajectory (debug/flywheel_refgap.py) no longer tells the
whole story — what matters is which *shards/sources* are emerging as good vs bad. This
report answers that:

  - per-source curation summary (which SOURCE conditions best — directly informs the
    production data recipe);
  - top winners and bottom "stinkers" by cond_gap (the data the bandit should keep/drop);
  - attribution warmth (how close the causal signal is to usable) + the n_scored
    distribution (how the included-observation count — the binding constraint — is filling);
  - selection concentration (is one shard dominating? runbook warm-up criterion: top shard
    n_selected <= ~8).

Metric note: `cond_gap_mean` is the iteration-level cond_gap smeared across that iteration's
included shards (coarse, correlational). `attributed_cond_gap` is the contrastive
included-minus-excluded estimate (causal), trustworthy only once attr_confidence >= 1.0.

Usage:
    data_selection_report.py [db_path] [--top N]
    (default db: /Volumes/2TBSSD/shard_scores.db, N=12)
"""

import argparse
import os
import sqlite3
import sys

_SRC = "COALESCE(manifest_source, source)"


def _f(x, w=8, p=4):
    return f"{x:>{w}.{p}f}" if x is not None else f"{'—':>{w}}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("db", nargs="?", default="/Volumes/2TBSSD/shard_scores.db")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"shard_scores.db not found: {args.db}")
        return 1
    c = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row

    total, scored, attr_ready = c.execute(
        "SELECT COUNT(*), SUM(n_scored>0), SUM(attr_confidence>=1.0) FROM shards"
    ).fetchone()
    scored = scored or 0
    attr_ready = attr_ready or 0
    print(f"=== data-selection report ({args.db}) ===")
    print(f"pool: {total} shards | {scored} touched ({100*scored/total:.1f}%) | "
          f"{attr_ready} attribution-ready (attr_confidence>=1.0)")

    # ── per-source curation summary ─────────────────────────────────────────────
    print(f"\n── per-source (which source conditions best) ──")
    print(f"{'source':<22}{'shards':>7}{'scored':>7}{'mean_cg':>9}{'mean_attr_cg':>13}{'attr_rdy':>9}")
    rows = c.execute(f"""
        SELECT {_SRC} src, COUNT(*) n,
               SUM(n_scored>0) scored,
               AVG(CASE WHEN n_scored>0 THEN cond_gap_mean END) avg_cg,
               AVG(CASE WHEN attr_confidence>=1.0 THEN attributed_cond_gap END) avg_attr,
               SUM(attr_confidence>=1.0) ar
        FROM shards GROUP BY src ORDER BY avg_cg DESC NULLS LAST
    """).fetchall()
    for r in rows:
        print(f"{(r['src'] or 'unknown'):<22}{r['n']:>7}{r['scored'] or 0:>7}"
              f"{_f(r['avg_cg'],9)}{_f(r['avg_attr'],13)}{r['ar'] or 0:>9}")

    # ── winners / stinkers (need >=2 included obs for a non-trivial mean) ────────
    def _shard_table(title, order):
        print(f"\n── {title} (n_scored>=2) ──")
        print(f"{'shard':<14}{'source':<20}{'inc':>4}{'exc':>4}{'cond_gap':>9}{'attr_cg':>9}{'conf':>6}{'sel':>4}")
        for r in c.execute(f"""
            SELECT shard_id, {_SRC} src, n_scored, n_excluded, cond_gap_mean,
                   attributed_cond_gap, attr_confidence, n_selected
            FROM shards WHERE n_scored>=2 ORDER BY cond_gap_mean {order} LIMIT ?
        """, (args.top,)).fetchall():
            sid = (r["shard_id"] or "")[-13:]
            print(f"{sid:<14}{(r['src'] or 'unknown')[:19]:<20}{r['n_scored']:>4}"
                  f"{r['n_excluded']:>4}{_f(r['cond_gap_mean'],9)}{_f(r['attributed_cond_gap'],9)}"
                  f"{_f(r['attr_confidence'],6,2)}{r['n_selected']:>4}")

    has2 = c.execute("SELECT COUNT(*) FROM shards WHERE n_scored>=2").fetchone()[0]
    if has2:
        _shard_table("top winners", "DESC")
        _shard_table("bottom stinkers", "ASC")
    else:
        print("\n(no shard has >=2 included observations yet — winners/stinkers need more iters)")

    # ── n_scored distribution (the binding constraint for attribution) ──────────
    print(f"\n── included-observation (n_scored) distribution ──")
    dist = {0: 0, 1: 0, 2: 0, 3: 0}
    for r in c.execute("SELECT n_scored, COUNT(*) n FROM shards GROUP BY n_scored").fetchall():
        k = r["n_scored"] if r["n_scored"] < 3 else 3
        dist[k] = dist.get(k, 0) + r["n"]
    print(f"  0 obs: {dist.get(0,0):>5}   1 obs: {dist.get(1,0):>5}   "
          f"2 obs: {dist.get(2,0):>5}   3+ obs: {dist.get(3,0):>5}  "
          f"(3+ ⇒ attribution can fire)")

    # ── selection concentration (runbook: top shard n_selected should stay <=~8) ─
    print(f"\n── selection concentration (over-picked shards?) ──")
    top_sel = c.execute(f"""
        SELECT shard_id, {_SRC} src, n_selected FROM shards
        WHERE n_selected>0 ORDER BY n_selected DESC LIMIT 5
    """).fetchall()
    total_sel = c.execute("SELECT SUM(n_selected) FROM shards").fetchone()[0] or 0
    if top_sel:
        mx = top_sel[0]["n_selected"]
        flag = "  ⚠ head over-concentrated (>8)" if mx > 8 else ""
        print(f"  total selections: {total_sel} | most-picked shard: {mx}{flag}")
        for r in top_sel:
            print(f"    {(r['shard_id'] or '')[-13:]:<14}{(r['src'] or '')[:19]:<20} n_selected={r['n_selected']}")
    else:
        print("  no selections recorded yet")

    c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
