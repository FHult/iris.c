#!/usr/bin/env python
"""
rescope_shard_scores.py — rebuild shard_scores.db to ONE campaign's observations.

WHY: shard_scores.db is the cross-campaign append-only knowledge base, so its
cumulative per-shard EMAs (cond_gap_mean, *_excl_mean, attributed_*, composite,
effective_score) BLEND cond_gap conventions across campaigns. When a convention
changes (held-out-EMA vs old train-batch) or a bad/test run scores into it
(e.g. the EMA-lag smoke wrote cond_gap -5.066 to all 1280 shards' excluded EMA),
the blended scores silently corrupt SELECTION for the live campaign.

This deletes every score_updates row NOT belonging to --keep-campaign, then
rebuilds each shard's stored aggregates from the surviving rows — using the
production recompute path (ShardScoreDB._recompute_attributed). Safe because the
flywheel EMA is a cumulative equal-weight mean when temporal_decay is unset
(order-independent → AVG over score_updates is identical to the live EMA).

  Dry-run (default): report what WOULD change, write nothing.
  --apply: back up the DB first (shard_scores.db.bak-<ts>), then rewrite.

PRECONDITIONS for --apply: the campaign must be PAUSED (pause --free-gpu) so the
orchestrator isn't writing the DB. Verify the DB is quiescent first.

Usage:
  rescope_shard_scores.py --keep-campaign warmup-run5 [--db PATH] [--apply]
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from shard_selector import ShardScoreDB, SHARD_SCORES_DB_PATH, _compute_raw_composite

# Shard score columns reset to a clean slate before rebuild (everything derived
# from score_updates). NON-score columns (source, path, siglip_mean_emb,
# hard_example_count, loss_p95, selection bookkeeping) are preserved.
_RESET_SQL = """
UPDATE shards SET
  ref_gap_mean=NULL, cond_gap_mean=NULL, loss_mean=NULL, n_scored=0,
  ref_gap_last=NULL, cond_gap_last=NULL, loss_last=NULL,
  n_excluded=0, ref_gap_excl_mean=NULL, cond_gap_excl_mean=NULL,
  attributed_ref_gap=NULL, attributed_cond_gap=NULL, attributed_composite=NULL,
  attr_confidence=NULL, composite_score=NULL, effective_score=NULL
"""


def _campaign_counts(con) -> list[tuple]:
    return con.execute(
        "SELECT COALESCE(flywheel_name,'(null)'), COUNT(*) FROM score_updates "
        "GROUP BY flywheel_name ORDER BY COUNT(*) DESC").fetchall()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep-campaign", required=True)
    ap.add_argument("--db", default=str(SHARD_SCORES_DB_PATH))
    ap.add_argument("--apply", action="store_true",
                    help="actually rewrite (default: dry-run). Backs up first.")
    args = ap.parse_args()
    db_path = Path(args.db)
    keep = args.keep_campaign

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    counts = _campaign_counts(con)
    total = sum(c for _, c in counts)
    kept = sum(c for n, c in counts if n == keep)
    print(f"score_updates by campaign in {db_path.name}:")
    for n, c in counts:
        mark = "  KEEP" if n == keep else "  drop"
        print(f"  {mark}  {n:22} {c:>7}")
    print(f"keeping {kept:,} / {total:,} rows ({keep!r})")
    if kept == 0:
        print(f"ERROR: no rows for campaign {keep!r} — refusing.", file=sys.stderr)
        return 1

    n_shards_with_keep = con.execute(
        "SELECT COUNT(DISTINCT shard_id) FROM score_updates WHERE flywheel_name=?",
        (keep,)).fetchone()[0]
    print(f"{n_shards_with_keep:,} shards have {keep!r} observations (others reset to clean).")

    if not args.apply:
        print("\nDRY-RUN — nothing written. Re-run with --apply (campaign must be PAUSED).")
        con.close()
        return 0

    # ── apply ────────────────────────────────────────────────────────────────
    con.close()
    bak = db_path.with_name(db_path.name + f".bak-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(db_path, bak)
    print(f"\nbacked up → {bak}")

    db = ShardScoreDB(db_path)
    c = db._conn
    c.execute("DELETE FROM score_updates WHERE flywheel_name IS NULL OR flywheel_name<>?",
              (keep,))
    c.execute(_RESET_SQL)
    c.commit()

    # Rebuild per-shard aggregates from surviving rows. temporal_decay unset →
    # cumulative mean == AVG; *_last == the value at the max iteration.
    rows = c.execute("""
        SELECT shard_id, role,
               AVG(ref_gap) ag, AVG(cond_gap) cg, AVG(loss) lg, COUNT(*) n
        FROM score_updates GROUP BY shard_id, role
    """).fetchall()
    per: dict = {}
    for r in rows:
        per.setdefault(r["shard_id"], {})[r["role"]] = r
    # latest included observation per shard for *_last
    last = {r["shard_id"]: r for r in c.execute("""
        SELECT s.shard_id, s.ref_gap, s.cond_gap, s.loss FROM score_updates s
        JOIN (SELECT shard_id, MAX(iteration) mi FROM score_updates
              WHERE role='included' GROUP BY shard_id) m
        ON s.shard_id=m.shard_id AND s.iteration=m.mi AND s.role='included'
    """).fetchall()}

    for sid, roles in per.items():
        inc, exc = roles.get("included"), roles.get("excluded")
        ref_m = inc["ag"] if inc else None
        cond_m = inc["cg"] if inc else None
        loss_m = inc["lg"] if inc else None
        comp = _compute_raw_composite(ref_m, cond_m, loss_m)
        lr = last.get(sid)
        c.execute("""
            UPDATE shards SET
              ref_gap_mean=?, cond_gap_mean=?, loss_mean=?, n_scored=?,
              ref_gap_last=?, cond_gap_last=?, loss_last=?,
              n_excluded=?, ref_gap_excl_mean=?, cond_gap_excl_mean=?,
              composite_score=?
            WHERE shard_id=?
        """, (ref_m, cond_m, loss_m, inc["n"] if inc else 0,
              lr["ref_gap"] if lr else None, lr["cond_gap"] if lr else None,
              lr["loss"] if lr else None,
              exc["n"] if exc else 0,
              exc["ag"] if exc else None, exc["cg"] if exc else None,
              comp, sid))
    db._conn.commit()

    # Production recompute of attributed/effective from the corrected means.
    all_ids = [r[0] for r in c.execute("SELECT shard_id FROM shards").fetchall()]
    db._recompute_attributed(all_ids)
    print(f"rebuilt {len(per):,} shards' aggregates + recomputed attribution "
          f"for {len(all_ids):,} shards.")
    # Verify: no implausible excluded mean survives.
    bad = c.execute("SELECT COUNT(*) FROM shards WHERE cond_gap_excl_mean < -0.5").fetchone()[0]
    print(f"shards with cond_gap_excl_mean < -0.5 after rebuild: {bad} (want 0)")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
