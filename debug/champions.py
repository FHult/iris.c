#!/usr/bin/env python
"""
champions.py — cross-campaign champion read-out (insights + ablation seeds).

Reconstructs the champion (max held-out cond_gap done-iteration) of every flywheel
campaign from flywheel_history.db, joined to shard sources from shard_scores.db,
with the metadata needed to actually learn from them: the exact shard mix by
SOURCE, the hyperparameters, the checkpoint + git commit, exploration (first-
contact), and timing.

THREE honesty rails baked in (a naive cross-campaign table is misleading):
  1. cond_gap CONVENTION changed — campaigns before the held-out-EMA switch
     (HELDOUT_EMA_SINCE) recorded the OLD train-batch gap; their cond_gap MAGNITUDES
     are NOT comparable to run5+. The tool tags each champion's era and refuses to
     rank across the boundary (shard-mix / hparam comparisons stay valid).
  2. EPHEMERAL campaigns (smoke*/test*) are filtered by default — their cond_gap is
     test noise (e.g. the EMA-lag smoke's -5.066).
  3. cond_gap is a per-ITERATION whole-adapter metric, so "these shards made it a
     champion" is correlational, not causal — the source mix shows what mix produced
     the best iteration, not that those shards caused it.

Usage:
  champions.py                      # cross-campaign table (real campaigns)
  champions.py --include-ephemeral  # include smoke/test
  champions.py --seeds              # emit (hparams -> objective) for ablation warm-start
  champions.py --json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "train" / "scripts"))
from pipeline_lib import DATA_ROOT  # noqa: E402

FLYWHEEL_DB = DATA_ROOT / "flywheel_history.db"
SHARD_DB = DATA_ROOT / "shard_scores.db"

# Held-out-EMA cond_gap convention (review M1 + the EMA-lag fix) took effect for
# campaigns started on/after this date; earlier campaigns recorded the old
# train-batch gap and their cond_gap magnitudes are a DIFFERENT ruler.
HELDOUT_EMA_SINCE = "2026-06-11"


def is_ephemeral(name: str) -> bool:
    return name.lower().startswith(("smoke", "test"))


def era_of(ts_first: str | None) -> str:
    """'held-out-EMA' for campaigns on/after the convention switch, else 'train-batch'."""
    if ts_first and ts_first[:10] >= HELDOUT_EMA_SINCE:
        return "held-out-EMA"
    return "train-batch"


def champions(fh: sqlite3.Connection, include_ephemeral: bool = False) -> list[dict]:
    """One champion per campaign: the done-iteration with max cond_gap. Enriched
    with source mix (from shard_scores.shards), hparams, era, lineage."""
    fh.row_factory = sqlite3.Row
    rows = fh.execute("""
        WITH c AS (SELECT flywheel_name, MAX(cond_gap) b FROM iterations
                   WHERE status='done' AND cond_gap IS NOT NULL GROUP BY flywheel_name)
        SELECT i.*, cs.ts_first, cs.config_path, cs.status AS campaign_status
        FROM iterations i
        JOIN c ON i.flywheel_name=c.flywheel_name AND i.cond_gap=c.b AND i.status='done'
        LEFT JOIN campaign_summary cs ON cs.flywheel_name=i.flywheel_name
        GROUP BY i.flywheel_name ORDER BY i.ts_start
    """).fetchall()

    src = {}
    if SHARD_DB.exists():
        _ss = sqlite3.connect(SHARD_DB)
        src = {r[0]: (r[1] or "?") for r in
               _ss.execute("SELECT shard_id, source FROM shards")}
        _ss.close()

    out = []
    for r in rows:
        if not include_ephemeral and is_ephemeral(r["flywheel_name"]):
            continue
        ids = json.loads(r["selected_shards"]) if r["selected_shards"] else []
        hp = json.loads(r["hyperparams"]) if r["hyperparams"] else {}
        mix = Counter(src.get(s, "unknown") for s in ids)
        out.append({
            "campaign": r["flywheel_name"], "iteration": r["iteration"],
            "era": era_of(r["ts_first"]),
            "cond_gap": r["cond_gap"], "ref_gap": r["ref_gap"],
            "train_loss": r["train_loss"], "n_shards": len(ids),
            "first_contact": r["n_first_contact"],
            "source_mix": dict(sorted(mix.items())),
            "hyperparams": hp, "checkpoint": r["checkpoint"],
            "checkpoint_hash": r["checkpoint_hash"], "git_commit": r["git_commit"],
            "ts": r["ts_start"], "config_path": r["config_path"],
            "campaign_status": r["campaign_status"],
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(FLYWHEEL_DB))
    ap.add_argument("--include-ephemeral", action="store_true")
    ap.add_argument("--seeds", action="store_true",
                    help="emit (hparams -> cond_gap) for the held-out-EMA era only — "
                         "warm-start seeds for the ablation Bayesian search (ABL-3)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    fh = sqlite3.connect(args.db)
    champs = champions(fh, include_ephemeral=args.include_ephemeral)
    fh.close()
    if not champs:
        print("no champions found", file=sys.stderr)
        return 1

    if args.seeds:
        seeds = [{"campaign": c["campaign"], "hyperparams": c["hyperparams"],
                  "cond_gap": c["cond_gap"]}
                 for c in champs if c["era"] == "held-out-EMA"]
        if args.json:
            print(json.dumps(seeds, indent=2))
        else:
            print(f"# ablation warm-start seeds — held-out-EMA era only "
                  f"({len(seeds)} champion(s); same convention = safe to compare)")
            for s in seeds:
                hp = " ".join(f"{k}={v}" for k, v in sorted(s["hyperparams"].items()))
                print(f"  {s['campaign']:14} cond_gap={s['cond_gap']:+.4f}  {hp}")
            if len(seeds) < 2:
                print("\n  NOTE: <2 distinct-hparam seeds — flywheel campaigns FIX "
                      "hyperparams (only data varies), so they barely seed a HYPERPARAM\n"
                      "  search. The ablation harness exists precisely to vary hparams; "
                      "these seeds grow useful as ablation/varied campaigns accumulate.")
        return 0

    if args.json:
        print(json.dumps(champs, indent=2))
        return 0

    # Grouped by era; cond_gap only ranked WITHIN an era.
    print("Cross-campaign champions (cond_gap comparable only WITHIN an era):\n")
    for era in ("held-out-EMA", "train-batch"):
        grp = [c for c in champs if c["era"] == era]
        if not grp:
            continue
        ruler = ("held-out paired EMA gap — the production-valid metric"
                 if era == "held-out-EMA" else
                 "OLD train-batch gap — NOT comparable to held-out-EMA magnitudes")
        print(f"── era: {era}  ({ruler}) ──")
        for c in sorted(grp, key=lambda d: -(d["cond_gap"] or -9)):
            mix = " ".join(f"{k}={v}" for k, v in c["source_mix"].items())
            hp = c["hyperparams"]
            print(f"  {c['campaign']:14} iter{c['iteration']:>3}  "
                  f"cond_gap={c['cond_gap']:+.4f}  ref_gap="
                  f"{(c['ref_gap'] if c['ref_gap'] is not None else float('nan')):+.4f}  "
                  f"shards[{c['n_shards']}] first_contact={c['first_contact']}")
            print(f"                 sources: {mix}")
            print(f"                 hparams: cross_ref={hp.get('cross_ref_prob')} "
                  f"style_w={hp.get('style_loss_weight')}   ckpt={c['checkpoint']}  "
                  f"git={(c['git_commit'] or '')[:8]}")
        print()
    print("Insights: compare SOURCE MIX + HPARAMS across eras freely; compare cond_gap\n"
          "VALUES only within an era. Champion shards are correlational (cond_gap is a\n"
          "per-iteration metric), not proof those shards caused the win.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
