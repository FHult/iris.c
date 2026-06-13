#!/usr/bin/env python
"""
source_attribution.py — per-source data-selection read-out for a flywheel campaign.

Answers "which DATA SOURCES (journeydb / coyo / laion / wikiart mixes) condition
the adapter best?" — the question behind run5's per_source_min and the eventual
production data recipe. Reads ShardScoreDB.source_attribution (the single source
of truth) + the selection-mix trend; adds no state (all reconstructed from
score_updates, single-campaign-scoped to avoid the cross-campaign convention mix).

It is deliberately CONSERVATIVE: cond_gap is a per-iteration metric, so per-source
contrast is only meaningful when a source is sometimes EXCLUDED. The tool reports
ubiquity (fraction of iterations a source appeared in) and refuses to call a
"best source" until at least one source clears the trust + separability bars.

Usage:
  debug/source_attribution.py [--campaign warmup-run5] [--db PATH] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "train" / "scripts"))
from shard_selector import ShardScoreDB, SHARD_SCORES_DB_PATH, SOURCE_TRUST_FLOOR


def _latest_campaign(db: ShardScoreDB) -> str | None:
    row = db._conn.execute(
        "SELECT flywheel_name FROM score_updates "
        "WHERE flywheel_name IS NOT NULL ORDER BY id DESC LIMIT 1").fetchone()
    return row[0] if row else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", default=None,
                    help="flywheel name (default: most recent in score_updates)")
    ap.add_argument("--db", default=str(SHARD_SCORES_DB_PATH))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    db = ShardScoreDB(Path(args.db))
    campaign = args.campaign or _latest_campaign(db)
    if not campaign:
        print("no campaign with score_updates found", file=sys.stderr)
        return 1

    rollup = db.source_attribution(campaign)
    mix = db.source_iteration_mix(campaign)

    if args.json:
        print(json.dumps({"campaign": campaign, "sources": rollup,
                          "iteration_mix": mix}, indent=2))
        db.close()
        return 0

    iters = sorted({m["it"] for m in mix})
    print(f"=== source attribution — campaign '{campaign}' ({len(iters)} iterations) ===\n")

    # Section 1 — selection MIX trend (the clean, unconfounded signal).
    sources = [r["source"] for r in rollup]
    print("Selection mix (shards INCLUDED per iteration by source) — is the meta-")
    print("flywheel drifting toward better sources? This signal is confound-free.\n")
    hdr = "  iter " + "".join(f"{s[:16]:>18}" for s in sources)
    print(hdr)
    by_it: dict = {}
    for m in mix:
        by_it.setdefault(m["it"], {})[m["source"]] = m["n"]
    for it in iters[-12:]:
        row = "".join(f"{by_it.get(it, {}).get(s, 0):>18}" for s in sources)
        print(f"  {it:>4} {row}")
    # first vs last-third share, per source
    if len(iters) >= 6:
        third = max(1, len(iters) // 3)
        early, late = iters[:third], iters[-third:]
        def _share(its):
            tot = sum(by_it.get(i, {}).get(s, 0) for i in its for s in sources) or 1
            return {s: sum(by_it.get(i, {}).get(s, 0) for i in its) / tot for s in sources}
        es, ls = _share(early), _share(late)
        print("\n  source share drift (first third → last third):")
        for s in sources:
            d = ls[s] - es[s]
            arrow = "↑" if d > 0.02 else ("↓" if d < -0.02 else "·")
            print(f"    {s:22} {es[s]*100:5.1f}% → {ls[s]*100:5.1f}%  {arrow}")

    # Section 2 — contrastive attribution (with loud caveats).
    print("\nContrastive attribution (mean cond_gap in iters INCLUDING vs EXCLUDING a")
    print("source's shards). Trust gate: ≥%d attributed shards. ubiquity≈1.0 means the" % SOURCE_TRUST_FLOOR)
    print("source is in nearly every iteration → excluded baseline thin → unreliable.\n")
    print(f"  {'source':22} {'shards':>7} {'attr':>5} {'attr_cond_gap':>14} {'ubiq':>5}  trust")
    any_separable = False
    for r in rollup:
        acg = r["attr_cond_gap_mean"]
        std = r["attr_cond_gap_std"]
        cell = "—" if acg is None else f"{acg:+.4f}±{std:.4f}"
        sep = (r["trust"] == "trusted" and r["ubiquity"] < 0.85
               and acg is not None and abs(acg) > (std or 0))
        any_separable = any_separable or sep
        flag = "  ★ separable" if sep else ""
        print(f"  {r['source']:22} {r['n_shards']:>7} {r['n_attributed']:>5} "
              f"{cell:>14} {r['ubiquity']:>5}  {r['trust']}{flag}")

    # Section 3 — honest interpretation.
    print("\nInterpretation:")
    if any_separable:
        best = max((r for r in rollup if r["attr_cond_gap_mean"] is not None),
                   key=lambda r: r["attr_cond_gap_mean"])
        print(f"  • {best['source']} leads on contrastive cond_gap "
              f"({best['attr_cond_gap_mean']:+.4f}) and is separable — candidate to "
              f"RAISE in per_source_min / the production recipe.")
    else:
        print("  • No source is separable yet. Every source is near-ubiquitous "
              "(in ~all iters),")
        print("    so the campaign is not A/B-ing sources — per_source_min + pool "
              "dominance keep")
        print("    them all in every iteration. Contrastive cond_gap can't separate "
              "what is never")
        print("    varied. Watch the SELECTION-MIX drift above for the meta-flywheel's "
              "own preference,")
        print("    and revisit once exploration produces source-poor iterations (or "
              "via an ablation")
        print("    arm that holds a source out).")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
