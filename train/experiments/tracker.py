"""
train/experiments/tracker.py — experiment orchestration + CLI (v3.21.0).

Thin layer over ExperimentRegistry that:
  - creates an experiment record from a finished campaign (pulling the
    campaign's best metrics from FlywheelDB + the run's proxy fallback rate);
  - renders the Champion/Challenger report and pairwise comparisons.

CLI:
    python -m experiments.tracker list
    python -m experiments.tracker report --metric clip_i
    python -m experiments.tracker champion --metric clip_i
    python -m experiments.tracker compare exp_0001 exp_0002
    python -m experiments.tracker promote --metric clip_i --min-margin 0.01

Pure: registry I/O only. No GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from experiments.registry import ExperimentRegistry, GOLDEN_METRICS, _LOWER_IS_BETTER


# ---------------------------------------------------------------------------
# Record creation from a finished campaign
# ---------------------------------------------------------------------------

def record_from_campaign(
    registry: ExperimentRegistry,
    campaign: str,
    weights_path: Optional[str] = None,
    proxy_enabled: bool = False,
    proxy_mode: Optional[str] = None,
    proxy_fallback_rate: Optional[float] = None,
    flywheel_db=None,
    git_sha: str = "",
    manifest: Optional[str] = None,
    notes: str = "",
) -> str:
    """Register an experiment for a finished campaign.

    When a FlywheelDB is provided, the campaign's best cond_gap / ref_gap /
    train_loss / total_steps + best_hyperparams are pulled from it so the
    experiment record carries the training outcome without re-deriving it.
    """
    hyperparams = None
    cond_gap = ref_gap = train_loss = None
    total_steps = None

    if flywheel_db is not None:
        try:
            best = flywheel_db.get_best(campaign)
            if best:
                cond_gap   = best.get("cond_gap")
                ref_gap    = best.get("ref_gap")
                train_loss = best.get("train_loss")
                total_steps = best.get("steps")
            hp = flywheel_db.get_best_hyperparams(campaign)
            if hp:
                hyperparams = hp
        except Exception:
            pass  # best-effort enrichment

    return registry.register(
        campaign=campaign, weights_path=weights_path, manifest=manifest,
        git_sha=git_sha, proxy_enabled=proxy_enabled, proxy_mode=proxy_mode,
        proxy_fallback_rate=proxy_fallback_rate, hyperparams=hyperparams,
        train_loss=train_loss, cond_gap=cond_gap, ref_gap=ref_gap,
        total_steps=total_steps, notes=notes,
    )


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def format_report(registry: ExperimentRegistry, metric: str = "clip_i") -> str:
    """Champion/Challenger table ordered by a golden metric."""
    ranked = registry.rank(metric)
    if not ranked:
        return f"No evaluated experiments (metric={metric}). Run golden-set eval first."

    col = f"golden_{metric}"
    arrow = "↓ lower=better" if metric in _LOWER_IS_BETTER else "↑ higher=better"
    lines = [f"Experiment ranking by golden {metric} ({arrow}):", ""]
    lines.append(f"  {'rank':<5}{'id':<10}{'status':<11}{'campaign':<18}"
                 f"{metric:<10}{'cond_gap':<10}{'fb_rate':<8}{'proxy':<8}")
    lines.append("  " + "-" * 78)
    for i, e in enumerate(ranked, 1):
        crown = "★" if e["status"] == "champion" else " "
        lines.append(
            f"  {crown}{i:<4}{e['id']:<10}{e['status']:<11}"
            f"{(e.get('campaign') or '—')[:17]:<18}"
            f"{_fmt(e.get(col)):<10}{_fmt(e.get('cond_gap')):<10}"
            f"{_fmt(e.get('proxy_fallback_rate'), 3):<8}"
            f"{e.get('proxy_mode') or '—':<8}"
        )
    return "\n".join(lines)


def format_compare(registry: ExperimentRegistry, id_a: str, id_b: str) -> str:
    cmp = registry.compare(id_a, id_b)
    lines = [
        f"Compare {id_a} ({cmp['campaign_a']}) vs {id_b} ({cmp['campaign_b']}):", "",
        f"  {'metric':<16}{id_a:<12}{id_b:<12}{'delta':<12}{'winner':<10}",
        "  " + "-" * 60,
    ]
    for col, e in cmp["fields"].items():
        if e["a"] is None and e["b"] is None:
            continue
        name = col.replace("golden_", "")
        winner = e["better"] or "—"
        lines.append(f"  {name:<16}{_fmt(e['a']):<12}{_fmt(e['b']):<12}"
                     f"{_fmt(e['delta']):<12}{winner:<10}")
    # Overall tally
    wins = cmp["winner_by_metric"]
    a_wins = sum(1 for w in wins.values() if w == id_a)
    b_wins = sum(1 for w in wins.values() if w == id_b)
    lines += ["", f"  Tally: {id_a}={a_wins}  {id_b}={b_wins}  "
              f"({'tie' if a_wins == b_wins else (id_a if a_wins > b_wins else id_b)} leads)"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment tracker (v3.21.0)")
    ap.add_argument("--db", default=None, help="experiments.db path (default: hot volume)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List all experiments")
    p_rep = sub.add_parser("report", help="Champion/Challenger ranking")
    p_rep.add_argument("--metric", default="clip_i", choices=list(GOLDEN_METRICS))
    p_ch = sub.add_parser("champion", help="Show the current champion")
    p_ch.add_argument("--metric", default="clip_i", choices=list(GOLDEN_METRICS))
    p_pr = sub.add_parser("promote", help="Recompute champion/challenger statuses")
    p_pr.add_argument("--metric", default="clip_i", choices=list(GOLDEN_METRICS))
    p_pr.add_argument("--min-margin", type=float, default=0.0)
    p_cmp = sub.add_parser("compare", help="Compare two experiments")
    p_cmp.add_argument("id_a")
    p_cmp.add_argument("id_b")

    args = ap.parse_args()
    reg = ExperimentRegistry(db_path=Path(args.db) if args.db else None)

    if args.cmd == "list":
        rows = reg.list()
        if not rows:
            print("No experiments registered.")
            return
        print(f"{'id':<10}{'status':<12}{'campaign':<20}{'clip_i':<9}{'proxy':<8}")
        for r in rows:
            print(f"{r['id']:<10}{r['status']:<12}{(r.get('campaign') or '—')[:19]:<20}"
                  f"{_fmt(r.get('golden_clip_i')):<9}{r.get('proxy_mode') or '—':<8}")
    elif args.cmd == "report":
        print(format_report(reg, args.metric))
    elif args.cmd == "champion":
        c = reg.champion(args.metric)
        print(f"No champion (no evaluated experiments)." if not c
              else f"Champion by {args.metric}: {c['id']} ({c.get('campaign')})  "
                   f"{args.metric}={_fmt(c.get('golden_' + args.metric))}")
    elif args.cmd == "promote":
        cid = reg.promote_champion(args.metric, args.min_margin)
        print(f"Champion: {cid}" if cid else "Nothing to promote.")
    elif args.cmd == "compare":
        print(format_compare(reg, args.id_a, args.id_b))


if __name__ == "__main__":
    main()
