"""
quality_gate.py — cross-run output-quality regression gate.

Closes the loop the ablation search can't: ablation picks hyperparameters on a
short (1000-step, cold-start) proxy; this checks whether the *long* run that
adopted them actually produced better OUTPUT than the previous long run, on a
fixed golden set — the apples-to-apples measure (clip_i/clip_t/aesthetic/lpips/fid)
independent of the training objective.

Pipeline (run at the end of a long run):
    1. evaluate_golden_set.run_golden_eval(checkpoint) → metrics  [GPU]
    2. weight_registry.register_snapshot(... validation_metrics=metrics)
    3. fetch the previous snapshot's golden metrics for this lineage
    4. compare_quality(current, previous) → verdict (REGRESSION / IMPROVED / NEUTRAL)

`compare_quality` is pure and fully unit-tested. The GPU eval + registry I/O live
in `run_quality_gate`, behind injectable callables so the glue is testable without
a model or touching the live registry.

Metric directions: clip_i / clip_t / aesthetic are higher-is-better; lpips / fid
are lower-is-better (mirrors evaluate_golden_set._LOWER_IS_BETTER).
"""

from __future__ import annotations

from typing import Callable, Optional

LOWER_IS_BETTER = frozenset({"lpips", "fid"})


def compare_quality(
    current: dict,
    previous: Optional[dict],
    *,
    lower_is_better=LOWER_IS_BETTER,
    rel_threshold: float = 0.01,
) -> dict:
    """Compare a run's golden metrics to the previous run's. Pure.

    For each metric present (non-None) in BOTH:
      - relative change |Δ| <= rel_threshold  → "neutral"
      - moved in the better direction beyond threshold → "improved"
      - moved in the worse direction beyond threshold → "regressed"
    Direction is inverted for lower-is-better metrics (lpips/fid).

    Overall verdict: REGRESSION if any metric regressed, else IMPROVED if any
    improved, else NEUTRAL (or NO_BASELINE when there's nothing to compare).
    """
    per: dict[str, dict] = {}
    any_reg = any_imp = compared = False

    for k, cur in current.items():
        prev = (previous or {}).get(k)
        if cur is None or prev is None:
            per[k] = {"cur": cur, "prev": prev, "status": "no_baseline"}
            continue
        compared = True
        delta = cur - prev
        better = (delta < 0) if k in lower_is_better else (delta > 0)
        rel = abs(delta) / (abs(prev) if prev else 1.0)
        if rel <= rel_threshold:
            status = "neutral"
        elif better:
            status = "improved"
            any_imp = True
        else:
            status = "regressed"
            any_reg = True
        per[k] = {"cur": cur, "prev": prev, "delta": delta, "rel": rel, "status": status}

    if not compared:
        verdict = "NO_BASELINE"
    elif any_reg:
        verdict = "REGRESSION"
    elif any_imp:
        verdict = "IMPROVED"
    else:
        verdict = "NEUTRAL"
    return {"per_metric": per, "verdict": verdict,
            "regressed": any_reg, "improved_any": any_imp}


def format_verdict(cmp: dict) -> str:
    lines = [f"quality gate: {cmp['verdict']}"]
    for k, d in sorted(cmp["per_metric"].items()):
        if d["status"] == "no_baseline":
            lines.append(f"  {k:<9} {d['cur']}  (no baseline)")
        else:
            mark = {"improved": "✓", "regressed": "✗", "neutral": "·"}[d["status"]]
            lines.append(f"  {k:<9} {d['prev']:.4f} → {d['cur']:.4f}  "
                         f"(Δ {d['delta']:+.4f}, {d['rel']*100:.1f}%) {mark} {d['status']}")
    return "\n".join(lines)


def _golden_metrics_from_result(result: dict, key_metrics) -> dict:
    """Extract the flat {metric: value} dict from evaluate_golden_set's result.

    Accepts either a flat metrics dict or a result with a champion/head section.
    """
    head = result.get("champion") or result.get("head") or result
    return {m: head.get(m) for m in key_metrics if head.get(m) is not None}


def run_quality_gate(
    checkpoint: str,
    campaign: str,
    *,
    golden_eval: Optional[Callable] = None,
    fetch_previous: Optional[Callable] = None,
    register: Optional[Callable] = None,
    rel_threshold: float = 0.01,
) -> dict:
    """Run the gate end-to-end. The three side-effecting steps are injectable so
    this is testable without a GPU or the live registry; defaults wire the real
    evaluate_golden_set / weight_registry.

    Returns {"metrics": current, "previous": prev, "comparison": compare_quality(...)}.
    """
    # 1. golden-set eval (GPU). Lazy import so this module stays import-light.
    if golden_eval is None:
        from evaluate_golden_set import run_golden_eval, KEY_METRICS
        result = run_golden_eval(checkpoint)
        current = _golden_metrics_from_result(result, KEY_METRICS)
    else:
        current = golden_eval(checkpoint)

    # 2. previous run's metrics for this lineage
    if fetch_previous is None:
        fetch_previous = _latest_prior_metrics
    previous = fetch_previous(campaign)

    # 3. register this run
    if register is None:
        register = _register_default
    try:
        register(checkpoint, campaign, current)
    except Exception:
        pass  # registration must never block the verdict

    # 4. compare
    cmp = compare_quality(current, previous, rel_threshold=rel_threshold)
    return {"metrics": current, "previous": previous, "comparison": cmp}


_GOLDEN_KEYS = ("clip_i", "clip_t", "aesthetic", "lpips", "fid")


def _latest_prior_metrics(campaign: str) -> Optional[dict]:
    """Most recent prior snapshot's golden metrics for this campaign.

    The registry index only summarises val_loss/cond_gap, so we walk the index
    (newest-first) and load each snapshot until we find one carrying golden
    metrics. Called BEFORE the current run is registered, so the newest match is
    the previous run.
    """
    try:
        import weight_registry as wr
        idx = wr._read_index()  # newest-first, entries have snapshot_id + campaign
    except Exception:
        return None
    for e in idx:
        if e.get("campaign") != campaign:
            continue
        snap = wr.load_snapshot(e.get("snapshot_id", ""))
        vm = (snap or {}).get("validation_metrics") or {}
        if any(vm.get(k) is not None for k in _GOLDEN_KEYS):
            return vm
    return None


def _register_default(checkpoint: str, campaign: str, metrics: dict) -> None:
    import os
    import weight_registry as wr
    snapshot_id = f"{campaign}_{os.path.splitext(os.path.basename(checkpoint))[0]}"
    wr.register_snapshot(snapshot_id=snapshot_id, campaign=campaign,
                         weights_path=checkpoint, validation_metrics=metrics)


def main(argv=None) -> int:
    """Run the gate as a step at the end of a long run.

    GPU step (golden-set eval) runs here — invoke only when the GPU is free.
    Exits non-zero on REGRESSION so it can gate a pipeline.
    """
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="trained adapter .safetensors")
    ap.add_argument("--campaign", required=True, help="lineage name for cross-run comparison")
    ap.add_argument("--rel-threshold", type=float, default=0.01,
                    help="relative change below which a metric is 'neutral' (default 1%%)")
    ap.add_argument("--fail-on-regression", action="store_true",
                    help="exit 1 if any metric regressed (for pipeline gating)")
    args = ap.parse_args(argv)

    out = run_quality_gate(args.checkpoint, args.campaign, rel_threshold=args.rel_threshold)
    print(format_verdict(out["comparison"]))
    if args.fail_on_regression and out["comparison"]["regressed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
