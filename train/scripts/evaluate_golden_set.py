#!/usr/bin/env python3
"""
train/scripts/evaluate_golden_set.py — golden-set quality validation (v3.21.0 phase 3).

Proves the proxy VAE does not degrade final model quality, via a controlled
3-arm downstream A/B on a fixed, stratified "Golden Set" (~2000–5000 images):

  arm "real"           — real Flux VAE latents
  arm "proxy_fallback" — proxy with confidence fallback (the production path)
  arm "proxy_forced"   — proxy with NO fallback (worst case / pure-proxy ceiling)

Each arm trains a short, identical IP-Adapter (same shards/seed/steps, differing
only in the VAE latents) and is scored on CLIP-I, CLIP-T, aesthetic, LPIPS, FID.
Results are written to the experiment registry and the monitoring trend store,
and a REGRESSION GATE decides whether the proxy is safe — auto-disabling it for
future runs if it degrades any key metric beyond tolerance.

The training + scoring arms need the GPU (run when the pipeline is idle). The
regression-gate decision and the registry/trend integration are pure and tested
(test_golden_gate.py).

Usage:
    python train/scripts/evaluate_golden_set.py \\
        --proxy /Volumes/2TBSSD/checkpoints/vae_proxy/proxy_final.safetensors \\
        --golden-manifest /Volumes/16TBCold/campaigns/golden/manifest.json \\
        --config train/configs/v2_pipeline.yaml \\
        --steps 500 --seed 1234 --campaign golden-eval-2026-06

See: compare_downstream_quality.py (the 2-arm primitive this extends),
     plans/proxy-vae-v3.19-migration.md.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


# Metrics that gate the proxy. Higher-is-better except lpips/fid.
KEY_METRICS = ("clip_i", "clip_t", "aesthetic", "lpips", "fid")
_LOWER_IS_BETTER = {"lpips", "fid"}


# ---------------------------------------------------------------------------
# Regression gate (pure) — the trustworthiness decision
# ---------------------------------------------------------------------------

def regression_gate(
    arm_metrics: dict,
    tolerance: float = 0.03,
    key_metrics=KEY_METRICS,
    reference_arm: str = "real",
    proxy_arm: str = "proxy_fallback",
) -> dict:
    """Decide whether the proxy arm is within tolerance of the real-VAE arm.

    arm_metrics: {arm_name: {metric: value}}.
    tolerance:   max allowed RELATIVE degradation vs the reference arm (0.03 = 3%).

    For each key metric present in both arms, computes the relative degradation
    (signed so that a worse proxy is positive) and flags a failure when it
    exceeds tolerance. Returns:
        {passed, tolerance, reference_arm, proxy_arm,
         per_metric: {m: {ref, proxy, rel_degradation, failed}},
         failures: [m, ...], recommend_disable: bool}
    """
    ref   = arm_metrics.get(reference_arm, {}) or {}
    proxy = arm_metrics.get(proxy_arm, {}) or {}

    per_metric = {}
    failures = []
    for m in key_metrics:
        rv, pv = ref.get(m), proxy.get(m)
        if rv is None or pv is None:
            continue
        # Relative degradation, signed so >0 means proxy is worse.
        if m in _LOWER_IS_BETTER:
            # lower is better → proxy worse when pv > rv
            denom = abs(rv) if rv != 0 else 1.0
            rel = (pv - rv) / denom
        else:
            # higher is better → proxy worse when pv < rv
            denom = abs(rv) if rv != 0 else 1.0
            rel = (rv - pv) / denom
        failed = rel > tolerance
        per_metric[m] = {
            "ref": rv, "proxy": pv,
            "rel_degradation": round(rel, 6), "failed": failed,
        }
        if failed:
            failures.append(m)

    return {
        "passed": len(failures) == 0,
        "tolerance": tolerance,
        "reference_arm": reference_arm,
        "proxy_arm": proxy_arm,
        "per_metric": per_metric,
        "failures": failures,
        "recommend_disable": len(failures) > 0,
    }


def format_gate(gate: dict) -> str:
    """Human-readable gate report."""
    head = ("PASS — proxy within tolerance" if gate["passed"]
            else f"FAIL — proxy degrades {', '.join(gate['failures'])}")
    lines = [f"Regression gate ({gate['proxy_arm']} vs {gate['reference_arm']}, "
             f"tol {gate['tolerance']:.0%}): {head}", ""]
    lines.append(f"  {'metric':<12}{'real':<10}{'proxy':<10}{'rel_deg':<10}{'status':<6}")
    lines.append("  " + "-" * 48)
    for m, d in gate["per_metric"].items():
        status = "✗" if d["failed"] else "✓"
        lines.append(f"  {m:<12}{d['ref']:<10.4g}{d['proxy']:<10.4g}"
                     f"{d['rel_degradation']*100:<9.2f}% {status}")
    if gate["recommend_disable"]:
        lines += ["", "  → RECOMMEND: disable proxy (set proxy_vae.enabled=false) "
                  "until retrained."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Results integration (pure) — write to registry + trend store
# ---------------------------------------------------------------------------

def write_results(arm_metrics: dict, gate: dict, campaign: str,
                  proxy_path: Optional[str] = None,
                  proxy_fallback_rate: Optional[float] = None,
                  registry=None, trends=None) -> Optional[str]:
    """Record golden-set results: a new experiment + champion-metric trend point.

    Returns the experiment id (or None if no registry given).
    """
    exp_id = None
    if registry is not None:
        exp_id = registry.register(
            campaign=campaign, weights_path=proxy_path,
            proxy_enabled=True, proxy_mode="balanced",
            proxy_fallback_rate=proxy_fallback_rate,
            notes="golden-set 3-arm evaluation",
        )
        registry.attach_golden(exp_id, arm_metrics, headline_arm=gate["proxy_arm"])

    if trends is not None:
        head = arm_metrics.get(gate["proxy_arm"], {})
        if head.get("clip_i") is not None:
            trends.record("champion_clip_i", head["clip_i"], campaign=campaign,
                          meta={"experiment": exp_id, "gate_passed": gate["passed"]})
        if proxy_fallback_rate is not None:
            trends.record("proxy_fallback_rate", proxy_fallback_rate, campaign=campaign)

    return exp_id


def maybe_disable_proxy(gate: dict, config_path: Optional[str]) -> bool:
    """If the gate recommends disabling, set proxy_vae.enabled=false in the config.

    Returns True if the config was modified. Safe no-op when the gate passes or no
    config path is given.
    """
    if not gate["recommend_disable"] or not config_path:
        return False
    import yaml
    p = Path(config_path)
    cfg = yaml.safe_load(p.read_text())
    pv = cfg.setdefault("proxy_vae", {})
    if pv.get("enabled"):
        pv["enabled"] = False
        pv["_auto_disabled_reason"] = (
            f"golden-set regression: {', '.join(gate['failures'])} "
            f"degraded > {gate['tolerance']:.0%}")
        p.write_text(yaml.dump(cfg, default_flow_style=False))
        return True
    return False


# ---------------------------------------------------------------------------
# 3-arm runner (GPU-gated orchestration) — extends compare_downstream_quality
# ---------------------------------------------------------------------------

def run_golden_eval(args) -> dict:
    """Orchestrate the 3 training+scoring arms, then gate + record.

    GPU-gated: each arm trains a short IP-Adapter and scores it. Refuses to run
    while the GPU lock is held (it launches training) unless --force.
    """
    from pipeline_lib import gpu_lock_holder

    holder = gpu_lock_holder()
    if holder and not args.force:
        print(f"ERROR: GPU lock held by '{holder.get('label','?')}'. The golden "
              f"evaluation launches training — run on an idle GPU or pass --force.",
              file=sys.stderr)
        sys.exit(1)

    # NOTE: the per-arm training + scoring is delegated to the same machinery as
    # compare_downstream_quality.py (precompute the golden shards per arm, train,
    # then score with CLIP-I/T + aesthetic + LPIPS). That path is GPU-bound and is
    # exercised when the pipeline is idle; this function wires the 3 arms, the
    # gate, and the registry/trend recording around it.
    raise NotImplementedError(
        "run_golden_eval requires an idle GPU and the per-arm training/scoring "
        "harness (shared with compare_downstream_quality.py). The pure regression "
        "gate + result recording are available via regression_gate()/write_results() "
        "and are unit-tested. Wire the arm execution when the pipeline is idle."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Golden-set 3-arm proxy quality validation")
    ap.add_argument("--proxy", required=True)
    ap.add_argument("--golden-manifest", required=True)
    ap.add_argument("--config", default="train/configs/v2_pipeline.yaml")
    ap.add_argument("--campaign", default="golden-eval")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tolerance", type=float, default=0.03)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    result = run_golden_eval(args)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
