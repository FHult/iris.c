#!/usr/bin/env python3
"""
train/scripts/validator.py — Post-training validation entry point (MLX-22).

V-01: weight integrity (inline — no external script required).
V-02/03/04: inference + CLIP scoring (not yet implemented for V2; skipped with WARN).
V-07: visual grid (not yet implemented for V2; skipped with WARN).

Exit codes:
    0 — PASS or WARN (orchestrator may proceed)
    1 — FAIL (orchestrator escalates to dispatch)
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_lib import (
    DATA_ROOT, LOG_DIR,
    mark_done, mark_error, log_event, log_orch, dispatch_issue,
    now_iso,
)


# ---------------------------------------------------------------------------
# V-01: weight integrity — inline, no subprocess required
# ---------------------------------------------------------------------------

def run_v01(checkpoint: str) -> dict:
    """Load the checkpoint and verify no NaN/Inf values and no zero weight tensors."""
    log_orch("V-01: checking weight integrity")
    errors: list[str] = []
    warnings: list[str] = []

    try:
        import mlx.core as mx
        weights = mx.load(checkpoint)
    except Exception as e:
        return {"ok": False, "errors": [f"failed to load checkpoint: {e}"], "warnings": []}

    if not weights:
        return {"ok": False, "errors": ["checkpoint is empty"], "warnings": []}

    n_params = 0
    for key, tensor in weights.items():
        try:
            arr = tensor.astype(mx.float32)
            mx.eval(arr)
            flat = arr.flatten()
            n = flat.shape[0]
            n_params += n

            has_nan = bool(mx.any(mx.isnan(flat)).item())
            has_inf = bool(mx.any(mx.isinf(flat)).item())
            if has_nan:
                errors.append(f"{key}: contains NaN")
            if has_inf:
                errors.append(f"{key}: contains Inf")

            # All-zero tensors on non-bias keys are suspicious
            if not has_nan and not has_inf and "bias" not in key:
                max_abs = float(mx.max(mx.abs(flat)).item())
                if max_abs == 0.0:
                    warnings.append(f"{key}: all zeros (may indicate failed initialisation)")
        except Exception as e:
            warnings.append(f"{key}: could not check ({e})")

    ok = len(errors) == 0
    log_event("validator", "v01_done", ok=ok, n_params=n_params,
              errors=errors[:5], warnings=warnings[:5])
    log_orch(f"V-01: {'PASS' if ok else 'FAIL'} — {n_params:,} params checked, "
             f"{len(errors)} errors, {len(warnings)} warnings")
    return {"ok": ok, "errors": errors, "warnings": warnings, "n_params": n_params}


# ---------------------------------------------------------------------------
# V-02/03/04: inference + CLIP scoring (not yet implemented for V2)
# ---------------------------------------------------------------------------

def run_inference_and_score(chunk: int, checkpoint: str, **_) -> dict:
    msg = ("V-02/03/04: inference + CLIP scoring not yet implemented for V2 pipeline — "
           "skipping (WARN, not FAIL)")
    log_orch(msg, level="warning")
    log_event("validator", "v03_04_skipped", chunk=chunk, reason="not_implemented_v2")
    return {"ok": True, "skipped": True, "reason": "not_implemented_v2"}


# ---------------------------------------------------------------------------
# V-07: visual grid (not yet implemented for V2)
# ---------------------------------------------------------------------------

def run_v07(val_dir: Path, **_) -> None:
    log_orch("V-07: visual grid not yet implemented for V2 — skipping", level="warning")
    log_event("validator", "v07_skipped", reason="not_implemented_v2")


# ---------------------------------------------------------------------------
# Write summary report
# ---------------------------------------------------------------------------

def _write_report(val_dir: Path, summary: dict) -> None:
    val_dir.mkdir(parents=True, exist_ok=True)
    report_path = val_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    log_orch(f"Validation report: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Post-training validator (MLX-22)")
    ap.add_argument("--chunk",        type=int, required=True)
    ap.add_argument("--checkpoint",   required=True)
    ap.add_argument("--config",       default="train/configs/v2_pipeline.yaml")
    ap.add_argument("--prompts",      default="train/configs/eval_prompts.txt")
    ap.add_argument("--prev-val-dir", default=None)
    args = ap.parse_args()

    chunk = args.chunk
    ckpt  = args.checkpoint

    val_dir  = Path(LOG_DIR) / f"val_chunk{chunk}"
    prev_dir = Path(args.prev_val_dir) if args.prev_val_dir else None

    log_orch(f"Validator starting for chunk {chunk}, checkpoint={ckpt}")
    log_event("validator", "start", chunk=chunk, checkpoint=ckpt)

    t_start = time.time()
    summary: dict = {
        "chunk":      chunk,
        "checkpoint": ckpt,
        "timestamp":  now_iso(),
        "v01":        None,
        "v03_04":     None,
        "verdict":    None,
    }

    # V-01: weight integrity (blocking)
    v01 = run_v01(ckpt)
    summary["v01"] = v01

    # V-02/03/04: inference + CLIP scoring (skipped until V2 inference is built)
    v03 = run_inference_and_score(chunk=chunk, checkpoint=ckpt,
                                  config_path=args.config,
                                  val_dir=val_dir,
                                  prompts_path=args.prompts,
                                  prev_val_dir=prev_dir)
    summary["v03_04"] = v03

    # V-07: visual grid (best-effort, non-blocking)
    try:
        run_v07(val_dir, prompts_path=args.prompts, prev_val_dir=prev_dir)
    except Exception as e:
        log_orch(f"V-07: non-fatal error: {e}")

    # Final verdict
    if not v01["ok"]:
        verdict = "FAIL"
        reason  = f"weight errors: {v01['errors'][:3]}"
    else:
        verdict = "PASS"
        skipped = [k for k, v in [("v03_04", v03)] if v.get("skipped")]
        reason  = (f"{v01['n_params']:,} params OK"
                   + (f"; skipped: {', '.join(skipped)}" if skipped else ""))

    summary["verdict"]      = verdict
    summary["reason"]       = reason
    summary["elapsed_secs"] = round(time.time() - t_start, 1)
    _write_report(val_dir, summary)

    log_event("validator", "done", chunk=chunk, verdict=verdict, reason=reason)
    log_orch(f"Validator chunk {chunk}: {verdict} — {reason}")

    if verdict == "FAIL":
        mark_error(chunk, "validate")
        dispatch_issue(
            f"val_fail_chunk{chunk}", "error",
            f"Chunk {chunk} validation FAILED: {reason}",
            chunk=chunk,
        )
        sys.exit(1)
    else:
        mark_done(chunk, "validate")
        sys.exit(0)


if __name__ == "__main__":
    main()
