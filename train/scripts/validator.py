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
import subprocess
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
# V-02/03/04: inference + CLIP scoring
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).parent


def _run_subprocess(label: str, cmd: list[str], timeout: int = 600) -> tuple[bool, str]:
    """Run a subprocess, stream stdout, return (success, stderr_tail)."""
    log_orch(f"{label}: {' '.join(cmd[:4])} ...")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "")[-500:]
            log_orch(f"{label}: FAILED (exit {result.returncode}): {tail}", level="warning")
            return False, tail
        return True, ""
    except subprocess.TimeoutExpired:
        log_orch(f"{label}: timed out after {timeout}s", level="warning")
        return False, "timeout"
    except Exception as e:
        log_orch(f"{label}: exception: {e}", level="warning")
        return False, str(e)


def run_inference_and_score(
    chunk: int,
    checkpoint: str,
    config_path: str = "train/configs/stage1_512px.yaml",
    val_dir: "Path | None" = None,
    prompts_path: str = "train/configs/eval_prompts.txt",
    prev_val_dir: "Path | None" = None,
    **_,
) -> dict:
    """Run inference with the trained adapter, compute CLIP scores, save metrics.json."""
    if val_dir is None:
        val_dir = LOG_DIR / f"val_chunk{chunk}"
    val_dir = Path(val_dir)
    val_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable

    # V-02: generate images
    infer_ok, infer_err = _run_subprocess(
        "V-02 inference",
        [python, str(_SCRIPTS_DIR / "run_inference.py"),
         "--checkpoint", checkpoint,
         "--prompts",    prompts_path,
         "--output",     str(val_dir),
         "--config",     config_path],
        timeout=900,
    )
    if not infer_ok:
        log_event("validator", "v02_failed", chunk=chunk, error=infer_err)
        return {"ok": True, "skipped": True,
                "reason": f"inference failed: {infer_err[:120]}"}

    # V-03/04: compute CLIP scores
    scores_path = val_dir / "scores.json"
    score_ok, score_err = _run_subprocess(
        "V-03/04 CLIP scoring",
        [python, str(_SCRIPTS_DIR / "score_validation.py"),
         "--images",  str(val_dir),
         "--prompts", prompts_path,
         "--output",  str(scores_path)],
        timeout=300,
    )
    if not score_ok:
        log_event("validator", "v03_04_failed", chunk=chunk, error=score_err)
        return {"ok": True, "skipped": True,
                "reason": f"CLIP scoring failed: {score_err[:120]}"}

    try:
        scores = json.loads(scores_path.read_text())
    except Exception as e:
        log_orch(f"V-03/04: could not read scores JSON: {e}", level="warning")
        return {"ok": True, "skipped": True, "reason": f"scores unreadable: {e}"}

    verdict_info = scores.get("verdict", {})
    mean_clip_i = verdict_info.get("mean_clip_i")
    mean_delta  = verdict_info.get("mean_adapter_delta")

    # Build metrics.json: scores + optional delta vs previous chunk
    metrics: dict = {
        "ts":            now_iso(),
        "chunk":         chunk,
        "mean_clip_i":   mean_clip_i,
        "mean_clip_t":   verdict_info.get("mean_clip_t"),
        "mean_adapter_delta": mean_delta,
        "verdict":       verdict_info.get("verdict"),
    }

    if prev_val_dir is not None:
        prev_metrics_path = Path(prev_val_dir) / "metrics.json"
        if prev_metrics_path.exists():
            try:
                prev = json.loads(prev_metrics_path.read_text())
                prev_clip_i = prev.get("mean_clip_i")
                if mean_clip_i is not None and prev_clip_i is not None:
                    metrics["clip_i_delta_vs_prev"] = round(mean_clip_i - prev_clip_i, 4)
            except Exception:
                pass

    metrics_path = val_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    log_event("validator", "v03_04_done", chunk=chunk,
              mean_clip_i=mean_clip_i, mean_adapter_delta=mean_delta)
    log_orch(f"V-03/04: CLIP-I={mean_clip_i}  delta={mean_delta}  "
             f"verdict={verdict_info.get('verdict')}")

    clip_i_delta = metrics.get("clip_i_delta_vs_prev")
    if clip_i_delta is not None and clip_i_delta < -0.05:
        log_orch(
            f"V-03/04: WARNING — CLIP-I dropped {clip_i_delta:+.3f} vs chunk {chunk - 1}. "
            "Quality regression possible.",
            level="warning"
        )

    return {
        "ok":      verdict_info.get("verdict") != "FAIL",
        "skipped": False,
        "mean_clip_i":           mean_clip_i,
        "mean_adapter_delta":    mean_delta,
        "clip_i_delta_vs_prev":  clip_i_delta,
        "verdict":               verdict_info.get("verdict"),
        "metrics_path":          str(metrics_path),
    }


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
    ap.add_argument("--ai",           action="store_true",
                    help="Emit compact JSON to stdout only; all progress goes to stderr")
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
    elif not v03.get("ok", True):
        verdict = "FAIL"
        reason  = f"CLIP scoring FAIL: {v03.get('verdict')}"
    else:
        verdict = "PASS"
        parts = [f"{v01['n_params']:,} params OK"]
        if v03.get("skipped"):
            parts.append(f"CLIP skipped: {v03.get('reason', '?')[:60]}")
        elif v03.get("mean_clip_i") is not None:
            parts.append(f"CLIP-I={v03['mean_clip_i']} delta={v03.get('mean_adapter_delta')}")
            delta_prev = v03.get("clip_i_delta_vs_prev")
            if delta_prev is not None:
                parts.append(f"vs-prev={delta_prev:+.3f}")
        reason = "; ".join(parts)

    summary["verdict"]      = verdict
    summary["reason"]       = reason
    summary["elapsed_secs"] = round(time.time() - t_start, 1)
    _write_report(val_dir, summary)

    log_event("validator", "done", chunk=chunk, verdict=verdict, reason=reason)
    log_orch(f"Validator chunk {chunk}: {verdict} — {reason}")

    if args.ai:
        v03 = summary.get("v03_04") or {}
        v01 = summary.get("v01") or {}
        issues = list(v01.get("errors", [])) + list(v01.get("warnings", []))
        if not v03.get("ok", True) and v03.get("verdict"):
            issues.append(f"clip: {v03['verdict']}")
        ai_out = {
            "passed": verdict != "FAIL",
            "verdict": verdict,
            "reason": reason,
            "clip_i_mean": v03.get("mean_clip_i"),
            "weight_ok": v01.get("ok", False),
            "issues": issues,
            "chunk": chunk,
            "elapsed_secs": summary["elapsed_secs"],
        }
        print(json.dumps(ai_out))

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
