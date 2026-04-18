#!/usr/bin/env python3
"""
train/scripts/validator.py — Post-training validation entry point (MLX-22).

Runs V-01 (weight integrity), V-02/03/04 (CLIP-I/CLIP-T via inference), V-07 (visual grid).
Writes validation.done or validation.error sentinel for the orchestrator.

Usage:
    python train/scripts/validator.py \
        --chunk 1 \
        --checkpoint /Volumes/2TBSSD/checkpoints/stage1/step_050000.safetensors \
        --config train/configs/v2_pipeline.yaml \
        [--prev-val-dir /tmp/val_chunk0]   # for V-08 regression vs prev chunk

Exit codes:
    0 — PASS or WARN (orchestrator may proceed)
    1 — FAIL (orchestrator escalates to dispatch)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_lib import (
    DATA_ROOT, LOG_DIR, SENTINEL_DIR, CKPT_DIR,
    mark_done, mark_error, log_event, log_orch, dispatch_issue,
    load_config, now_iso,
)


# ---------------------------------------------------------------------------
# Run a sub-script with timeout, capture stdout/stderr
# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: int = 600) -> tuple[int, str, str]:
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=timeout,
        )
        elapsed = time.time() - t0
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"timed out after {timeout}s"
    except Exception as e:
        return -1, "", str(e)


SCRIPTS_DIR = Path(__file__).parent
PYTHON = sys.executable


# ---------------------------------------------------------------------------
# V-01: weight integrity
# ---------------------------------------------------------------------------

def run_v01(checkpoint: str) -> dict:
    log_orch("V-01: checking weight integrity")
    rc, out, err = _run([
        PYTHON, str(SCRIPTS_DIR / "validate_weights.py"),
        "--checkpoint", checkpoint,
    ], timeout=60)
    ok = rc == 0
    errors: list[str] = []
    for line in out.splitlines():
        if "ERROR:" in line:
            errors.append(line.strip())
    log_event("validator", "v01_done", ok=ok, errors=errors)
    return {"ok": ok, "errors": errors, "stdout": out, "stderr": err}


# ---------------------------------------------------------------------------
# V-03/04: inference + CLIP scoring
# ---------------------------------------------------------------------------

def run_inference_and_score(
    chunk: int,
    checkpoint: str,
    config_path: str,
    val_dir: Path,
    prompts_path: str,
    prev_val_dir: Path = None,
) -> dict:
    val_dir.mkdir(parents=True, exist_ok=True)

    # Inference
    log_orch(f"V-02/03/04: running inference for chunk {chunk}")
    rc, out, err = _run([
        PYTHON, str(SCRIPTS_DIR / "run_inference.py"),
        "--checkpoint", checkpoint,
        "--prompts",    prompts_path,
        "--output",     str(val_dir),
        "--config",     config_path,
    ], timeout=900)

    if rc != 0:
        log_event("validator", "inference_failed", chunk=chunk, rc=rc)
        return {"ok": False, "reason": f"inference failed (rc={rc})", "stderr": err}

    log_event("validator", "inference_done", chunk=chunk)

    # CLIP scoring
    scores_path = str(val_dir / "scores.json")
    log_orch(f"V-03/04: CLIP scoring for chunk {chunk}")
    rc, out, err = _run([
        PYTHON, str(SCRIPTS_DIR / "score_validation.py"),
        "--images",  str(val_dir),
        "--prompts", prompts_path,
        "--output",  scores_path,
    ], timeout=300)

    if not os.path.exists(scores_path):
        return {"ok": False, "reason": "score_validation.py produced no output"}

    with open(scores_path) as f:
        scores = json.load(f)

    log_event("validator", "scoring_done", chunk=chunk,
              verdict=scores["verdict"].get("verdict"),
              mean_clip_i=scores["verdict"].get("mean_clip_i"))
    return {"ok": rc == 0, "scores": scores}


# ---------------------------------------------------------------------------
# V-07: visual grid
# ---------------------------------------------------------------------------

def run_v07(val_dir: Path, prompts_path: str, prev_val_dir: Path = None) -> None:
    grid_path = str(val_dir / "validation_grid.png")
    cmd = [
        PYTHON, str(SCRIPTS_DIR / "render_validation_grid.py"),
        "--images",  str(val_dir),
        "--prompts", prompts_path,
        "--output",  grid_path,
    ]
    if prev_val_dir:
        cmd += ["--prev-images", str(prev_val_dir)]
    rc, out, err = _run(cmd, timeout=120)
    if rc == 0:
        log_orch(f"V-07: grid written to {grid_path}")
    else:
        log_orch(f"V-07: grid render failed (rc={rc}): {err[:200]}")


# ---------------------------------------------------------------------------
# Write summary report
# ---------------------------------------------------------------------------

def _write_report(val_dir: Path, summary: dict) -> None:
    report_path = val_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    log_orch(f"Validation report: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Post-training validator (MLX-22)")
    ap.add_argument("--chunk",       type=int, required=True)
    ap.add_argument("--checkpoint",  required=True)
    ap.add_argument("--config",      default="train/configs/v2_pipeline.yaml")
    ap.add_argument("--prompts",     default="train/configs/eval_prompts.txt")
    ap.add_argument("--prev-val-dir", default=None,
                    help="Previous chunk's val dir for regression comparison (V-08)")
    args = ap.parse_args()

    chunk    = args.chunk
    ckpt     = args.checkpoint
    cfg_path = args.config

    val_dir  = Path(LOG_DIR) / f"val_chunk{chunk}"
    prev_dir = Path(args.prev_val_dir) if args.prev_val_dir else None

    log_orch(f"Validator starting for chunk {chunk}, checkpoint={ckpt}")
    log_event("validator", "start", chunk=chunk, checkpoint=ckpt)

    t_start = time.time()
    summary: dict = {
        "chunk": chunk,
        "checkpoint": ckpt,
        "timestamp": now_iso(),
        "v01": None,
        "v03_04": None,
        "verdict": None,
    }

    # V-01
    v01 = run_v01(ckpt)
    summary["v01"] = v01

    # V-02/03/04 (inference + CLIP scoring)
    v03 = run_inference_and_score(
        chunk=chunk,
        checkpoint=ckpt,
        config_path=cfg_path,
        val_dir=val_dir,
        prompts_path=args.prompts,
        prev_val_dir=prev_dir,
    )
    summary["v03_04"] = v03

    # V-07 (visual grid) — best-effort, non-blocking
    try:
        run_v07(val_dir, args.prompts, prev_dir)
    except Exception as e:
        log_orch(f"V-07: non-fatal error: {e}")

    # Determine final verdict
    if not v01["ok"]:
        verdict = "FAIL"
        reason  = f"weight errors: {v01['errors']}"
    elif not v03.get("ok"):
        verdict = "FAIL"
        reason  = v03.get("reason", "inference/scoring failed")
    else:
        scores_verdict = v03.get("scores", {}).get("verdict", {})
        verdict = scores_verdict.get("verdict", "FAIL")
        reason  = f"mean_clip_i={scores_verdict.get('mean_clip_i')} delta={scores_verdict.get('mean_adapter_delta')}"

    summary["verdict"] = verdict
    summary["reason"]  = reason
    summary["elapsed_secs"] = round(time.time() - t_start, 1)
    _write_report(val_dir, summary)

    log_event("validator", "done", chunk=chunk, verdict=verdict, reason=reason)
    log_orch(f"Validator chunk {chunk}: {verdict} — {reason}")

    if verdict == "FAIL":
        mark_error(chunk, "validate")
        dispatch_issue(
            id=f"val_fail_chunk{chunk}",
            severity="error",
            message=f"Chunk {chunk} validation FAILED: {reason}",
            chunk=chunk,
        )
        sys.exit(1)
    else:
        mark_done(chunk, "validate")
        sys.exit(0)


if __name__ == "__main__":
    main()
