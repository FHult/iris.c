#!/usr/bin/env python3
"""
train/scripts/validate_weights.py — V-01 weight integrity check.

Verifies:
- No NaN/Inf in any tensor
- ip_scale values in plausible range [0, 10]
- Required adapter keys present

Usage:
    python train/scripts/validate_weights.py --checkpoint path/to/step_050000.safetensors
"""

import argparse
import sys
from pathlib import Path

import numpy as np


REQUIRED_KEY_PREFIXES = ["ip_proj", "resampler", "scale"]


def check_weights(checkpoint_path: str) -> dict:
    """
    Returns:
      ok        — True if no hard errors
      errors    — list of blocking problems (cause FAIL)
      warnings  — list of soft issues (cause WARN)
      stats     — {key: {shape, dtype, min, max}} summary
    """
    from safetensors import safe_open

    errors: list[str] = []
    warnings: list[str] = []
    stats: dict = {}

    # Load all tensors
    tensors: dict[str, np.ndarray] = {}
    with safe_open(checkpoint_path, framework="numpy") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    live_keys = [k for k in tensors if not k.startswith("ema.")]

    # V-01a: NaN / Inf
    for k, t in tensors.items():
        has_nan = bool(np.any(np.isnan(t)))
        has_inf = bool(np.any(np.isinf(t)))
        if has_nan or has_inf:
            errors.append(f"{k}: contains {'NaN' if has_nan else ''}{'Inf' if has_inf else ''}")
        stats[k] = {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "min": round(float(t.min()), 6),
            "max": round(float(t.max()), 6),
        }

    # V-01b: ip_scale range
    for k, t in tensors.items():
        if "scale" in k and not k.startswith("ema."):
            if np.any(np.abs(t) > 10.0):
                errors.append(f"{k}: ip_scale magnitude > 10: {t.tolist()}")
            elif np.any(t < 0):
                warnings.append(f"{k}: negative ip_scale values: {t.tolist()}")

    # V-01c: required key prefixes present
    for prefix in REQUIRED_KEY_PREFIXES:
        if not any(k.startswith(prefix) or f".{prefix}" in k for k in live_keys):
            warnings.append(f"No keys matching prefix '{prefix}' in non-EMA weights")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
        "num_keys": len(tensors),
    }


def main() -> None:
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--ai", action="store_true",
                    help="Emit compact JSON to stdout only; all output goes to stderr")
    args = ap.parse_args()

    result = check_weights(args.checkpoint)

    if args.ai:
        ai_out = {
            "passed": result["ok"],
            "issues": result["errors"] + result["warnings"],
            "num_keys": result["num_keys"],
        }
        print(json.dumps(ai_out))
        sys.exit(0 if result["ok"] else 1)

    _out = sys.stdout
    if result["errors"]:
        print("FAIL — weight integrity errors:", file=_out)
        for e in result["errors"]:
            print(f"  ERROR: {e}", file=_out)
    else:
        print("PASS — no weight integrity errors", file=_out)

    if result["warnings"]:
        for w in result["warnings"]:
            print(f"  WARN: {w}", file=_out)

    print(f"  {result['num_keys']} keys checked", file=_out)

    if args.verbose:
        for k, s in sorted(result["stats"].items()):
            print(f"  {k}: shape={s['shape']} dtype={s['dtype']} min={s['min']:.4f} max={s['max']:.4f}",
                  file=_out)

    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
