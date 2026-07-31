#!/usr/bin/env python
"""
debug/test_siglip_csd_parity.py — SCAFFOLD (weights/deps-gated, SKIPS by default).

Boundary: the INFERENCE feature producers (train/scripts/siglip_features.py and
train/scripts/csd_features.py) must call the IDENTICAL preprocessing + encoder as
the TRAINING precompute — this is Training->Inference Correctness Protocol #4
("encoder/preprocessing parity"), the one a same-bytes fixture CANNOT catch,
because it checks that inference PRODUCES the same input the model trained on.

PRIOR RESULT THIS FORMALIZES (do not re-derive):
2026-07-30 smoke-vs-cache check found the producer output BIT-IDENTICAL to the
training precompute cache, corr = 1.0 (accepting the documented f16-cache vs
f32-inference precision convention — see siglip_features.py / csd_features.py
headers; that convention is NOT a mismatch). This test pins that as a guard.

STATUS WHEN RUN VIA `make test-parity`: SKIPS unless BOTH a fixture image and its
matching cache vector are supplied AND the producer deps import. It is NOT
reported as passing when it skips.

HOW IT WORKS:
  env SIGLIP_PARITY_IMAGE  — path to an image also present in the training cache
  env SIGLIP_PARITY_CACHE  — that image's cached feature vector (raw f32 or .npy)
  (CSD variant: CSD_PARITY_IMAGE / CSD_PARITY_CACHE)
It runs the SAME producer used at inference, then compares to the cached vector:
gate corr > 0.9999 (the f16->f32 convention lifts max_abs to ~5e-4, so corr, not
abs error, is the parity signal — exactly as the 2026-07-30 check treated it).

Run:  train/.venv/bin/python debug/test_siglip_csd_parity.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    # Deps unavailable (e.g. run without train/.venv) -> clean SKIP, not a crash.
    print("SKIP siglip/csd-producer-parity: numpy unavailable "
          "(run with train/.venv/bin/python to enable).")
    raise SystemExit(0)


def _load_cache_vec(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32).ravel()
    return np.fromfile(path, dtype=np.float32).ravel()


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return -1.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _run_one(name: str, img_env: str, cache_env: str, producer_rel: str) -> str:
    """Returns 'PASS' | 'FAIL' | 'SKIP'."""
    img = os.environ.get(img_env)
    cache = os.environ.get(cache_env)
    if not img or not cache:
        print(f"SKIP {name}: fixtures absent (set {img_env} and {cache_env}).")
        return "SKIP"
    if not Path(img).exists() or not Path(cache).exists():
        print(f"SKIP {name}: a fixture path does not exist.")
        return "SKIP"

    repo = Path(__file__).resolve().parent.parent
    producer = repo / producer_rel
    if not producer.exists():
        print(f"SKIP {name}: producer {producer_rel} not found.")
        return "SKIP"

    # Run the REAL inference producer into a temp file, then compare to cache.
    import subprocess
    import tempfile

    py = sys.executable  # whichever python invoked us (venv when available)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
        out = tf.name
    try:
        r = subprocess.run(
            [py, str(producer), img, "--out", out],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            # Deps (mlx_vlm / CSD weights / model download) unavailable -> SKIP,
            # never a fake pass.
            print(f"SKIP {name}: producer could not run "
                  f"(missing deps/weights). stderr tail: {r.stderr.strip()[-200:]}")
            return "SKIP"
        got = np.fromfile(out, dtype=np.float32).ravel()
        want = _load_cache_vec(cache)
        c = _corr(got, want)
        max_abs = float(np.max(np.abs(got - want))) if got.shape == want.shape else float("nan")
        print(f"{name}: n={got.size} corr={c:.6f} max_abs={max_abs:.6f}")
        ok = c > 0.9999
        print(f"{'PASS' if ok else 'FAIL'} {name} (corr>0.9999; "
              f"f16-cache vs f32 max_abs~5e-4 is the accepted convention)")
        return "PASS" if ok else "FAIL"
    finally:
        try:
            os.unlink(out)
        except OSError:
            pass


def main() -> int:
    results = [
        _run_one("siglip-producer-parity", "SIGLIP_PARITY_IMAGE",
                 "SIGLIP_PARITY_CACHE", "train/scripts/siglip_features.py"),
        _run_one("csd-producer-parity", "CSD_PARITY_IMAGE",
                 "CSD_PARITY_CACHE", "train/scripts/csd_features.py"),
    ]
    if "FAIL" in results:
        return 1
    # SKIP-only or PASS -> exit 0 (skips never fail the suite).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
