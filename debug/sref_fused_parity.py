#!/usr/bin/env python3
"""On-demand parity guard for the B2 fused-path IP-Adapter inject.

Renders the SAME sref generation twice — once through the fast fused bf16 path (default)
and once through the per-block path (IRIS_NO_FUSED_IP=1) — and asserts the two images match
(the fused path uses f16/bf16 for the inject vs f32 on the per-block path, so this is a
"noise not mismatch" check, not bit-identical). Needs the real model + adapter bundle, so it
is NOT part of `make test`; run it after touching the inject or the fused single-block path.

Usage:
  debug/sref_fused_parity.py \
      [--model flux-klein-model] \
      [--bundle /Volumes/2TBSSD/sref_eval/clean_concentrate_leak/bundle] \
      [--features /Volumes/2TBSSD/sref_eval/refs_feat_hybrid/artnouveau.bin] \
      [--scale 0.45] [--size 576] [--steps 4] [--seed 7] [--min-corr 0.99]

Exit 0 on parity, 1 on mismatch / error.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def render(out_path, args, force_per_block):
    env_prefix = {"IRIS_NO_FUSED_IP": "1"} if force_per_block else {}
    import os
    env = {**os.environ, **env_prefix}
    cmd = [
        str(ROOT / "iris"), "-d", args.model, "-p", args.prompt,
        "-W", str(args.size), "-H", str(args.size),
        "--steps", str(args.steps), "--seed", str(args.seed),
        "--ip", args.bundle, "--ip-features", args.features,
        "--ip-scale", str(args.scale), "-o", str(out_path),
    ]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0 or not Path(out_path).exists():
        sys.stderr.write(f"render failed (per_block={force_per_block}):\n{r.stderr[-1500:]}\n")
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="flux-klein-model")
    ap.add_argument("--bundle", default="/Volumes/2TBSSD/sref_eval/clean_concentrate_leak/bundle")
    ap.add_argument("--features", default="/Volumes/2TBSSD/sref_eval/refs_feat_hybrid/artnouveau.bin")
    ap.add_argument("--prompt", default="a cat sitting on a chair")
    ap.add_argument("--scale", type=float, default=0.45)
    ap.add_argument("--size", type=int, default=576)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-corr", type=float, default=0.99)
    ap.add_argument("--max-abs", type=float, default=20.0)  # /255
    args = ap.parse_args()

    if not Path(args.bundle).exists() or not Path(args.features).exists():
        sys.stderr.write(f"SKIP: bundle/features not found ({args.bundle}, {args.features})\n")
        return 0  # not a failure — the assets just aren't on this machine

    with tempfile.TemporaryDirectory() as td:
        fused = Path(td) / "fused.png"
        perblk = Path(td) / "perblock.png"
        print(f"rendering fused path -> {fused.name}")
        if not render(fused, args, force_per_block=False):
            return 1
        print(f"rendering per-block path (IRIS_NO_FUSED_IP=1) -> {perblk.name}")
        if not render(perblk, args, force_per_block=True):
            return 1

        a = np.asarray(Image.open(fused).convert("RGB"), dtype=np.float64)
        b = np.asarray(Image.open(perblk).convert("RGB"), dtype=np.float64)
        if a.shape != b.shape:
            sys.stderr.write(f"FAIL: shape mismatch {a.shape} vs {b.shape}\n")
            return 1
        corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        mad = float(np.abs(a - b).mean())
        mx = float(np.abs(a - b).max())
        print(f"fused vs per-block: corr={corr:.4f}  mean|Δ|={mad:.2f}/255  max|Δ|={mx:.0f}/255")
        if corr < args.min_corr or mx > args.max_abs:
            sys.stderr.write(f"FAIL: parity below threshold (corr<{args.min_corr} or max>{args.max_abs})\n")
            return 1
        print("PASS: fused-path inject matches per-block within tolerance.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
