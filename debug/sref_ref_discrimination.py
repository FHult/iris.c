#!/usr/bin/env python3
"""sref_ref_discrimination.py — the mode-collapse gate (SREF-CHAMPION-COLLAPSE, 2026-06-29).

A high sref_score means nothing if the adapter ignores the reference. This gate holds the
prompt, seed, and ip-scale FIXED and varies ONLY the reference, then pixel-correlates the
outputs. A real style-reference adapter produces DIFFERENT images for different references
(low cross-ref corr). A mode-collapsed adapter applies a near-constant transform regardless
of reference (cross-ref corr ~0.99) — that's a FALSE-POSITIVE sref_score.

Discovery numbers (champion clean_concentrate_leak @0.38, 512px): 7 wildly different refs
(5 WikiArt paintings + line-art + sticker) → cross-ref output corr >= 0.984, while each vs
the no-adapter baseline was only ~0.375. i.e. strong but reference-INDEPENDENT injection.

Run this on any candidate BEFORE trusting an A/B. PASS = cross-ref corr below --max-corr
(default 0.90) AND clearly above the no-adapter floor (the adapter does *something*).

Usage:
  debug/sref_ref_discrimination.py --bundle BUNDLE \
      --feat refA.bin refB.bin refC.bin [refD.bin ...] \
      [--scale 0.38] [--seed 42] [--size 512] [--prompt "a cat sitting on a chair"] \
      [--model flux-klein-model] [--max-corr 0.90]

Needs >= 3 features (more is better; mix in/out-of-distribution styles). Exit 0 = discriminates,
1 = mode-collapse / error.
"""
import argparse
import itertools
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def gen(model, bundle, feat, scale, prompt, seed, size, out):
    cmd = [str(ROOT / "iris"), "-d", model, "--ip", bundle, "--ip-features", str(feat),
           "--ip-scale", str(scale), "-p", prompt, "-S", str(seed),
           "-W", str(size), "-H", str(size), "-o", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not Path(out).exists():
        sys.stderr.write(f"gen failed ({Path(feat).name}):\n{r.stderr[-800:]}\n")
        return False
    return True


def flat(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--feat", nargs="+", required=True, help="reference feature .bin files (>=3)")
    ap.add_argument("--model", default="flux-klein-model")
    ap.add_argument("--prompt", default="a cat sitting on a chair")
    ap.add_argument("--scale", type=float, default=0.38)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--max-corr", type=float, default=0.90,
                    help="cross-ref output corr at/above this = mode collapse (fail). Default 0.90.")
    args = ap.parse_args()

    feats = [Path(f) for f in args.feat]
    missing = [f for f in feats if not f.exists()]
    if missing:
        sys.stderr.write(f"missing features: {missing}\n"); return 1
    if len(feats) < 3:
        sys.stderr.write("need >= 3 references to measure discrimination\n"); return 1

    with tempfile.TemporaryDirectory() as td:
        # no-adapter floor (scale 0) + one styled gen per reference, identical seed/prompt/scale
        base = Path(td) / "noadapter.png"
        print(f"baseline (scale 0) ...", flush=True)
        if not gen(args.model, args.bundle, feats[0], 0.0, args.prompt, args.seed, args.size, base):
            return 1
        styled = {}
        for f in feats:
            o = Path(td) / f"{f.stem}.png"
            print(f"gen {f.stem} @scale {args.scale} ...", flush=True)
            if not gen(args.model, args.bundle, f, args.scale, args.prompt, args.seed, args.size, o):
                return 1
            styled[f.stem] = flat(o)
        b = flat(base)

        names = list(styled)
        cross = [float(np.corrcoef(styled[x], styled[y])[0, 1]) for x, y in itertools.combinations(names, 2)]
        vs_base = [float(np.corrcoef(styled[n], b)[0, 1]) for n in names]
        mean_cross, max_cross = float(np.mean(cross)), float(np.max(cross))
        mean_base = float(np.mean(vs_base))
        print(f"\ncross-reference output corr: mean {mean_cross:.3f}  max {max_cross:.3f}  (n_refs={len(names)})")
        print(f"styled-vs-no-adapter corr:   mean {mean_base:.3f}")
        collapsed = max_cross >= args.max_corr
        inert = mean_base > 0.95  # adapter barely changes the image at all
        if collapsed:
            print(f"FAIL: MODE COLLAPSE — refs produce ~identical outputs (max cross-corr "
                  f"{max_cross:.3f} >= {args.max_corr}). sref_score is a false positive.")
            return 1
        if inert:
            print(f"FAIL: adapter is near-inert (styled ~= no-adapter, corr {mean_base:.3f}).")
            return 1
        print(f"PASS: references discriminate (max cross-corr {max_cross:.3f} < {args.max_corr}) "
              f"and the adapter transforms the image (vs-baseline {mean_base:.3f}).")
        return 0


if __name__ == "__main__":
    sys.exit(main())
