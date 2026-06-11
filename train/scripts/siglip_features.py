#!/usr/bin/env python
"""
siglip_features.py — one image -> raw f32 SigLIP features for `iris --ip-features`
(SREF-3 sidecar).

The C engine consumes image conditioning as a precomputed features file
(raw float32 [729, 1152], no header — see main.c --ip-features) until G-1
Phase 3 brings SigLIP in-process. This script is the canonical producer:
the web server shells out to it per style-reference image (content-hash
cached), and it doubles as the manual CLI for testing `iris --ip`.

Uses the SAME model + preprocessing as training precompute
(google/siglip-so400m-patch14-384 via mlx_vlm, _preprocess_siglip) so the
features match what the adapter was trained on. UNQUANTIZED — inference
should not inherit the 4-bit training-cache compression.

Usage:
  train/.venv/bin/python train/scripts/siglip_features.py IMAGE --out FEATURES.bin
Prints a JSON line {"out":..., "tokens":729, "dim":1152} on success (parsed
by web/server.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

SIGLIP_MODEL = "google/siglip-so400m-patch14-384"
TOKENS, DIM = 729, 1152


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", help="input image (jpg/png)")
    ap.add_argument("--out", required=True, help="output raw f32 .bin path")
    args = ap.parse_args()

    raw = Path(args.image).read_bytes()
    from precompute_all import _preprocess_siglip
    img = _preprocess_siglip(raw)                       # [1,3,384,384] f32

    import mlx.core as mx
    from mlx_vlm import load as vlm_load
    model, _ = vlm_load(SIGLIP_MODEL)
    model.eval()
    feats = model.vision_model(mx.array(img))           # [1,729,1152]
    mx.eval(feats)
    feats_np = np.array(feats)[0].astype(np.float32)
    if feats_np.shape != (TOKENS, DIM):
        print(f"unexpected feature shape {feats_np.shape}", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    feats_np.tofile(tmp)
    tmp.rename(out)
    print(json.dumps({"out": str(out), "tokens": TOKENS, "dim": DIM}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
