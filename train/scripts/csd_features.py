#!/usr/bin/env python
"""
csd_features.py — one image → raw f32 CSD style vector for `iris --ip-features` (CSD adapters).

The CSD-conditioned IP-adapter (cond_mode="csd", SREF-LEAK-2) conditions on a single 768-d
content-invariant CSD style embedding, NOT SigLIP patch tokens. iris consumes it as a raw
float32 [csd_dim] file (no header — see iris.c iris_load_ip_adapter, which reads csd_dim rows).
This is the CSD analogue of siglip_features.py.

Uses the SAME CSD encoder + preprocessing as the training cache (train/style_encoder/csd_mlx.py
via style_precompute.py) so the inference vector matches what the adapter trained on. The
training cache stored f16; inference writes f32 (higher precision — consistent with the
siglip_features.py convention that inference does not inherit the training-cache compression;
the FiLM projection is linear and robust to the ~5e-4 difference).

Usage:
  train/.venv/bin/python train/scripts/csd_features.py IMAGE --out FEATURES.bin
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style_encoder.csd_mlx import CSDStyleEncoder, preprocess


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="input image (jpg/png)")
    ap.add_argument("--out", required=True, help="output raw f32 .bin path")
    ap.add_argument("--weights", default="/Volumes/2TBSSD/models/csd_vit_l_style.safetensors")
    args = ap.parse_args()

    enc = CSDStyleEncoder(args.weights)
    img = Image.open(args.image).convert("RGB")
    x = preprocess(img)                       # [3,224,224] f32, CLIP-normalised
    style = enc.encode(np.stack([x]))         # [1, 768] L2-normalised (same as the cache)
    vec = np.asarray(style)[0].astype(np.float32)
    if vec.ndim != 1:
        print(f"unexpected CSD shape {vec.shape}", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(out.parent), suffix=".bin.tmp")
    os.close(fd)
    vec.tofile(tmp)
    os.replace(tmp, out)
    print(json.dumps({"out": str(out), "dim": int(vec.shape[0])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
