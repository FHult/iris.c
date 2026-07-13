#!/usr/bin/env python3
"""dump_csd.py — reference image → 768-d L2-normalised CSD style vector as raw little-endian f32,
for the C sref joint adapter (`iris --sref-csd <file>`). SAME encoder + preprocessing as the
training cache / the Python gate (style_encoder.csd_mlx), so the C inference sees the exact vector
the adapter was trained against.

  train/.venv/bin/python train/lora/dump_csd.py --image ref.jpg --out ref.csd.f32
"""
import argparse, os, sys
import numpy as np
import mlx.core as mx
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from style_encoder.csd_mlx import CSDStyleEncoder, preprocess

CSD_WEIGHTS = "/Volumes/2TBSSD/models/csd_vit_l_style.safetensors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=CSD_WEIGHTS)
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB")
    x = preprocess(img)                                   # [3,224,224] f32, CLIP-normalised
    enc = CSDStyleEncoder(args.weights)
    style = enc.encode(np.stack([x]))                     # [1,768] L2-normalised
    v = np.array(mx.array(style).astype(mx.float32))[0].astype("<f4")
    v.tofile(args.out)
    print(f"wrote csd[{v.shape[0]}] norm={float(np.linalg.norm(v)):.4f} -> {args.out}")


if __name__ == "__main__":
    main()
