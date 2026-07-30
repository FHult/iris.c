#!/usr/bin/env python
"""
hybrid_features.py — one image → raw f32 packed hybrid feature for `iris --ip-features`
(cond_mode="hybrid", SREF-COMBINE-1).

The hybrid IP-adapter conditions on BOTH 729 SigLIP patch tokens (local detail/texture,
via the PerceiverResampler) AND a single 768-d content-invariant CSD style vector (global
style direction, via FiLM). iris consumes them as ONE packed float32 file shaped
[730, 1152]: rows 0..728 = SigLIP features, row 729 = the CSD vector zero-padded to 1152
(see iris.c iris_load_ip_adapter / iris_ip_adapter.c perceive hybrid path, which slices the
last row's first csd_dim values back out).

To guarantee the SAME preprocessing + encoders as training precompute, this composes the two
canonical producers — siglip_features.py and csd_features.py — rather than re-implementing
either. Both write f32 (inference does not inherit the training-cache compression).

Usage:
  train/.venv/bin/python train/scripts/hybrid_features.py IMAGE --out FEATURES.bin
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

TOKENS, DIM, CSD_DIM = 729, 1152, 768


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="input image (jpg/png)")
    ap.add_argument("--out", required=True, help="output raw f32 .bin path")
    ap.add_argument("--csd-weights", default="/Volumes/2TBSSD/models/csd_vit_l_style.safetensors")
    ap.add_argument("--siglip-pool-grid", type=int, default=None,
                    help="option-2: avg-pool the 729 SigLIP tokens to grid*grid (e.g. 9 → 81). "
                         "MUST match data.siglip_pool_grid the adapter was trained with.")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    py = sys.executable

    with tempfile.TemporaryDirectory() as td:
        sig_bin = Path(td) / "sig.bin"
        csd_bin = Path(td) / "csd.bin"
        # SigLIP (729x1152) — same encoder/preprocess as precompute_all.
        subprocess.run([py, str(here / "siglip_features.py"), args.image,
                        "--out", str(sig_bin)], check=True)
        # CSD (768) — same encoder/preprocess as style_precompute.py.
        subprocess.run([py, str(here / "csd_features.py"), args.image,
                        "--out", str(csd_bin), "--weights", args.csd_weights], check=True)

        sig = np.fromfile(sig_bin, dtype=np.float32)
        csd = np.fromfile(csd_bin, dtype=np.float32)

    if sig.size != TOKENS * DIM:
        print(f"unexpected SigLIP size {sig.size} (want {TOKENS*DIM})", file=sys.stderr)
        return 1
    if csd.size != CSD_DIM:
        print(f"unexpected CSD size {csd.size} (want {CSD_DIM})", file=sys.stderr)
        return 1

    sig = sig.reshape(TOKENS, DIM)
    if args.siglip_pool_grid:                     # option-2: identical pooling to training
        sys.path.insert(0, str(here.parent))
        from ip_adapter.dataset import pool_siglip_rows
        sig = pool_siglip_rows(sig, args.siglip_pool_grid)
    row = np.zeros((1, DIM), dtype=np.float32)
    row[0, :CSD_DIM] = csd                        # CSD in the first csd_dim slots, rest zero
    packed = np.concatenate([sig, row], axis=0)   # [grid²+1, 1152] f32

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(out.parent), suffix=".bin.tmp")
    import os
    os.close(fd)
    packed.astype(np.float32).tofile(tmp)
    os.replace(tmp, out)
    # rows = actual packed row count: 730 normally, grid*grid+1 when --siglip-pool-grid pools the
    # SigLIP tokens (mirrors dataset.py option-2 for train<->infer parity). Was hardcoded TOKENS+1.
    print(json.dumps({"out": str(out), "rows": int(packed.shape[0]), "dim": DIM, "csd_dim": CSD_DIM}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
