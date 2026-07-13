#!/usr/bin/env python3
"""gen_csdmod_fixture.py — golden fixture for the C csd_mod parity test (debug/test_csdmod.c).

Loads the EXPORTED csd_mod weights (the same safetensors the C loads), runs the REAL Python
CSDModulation forward on a seeded L2-normalised csd vector, and writes the input + golden output
as raw little-endian f32 so the C side can reproduce it to a tight tolerance.

  train/.venv/bin/python debug/gen_csdmod_fixture.py \
      --weights /Volumes/2TBSSD/sref_eval/joint_v1_c_export/csd_mod.safetensors \
      --out     /Volumes/2TBSSD/sref_eval/joint_v1_c_export
"""
import argparse, os, sys
import numpy as np
import mlx.core as mx
from mlx.utils import tree_unflatten

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
from ip_adapter.model import CSDModulation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    w = mx.load(args.weights)
    mlp_dim, csd_dim = w["fc1.weight"].shape       # [1024, 768]
    hidden_dim = w["fc2.weight"].shape[0]          # 3072
    mod = CSDModulation(hidden_dim=hidden_dim, csd_dim=csd_dim, mlp_dim=mlp_dim)
    mod.update(tree_unflatten(list(w.items())))
    mod.eval()
    mx.eval(mod.parameters())

    # seeded, L2-normalised csd (matches the inference convention)
    rng = np.random.default_rng(args.seed)
    csd = rng.standard_normal(csd_dim).astype(np.float32)
    csd /= np.linalg.norm(csd)

    delta = np.array(mod(mx.array(csd)[None]).astype(mx.float32))[0]   # [hidden_dim] golden
    assert delta.shape == (hidden_dim,), delta.shape

    os.makedirs(args.out, exist_ok=True)
    csd.astype("<f4").tofile(os.path.join(args.out, "csdmod_input.f32"))
    delta.astype("<f4").tofile(os.path.join(args.out, "csdmod_golden.f32"))
    print(f"  csd_dim={csd_dim} mlp_dim={mlp_dim} hidden_dim={hidden_dim}")
    print(f"  golden delta: mean={delta.mean():.6f} std={delta.std():.6f} "
          f"absmax={np.abs(delta).max():.6f}")
    print(f"  wrote csdmod_input.f32 ({csd_dim}) + csdmod_golden.f32 ({hidden_dim}) -> {args.out}")


if __name__ == "__main__":
    main()
