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
    ap.add_argument("--weights", help="exported csd_mod.safetensors (omit with --synthetic)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true",
                    help="build a SMALL random CSDModulation (hermetic make-test fixture) instead of "
                         "loading real weights; randomises the zero-init fc2 so the matmul is exercised, "
                         "and writes csd_mod.safetensors alongside the input/golden.")
    args = ap.parse_args()

    if args.synthetic:
        # Small, non-degenerate dims (all distinct, none a multiple of another) — enough to exercise
        # fc1/SiLU/fc2 + biases without a large committed fixture. CRUCIAL: randomise fc2 (adaLN-zero
        # at init) so the golden is non-trivial and the C fc2 matmul is actually tested.
        csd_dim, mlp_dim, hidden_dim = 24, 40, 48
        mod = CSDModulation(hidden_dim=hidden_dim, csd_dim=csd_dim, mlp_dim=mlp_dim)
        rng0 = np.random.default_rng(args.seed + 1)
        params = {
            "fc1.weight": mx.array((rng0.standard_normal((mlp_dim, csd_dim)) * 0.2).astype(np.float32)),
            "fc1.bias":   mx.array((rng0.standard_normal(mlp_dim) * 0.1).astype(np.float32)),
            "fc2.weight": mx.array((rng0.standard_normal((hidden_dim, mlp_dim)) * 0.2).astype(np.float32)),
            "fc2.bias":   mx.array((rng0.standard_normal(hidden_dim) * 0.1).astype(np.float32)),
        }
        mod.update(tree_unflatten(list(params.items())))
        os.makedirs(args.out, exist_ok=True)
        mx.save_safetensors(os.path.join(args.out, "csd_mod.safetensors"), params)
    else:
        if not args.weights:
            ap.error("--weights is required unless --synthetic is given")
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
