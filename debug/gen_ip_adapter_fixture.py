#!/usr/bin/env python3
"""
debug/gen_ip_adapter_fixture.py — golden fixtures for the C IP-Adapter port (G-1, Phase 0).

Builds a tiny *synthetic* IPAdapterKlein (seeded, small dims), exports it through the
real export_adapter.py (so the bundle format is exactly what iris_ip_adapter.c will
load), reloads the quantised weights into the model (so the goldens are computed from
the SAME bytes C will load — tight parity, no quant skew), and dumps the reference
outputs of each stage the C must reproduce:

  perceive : SigLIP feats  -> ip_embeds        (MHA over learned queries + LayerNorm)
  get_kv   : ip_embeds     -> k_ip, v_ip       (per-block projection)
  inject   : img_q,k,v     -> contribution     (scale * SDPA(q,k,v))

Outputs (raw little-endian float32 + a shapes.json) under <out>/, plus the bundle.
CPU-only (mx.cpu) and synthetic — safe to run alongside a live flywheel, no GPU, no
real checkpoint. The C test (debug/test_ip_adapter.c, Phase 1) asserts against these.

Usage: train/.venv/bin/python debug/gen_ip_adapter_fixture.py --out debug/fixtures/ip_adapter
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "train" / "export"))

import mlx.core as mx
mx.set_default_device(mx.cpu)                      # no GPU contention with the flywheel
from mlx.utils import tree_flatten, tree_unflatten
from ip_adapter.model import IPAdapterKlein

# Small-but-real dims (head_dim stays 128 — the inference invariant). Tiny fixtures.
HIDDEN, HEADS = 256, 2                              # head_dim = 128 (inference invariant), 2 heads
N_BLOCKS, N_DOUBLE = 5, 1                           # export sets num_double = num_blocks//5 = 1; small bundle
N_TOKENS, SIGLIP_DIM, SIGLIP_SEQ = 8, 64, 16
IMG_SEQ = 12
HEAD_DIM = HIDDEN // HEADS                          # = 128

# export key -> training param key (inverse of export_adapter._KEY_MAP)
_INV = {
    "perceiver.query_tokens": "image_proj.query_tokens",
    "perceiver.query_proj":   "image_proj.cross_attn.query_proj.weight",
    "perceiver.key_proj":     "image_proj.cross_attn.key_proj.weight",
    "perceiver.value_proj":   "image_proj.cross_attn.value_proj.weight",
    "perceiver.out_proj":     "image_proj.cross_attn.out_proj.weight",
    "perceiver.norm_weight":  "image_proj.norm.weight",
    "perceiver.norm_bias":    "image_proj.norm.bias",
    "ip_k_stacked":           "to_k_ip_stacked",
    "ip_v_stacked":           "to_v_ip_stacked",
    "ip_scale":               "scale",
}


def _save(arr, path):
    np.asarray(arr, dtype=np.float32).tofile(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "debug" / "fixtures" / "ip_adapter"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    bundle = out / "bundle"

    mx.random.seed(1234)
    model = IPAdapterKlein(num_blocks=N_BLOCKS, hidden_dim=HIDDEN,
                           num_image_tokens=N_TOKENS, siglip_dim=SIGLIP_DIM,
                           perceiver_heads=HEADS, num_double_blocks=N_DOUBLE)
    mx.eval(model.parameters())

    # 1. Save a training-style checkpoint and export the real bundle (float16).
    flat = dict(tree_flatten(model.parameters()))
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "step_0000001.safetensors"
        mx.save_safetensors(str(ckpt), flat)
        rc = subprocess.run(
            [sys.executable, str(ROOT / "train" / "export" / "export_adapter.py"),
             "--checkpoint", str(ckpt), "--output", str(bundle), "--quant", "float16",
             "--perceiver-heads", str(HEADS)],
            cwd=str(ROOT)).returncode
        if rc != 0:
            print("export_adapter failed", file=sys.stderr); return 1

    # 2. Reload the *bundle* weights (the exact bytes C will load) back into the model,
    #    so goldens are computed from f16-rounded weights — tight parity with C.
    bw = mx.load(str(bundle / "adapter_weights.safetensors"))
    upd = {}
    for ek, tk in _INV.items():
        if ek in bw:
            upd[tk] = bw[ek].astype(mx.float32)
    model.update(tree_unflatten(list(upd.items())))
    mx.eval(model.parameters())

    # 3. Goldens via the real model forward.
    siglip = mx.random.normal((1, SIGLIP_SEQ, SIGLIP_DIM))
    ip_embeds = model.get_image_embeds(siglip)            # [1, N_TOKENS, HIDDEN]
    k_all, v_all = model.get_kv_all(ip_embeds)            # [1, N_BLOCKS, N_TOKENS, HIDDEN]
    blk = 0
    k_ip = k_all[:, blk]                                  # [1, N_TOKENS, HIDDEN]
    v_ip = v_all[:, blk]
    # inject: img_q in C's flat [IMG_SEQ, HIDDEN] -> [1, HEADS, IMG_SEQ, HEAD_DIM]
    img_q_flat = mx.random.normal((IMG_SEQ, HIDDEN))
    img_q = img_q_flat.reshape(IMG_SEQ, HEADS, HEAD_DIM).transpose(1, 0, 2)[None]
    contrib = model.inject(img_q, k_ip, v_ip, blk)        # [1, IMG_SEQ, HIDDEN]
    mx.eval(ip_embeds, k_ip, v_ip, contrib)

    # 4. Dump inputs + goldens (float32 raw) + shapes.
    _save(siglip,            out / "in_siglip.bin")
    _save(ip_embeds,         out / "gold_ip_embeds.bin")
    _save(k_ip,              out / "gold_k_ip_b0.bin")
    _save(v_ip,              out / "gold_v_ip_b0.bin")
    _save(img_q_flat,        out / "in_img_q.bin")
    _save(contrib,           out / "gold_inject_b0.bin")
    shapes = {
        "hidden": HIDDEN, "heads": HEADS, "head_dim": HEAD_DIM,
        "num_blocks": N_BLOCKS, "num_double_blocks": N_DOUBLE,
        "num_image_tokens": N_TOKENS, "siglip_dim": SIGLIP_DIM, "siglip_seq": SIGLIP_SEQ,
        "img_seq": IMG_SEQ, "block": blk,
        "in_siglip": [1, SIGLIP_SEQ, SIGLIP_DIM],
        "gold_ip_embeds": [1, N_TOKENS, HIDDEN],
        "gold_k_ip_b0": [1, N_TOKENS, HIDDEN], "gold_v_ip_b0": [1, N_TOKENS, HIDDEN],
        "in_img_q": [IMG_SEQ, HIDDEN], "gold_inject_b0": [1, IMG_SEQ, HIDDEN],
    }
    (out / "shapes.json").write_text(json.dumps(shapes, indent=2))
    print(f"wrote fixtures + bundle to {out}")
    print(f"  ip_embeds std={float(ip_embeds.std()):.4f}  contrib std={float(contrib.std()):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
