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

CSD_DIM = 32                                       # synthetic CSD style dim (cond_mode="csd")

# CSD-mode export key -> training param key (FiLM instead of cross_attn; in_gamma/beta absent).
_INV_CSD = {
    "perceiver.query_tokens": "image_proj.query_tokens",
    "perceiver.film_weight":  "image_proj.film.weight",
    "perceiver.film_bias":    "image_proj.film.bias",
    "perceiver.norm_weight":  "image_proj.norm.weight",
    "perceiver.norm_bias":    "image_proj.norm.bias",
    "ip_k_stacked":           "to_k_ip_stacked",
    "ip_v_stacked":           "to_v_ip_stacked",
    "ip_scale":               "scale",
}

# Small-but-real dims (Flux block head_dim stays 128 — the inference invariant).
# PERCEIVER_HEADS is deliberately != HIDDEN/128 so the goldens exercise the real
# perceiver grouping (head_dim = HIDDEN/PERCEIVER_HEADS), not the Flux block's 128.
# A fixture with PERCEIVER_HEADS == HIDDEN/128 would mask IP-ADAPTER-INFER-1.
HIDDEN, HEADS = 256, 2                              # Flux block: head_dim = 128, 2 heads
PERCEIVER_HEADS = 4                                 # != HIDDEN/128 (=2); perceiver head_dim = 64
N_BLOCKS, N_DOUBLE = 5, 1                           # export sets num_double = num_blocks//5 = 1; small bundle
N_TOKENS, SIGLIP_DIM, SIGLIP_SEQ = 8, 64, 16
IMG_SEQ = 12
HEAD_DIM = HIDDEN // HEADS                          # = 128 (Flux block / inject path)

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


def _export(ckpt, bundle_dir, quant):
    rc = subprocess.run(
        [sys.executable, str(ROOT / "train" / "export" / "export_adapter.py"),
         "--checkpoint", str(ckpt), "--output", str(bundle_dir), "--quant", quant,
         "--perceiver-heads", str(PERCEIVER_HEADS)], cwd=str(ROOT)).returncode
    if rc != 0:
        raise SystemExit(f"export_adapter ({quant}) failed")


def _reload_from_bundle(model, bundle_dir, inv=_INV):
    """Set model weights from the EXACT bundle bytes C will load — dequantising
    int8 (q * per-row scale) so goldens match the C int8 path; f16/f32 pass through."""
    w = mx.load(str(bundle_dir / "adapter_weights.safetensors"))
    upd = {}
    for ek, tk in inv.items():
        if ek not in w:
            continue
        sk = ek + ".scale"
        if sk in w:                                    # int8: per-row dequant
            q = w[ek].astype(mx.float32)
            cols = q.shape[-1]
            scale = w[sk].astype(mx.float32).reshape(-1, 1)
            upd[tk] = (q.reshape(-1, cols) * scale).reshape(q.shape)
        else:
            upd[tk] = w[ek].astype(mx.float32)
    model.update(tree_unflatten(list(upd.items())))
    mx.eval(model.parameters())


def _dump_goldens(model, out, suffix, siglip, img_q):
    ip_embeds = model.get_image_embeds(siglip)
    k_all, v_all = model.get_kv_all(ip_embeds)
    k_ip, v_ip = k_all[:, 0], v_all[:, 0]
    img_q4 = img_q.reshape(IMG_SEQ, HEADS, HEAD_DIM).transpose(1, 0, 2)[None]
    contrib = model.inject(img_q4, k_ip, v_ip, 0)
    mx.eval(ip_embeds, k_ip, v_ip, contrib)
    _save(ip_embeds, out / f"gold_ip_embeds{suffix}.bin")
    _save(k_ip,      out / f"gold_k_ip_b0{suffix}.bin")
    _save(v_ip,      out / f"gold_v_ip_b0{suffix}.bin")
    _save(contrib,   out / f"gold_inject_b0{suffix}.bin")
    return float(ip_embeds.std()), float(contrib.std())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "debug" / "fixtures" / "ip_adapter"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    bundle = out / "bundle"

    mx.random.seed(1234)
    model = IPAdapterKlein(num_blocks=N_BLOCKS, hidden_dim=HIDDEN,
                           num_image_tokens=N_TOKENS, siglip_dim=SIGLIP_DIM,
                           perceiver_heads=PERCEIVER_HEADS, num_double_blocks=N_DOUBLE)
    mx.eval(model.parameters())

    # Shared inputs (same for every quant mode so the C test reuses them).
    siglip     = mx.random.normal((1, SIGLIP_SEQ, SIGLIP_DIM))
    img_q_flat = mx.random.normal((IMG_SEQ, HIDDEN))
    mx.eval(siglip, img_q_flat)
    _save(siglip,     out / "in_siglip.bin")
    _save(img_q_flat, out / "in_img_q.bin")

    # Export the real bundle in each quant mode; compute goldens from the reloaded
    # (quantised) weights so they match exactly what the C loader will dequantise.
    flat = dict(tree_flatten(model.parameters()))
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "step_0000001.safetensors"
        mx.save_safetensors(str(ckpt), flat)
        for sub, quant, suffix in (("bundle", "float16", ""),
                                   ("bundle_int8", "int8", "_int8")):
            bdir = out / sub
            _export(ckpt, bdir, quant)
            _reload_from_bundle(model, bdir)
            es, cs = _dump_goldens(model, out, suffix, siglip, img_q_flat)
            print(f"  {quant:8s}: ip_embeds std={es:.4f}  contrib std={cs:.4f}")

    # ── CSD-mode fixture (cond_mode="csd"): FiLM over a content-invariant [CSD_DIM] vector ──
    mx.random.seed(4321)
    csd_model = IPAdapterKlein(num_blocks=N_BLOCKS, hidden_dim=HIDDEN,
                               num_image_tokens=N_TOKENS, num_double_blocks=N_DOUBLE,
                               cond_mode="csd", csd_dim=CSD_DIM)
    # FiLM is ZERO-initialised in the module (stable training start) → goldens would be trivial
    # (csd input ignored, all tokens = LN(query_tokens)). Randomise FiLM so the fixture actually
    # exercises the C FiLM matmul + scale/shift split (the train↔infer parity surface).
    fw = mx.random.normal((2 * HIDDEN, CSD_DIM)) * 0.1
    fb = mx.random.normal((2 * HIDDEN,)) * 0.1
    csd_model.update(tree_unflatten([("image_proj.film.weight", fw),
                                     ("image_proj.film.bias", fb)]))
    mx.eval(csd_model.parameters())
    csd_vec = mx.random.normal((1, CSD_DIM))
    csd_vec = csd_vec / mx.linalg.norm(csd_vec, axis=-1, keepdims=True)   # L2-normed, like real CSD
    mx.eval(csd_vec)
    _save(csd_vec, out / "in_csd.bin")
    flat_csd = dict(tree_flatten(csd_model.parameters()))
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "step_0000001.safetensors"
        mx.save_safetensors(str(ckpt), flat_csd)
        bdir = out / "bundle_csd"
        _export(ckpt, bdir, "float16")
        _reload_from_bundle(csd_model, bdir, inv=_INV_CSD)
        es, cs = _dump_goldens(csd_model, out, "_csd", csd_vec, img_q_flat)
        print(f"  csd     : ip_embeds std={es:.4f}  contrib std={cs:.4f}")

    shapes = {
        "hidden": HIDDEN, "heads": HEADS, "head_dim": HEAD_DIM,
        "perceiver_heads": PERCEIVER_HEADS,
        "num_blocks": N_BLOCKS, "num_double_blocks": N_DOUBLE,
        "num_image_tokens": N_TOKENS, "siglip_dim": SIGLIP_DIM, "siglip_seq": SIGLIP_SEQ,
        "img_seq": IMG_SEQ, "block": 0,
        "in_siglip": [1, SIGLIP_SEQ, SIGLIP_DIM],
        "gold_ip_embeds": [1, N_TOKENS, HIDDEN],
        "gold_k_ip_b0": [1, N_TOKENS, HIDDEN], "gold_v_ip_b0": [1, N_TOKENS, HIDDEN],
        "in_img_q": [IMG_SEQ, HIDDEN], "gold_inject_b0": [1, IMG_SEQ, HIDDEN],
        "csd_dim": CSD_DIM, "in_csd": [1, CSD_DIM],
        "gold_ip_embeds_csd": [1, N_TOKENS, HIDDEN],
        "gold_k_ip_b0_csd": [1, N_TOKENS, HIDDEN], "gold_v_ip_b0_csd": [1, N_TOKENS, HIDDEN],
        "gold_inject_b0_csd": [1, IMG_SEQ, HIDDEN],
    }
    (out / "shapes.json").write_text(json.dumps(shapes, indent=2))
    print(f"wrote fixtures + bundle to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
