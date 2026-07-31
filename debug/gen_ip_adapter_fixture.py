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

# Hybrid (cond_mode="hybrid", SREF-COMBINE-1): SigLIP perceiver (image_proj.*) + the separate
# CSD module (csd_proj.*). Export key -> training param key for the reload.
_INV_HYBRID = {
    "perceiver.query_tokens": "image_proj.query_tokens",
    "perceiver.query_proj":   "image_proj.cross_attn.query_proj.weight",
    "perceiver.key_proj":     "image_proj.cross_attn.key_proj.weight",
    "perceiver.value_proj":   "image_proj.cross_attn.value_proj.weight",
    "perceiver.out_proj":     "image_proj.cross_attn.out_proj.weight",
    "perceiver.norm_weight":  "image_proj.norm.weight",
    "perceiver.norm_bias":    "image_proj.norm.bias",
    "csd.query_tokens":       "csd_proj.query_tokens",
    "csd.film_weight":        "csd_proj.film.weight",
    "csd.film_bias":          "csd_proj.film.bias",
    "csd.norm_weight":        "csd_proj.norm.weight",
    "csd.norm_bias":          "csd_proj.norm.bias",
    "group_gate":             "group_gate",
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


# ── Per-block IP-injection propagation golden (M3 parity guard) ──────────────────────────────
# The existing goldens above test each IP stage in ISOLATION at block 0. They do NOT test that a
# block's IP contribution PROPAGATES: C inference (iris_transformer_flux.c) injects k_ip/v_ip PER
# BLOCK into the post-block hidden state, so block i+1 derives its image-Q from a state that
# already carries block i's injection. This mirrors the Python `use_block_injection=True` path
# (_flux_forward_with_ip). The DEFAULT trainer path (_pred_from_embeds, use_block_injection=False)
# is an APPROXIMATION: every block's Q comes from the IP-free hidden and contributions are summed
# ONCE at the end — a different result. This fixture guards the CORRECT per-block-injected forward.
#
# Hermetic: no Flux weights. The frozen-Flux per-block transform is stood in by identity, and the
# image-Q is derived by a per-head RMSNorm (head_dim=128, the Flux invariant) of the CURRENT
# hidden — the minimal op that makes Q depend on accumulated injections (so propagation is
# actually exercised). The IP math itself (perceive / get_kv / inject SDPA + per-block scale) is
# the REAL parity surface, computed by the same model methods the C functions mirror.
BP_EPS = 1e-6


def _derive_q(h, gamma):
    """Per-head RMSNorm(head_dim=128) of the current image hidden → [1, HEADS, S, HEAD_DIM].
    Stand-in for the block's post-QK-norm PRE-RoPE image-Q; makes Q depend on the accumulated
    hidden so the per-block injection propagation is testable. Mirrored bit-for-bit in C."""
    S = h.shape[1]
    qh = h.reshape(1, S, HEADS, HEAD_DIM)
    ms = mx.mean(qh * qh, axis=-1, keepdims=True)
    qn = qh * mx.rsqrt(ms + BP_EPS) * gamma          # gamma broadcasts over HEAD_DIM
    return qn.transpose(0, 2, 1, 3)                  # [1, HEADS, S, HEAD_DIM]


def _forward_block_injected(model, ip_embeds, h0, gamma):
    """Correct per-block-injected forward (matches C + use_block_injection=True):
    h_{i+1} = h_i + scale[i] * SDPA(derive_q(h_i), k_i, v_i)  — Q sees earlier injections."""
    k_all, v_all = model.get_kv_all(ip_embeds)
    h = h0
    for i in range(model.num_blocks):
        q4 = _derive_q(h, gamma)
        h = h + model.inject(q4, k_all[:, i], v_all[:, i], i)   # inject() applies scale[i]
    return h


def _forward_end_sum(model, ip_embeds, h0, gamma):
    """The use_block_injection=False APPROXIMATION (negative control): every block's Q comes
    from the IP-free initial hidden and contributions are summed ONCE. Must differ from the
    per-block forward, else the fixture would not discriminate a regression to end-sum."""
    k_all, v_all = model.get_kv_all(ip_embeds)
    q4 = _derive_q(h0, gamma)                          # ALL blocks reuse the initial Q
    ip_total = mx.zeros_like(h0)
    for i in range(model.num_blocks):
        ip_total = ip_total + model.inject(q4, k_all[:, i], v_all[:, i], i)
    return h0 + ip_total


def _dump_block_prop(model, out, siglip, img_q_flat, gamma_np):
    ip_embeds = model.get_image_embeds(siglip)
    gamma = mx.array(gamma_np.astype(np.float32))
    h0 = img_q_flat[None]                              # [1, IMG_SEQ, HIDDEN]
    h_perblock = _forward_block_injected(model, ip_embeds, h0, gamma)
    h_endsum   = _forward_end_sum(model, ip_embeds, h0, gamma)
    mx.eval(h_perblock, h_endsum)
    _save(h_perblock, out / "gold_blockprop.bin")
    # Discrimination: prove the fixture would CATCH a regression to the end-sum approximation.
    diff = float(mx.linalg.norm(h_perblock - h_endsum))
    base = float(mx.linalg.norm(h_perblock))
    rel = diff / max(base, 1e-9)
    assert rel > 0.01, f"block-injection vs end-sum differ by only {rel:.4%} — fixture too weak"
    return float(h_perblock.std()), rel


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
        _export(ckpt, bdir, "bfloat16")   # the REAL arm exports bf16 — guard that load path
        _reload_from_bundle(csd_model, bdir, inv=_INV_CSD)
        es, cs = _dump_goldens(csd_model, out, "_csd", csd_vec, img_q_flat)
        print(f"  csd     : ip_embeds std={es:.4f}  contrib std={cs:.4f}")

    # ── Hybrid-mode fixture (cond_mode="hybrid", SREF-COMBINE-1): SigLIP perceiver (first half)
    #    + CSD FiLM (second half), concatenated → 2*N_TOKENS image tokens. Input is the packed
    #    [SIGLIP_SEQ+1, SIGLIP_DIM] feature (rows 0..SIGLIP_SEQ-1 = SigLIP, last row = CSD padded). ──
    mx.random.seed(7777)
    HYB_TOKENS = 2 * N_TOKENS
    hyb_model = IPAdapterKlein(num_blocks=N_BLOCKS, hidden_dim=HIDDEN,
                               num_image_tokens=HYB_TOKENS, siglip_dim=SIGLIP_DIM,
                               perceiver_heads=PERCEIVER_HEADS, num_double_blocks=N_DOUBLE,
                               cond_mode="hybrid", csd_dim=CSD_DIM)
    # Randomise the CSD FiLM (ZERO-initialised in the module) so the CSD half is non-trivial.
    fw_h = mx.random.normal((2 * HIDDEN, CSD_DIM)) * 0.1
    fb_h = mx.random.normal((2 * HIDDEN,)) * 0.1
    # Non-trivial per-block per-group injection gate (default is all-ones = no-op) so the
    # parity test exercises the get_kv V-scaling path. Distinct per block AND per group.
    gate_h = mx.random.uniform(low=0.2, high=1.0, shape=(N_BLOCKS, 2))
    hyb_model.update(tree_unflatten([("csd_proj.film.weight", fw_h),
                                     ("csd_proj.film.bias", fb_h),
                                     ("group_gate", gate_h)]))
    mx.eval(hyb_model.parameters())
    # Packed feature: SigLIP rows + one CSD row (L2-normed CSD in the first CSD_DIM slots, rest 0).
    sig_h = mx.random.normal((1, SIGLIP_SEQ, SIGLIP_DIM))
    csd_h = mx.random.normal((1, CSD_DIM))
    csd_h = csd_h / mx.linalg.norm(csd_h, axis=-1, keepdims=True)
    csd_row = mx.concatenate([csd_h, mx.zeros((1, SIGLIP_DIM - CSD_DIM))], axis=-1)[:, None, :]
    hyb_feat = mx.concatenate([sig_h, csd_row], axis=1)            # [1, SIGLIP_SEQ+1, SIGLIP_DIM]
    mx.eval(hyb_feat)
    _save(hyb_feat, out / "in_hybrid.bin")
    flat_hyb = dict(tree_flatten(hyb_model.parameters()))
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "step_0000001.safetensors"
        mx.save_safetensors(str(ckpt), flat_hyb)
        bdir = out / "bundle_hybrid"
        _export(ckpt, bdir, "bfloat16")
        _reload_from_bundle(hyb_model, bdir, inv=_INV_HYBRID)
        es, cs = _dump_goldens(hyb_model, out, "_hybrid", hyb_feat, img_q_flat)
        print(f"  hybrid  : ip_embeds std={es:.4f}  contrib std={cs:.4f}")

    # ── Block-injection propagation fixture (M3): siglip perceiver, RANDOMISED per-block scale ──
    #    (so the per-block scale application/indexing is guarded, not a constant), ≥3 blocks
    #    (N_BLOCKS=5) so propagation across blocks is exercised. Uses its OWN small initial hidden
    #    and boosted scales so each block's injection meaningfully rotates the propagated image-Q
    #    — otherwise (h0≈unit, injection≪h0) per-head RMSNorm makes per-block ≈ end-sum and the
    #    fixture cannot discriminate the two forwards.
    mx.random.seed(9182)
    bp_model = IPAdapterKlein(num_blocks=N_BLOCKS, hidden_dim=HIDDEN,
                              num_image_tokens=N_TOKENS, siglip_dim=SIGLIP_DIM,
                              perceiver_heads=PERCEIVER_HEADS, num_double_blocks=N_DOUBLE)
    # Distinct per-block scale (default ip_scale_init makes all blocks 1.0 → indexing untested).
    bp_scale = mx.random.uniform(low=1.5, high=4.0, shape=(N_BLOCKS,))
    bp_model.update(tree_unflatten([("scale", bp_scale)]))
    mx.eval(bp_model.parameters())
    # gamma: per-head RMSNorm weight (RMSNorm inits to ones → randomise so it is exercised).
    gamma_np = (1.0 + np.random.default_rng(9182).normal(size=(HEAD_DIM,)) * 0.2).astype(np.float32)
    _save(gamma_np, out / "in_blockprop_gamma.bin")
    # Small initial hidden so per-block injections dominate the propagated Q direction.
    bp_h0 = mx.random.normal((IMG_SEQ, HIDDEN)) * 0.1
    mx.eval(bp_h0)
    _save(bp_h0, out / "in_blockprop_h0.bin")
    flat_bp = dict(tree_flatten(bp_model.parameters()))
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "step_0000001.safetensors"
        mx.save_safetensors(str(ckpt), flat_bp)
        bdir = out / "bundle_blockprop"
        _export(ckpt, bdir, "float16")
        _reload_from_bundle(bp_model, bdir)
        hs, rel = _dump_block_prop(bp_model, out, siglip, bp_h0, gamma_np)
        print(f"  blockprop: h_final std={hs:.4f}  perblock-vs-endsum rel-diff={rel:.2%}")

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
        # Hybrid: 2*N_TOKENS image tokens; packed input has SIGLIP_SEQ+1 rows.
        "hybrid_tokens": 2 * N_TOKENS, "hybrid_seq": SIGLIP_SEQ + 1,
        "in_hybrid": [1, SIGLIP_SEQ + 1, SIGLIP_DIM],
        "gold_ip_embeds_hybrid": [1, 2 * N_TOKENS, HIDDEN],
        "gold_k_ip_b0_hybrid": [1, 2 * N_TOKENS, HIDDEN],
        "gold_v_ip_b0_hybrid": [1, 2 * N_TOKENS, HIDDEN],
        "gold_inject_b0_hybrid": [1, IMG_SEQ, HIDDEN],
        # Block-injection propagation (M3): per-head RMSNorm derive_q, per-block inject over
        # N_BLOCKS, gamma = [HEAD_DIM] RMSNorm weight, golden = final hidden [1, IMG_SEQ, HIDDEN].
        "blockprop_eps": BP_EPS,
        "in_blockprop_gamma": [HEAD_DIM],
        "in_blockprop_h0": [IMG_SEQ, HIDDEN],
        "gold_blockprop": [1, IMG_SEQ, HIDDEN],
    }
    (out / "shapes.json").write_text(json.dumps(shapes, indent=2))
    print(f"wrote fixtures + bundle to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
