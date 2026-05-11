#!/usr/bin/env python3
"""
train/scripts/test_ip_adapter_inference.py — Validate exported IP-Adapter bundle.

Generates a baseline image (no adapter) and an adapter-conditioned image from
the same seed and prompt, prints detailed diagnostics, and saves a three-panel
comparison.

Usage:
    python train/scripts/test_ip_adapter_inference.py \\
        --adapter /path/to/bundle/ \\
        --style-image /path/to/style.jpg \\
        --prompt "an oil painting of a mountain landscape" \\
        --flux-model /path/to/flux-klein-4b \\
        [--strength 1.0] [--steps 4] [--seed 42] [--width 512] [--height 512] \\
        [--output-dir /tmp/ip_adapter_test] [--style-only]

Or use a training config to supply the model path:
    python train/scripts/test_ip_adapter_inference.py \\
        --adapter /path/to/bundle/ \\
        --style-image /path/to/style.jpg \\
        --prompt "..." \\
        --config train/configs/stage1_512px.yaml

SigLIP feature extraction and CLIP-I scoring require 'transformers' and 'torch'.
If unavailable, generation still runs with zero style conditioning (smoke test only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import mlx.core as mx
    import numpy as np
except ImportError:
    print("Error: MLX not found. Run: source train/.venv/bin/activate", file=sys.stderr)
    sys.exit(1)

# Insert train/ so local imports resolve correctly.
_TRAIN_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_TRAIN_DIR))

try:
    from ip_adapter.model import IPAdapterKlein
    from train_ip_adapter import _flux_forward_no_ip
except ImportError as e:
    print(f"Error: cannot import training code: {e}", file=sys.stderr)
    print("  Ensure this script is run with train/.venv active.", file=sys.stderr)
    sys.exit(1)

try:
    from mflux.models.flux2 import Flux2Klein
    _HAS_MFLUX = True
except ImportError:
    _HAS_MFLUX = False


# ── Export bundle key map ─────────────────────────────────────────────────────
# export_adapter.py writes "export" key names; IPAdapterKlein uses "training" names.
_EXPORT_TO_TRAIN: dict[str, str] = {
    "perceiver.query_tokens":  "image_proj.query_tokens",
    "perceiver.query_proj":    "image_proj.cross_attn.query_proj.weight",
    "perceiver.key_proj":      "image_proj.cross_attn.key_proj.weight",
    "perceiver.value_proj":    "image_proj.cross_attn.value_proj.weight",
    "perceiver.out_proj":      "image_proj.cross_attn.out_proj.weight",
    "perceiver.norm_weight":   "image_proj.norm.weight",
    "perceiver.norm_bias":     "image_proj.norm.bias",
    "ip_k_stacked":            "to_k_ip_stacked",
    "ip_v_stacked":            "to_v_ip_stacked",
    "ip_scale":                "scale",
}

_INT8_QUANT_KEYS = frozenset({
    "perceiver.query_proj", "perceiver.key_proj",
    "perceiver.value_proj", "perceiver.out_proj",
    "ip_k_stacked", "ip_v_stacked",
})


# ─────────────────────────────────────────────────────────────────────────────
# Bundle loading
# ─────────────────────────────────────────────────────────────────────────────

def load_bundle(adapter_dir: str) -> tuple[dict, IPAdapterKlein]:
    """Load adapter_meta.json + adapter_weights.safetensors → (meta, IPAdapterKlein)."""
    meta_path = os.path.join(adapter_dir, "adapter_meta.json")
    wts_path  = os.path.join(adapter_dir, "adapter_weights.safetensors")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"adapter_meta.json not found in {adapter_dir}")
    if not os.path.exists(wts_path):
        raise FileNotFoundError(f"adapter_weights.safetensors not found in {adapter_dir}")

    with open(meta_path) as f:
        meta = json.load(f)

    raw: dict[str, mx.array] = dict(mx.load(wts_path))

    # INT8: dequantize large linear tensors to bfloat16 before loading,
    # matching the dtype used during training for SDPA compatibility.
    if meta.get("quant") == "int8":
        for k in _INT8_QUANT_KEYS:
            sk = f"{k}.scale"
            if k not in raw or sk not in raw:
                continue
            q = np.array(raw[k]).astype(np.int32)
            s = np.array(raw[sk].astype(mx.float32))
            if q.ndim == 2:
                dq = (q * s[:, None]).astype(np.float32)
            elif q.ndim == 3:
                dq = (q * s[:, :, None]).astype(np.float32)
            else:
                dq = q.astype(np.float32)
            raw[k] = mx.array(dq).astype(mx.bfloat16)
            del raw[sk]

    # Reverse key map: export names → IPAdapterKlein parameter paths.
    # Keep original dtype from bundle (BF16 for large tensors, F32 for ip_scale)
    # so SDPA dtype matches the BF16 Q vectors from the Flux transformer.
    train_weights: list[tuple[str, mx.array]] = []
    missing: list[str] = []
    for ek, tk in _EXPORT_TO_TRAIN.items():
        if ek in raw:
            train_weights.append((tk, raw[ek]))
        else:
            missing.append(ek)
    if missing:
        raise ValueError(f"Bundle missing expected tensors: {missing}")

    adapter = IPAdapterKlein(
        num_blocks       =meta["num_blocks"],
        num_double_blocks=meta.get("num_double_blocks", 5),
        hidden_dim       =meta["hidden_dim"],
        num_image_tokens =meta["num_image_tokens"],
        siglip_dim       =meta["siglip_dim"],
        perceiver_heads  =meta.get("perceiver_heads", meta.get("num_heads", 24)),
    )
    adapter.load_weights(train_weights)
    adapter.eval()
    adapter.freeze()
    mx.eval(adapter.parameters())
    return meta, adapter


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def print_meta(meta: dict) -> None:
    step = meta.get("training_step")
    nd   = meta.get("num_double_blocks", 5)
    nb   = meta.get("num_blocks", 25)
    print(f"  adapter_type:      {meta.get('adapter_type', '?')}")
    print(f"  model_target:      {meta.get('model_target', '?')}")
    print(f"  training_step:     {step:,}" if step else "  training_step:     ?")
    print(f"  quant:             {meta.get('quant', '?')}")
    print(f"  num_blocks:        {nb}  ({nd} double-stream + {nb - nd} single-stream)")
    print(f"  hidden_dim:        {meta.get('hidden_dim', '?')}")
    print(f"  num_image_tokens:  {meta.get('num_image_tokens', '?')}")
    if meta.get("style_only"):
        print(f"  mode:              style_only  (double-stream ip_scale zeroed in bundle)")
    src = meta.get("source_checkpoint")
    if src:
        print(f"  source:            {src}")


def print_ip_scale_stats(adapter: IPAdapterKlein, meta: dict) -> None:
    nd  = meta.get("num_double_blocks", 5)
    nb  = meta["num_blocks"]
    sc  = np.array(adapter.scale.astype(mx.float32))
    fmt = lambda v: f"{v:.4f}"

    if not np.all(np.isfinite(sc)):
        print("  ERROR: ip_scale contains NaN/Inf — bundle may be corrupt!")
        return

    ds = sc[:nd]
    ss = sc[nd:]

    print(f"\nip_scale ({nb} blocks):")
    print(f"  Double-stream (blocks 0–{nd-1}):")
    print("    " + "  ".join(f"[{i}] {fmt(v)}" for i, v in enumerate(ds)))
    print(f"    mean={ds.mean():.4f}  min={ds.min():.4f}  max={ds.max():.4f}")

    print(f"  Single-stream (blocks {nd}–{nb-1}):")
    if len(ss) <= 10:
        print("    " + "  ".join(f"[{nd+i}] {fmt(v)}" for i, v in enumerate(ss)))
    else:
        head = "  ".join(f"[{nd+i}] {fmt(v)}" for i, v in enumerate(ss[:5]))
        tail = "  ".join(f"[{nb-5+i}] {fmt(v)}" for i, v in enumerate(ss[-5:]))
        print(f"    (first 5) {head}")
        print(f"    (last 5)  {tail}")
    print(f"    mean={ss.mean():.4f}  min={ss.min():.4f}  max={ss.max():.4f}")

    # Anomaly warnings
    style_only_bundle = meta.get("style_only", False)
    near_zero = np.where(sc < 0.01)[0].tolist()
    very_large = np.where(sc > 5.0)[0].tolist()

    if near_zero:
        ds_zero = [i for i in near_zero if i < nd]
        ss_zero = [i for i in near_zero if i >= nd]
        if style_only_bundle and ds_zero and not ss_zero:
            print(f"  INFO:  blocks {ds_zero} have ip_scale ≈ 0  (expected: style_only bundle)")
        else:
            print(f"  WARN:  blocks {near_zero} have ip_scale < 0.01 — style transfer may be negligible")

    if very_large:
        print(f"  WARN:  blocks {very_large} have ip_scale > 5.0 — may cause saturation artifacts")

    if not style_only_bundle and ds.mean() < 0.05:
        print(f"  WARN:  double-stream mean scale {ds.mean():.4f} is very low — layout conditioning is absent")


def print_token_stats(ip_embeds: mx.array, k_ip: mx.array) -> None:
    emb     = np.array(ip_embeds.astype(mx.float32))   # [1, 128, 3072]
    norms   = np.linalg.norm(emb[0], axis=-1)           # [128]
    k_np    = np.array(k_ip.astype(mx.float32))         # [1, 25, 128, 3072]
    k_norms = np.linalg.norm(k_np[0], axis=-1)          # [25, 128]

    print(f"  Perceiver output:  [{emb.shape[1]} tokens, {emb.shape[2]}d]  "
          f"norm mean={norms.mean():.2f}  std={norms.std():.2f}  "
          f"min={norms.min():.2f}  max={norms.max():.2f}")
    print(f"  IP-K tokens:       [{k_np.shape[1]} blocks × {k_np.shape[2]} tokens]  "
          f"norm mean={k_norms.mean():.2f}  std={k_norms.std():.2f}")

    if norms.mean() < 1e-3:
        print("  WARN:  perceiver output norms near-zero — style signal not propagating (check SigLIP features)")
    if k_norms.mean() < 1e-3:
        print("  WARN:  IP-K norms near-zero — style conditioning will have no effect")


# ─────────────────────────────────────────────────────────────────────────────
# SigLIP
# ─────────────────────────────────────────────────────────────────────────────

def _load_siglip(model_name: str):
    """Return (SiglipModel, SiglipProcessor) or (None, None) if unavailable."""
    try:
        from transformers import SiglipModel, SiglipProcessor
        model = SiglipModel.from_pretrained(model_name).eval()
        proc  = SiglipProcessor.from_pretrained(model_name)
        return model, proc
    except ImportError:
        print("  WARN:  'transformers' not installed — SigLIP unavailable", file=sys.stderr)
        print("         Style features will be zero; CLIP-I skipped.", file=sys.stderr)
        print("         Install: pip install transformers torch", file=sys.stderr)
    except Exception as e:
        print(f"  WARN:  SigLIP load failed ({e}) — style features zeroed", file=sys.stderr)
    return None, None


def _siglip_feats(
    model, proc, pil_img,
    siglip_dim: int = 1152,
    num_patches: int = 729,
) -> mx.array:
    """Extract SigLIP patch tokens [1, 729, 1152]. Returns zeros if model is None."""
    if model is None:
        return mx.zeros((1, num_patches, siglip_dim), dtype=mx.bfloat16)
    import torch
    inputs = proc(images=[pil_img], return_tensors="pt")
    with torch.no_grad():
        out = model.vision_model(**inputs)
    feats = out.last_hidden_state.numpy().astype(np.float32)
    return mx.array(feats).astype(mx.bfloat16)


def _clip_i(model, proc, ref_pil, gen_pil) -> Optional[float]:
    """CLIP-I: cosine similarity between SigLIP global image embeddings."""
    if model is None:
        return None
    try:
        import torch
        inp = proc(images=[ref_pil, gen_pil], return_tensors="pt")
        with torch.no_grad():
            out = model.get_image_features(**inp)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return float((feats[0] * feats[1]).sum())
    except Exception as e:
        print(f"  WARN:  CLIP-I computation failed: {e}", file=sys.stderr)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Text encoding
# ─────────────────────────────────────────────────────────────────────────────

def _encode_text(flux, prompt: str) -> mx.array:
    tokenizer = flux.tokenizers["qwen3"]
    tokens    = tokenizer.tokenize([prompt])
    embeds    = flux.text_encoder.get_prompt_embeds(
        input_ids=tokens.input_ids,
        attention_mask=tokens.attention_mask,
        hidden_state_layers=(9, 18, 27),
    )
    return embeds.astype(mx.bfloat16)


# ─────────────────────────────────────────────────────────────────────────────
# Euler denoising (inlined from eval.py to avoid shadowing builtin `eval`)
# ─────────────────────────────────────────────────────────────────────────────

def _euler_step(
    x_t: mx.array, t: int, t_next: int, v_pred: mx.array,
) -> mx.array:
    alpha_t = 1.0 - t      / 1000.0
    sigma_t =       t      / 1000.0
    alpha_n = 1.0 - t_next / 1000.0
    sigma_n =       t_next / 1000.0
    denom = max(alpha_t ** 2 + sigma_t ** 2, 1e-6)
    x0    = (alpha_t * x_t.astype(mx.float32) - sigma_t * v_pred.astype(mx.float32)) / denom
    if sigma_t < 1e-6:
        return x0.astype(x_t.dtype)
    noise  = (x_t.astype(mx.float32) - alpha_t * x0) / sigma_t
    x_next = alpha_n * x0 + sigma_n * noise
    return x_next.astype(x_t.dtype)


def _ip_forward(
    flux,
    adapter: IPAdapterKlein,
    noisy_latents: mx.array,
    text_embeds: mx.array,
    t_int: mx.array,
    k_ip_all: mx.array,
    v_ip_all: mx.array,
    ip_scale: mx.array,
) -> mx.array:
    """
    Inference forward pass matching the training computation exactly.

    Training uses _flux_forward_no_ip (runs Flux fully, collects Q at each block)
    then sums all IP contributions and adds to h_final before norm_out/proj_out.
    This function replicates that computation for correct inference.
    """
    flux_state = _flux_forward_no_ip(flux, noisy_latents, text_embeds, t_int)

    qs      = flux_state["qs"]
    h_final = flux_state["h_final"]
    temb    = flux_state["temb"]
    B       = flux_state["B"]
    C       = flux_state["C"]
    Lh      = flux_state["Lh"]
    Lw      = flux_state["Lw"]
    pH      = flux_state["pH"]
    pW      = flux_state["pW"]
    seq_img = flux_state["seq_img"]
    d_inner = h_final.shape[2]

    ip_total = mx.zeros((B, seq_img, d_inner), dtype=h_final.dtype)
    for i, q_i in enumerate(qs):
        H_i  = q_i.shape[1]
        Hd_i = q_i.shape[3]
        k_i  = k_ip_all[:, i].reshape(B, -1, H_i, Hd_i).transpose(0, 2, 1, 3)
        v_i  = v_ip_all[:, i].reshape(B, -1, H_i, Hd_i).transpose(0, 2, 1, 3)
        ip_out = mx.fast.scaled_dot_product_attention(
            q_i, k_i, v_i, scale=Hd_i ** -0.5,
        )
        ip_out = ip_out.transpose(0, 2, 1, 3).reshape(B, seq_img, d_inner)
        ip_total = ip_total + ip_scale[i] * ip_out

    tr = flux.transformer
    h_with_ip = tr.norm_out(h_final + ip_total, temb)
    pred_seq  = tr.proj_out(h_with_ip)

    pred = pred_seq.transpose(0, 2, 1).reshape(B, C * 4, pH, pW)
    pred = pred.reshape(B, C, 2, 2, pH, pW)
    pred = pred.transpose(0, 1, 4, 2, 5, 3)
    pred = pred.reshape(B, C, Lh, Lw)
    return pred


def _generate(
    flux,
    adapter: IPAdapterKlein,
    text_embeds: mx.array,
    siglip_feats: mx.array,
    width: int,
    height: int,
    n_steps: int,
    seed: int,
    sref_strength: float,
    style_only: bool = False,
) -> mx.array:
    """
    Full Euler denoising loop with IP-Adapter.

    Uses the training-matching inference approach: run Flux without IP to
    collect Q vectors and h_final, then sum IP contributions at h_final.
    sref_strength=0.0 gives a numerically exact baseline (ip_scale all zeros).
    """
    mx.random.seed(seed)
    H_lat, W_lat = height // 8, width // 8
    x = mx.random.normal((1, 32, H_lat, W_lat)).astype(mx.bfloat16)

    ip_embeds = adapter.get_image_embeds(siglip_feats)
    k_ip, v_ip = adapter.get_kv_all(ip_embeds)
    ip_scale = adapter.effective_scale(style_only=style_only, sref_strength=sref_strength)
    mx.eval(ip_embeds, k_ip, v_ip, ip_scale)

    timesteps = [int(1000 * (1 - i / n_steps)) for i in range(n_steps + 1)]
    for t_curr, t_next in zip(timesteps[:-1], timesteps[1:]):
        t_arr  = mx.array([t_curr], dtype=mx.int32)
        v_pred = _ip_forward(flux, adapter, x, text_embeds, t_arr, k_ip, v_ip, ip_scale)
        mx.eval(v_pred)
        x = _euler_step(x, t_curr, t_next, v_pred)
        mx.eval(x)

    return x


# ─────────────────────────────────────────────────────────────────────────────
# Image I/O
# ─────────────────────────────────────────────────────────────────────────────

def _decode(vae, latents: mx.array) -> np.ndarray:
    img = vae.decode(latents)
    mx.eval(img)
    arr = np.array(img[0].astype(mx.float32)).transpose(1, 2, 0)  # [H, W, 3]
    return np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)


def _make_comparison(
    style_pil,
    baseline: np.ndarray,
    adapter_np: np.ndarray,
    path: str,
) -> None:
    from PIL import Image, ImageDraw
    H, W     = baseline.shape[:2]
    style_rs = np.array(style_pil.resize((W, H), Image.LANCZOS).convert("RGB"))
    sep      = np.full((H, 4, 3), 64, dtype=np.uint8)
    canvas   = np.concatenate([style_rs, sep, baseline, sep, adapter_np], axis=1)
    img      = Image.fromarray(canvas)
    draw     = ImageDraw.Draw(img)
    label_y  = max(0, H - 18)
    for text, x in [("style ref", 4), ("baseline", W + 8), ("adapter", W * 2 + 12)]:
        draw.text((x, label_y), text, fill=(255, 255, 255))
    img.save(path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate an exported IP-Adapter bundle via side-by-side inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--adapter", required=True,
                   help="Directory with adapter_weights.safetensors + adapter_meta.json")
    p.add_argument("--style-image", required=True,
                   help="Path to style reference image")
    p.add_argument("--prompt", required=True,
                   help="Text prompt for both baseline and adapter generation")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--flux-model",
                   help="Path to Flux Klein model directory")
    g.add_argument("--config",
                   help="Training config YAML (reads model.flux_model_dir)")

    p.add_argument("--strength",     type=float, default=1.0,
                   help="IP-Adapter style strength (default: 1.0)")
    p.add_argument("--steps",        type=int,   default=4,
                   help="Denoising steps (default: 4 for distilled Flux Klein)")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--width",        type=int,   default=512)
    p.add_argument("--height",       type=int,   default=512)
    p.add_argument("--output-dir",   default="/tmp/ip_adapter_test",
                   help="Output directory (default: /tmp/ip_adapter_test)")
    p.add_argument("--style-only",   action="store_true",
                   help="Zero double-stream ip_scale (style-only inference mode)")
    p.add_argument("--siglip-model", default="google/siglip-so400m-patch14-384",
                   help="SigLIP model for style encoding and CLIP-I scoring")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # Resolve Flux model path
    if args.flux_model:
        flux_model_dir = args.flux_model
    else:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        flux_model_dir = cfg["model"]["flux_model_dir"]

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load bundle ─────────────────────────────────────────────────────────
    print("\n=== IP-Adapter Inference Test ===")
    print(f"\nLoading adapter bundle: {args.adapter}")
    meta, adapter = load_bundle(args.adapter)
    print_meta(meta)
    print_ip_scale_stats(adapter, meta)

    # ── 2. Load Flux ───────────────────────────────────────────────────────────
    if not _HAS_MFLUX:
        print("Error: mflux not installed. Run: pip install mflux", file=sys.stderr)
        sys.exit(1)
    print(f"\nLoading Flux Klein: {flux_model_dir}")
    t0   = time.monotonic()
    flux = Flux2Klein(model_path=flux_model_dir, quantize=None)
    print(f"  loaded in {time.monotonic() - t0:.1f}s")

    # ── 3. Load SigLIP ─────────────────────────────────────────────────────────
    print(f"\nLoading SigLIP ({args.siglip_model}) ...")
    siglip_model, siglip_proc = _load_siglip(args.siglip_model)
    if siglip_model is not None:
        print("  loaded")

    # ── 4. Style image → SigLIP features ──────────────────────────────────────
    from PIL import Image
    style_pil = Image.open(args.style_image).convert("RGB")
    print(f"\nStyle image: {args.style_image}  ({style_pil.size[0]}×{style_pil.size[1]})")
    if siglip_model is None:
        print("  WARN:  using zero SigLIP features (transformers unavailable) — adapter output will not reflect style")

    siglip_feats = _siglip_feats(
        siglip_model, siglip_proc, style_pil,
        siglip_dim=meta["siglip_dim"],
    )
    mx.eval(siglip_feats)

    # ── 5. Perceiver token diagnostics ────────────────────────────────────────
    print("\nRunning perceiver ...")
    ip_embeds_diag      = adapter.get_image_embeds(siglip_feats)
    k_ip_diag, v_ip_diag = adapter.get_kv_all(ip_embeds_diag)
    mx.eval(ip_embeds_diag, k_ip_diag)
    print_token_stats(ip_embeds_diag, k_ip_diag)
    del ip_embeds_diag, k_ip_diag, v_ip_diag

    # ── 6. Encode text ─────────────────────────────────────────────────────────
    print(f"\nPrompt: {args.prompt!r}")
    print("Encoding text ...")
    text_embeds = _encode_text(flux, args.prompt)
    mx.eval(text_embeds)
    print(f"  text embeds: {list(text_embeds.shape)}")

    # ── 7. Baseline generation (sref_strength=0 → numerically exact no-IP run) ─
    print(f"\nGenerating baseline  (strength=0.0, seed={args.seed}, {args.steps} steps) ...")
    t0 = time.monotonic()
    x_base = _generate(
        flux, adapter, text_embeds, siglip_feats,
        args.width, args.height, args.steps, args.seed, 0.0,
    )
    mx.eval(x_base)
    t_base = time.monotonic() - t0
    print(f"  done in {t_base:.1f}s")

    # ── 8. Adapter generation ──────────────────────────────────────────────────
    print(f"\nGenerating with adapter (strength={args.strength}, seed={args.seed}, {args.steps} steps) ...")
    t0 = time.monotonic()
    x_adapter = _generate(
        flux, adapter, text_embeds, siglip_feats,
        args.width, args.height, args.steps, args.seed,
        args.strength, style_only=args.style_only,
    )
    mx.eval(x_adapter)
    t_adapter = time.monotonic() - t0
    print(f"  done in {t_adapter:.1f}s")

    # ── 9. Decode + save ───────────────────────────────────────────────────────
    print("\nDecoding and saving images ...")
    vae         = flux.vae
    base_arr    = _decode(vae, x_base)
    adapter_arr = _decode(vae, x_adapter)

    base_path    = os.path.join(args.output_dir, "baseline.png")
    adapter_path = os.path.join(args.output_dir, "adapter.png")
    comp_path    = os.path.join(args.output_dir, "comparison.png")

    from PIL import Image as _PIL
    _PIL.fromarray(base_arr).save(base_path)
    _PIL.fromarray(adapter_arr).save(adapter_path)
    _make_comparison(style_pil, base_arr, adapter_arr, comp_path)

    print(f"  → {base_path}")
    print(f"  → {adapter_path}")
    print(f"  → {comp_path}")

    # ── 10. CLIP-I ─────────────────────────────────────────────────────────────
    clip_i: Optional[float] = None
    if siglip_model is not None:
        print("\nComputing CLIP-I (style image vs adapter output) ...")
        adapter_pil = _PIL.fromarray(adapter_arr)
        clip_i = _clip_i(siglip_model, siglip_proc, style_pil, adapter_pil)
        if clip_i is not None:
            if clip_i < 0.5:
                flag = "  WARN: low similarity — adapter may not be transferring style effectively"
            elif clip_i > 0.85:
                flag = "  (high — strong style match)"
            else:
                flag = ""
            print(f"  CLIP-I = {clip_i:.4f}{flag}")

    # ── 11. Summary ────────────────────────────────────────────────────────────
    overhead = t_adapter - t_base
    ovhd_pct = 100.0 * overhead / max(t_base, 0.01)
    print("\n=== Summary ===")
    print(f"Baseline time:   {t_base:.1f}s")
    print(f"Adapter time:    {t_adapter:.1f}s  (overhead: +{overhead:.1f}s, {ovhd_pct:+.0f}%)")
    if clip_i is not None:
        print(f"CLIP-I:          {clip_i:.4f}  (style similarity: reference vs adapter output)")
    print(f"Output dir:      {args.output_dir}/")
    print(f"  baseline.png   — Flux generation without adapter (strength=0)")
    print(f"  adapter.png    — Flux generation with adapter (strength={args.strength})")
    print(f"  comparison.png — style ref | baseline | adapter (side-by-side)")


if __name__ == "__main__":
    main()
