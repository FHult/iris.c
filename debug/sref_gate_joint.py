#!/usr/bin/env python3
"""debug/sref_gate_joint.py — Stage-0.5 RENDER GATE for the joint-backbone probe.

The in-loop gate (probe_joint_contrastive.py) proved the model DISCRIMINATES references in
projector space. This gate confirms the discrimination shows up as visible STYLE in generated
images — the check the plan (plans/sref-joint-backbone-project.md §6, gate 2) requires before any
cloud spend, and the exact check the 3 prior collapsed adapters failed.

Unlike debug/sref_infer_film.py (frozen base + csd_mod only — the FILM-only rail that collapsed,
SREF-FILM-1), this injects the TRAINED LoRA r64 backbone AND the CSDModulation from the joint
checkpoint, so the render uses the SAME forward the probe trained (`flux_forward_film` on the
LoRA-injected transformer + csd_mod temb injection).

    style ref ─► CSD(ref) ─► [LoRA-injected DiT ; temb += csd_mod(csd)] denoise on 4B BASE
              ─► un-BN (inverse of VAE-Q1 bn_pack) ─► vae.decode ─► PNG

Two numbers gate it (plan §6):
  cross-ref output corr  < 0.90   (refs produce DIFFERENT images — not reference-inert)
  styleCSD Δ            > 0.009   (the reference's LOOK transferred; via sref_scorecard --score-only)

Usage:
  train/.venv/bin/python debug/sref_gate_joint.py --ckpt <joint_probe_0007000.safetensors> --label joint_7000
  train/.venv/bin/python debug/sref_scorecard.py  --label joint_7000 --score-only
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "train" / "scripts"))
sys.path.insert(0, str(ROOT / "debug"))

import mlx.core as mx
from mlx.utils import tree_unflatten
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

from ip_adapter.model import CSDModulation
from lora.lora import inject_lora_double_blocks, inject_lora_single_blocks
from lora.probe_joint_contrastive import _Joint            # same container the ckpt was saved from
from eval import _decode_latents, _save_png
# reuse the EXACT render loop + CSD encoder + BN/corr helpers — no divergence from the FILM gate
from sref_infer_film import _CSD, generate_film, EVAL_SET
from sref_infer_style import _load_vae_bn_stats, _bn_unpack, _corr
from train_ip_adapter import _encode_text, _TextEncoderBundle


class _ScaledCSDMod:
    """Wraps csd_mod so the temb injection becomes `temb += scale * csd_mod(csd)` — an INFERENCE-time
    style strength knob (SREF-JOINT-V2-CONTENT: content-vs-style is not a train-time t-band slider; it's
    this). scale=1.0 == the trained model; lower lets the base model's prompt-following reassert. Only
    __call__ is used by flux_forward_film, so a plain callable suffices."""
    def __init__(self, mod, scale):
        self.mod, self.scale = mod, scale
    def __call__(self, x):
        return self.scale * self.mod(x)


def load_joint(model_dir, ckpt_path, rank, csd_dim, mlp_dim):
    """Base flux + injected LoRA(rank) + CSDModulation, with BOTH weight groups loaded from the
    joint checkpoint (flux.* -> LoRA, csd_mod.* -> module). Frozen for inference."""
    flux = Flux2Klein(model_path=model_dir, quantize=None)
    inject_lora_double_blocks(flux, rank=rank)
    inject_lora_single_blocks(flux, rank=rank)
    csd_mod = CSDModulation(hidden_dim=3072, csd_dim=csd_dim, mlp_dim=mlp_dim)
    joint = _Joint(flux, csd_mod)
    ckpt = list(mx.load(ckpt_path).items())
    joint.update(tree_unflatten(ckpt))                     # applies trainable params onto the module tree
    flux.freeze(); csd_mod.freeze(); csd_mod.eval()
    mx.eval(flux.parameters(), csd_mod.parameters())
    n_flux = sum(1 for k, _ in ckpt if k.startswith("flux"))
    n_csd = sum(1 for k, _ in ckpt if k.startswith("csd_mod"))
    print(f"  loaded joint ckpt: {n_flux} LoRA tensors + {n_csd} csd_mod tensors", flush=True)
    return flux, csd_mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="joint_probe_*.safetensors (LoRA + CSDModulation)")
    ap.add_argument("--label", required=True, help="scorecard label (output subdir name)")
    ap.add_argument("--model", default="flux-klein-4b-base")
    ap.add_argument("--out-dir", default="/tmp/sref_scorecard")
    ap.add_argument("--refs", nargs="*", default=None, help="subset of ref ids (default: all)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--guidance", type=float, default=3.5)     # matches the training config
    ap.add_argument("--style-scale", type=float, default=1.0,
                    help="INFERENCE style strength: temb += scale*csd_mod(csd). 1.0=trained; <1 lifts content.")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--csd-dim", type=int, default=768)
    ap.add_argument("--mlp-dim", type=int, default=1024)
    ap.add_argument("--regen", action="store_true")
    args = ap.parse_args()

    spec = json.loads(EVAL_SET.read_text())
    prompt, seed = spec["eval_prompt"], spec["eval_seed"]
    refs = spec["refs"]
    if args.refs:
        refs = [r for r in refs if r["id"] in args.refs]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / args.label; md.mkdir(exist_ok=True)

    model_dir = str(ROOT / args.model)
    print(f"loading Flux2Klein ({args.model}) + LoRA r{args.rank} ...", flush=True)
    t0 = time.time()
    flux, csd_mod = load_joint(model_dir, args.ckpt, args.rank, args.csd_dim, args.mlp_dim)
    if args.style_scale != 1.0:
        csd_mod = _ScaledCSDMod(csd_mod, args.style_scale)
        print(f"  style-scale = {args.style_scale}", flush=True)
    print(f"  ready in {time.time()-t0:.1f}s", flush=True)
    bn_m, bn_s = _load_vae_bn_stats(model_dir)
    csd = _CSD()
    text_enc = _TextEncoderBundle(flux.text_encoder, flux.tokenizers["qwen3"])
    text_embeds = _encode_text(text_enc, [prompt]); mx.eval(text_embeds)
    print(f"gate: {Path(args.ckpt).name}   prompt={prompt!r}  seed={seed}  "
          f"steps={args.steps}  guidance={args.guidance}  size={args.size}", flush=True)

    # ── null-style baseline (zeroed CSD) — the reference's effect is measured vs this ──
    base_png = out_dir / f"_baseline_s{seed}.png"
    if not base_png.exists() or args.regen:
        lat = generate_film(flux, csd_mod, text_embeds, mx.zeros((1, args.csd_dim), dtype=mx.bfloat16),
                            args.size, args.size, args.steps, seed, args.guidance, bn_m, bn_s, null=True)
        arr = _decode_latents(flux.vae, _bn_unpack(lat, bn_m, bn_s).astype(mx.bfloat16))
        _save_png(str(base_png), arr)
        print(f"  baseline -> {base_png.name}", flush=True)

    # ── one render per reference (seed+prompt fixed; only the ref varies) ──
    outs = {}
    for r in refs:
        op = md / f"{r['id']}.png"
        if op.exists() and not args.regen:
            outs[r["id"]] = (r["type"], np.asarray(Image.open(op).convert("RGB")))
            continue
        cv = csd.feats(r["path"])
        lat = generate_film(flux, csd_mod, text_embeds, cv, args.size, args.size,
                            args.steps, seed, args.guidance, bn_m, bn_s, null=False)
        arr = _decode_latents(flux.vae, _bn_unpack(lat, bn_m, bn_s).astype(mx.bfloat16))
        _save_png(str(op), arr)
        outs[r["id"]] = (r["type"], arr)
        print(f"  {r['id']:<20} ({r['type']}) -> {op.name}", flush=True)
        mx.clear_cache()

    # ── GATE 2a: cross-reference output correlation (COLLAPSE check) ──
    ids = list(outs.keys())
    n = len(ids)
    if n >= 2:
        cc = np.array([_corr(outs[ids[i]][1], outs[ids[j]][1])
                       for i in range(n) for j in range(i + 1, n)])
        print(f"\n=== GATE 2a — cross-ref output corr ({n} refs, {len(cc)} pairs; seed+prompt fixed) ===")
        print(f"cross-ref output corr:  mean {cc.mean():.4f}   max {cc.max():.4f}   min {cc.min():.4f}")
        print("  PASS (< 0.90 => refs discriminate)" if cc.max() < 0.90
              else "  FAIL (>= 0.90 => reference-inert collapse)")
    print(f"\nNext: train/.venv/bin/python debug/sref_scorecard.py --label {args.label} --score-only "
          f"--model {args.model}   # GATE 2b: styleCSD Δ > 0.009", flush=True)


if __name__ == "__main__":
    main()
