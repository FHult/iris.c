#!/usr/bin/env python3
"""debug/sref_infer_film.py — SREF experiment B inference / gate harness (CSD→modulation FiLM).

Companion to debug/sref_infer_style.py (the CLOSED in-sequence style-token gate). Renders the frozen
SREF eval set through the CSD→timestep-modulation path so debug/sref_scorecard.py can GATE a
CSDModulation checkpoint:

    style ref ─► CSD encoder (SAME as the training cache) ─► 768-d L2 style vector
              ─► flux_forward_film  (temb += CSDModulation(csd); DiT frozen)  denoise on the 4B BASE
              ─► un-BN (inverse of the trainer's VAE-Q1 _bn_pack) ─► vae.decode ─► PNG

Two commands to gate (identical layout to the style-token harness):
    train/.venv/bin/python debug/sref_infer_film.py --ckpt <style_film.safetensors> --label film_4000
    train/.venv/bin/python debug/sref_scorecard.py  --label film_4000 --score-only

Also prints the hard-kill COLLAPSE metric in-process (seed+prompt fixed; only the ref varies → if
outputs still correlate ≥ ~0.90 the module went reference-inert). Unlike the style-token gate this
uses NO CFG doubling: the CSD is injected into `temb`, which is applied at every step/every block, so
the conditioning is always on — a single forward is faithful. Run with the train venv (MLX).
"""
import argparse, json, sys, os, time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "train" / "scripts"))
sys.path.insert(0, str(ROOT / "debug"))

import mlx.core as mx
from mlx.utils import tree_unflatten
from mflux.models.flux2 import Flux2Klein

from ip_adapter.model import CSDModulation
from lora.film_step import flux_forward_film
from lora.train_step import patchify_pack, unpatchify, make_position_ids
from train_ip_adapter import _encode_text, _TextEncoderBundle
from eval import _euler_step, _decode_latents, _save_png
from style_encoder.csd_mlx import CSDStyleEncoder, preprocess

# reuse the exact BN + correlation helpers from the style-token harness (parity, no divergence)
from sref_infer_style import _load_vae_bn_stats, _bn_unpack, _corr

EVAL_SET = ROOT / "debug" / "sref_eval_set.json"
CSD_WEIGHTS = "/Volumes/2TBSSD/models/csd_vit_l_style.safetensors"


class _CSD:
    """CSD style vector — SAME encoder + preprocessing as the training cache (parity)."""
    def __init__(self, weights=CSD_WEIGHTS):
        self.enc = CSDStyleEncoder(weights)

    def feats(self, image_path):
        img = Image.open(image_path).convert("RGB")
        x = preprocess(img)                          # [3,224,224] f32, CLIP-normalised
        style = self.enc.encode(np.stack([x]))       # [1,768] L2-normalised (== cache)
        return mx.array(np.asarray(style)[0][None], dtype=mx.bfloat16)   # [1,768]


def load_film(ckpt_path, csd_dim, mlp_dim):
    mod = CSDModulation(hidden_dim=3072, csd_dim=csd_dim, mlp_dim=mlp_dim)
    mod.update(tree_unflatten(list(mx.load(ckpt_path).items())))
    mod.eval(); mod.freeze()
    mx.eval(mod.parameters())
    return mod


def generate_film(flux, csd_mod, text_embeds, csd_vec, W, H, steps, seed, guidance,
                  bn_m, bn_s, null=False):
    """Full Euler denoise with CSD FiLM'd into temb. Returns a BN-packed latent [1,32,Lh,Lw]."""
    mx.random.seed(seed)
    Lh, Lw = H // 8, W // 8
    x = mx.random.normal((1, 32, Lh, Lw)).astype(mx.bfloat16)
    csd = mx.zeros_like(csd_vec) if null else csd_vec
    img_ids, txt_ids = make_position_ids(Lh // 2, Lw // 2, text_embeds.shape[1])
    timesteps = [int(1000 * (1 - i / steps)) for i in range(steps + 1)]
    for t_curr, t_next in zip(timesteps[:-1], timesteps[1:]):
        hidden = patchify_pack(x.astype(text_embeds.dtype))
        t_arr = mx.array([t_curr], dtype=mx.int32)
        pred_seq = flux_forward_film(flux.transformer, hidden, text_embeds, csd_mod, csd, t_arr,
                                     img_ids, txt_ids, None, None, guidance)
        v = unpatchify(pred_seq, 1, 32, Lh, Lw)
        mx.eval(v)
        x = _euler_step(x, t_curr, t_next, v)
        mx.eval(x)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="CSDModulation .safetensors checkpoint")
    ap.add_argument("--label", required=True, help="scorecard label (output subdir name)")
    ap.add_argument("--model", default="flux-klein-4b-base", help="flux model dir (train/gate on BASE)")
    ap.add_argument("--out-dir", default="/tmp/sref_scorecard")
    ap.add_argument("--refs", nargs="*", default=None, help="subset of ref ids (default: all)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--guidance", type=float, default=3.5)
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
    print(f"loading Flux2Klein ({args.model}) ...", flush=True)
    t0 = time.time()
    flux = Flux2Klein(model_path=model_dir, quantize=None); flux.freeze()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
    bn_m, bn_s = _load_vae_bn_stats(model_dir)
    csd_mod = load_film(args.ckpt, args.csd_dim, args.mlp_dim)
    csd = _CSD()
    text_enc = _TextEncoderBundle(flux.text_encoder, flux.tokenizers["qwen3"])
    text_embeds = _encode_text(text_enc, [prompt]); mx.eval(text_embeds)
    print(f"film: {Path(args.ckpt).name}   prompt={prompt!r}  seed={seed}  "
          f"steps={args.steps}  guidance={args.guidance}", flush=True)

    # ── null-style baseline (zeroed CSD) — the reference's effect is measured vs this ──
    base_png = out_dir / f"_baseline_s{seed}.png"
    if not base_png.exists() or args.regen:
        lat = generate_film(flux, csd_mod, text_embeds, mx.zeros((1, args.csd_dim), dtype=mx.bfloat16),
                            args.size, args.size, args.steps, seed, args.guidance, bn_m, bn_s, null=True)
        arr = _decode_latents(flux.vae, _bn_unpack(lat, bn_m, bn_s).astype(mx.bfloat16))
        _save_png(str(base_png), arr)
        print(f"  baseline -> {base_png.name}", flush=True)

    # ── one render per reference ──
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

    # ── COLLAPSE metric: cross-reference output correlation (seed+prompt fixed) ──
    ids = list(outs.keys())
    n = len(ids)
    if n >= 2:
        cc = []
        for i in range(n):
            for j in range(i + 1, n):
                cc.append(_corr(outs[ids[i]][1], outs[ids[j]][1]))
        cc = np.array(cc)
        print(f"\n=== COLLAPSE check ({n} refs, {len(cc)} pairs; seed+prompt fixed) ===")
        print(f"cross-ref output corr:  mean {cc.mean():.4f}   max {cc.max():.4f}   min {cc.min():.4f}")
        verdict = ("COLLAPSE — reference-inert (>=0.90); STOP, log a negative result"
                   if cc.max() >= 0.90 else
                   "DISCRIMINATES references (max < 0.90) — proceed to the scorecard gate")
        print(f"verdict: {verdict}")
    print(f"\nrenders in {md}  —  now gate with:\n"
          f"  train/.venv/bin/python debug/sref_scorecard.py --label {args.label} --score-only", flush=True)


if __name__ == "__main__":
    main()
