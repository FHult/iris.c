#!/usr/bin/env python3
"""debug/sref_infer_style.py — SREF Phase-1 inference / gate harness (learned style projector).

The learned in-sequence style rail does NOT exist in the C `iris` binary yet (that's the Phase-4 C
port). It lives only in Python/MLX. This harness renders the frozen SREF eval set
(debug/sref_eval_set.json) through that path so debug/sref_scorecard.py can GATE a projector
checkpoint:

    style ref ─► SigLIP (precompute-parity preprocessing) ─► StyleProjector(ckpt) ─► 192 style tokens
              ─► flux_forward_style  (TEXT | STYLE | IMAGE, text-like RoPE)  denoise on the 4B BASE
              ─► un-BN (inverse of the trainer's VAE-Q1 _bn_pack) ─► vae.decode ─► PNG

It writes outputs in the EXACT filename layout the scorecard consumes, so the gate is two commands:

    train/.venv/bin/python debug/sref_infer_style.py --ckpt <projector.safetensors> --label learned_4000
    train/.venv/bin/python debug/sref_scorecard.py   --label learned_4000 --score-only

It also computes the hard-kill COLLAPSE metric in-process. Seed + prompt are held constant across all
references, so the ONLY thing varying is the reference: if outputs still correlate ≥ ~0.90 the projector
went reference-inert (the SREF-CHAMPION-COLLAPSE failure) → stop, log a negative result, don't bother
scoring style. Below the line = the reference changes the output = discriminating (necessary, not
sufficient — style quality is the scorecard's job).

Train↔infer parity (CLAUDE.md protocol): reuses the SAME modules the trainer used — StyleProjector,
lora.style_step.flux_forward_style, patchify/unpatchify/position-ids, the VAE-Q1 BN latent space
(un-BN'd here, the exact inverse of train_style_projector._bn_pack), Qwen3 _encode_text, and the
guidance EMBEDDING with a single forward per step (NOT CFG-doubled — matching how the projector was
trained; see NOTE below). SigLIP uses precompute_all._preprocess_siglip so the features match the
cache the projector trained on. Run with the train venv (MLX): train/.venv/bin/python debug/sref_infer_style.py ...

NOTE (guidance): training passes `guidance` into tr.time_guidance_embed and runs ONE forward. This
harness does the same for train↔infer consistency. If base renders look washed out, revisit true CFG
(two forwards) — but then the trainer must match. Logged as an open question in plans/sref-phase1-projector.md.
"""
import argparse, io, json, sys, os, time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "train" / "scripts"))

import mlx.core as mx
from mlx.utils import tree_unflatten
from mflux.models.flux2 import Flux2Klein

from ip_adapter.model import StyleProjector
from lora.style_step import flux_forward_style, _style_ids
from lora.train_step import patchify_pack, unpatchify, make_position_ids
from train_ip_adapter import _encode_text, _TextEncoderBundle
from eval import _euler_step, _decode_latents, _save_png

EVAL_SET = ROOT / "debug" / "sref_eval_set.json"


# ── VAE-Q1 BN stats (same load as the trainer) + the INVERSE of _bn_pack ──────
def _load_vae_bn_stats(flux_model_dir):
    st = mx.load(os.path.join(flux_model_dir, "vae", "diffusion_pytorch_model.safetensors"))
    m = st["bn.running_mean"].astype(mx.float32).reshape(32, 2, 2)
    s = mx.sqrt(st["bn.running_var"].astype(mx.float32).reshape(32, 2, 2) + 1e-4)
    mx.eval(m, s)
    return m, s


def _bn_unpack(x, m, s):
    """Inverse of train_style_projector._bn_pack: BN-packed latent (std≈1) → raw VAE latent (std≈1.7),
    which is what flux.vae.decode expects (equivalent to vae.decode_packed_latents' un-BN, in the
    un-patchified 32ch layout)."""
    _, _, Lh, Lw = x.shape
    return (x.astype(mx.float32) * mx.tile(s, (1, Lh // 2, Lw // 2))
            + mx.tile(m, (1, Lh // 2, Lw // 2)))


# ── SigLIP features (precompute-parity: same preprocessing + tower as the cache) ──
class _Siglip:
    def __init__(self):
        from precompute_all import _preprocess_siglip
        import torch
        from transformers import AutoModel
        self._pre = _preprocess_siglip
        self._torch = torch
        self._dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self._m = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").vision_model.eval().to(self._dev)

    def feats(self, image_path):
        raw = Path(image_path).read_bytes()
        try:
            img = self._pre(raw)                   # JPEG path (TurboJPEG) — exact precompute parity
        except OSError:
            # non-JPEG eval refs (e.g. PNG): TurboJPEG rejects them; decode with PIL and apply the
            # IDENTICAL resize + SigLIP normalize as _preprocess_siglip so features still match.
            from precompute_all import _resize, SIGLIP_IMAGE_SIZE, _SIGLIP_MEAN, _SIGLIP_STD
            arr = _resize(np.array(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8),
                          SIGLIP_IMAGE_SIZE)
            img = ((arr.astype(np.float32) / 255.0 - _SIGLIP_MEAN) / _SIGLIP_STD).transpose(2, 0, 1)[None]
        t = self._torch
        with t.no_grad():
            out = self._m(pixel_values=t.from_numpy(img).to(self._dev))
        f = out.last_hidden_state[0].float().cpu().numpy()   # [729,1152]
        return mx.array(f[None], dtype=mx.bfloat16)          # [1,729,1152]


def load_projector(ckpt_path, n_style):
    projector = StyleProjector(hidden_dim=3072, num_heads=24, num_style_tokens=n_style, siglip_dim=1152)
    weights = list(mx.load(ckpt_path).items())
    projector.update(tree_unflatten(weights))
    projector.eval()
    projector.freeze()
    mx.eval(projector.parameters())
    return projector


def generate_style(flux, projector, text_embeds, siglip_feats, W, H, steps, seed, guidance,
                   bn_m, bn_s, null=False):
    """Full Euler denoise with in-sequence style tokens. Returns a BN-packed latent [1,32,Lh,Lw]."""
    mx.random.seed(seed)
    Lh, Lw = H // 8, W // 8
    x = mx.random.normal((1, 32, Lh, Lw)).astype(mx.bfloat16)
    sig = mx.zeros_like(siglip_feats) if null else siglip_feats
    style_tokens = projector(sig)                                        # [1, n_style, hidden]
    style_ids = _style_ids(text_embeds.shape[1], style_tokens.shape[1])
    img_ids, txt_ids = make_position_ids(Lh // 2, Lw // 2, text_embeds.shape[1])
    mx.eval(style_tokens)
    timesteps = [int(1000 * (1 - i / steps)) for i in range(steps + 1)]
    for t_curr, t_next in zip(timesteps[:-1], timesteps[1:]):
        hidden = patchify_pack(x.astype(text_embeds.dtype))
        t_arr = mx.array([t_curr], dtype=mx.int32)
        pred_seq = flux_forward_style(flux.transformer, hidden, text_embeds, style_tokens, t_arr,
                                      img_ids, txt_ids, style_ids, None, None, guidance)
        v = unpatchify(pred_seq, 1, 32, Lh, Lw)
        mx.eval(v)
        x = _euler_step(x, t_curr, t_next, v)
        mx.eval(x)
    return x


def _corr(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="StyleProjector .safetensors checkpoint")
    ap.add_argument("--label", required=True, help="scorecard label (output subdir name)")
    ap.add_argument("--model", default="flux-klein-4b-base", help="flux model dir (train on/gate on BASE)")
    ap.add_argument("--out-dir", default="/tmp/sref_scorecard")
    ap.add_argument("--refs", nargs="*", default=None, help="subset of ref ids (default: all)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=24, help="denoise steps (base is steerable; 24 default)")
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--n-style", type=int, default=192)
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
    flux = Flux2Klein(model_path=model_dir, quantize=None)
    flux.freeze()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
    bn_m, bn_s = _load_vae_bn_stats(model_dir)
    projector = load_projector(args.ckpt, args.n_style)
    siglip = _Siglip()
    text_enc = _TextEncoderBundle(flux.text_encoder, flux.tokenizers["qwen3"])
    text_embeds = _encode_text(text_enc, [prompt]); mx.eval(text_embeds)
    print(f"projector: {Path(args.ckpt).name}   prompt={prompt!r}  seed={seed}  "
          f"steps={args.steps}  guidance={args.guidance}", flush=True)

    # ── null-style baseline (zeroed SigLIP) — the reference's effect is measured vs this ──
    base_png = out_dir / f"_baseline_s{seed}.png"
    if not base_png.exists() or args.regen:
        lat = generate_style(flux, projector, text_embeds, mx.zeros((1, 729, 1152), dtype=mx.bfloat16),
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
        sig = siglip.feats(r["path"])
        lat = generate_style(flux, projector, text_embeds, sig, args.size, args.size,
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
