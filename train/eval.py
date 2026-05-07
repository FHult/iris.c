#!/usr/bin/env python3
"""
train/eval.py — Checkpoint evaluation for IP-Adapter training.

Generates images for each entry in eval_prompts.txt using the current
checkpoint, computes CLIP-I (style fidelity) and CLIP-T (prompt adherence)
via SigLIP SO400M, and writes eval_results.json + report.html.

Usage (standalone):
    python train/eval.py \\
        --checkpoint train/data/checkpoints/stage1/step_10000.safetensors \\
        --config train/configs/stage1_512px.yaml \\
        [--prompts train/configs/eval_prompts.txt] \\
        [--output train/data/checkpoints/stage1/eval/step_0010000/] \\
        [--steps 4] [--width 512] [--height 512] [--seed 42]

Called automatically from train_ip_adapter.py when eval.enabled=true in the
training config (every eval.every_steps steps).

CLIP metrics require 'transformers' and 'torch'. If unavailable, images are
still generated but CLIP-I and CLIP-T are omitted from the report.
"""

import argparse
import base64
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

sys.path.insert(0, str(Path(__file__).parent))
from ip_adapter.model import IPAdapterKlein
from ip_adapter.ema import _flatten
from train_ip_adapter import (
    _flux_forward_with_ip,
    _encode_text,
    _TextEncoderBundle,
)

try:
    from mflux.models.flux2 import Flux2Klein
    _HAS_MFLUX = True
except ImportError:
    _HAS_MFLUX = False


# ─────────────────────────────────────────────────────────────────────────────
# SigLIP (used for IP conditioning and CLIP metrics)
# ─────────────────────────────────────────────────────────────────────────────

def _load_siglip_full(model_name: str):
    """
    Load full SigLIP model (vision + text) via transformers.
    Returns (SiglipModel, SiglipProcessor) or (None, None) if unavailable.
    """
    try:
        from transformers import SiglipModel, SiglipProcessor
        model = SiglipModel.from_pretrained(model_name).eval()
        proc = SiglipProcessor.from_pretrained(model_name)
        return model, proc
    except ImportError:
        return None, None
    except Exception as e:
        print(f"  eval: cannot load SigLIP full model: {e}", file=sys.stderr)
        return None, None


def _siglip_vision_feats(model, processor, pil_img) -> Optional[np.ndarray]:
    """
    Extract SigLIP patch features [1, 729, 1152] as float32 numpy.
    pil_img: PIL.Image.Image (will be resized internally by processor).
    """
    try:
        import torch
        inputs = processor(images=[pil_img], return_tensors="pt")
        with torch.no_grad():
            out = model.vision_model(**inputs)
        # last_hidden_state: [1, 729, 1152] (SigLIP has no CLS token)
        return out.last_hidden_state.numpy().astype(np.float32)
    except Exception as e:
        print(f"  eval: SigLIP vision feats failed: {e}", file=sys.stderr)
        return None


def _compute_clip_i(model, processor, ref_pil, gen_pil) -> Optional[float]:
    """CLIP-I: cosine similarity between SigLIP image embeddings."""
    try:
        import torch
        inputs = processor(images=[ref_pil, gen_pil], return_tensors="pt")
        with torch.no_grad():
            feats = model.get_image_features(**inputs)  # [2, 1152]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return float((feats[0] * feats[1]).sum().item())
    except Exception as e:
        print(f"  eval: CLIP-I failed: {e}", file=sys.stderr)
        return None


def _compute_clip_t(model, processor, prompt: str, gen_pil) -> Optional[float]:
    """CLIP-T: cosine similarity between SigLIP text and generated image embeddings."""
    try:
        import torch
        t_inputs = processor(
            text=[prompt], return_tensors="pt",
            padding="max_length", truncation=True, max_length=64,
        )
        i_inputs = processor(images=[gen_pil], return_tensors="pt")
        with torch.no_grad():
            t_feat = model.get_text_features(**t_inputs)   # [1, 1152]
            i_feat = model.get_image_features(**i_inputs)  # [1, 1152]
        t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
        i_feat = i_feat / i_feat.norm(dim=-1, keepdim=True)
        return float((t_feat[0] * i_feat[0]).sum().item())
    except Exception as e:
        print(f"  eval: CLIP-T failed: {e}", file=sys.stderr)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Euler denoising
# ─────────────────────────────────────────────────────────────────────────────

def _euler_step(
    x_t: mx.array, t: int, t_next: int, v_pred: mx.array,
) -> mx.array:
    """
    One Euler step in v-prediction flow-matching latent space.

    x_t:    [B, C, H, W] current noisy latent
    t:      current integer timestep in [0, 1000]
    t_next: next integer timestep (lower = more denoised)
    v_pred: [B, C, H, W] predicted velocity from model

    Returns x at t_next.
    """
    alpha_t = 1.0 - t      / 1000.0
    sigma_t =       t      / 1000.0
    alpha_n = 1.0 - t_next / 1000.0
    sigma_n =       t_next / 1000.0

    # Unbiased x0 estimate: same formula as reconstruct_x0() in loss.py
    denom = max(alpha_t * alpha_t + sigma_t * sigma_t, 1e-6)
    x0 = (alpha_t * x_t.astype(mx.float32) - sigma_t * v_pred.astype(mx.float32)) / denom

    if sigma_t < 1e-6:
        return x0.astype(x_t.dtype)

    # Re-extract noise from current latent, then re-interpolate at t_next
    noise = (x_t.astype(mx.float32) - alpha_t * x0) / sigma_t
    x_next = alpha_n * x0 + sigma_n * noise
    return x_next.astype(x_t.dtype)


def _generate(
    flux,
    adapter: IPAdapterKlein,
    text_embeds: mx.array,      # [1, seq, 7680]
    siglip_feats: mx.array,     # [1, 729, 1152]
    width: int,
    height: int,
    n_steps: int,
    seed: int,
) -> mx.array:
    """
    Full Euler denoising loop with IP-Adapter.
    Returns latents [1, 32, H/8, W/8] bfloat16.
    """
    mx.random.seed(seed)
    H_lat, W_lat = height // 8, width // 8
    x = mx.random.normal((1, 32, H_lat, W_lat)).astype(mx.bfloat16)

    ip_embeds = adapter.get_image_embeds(siglip_feats)
    k_ip, v_ip = adapter.get_kv_all(ip_embeds)
    ip_scale = adapter.scale
    mx.eval(ip_embeds, k_ip, v_ip, ip_scale)

    # Uniform schedule: 1000 → 0 in n_steps steps
    timesteps = [int(1000 * (1 - i / n_steps)) for i in range(n_steps + 1)]

    for t_curr, t_next in zip(timesteps[:-1], timesteps[1:]):
        t_arr = mx.array([t_curr], dtype=mx.int32)
        v_pred = _flux_forward_with_ip(flux, x, text_embeds, t_arr, k_ip, v_ip, ip_scale)
        mx.eval(v_pred)
        x = _euler_step(x, t_curr, t_next, v_pred)
        mx.eval(x)

    return x


# ─────────────────────────────────────────────────────────────────────────────
# Image I/O
# ─────────────────────────────────────────────────────────────────────────────

def _load_pil(path: str, size: Optional[int] = None):
    """Load image as RGB PIL.Image, optionally resizing."""
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if size is not None:
            img = img.resize((size, size), Image.LANCZOS)
        return img
    except Exception as e:
        print(f"  eval: cannot load image {path}: {e}", file=sys.stderr)
        return None


def _decode_latents(vae, latents: mx.array) -> Optional[np.ndarray]:
    """
    Decode VAE latents → uint8 [H, W, 3].
    Returns None if decode fails.
    """
    try:
        img = vae.decode(latents)   # [1, 3, H, W] in [-1, 1]
        mx.eval(img)
        arr = np.array(img[0].astype(mx.float32))   # [3, H, W]
        arr = arr.transpose(1, 2, 0)                 # [H, W, 3]
        return np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"  eval: VAE decode failed: {e}", file=sys.stderr)
        return None


def _save_png(path: str, arr: np.ndarray):
    from PIL import Image
    Image.fromarray(arr).save(path)


def _to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# HTML report
# ─────────────────────────────────────────────────────────────────────────────

def _make_html(results: list[dict], step: int, mean_clip_i, mean_clip_t) -> str:
    ci_str = f"{mean_clip_i:.3f}" if mean_clip_i is not None else "—"
    ct_str = f"{mean_clip_t:.3f}" if mean_clip_t is not None else "—"
    rows = []
    for r in results:
        ref_src = r.get("ref_b64", "")
        gen_src = r.get("gen_b64", "")
        ref_img = f'<img src="{ref_src}" width="240">' if ref_src else "(missing)"
        gen_img = f'<img src="{gen_src}" width="240">' if gen_src else "(VAE unavailable)"
        ci = f'{r["clip_i"]:.3f}' if r.get("clip_i") is not None else "—"
        ct = f'{r["clip_t"]:.3f}' if r.get("clip_t") is not None else "—"
        rows.append(f"""
  <tr>
    <td style="max-width:280px">{r['prompt']}</td>
    <td>{ref_img}</td>
    <td>{gen_img}</td>
    <td style="text-align:center">{ci}</td>
    <td style="text-align:center">{ct}</td>
  </tr>""")
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8">
<title>IP-Adapter Eval — step {step:,}</title>
<style>
body{{font-family:monospace;background:#111;color:#eee;margin:20px}}
h2{{color:#ccc}}
.summary{{margin:10px 0;color:#aaa}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #333;padding:8px;vertical-align:top}}
th{{background:#1e1e1e;color:#ccc}}
img{{border-radius:3px;display:block}}
</style>
</head>
<body>
<h2>IP-Adapter Eval — step {step:,}</h2>
<div class="summary">CLIP-I (style): {ci_str} &nbsp;|&nbsp; CLIP-T (prompt): {ct_str} &nbsp;|&nbsp; {len(results)} prompts</div>
<table>
<tr><th>Prompt</th><th>Reference</th><th>Generated</th><th>CLIP-I</th><th>CLIP-T</th></tr>
{"".join(rows)}
</table>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation function (called by training hook and standalone CLI)
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(
    flux,
    adapter_cfg: dict,
    adapter_params: dict,
    prompts_file: str,
    output_dir: str,
    step: int,
    width: int = 512,
    height: int = 512,
    n_steps: int = 4,
    seed: int = 42,
    siglip_model_name: str = "google/siglip-so400m-patch14-384",
    vae=None,
) -> dict:
    """
    Run evaluation against a fixed prompt set.

    flux:            loaded Flux2Klein (frozen, shared with training)
    adapter_cfg:     'adapter' section from training config YAML
    adapter_params:  EMA weight dict (str → mx.array), as from mx.load()
    prompts_file:    path to eval_prompts.txt
    output_dir:      where to write images, JSON, HTML
    step:            training step (used in filenames and report)
    vae:             VAE decoder; defaults to flux.vae if None
    """
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)

    if vae is None:
        vae = getattr(flux, "vae", None)

    # ── Load prompts ─────────────────────────────────────────────────────────
    prompts: list[str] = []
    ref_paths: list[str] = []
    prompts_dir = str(Path(prompts_file).parent)
    with open(prompts_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|", 1)
            if len(parts) != 2:
                continue
            prompts.append(parts[0].strip())
            ref_paths.append(os.path.join(prompts_dir, parts[1].strip()))

    if not prompts:
        print("  eval: no prompts found in eval_prompts.txt — skipping", flush=True)
        return {}

    # ── Build adapter from EMA params ────────────────────────────────────────
    adapter = IPAdapterKlein(
        num_blocks=adapter_cfg["num_blocks"],
        hidden_dim=adapter_cfg["hidden_dim"],
        num_image_tokens=adapter_cfg["num_image_tokens"],
        siglip_dim=adapter_cfg["siglip_dim"],
        perceiver_heads=adapter_cfg["perceiver_heads"],
    )
    # Strip "ema." prefix if checkpoint was saved with prefix
    params = {(k[4:] if k.startswith("ema.") else k): v for k, v in adapter_params.items()}
    adapter.load_weights(list(params.items()))
    adapter.eval()
    adapter.freeze()
    mx.eval(adapter.parameters())

    # ── Load SigLIP (for IP conditioning and CLIP metrics) ───────────────────
    siglip_model, siglip_proc = _load_siglip_full(siglip_model_name)
    if siglip_model is None:
        print("  eval: SigLIP unavailable — CLIP metrics skipped, IP conditioning zeroed",
              flush=True)

    # ── Text encoder (reuse from flux) ────────────────────────────────────────
    text_enc = _TextEncoderBundle(flux.text_encoder, flux.tokenizers["qwen3"])

    results: list[dict] = []
    clip_i_scores: list[float] = []
    clip_t_scores: list[float] = []

    for idx, (prompt, ref_path) in enumerate(zip(prompts, ref_paths)):
        print(f"  eval [{idx+1}/{len(prompts)}]: {prompt[:60]}", flush=True)

        # ── Encode text ───────────────────────────────────────────────────────
        text_embeds = _encode_text(text_enc, [prompt])  # [1, seq, 7680]
        mx.eval(text_embeds)

        # ── Load reference image → SigLIP features ────────────────────────────
        ref_pil = _load_pil(ref_path) if os.path.exists(ref_path) else None
        siglip_np = None
        if ref_pil is not None and siglip_model is not None:
            siglip_np = _siglip_vision_feats(siglip_model, siglip_proc, ref_pil)

        if siglip_np is not None:
            siglip_feats = mx.array(siglip_np).astype(mx.bfloat16)  # [1, 729, 1152]
        else:
            siglip_feats = mx.zeros((1, 729, adapter_cfg["siglip_dim"]), dtype=mx.bfloat16)
        mx.eval(siglip_feats)

        # ── Generate latents ──────────────────────────────────────────────────
        latents = _generate(flux, adapter, text_embeds, siglip_feats,
                            width, height, n_steps, seed + idx)

        # ── Decode + save ─────────────────────────────────────────────────────
        gen_arr = _decode_latents(vae, latents) if vae is not None else None
        gen_path = os.path.join(output_dir, f"gen_{idx:02d}.png")
        gen_b64 = ""
        if gen_arr is not None:
            _save_png(gen_path, gen_arr)
            gen_b64 = _to_b64(gen_path)

        ref_b64 = ""
        ref_arr = None
        if ref_pil is not None:
            ref_out = os.path.join(output_dir, f"ref_{idx:02d}.png")
            ref_arr = np.array(ref_pil)
            _save_png(ref_out, ref_arr)
            ref_b64 = _to_b64(ref_out)

        # ── CLIP metrics ──────────────────────────────────────────────────────
        from PIL import Image as _PIL
        gen_pil = _PIL.fromarray(gen_arr) if gen_arr is not None else None

        clip_i = clip_t = None
        if siglip_model is not None:
            if ref_pil is not None and gen_pil is not None:
                clip_i = _compute_clip_i(siglip_model, siglip_proc, ref_pil, gen_pil)
                if clip_i is not None:
                    clip_i_scores.append(clip_i)
            if gen_pil is not None:
                clip_t = _compute_clip_t(siglip_model, siglip_proc, prompt, gen_pil)
                if clip_t is not None:
                    clip_t_scores.append(clip_t)

        results.append({
            "idx": idx,
            "prompt": prompt,
            "ref_path": ref_path,
            "gen_path": gen_path if gen_arr is not None else None,
            "clip_i": round(clip_i, 4) if clip_i is not None else None,
            "clip_t": round(clip_t, 4) if clip_t is not None else None,
            "ref_b64": ref_b64,
            "gen_b64": gen_b64,
        })

        del latents, text_embeds, siglip_feats
        mx.clear_cache()

    # ── Summary + output ──────────────────────────────────────────────────────
    mean_ci = float(np.mean(clip_i_scores)) if clip_i_scores else None
    mean_ct = float(np.mean(clip_t_scores)) if clip_t_scores else None
    elapsed = round(time.time() - t0, 1)

    summary = {
        "step": step,
        "num_prompts": len(prompts),
        "mean_clip_i": round(mean_ci, 4) if mean_ci is not None else None,
        "mean_clip_t": round(mean_ct, 4) if mean_ct is not None else None,
        "elapsed_s": elapsed,
        "results": [
            {k: v for k, v in r.items() if not k.endswith("_b64")}
            for r in results
        ],
    }

    json_path = os.path.join(output_dir, "eval_results.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    html_path = os.path.join(output_dir, "report.html")
    with open(html_path, "w") as fh:
        fh.write(_make_html(results, step, mean_ci, mean_ct))

    ci_str = f"{mean_ci:.3f}" if mean_ci is not None else "—"
    ct_str = f"{mean_ct:.3f}" if mean_ct is not None else "—"
    print(
        f"  eval done: step={step}  CLIP-I={ci_str}  CLIP-T={ct_str}"
        f"  ({len(prompts)} prompts, {elapsed}s)"
        f"  → {output_dir}",
        flush=True,
    )

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate IP-Adapter checkpoint: generate images + CLIP-I/T metrics"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="EMA checkpoint .safetensors (e.g. step_010000.safetensors)")
    parser.add_argument("--config", required=True,
                        help="Training config YAML (stage1_512px.yaml)")
    parser.add_argument("--prompts", default=None,
                        help="eval_prompts.txt path (default: <config-dir>/eval_prompts.txt)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: <ckpt-dir>/eval/step_NNNNNNN/)")
    parser.add_argument("--steps", type=int, default=4,
                        help="Denoising steps (default 4 for distilled Flux)")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import yaml
    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    mcfg = config["model"]
    acfg = config["adapter"]

    prompts_file = args.prompts or str(Path(args.config).parent / "eval_prompts.txt")

    # Infer step from checkpoint filename (e.g. "step_010000" or "010000")
    stem = Path(args.checkpoint).stem
    step = 0
    for tok in stem.replace("_", " ").replace("-", " ").split():
        try:
            step = int(tok)
        except ValueError:
            pass

    output_dir = args.output
    if output_dir is None:
        ckpt_dir = str(Path(args.checkpoint).parent)
        output_dir = os.path.join(ckpt_dir, "eval", f"step_{step:07d}")

    if not _HAS_MFLUX:
        print("Error: mflux not installed. Run: pip install mflux", file=sys.stderr)
        sys.exit(1)

    print(f"Loading Flux Klein from {mcfg['flux_model_dir']} ...")
    flux = Flux2Klein.from_pretrained(mcfg["flux_model_dir"])

    print(f"Loading adapter checkpoint {args.checkpoint} ...")
    adapter_params = dict(mx.load(args.checkpoint))

    run_eval(
        flux=flux,
        adapter_cfg=acfg,
        adapter_params=adapter_params,
        prompts_file=prompts_file,
        output_dir=output_dir,
        step=step,
        width=args.width,
        height=args.height,
        n_steps=args.steps,
        seed=args.seed,
        siglip_model_name=mcfg["siglip_model"],
        vae=flux.vae,
    )


if __name__ == "__main__":
    main()
