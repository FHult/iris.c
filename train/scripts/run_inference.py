#!/usr/bin/env python3
"""
train/scripts/run_inference.py — Python-side IP-Adapter inference (MLX-22/23).

Runs Flux Klein 4B with loaded IP-Adapter weights for validation image generation.
This is the Python-side --sref path before MLX-23 (iris.c --sref) is implemented.

Usage:
    python train/scripts/run_inference.py \
        --checkpoint /Volumes/2TBSSD/checkpoints/stage1/step_050000.safetensors \
        --prompts train/configs/eval_prompts.txt \
        --output /tmp/val_chunk1/ \
        --config train/configs/stage1_512px.yaml

Output:
    /tmp/val_chunk1/{idx:02d}_{prompt_slug}_with_adapter.png
    /tmp/val_chunk1/{idx:02d}_{prompt_slug}_no_adapter.png
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

try:
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
except ImportError:
    print("MLX not found. Run: source train/.venv/bin/activate", file=sys.stderr)
    sys.exit(1)

import yaml


# ---------------------------------------------------------------------------
# Sigma schedule (distilled Klein 4B: 4 steps)
# ---------------------------------------------------------------------------

DISTILLED_SIGMAS = [1.000, 0.750, 0.500, 0.250, 0.000]


def _euler_step(x: mx.array, sigma_cur: float, sigma_next: float,
                v_pred: mx.array) -> mx.array:
    """Single Euler step in flow matching: x_next = x + (sigma_next - sigma_cur) * v."""
    return x + (sigma_next - sigma_cur) * v_pred


# ---------------------------------------------------------------------------
# IP-Adapter weight loading
# ---------------------------------------------------------------------------

def load_adapter_from_checkpoint(checkpoint_path: str, adapter):
    """Load adapter weights from a step_*.safetensors checkpoint (skip ema.* keys)."""
    from safetensors import safe_open
    params: dict = {}
    with safe_open(checkpoint_path, framework="numpy") as f:
        for k in f.keys():
            if not k.startswith("ema."):
                params[k] = mx.array(f.get_tensor(k))
    # Rebuild nested dict and update model
    nested: dict = {}
    for key, val in params.items():
        parts = key.split(".")
        d = nested
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = val
    adapter.update(nested)
    print(f"Loaded adapter from {checkpoint_path}")


# ---------------------------------------------------------------------------
# Reference image → SigLIP features
# ---------------------------------------------------------------------------

def load_reference_image(path: str, size: int = 384) -> "PIL.Image.Image":
    from PIL import Image
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return img


def encode_siglip(image, siglip_model_name: str) -> mx.array:
    """Encode a PIL image to SigLIP features [1, 729, 1152]."""
    try:
        from transformers import AutoProcessor, AutoModel
        import torch

        processor = AutoProcessor.from_pretrained(siglip_model_name)
        model = AutoModel.from_pretrained(siglip_model_name).vision_model.eval()

        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            feats = model(**inputs).last_hidden_state  # [1, 729, 1152]
        return mx.array(feats.numpy()).astype(mx.bfloat16)
    except ImportError as e:
        raise RuntimeError(f"transformers/torch required for SigLIP: {e}") from e


# ---------------------------------------------------------------------------
# Full sampling loop with IP-Adapter
# ---------------------------------------------------------------------------

def generate_with_adapter(
    flux,
    adapter,
    text_embeds: mx.array,
    siglip_feats: mx.array,
    width: int = 512,
    height: int = 512,
    seed: int = 42,
    use_adapter: bool = True,
) -> mx.array:
    """
    Full 4-step Euler denoising with optional IP-Adapter injection.

    Returns: [1, 32, H/8, W/8] clean latent.
    """
    mx.random.seed(seed)
    Lh, Lw = height // 8, width // 8
    latent = mx.random.normal((1, 32, Lh, Lw), dtype=mx.bfloat16)

    if use_adapter:
        ip_embeds = adapter.get_image_embeds(siglip_feats)
        k_ip_all, v_ip_all = adapter.get_kv_all(ip_embeds)
        ip_scale = adapter.scale
    else:
        k_ip_all = v_ip_all = ip_scale = None

    for i in range(len(DISTILLED_SIGMAS) - 1):
        sigma_cur  = DISTILLED_SIGMAS[i]
        sigma_next = DISTILLED_SIGMAS[i + 1]
        t_int = mx.array([int(sigma_cur * 1000)], dtype=mx.int32)

        if use_adapter and k_ip_all is not None:
            from train_ip_adapter import _flux_forward_with_ip
            v_pred = _flux_forward_with_ip(
                flux, latent, text_embeds, t_int,
                k_ip_all, v_ip_all, ip_scale,
            )
        else:
            from train_ip_adapter import _flux_forward_no_ip
            flux_state = _flux_forward_no_ip(flux, latent, text_embeds, t_int)
            # Reconstruct prediction from flux_state for no-adapter path
            tr = flux.transformer
            h_final = flux_state["h_final"]
            temb    = flux_state["temb"]
            B  = flux_state["B"]
            C  = flux_state["C"]
            pH = flux_state["pH"]
            pW = flux_state["pW"]
            h_normed = tr.norm_out(h_final, temb)
            pred_seq = tr.proj_out(h_normed)
            pred = pred_seq.transpose(0, 2, 1).reshape(B, C * 4, pH, pW)
            pred = pred.reshape(B, C, 2, 2, pH, pW)
            pred = pred.transpose(0, 1, 4, 2, 5, 3)
            v_pred = pred.reshape(B, C, flux_state["Lh"], flux_state["Lw"])

        mx.eval(v_pred)
        latent = _euler_step(latent, sigma_cur, sigma_next, v_pred)
        mx.eval(latent)

    return latent


# ---------------------------------------------------------------------------
# Decode latent to image
# ---------------------------------------------------------------------------

def decode_latent(flux, latent: mx.array) -> "PIL.Image.Image":
    """VAE decode [1, 32, H/8, W/8] → PIL RGB image."""
    import numpy as np
    from PIL import Image

    decoded = flux.vae.decode(latent)  # [1, 3, H, W] in [-1, 1]
    mx.eval(decoded)
    img_np = np.array(decoded[0].astype(mx.float32))  # [3, H, W]
    img_np = ((img_np.transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="IP-Adapter inference for validation")
    ap.add_argument("--checkpoint", required=True, help="Adapter checkpoint .safetensors")
    ap.add_argument("--prompts", default="train/configs/eval_prompts.txt")
    ap.add_argument("--output",  required=True, help="Output directory for images")
    ap.add_argument("--config",  default="train/configs/stage1_512px.yaml")
    ap.add_argument("--width",   type=int, default=512)
    ap.add_argument("--height",  type=int, default=512)
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output, exist_ok=True)

    print("Loading Flux Klein 4B (frozen)...")
    try:
        from mflux.models.flux2 import Flux2Klein
    except ImportError:
        print("mflux not installed — pip install mflux", file=sys.stderr)
        sys.exit(1)

    flux = Flux2Klein(model_path=cfg["model"]["flux_model_dir"], quantize=None)
    flux.freeze()
    mx.eval(flux.transformer.parameters())

    print("Loading IP-Adapter...")
    from ip_adapter.model import IPAdapterKlein
    acfg = cfg["adapter"]
    adapter = IPAdapterKlein(
        num_blocks=acfg["num_blocks"],
        hidden_dim=acfg["hidden_dim"],
        num_image_tokens=acfg["num_image_tokens"],
        siglip_dim=acfg["siglip_dim"],
        perceiver_heads=acfg["perceiver_heads"],
    )
    load_adapter_from_checkpoint(args.checkpoint, adapter)
    mx.eval(adapter.parameters())

    siglip_model_name = cfg["model"].get("siglip_model", "google/siglip-so400m-patch14-384")

    # Parse eval prompts
    prompt_pairs: list[tuple[str, str]] = []
    with open(args.prompts) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) >= 2:
                prompt_pairs.append((parts[0].strip(), parts[1].strip()))

    print(f"Generating {len(prompt_pairs)} pairs (with + without adapter)...")

    for idx, (prompt, ref_path) in enumerate(prompt_pairs):
        slug = prompt[:40].replace(" ", "_").replace("/", "-")
        print(f"  [{idx+1}/{len(prompt_pairs)}] {prompt[:60]}")

        # Text embedding
        try:
            from ip_adapter.utils import encode_text_qwen3
            text_embeds = encode_text_qwen3(flux, prompt)
        except Exception:
            # Fallback: zero text embeds for smoke test
            text_embeds = mx.zeros((1, 64, cfg["model"].get("text_dim", 7680)),
                                   dtype=mx.bfloat16)

        # SigLIP features
        ref_full = ref_path if os.path.isabs(ref_path) else os.path.join("train", ref_path)
        if os.path.exists(ref_full):
            ref_img = load_reference_image(ref_full)
            siglip_feats = encode_siglip(ref_img, siglip_model_name)
        else:
            print(f"    WARNING: ref not found: {ref_full} — using zero features")
            siglip_feats = mx.zeros((1, 729, acfg["siglip_dim"]), dtype=mx.bfloat16)

        # Generate with adapter
        t0 = time.time()
        latent_with = generate_with_adapter(
            flux, adapter, text_embeds, siglip_feats,
            width=args.width, height=args.height, seed=args.seed, use_adapter=True,
        )
        img_with = decode_latent(flux, latent_with)
        out_with = os.path.join(args.output, f"{idx:02d}_{slug}_with_adapter.png")
        img_with.save(out_with)

        # Generate without adapter (baseline)
        latent_no = generate_with_adapter(
            flux, adapter, text_embeds, siglip_feats,
            width=args.width, height=args.height, seed=args.seed, use_adapter=False,
        )
        img_no = decode_latent(flux, latent_no)
        out_no = os.path.join(args.output, f"{idx:02d}_{slug}_no_adapter.png")
        img_no.save(out_no)

        print(f"    done ({time.time()-t0:.1f}s): {out_with}")

    print(f"Inference complete. Images in {args.output}")


if __name__ == "__main__":
    main()
