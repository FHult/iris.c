#!/usr/bin/env python3
"""
train/train_ip_adapter.py — Main IP-Adapter training loop for Flux Klein 4B.

Usage:
    source train/.venv/bin/activate
    python train/train_ip_adapter.py --config train/configs/stage1_512px.yaml

Stage 2 (768px fine-tune):
    python train/train_ip_adapter.py --config train/configs/stage2_768px.yaml

Run under caffeinate for unattended multi-day training:
    caffeinate -i -d python train/train_ip_adapter.py --config ... \
      2>&1 | tee logs/train_stage1.log &

Architecture: plans/ip-adapter-training.md §3
Performance: plans/ip-adapter-training.md Metal Optimisations section
"""

import argparse
import glob
import math
import os
import random
import sys
import threading
import time
from pathlib import Path

import yaml

# ── MLX imports ──────────────────────────────────────────────────────────────
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
except ImportError:
    print("Error: MLX not found. Run: source train/.venv/bin/activate", file=sys.stderr)
    sys.exit(1)

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from ip_adapter.model import IPAdapterKlein
from ip_adapter.loss import fused_flow_noise, get_schedule_values
from ip_adapter.ema import update_ema, save_ema, _flatten
from ip_adapter.dataset import make_prefetch_loader, augment_mlx, BUCKETS

# ── mflux: Flux Klein 4B MLX inference ───────────────────────────────────────
try:
    from mflux import Flux1
    _HAS_MFLUX = True
except ImportError:
    _HAS_MFLUX = False
    print("Warning: mflux not installed. Run: pip install mflux", file=sys.stderr)

try:
    from mlx_lm.tuner.trainer import grad_checkpoint
    _HAS_GRAD_CHECKPOINT = True
except ImportError:
    _HAS_GRAD_CHECKPOINT = False


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule (plans/ip-adapter-training.md §3.11)
# ─────────────────────────────────────────────────────────────────────────────

def make_lr_schedule(lr_max: float, warmup_steps: int, total_steps: int):
    """
    Linear warmup then cosine decay.
    Uses optim.join_schedules matching plans/ip-adapter-training.md §3.11.
    """
    decay_steps = total_steps - warmup_steps
    return optim.join_schedules(
        [
            optim.linear_schedule(1e-6, lr_max, steps=warmup_steps),
            optim.cosine_decay(lr_max, decay_steps=decay_steps, eta_min=1e-6),
        ],
        [warmup_steps],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Async checkpoint save (plans/ip-adapter-training.md §3.12)
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint_async(
    adapter: IPAdapterKlein,
    ema_params: dict,
    step: int,
    output_dir: str,
    keep_last_n: int = 5,
) -> None:
    """
    Save adapter weights + EMA on a background thread so GPU continues training.
    mx.eval() completes pending async_eval ops before the numpy copy.
    Based on plans/ip-adapter-training.md §3.12.
    """
    from safetensors.numpy import save_file

    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"step_{step:07d}.safetensors")

    # mx.eval ensures all pending async_eval ops flush before numpy copy
    mx.eval(adapter.parameters())
    mx.eval(ema_params)

    flat_adapter = {k: np.array(v) for k, v in _flatten(adapter.parameters())}
    flat_ema = {f"ema.{k}": np.array(v) for k, v in _flatten(ema_params)}
    payload = {**flat_adapter, **flat_ema}

    def _write():
        save_file(payload, ckpt_path)
        size_mb = os.path.getsize(ckpt_path) / 1e6
        print(f"  checkpoint saved: step_{step:07d}.safetensors ({size_mb:.0f} MB)")
        _purge_old_checkpoints(output_dir, keep_last_n)

    threading.Thread(target=_write, daemon=True).start()


def _purge_old_checkpoints(directory: str, keep_last_n: int) -> None:
    files = sorted(f for f in glob.glob(os.path.join(directory, "step_*.safetensors"))
                   if "ema." not in f)
    for f in files[:-keep_last_n]:
        os.remove(f)


def load_checkpoint(adapter: IPAdapterKlein, path: str) -> None:
    from safetensors import safe_open
    params = {}
    with safe_open(path, framework="numpy") as f:
        for k in f.keys():
            if not k.startswith("ema."):
                params[k] = mx.array(f.get_tensor(k))
    _nested_update(adapter, params)
    print(f"Loaded checkpoint: {path}")


def _nested_update(model: nn.Module, flat: dict) -> None:
    nested = {}
    for key, val in flat.items():
        parts = key.split(".")
        d = nested
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = val
    model.update(nested)


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(config: dict) -> None:
    mcfg = config["model"]
    acfg = config["adapter"]
    dcfg = config["data"]
    tcfg = config["training"]
    ocfg = config["output"]
    lcfg = config.get("logging", {})

    # ── Optional W&B logging ──────────────────────────────────────────────────
    wandb_run = None
    if lcfg.get("wandb_project"):
        try:
            import wandb
            wandb_run = wandb.init(
                project=lcfg["wandb_project"],
                name=lcfg.get("wandb_run_name"),
                config=config,
            )
        except ImportError:
            print("wandb not installed — logging to stdout only")

    # ── Load frozen base models ───────────────────────────────────────────────
    if not _HAS_MFLUX:
        raise RuntimeError("pip install mflux")

    print("Loading Flux Klein 4B (frozen) ...")
    flux = Flux1.from_pretrained(path=mcfg["flux_model_dir"], quantize=None)
    flux.freeze()

    print("Loading SigLIP SO400M (frozen) ...")
    siglip = _load_siglip(mcfg["siglip_model"])
    siglip.freeze()

    print("Loading VAE (frozen) ...")
    vae = _load_vae(flux)
    vae.freeze()

    print("Loading Qwen3 text encoder (Q4, frozen) ...")
    text_encoder = _load_text_encoder(flux)
    text_encoder.freeze()

    # ── Apply gradient checkpointing to Flux transformer blocks ──────────────
    # Saves ~600MB at 512px; enables batch_size=2.
    # Uses grad_checkpoint from mlx_lm (plans/ip-adapter-training.md §3.2).
    if _HAS_GRAD_CHECKPOINT:
        try:
            from mflux.models.transformer.transformer_block import Flux2TransformerBlock
            from mflux.models.transformer.single_transformer_block import Flux2SingleTransformerBlock
            grad_checkpoint(Flux2TransformerBlock)
            grad_checkpoint(Flux2SingleTransformerBlock)
            print("Gradient checkpointing applied to Flux transformer blocks")
        except ImportError as e:
            print(f"Warning: gradient checkpointing unavailable: {e}")
    else:
        print("Warning: mlx_lm not installed — gradient checkpointing disabled")

    # ── Build IP-Adapter (trainable) ──────────────────────────────────────────
    print("Building IPAdapterKlein ...")
    warmstart = mcfg.get("warmstart_path")
    if warmstart and os.path.isdir(warmstart):
        # Warmstart Perceiver Resampler from InstantX Flux.1-dev weights
        adapter = IPAdapterKlein.from_pretrained_warmstart(
            instantx_path=warmstart,
            num_blocks=acfg["num_blocks"],
            hidden_dim=acfg["hidden_dim"],
            num_image_tokens=acfg["num_image_tokens"],
            siglip_dim=acfg["siglip_dim"],
            perceiver_heads=acfg["perceiver_heads"],
        )
    else:
        adapter = IPAdapterKlein(
            num_blocks=acfg["num_blocks"],
            hidden_dim=acfg["hidden_dim"],
            num_image_tokens=acfg["num_image_tokens"],
            siglip_dim=acfg["siglip_dim"],
            perceiver_heads=acfg["perceiver_heads"],
        )
        if warmstart and os.path.isfile(warmstart):
            # Resume from a previous checkpoint (stage 2)
            load_checkpoint(adapter, warmstart)

    # ── Optimizer with built-in cosine+warmup schedule ────────────────────────
    # MLX schedule object passed directly to AdamW — it advances each step.
    # (plans/ip-adapter-training.md §3.11)
    lr_schedule = make_lr_schedule(
        tcfg["learning_rate"],
        tcfg["warmup_steps"],
        tcfg["num_steps"],
    )
    optimizer = optim.AdamW(
        learning_rate=lr_schedule,
        betas=(0.9, 0.999),
        weight_decay=tcfg["weight_decay"],
    )

    # ── EMA initialised to model's current parameters ─────────────────────────
    ema_params = adapter.parameters()

    # ── Compiled loss + backward ──────────────────────────────────────────────
    # state list must include mx.random.state so noise sampling isn't baked
    # as a constant during graph compilation.
    # (plans/ip-adapter-training.md §3.7)
    state = [adapter.state, optimizer.state, mx.random.state]

    def loss_fn(adapter_params, images, text_embeds, latents, siglip_feats,
                use_null_image: mx.array, use_null_text: mx.array):
        adapter.update(adapter_params)

        # Null conditioning dropout — decided outside compiled region,
        # passed as mx.array so mx.where works inside compiled graph.
        ip_embeds = adapter.get_image_embeds(siglip_feats)
        zero_embeds = mx.zeros_like(ip_embeds)
        ip_embeds = mx.where(use_null_image, zero_embeds, ip_embeds)

        # Batched K/V: 2 Metal dispatches for all 25 blocks
        k_ip_all, v_ip_all = adapter.get_kv_all(ip_embeds)

        # Sample timestep in [0, 1000]; get alpha/sigma for v-prediction
        t_int = mx.random.randint(0, 1000, shape=(1,))
        alpha_t, sigma_t = get_schedule_values(t_int)

        noise = mx.random.normal(latents.shape, dtype=latents.dtype)
        noisy, target = fused_flow_noise(latents, noise, alpha_t, sigma_t)

        # Forward through frozen Klein with IP injection (stub — see below)
        pred = _flux_forward_with_ip(
            flux, noisy, text_embeds, t_int,
            k_ip_all=k_ip_all,
            v_ip_all=v_ip_all,
            ip_scale=adapter.scale,
        )

        return mx.mean((pred - target) ** 2)

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)
    compiled_loss_and_grad = mx.compile(loss_and_grad, inputs=state, outputs=state)

    # ── Data loader ───────────────────────────────────────────────────────────
    shard_paths = sorted(glob.glob(os.path.join(dcfg["shard_path"], "*.tar")))
    if not shard_paths:
        raise RuntimeError(f"No .tar shards found in {dcfg['shard_path']}")

    loader = make_prefetch_loader(
        shard_paths=shard_paths,
        batch_size=dcfg["batch_size"],
        image_dropout_prob=tcfg["image_dropout_prob"],
        text_dropout_prob=tcfg["text_dropout_prob"],
        sample_buffer=dcfg.get("prefetch_batches", 6),
        qwen3_cache_dir=dcfg.get("qwen3_cache_dir"),
        vae_cache_dir=dcfg.get("vae_cache_dir"),
        siglip_cache_dir=dcfg.get("siglip_cache_dir"),
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining: {tcfg['num_steps']:,} steps, batch_size={dcfg['batch_size']}\n")

    step = 0
    t0 = time.time()
    log_interval = ocfg["log_every"]

    for images_np, captions, style_refs_np, text_np, vae_np, siglip_np, bucket_hw in loader:
        if step >= tcfg["num_steps"]:
            break

        bH, bW = bucket_hw

        # Convert to MLX — images are padded (H+32, W+32) for GPU crop augmentation
        images = mx.array(images_np, dtype=mx.bfloat16)   # [B, C, H+32, W+32]

        # GPU augmentation: random horizontal flip + random crop (§3.10)
        # Done in MLX on GPU — no CPU→GPU copy penalty
        images = augment_mlx(images, bH, bW)               # [B, C, bH, bW]

        # Null conditioning flags — decided OUTSIDE compiled region (§3.7)
        use_null_image = mx.array(random.random() < tcfg["image_dropout_prob"])
        use_null_text = mx.array(random.random() < tcfg["text_dropout_prob"])

        # Encode frozen models — use pre-computed cache if available (§2.7)
        if vae_np is not None:
            latents = mx.array(vae_np, dtype=mx.bfloat16)
        else:
            latents = _vae_encode(vae, images)

        if text_np is not None:
            text_embeds = mx.array(text_np, dtype=mx.bfloat16)
        else:
            captions_in = [""] * len(captions) if bool(use_null_text.item()) else captions
            text_embeds = _encode_text(text_encoder, captions_in)

        if siglip_np is not None:
            siglip_feats = mx.array(siglip_np, dtype=mx.bfloat16)
        else:
            siglip_feats = siglip(images)

        # Forward + backward (compiled)
        loss_val, grads = compiled_loss_and_grad(
            adapter.trainable_parameters(),
            images, text_embeds, latents, siglip_feats,
            use_null_image, use_null_text,
        )

        # Gradient clip then optimizer step (plans §3.6)
        grads, _ = mx.clip_by_global_norm(grads, max_norm=tcfg["grad_clip"])
        optimizer.update(adapter, grads)

        # async_eval: overlaps ~5–10ms Python overhead with Metal execution
        mx.async_eval(adapter.parameters(), optimizer.state, loss_val)

        step += 1

        # EMA update (every 10 steps saves ~23 minutes; plans §3.11)
        if step % tcfg["ema_update_every"] == 0:
            ema_params = update_ema(ema_params, adapter, decay=tcfg["ema_decay"])

        # Logging
        if step % log_interval == 0:
            mx.eval(loss_val)  # materialise for logging
            elapsed = time.time() - t0
            steps_per_sec = log_interval / elapsed
            loss_scalar = float(loss_val.item())
            lr_now = float(optimizer.learning_rate)
            print(f"step {step:>7,}  loss {loss_scalar:.4f}  "
                  f"lr {lr_now:.2e}  {steps_per_sec:.2f} steps/s")
            if wandb_run:
                wandb_run.log(
                    {"loss": loss_scalar, "lr": lr_now,
                     "steps_per_sec": steps_per_sec},
                    step=step,
                )
            t0 = time.time()

        # Checkpoint (async background write; plans §3.12)
        if step % ocfg["checkpoint_every"] == 0:
            save_checkpoint_async(adapter, ema_params, step,
                                  ocfg["checkpoint_dir"], ocfg["keep_last_n"])

    # Final checkpoint + EMA export
    save_checkpoint_async(adapter, ema_params, step,
                          ocfg["checkpoint_dir"], keep_last_n=999)

    # Save best EMA as final export
    mx.eval(ema_params)
    save_ema(ema_params, os.path.join(ocfg["checkpoint_dir"], "best.safetensors"))
    print(f"\nTraining complete. EMA weights: {ocfg['checkpoint_dir']}/best.safetensors")


# ─────────────────────────────────────────────────────────────────────────────
# Integration stubs — filled in once mflux API is confirmed
# ─────────────────────────────────────────────────────────────────────────────

def _load_siglip(model_name: str):
    """
    Load SigLIP SO400M vision encoder (frozen).
    Implementation: load via mflux or transformers.
    Input: [B, 3, 384, 384] normalised to [-1, 1]
    Output: [B, 729, 1152]
    """
    raise NotImplementedError(
        "Implement _load_siglip:\n"
        "  Option A: from mlx_vlm or transformers (CLIPVisionModel equivalent)\n"
        "  Option B: iris_siglip.py — pure MLX SigLIP from safetensors weights"
    )


def _load_vae(flux):
    """Extract VAE encoder from the loaded Flux object."""
    return flux.vae  # mflux exposes .vae


def _load_text_encoder(flux):
    """Extract Qwen3 text encoder from the loaded Flux object."""
    return flux.text_encoder  # mflux exposes .text_encoder or similar


def _vae_encode(vae, images: mx.array) -> mx.array:
    """
    Encode images to VAE latents.
    images: [B, 3, H, W] BF16 in [-1, 1]
    Returns: [B, 32, H/8, W/8] latents (Flux VAE: 32 latent channels)
    """
    raise NotImplementedError("Wire to mflux VAE encoder")


def _encode_text(text_encoder, captions: list) -> mx.array:
    """
    Encode text captions using Qwen3 (Q4 quantized, frozen).
    Returns: [B, seq, text_dim] text embeddings
    """
    raise NotImplementedError("Wire to mflux Qwen3 text encoder")


def _flux_forward_with_ip(flux, noisy_latents, text_embeds, t_int,
                           k_ip_all, v_ip_all, ip_scale) -> mx.array:
    """
    Run Flux Klein 4B forward pass with IP-Adapter injection.

    Requires subclassing mflux Flux2TransformerBlock to accept and inject
    k_ip / v_ip at each block (plans/ip-adapter-training.md §3.5):

        if k_ip is not None:
            ip_attn = mx.fast.scaled_dot_product_attention(
                img_q, k_ip, v_ip, scale=(head_dim ** -0.5)
            )
            hidden_states = hidden_states + ip_scale * ip_attn.reshape(...)

    See plans/ip-adapter-training.md §3.5 for the full patched block signature.
    """
    raise NotImplementedError(
        "Subclass Flux2Transformer to inject IP K/V — see plans §3.5"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train IP-Adapter for Flux Klein 4B")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--resume", default=None,
                        help="Resume from checkpoint .safetensors path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config without training")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.resume:
        config["model"]["warmstart_path"] = args.resume

    if args.dry_run:
        print("Config OK:")
        print(yaml.dump(config, default_flow_style=False))
        return

    train(config)


if __name__ == "__main__":
    main()
