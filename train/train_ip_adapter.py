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
from functools import partial
from pathlib import Path
from typing import Optional

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
    from mflux.models.flux2 import Flux2Klein
    _HAS_MFLUX = True
except ImportError:
    _HAS_MFLUX = False
    print("Warning: mflux not installed. Run: pip install mflux", file=sys.stderr)



# ─────────────────────────────────────────────────────────────────────────────
# LR schedule (plans/ip-adapter-training.md §3.11)
# ─────────────────────────────────────────────────────────────────────────────

def make_lr_schedule(lr_max: float, warmup_steps: int, total_steps: int,
                      start_step: int = 0):
    """
    Linear warmup then cosine decay.
    Uses optim.join_schedules matching plans/ip-adapter-training.md §3.11.

    start_step: if resuming, pass the loaded step so the schedule continues
    from the correct position rather than restarting warmup from scratch.

    eta_min is set proportionally to lr_max (1% of peak) so that chunks with
    lower lr (e.g. 1e-5) still have a meaningful decay range instead of
    collapsing to the hardcoded 1e-6 floor.
    """
    eta_min = max(1e-8, lr_max * 0.01)
    decay_steps = total_steps - warmup_steps

    if start_step == 0:
        return optim.join_schedules(
            [
                optim.linear_schedule(1e-8, lr_max, steps=warmup_steps),
                optim.cosine_decay(lr_max, decay_steps=decay_steps, end=eta_min),
            ],
            [warmup_steps],
        )

    # Resume: fast-forward to the correct LR position
    if start_step < warmup_steps:
        current_lr = lr_max * start_step / max(1, warmup_steps)
        remaining_warmup = warmup_steps - start_step
        return optim.join_schedules(
            [
                optim.linear_schedule(current_lr, lr_max, steps=remaining_warmup),
                optim.cosine_decay(lr_max, decay_steps=decay_steps, end=eta_min),
            ],
            [remaining_warmup],
        )
    else:
        # Already in cosine decay phase
        decay_done = start_step - warmup_steps
        decay_total = max(1, total_steps - warmup_steps)
        progress = min(1.0, decay_done / decay_total)
        current_lr = eta_min + 0.5 * (lr_max - eta_min) * (1.0 + math.cos(math.pi * progress))
        remaining_decay = max(1, decay_total - decay_done)
        return optim.cosine_decay(current_lr, decay_steps=remaining_decay, end=eta_min)


# ─────────────────────────────────────────────────────────────────────────────
# Async checkpoint save (plans/ip-adapter-training.md §3.12)
# ─────────────────────────────────────────────────────────────────────────────

# Lock serialises _purge_old_checkpoints across concurrent async writer threads.
_ckpt_lock = threading.Lock()


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
        # Serialise purge so two concurrent threads don't race on the file list
        with _ckpt_lock:
            _purge_old_checkpoints(output_dir, keep_last_n)

    threading.Thread(target=_write, daemon=True).start()


def _purge_old_checkpoints(directory: str, keep_last_n: int) -> None:
    files = sorted(f for f in glob.glob(os.path.join(directory, "step_*.safetensors"))
                   if "ema." not in f)
    for f in files[:-keep_last_n]:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass  # already removed by a concurrent thread


def load_checkpoint(adapter: IPAdapterKlein, path: str) -> None:
    """Load adapter weights from a step_*.safetensors checkpoint (skips ema.* keys)."""
    from safetensors import safe_open
    params = {}
    with safe_open(path, framework="numpy") as f:
        for k in f.keys():
            if not k.startswith("ema."):
                params[k] = mx.array(f.get_tensor(k))
    _nested_update(adapter, params)
    print(f"Loaded checkpoint: {path}")


def load_ema_from_checkpoint(path: str) -> Optional[dict]:
    """
    Load EMA parameters from a step_*.safetensors checkpoint.
    Returns a nested dict matching adapter.parameters() structure, or None if
    the file contains no ema.* keys (e.g. a best.safetensors EMA export).
    """
    from safetensors import safe_open
    flat: dict = {}
    with safe_open(path, framework="numpy") as f:
        for k in f.keys():
            if k.startswith("ema."):
                flat[k[4:]] = mx.array(f.get_tensor(k))  # strip "ema." prefix
    if not flat:
        return None
    return _flat_to_nested(flat)


def step_from_checkpoint_path(path: str) -> int:
    """Parse the step number from a step_NNNNNNN.safetensors filename, or 0."""
    name = os.path.basename(path)
    if name.startswith("step_") and name.endswith(".safetensors"):
        try:
            return int(name[5:-len(".safetensors")])
        except ValueError:
            pass
    return 0


def _nested_update(model: nn.Module, flat: dict) -> None:
    nested = {}
    for key, val in flat.items():
        parts = key.split(".")
        d = nested
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = val
    model.update(nested)


def _flat_to_nested(flat: dict) -> dict:
    """Convert a flat dot-separated key dict to a nested dict."""
    nested: dict = {}
    for key, val in flat.items():
        parts = key.split(".")
        d = nested
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = val
    return nested


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(config: dict) -> None:
    mcfg = config["model"]
    acfg = config["adapter"]
    dcfg = config["data"]
    tcfg = config["training"]
    ocfg = config["output"]
    lcfg = config.get("logging") or {}

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
    flux = Flux2Klein(model_path=mcfg["flux_model_dir"], quantize=None)
    flux.freeze()

    use_siglip_live = dcfg.get("siglip_cache_dir") is None
    siglip = None
    if use_siglip_live:
        print("Loading SigLIP SO400M (frozen, live inference — consider precomputing) ...")
        siglip = _load_siglip(mcfg["siglip_model"])
        siglip.freeze()
    else:
        print("SigLIP: using precomputed cache — skipping model load.")

    print("Loading VAE (frozen) ...")
    vae = _load_vae(flux)
    vae.freeze()

    print("Loading Qwen3 text encoder (Q4, frozen) ...")
    text_encoder = _load_text_encoder(flux)
    text_encoder.freeze()

    # ── Force-evaluate all frozen model weights ───────────────────────────────
    # Flux weights are mmap'd (lazy, loaded from disk via ParallelFileReader).
    # mx.eval here pre-fetches all weights into Metal memory so the first
    # training step doesn't stall on disk I/O.
    print("Pre-evaluating frozen weights (flushing mmap → GPU) ...")
    mx.eval(flux.transformer.parameters())

    # ── Apply gradient checkpointing to Flux transformer blocks ──────────────
    # mflux 0.17.4 has immutable types — mlx_lm's grad_checkpoint() which patches
    # type.__call__ won't work. Use mlx.nn.checkpoint() instead, which wraps
    # individual block instances and returns checkpointed callables.
    # These are passed into _flux_forward_with_ip and called in place of blocks.
    ckpt_double = None
    ckpt_single = None
    try:
        from mlx.nn import checkpoint as nn_checkpoint
        ckpt_double = [nn_checkpoint(b) for b in flux.transformer.transformer_blocks]
        ckpt_single = [nn_checkpoint(b) for b in flux.transformer.single_transformer_blocks]
        print("Gradient checkpointing applied to Flux transformer blocks")
    except Exception as e:
        print(f"Warning: gradient checkpointing unavailable: {e}")

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

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_step = 0
    loaded_ema: Optional[dict] = None
    if warmstart and os.path.isfile(warmstart):
        load_checkpoint(adapter, warmstart)
        start_step = step_from_checkpoint_path(warmstart)
        loaded_ema = load_ema_from_checkpoint(warmstart)
        if start_step > 0:
            print(f"  Resuming from step {start_step:,}")
        if loaded_ema is not None:
            print(f"  Loaded EMA from checkpoint")

    # ── Optimizer with built-in cosine+warmup schedule ────────────────────────
    # MLX schedule object passed directly to AdamW — it advances each step.
    # start_step fast-forwards the schedule so LR continues from where it left
    # off rather than restarting warmup.  (plans/ip-adapter-training.md §3.11)
    lr_schedule = make_lr_schedule(
        tcfg["learning_rate"],
        tcfg["warmup_steps"],
        tcfg["num_steps"],
        start_step=start_step,
    )
    optimizer = optim.AdamW(
        learning_rate=lr_schedule,
        betas=(0.9, 0.999),
        weight_decay=tcfg["weight_decay"],
    )

    # ── EMA: use loaded EMA weights if resuming, else start from adapter ──────
    if loaded_ema is not None:
        ema_params = loaded_ema
    else:
        ema_params = adapter.parameters()

    # ── Pre-evaluate adapter parameters ───────────────────────────────────────
    # Flush all lazy init arrays (mx.random.normal * scale) before training.
    mx.eval(adapter.parameters())
    mx.eval(ema_params)

    def loss_fn(images, text_embeds, latents, siglip_feats,
                use_null_image: mx.array, use_null_text: mx.array):
        # Null conditioning dropout — decided outside compiled region,
        # passed as mx.array so mx.where works inside compiled graph.
        ip_embeds = adapter.get_image_embeds(siglip_feats)
        zero_embeds = mx.zeros_like(ip_embeds)
        ip_embeds = mx.where(use_null_image, zero_embeds, ip_embeds)

        # Batched K/V: 2 Metal dispatches for all 25 blocks
        k_ip_all, v_ip_all = adapter.get_kv_all(ip_embeds)

        # Sample per-sample timestep in [0, 1000]; get alpha/sigma for v-prediction
        B = latents.shape[0]
        t_int = mx.random.randint(0, 1000, shape=(B,))
        alpha_t, sigma_t = get_schedule_values(t_int)

        noise = mx.random.normal(latents.shape, dtype=latents.dtype)
        noisy, target = fused_flow_noise(latents, noise, alpha_t, sigma_t)

        # Forward through frozen Klein with IP injection (stub — see below)
        pred = _flux_forward_with_ip(
            flux, noisy, text_embeds, t_int,
            k_ip_all=k_ip_all,
            v_ip_all=v_ip_all,
            ip_scale=adapter.scale,
            ckpt_double=ckpt_double,
            ckpt_single=ckpt_single,
        )

        return mx.mean((pred - target) ** 2)

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)

    # Disable MLX auto-compilation for the entire training loop.
    # MLX auto-fuses element-wise ops into Compiled kernels; mx.checkpoint also
    # creates Compiled nodes for the recomputed backward pass. Both paths crash
    # with SIGSEGV (null Metal buffer) when mmap'd Flux weights haven't been
    # promoted to GPU yet. Disabling compile prevents kernel fusion SIGSEGV while
    # still allowing mx.checkpoint to recompute activations (memory savings preserved).
    mx.disable_compile()

    def compiled_step(images, text_embeds, latents, siglip_feats,
                      use_null_image, use_null_text):
        loss_val, grads = loss_and_grad(
            images, text_embeds, latents, siglip_feats,
            use_null_image, use_null_text,
        )
        grads, _ = optim.clip_grad_norm(grads, max_norm=tcfg["grad_clip"])
        optimizer.update(adapter, grads)
        return loss_val

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
        anchor_shard_dir=dcfg.get("anchor_shard_dir"),
        anchor_mix_ratio=dcfg.get("anchor_mix_ratio", 0.20),
        hard_example_dir=dcfg.get("hard_example_dir"),
        hard_mix_ratio=dcfg.get("hard_mix_ratio", 0.05),
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining: {tcfg['num_steps']:,} steps, batch_size={dcfg['batch_size']}\n")

    step = start_step
    t0 = time.time()
    log_interval = ocfg["log_every"]
    loss_history: list[float] = []  # rolling window for smoothed loss

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
        # Keep as Python bools for use in Python conditionals (no GPU sync).
        # Wrap in mx.array only for passing into the compiled loss fn (mx.where).
        null_image = random.random() < tcfg["image_dropout_prob"]
        null_text  = random.random() < tcfg["text_dropout_prob"]
        use_null_image = mx.array(null_image)
        use_null_text  = mx.array(null_text)

        # Encode frozen models — use pre-computed cache if available (§2.7)
        if vae_np is not None:
            latents = mx.array(vae_np, dtype=mx.bfloat16)
        else:
            latents = _vae_encode(vae, images)

        if text_np is not None:
            text_embeds = mx.array(text_np, dtype=mx.bfloat16)
            if null_text:
                text_embeds = mx.zeros_like(text_embeds)
        else:
            captions_in = [""] * len(captions) if null_text else captions
            text_embeds = _encode_text(text_encoder, captions_in)

        if siglip_np is not None:
            siglip_feats = mx.array(siglip_np, dtype=mx.bfloat16)
        elif siglip is not None:
            siglip_feats = siglip(images)
        else:
            # Cache miss — use zero features (neutral image conditioning).
            # Happens when siglip precompute is partial; treated as null-image dropout.
            B = images.shape[0]
            siglip_feats = mx.zeros((B, 729, acfg["siglip_dim"]), dtype=mx.bfloat16)

        # Forward + backward + optimizer update (compiled step)
        loss_val = compiled_step(
            images, text_embeds, latents, siglip_feats,
            use_null_image, use_null_text,
        )

        # async_eval: overlaps ~5–10ms Python overhead with Metal execution
        mx.async_eval(loss_val, adapter.parameters())

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
            loss_history.append(loss_scalar)
            if len(loss_history) > 20:
                loss_history.pop(0)
            loss_smooth = sum(loss_history) / len(loss_history)
            steps_remaining = tcfg["num_steps"] - step
            eta_s = steps_remaining / steps_per_sec if steps_per_sec > 0 else 0
            eta_h, eta_m = divmod(int(eta_s) // 60, 60)
            print(
                f"step {step:>7,}/{tcfg['num_steps']:,}"
                f"  loss {loss_scalar:.4f} (avg {loss_smooth:.4f})"
                f"  lr {lr_now:.2e}"
                f"  {steps_per_sec:.2f} steps/s"
                f"  ETA {eta_h}h{eta_m:02d}m",
                flush=True,
            )
            if wandb_run:
                wandb_run.log(
                    {"loss": loss_scalar, "loss_smooth": loss_smooth,
                     "lr": lr_now, "steps_per_sec": steps_per_sec},
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
# SigLIP wrappers
# ─────────────────────────────────────────────────────────────────────────────

class _SigLIPMLXWrapper:
    """Thin wrapper around an mlx_vlm SigLIP vision model."""

    def __init__(self, vision_model):
        self._model = vision_model

    def freeze(self):
        if hasattr(self._model, "freeze"):
            self._model.freeze()

    def __call__(self, images: mx.array) -> mx.array:
        return self._model(images).astype(mx.bfloat16)  # [B, 729, 1152]


class _SigLIPTorchWrapper:
    """Wraps a transformers SigLIP vision model with numpy bridge for MLX."""

    def __init__(self, hf_model):
        self._model = hf_model

    def freeze(self):
        pass  # always frozen; no MLX-style freeze needed

    def __call__(self, images: mx.array) -> mx.array:
        import torch
        import numpy as np
        imgs_np = np.array(images.astype(mx.float32))
        imgs_t = torch.from_numpy(imgs_np)
        with torch.no_grad():
            out = self._model(pixel_values=imgs_t).last_hidden_state  # [B, 729, 1152]
        return mx.array(out.numpy()).astype(mx.bfloat16)


# ─────────────────────────────────────────────────────────────────────────────
# Text encoder bundle
# ─────────────────────────────────────────────────────────────────────────────

class _TextEncoderBundle:
    """Bundles Qwen3 encoder + tokenizer so _encode_text can tokenise and encode."""

    def __init__(self, encoder, tokenizer):
        self.encoder = encoder
        self.tokenizer = tokenizer

    def freeze(self):
        self.encoder.freeze()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_siglip(model_name: str):
    """
    Load SigLIP SO400M vision encoder (frozen).
    Tries mlx_vlm first; falls back to transformers + torch.

    Input:  [B, 3, 384, 384] normalised to [-1, 1]
    Output: [B, 729, 1152]

    In practice SigLIP features are pre-computed by precompute_siglip.py.
    This path is only hit when siglip_cache_dir is None.
    """
    try:
        from mlx_vlm import load as vlm_load
        model, _ = vlm_load(model_name)
        model.eval()
        vm = model.vision_model if hasattr(model, "vision_model") else model
        return _SigLIPMLXWrapper(vm)
    except Exception:
        pass

    try:
        from transformers import AutoModel
        hf_model = AutoModel.from_pretrained(model_name).vision_model.eval()
        return _SigLIPTorchWrapper(hf_model)
    except Exception as e:
        raise RuntimeError(
            f"Cannot load SigLIP '{model_name}': {e}\n"
            "Install mlx-vlm: pip install mlx-vlm\n"
            "Or pre-compute features: python train/scripts/precompute_siglip.py "
            "--shards train/data/shards --output train/data/precomputed/siglip"
        ) from e


def _load_vae(flux):
    """Extract VAE encoder from the loaded Flux object."""
    return flux.vae


def _load_text_encoder(flux):
    """Return a _TextEncoderBundle (encoder + tokenizer) from the Flux object."""
    return _TextEncoderBundle(flux.text_encoder, flux.tokenizers["qwen3"])


# ─────────────────────────────────────────────────────────────────────────────
# Encode helpers
# ─────────────────────────────────────────────────────────────────────────────

def _vae_encode(vae, images: mx.array) -> mx.array:
    """
    Encode images to VAE latents.
    images: [B, 3, H, W] BF16 in [-1, 1]
    Returns: [B, 32, H/8, W/8]
    """
    return vae.encode(images)


def _encode_text(text_encoder: _TextEncoderBundle, captions: list) -> mx.array:
    """
    Encode text captions using Qwen3 (frozen).
    Returns: [B, seq, text_dim] BF16
    """
    tokens = text_encoder.tokenizer.tokenize(captions)
    embeds = text_encoder.encoder.get_prompt_embeds(
        input_ids=tokens.input_ids,
        attention_mask=tokens.attention_mask,
        hidden_state_layers=(9, 18, 27),
    )
    return embeds.astype(mx.bfloat16)


# ─────────────────────────────────────────────────────────────────────────────
# Flux forward pass with IP-Adapter injection
# ─────────────────────────────────────────────────────────────────────────────

def _flux_forward_with_ip(
    flux,
    noisy_latents: mx.array,
    text_embeds: mx.array,
    t_int: mx.array,
    k_ip_all: mx.array,
    v_ip_all: mx.array,
    ip_scale: mx.array,
    ckpt_double=None,
    ckpt_single=None,
) -> mx.array:
    """
    Flux Klein 4B forward pass with IP-Adapter injection.

    Manually replays the transformer forward pass so that after each double-stream
    block's standard attention we add:
        ip_out = SDPA(img_Q, k_ip[i], v_ip[i])
        hidden_states += ip_scale[i] * ip_out

    For single-stream blocks the same injection is applied to the image portion of
    the concatenated [TEXT | IMAGE] sequence.

    Args:
        flux:           frozen Flux2Klein model
        noisy_latents:  [B, 32, H/8, W/8] noisy latents in VAE space
        text_embeds:    [B, seq_txt, 7680] Qwen3 embeddings
        t_int:          [B] or [1] integer timestep in [0, 1000)
        k_ip_all:       [B, num_blocks, 128, 3072] IP keys
        v_ip_all:       [B, num_blocks, 128, 3072] IP values
        ip_scale:       [num_blocks] per-block scale (from adapter.scale)

    Returns:
        pred: [B, 32, H/8, W/8] velocity prediction in VAE latent space
    """
    tr = flux.transformer

    # ── Step 1: patchify + pack noisy latents ─────────────────────────────────
    # VAE latents: [B, 32, Lh, Lw]  (Lh = H/8, Lw = W/8)
    # Patchify:    [B, 128, pH, pW]  (pH = Lh/2 = H/16)
    # Pack:        [B, seq_img, 128]
    B, C, Lh, Lw = noisy_latents.shape
    pH, pW = Lh // 2, Lw // 2

    h = noisy_latents.reshape(B, C, pH, 2, pW, 2)   # [B, 32, pH, 2, pW, 2]
    h = h.transpose(0, 1, 3, 5, 2, 4)               # [B, 32, 2, 2, pH, pW]
    h = h.reshape(B, C * 4, pH, pW)                  # [B, 128, pH, pW]
    hidden_states = h.reshape(B, 128, pH * pW).transpose(0, 2, 1)  # [B, seq_img, 128]
    seq_img = pH * pW

    # ── Step 2: position IDs ──────────────────────────────────────────────────
    h_grid = mx.broadcast_to(
        mx.arange(pH, dtype=mx.int32)[:, None], (pH, pW)
    ).reshape(-1)
    w_grid = mx.broadcast_to(
        mx.arange(pW, dtype=mx.int32)[None, :], (pH, pW)
    ).reshape(-1)
    img_ids = mx.stack(
        [mx.zeros(seq_img, dtype=mx.int32), h_grid, w_grid,
         mx.zeros(seq_img, dtype=mx.int32)],
        axis=1,
    )  # [seq_img, 4]

    seq_txt = text_embeds.shape[1]
    txt_ids = mx.stack(
        [mx.zeros(seq_txt, dtype=mx.int32),
         mx.zeros(seq_txt, dtype=mx.int32),
         mx.zeros(seq_txt, dtype=mx.int32),
         mx.arange(seq_txt, dtype=mx.int32)],
        axis=1,
    )  # [seq_txt, 4]

    # ── Step 3: timestep embedding ────────────────────────────────────────────
    # Klein 4B: guidance_embeds=False, so pass guidance=None
    if not isinstance(t_int, mx.array):
        timestep = mx.array(t_int, dtype=hidden_states.dtype)
    else:
        timestep = t_int.astype(hidden_states.dtype)
    if timestep.ndim == 0:
        timestep = mx.full((B,), float(timestep.item()), dtype=hidden_states.dtype)
    elif timestep.shape[0] == 1 and B > 1:
        timestep = mx.broadcast_to(timestep, (B,))

    temb = tr.time_guidance_embed(timestep, guidance=None)
    temb = temb.astype(mx.bfloat16)

    # ── Step 4: project inputs ────────────────────────────────────────────────
    hidden_states = tr.x_embedder(hidden_states)            # [B, seq_img, inner_dim]
    encoder_hidden_states = tr.context_embedder(text_embeds)  # [B, seq_txt, inner_dim]

    # ── Step 5: RoPE ──────────────────────────────────────────────────────────
    image_rotary_emb = tr.pos_embed(img_ids)
    text_rotary_emb = tr.pos_embed(txt_ids)
    concat_rotary_emb = (
        mx.concatenate([text_rotary_emb[0], image_rotary_emb[0]], axis=0),
        mx.concatenate([text_rotary_emb[1], image_rotary_emb[1]], axis=0),
    )

    # ── Step 6: double-stream modulation (shared across all double blocks) ────
    temb_mod_params_img = tr.double_stream_modulation_img(temb)
    temb_mod_params_txt = tr.double_stream_modulation_txt(temb)
    # Unpack the AdaLN-Zero shift/scale for IP Q extraction (same for all blocks)
    (shift_msa_img, scale_msa_img, _), _ = temb_mod_params_img

    # ── Step 7: double-stream blocks + IP injection ───────────────────────────
    for i, block in enumerate(tr.transformer_blocks):
        h_before = hidden_states  # save pre-block state to recompute Q

        call_block = ckpt_double[i] if ckpt_double is not None else block
        encoder_hidden_states, hidden_states = call_block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb_mod_params_img=temb_mod_params_img,
            temb_mod_params_txt=temb_mod_params_txt,
            image_rotary_emb=concat_rotary_emb,
        )

        # Recompute normalised image hidden state to extract Q for IP attention
        norm_h = block.norm1(h_before)
        norm_h = (1 + scale_msa_img) * norm_h + shift_msa_img  # [B, seq_img, inner_dim]

        # Q projection + RMSNorm (matches double-block attention path)
        q_ip = block.attn.to_q(norm_h)
        bsz, s_img, d_inner = q_ip.shape
        H = block.attn.heads
        Hd = block.attn.dim_head
        q_ip = q_ip.reshape(bsz, s_img, H, Hd)
        q_ip = block.attn.norm_q(q_ip.astype(mx.float32)).astype(mx.bfloat16)
        q_ip = q_ip.transpose(0, 2, 1, 3)  # [B, heads, seq_img, head_dim]

        # IP K/V for this block: k_ip_all[:, i] is [B, 128, inner_dim]
        k_i = k_ip_all[:, i, :, :]  # [B, 128, inner_dim]
        v_i = v_ip_all[:, i, :, :]
        k_i = k_i.reshape(bsz, -1, H, Hd).transpose(0, 2, 1, 3)
        v_i = v_i.reshape(bsz, -1, H, Hd).transpose(0, 2, 1, 3)

        ip_out = mx.fast.scaled_dot_product_attention(
            q_ip, k_i, v_i, scale=Hd ** -0.5,
        )  # [B, heads, seq_img, head_dim]
        ip_out = ip_out.transpose(0, 2, 1, 3).reshape(bsz, s_img, d_inner)

        hidden_states = hidden_states + ip_scale[i] * ip_out

    # ── Step 8: merge streams ─────────────────────────────────────────────────
    hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)

    # ── Step 9: single-stream modulation (shared across all single blocks) ────
    temb_mod_params_single = tr.single_stream_modulation(temb)[0]
    mod_shift_s, mod_scale_s, _ = temb_mod_params_single

    n_double = len(tr.transformer_blocks)  # 5 for Klein 4B

    # ── Step 10: single-stream blocks + IP injection ──────────────────────────
    for j, block in enumerate(tr.single_transformer_blocks):
        h_before_s = hidden_states

        call_block_s = ckpt_single[j] if ckpt_single is not None else block
        hidden_states = call_block_s(
            hidden_states=hidden_states,
            temb_mod_params=temb_mod_params_single,
            image_rotary_emb=concat_rotary_emb,
        )

        # IP injection for image tokens: recompute Q from image portion
        block_ip_idx = n_double + j

        norm_h_full = block.norm(h_before_s)
        norm_h_full = (1 + mod_scale_s) * norm_h_full + mod_shift_s
        img_norm_h = norm_h_full[:, seq_txt:, :]  # [B, seq_img, inner_dim]

        # Fused QKV+MLP projection — extract Q only
        proj_img = block.attn.to_qkv_mlp_proj(img_norm_h)
        qkv_img, _ = mx.split(proj_img, [block.attn.inner_dim * 3], axis=-1)
        q_ip_s, _, _ = mx.split(qkv_img, 3, axis=-1)

        bsz2, s2, d2 = q_ip_s.shape
        H_s = block.attn.heads
        Hd_s = block.attn.dim_head
        q_ip_s = q_ip_s.reshape(bsz2, s2, H_s, Hd_s)
        q_ip_s = block.attn.norm_q(q_ip_s.astype(mx.float32)).astype(mx.bfloat16)
        q_ip_s = q_ip_s.transpose(0, 2, 1, 3)  # [B, heads, seq_img, head_dim]

        k_i2 = k_ip_all[:, block_ip_idx, :, :]
        v_i2 = v_ip_all[:, block_ip_idx, :, :]
        k_i2 = k_i2.reshape(bsz2, -1, H_s, Hd_s).transpose(0, 2, 1, 3)
        v_i2 = v_i2.reshape(bsz2, -1, H_s, Hd_s).transpose(0, 2, 1, 3)

        ip_out_s = mx.fast.scaled_dot_product_attention(
            q_ip_s, k_i2, v_i2, scale=Hd_s ** -0.5,
        )
        ip_out_s = ip_out_s.transpose(0, 2, 1, 3).reshape(bsz2, s2, d2)

        # Add IP contribution to image portion of hidden_states
        hidden_states = mx.concatenate([
            hidden_states[:, :seq_txt, :],
            hidden_states[:, seq_txt:, :] + ip_scale[block_ip_idx] * ip_out_s,
        ], axis=1)

    # ── Step 11: extract image tokens, final norm + proj ─────────────────────
    hidden_states = hidden_states[:, seq_txt:, :]
    hidden_states = tr.norm_out(hidden_states, temb)
    hidden_states = tr.proj_out(hidden_states)  # [B, seq_img, 128]

    # ── Step 12: unpack → unatchify → latent space ────────────────────────────
    # [B, seq_img, 128] → [B, 128, pH, pW] → [B, 32, H/8, W/8]
    pred = hidden_states.transpose(0, 2, 1).reshape(B, C * 4, pH, pW)
    pred = pred.reshape(B, C, 2, 2, pH, pW)
    pred = pred.transpose(0, 1, 4, 2, 5, 3)  # [B, C, pH, 2, pW, 2]
    pred = pred.reshape(B, C, Lh, Lw)         # [B, 32, H/8, W/8]

    return pred


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train IP-Adapter for Flux Klein 4B")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--resume", default=None,
                        help="Resume from checkpoint .safetensors path")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate from config (use lower LR for later chunks)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max_steps from config")
    parser.add_argument("--anchor-shards", default=None,
                        help="Path to anchor shard directory mixed into every chunk (20%% of batches)")
    parser.add_argument("--hard-examples", default=None,
                        help="Path to hard example shard directory (from mine_hard_examples.py); "
                             "mixed at hard_mix_ratio (default 5%%)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config without training")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.resume:
        config["model"]["warmstart_path"] = args.resume
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.max_steps is not None:
        config["training"]["num_steps"] = args.max_steps
    if args.anchor_shards is not None:
        config["data"]["anchor_shard_dir"] = args.anchor_shards
    if args.hard_examples is not None:
        config["data"]["hard_example_dir"] = args.hard_examples

    if args.dry_run:
        print("Config OK:")
        print(yaml.dump(config, default_flow_style=False))
        return

    train(config)


if __name__ == "__main__":
    main()
