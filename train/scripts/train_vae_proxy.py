#!/usr/bin/env python3
"""
train/scripts/train_vae_proxy.py — Proxy VAE distillation trainer.

Trains a lightweight StudentEncoder (~6-10M params) to approximate the
Flux2 VAE encoder via output + perceptual + optional feature distillation.

Usage:
    source train/.venv/bin/activate
    python train/scripts/train_vae_proxy.py \\
        --config train/configs/vae_proxy_512px.yaml

Resume from checkpoint:
    python train/scripts/train_vae_proxy.py \\
        --config train/configs/vae_proxy_512px.yaml \\
        --resume /Volumes/2TBSSD/checkpoints/vae_proxy/step_0010000.safetensors

See plans/precomp2-proxy-vae-design.md for architecture and loss rationale.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mx_utils

from vae_distill.teacher  import TeacherEncoder
from vae_distill.student  import build_student
from vae_distill.losses   import DistillationLoss, compute_channel_stats
from vae_distill.dataset  import build_index, VAEDistillDataset
from vae_distill.proxy    import save_proxy_checkpoint

from pipeline_lib import write_heartbeat, now_iso, DATA_ROOT


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _cfg(cfg: dict, *keys, default=None):
    """Nested dict get with dot-path keys list."""
    d = cfg
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
        if d is default:
            return default
    return d


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    step: int,
    student,
    optimizer,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    teacher: TeacherEncoder,
    cfg: dict,
    ckpt_dir: Path,
    feat_adapters=None,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:07d}.safetensors"

    # Full training checkpoint: student weights + optimizer state + optional adapters
    payload: dict[str, mx.array] = {}
    for k, v in mx_utils.tree_flatten(student.parameters()):
        payload[f"student.{k}"] = v
    # Save feature adapter weights when present (separate prefix for clean loading)
    if feat_adapters:
        for ai, adapter in enumerate(feat_adapters):
            for k, v in mx_utils.tree_flatten(adapter.parameters()):
                payload[f"adapter{ai}.{k}"] = v
    opt_state = optimizer.state
    for k, v in mx_utils.tree_flatten(opt_state):
        if isinstance(v, mx.array):
            payload[f"opt.{k}"] = v

    payload["meta.channel_mean"]   = mx.array(channel_mean.astype(np.float32))
    payload["meta.channel_std"]    = mx.array(channel_std.astype(np.float32))
    payload["meta.scaling_factor"] = mx.array(np.float32(teacher.scaling_factor))
    payload["meta.shift_factor"]   = mx.array(np.float32(teacher.shift_factor))
    payload["meta.step"]           = mx.array(np.int32(step))
    payload["meta.config_json"]    = mx.array(
        np.frombuffer(json.dumps(cfg).encode(), dtype=np.uint8)
    )

    mx.eval(list(payload.values()))
    tmp = str(path) + ".tmp"
    mx.save_safetensors(tmp, payload)
    os.replace(tmp, str(path))
    size_mb = path.stat().st_size / 1e6
    print(f"  checkpoint: {path.name}  ({size_mb:.0f} MB)")

    _purge_old_checkpoints(ckpt_dir,
                           keep=_cfg(cfg, "output", "keep_last_n", default=3))
    return path


def _purge_old_checkpoints(ckpt_dir: Path, keep: int = 3) -> None:
    ckpts = sorted(ckpt_dir.glob("step_*.safetensors"),
                   key=lambda p: int(p.stem.split("_")[1]))
    for old in ckpts[:-keep]:
        old.unlink(missing_ok=True)


def _resume(path: str, student, optimizer, feat_adapters=None) -> int:
    weights = mx.load(path)
    # Restore student weights
    student_w = {k[len("student."):]: v
                 for k, v in weights.items() if k.startswith("student.")}
    student.load_weights(list(student_w.items()))
    # Restore feature adapter weights if present
    if feat_adapters:
        for ai, adapter in enumerate(feat_adapters):
            prefix = f"adapter{ai}."
            adapter_w = {k[len(prefix):]: v
                         for k, v in weights.items() if k.startswith(prefix)}
            if adapter_w:
                adapter.load_weights(list(adapter_w.items()))
    # Restore optimizer state (AdamW moments) so momentum carries over correctly
    opt_saved = {k[len("opt."):]: v
                 for k, v in weights.items() if k.startswith("opt.")}
    if opt_saved:
        try:
            # Reconstruct the nested state dict and assign it to the optimizer.
            # mx_utils.tree_unflatten converts a flat {key: array} dict back to
            # the nested structure that optimizer.state expects.
            nested = mx_utils.tree_unflatten(list(opt_saved.items()))
            optimizer.state = nested
        except Exception as e:
            print(f"  WARNING: could not restore optimizer state ({e}); "
                  f"optimizer starts from scratch")
    start_step = int(np.array(weights.get("meta.step", np.int32(0))))
    print(f"  resumed from {path} at step {start_step:,}")
    return start_step


# ---------------------------------------------------------------------------
# Channel statistics bootstrap
# ---------------------------------------------------------------------------

def _bootstrap_channel_stats(
    vae_cache_dir: str,
    n_samples: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean/std by sampling n_samples random latent npz files.
    Cached to <vae_cache_dir>/channel_stats.npz to avoid repeated computation.
    """
    stats_path = Path(vae_cache_dir) / "channel_stats.npz"
    if stats_path.exists():
        d = np.load(str(stats_path))
        print(f"  [stats] loaded channel stats from {stats_path}")
        return d["mean"], d["std"]

    print(f"  [stats] computing channel stats from {n_samples} samples ...")
    npz_files = list(Path(vae_cache_dir).glob("*.npz"))
    if not npz_files:
        raise RuntimeError(f"No .npz files found in {vae_cache_dir}")

    rng = np.random.default_rng(0)
    sample = rng.choice(npz_files, size=min(n_samples, len(npz_files)),
                        replace=False)

    latents = []
    for p in sample:
        try:
            lat = np.load(str(p))["latent"].astype(np.float32)
            latents.append(lat)
        except Exception:
            pass

    mean, std = compute_channel_stats(latents)
    np.savez(str(stats_path), mean=mean, std=std)
    print(f"  [stats] channel mean range: [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"  [stats] channel std  range: [{std.min():.3f},  {std.max():.3f}]")
    return mean, std


# ---------------------------------------------------------------------------
# Feature adapter modules (for optional feature distillation)
# ---------------------------------------------------------------------------

def _build_feature_adapters(student_channels, teacher_channels, norm_groups=16):
    """1×1 projection convs: student_ch → teacher_ch per feature level."""
    adapters = []
    for sc, tc in zip(student_channels, teacher_channels):
        adapters.append(nn.Conv2d(sc, tc, 1, padding=0))
    return adapters


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict, resume_path: Optional[str] = None) -> None:
    # ── Paths ──────────────────────────────────────────────────────────────
    flux_model   = _cfg(cfg, "teacher", "flux_model_dir", default="flux-klein-model")
    vae_cache    = _cfg(cfg, "data", "vae_cache_dir",
                        default=str(DATA_ROOT / "precomputed" / "vae" / "current"))
    shard_dirs   = _cfg(cfg, "data", "shard_dirs", default=[str(DATA_ROOT / "shards")])
    ckpt_dir     = Path(_cfg(cfg, "output", "checkpoint_dir",
                             default=str(DATA_ROOT / "checkpoints" / "vae_proxy")))
    image_size   = _cfg(cfg, "data", "image_size",   default=512)
    batch_size   = _cfg(cfg, "data", "batch_size",   default=8)
    min_score    = _cfg(cfg, "data", "min_unified_score", default=0.0)
    max_shards   = _cfg(cfg, "data", "max_shards",   default=None)
    num_steps    = _cfg(cfg, "training", "num_steps", default=50000)
    warmup_steps = _cfg(cfg, "training", "warmup_steps", default=500)
    lr           = float(_cfg(cfg, "training", "learning_rate", default=2e-4))
    wd           = float(_cfg(cfg, "training", "weight_decay",  default=1e-2))
    grad_clip    = float(_cfg(cfg, "training", "grad_clip",     default=1.0))
    log_every    = _cfg(cfg, "training", "log_every",    default=100)
    ckpt_every   = _cfg(cfg, "training", "checkpoint_every", default=2000)
    qat_bits     = _cfg(cfg, "training", "qat_bits",     default=None)
    use_feats    = _cfg(cfg, "losses",   "feature_weight", default=0.0) > 0.0
    seed         = _cfg(cfg, "data",     "seed",          default=42)

    # ── Memory cap ─────────────────────────────────────────────────────────
    try:
        mx.set_memory_limit(14 * 1024 ** 3)
    except AttributeError:
        pass

    # ── Teacher ────────────────────────────────────────────────────────────
    print("[train_vae_proxy] Loading teacher ...")
    teacher = TeacherEncoder.load(flux_model)
    teacher._vae.eval()
    for p in mx_utils.tree_flatten(teacher._vae.parameters()):
        pass  # ensure loaded

    # ── Channel statistics ─────────────────────────────────────────────────
    channel_mean, channel_std = _bootstrap_channel_stats(vae_cache)

    # ── Student ────────────────────────────────────────────────────────────
    print("[train_vae_proxy] Building student ...")
    student = build_student(cfg)
    n_params = student.param_count()
    print(f"  student parameters: {n_params:,} (~{n_params/1e6:.1f}M)")

    # ── Feature adapters ──────────────────────────────────────────────────
    feat_adapters = None
    if use_feats:
        teacher_channels = [128, 256, 512, 512]
        student_channels = _cfg(cfg, "student", "channels", default=[64, 128, 256, 256])
        feat_adapters = _build_feature_adapters(student_channels, teacher_channels)
        print(f"  feature distillation enabled ({len(feat_adapters)} adapter levels)")

    # ── Loss ───────────────────────────────────────────────────────────────
    loss_cfg    = cfg.get("losses", {})
    distill_loss = DistillationLoss(
        channel_std       = channel_std,
        latent_mse_weight = float(loss_cfg.get("latent_mse_weight",  1.0)),
        decoded_mse_weight= float(loss_cfg.get("decoded_mse_weight", 0.1)),
        freq_alpha        = float(loss_cfg.get("freq_alpha",         0.5)),
        dist_weight       = float(loss_cfg.get("dist_weight",        0.01)),
        feature_weight    = float(loss_cfg.get("feature_weight",     0.0)),
    )

    def _lr_schedule(step: int) -> float:
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        return lr  # constant after warmup (cosine decay can be added later)

    optimizer = optim.AdamW(learning_rate=lr, weight_decay=wd)

    # ── Dataset ────────────────────────────────────────────────────────────
    print("[train_vae_proxy] Building dataset index ...")
    index = build_index(
        shard_dirs=shard_dirs,
        vae_cache_dir=vae_cache,
        min_unified_score=min_score,
        max_shards=max_shards,
    )
    if not index:
        print("ERROR: no training pairs found. Run generate_teacher_latents.py first.",
              file=sys.stderr)
        sys.exit(1)

    dataset = VAEDistillDataset(
        index=index,
        vae_cache=vae_cache,
        image_size=image_size,
        batch_size=batch_size,
        hflip_prob=0.5,
        seed=seed,
    )

    # ── Resume ─────────────────────────────────────────────────────────────
    start_step = 0
    if resume_path:
        start_step = _resume(resume_path, student, optimizer, feat_adapters)

    # ── Loss/grad function ─────────────────────────────────────────────────
    # Freeze teacher — no gradients flow through it.
    teacher._vae.freeze()

    # Auxiliary loss components captured via mutable reference (MLX pattern
    # from train_ip_adapter.py) since nn.value_and_grad returns only a scalar.
    _aux: dict[str, mx.array] = {}

    def _loss_fn(images_mx, teacher_latents_mx):
        if use_feats:
            proxy_latent, s_feats = student(
                images_mx, return_features=True, qat_bits=qat_bits)
            t_latent, t_feats = teacher.encode_with_features(images_mx)
        else:
            proxy_latent = student(images_mx, qat_bits=qat_bits)
            t_latent     = teacher_latents_mx
            s_feats = t_feats = None

        total, components = distill_loss(
            proxy_latent   = proxy_latent,
            teacher_latent = mx.stop_gradient(t_latent),
            teacher_decoder= teacher.decode if distill_loss.w_decoded > 0 else None,
            scaling_factor = teacher.scaling_factor,
            shift_factor   = teacher.shift_factor,
            student_feats  = s_feats,
            teacher_feats  = t_feats,
            feat_adapters  = feat_adapters,
        )
        _aux.update(components)
        return total  # scalar — nn.value_and_grad requires scalar return

    # nn.value_and_grad(model, fn): computes gradients of fn w.r.t. model's
    # trainable parameters. fn closes over student; images/latents are args.
    loss_and_grad = nn.value_and_grad(student, _loss_fn)

    # When feature adapters are used they are separate nn.Module instances and
    # need their own gradient pass.  Build a combined model wrapper so the
    # optimizer sees all parameters in one update.
    if feat_adapters:
        class _Combined(nn.Module):
            def __init__(self):
                super().__init__()
                self.student = student
                self.adapters = feat_adapters
        _combined = _Combined()
        loss_and_grad = nn.value_and_grad(_combined, _loss_fn)

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"[train_vae_proxy] Training {num_steps:,} steps  "
          f"(batch={batch_size}, lr={lr:.1e})  ...")
    print(f"  checkpoint_dir: {ckpt_dir}")
    print(f"  dataset: {len(index):,} pairs, {len(dataset):,} batches/epoch")

    step        = start_step
    log_loss    = 0.0
    log_terms: dict[str, float] = {}
    log_t0      = time.time()

    def _data_loop():
        while True:
            for batch in dataset:
                yield batch

    data_iter = _data_loop()

    while step < num_steps:
        images_np, latents_np = next(data_iter)
        images_mx  = mx.array(images_np)
        latents_mx = mx.array(latents_np)

        # Update learning rate
        new_lr = _lr_schedule(step)
        optimizer.learning_rate = new_lr

        # Forward + backward — loss_and_grad differentiates w.r.t. student params
        _aux.clear()
        loss_val, grads = loss_and_grad(images_mx, latents_mx)

        # Gradient clip
        grads, grad_norm = optim.clip_grad_norm(grads, max_norm=grad_clip)

        # Optimizer step
        model_to_update = _combined if feat_adapters else student
        optimizer.update(model_to_update, grads)
        mx.eval(student.parameters(), optimizer.state, loss_val)

        # Collect logged component values (evaluated lazily above)
        mx.eval(list(_aux.values()))
        loss_f = float(loss_val)
        log_loss += loss_f
        for k, v in _aux.items():
            log_terms[k] = log_terms.get(k, 0.0) + float(v)

        step += 1

        if step % log_every == 0:
            elapsed = time.time() - log_t0
            steps_per_sec = log_every / elapsed
            avg_loss  = log_loss / log_every
            term_strs = "  ".join(
                f"{k}={v/log_every:.4f}" for k, v in log_terms.items()
            )
            print(
                f"  step {step:>7,}/{num_steps:,}  "
                f"loss={avg_loss:.4f}  {term_strs}  "
                f"lr={new_lr:.2e}  grad_norm={float(grad_norm):.3f}  "
                f"{steps_per_sec:.1f}s/s  {now_iso()}"
            )
            write_heartbeat("vae_proxy_trainer", status="training",
                            step=step, loss=avg_loss)
            log_loss = 0.0
            log_terms.clear()
            log_t0 = time.time()

        if step % ckpt_every == 0:
            _save_checkpoint(step, student, optimizer,
                             channel_mean, channel_std,
                             teacher, cfg, ckpt_dir, feat_adapters)
            # Also save inference-ready proxy checkpoint
            proxy_path = ckpt_dir / f"proxy_step_{step:07d}.safetensors"
            save_proxy_checkpoint(
                str(proxy_path), student,
                channel_mean, channel_std,
                teacher.scaling_factor, teacher.shift_factor,
                cfg,
            )

    # Final checkpoint
    _save_checkpoint(step, student, optimizer, channel_mean, channel_std,
                     teacher, cfg, ckpt_dir, feat_adapters)
    proxy_final = ckpt_dir / "proxy_final.safetensors"
    save_proxy_checkpoint(
        str(proxy_final), student, channel_mean, channel_std,
        teacher.scaling_factor, teacher.shift_factor, cfg,
    )
    print(f"[train_vae_proxy] Training complete.  Proxy: {proxy_final}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train proxy VAE encoder")
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--resume", default=None, help="Resume from checkpoint")
    args = ap.parse_args()

    cfg = load_config(args.config)
    train(cfg, resume_path=args.resume)
