#!/usr/bin/env python3
"""train/lora/train_lora.py — train a LoRA on the frozen Flux Klein transformer (framework Phase 1).

Reuses the IP-adapter's VALIDATED data path: make_prefetch_loader (VAE+Qwen3 cache) + the VAE-Q1
_bn_pack_latents transform, so LoRA trains in the exact same latent space the frozen base operates in.
Trains only the LoRA (piece-1 freeze), exports to the Diffusers safetensors iris_lora.c loads.

  train/.venv/bin/python train/lora/train_lora.py --config <cfg.yaml> --out <lora.safetensors> [--max-steps N]
"""
import argparse, glob, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from ip_adapter.dataset import make_prefetch_loader
from train_ip_adapter import _load_vae_bn_stats, _bn_pack_latents   # reuse the VAE-Q1 prep (single source)

from lora.lora import inject_lora_double_blocks, inject_lora_single_blocks, lora_param_count
from lora.train_step import lora_loss, make_ckpt_blocks
from lora.export import export_lora_diffusers


def _set_mlx_memory(pct: float):
    """Regular-pipeline lesson (BUGS MLX-2): cap MLX GPU allocation + cache so the allocator doesn't
    churn-reclaim into the huge-graph 'wedge'. mlx_memory_pct ~0.6–0.66 (21 GB on 32 GB)."""
    try:
        import subprocess
        ram = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
        mx.set_memory_limit(int(ram * pct))
        mx.set_cache_limit(int(ram * 0.06))
        print(f"  MLX memory: limit {ram*pct/1e9:.1f} GB ({pct:.0%}), cache {ram*0.06/1e9:.1f} GB", flush=True)
    except Exception as e:
        print(f"  WARNING: could not set MLX memory limit ({e})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="export path for the trained LoRA (.safetensors)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--no-ema", action="store_true", help="disable EMA; export the online weights")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    d, t, m, a = cfg["data"], cfg.get("training", {}), cfg["model"], cfg.get("adapter", {})
    model_dir = os.path.abspath(m["flux_model_dir"])
    steps = args.max_steps or int(t.get("num_steps", 500))
    rank = int(a.get("rank", 16)); alpha = float(a.get("alpha", rank))

    _set_mlx_memory(float(t.get("mlx_memory_pct", 0.6)))   # lesson: set BEFORE the model loads
    print("loading Flux2Klein ...", flush=True)
    flux = Flux2Klein(model_path=model_dir, quantize=None)
    inj = inject_lora_double_blocks(flux, rank=rank, alpha=alpha)
    if a.get("single_blocks", True):                       # full-coverage default (20 single blocks)
        inj += inject_lora_single_blocks(flux, rank=rank, alpha=alpha)
    print(f"injected {len(inj)} LoRA modules (rank {rank}, alpha {alpha}); "
          f"{lora_param_count(flux):,} trainable params", flush=True)
    mx.eval(flux.transformer.parameters())
    # Gradient checkpointing — backprop the LoRA through the full frozen transformer at 512px
    # without the huge-graph stall (regular-pipeline lesson: block_gradient_checkpointing).
    ckpt_double, ckpt_single = make_ckpt_blocks(flux)
    print(f"  gradient checkpointing: {len(ckpt_double)} double + {len(ckpt_single)} single blocks",
          flush=True)

    bn_mean, bn_std = _load_vae_bn_stats(model_dir)
    shard_paths = sorted(glob.glob(os.path.join(d["shard_path"], "*.tar")))
    if not shard_paths:
        raise RuntimeError(f"no .tar shards in {d['shard_path']} (must be hot SSD — AGENT #6)")
    # record_allowlist: train ONLY on the curated rec_ids (curate_style_subset.py → {"rec_ids":[...]}).
    # Without this the loader yields the WHOLE pool — the targeted-style curation would be inert.
    allow = None
    if d.get("record_allowlist"):
        import json
        al = json.load(open(d["record_allowlist"]))
        allow = set(al["rec_ids"] if isinstance(al, dict) else al)
        print(f"record_allowlist: {len(allow)} rec_ids from {d['record_allowlist']}", flush=True)
    print(f"{len(shard_paths)} shards; bucket {d.get('bucket', [512,512])}", flush=True)
    loader = make_prefetch_loader(
        shard_paths=shard_paths, batch_size=int(d.get("batch_size", 1)),
        qwen3_cache_dir=d["qwen3_cache_dir"], vae_cache_dir=d["vae_cache_dir"],
        bucket=tuple(d.get("bucket", [512, 512])), seed=d.get("seed"),
        records_per_shard_visit=d.get("records_per_shard_visit"),
        record_allowlist=allow,
    )

    opt = optim.AdamW(learning_rate=float(t.get("learning_rate", 1e-4)),
                      weight_decay=float(t.get("weight_decay", 0.01)))
    clip = float(t.get("grad_clip", 1.0))

    # EMA over the LoRA params only (NOT update_ema, which averages model.parameters() = the whole
    # frozen 4B). Decay-warmup so short runs don't stay stuck near init (regular-pipeline lesson).
    use_ema = not args.no_ema
    ema_decay = float(t.get("ema_decay", 0.9999))
    ema_every = int(t.get("ema_update_every", 10))
    ema = tree_map(lambda p: mx.array(p), flux.trainable_parameters()) if use_ema else None
    ema_n = 0

    def loss_fn(mod, latents, txt, t_int, noise):
        return lora_loss(mod, latents, txt, t_int, noise, ckpt_double, ckpt_single)
    lg = nn.value_and_grad(flux, loss_fn)

    ckpt_every = int(t.get("checkpoint_every", 0))
    print(f"training {steps} steps ...", flush=True)
    step, t0, skipped = 0, time.time(), 0
    for images_np, captions, text_np, vae_np, *_ in loader:
        if vae_np is None or text_np is None:
            skipped += 1
            continue
        latents = _bn_pack_latents(mx.array(vae_np, dtype=mx.bfloat16), bn_mean, bn_std)
        txt = mx.array(text_np, dtype=mx.bfloat16)
        B = latents.shape[0]
        t_int = mx.clip((mx.sigmoid(mx.random.normal((B,))) * 1000).astype(mx.int32), 0, 999)  # logit-normal
        noise = mx.random.normal(latents.shape)
        loss, grads = lg(flux, latents, txt, t_int, noise)
        grads, gn = optim.clip_grad_norm(grads, max_norm=clip)
        opt.update(flux, grads)
        mx.eval(flux.trainable_parameters(), opt.state, loss)   # fence
        mx.clear_cache()                                        # lesson: bound peak; avoid reclaim churn
        step += 1
        if use_ema and step % ema_every == 0:
            ema_n += 1
            de = min(ema_decay ** ema_every, (1.0 + ema_n) / (10.0 + ema_n))   # decay warmup
            ema = tree_map(lambda e, m: de * e + (1.0 - de) * m, ema, flux.trainable_parameters())
            mx.eval(ema)
        if step == 1 or step % 25 == 0:
            print(f"  step {step}/{steps}  loss {float(loss):.4f}  gnorm {float(gn):.3f}  "
                  f"{step / (time.time() - t0):.2f} it/s  peak {mx.get_peak_memory()/1e9:.1f} GB", flush=True)
        if ckpt_every and step % ckpt_every == 0 and step < steps:
            export_lora_diffusers(flux, args.out)              # periodic checkpoint (overwrites)
            print(f"  checkpoint @ step {step} -> {args.out}", flush=True)
        if step >= steps:
            break

    if use_ema:
        flux.update(ema)                                        # promote EMA = the exported weights
        mx.eval(flux.trainable_parameters())
        print("  exporting EMA weights", flush=True)
    n = export_lora_diffusers(flux, args.out)
    print(f"EXPORTED {n} adapters -> {args.out}  (skipped {skipped} incomplete batches)", flush=True)
    print("TRAINDONE", flush=True)


if __name__ == "__main__":
    main()
