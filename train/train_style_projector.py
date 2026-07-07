"""train/train_style_projector.py — SREF Phase-1 Stage-1 trainer (learned in-sequence style tokens).

Trains the StyleProjector (SigLIP -> 192 style tokens concatenated into the DiT sequence) with the
DiT FROZEN, via full backprop through the frozen backbone (the mechanics are validated by
train/lora/smoke_style.py). Reuses the proven data machinery (make_prefetch_loader, VAE-Q1
BN-packing, augment) so the train<->infer latent space is inherited, not reinvented.

  # verify the trainer loop + checkpointing on synthetic data (no dataset needed):
  train/.venv/bin/python train/train_style_projector.py --smoke
  # real run (REQUIRES staged hot-SSD data — shards + siglip/vae/qwen3 caches):
  train/.venv/bin/python train/train_style_projector.py --config train/configs/sref_projector_v1.yaml

Gate the checkpoints with debug/sref_scorecard.py (kill on collapse; painterly must beat 0.009).
Train on the 4B BASE (steerable). NEVER train from cold storage (hot SSD only — CLAUDE.md).
"""
import argparse, os, sys, time, math, random
sys.path.insert(0, os.path.dirname(__file__))

import yaml
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

from ip_adapter.model import StyleProjector
from lora.train_step import make_ckpt_blocks
from lora.style_step import style_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── VAE-Q1 BN-pack (copied verbatim from train_ip_adapter.py to avoid importing that script) ──
def _load_vae_bn_stats(flux_model_dir):
    try:
        st = mx.load(os.path.join(flux_model_dir, "vae", "diffusion_pytorch_model.safetensors"))
        m = st["bn.running_mean"].astype(mx.float32).reshape(32, 2, 2)
        s = mx.sqrt(st["bn.running_var"].astype(mx.float32).reshape(32, 2, 2) + 1e-4)
        mx.eval(m, s); return m, s
    except Exception as e:
        print(f"  WARNING: VAE-Q1 bn stats unavailable ({e}); latents NOT BN-packed", flush=True)
        return None, None


def _bn_pack(lat, m, s):
    if m is None:
        return lat
    _, _, Lh, Lw = lat.shape
    return ((lat.astype(mx.float32) - mx.tile(m, (1, Lh // 2, Lw // 2)))
            / mx.tile(s, (1, Lh // 2, Lw // 2))).astype(lat.dtype)


def _sample_t(B):
    return mx.clip((mx.sigmoid(mx.random.normal((B,))) * 1000).astype(mx.int32), 0, 999)


def _save_ckpt(projector, ckpt_dir, step):
    os.makedirs(ckpt_dir, exist_ok=True)
    flat = dict(tree_flatten(projector.trainable_parameters()))
    path = os.path.join(ckpt_dir, f"style_projector_{step:07d}.safetensors")
    mx.save_safetensors(path, {k: v for k, v in flat.items()})
    print(f"  saved {path}", flush=True)


def build_lr(base_lr, warmup, total):
    def sched(step):
        if step < warmup:
            return base_lr * (step + 1) / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return sched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true", help="synthetic data — verify the loop/checkpoint")
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config)) if args.config else {}
    mcfg, dcfg, tcfg = cfg.get("model", {}), cfg.get("data", {}), cfg.get("training", {})
    model_dir = os.path.abspath(mcfg.get("flux_model_dir", os.path.join(ROOT, "flux-klein-4b-base")))
    if args.smoke and not args.config:
        model_dir = os.path.join(ROOT, "flux-klein-model")  # distilled: fast loop verification
    n_style = int(cfg.get("adapter", {}).get("num_style_tokens", 192))
    lr = float(tcfg.get("learning_rate", 1e-4))
    warmup = int(tcfg.get("warmup_steps", 200))
    total = args.steps or int(tcfg.get("num_steps", 4000))
    null_style_prob = float(tcfg.get("null_style_prob", 0.1))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    ckpt_every = int(tcfg.get("checkpoint_every", 500))
    ckpt_dir = tcfg.get("checkpoint_dir", "/Volumes/2TBSSD/checkpoints/sref_projector")
    B = int(dcfg.get("batch_size", 1))
    bucket = tuple(dcfg.get("bucket", [512, 512]))
    guidance = tcfg.get("guidance")  # base model CFG guidance (None ok in the flow forward)

    print(f"loading Flux2Klein ({model_dir}) ...", flush=True)
    t0 = time.time()
    flux = Flux2Klein(model_path=model_dir, quantize=None)
    flux.freeze()
    mx.eval(flux.transformer.parameters())
    print(f"  loaded + frozen in {time.time()-t0:.1f}s", flush=True)

    projector = StyleProjector(hidden_dim=3072, num_heads=24, num_style_tokens=n_style, siglip_dim=1152)
    n = sum(v.size for _, v in tree_flatten(projector.trainable_parameters()))
    print(f"StyleProjector: {n/1e6:.1f} M trainable (DiT frozen); n_style={n_style}", flush=True)

    # 512px full backprop through the frozen DiT → gradient checkpointing (TRAIN-7 ~19 GB).
    ckpt_d, ckpt_s = make_ckpt_blocks(flux)
    bn_m, bn_s = _load_vae_bn_stats(model_dir)
    sched = build_lr(lr, warmup, total)
    opt = optim.AdamW(learning_rate=lr, weight_decay=float(tcfg.get("weight_decay", 0.01)))

    def step_loss(p, latent, text, siglip, t_int, noise):
        return style_loss(flux, p, latent, text, siglip, t_int, noise, ckpt_d, ckpt_s, guidance)
    lg = nn.value_and_grad(projector, step_loss)

    def train_step(latent, text, siglip, t_int, noise):
        loss, grads = lg(projector, latent, text, siglip, t_int, noise)
        grads, gn = optim.clip_grad_norm(grads, max_norm=grad_clip)
        opt.update(projector, grads)
        mx.eval(projector.trainable_parameters(), opt.state, loss)
        return float(loss), float(gn)

    # ── data source ───────────────────────────────────────────────────────────
    if args.smoke:
        print("SMOKE: synthetic data (no dataset) — verifying the loop + checkpoint", flush=True)
        Lh = Lw = 16  # 128px, tiny
        mx.random.seed(0)
        fixed = (mx.random.normal((B, 32, Lh, Lw)) * 0.5).astype(mx.bfloat16)
        text = (mx.random.normal((B, 64, 7680)) * 0.1).astype(mx.bfloat16)
        sig = (mx.random.normal((B, 729, 1152))).astype(mx.bfloat16)
        def batches():
            while True:
                yield fixed, text, sig
    else:
        from ip_adapter.dataset import make_prefetch_loader, augment_mlx
        loader = make_prefetch_loader(
            shard_paths=dcfg["shard_paths"], batch_size=B,
            text_dropout_prob=tcfg.get("text_dropout_prob", 0.0),
            qwen3_cache_dir=dcfg.get("qwen3_cache_dir"), vae_cache_dir=dcfg.get("vae_cache_dir"),
            siglip_cache_dir=dcfg.get("siglip_cache_dir"), bucket=bucket,
            style_neighbors_db=dcfg.get("style_neighbors_db"), cond_mode="siglip",
            seed=dcfg.get("seed"))
        def batches():
            for images_np, captions, text_np, vae_np, siglip_np, style_ref_np, bhw, ids in loader:
                if vae_np is None or text_np is None or siglip_np is None:
                    continue  # skip cache-miss batches (real run needs full caches)
                latent = _bn_pack(mx.array(vae_np, dtype=mx.bfloat16), bn_m, bn_s)
                yield latent, mx.array(text_np, dtype=mx.bfloat16), mx.array(siglip_np, dtype=mx.bfloat16)

    # ── train loop ────────────────────────────────────────────────────────────
    print(f"training {total} steps (bucket {bucket}, B={B}, lr {lr}, warmup {warmup}) ...", flush=True)
    t0 = time.time(); step = 0
    for latent, text, siglip in batches():
        if step >= total:
            break
        opt.learning_rate = sched(step)
        # null-style (CFG) dropout: zero the SigLIP for this batch with prob null_style_prob
        sig = mx.zeros_like(siglip) if random.random() < null_style_prob else siglip
        t_int = _sample_t(latent.shape[0])
        noise = mx.random.normal(latent.shape)
        loss, gn = train_step(latent, text, sig, t_int, noise)
        if step % 20 == 0 or step == total - 1:
            sps = (step + 1) / (time.time() - t0)
            print(f"  step {step:6d}  loss {loss:.5f}  gnorm {gn:.3f}  lr {opt.learning_rate.item():.2e}"
                  f"  {sps:.2f} it/s", flush=True)
        if step > 0 and step % ckpt_every == 0:
            _save_ckpt(projector, ckpt_dir, step)
        step += 1
    _save_ckpt(projector, ckpt_dir, step)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
