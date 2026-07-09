"""train/train_style_film.py — SREF experiment B trainer: CSD→timestep-modulation (FiLM).

Trains the small CSDModulation MLP (content-invariant CSD style vector → a delta on the DiT's
timestep/guidance modulation `temb`) with the DiT FROZEN, via full backprop through the frozen
backbone. This is the follow-on to the CLOSED in-sequence style-token path (SREF-STYLE-CFG-PROBE):
the modulation channel is UNIGNORABLE (feeds every block's adaLN at every noise level), unlike the
optional token channel the DiT down-weighted at high noise.

  # verify the loop + that grads reach CSDModulation through the frozen DiT (no dataset, minimal heat):
  train/.venv/bin/python train/train_style_film.py --smoke --steps 30
  # real run (REQUIRES staged hot-SSD data — shards + vae/qwen3 caches + hot CSD cache):
  train/.venv/bin/python train/train_style_film.py --config train/configs/sref_film_v1.yaml

Gate the checkpoints with a CSD→modulation inference harness + debug/sref_scorecard.py (kill on
collapse; painterly must beat band-control 0.009). Train on the 4B BASE. NEVER train from cold
storage (hot SSD only — CLAUDE.md).
"""
import argparse, os, sys, time, math, random
sys.path.insert(0, os.path.dirname(__file__))

import yaml
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

from ip_adapter.model import CSDModulation
from lora.train_step import make_ckpt_blocks
from lora.film_step import film_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── VAE-Q1 BN-pack (copied verbatim from train_style_projector.py) ──
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


def _save_ckpt(csd_mod, ckpt_dir, step):
    os.makedirs(ckpt_dir, exist_ok=True)
    flat = dict(tree_flatten(csd_mod.trainable_parameters()))
    path = os.path.join(ckpt_dir, f"style_film_{step:07d}.safetensors")
    mx.save_safetensors(path, {k: v for k, v in flat.items()})
    print(f"  saved {path}", flush=True)


def build_lr(base_lr, warmup, total):
    def sched(step):
        if step < warmup:
            return base_lr * (step + 1) / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return sched


def _as_csd(x):
    """loader cond in csd mode is [B, csd_dim]; tolerate [B,1,csd_dim]."""
    a = mx.array(x, dtype=mx.bfloat16)
    if a.ndim == 3:
        a = a.reshape(a.shape[0], -1)
    return a


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
    csd_dim = int(cfg.get("adapter", {}).get("csd_dim", 768))
    mlp_dim = int(cfg.get("adapter", {}).get("mlp_dim", 1024))
    lr = float(tcfg.get("learning_rate", 1e-4))
    warmup = int(tcfg.get("warmup_steps", 50))
    total = args.steps or int(tcfg.get("num_steps", 4000))
    null_style_prob = float(tcfg.get("null_style_prob", 0.1))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    ckpt_every = int(tcfg.get("checkpoint_every", 500))
    ckpt_dir = tcfg.get("checkpoint_dir", "/Volumes/2TBSSD/checkpoints/sref_film")
    B = int(dcfg.get("batch_size", 1))
    bucket = tuple(dcfg.get("bucket", [512, 512]))
    guidance = tcfg.get("guidance")  # base model CFG guidance (None ok in the flow forward)

    print(f"loading Flux2Klein ({model_dir}) ...", flush=True)
    t0 = time.time()
    flux = Flux2Klein(model_path=model_dir, quantize=None)
    flux.freeze()
    mx.eval(flux.transformer.parameters())
    print(f"  loaded + frozen in {time.time()-t0:.1f}s", flush=True)

    csd_mod = CSDModulation(hidden_dim=3072, csd_dim=csd_dim, mlp_dim=mlp_dim)
    n = sum(v.size for _, v in tree_flatten(csd_mod.trainable_parameters()))
    print(f"CSDModulation: {n/1e6:.2f} M trainable (DiT frozen); csd_dim={csd_dim} mlp_dim={mlp_dim}", flush=True)

    ckpt_d, ckpt_s = make_ckpt_blocks(flux)
    bn_m, bn_s = _load_vae_bn_stats(model_dir)
    sched = build_lr(lr, warmup, total)
    opt = optim.AdamW(learning_rate=lr, weight_decay=float(tcfg.get("weight_decay", 0.01)))

    def step_loss(mod, latent, text, csd, t_int, noise):
        return film_loss(flux, mod, latent, text, csd, t_int, noise, ckpt_d, ckpt_s, guidance)
    lg = nn.value_and_grad(csd_mod, step_loss)

    def train_step(latent, text, csd, t_int, noise):
        loss, grads = lg(csd_mod, latent, text, csd, t_int, noise)
        grads, gn = optim.clip_grad_norm(grads, max_norm=grad_clip)
        opt.update(csd_mod, grads)
        mx.eval(csd_mod.trainable_parameters(), opt.state, loss)
        return float(loss), float(gn)

    # ── data source ───────────────────────────────────────────────────────────
    if args.smoke:
        print("SMOKE: synthetic data (no dataset) — verifying the loop + checkpoint", flush=True)
        Lh = Lw = 16  # 128px, tiny
        mx.random.seed(0)
        fixed = (mx.random.normal((B, 32, Lh, Lw)) * 0.5).astype(mx.bfloat16)
        text = (mx.random.normal((B, 64, 7680)) * 0.1).astype(mx.bfloat16)
        # per-step synthetic CSD so grad must route the (varying) style vec through the DiT.
        def batches():
            k = 0
            while True:
                c = mx.random.normal((B, csd_dim))
                c = c / mx.maximum(mx.linalg.norm(c, axis=-1, keepdims=True), 1e-6)  # L2 like real CSD
                yield fixed, text, c.astype(mx.bfloat16)
                k += 1
    else:
        import glob as _glob
        from ip_adapter.dataset import make_prefetch_loader
        shard_paths = dcfg.get("shard_paths") or sorted(
            _glob.glob(os.path.join(dcfg["shard_path"], "*.tar")))
        if not shard_paths:
            sys.exit(f"no .tar shards under {dcfg.get('shard_path')}")
        print(f"  {len(shard_paths)} shard tars (hot); cond_mode=csd, caches: vae/qwen3/csd", flush=True)
        loader = make_prefetch_loader(
            shard_paths=shard_paths, batch_size=B,
            text_dropout_prob=tcfg.get("text_dropout_prob", 0.0),
            qwen3_cache_dir=dcfg.get("qwen3_cache_dir"), vae_cache_dir=dcfg.get("vae_cache_dir"),
            siglip_cache_dir=None, bucket=bucket,
            style_neighbors_db=dcfg.get("style_neighbors_db"),
            cond_mode="csd", csd_cache_dir=dcfg.get("csd_cache_dir"),
            seed=dcfg.get("seed"))
        n_cross = [0, 0]
        def batches():
            for images_np, captions, text_np, vae_np, csd_np, style_ref_np, bhw, ids in loader:
                if vae_np is None or text_np is None or csd_np is None:
                    continue  # skip cache-miss batches (real run needs vae/qwen3/csd)
                latent = _bn_pack(mx.array(vae_np, dtype=mx.bfloat16), bn_m, bn_s)
                # DATA-SELECTION PRINCIPLE: prefer the look-NEIGHBOR's CSD (same style, different
                # content → forces style-only use of the channel); fall back to own CSD.
                if style_ref_np is not None:
                    csd = _as_csd(style_ref_np); n_cross[0] += 1
                else:
                    csd = _as_csd(csd_np); n_cross[1] += 1
                yield latent, mx.array(text_np, dtype=mx.bfloat16), csd
        batches.n_cross = n_cross  # exposed for the periodic log

    # ── train loop ────────────────────────────────────────────────────────────
    print(f"training {total} steps (bucket {bucket}, B={B}, lr {lr}, warmup {warmup}) ...", flush=True)
    t0 = time.time(); step = 0
    for latent, text, csd in batches():
        if step >= total:
            break
        opt.learning_rate = sched(step)
        # null-style (CFG) dropout: zero the CSD for this batch with prob null_style_prob
        c = mx.zeros_like(csd) if random.random() < null_style_prob else csd
        t_int = _sample_t(latent.shape[0])
        noise = mx.random.normal(latent.shape)
        loss, gn = train_step(latent, text, c, t_int, noise)
        if step % 20 == 0 or step == total - 1:
            sps = (step + 1) / (time.time() - t0)
            extra = ""
            if not args.smoke:
                nc = getattr(batches, "n_cross", [0, 0])
                extra = f"  cross/self {nc[0]}/{nc[1]}"
            print(f"  step {step:6d}  loss {loss:.5f}  gnorm {gn:.3f}  lr {opt.learning_rate.item():.2e}"
                  f"  {sps:.2f} it/s{extra}", flush=True)
        if step > 0 and step % ckpt_every == 0:
            _save_ckpt(csd_mod, ckpt_dir, step)
        step += 1
    _save_ckpt(csd_mod, ckpt_dir, step)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
