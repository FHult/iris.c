"""train/lora/probe_joint_contrastive.py — SREF Stage-0.5 go/no-go probe.

Tests the ONE lever we have never been able to try (plans/sref-joint-backbone-project.md §2): does a
TRAINABLE backbone (LoRA) + an IN-BATCH CONTRASTIVE loss (needs batch>1) break the reference-collapse
that killed every frozen-backbone adapter (SREF-FILM-1: the collapse is loss-bound, not channel-bound)?

Cheap + local: 256px (¼ the tokens of 512px) so batch 4 + LoRA fits on the M1 32GB. Joint-trains a LoRA
(r64, double+single attn) AND the CSDModulation (CSD→temb FiLM) via one `_Joint` value_and_grad. Loss =
flow-recon + λ·InfoNCE over the batch (each prediction's AdaIN style-stats must match its OWN reference's
target more than the other batch members' → a constant/collapsed output cannot classify → collapse punished).

PRIMARY SIGNAL (logged every --diag-every steps): fixed-content cross-ref x0 correlation — forward the SAME
latent+noise+t with each of the batch's B distinct CSD refs; if the outputs decorrelate (corr < ~0.9) the
reference finally matters. GO ⇒ proceed to the render+scorecard gate. NO-GO (stays ≥0.9) ⇒ the loss-bound
verdict is ironclad; stop. NEVER train from cold storage (hot SSD only — CLAUDE.md).

  train/.venv/bin/python train/lora/probe_joint_contrastive.py --smoke --steps 40
  train/.venv/bin/python train/lora/probe_joint_contrastive.py --config train/configs/sref_joint_probe_v1.yaml
"""
import argparse, os, sys, time, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # train/

import yaml
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

from ip_adapter.model import CSDModulation
from ip_adapter.loss import get_schedule_values, fused_flow_noise, reconstruct_x0, style_stats
from lora.lora import inject_lora_double_blocks, inject_lora_single_blocks
from lora.train_step import make_ckpt_blocks, patchify_pack, unpatchify, make_position_ids
from lora.film_step import flux_forward_film

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Joint(nn.Module):
    """Container so nn.value_and_grad differentiates BOTH the LoRA (unfrozen inside flux) and the
    fully-trainable CSDModulation in one pass. trainable_parameters() = LoRA + csd_mod."""
    def __init__(self, flux, csd_mod):
        super().__init__()
        self.flux = flux
        self.csd_mod = csd_mod


def _load_vae_bn_stats(model_dir):
    st = mx.load(os.path.join(model_dir, "vae", "diffusion_pytorch_model.safetensors"))
    m = st["bn.running_mean"].astype(mx.float32).reshape(32, 2, 2)
    s = mx.sqrt(st["bn.running_var"].astype(mx.float32).reshape(32, 2, 2) + 1e-4)
    mx.eval(m, s); return m, s


def _bn_pack(lat, m, s):
    _, _, Lh, Lw = lat.shape
    return ((lat.astype(mx.float32) - mx.tile(m, (1, Lh // 2, Lw // 2)))
            / mx.tile(s, (1, Lh // 2, Lw // 2))).astype(lat.dtype)


def _sample_t_highbias(bias=1.0):
    """One shared timestep, biased toward HIGH noise (where global style is set). sigmoid(N(bias,1))."""
    u = float(mx.sigmoid(mx.random.normal((1,)) + bias).item())
    return max(1, min(999, int(u * 1000)))


def _l2(x, axis=-1, eps=1e-6):
    return x / mx.maximum(mx.linalg.norm(x, axis=axis, keepdims=True), eps)


def _fix_seq(t, L):
    """Pad/truncate text embeds to a FIXED length L. Real captions have variable token counts; the
    loader pads to the per-BATCH max, so seq_txt changes every batch → MLX recompiles the whole
    (huge) graph each step and never progresses. A fixed shape → compile once, then step."""
    s = t.shape[1]
    if s == L:
        return t
    if s > L:
        return t[:, :L]
    return mx.concatenate([t, mx.zeros((t.shape[0], L - s, t.shape[2]), dtype=t.dtype)], axis=1)


def _forward_vpred(flux, csd_mod, noisy, text, csd, t_int, ckpt_d, ckpt_s, guidance):
    B, C, Lh, Lw = noisy.shape
    hidden = patchify_pack(noisy.astype(text.dtype))
    img_ids, txt_ids = make_position_ids(Lh // 2, Lw // 2, text.shape[1])
    seq = flux_forward_film(flux.transformer, hidden, text, csd_mod, csd, t_int,
                            img_ids, txt_ids, ckpt_d, ckpt_s, guidance)
    return unpatchify(seq, B, C, Lh, Lw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--diag-every", type=int, default=40)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config)) if args.config else {}
    mcfg, dcfg, tcfg = cfg.get("model", {}), cfg.get("data", {}), cfg.get("training", {})
    model_dir = os.path.abspath(mcfg.get("flux_model_dir", os.path.join(ROOT, "flux-klein-4b-base")))
    if args.smoke and not args.config:
        model_dir = os.path.join(ROOT, "flux-klein-model")
    rank = int(cfg.get("adapter", {}).get("lora_rank", 64))
    lr = float(tcfg.get("learning_rate", 1e-4))
    warmup = int(tcfg.get("warmup_steps", 50))
    total = args.steps or int(tcfg.get("num_steps", 8000))
    lambda_c = float(tcfg.get("contrastive_weight", 1.0))
    tau = float(tcfg.get("infonce_tau", 0.1))
    null_prob = float(tcfg.get("null_style_prob", 0.1))
    t_bias = float(tcfg.get("highnoise_bias", 1.0))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    ckpt_every = int(tcfg.get("checkpoint_every", 1000))
    ckpt_dir = tcfg.get("checkpoint_dir", "/Volumes/2TBSSD/checkpoints/sref_joint_probe")
    B = int(dcfg.get("batch_size", 4))
    bucket = tuple(dcfg.get("bucket", [256, 256]))
    text_seq = int(dcfg.get("text_seq", 128))     # FIXED text length → constant graph shape (no per-batch recompile)
    guidance = tcfg.get("guidance")

    print(f"loading Flux2Klein ({model_dir}) ...", flush=True)
    t0 = time.time()
    flux = Flux2Klein(model_path=model_dir, quantize=None)
    inj = inject_lora_double_blocks(flux, rank=rank)
    inj += inject_lora_single_blocks(flux, rank=rank)
    csd_mod = CSDModulation(hidden_dim=3072, csd_dim=768, mlp_dim=1024)
    joint = _Joint(flux, csd_mod)
    n = sum(v.size for _, v in tree_flatten(joint.trainable_parameters()))
    print(f"  loaded + injected {len(inj)} LoRA modules (r{rank}); {n/1e6:.2f} M trainable "
          f"(LoRA + CSDModulation); {time.time()-t0:.1f}s", flush=True)

    # 256px activations are small → grad checkpointing (needed at 512px) usually just costs a 2nd
    # forward per step. Default OFF here; flip training.grad_checkpoint if batch 4 OOMs.
    if bool(tcfg.get("grad_checkpoint", False)):
        ckpt_d, ckpt_s = make_ckpt_blocks(flux)
        print("  grad checkpointing ON", flush=True)
    else:
        ckpt_d, ckpt_s = None, None
    bn_m, bn_s = _load_vae_bn_stats(model_dir)

    def sched(step):
        if step < warmup:
            return lr * (step + 1) / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    opt = optim.AdamW(learning_rate=lr, weight_decay=float(tcfg.get("weight_decay", 0.01)))

    def loss_fn(J, latent, text, csd, t_int, noise):
        alpha, sigma = get_schedule_values(t_int)
        noisy, v_target = fused_flow_noise(latent, noise, alpha, sigma)
        v_pred = _forward_vpred(J.flux, J.csd_mod, noisy, text, csd, t_int, ckpt_d, ckpt_s, guidance)
        l_recon = mx.mean((v_pred.astype(mx.float32) - v_target.astype(mx.float32)) ** 2)
        x0 = reconstruct_x0(noisy, v_pred, alpha, sigma)
        s_pred = _l2(style_stats(x0))                       # [B, 2C]
        s_tgt = _l2(style_stats(latent.astype(mx.float32)))  # [B, 2C]
        logits = (s_pred @ s_tgt.T) / tau                    # [B, B]
        labels = mx.arange(latent.shape[0])
        l_con = mx.mean(nn.losses.cross_entropy(logits, labels))
        return l_recon + lambda_c * l_con

    lg = nn.value_and_grad(joint, loss_fn)

    def train_step(latent, text, csd, t_int, noise):
        loss, grads = lg(joint, latent, text, csd, t_int, noise)
        grads, gn = optim.clip_grad_norm(grads, max_norm=grad_clip)
        opt.update(joint, grads)
        mx.eval(joint.trainable_parameters(), opt.state, loss)
        return float(loss), float(gn)

    def collapse_metric(latent, text, csd):
        """Fixed CONTENT (sample 0), vary the reference across the batch's B CSDs → cross-ref x0 corr.
        Low (<~0.9) = the reference finally changes the output (collapse broken)."""
        Bn = latent.shape[0]
        t_int = mx.full((Bn,), 600, dtype=mx.int32)
        alpha, sigma = get_schedule_values(t_int)
        l0 = mx.broadcast_to(latent[0:1], latent.shape)
        mx.random.seed(1234)
        noise0 = mx.broadcast_to(mx.random.normal(latent[0:1].shape), latent.shape)
        t0b = mx.broadcast_to(text[0:1], text.shape)
        noisy, _ = fused_flow_noise(l0, noise0, alpha, sigma)
        v = _forward_vpred(flux, csd_mod, noisy, t0b, csd, t_int, ckpt_d, ckpt_s, guidance)
        x0 = np.array(reconstruct_x0(noisy, v, alpha, sigma).astype(mx.float32))
        cc = []
        for i in range(Bn):
            for j in range(i + 1, Bn):
                a, b = x0[i].ravel(), x0[j].ravel()
                a, b = a - a.mean(), b - b.mean()
                d = np.sqrt((a * a).sum() * (b * b).sum())
                cc.append(float((a * b).sum() / d) if d > 0 else 0.0)
        return float(np.mean(cc)), float(np.max(cc))

    # ── data ──────────────────────────────────────────────────────────────────
    if args.smoke:
        print("SMOKE: synthetic data — verify the joint loop + contrastive + collapse metric", flush=True)
        Lh = Lw = bucket[0] // 8
        mx.random.seed(0)
        def batches():
            while True:
                lat = (mx.random.normal((B, 32, Lh, Lw)) * 0.5).astype(mx.bfloat16)
                txt = (mx.random.normal((B, 48, 7680)) * 0.1).astype(mx.bfloat16)
                c = _l2(mx.random.normal((B, 768))).astype(mx.bfloat16)
                yield lat, txt, c
    else:
        import glob as _glob
        from ip_adapter.dataset import make_prefetch_loader
        shard_paths = dcfg.get("shard_paths") or sorted(_glob.glob(os.path.join(dcfg["shard_path"], "*.tar")))
        if not shard_paths:
            sys.exit(f"no .tar shards under {dcfg.get('shard_path')}")
        print(f"  {len(shard_paths)} shard tars (hot); cond_mode=csd, B={B}", flush=True)
        loader = make_prefetch_loader(
            shard_paths=shard_paths, batch_size=B,
            qwen3_cache_dir=dcfg.get("qwen3_cache_dir"), vae_cache_dir=dcfg.get("vae_cache_dir"),
            siglip_cache_dir=None, bucket=bucket,
            style_neighbors_db=dcfg.get("style_neighbors_db"),
            cond_mode="csd", csd_cache_dir=dcfg.get("csd_cache_dir"), seed=dcfg.get("seed"))
        def batches():
            for _img, _cap, text_np, vae_np, csd_np, sref_np, _bhw, _ids in loader:
                if vae_np is None or text_np is None or csd_np is None:
                    continue
                latent = _bn_pack(mx.array(vae_np, dtype=mx.bfloat16), bn_m, bn_s)
                csd = mx.array(sref_np if sref_np is not None else csd_np, dtype=mx.bfloat16)
                if csd.ndim == 3:
                    csd = csd.reshape(csd.shape[0], -1)
                text = _fix_seq(mx.array(text_np, dtype=mx.bfloat16), text_seq)
                yield latent, text, csd

    # ── train loop ──────────────────────────────────────────────────────────────
    print(f"training {total} steps (bucket {bucket}, B={B}, lr {lr}, λ_con {lambda_c}, τ {tau}, "
          f"high-noise bias {t_bias}) ...", flush=True)
    t0 = time.time(); step = 0
    for latent, text, csd in batches():
        if step >= total:
            break
        if latent.shape[0] < 2:
            continue  # contrastive needs ≥2
        opt.learning_rate = sched(step)
        c = mx.zeros_like(csd) if random.random() < null_prob else csd
        tv = _sample_t_highbias(t_bias)
        t_int = mx.full((latent.shape[0],), tv, dtype=mx.int32)
        noise = mx.random.normal(latent.shape)
        loss, gn = train_step(latent, text, c, t_int, noise)
        if step % 20 == 0 or step == total - 1:
            sps = (step + 1) / (time.time() - t0)
            print(f"  step {step:5d}  loss {loss:.4f}  gnorm {gn:.3f}  lr {opt.learning_rate.item():.2e}"
                  f"  t {tv}  {sps:.2f} it/s", flush=True)
        if step > 0 and step % args.diag_every == 0:
            cm_mean, cm_max = collapse_metric(latent, text, csd)
            flag = "  <<< COLLAPSE BROKEN" if cm_max < 0.90 else ""
            print(f"    [collapse] cross-ref x0 corr  mean {cm_mean:.4f}  max {cm_max:.4f}"
                  f"  (want < 0.90){flag}", flush=True)
        if step > 0 and step % ckpt_every == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
            flat = dict(tree_flatten(joint.trainable_parameters()))
            mx.save_safetensors(os.path.join(ckpt_dir, f"joint_probe_{step:07d}.safetensors"),
                                {k: v for k, v in flat.items()})
            print(f"  saved ckpt @ {step}", flush=True)
        step += 1
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
