"""train/lora/probe_joint_contrastive.py — SREF Stage-0.5 go/no-go probe.

Tests the ONE lever we have never been able to try (plans/sref-joint-backbone-project.md): does a
TRAINABLE backbone (LoRA) + a loss that CANNOT be satisfied by ignoring the reference break the
collapse that killed every frozen-backbone adapter?

THE OBJECTIVE (BACKLOG SREF-INFONCE-VOID / SREF-PAIR-VS-BANK — read before touching this file).
Forward the SAME noisy latent x_t TWICE: branch A on the correct reference `a` (a style-neighbour of
the target, never the target itself), branch B on a FOREIGN reference `b`. Each branch's output style
`z = proj(x0_pred)` must match the reference IT WAS HANDED.

    loss = L_recon(branch A)  +  λ·½·[ pair_row_ce(z_a, s_a, s_b) + pair_row_ce(z_b, s_b, s_a) ]

A reference-blind model emits z_a == z_b and pays ≥ ln 2 = 0.6931, ALWAYS (guard:
train/tests/test_pair_contrastive.py). Two superseded shapes that a reference-blind model beats,
and which must not creep back in:
  • contrasting each prediction against its OWN target  → a perfect ref-blind denoiser attains the
    minimum (acc 100%, flat in t). The reference was not even an argument of that loss.
  • a bank of negatives on the correct-ref branch alone → ref-blind scores 98.0% top-1 vs 1023
    negatives, because cos(CSD(target), CSD(its style-neighbour)) = 0.75. The foreign branch is the
    only term denoising cannot reach (cos(target, foreign) = 0.12).

WHY IT FITS AN M1. The two rows decouple (each touches only its own z plus frozen descriptors), so
they run as two SEQUENTIAL batch-1 forward/backward passes with summed grad trees: batch-1 memory,
2× compute. No co-resident batch — which is what wedged MLX's value_and_grad at batch 2/4.

t is sampled in [700, 950]: below t≈400 the input pins α²/(α²+σ²) of x0_pred (84.5% at t=300), so the
model cannot move style there even if it reads the reference (debug/sref_noise_band.py).

PRIMARY SIGNALS (logged every --diag-every steps):
  • foreign-row accuracy — EXACTLY 0.5 under collapse (the two rows' accuracies sum to 1). Rising
    off 0.5 = the reference is being used. Do NOT use a bank's top-1: it reads ~98% while collapsed.
  • cross-ref x0 correlation at fixed content — < 0.90 = collapse broken.
GO ⇒ render+scorecard gate. NO-GO ⇒ the loss-bound verdict is ironclad; stop.
NEVER train from cold storage (hot SSD only — CLAUDE.md).

  train/.venv/bin/python train/lora/probe_joint_contrastive.py --smoke --steps 40
  train/.venv/bin/python train/lora/probe_joint_contrastive.py --config train/configs/sref_joint_probe_v1.yaml
"""
import argparse, collections, math, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # train/

import yaml
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

from ip_adapter.model import CSDModulation
from ip_adapter.latent_csd import LatentCSDProjector, load_vae_bn_stats, bn_pack
from ip_adapter.loss import get_schedule_values, fused_flow_noise, reconstruct_x0, pair_row_ce
from lora.lora import inject_lora_double_blocks, inject_lora_single_blocks
from lora.train_step import patchify_pack, unpatchify, make_position_ids, make_ckpt_blocks
from lora.film_step import flux_forward_film

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Joint(nn.Module):
    """Container so nn.value_and_grad differentiates BOTH the LoRA (unfrozen inside flux) and the
    fully-trainable CSDModulation in one pass. The projector is NOT here: it is frozen, and gradient
    flows THROUGH it into x0_pred without updating it."""
    def __init__(self, flux, csd_mod):
        super().__init__()
        self.flux = flux
        self.csd_mod = csd_mod


def _l2(x, axis=-1, eps=1e-6):
    return x / mx.maximum(mx.linalg.norm(x, axis=axis, keepdims=True), eps)


def _fix_seq(t, L):
    """Pad/truncate text embeds to a FIXED length. Variable seq_txt → MLX rebuilds the (huge) graph
    every step and never progresses."""
    s = t.shape[1]
    if s == L:
        return t
    if s > L:
        return t[:, :L]
    return mx.concatenate([t, mx.zeros((t.shape[0], L - s, t.shape[2]), dtype=t.dtype)], axis=1)


def _forward_vpred(flux, csd_mod, noisy, text, csd, t_int, guidance, ckpt_d=None, ckpt_s=None):
    B, C, Lh, Lw = noisy.shape
    hidden = patchify_pack(noisy.astype(text.dtype))
    img_ids, txt_ids = make_position_ids(Lh // 2, Lw // 2, text.shape[1])
    seq = flux_forward_film(flux.transformer, hidden, text, csd_mod, csd, t_int,
                            img_ids, txt_ids, ckpt_d, ckpt_s, guidance)
    return unpatchify(seq, B, C, Lh, Lw)


class ForeignRefQueue:
    """Rolling bank of recently-seen CSD vectors, the SOURCE of branch B's foreign reference.

    This is the one correct use of a frozen-encoder bank: not as extra softmax negatives (those are
    easy and dilute the a-vs-b margin) but to supply a foreign reference at batch 1, so the pair
    never needs two co-resident samples. Picks the HARDEST safe negative: among candidates with
    cos(s_a, s_b) < max_cos (above that the "foreign" style may actually BE the target's style —
    label noise), take the most similar one."""

    def __init__(self, size=4096, pool=16, max_cos=0.5):
        self.q = collections.deque(maxlen=size)
        self.pool, self.max_cos = pool, max_cos

    def add(self, s):
        self.q.append(np.asarray(s, dtype=np.float32))

    def sample(self, s_a, rng):
        if len(self.q) < 2:
            return None
        idx = rng.choice(len(self.q), min(self.pool, len(self.q)), replace=False)
        cand = np.stack([self.q[i] for i in idx])
        cos = cand @ np.asarray(s_a, dtype=np.float32).ravel()
        ok = np.where(cos < self.max_cos)[0]
        if len(ok) == 0:
            return None
        return cand[ok[np.argmax(cos[ok])]], float(cos[ok].max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--diag-every", type=int, default=100)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config)) if args.config else {}
    mcfg, dcfg, tcfg = cfg.get("model", {}), cfg.get("data", {}), cfg.get("training", {})
    model_dir = os.path.abspath(mcfg.get("flux_model_dir", os.path.join(ROOT, "flux-klein-4b-base")))
    if args.smoke and not args.config:
        model_dir = os.path.join(ROOT, "flux-klein-model")
    rank = int(cfg.get("adapter", {}).get("lora_rank", 64))
    proj_ckpt = cfg.get("adapter", {}).get(
        "projector_ckpt", "/Volumes/2TBSSD/checkpoints/latent_csd/latent_csd_projector.safetensors")
    lr = float(tcfg.get("learning_rate", 1e-4))
    warmup = int(tcfg.get("warmup_steps", 50))
    total = args.steps or int(tcfg.get("num_steps", 8000))
    lambda_c = float(tcfg.get("contrastive_weight", 1.0))
    tau = float(tcfg.get("infonce_tau", 0.1))
    null_prob = float(tcfg.get("null_style_prob", 0.1))
    t_lo, t_hi = tcfg.get("t_range", [700, 950])
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    ckpt_every = int(tcfg.get("checkpoint_every", 1000))
    ckpt_dir = tcfg.get("checkpoint_dir", "/Volumes/2TBSSD/checkpoints/sref_joint_probe")
    bucket = tuple(dcfg.get("bucket", [256, 256]))
    text_seq = int(dcfg.get("text_seq", 128))
    guidance = tcfg.get("guidance")
    LN2 = math.log(2.0)

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

    proj = LatentCSDProjector()
    if os.path.exists(proj_ckpt):
        proj.load_weights(proj_ckpt)
        print(f"  latent->CSD projector loaded ({proj_ckpt})", flush=True)
    elif not args.smoke:
        sys.exit(f"missing projector checkpoint {proj_ckpt} — train it first "
                 f"(train/lora/train_latent_csd_projector.py)")
    proj.freeze()
    proj.eval()

    # Grad checkpointing: batch-1 backward peaks at 25.19 GB raw vs 15.72 GB checkpointed
    # (debug/sref_memory_ladder.py, 4B base @256px) for 21% slower steps — the 21.5 GB ceiling
    # needs it. The old wedge was checkpointing x a BATCHED graph; at batch 1 it compiles fine.
    if bool(tcfg.get("grad_checkpoint", True)):
        ckpt_d, ckpt_s = make_ckpt_blocks(flux)
        print("  grad checkpointing ON", flush=True)
    else:
        ckpt_d, ckpt_s = None, None

    bn_m, bn_s = load_vae_bn_stats(model_dir)

    def sched(step):
        if step < warmup:
            return lr * (step + 1) / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    opt = optim.AdamW(learning_rate=lr, weight_decay=float(tcfg.get("weight_decay", 0.01)))

    def _z(noisy, v_pred, alpha, sigma):
        return _l2(proj(reconstruct_x0(noisy, v_pred, alpha, sigma).astype(mx.float32)))

    # ── the two DECOUPLED rows. Each is a batch-1 forward/backward; grads are summed. ──────────
    def loss_a(J, noisy, text, s_a, s_b, t_int, alpha, sigma, v_target):
        v = _forward_vpred(J.flux, J.csd_mod, noisy, text, s_a, t_int, guidance, ckpt_d, ckpt_s)
        l_recon = mx.mean((v.astype(mx.float32) - v_target.astype(mx.float32)) ** 2)
        return l_recon + lambda_c * 0.5 * pair_row_ce(_z(noisy, v, alpha, sigma), s_a, s_b, tau)

    def loss_b(J, noisy, text, s_a, s_b, t_int, alpha, sigma):
        v = _forward_vpred(J.flux, J.csd_mod, noisy, text, s_b, t_int, guidance, ckpt_d, ckpt_s)
        return lambda_c * 0.5 * pair_row_ce(_z(noisy, v, alpha, sigma), s_b, s_a, tau)

    def loss_recon_only(J, noisy, text, csd, t_int, v_target):
        v = _forward_vpred(J.flux, J.csd_mod, noisy, text, csd, t_int, guidance, ckpt_d, ckpt_s)
        return mx.mean((v.astype(mx.float32) - v_target.astype(mx.float32)) ** 2)

    lg_a = nn.value_and_grad(joint, loss_a)
    lg_b = nn.value_and_grad(joint, loss_b)
    lg_r = nn.value_and_grad(joint, loss_recon_only)

    def train_step(latent, text, s_a, s_b, t_int, noise, drop_style):
        alpha, sigma = get_schedule_values(t_int)
        noisy, v_target = fused_flow_noise(latent, noise, alpha, sigma)
        if drop_style:
            # CFG dropout: no reference at all -> the contrastive is undefined; pure recon.
            l, g = lg_r(joint, noisy, text, mx.zeros_like(s_a), t_int, v_target)
            mx.eval(g)
            la, lb = float(l), 0.0
        else:
            l_a, g_a = lg_a(joint, noisy, text, s_a, s_b, t_int, alpha, sigma, v_target)
            mx.eval(g_a, l_a)                       # execute + free pass A's graph before pass B
            l_b, g_b = lg_b(joint, noisy, text, s_a, s_b, t_int, alpha, sigma)
            mx.eval(g_b, l_b)
            g = tree_map(lambda x, y: x + y, g_a, g_b)   # rows decouple -> gradients add
            la, lb = float(l_a), float(l_b)
        g, gn = optim.clip_grad_norm(g, max_norm=grad_clip)
        opt.update(joint, g)
        mx.eval(joint.trainable_parameters(), opt.state)
        return la, lb, float(gn)

    def diagnostics(latent, text, s_a, s_b):
        """Fixed CONTENT, two references, no gradients. Returns the three honest reads:
        (1) foreign-row accuracy — EXACTLY 0.5 under collapse (the rows' accuracies sum to 1);
        (2) cross-ref x0 correlation — < 0.90 = collapse broken;
        (3) the pair term itself — >= ln2 for ANY reference-blind model, so falling below ln2 is
            the unambiguous proof the reference is being used (test_pair_contrastive.py)."""
        t_int = mx.full((1,), 850, dtype=mx.int32)
        alpha, sigma = get_schedule_values(t_int)
        mx.random.seed(1234)
        noisy, _ = fused_flow_noise(latent, mx.random.normal(latent.shape), alpha, sigma)
        v_a = _forward_vpred(flux, csd_mod, noisy, text, s_a, t_int, guidance, ckpt_d, ckpt_s)
        v_b = _forward_vpred(flux, csd_mod, noisy, text, s_b, t_int, guidance, ckpt_d, ckpt_s)
        x0_a = reconstruct_x0(noisy, v_a, alpha, sigma).astype(mx.float32)
        x0_b = reconstruct_x0(noisy, v_b, alpha, sigma).astype(mx.float32)
        z_a, z_b = _l2(proj(x0_a)), _l2(proj(x0_b))
        acc_a = float(mx.sum(z_a * s_a) > mx.sum(z_a * s_b))
        acc_b = float(mx.sum(z_b * s_b) > mx.sum(z_b * s_a))
        pair = float(0.5 * (pair_row_ce(z_a, s_a, s_b, tau) + pair_row_ce(z_b, s_b, s_a, tau)))
        a, b = np.array(x0_a).ravel(), np.array(x0_b).ravel()
        a, b = a - a.mean(), b - b.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        corr = float((a * b).sum() / d) if d > 0 else 0.0
        return 0.5 * (acc_a + acc_b), corr, float(mx.sum(z_a * z_b)), pair

    # ── data. batch 1 CONTENT; the pair comes from two forwards, not two samples. ─────────────
    rng = np.random.default_rng(int(dcfg.get("seed") or 0))
    queue = ForeignRefQueue(size=int(tcfg.get("foreign_queue", 4096)),
                            max_cos=float(tcfg.get("foreign_max_cos", 0.5)))
    if args.smoke:
        print("SMOKE: synthetic data — verify the split-backward loop + pair loss + diagnostics", flush=True)
        Lh = Lw = bucket[0] // 8
        mx.random.seed(0)
        def batches():
            while True:
                lat = (mx.random.normal((1, 32, Lh, Lw)) * 0.5).astype(mx.bfloat16)
                txt = (mx.random.normal((1, text_seq, 7680)) * 0.1).astype(mx.bfloat16)
                yield lat, txt, _l2(mx.random.normal((1, 768))).astype(mx.float32)
    else:
        import glob as _glob
        from ip_adapter.dataset import make_prefetch_loader
        shard_paths = dcfg.get("shard_paths") or sorted(_glob.glob(os.path.join(dcfg["shard_path"], "*.tar")))
        if not shard_paths:
            sys.exit(f"no .tar shards under {dcfg.get('shard_path')}")
        print(f"  {len(shard_paths)} shard tars (hot); cond_mode=csd, batch 1 content + 2 forwards", flush=True)
        loader = make_prefetch_loader(
            shard_paths=shard_paths, batch_size=1,
            qwen3_cache_dir=dcfg.get("qwen3_cache_dir"), vae_cache_dir=dcfg.get("vae_cache_dir"),
            siglip_cache_dir=None, bucket=bucket,
            style_neighbors_db=dcfg.get("style_neighbors_db"),
            cond_mode="csd", csd_cache_dir=dcfg.get("csd_cache_dir"), seed=dcfg.get("seed"))
        def batches():
            skipped = 0
            for _img, _cap, text_np, vae_np, _csd_np, sref_np, _bhw, _ids in loader:
                # 100% CROSS-PAIRED. sref_np is a style-NEIGHBOUR's CSD; the old fallback to the
                # sample's own CSD made the reference redundant with the target being denoised
                # (~74% of samples) — precisely the shortcut that produces collapse.
                if vae_np is None or text_np is None or sref_np is None:
                    skipped += 1
                    continue
                latent = bn_pack(mx.array(vae_np, dtype=mx.bfloat16), bn_m, bn_s)
                s = mx.array(sref_np, dtype=mx.float32)
                if s.ndim == 3:
                    s = s.reshape(s.shape[0], -1)
                text = _fix_seq(mx.array(text_np, dtype=mx.bfloat16), text_seq)
                yield latent, text, _l2(s)
            print(f"  [data] exhausted (skipped {skipped} self-paired/missing)", flush=True)

    print(f"training {total} steps (bucket {bucket}, batch-1 x 2 passes, lr {lr}, λ_c {lambda_c}, "
          f"τ {tau}, t∈[{t_lo},{t_hi}]) ...", flush=True)
    print(f"  reference-blind floor on the pair term = λ·ln2 = {lambda_c*LN2:.4f}", flush=True)
    t0 = time.time(); step = 0; no_foreign = 0
    for latent, text, s_a in batches():
        if step >= total:
            break
        s_a_np = np.array(s_a).ravel()
        got = queue.sample(s_a_np, rng)
        queue.add(s_a_np)
        if got is None:                              # queue cold, or no safe foreign ref found
            no_foreign += 1
            continue
        s_b = mx.array(got[0], dtype=mx.float32)[None, :]

        opt.learning_rate = sched(step)
        tv = int(rng.integers(t_lo, t_hi + 1))
        t_int = mx.full((1,), tv, dtype=mx.int32)
        noise = mx.random.normal(latent.shape)
        drop = random.random() < null_prob
        l_a, l_b, gn = train_step(latent, text, s_a, s_b, t_int, noise, drop)

        if step % 20 == 0 or step == total - 1:
            sps = (step + 1) / (time.time() - t0)
            # NOTE: on a CFG-dropout step there is no reference, so the pair term is absent and
            # loss_b is exactly 0. Tag it — otherwise it reads as a floor violation.
            tag = "  [null]" if drop else ""
            print(f"  step {step:5d}  loss_a {l_a:.4f}  loss_b {l_b:.4f}  gnorm {gn:.3f}  "
                  f"lr {opt.learning_rate.item():.2e}  t {tv}  cos(a,b) {got[1]:.3f}  "
                  f"{sps:.3f} it/s  peak {mx.get_peak_memory()/2**30:.1f} GB{tag}", flush=True)
        if step > 0 and step % args.diag_every == 0:
            acc, corr, zcos, pair = diagnostics(latent, text, s_a, s_b)
            flag = "  <<< COLLAPSE BROKEN" if (corr < 0.90 and pair < LN2) else ""
            print(f"    [diag] foreign-row acc {acc:.3f} (0.5 = collapse)  cross-ref x0 corr {corr:.4f} "
                  f"(want < 0.90)  cos(z_a,z_b) {zcos:.4f}  pair {pair:.4f} "
                  f"(ref-blind floor {LN2:.4f}){flag}", flush=True)
        if step > 0 and step % ckpt_every == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
            mx.save_safetensors(os.path.join(ckpt_dir, f"joint_probe_{step:07d}.safetensors"),
                                dict(tree_flatten(joint.trainable_parameters())))
            print(f"  saved ckpt @ {step}", flush=True)
        step += 1
    print(f"DONE ({no_foreign} steps skipped for want of a foreign ref)", flush=True)


if __name__ == "__main__":
    main()
