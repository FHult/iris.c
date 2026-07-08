"""train/lora/probe_style_lora.py — SREF Phase-1 → Stage-2 GO/NO-GO PROBE (cheap).

Stage 1 (projector-only, DiT FROZEN) was near-inert: in-sequence style tokens barely moved the output
(BACKLOG SREF-LEARNED-STAGE1; cross-ref output corr 0.98, styleCSD Δ ~0). Hypothesis: the FROZEN DiT has
no pathway to USE the tokens; a jointly-trained DiT LoRA (the plan's Stage 2) provides that pathway. This
probe tests that hypothesis CHEAPLY before committing to a ~2-day full run.

Design (decisive by construction):
  • Inject a small LoRA (rank R) into the DiT attention Linears (double+single) AND train the StyleProjector
    jointly — both trainable, DiT base frozen.
  • Overfit ~N real look-paired samples with EMPTY text: the only input that distinguishes the N targets is
    the SigLIP, so the model CAN reduce the loss ONLY by routing SigLIP→style-tokens→output. If the pathway
    can carry information at all (with a LoRA), the loss falls; if it's structurally inert, it can't.
  • CAUSAL SWAP TEST (the same test that flagged Stage-1 collapse): fix seed + empty text, vary ONLY the
    SigLIP across the N samples, decode. Stage-1 gave ~identical outputs (corr 0.98). If the LoRA makes the
    tokens causal, the outputs now DIVERGE (corr well below 0.98) and each resembles its own target's style
    more than the others' (CSD diagonal dominance).

VERDICT: GREENLIGHT full Stage 2 if the swap outputs diverge (mean off-diag corr materially < 0.98) with
CSD diagonal-dominance above chance; KILL (in-sequence+LoRA can't bind style either) if they stay inert.
This does NOT prove style DISENTANGLEMENT (empty-text overfit; SigLIP carries content too) — that's the
full run's job with real style-paired data. It proves only that the tokens can influence the output.

Run (train venv, GPU free): train/.venv/bin/python train/lora/probe_style_lora.py --config train/configs/sref_projector_v1.yaml
Uses CACHED latents only (live VAE-encode segfaults MLX — BUGS.md MLX-1).
"""
import argparse, os, sys, time, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import numpy as np
import yaml
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mflux.models.flux2 import Flux2Klein

from ip_adapter.model import StyleProjector
from ip_adapter.dataset import make_prefetch_loader
from lora.lora import inject_lora_double_blocks, inject_lora_single_blocks, lora_param_count
from lora.train_step import make_ckpt_blocks
from lora.style_step import style_loss
from train_ip_adapter import _encode_text, _TextEncoderBundle

# reuse the exact inference path the gate uses (parity)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "debug"))
from sref_infer_style import generate_style, _load_vae_bn_stats, _bn_unpack
from eval import _decode_latents

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Joint(nn.Module):
    """Container so nn.value_and_grad differentiates BOTH the LoRA (unfrozen inside flux) and the
    fully-trainable projector in one pass. trainable_parameters() = LoRA + projector."""
    def __init__(self, flux, projector):
        super().__init__()
        self.flux = flux
        self.projector = projector


def _bn_pack(lat, m, s):
    _, _, Lh, Lw = lat.shape
    return ((lat.astype(mx.float32) - mx.tile(m, (1, Lh // 2, Lw // 2)))
            / mx.tile(s, (1, Lh // 2, Lw // 2))).astype(lat.dtype)


def _sample_t(B):
    return mx.clip((mx.sigmoid(mx.random.normal((B,))) * 1000).astype(mx.int32), 0, 999)


def _corr(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=8, help="# samples to overfit")
    ap.add_argument("--two-cluster", action="store_true",
                    help="pick N samples as 2 stylistically-distinct clusters (CSD) so the 'predict the "
                         "average' escape is high-loss and SigLIP is REQUIRED — the unconfounded probe.")
    ap.add_argument("--pool", type=int, default=16, help="candidate pool to CSD-cluster (two-cluster mode)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--infer-steps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default="/tmp/sref_probe")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    mcfg, dcfg, tcfg = cfg["model"], cfg["data"], cfg.get("training", {})
    model_dir = os.path.abspath(os.path.join(ROOT, mcfg["flux_model_dir"]))
    guidance = tcfg.get("guidance")
    bucket = tuple(dcfg.get("bucket", [512, 512]))
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"loading Flux2Klein ({mcfg['flux_model_dir']}) + injecting LoRA rank {args.rank} ...", flush=True)
    t0 = time.time()
    flux = Flux2Klein(model_path=model_dir, quantize=None)
    inj = inject_lora_double_blocks(flux, rank=args.rank)
    inj += inject_lora_single_blocks(flux, rank=args.rank)
    projector = StyleProjector(hidden_dim=3072, num_heads=24, num_style_tokens=192, siglip_dim=1152)
    joint = _Joint(flux, projector)
    n_lora = lora_param_count(flux)
    n_proj = sum(v.size for _, v in tree_flatten(projector.trainable_parameters()))
    print(f"  loaded in {time.time()-t0:.1f}s; LoRA {n_lora/1e6:.1f}M ({len(inj)} sites) + projector "
          f"{n_proj/1e6:.1f}M trainable, DiT base frozen", flush=True)

    bn_m, bn_s = _load_vae_bn_stats(model_dir); print("  bn stats loaded", flush=True)
    text_enc = _TextEncoderBundle(flux.text_encoder, flux.tokenizers["qwen3"])
    # Fixed neutral prompt (SAME for all N) so SigLIP is the only cross-sample distinguisher — like an
    # empty prompt but without the zero-length-sequence risk.
    empty_text = _encode_text(text_enc, ["a photograph"]); mx.eval(empty_text)
    print("  fixed prompt encoded", flush=True)

    # ── grab N cached look-paired samples (vae latent + siglip) ──
    shard_paths = sorted(glob.glob(os.path.join(dcfg["shard_path"], "*.tar")))
    loader = make_prefetch_loader(
        shard_paths=shard_paths, batch_size=1, text_dropout_prob=0.0,
        qwen3_cache_dir=dcfg.get("qwen3_cache_dir"), vae_cache_dir=dcfg.get("vae_cache_dir"),
        siglip_cache_dir=dcfg.get("siglip_cache_dir"), bucket=bucket,
        style_neighbors_db=dcfg.get("style_neighbors_db"), cond_mode="siglip", seed=dcfg.get("seed"))
    print("  loader created; collecting samples ...", flush=True)
    n_collect = args.pool if args.two_cluster else args.n
    samples = []
    for images_np, captions, text_np, vae_np, siglip_np, style_ref_np, bhw, ids in loader:
        if vae_np is None or siglip_np is None:
            continue
        lat = _bn_pack(mx.array(vae_np, dtype=mx.bfloat16), bn_m, bn_s)
        sig = mx.array(siglip_np, dtype=mx.bfloat16)
        mx.eval(lat, sig)
        samples.append((lat, sig))
        if len(samples) >= n_collect:
            break

    labels = None
    if args.two_cluster:
        # CSD-cluster the pool into 2 stylistically-distinct groups of n//2, so their AVERAGE is a muddy
        # high-loss blend → the model must read SigLIP to tell the clusters apart (closes the average escape).
        from style_encoder.csd_mlx import CSDStyleEncoder, preprocess
        from PIL import Image
        enc = CSDStyleEncoder("/Volumes/2TBSSD/models/csd_vit_l_style.safetensors")
        vecs = []
        for lat, _ in samples:
            img = _decode_latents(flux.vae, _bn_unpack(lat, bn_m, bn_s).astype(mx.bfloat16))
            vecs.append(np.asarray(enc.encode(np.stack([preprocess(Image.fromarray(img).resize((512, 512)))])))[0])
            mx.clear_cache()
        V = np.stack(vecs)                                   # [pool, 768] L2-normalised
        D = 1.0 - V @ V.T                                    # cosine distance
        i0, j0 = np.unravel_index(np.argmax(D), D.shape)     # two farthest = cluster seeds
        half = args.n // 2
        a = list(np.argsort(D[i0])[:half])                   # nearest to seed A (incl i0)
        b = [k for k in np.argsort(D[j0]) if k not in a][:half]
        idx = a + b
        samples = [samples[k] for k in idx]
        labels = [0] * len(a) + [1] * len(b)
        sep = float(np.mean([1.0 - float(V[p] @ V[q]) for p in a for q in b]))   # mean cross-cluster CSD dist
        print(f"  2 CSD clusters of {half} (cross-cluster CSD dist {sep:.3f}; bigger=more distinct)", flush=True)
    print(f"  {len(samples)} samples cached (fixed-prompt overfit)", flush=True)

    ckpt_d, ckpt_s = make_ckpt_blocks(flux)
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.0)

    def loss_fn(mod, latent, sig, t_int, noise):
        return style_loss(mod.flux, mod.projector, latent, empty_text, sig, t_int, noise,
                          ckpt_d, ckpt_s, guidance)
    lg = nn.value_and_grad(joint, loss_fn)

    # ── overfit ──
    print(f"overfitting {len(samples)} samples for {args.steps} steps (lr {args.lr}) ...", flush=True)
    t0 = time.time(); first = None
    for step in range(args.steps):
        lat, sig = samples[step % len(samples)]
        t_int = _sample_t(1)
        noise = mx.random.normal(lat.shape)
        loss, grads = lg(joint, lat, sig, t_int, noise)
        grads, gn = optim.clip_grad_norm(grads, max_norm=1.0)
        opt.update(joint, grads)
        mx.eval(joint.trainable_parameters(), opt.state, loss)
        lv = float(loss)
        if first is None:
            first = lv
        if step % 25 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}  loss {lv:.5f}  gnorm {float(gn):.3f}  "
                  f"{(step+1)/(time.time()-t0):.2f} it/s", flush=True)
    last = lv
    print(f"overfit loss {first:.4f} -> {last:.4f}", flush=True)

    # ── CAUSAL SWAP TEST: fixed seed + empty text, vary only SigLIP ──
    print("causal swap test (fixed seed+text, vary SigLIP) ...", flush=True)
    size = bucket[0]
    outs, targets = [], []
    for i, (lat, sig) in enumerate(samples):
        o_lat = generate_style(flux, projector, empty_text, sig, size, size, args.infer_steps,
                               args.seed, guidance, bn_m, bn_s, null=False)
        o = _decode_latents(flux.vae, _bn_unpack(o_lat, bn_m, bn_s).astype(mx.bfloat16))
        t = _decode_latents(flux.vae, _bn_unpack(lat, bn_m, bn_s).astype(mx.bfloat16))
        from PIL import Image
        Image.fromarray(o).save(os.path.join(args.out_dir, f"out_{i}.png"))
        Image.fromarray(t).save(os.path.join(args.out_dir, f"target_{i}.png"))
        outs.append(o); targets.append(t)
        mx.clear_cache()

    # (1) do the swap outputs DIVERGE? (Stage-1 baseline: 0.98)
    n = len(outs)
    offdiag = [_corr(outs[i], outs[j]) for i in range(n) for j in range(i + 1, n)]
    offdiag = np.array(offdiag)
    # (2) CSD diagonal dominance: is out_i closest in STYLE to target_i?
    diag_wins, csd_note = None, ""
    try:
        sys.path.insert(0, os.path.join(ROOT, "train"))
        from style_encoder.csd_mlx import CSDStyleEncoder, preprocess
        from PIL import Image
        enc = CSDStyleEncoder("/Volumes/2TBSSD/models/csd_vit_l_style.safetensors")
        def csd_vec(arr):
            return np.asarray(enc.encode(np.stack([preprocess(Image.fromarray(arr).resize((512, 512)))])))[0]
        ov = [csd_vec(o) for o in outs]; tv = [csd_vec(t) for t in targets]
        S = np.array([[float(np.dot(ov[i], tv[j])) for j in range(n)] for i in range(n)])
        diag_wins = int(sum(int(np.argmax(S[i]) == i) for i in range(n)))
        csd_note = f"CSD diagonal-dominance {diag_wins}/{n} (chance {1.0:.1f}/{n}); mean diag {np.mean(np.diag(S)):.3f} vs mean off-diag {np.mean(S[~np.eye(n,dtype=bool)]):.3f}"
    except Exception as e:
        csd_note = f"(CSD dominance skipped: {e})"

    print("\n=== STAGE-2 PROBE VERDICT ===")
    print(f"overfit loss:           {first:.4f} -> {last:.4f}")
    print(f"swap output corr:       mean off-diag {offdiag.mean():.4f}  max {offdiag.max():.4f}  "
          f"(Stage-1 frozen-DiT baseline: ~0.98)")
    print(f"style binding:          {csd_note}")

    if labels is not None:
        # 2-cluster decisive signal: if SigLIP routes style, swap outputs CLUSTER by style — within-cluster
        # pairs more similar than cross-cluster pairs. Inert ⇒ within ≈ cross ≈ ~0.99.
        within = [_corr(outs[i], outs[j]) for i in range(n) for j in range(i + 1, n) if labels[i] == labels[j]]
        cross = [_corr(outs[i], outs[j]) for i in range(n) for j in range(i + 1, n) if labels[i] != labels[j]]
        wmean, cmean = float(np.mean(within)), float(np.mean(cross))
        gap = wmean - cmean
        print(f"cluster routing:        within-cluster corr {wmean:.4f}  cross-cluster corr {cmean:.4f}  "
              f"gap {gap:+.4f}  (gap > ~0.02 = SigLIP routes style)")
        if gap > 0.02 and cmean < 0.97:
            verdict = ("GREENLIGHT — swap outputs CLUSTER BY STYLE (within>cross); the LoRA makes in-sequence "
                       "tokens route style. Run full Stage 2.")
        elif gap > 0.005:
            verdict = ("PARTIAL — weak but present clustering (gap {:+.4f}); Stage 2 plausible but marginal. "
                       "Consider more steps/rank before a full commit.".format(gap))
        else:
            verdict = ("KILL — no style routing even with the average-escape closed (within≈cross≈inert). "
                       "In-sequence+LoRA does not bind style on this stack. Fall back to retrieval-hybrid.")
    else:
        diverged = offdiag.mean() < 0.90
        bind = (diag_wins is not None and diag_wins > max(1, n // 4))
        if diverged and bind:
            verdict = "GREENLIGHT — the LoRA makes in-sequence style tokens CAUSAL and style-binding; run full Stage 2."
        elif diverged:
            verdict = "PARTIAL — outputs diverge (tokens are causal) but style-binding is weak; Stage 2 plausible, watch disentanglement."
        else:
            verdict = "KILL — outputs stay inert even with a LoRA (corr ~Stage-1); in-sequence+LoRA does not bind. Fall back to retrieval-hybrid."
    print(f"VERDICT: {verdict}")
    print(f"renders in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
