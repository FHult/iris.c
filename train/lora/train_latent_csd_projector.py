"""Distil a latent→CSD projector from the cached (VAE latent, CSD) pairs.

Prerequisite for the corrected SREF contrastive (BACKLOG SREF-PAIR-VS-BANK): the content-shared-pair
loss compares `z = proj(x0_pred)` against the frozen `CSD(ref)`, so the output descriptor must live in
CSD space. Training data is already on disk — `precomputed/vae/v_2232c1` (raw latents) x `universe_csd`
(L2-normalised CSD, 200/shard x 1280 shards) — so this costs hours, not days, and never loads the 4B DiT.

The projector doubles as a cheap in-loop style metric (no VAE decode, no ViT).

  train/.venv/bin/python train/lora/train_latent_csd_projector.py --steps 20 --smoke
  train/.venv/bin/python train/lora/train_latent_csd_projector.py

Validation is the thing that matters, not the training cosine: the projector must PRESERVE the geometry
the pair loss depends on — cos(proj(target), CSD(style-neighbour)) ~ 0.75 vs cos(proj(target),
CSD(foreign)) ~ 0.12. If it flattens that gap, the contrastive has nothing to push on.
"""
import argparse, glob, json, math, os, random, sqlite3, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # train/

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ip_adapter.latent_csd import LatentCSDProjector, load_vae_bn_stats, bn_pack

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VAE_DIR = "/Volumes/2TBSSD/precomputed/vae/v_2232c1"
CSD_DIR = "/Volumes/2TBSSD/universe_csd"
EVAL_CSD_DIR = "/Volumes/2TBSSD/sref_eval/style_cache"      # full 5000/shard CSD + neighbors.sqlite
CKPT_DIR = "/Volumes/2TBSSD/checkpoints/latent_csd"
VAL_SHARD_MOD = 20                                           # shard % 20 == 19 -> held out


def _l2(v, axis=-1):
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), 1e-9)


def build_index(limit_shards=None):
    """rec_ids that have BOTH a VAE latent and a CSD vector. Split by SHARD (no style leakage).

    Only ~520 of the 1280 CSD shards have precomputed latents, and the covered shard ids are NOT
    contiguous -- so the split must hold out every Nth *populated* shard. Keying the split on the raw
    shard id instead silently yields an empty (or unrepresentative) val set."""
    have_lat = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(VAE_DIR, "*.npz"))}
    per_shard, csd = {}, {}
    shards = sorted(glob.glob(os.path.join(CSD_DIR, "*.npz")))
    if limit_shards:
        shards = shards[:limit_shards]
    for p in shards:
        d = np.load(p)
        ids = [k for k in d.files if k in have_lat]
        if not ids:
            continue
        for k in ids:
            csd[k] = d[k].astype(np.float32)
        per_shard[os.path.basename(p)[:-4]] = ids

    train, val = [], []
    for i, sid in enumerate(sorted(per_shard)):
        (val if i % VAL_SHARD_MOD == VAL_SHARD_MOD - 1 else train).extend(per_shard[sid])
    print(f"  {len(per_shard)} populated shards ({len(per_shard)//VAL_SHARD_MOD} held out)", flush=True)
    return train, val, csd


def load_latents(rec_ids, pool):
    def rd(r):
        return np.load(os.path.join(VAE_DIR, f"{r}.npz"))["latent"]
    return np.stack(list(pool.map(rd, rec_ids)))            # [B,32,64,64] float32


def augment(lat, rng, crop_prob=0.5, crop=32):
    """Random spatial crop with EVEN offsets. Style is spatially stationary, so a crop keeps the
    image's style -> this is a correct invariance, and it is what makes the projector work at the
    256px (32x32 latent) training resolution. Offsets MUST be even: BN packing is positional
    (feature = c*4 + (h%2)*2 + (w%2)), so an odd offset shifts the packing phase. For the same reason
    there is NO horizontal flip here -- it would invert the w-parity."""
    if rng.random() > crop_prob:
        return lat
    H = lat.shape[-1]
    oy = 2 * rng.integers(0, (H - crop) // 2 + 1)
    ox = 2 * rng.integers(0, (H - crop) // 2 + 1)
    return lat[:, :, oy:oy + crop, ox:ox + crop]


def evaluate(proj, val_ids, csd, bn_m, bn_s, pool, n=1024, bs=64):
    """Held-out cosine(proj(latent), CSD(same image)), at full res and at 32x32 crop."""
    ids = val_ids[:n]
    tot, totc, seen = 0.0, 0.0, 0
    for i in range(0, len(ids), bs):
        b = ids[i:i + bs]
        lat = mx.array(load_latents(b, pool))
        tgt = mx.array(_l2(np.stack([csd[r] for r in b])))
        packed = bn_pack(lat, bn_m, bn_s)
        z = proj(packed)
        zc = proj(packed[:, :, 16:48, 16:48])               # centre 32x32 crop (even offset)
        tot += float(mx.sum(mx.sum(z * tgt, axis=1)))
        totc += float(mx.sum(mx.sum(zc * tgt, axis=1)))
        seen += len(b)
    return tot / seen, totc / seen


def geometry_check(proj, bn_m, bn_s, pool, n=384, seed=0):
    """THE validation. Does the projector preserve the separation the pair loss needs?
    cos(proj(target), CSD(style-neighbour))  vs  cos(proj(target), CSD(foreign)).
    Ground truth (true CSD on both sides, BACKLOG SREF-PAIR-VS-BANK): 0.7526 vs 0.1205."""
    rng = np.random.default_rng(seed)
    tab = {}
    for p in sorted(glob.glob(os.path.join(EVAL_CSD_DIR, "*.npz"))):
        tab.update({k: v.astype(np.float32) for k, v in np.load(p).items()})
    have_lat = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(VAE_DIR, "*.npz"))}
    con = sqlite3.connect(os.path.join(EVAL_CSD_DIR, "neighbors.sqlite"))
    pairs = []
    for rec, nids, _ in con.execute("select rec_id, neighbor_ids, neighbor_cos from neighbors"):
        if rec not in tab or rec not in have_lat:
            continue
        ns = [x for x in json.loads(nids) if x in tab and x != rec]
        if ns:
            pairs.append((rec, ns[0]))
    if not pairs:
        return None
    rng.shuffle(pairs)
    pairs = pairs[:n]
    keys = list(tab)
    tgt_ids = [t for t, _ in pairs]
    A = _l2(np.stack([tab[a] for _, a in pairs]))                                   # correct ref CSD
    B = _l2(np.stack([tab[keys[i]] for i in rng.integers(0, len(keys), len(pairs))]))  # foreign ref CSD
    T_true = _l2(np.stack([tab[t] for t in tgt_ids]))                               # true CSD of target

    zs = []
    for i in range(0, len(tgt_ids), 64):
        lat = mx.array(load_latents(tgt_ids[i:i + 64], pool))
        zs.append(np.array(proj(bn_pack(lat, bn_m, bn_s))))
    Z = np.concatenate(zs)

    return {
        "proj_vs_neighbour": float(np.mean(np.sum(Z * A, 1))),
        "proj_vs_foreign":   float(np.mean(np.sum(Z * B, 1))),
        "true_vs_neighbour": float(np.mean(np.sum(T_true * A, 1))),
        "true_vs_foreign":   float(np.mean(np.sum(T_true * B, 1))),
        "proj_vs_true":      float(np.mean(np.sum(Z * T_true, 1))),
        "n": len(pairs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--smoke", action="store_true", help="40 shards (>=1 val shard), tiny run, no checkpoint")
    ap.add_argument("--model-dir", default=os.path.join(ROOT, "flux-klein-4b-base"))
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    random.seed(0); mx.random.seed(0)

    t0 = time.time()
    train_ids, val_ids, csd = build_index(limit_shards=40 if args.smoke else None)
    print(f"index: {len(train_ids)} train / {len(val_ids)} val pairs "
          f"({len(csd)} CSD vectors)  [{time.time()-t0:.1f}s]", flush=True)
    if not train_ids or not val_ids:
        sys.exit("empty index -- check VAE_DIR / CSD_DIR")

    bn_m, bn_s = load_vae_bn_stats(args.model_dir)
    proj = LatentCSDProjector(width=args.width)
    n_par = sum(v.size for _, v in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(
        proj.trainable_parameters()))
    print(f"projector: {n_par/1e6:.2f} M params", flush=True)

    def sched(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)

    def loss_fn(P, lat, tgt):
        return mx.mean(1.0 - mx.sum(P(lat) * tgt, axis=1))   # cosine distance; both L2-normalised

    lg = nn.value_and_grad(proj, loss_fn)

    pool = ThreadPoolExecutor(16)
    print(f"training {args.steps} steps, batch {args.batch_size} ...", flush=True)
    t0 = time.time()
    for step in range(args.steps):
        b = [train_ids[i] for i in rng.integers(0, len(train_ids), args.batch_size)]
        lat = augment(load_latents(b, pool), rng)
        lat = bn_pack(mx.array(lat), bn_m, bn_s)
        tgt = mx.array(_l2(np.stack([csd[r] for r in b])))
        opt.learning_rate = sched(step)
        loss, grads = lg(proj, lat, tgt)
        opt.update(proj, grads)
        mx.eval(proj.trainable_parameters(), opt.state, loss)   # never let the lazy graph accumulate
        if step % 50 == 0 or step == args.steps - 1:
            print(f"  step {step:5d}  cos_dist {float(loss):.4f}  "
                  f"lr {opt.learning_rate.item():.2e}  {(step+1)/(time.time()-t0):.2f} it/s", flush=True)
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            full, crop = evaluate(proj, val_ids, csd, bn_m, bn_s, pool)
            print(f"    [val] cos(proj,CSD) full {full:.4f}  32x32-crop {crop:.4f}", flush=True)

    if not args.smoke:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        out = os.path.join(args.ckpt_dir, "latent_csd_projector.safetensors")
        proj.save_weights(out)
        print(f"saved {out}", flush=True)

    g = geometry_check(proj, bn_m, bn_s, pool)
    if g:
        print("\n=== GEOMETRY CHECK (does proj preserve what the pair loss pushes on?) ===", flush=True)
        print(f"  n={g['n']}   cos(proj(target), proj-space vs true CSD of same image) = {g['proj_vs_true']:.4f}")
        print(f"  cos(proj(target), CSD(style-neighbour)) = {g['proj_vs_neighbour']:.4f}   "
              f"(true-CSD reference: {g['true_vs_neighbour']:.4f})")
        print(f"  cos(proj(target), CSD(foreign ref))     = {g['proj_vs_foreign']:.4f}   "
              f"(true-CSD reference: {g['true_vs_foreign']:.4f})")
        sep_p = g["proj_vs_neighbour"] - g["proj_vs_foreign"]
        sep_t = g["true_vs_neighbour"] - g["true_vs_foreign"]
        print(f"  separation: proj {sep_p:.4f}  vs true {sep_t:.4f}  "
              f"({sep_p/sep_t:.0%} retained)  -- the pair loss needs this gap", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
