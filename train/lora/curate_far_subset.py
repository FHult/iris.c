"""Curate a coherent style cluster that is FAR from the base model's CSD prior.

Unlike curate_style_subset.py (densest cluster anywhere — which landed ON the photographic prior),
this measures the base model's own style signature (CSD-encode N base-model generations → prior
centroid) and picks the densest pool cluster among those FARTHEST from that prior. Output = the same
{"rec_ids":[...]} allowlist the LoRA trainer consumes.

  train/.venv/bin/python curate_far_subset.py --prior-dir <base_imgs> --out subset.json
"""
import argparse, glob, json, os, sys
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # train/ (for style_encoder)
from style_encoder.csd_mlx import CSDStyleEncoder, preprocess

HOT = "/Volumes/2TBSSD/baseline_pool_hot"
CSD = "/Volumes/2TBSSD/sref_eval/style_cache"
WEIGHTS = "/Volumes/2TBSSD/models/csd_vit_l_style.safetensors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior-dir", required=True, help="dir of base-model PNGs (the prior)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=250, help="cluster size")
    ap.add_argument("--far-frac", type=float, default=0.20, help="seeds from the farthest this fraction")
    args = ap.parse_args()

    # --- base prior centroid: CSD-encode the base generations (same encoder as the cache) ---
    enc = CSDStyleEncoder(WEIGHTS)
    pimgs = sorted(glob.glob(os.path.join(args.prior_dir, "*.png")))
    if not pimgs:
        raise RuntimeError(f"no PNGs in {args.prior_dir}")
    px = np.stack([preprocess(Image.open(p).convert("RGB")) for p in pimgs])
    pf = np.asarray(enc.encode(px)).astype(np.float32)          # [P,768] L2-normalised
    prior = pf.mean(0); prior /= (np.linalg.norm(prior) + 1e-8)  # centroid, renormalised
    print(f"prior: {len(pimgs)} base imgs; intra-prior mean cos {float((pf @ prior).mean()):.3f}", flush=True)

    # --- pool CSD vectors (same shards curate_style_subset used) ---
    shards = [os.path.basename(p)[:-4] for p in sorted(glob.glob(f"{HOT}/*.tar"))]
    ids, vecs = [], []
    for s in shards:
        z = np.load(f"{CSD}/{s}.npz")
        for k in z.files:
            ids.append(k); vecs.append(z[k])
    V = np.stack(vecs).astype(np.float32)
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    prior_cos = V @ prior                                        # [M] cosine to base prior; low = far
    print(f"loaded {len(ids)} pool CSD vecs; prior_cos min {prior_cos.min():.3f} "
          f"median {np.median(prior_cos):.3f} max {prior_cos.max():.3f}", flush=True)

    # --- candidates: broad coverage (random pool + the farthest seeds). For each, build its N-cluster
    #     and score = coherence − λ·cluster_prior_cos, subject to a coherence FLOOR. This maximises
    #     distance-from-prior while still demanding a real, tight style. ---
    N, FLOOR, LAM = args.n, 0.60, 2.0
    order = np.argsort(prior_cos)                                # ascending → farthest first
    rng = np.random.default_rng(0)
    far_seeds = order[:max(N, int(len(ids) * args.far_frac))]
    cand = np.unique(np.concatenate([
        far_seeds[rng.choice(len(far_seeds), size=min(400, len(far_seeds)), replace=False)],
        rng.choice(len(ids), size=600, replace=False)]))
    sims = V[cand] @ V.T                                         # [C, M]
    topk = np.argpartition(-sims, N, axis=1)[:, :N]              # top-N neighbour indices per candidate
    coh = np.take_along_axis(sims, topk, axis=1).mean(axis=1)    # cluster coherence (density)
    cpc = prior_cos[topk].mean(axis=1)                           # cluster mean cosine to prior
    score = np.where(coh >= FLOOR, coh - LAM * cpc, -1e9)        # far + coherent
    bi = int(np.argmax(score))
    best = cand[bi]
    sim = V[best] @ V.T
    top = np.argsort(-sim)[:N]
    rec_ids = [ids[i] for i in top]
    intra = float(sim[top].mean()); cluster_prior_cos = float(prior_cos[top].mean())

    json.dump({"rec_ids": rec_ids, "seed": ids[best], "intra_cos": intra,
               "cluster_prior_cos": cluster_prior_cos}, open(args.out, "w"))
    print(f"SELECTED seed {ids[best]}  N={N}  (coherence floor {FLOOR}, λ={LAM})", flush=True)
    print(f"  intra-cluster cos {intra:.3f}  (coherence)", flush=True)
    print(f"  cluster prior_cos {cluster_prior_cos:+.3f}  vs pool median {np.median(prior_cos):+.3f}", flush=True)
    # compare to the previous (densest-anywhere = photographic) cluster, if present
    prev = "/Volumes/2TBSSD/sref_eval/lora_style_subset.json"
    if os.path.exists(prev):
        pids = set(json.load(open(prev))["rec_ids"])
        idx = [i for i, k in enumerate(ids) if k in pids]
        if idx:
            print(f"  (prev photographic cluster prior_cos {float(prior_cos[idx].mean()):+.3f})", flush=True)
    print(f"  shard spread: {len(set(r.split('_')[0] for r in rec_ids))} of {len(shards)} shards", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
