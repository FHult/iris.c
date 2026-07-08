"""train/lora/curate_style_seeded.py — SEEDED style-cluster curation for the retrieval-hybrid
(plans/sref-retrieval-hybrid-project.md, Phase 0).

curate_style_subset.py finds the DENSEST cluster overall (likely generic photos). For a targeted
per-style LoRA we want a cluster of a SPECIFIC style, so seed from an exemplar image: compute its CSD
style vector, then take the N hot-pool records with the highest CSD cosine → {"rec_ids":[...]} for
train_lora.py's record_allowlist. Same output format + CSD cache as curate_style_subset.py.

  train/.venv/bin/python train/lora/curate_style_seeded.py \
      --seed-image /Volumes/2TBSSD/sref_eval/refs/impressionism_landscape.jpg \
      --out /Volumes/2TBSSD/sref_eval/lora_painterly_subset.json --n 250
"""
import argparse, glob, json, os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from style_encoder.csd_mlx import CSDStyleEncoder, preprocess

HOT = "/Volumes/2TBSSD/baseline_pool_hot"
CSD_CACHE = "/Volumes/2TBSSD/sref_eval/style_cache"
CSD_WEIGHTS = "/Volumes/2TBSSD/models/csd_vit_l_style.safetensors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-image", required=True, help="style exemplar to cluster around")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=250, help="cluster size")
    args = ap.parse_args()

    # seed CSD vector
    enc = CSDStyleEncoder(CSD_WEIGHTS)
    img = Image.open(args.seed_image).convert("RGB")
    seed_vec = np.asarray(enc.encode(np.stack([preprocess(img)])))[0].astype(np.float32)
    seed_vec /= (np.linalg.norm(seed_vec) + 1e-8)

    # hot-pool CSD vectors (npz keys = rec_ids)
    shards = [os.path.basename(p)[:-4] for p in sorted(glob.glob(f"{HOT}/*.tar"))]
    ids, vecs = [], []
    for s in shards:
        z = np.load(f"{CSD_CACHE}/{s}.npz")
        for k in z.files:
            ids.append(k); vecs.append(z[k])
    V = np.stack(vecs).astype(np.float32)
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    print(f"loaded {len(ids)} CSD vectors from {len(shards)} hot shards", flush=True)

    sim = V @ seed_vec                                   # cosine to the seed
    top = np.argsort(-sim)[:args.n]
    rec_ids = [ids[i] for i in top]
    intra = float(sim[top].mean())
    json.dump({"rec_ids": rec_ids, "seed_image": args.seed_image, "seed_cos_mean": intra}, open(args.out, "w"))
    print(f"seed {os.path.basename(args.seed_image)}  cluster N={len(rec_ids)}  "
          f"mean cosine-to-seed {intra:.3f}  (min {float(sim[top].min()):.3f})", flush=True)
    print(f"  shard spread: {len(set(r.split('_')[0] for r in rec_ids))} of {len(shards)} hot shards", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
