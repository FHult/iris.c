"""Assemble a coherent single-style subset from the hot pool's CSD features → a record_allowlist.
Loads CSD (style descriptor) for the 22 baseline_pool_hot shards, finds the densest style cluster
(seed with the tightest top-N neighborhood), writes {"rec_ids":[...]} for the LoRA trainer."""
import glob, json, os
import numpy as np

HOT = "/Volumes/2TBSSD/baseline_pool_hot"
CSD = "/Volumes/2TBSSD/sref_eval/style_cache"
OUT = "/Volumes/2TBSSD/sref_eval/lora_style_subset.json"
N = 250                                  # cluster size (same-style images)

shards = [os.path.basename(p)[:-4] for p in sorted(glob.glob(f"{HOT}/*.tar"))]
ids, vecs = [], []
for s in shards:
    z = np.load(f"{CSD}/{s}.npz")
    for k in z.files:
        ids.append(k); vecs.append(z[k])
V = np.stack(vecs).astype(np.float32)
V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)        # L2-normalize → cosine = dot
print(f"loaded {len(ids)} CSD vectors from {len(shards)} hot shards", flush=True)

rng = np.random.default_rng(0)
cand = rng.choice(len(ids), size=400, replace=False)          # candidate seeds
sims = V[cand] @ V.T                                          # [400, M] cosine to all
# each candidate's top-N mean cosine (density of its style neighborhood)
part = np.partition(sims, -N, axis=1)[:, -N:]
dens = part.mean(axis=1)
best = cand[int(np.argmax(dens))]
sim = V[best] @ V.T
top = np.argsort(-sim)[:N]                                    # the N most stylistically-similar records
rec_ids = [ids[i] for i in top]
intra = float(sim[top].mean())
json.dump({"rec_ids": rec_ids, "seed": ids[best], "intra_cos": intra}, open(OUT, "w"))
print(f"seed {ids[best]}  cluster N={N}  intra-cluster mean cosine {intra:.3f}", flush=True)
print(f"  (shard spread: {len(set(r.split('_')[0] for r in rec_ids))} of {len(shards)} hot shards)", flush=True)
print(f"wrote {OUT}", flush=True)
