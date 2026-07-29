"""train/lora/cluster_hot_styles.py — discover distinct dense STYLE clusters in the hot pool (CSD).

Retrieval-hybrid prove-out (plans/sref-retrieval-hybrid-project.md): the 22-shard hot pool is mostly
journeydb and already has full VAE/Qwen3/CSD precompute, so we can find 2-3 distinct, coherent style
clusters and train a per-cluster LoRA each with NO staging. K-means over the hot CSD cache; rank clusters
by tightness (intra-cluster cosine) and mutual separation; emit per-cluster rec_ids (for train_lora's
record_allowlist) + centroids + exemplar rec_ids (nearest the centroid, to eyeball the style).

  train/.venv/bin/python train/lora/cluster_hot_styles.py --k 24 --pick 3 --out /Volumes/2TBSSD/sref_eval/hot_style_clusters
"""
import argparse, glob, json, os, sys
import numpy as np

CSD_CACHE = "/Volumes/2TBSSD/sref_eval/style_cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=24, help="kmeans clusters to discover")
    ap.add_argument("--pick", type=int, default=3, help="how many tight+separated clusters to emit")
    ap.add_argument("--cluster-size", type=int, default=250, help="rec_ids per emitted LoRA subset")
    ap.add_argument("--out", default="/Volumes/2TBSSD/sref_eval/hot_style_clusters")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = args.out; os.makedirs(out, exist_ok=True)

    ids, vecs = [], []
    for f in sorted(glob.glob(f"{CSD_CACHE}/*.npz")):
        z = np.load(f)
        for k in z.files:
            ids.append(k); vecs.append(z[k])
    ids = np.array(ids)
    V = np.stack(vecs).astype(np.float32)
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    print(f"loaded {len(ids)} CSD vectors", flush=True)

    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=args.k, random_state=args.seed, batch_size=4096, n_init=5)
    lab = km.fit_predict(V)
    C = km.cluster_centers_
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)

    # per-cluster tightness = mean cosine of members to their (normalised) centroid
    rows = []
    for c in range(args.k):
        m = np.where(lab == c)[0]
        if len(m) < args.cluster_size:
            continue
        sim = V[m] @ C[c]
        rows.append({"cluster": c, "size": int(len(m)), "intra_cos": float(sim.mean()),
                     "members": m, "sim": sim})
    rows.sort(key=lambda r: r["intra_cos"], reverse=True)

    # greedily pick tight clusters that are mutually separated (centroid cos < 0.55)
    picked = []
    for r in rows:
        if all(float(C[r["cluster"]] @ C[p["cluster"]]) < 0.55 for p in picked):
            picked.append(r)
        if len(picked) >= args.pick:
            break

    print(f"\n{'cluster':>7}{'size':>7}{'intra_cos':>11}   exemplars (nearest centroid)")
    manifest = {"k": args.k, "picked": []}
    for r in picked:
        c = r["cluster"]; m = r["members"]; sim = r["sim"]
        order = m[np.argsort(-sim)]
        top = order[:args.cluster_size]
        exemplars = [ids[i] for i in order[:4]]
        name = f"cluster{c:02d}"
        rec_ids = [ids[i] for i in top]
        json.dump({"rec_ids": rec_ids, "cluster": int(c), "intra_cos": r["intra_cos"]},
                  open(f"{out}/{name}_subset.json", "w"))
        manifest["picked"].append({"name": name, "cluster": int(c), "size": r["size"],
                                   "intra_cos": r["intra_cos"], "centroid": C[c].tolist(),
                                   "exemplars": exemplars, "subset": f"{out}/{name}_subset.json"})
        print(f"{c:>7}{r['size']:>7}{r['intra_cos']:>11.3f}   {exemplars}")
    # pairwise separation of picked
    if len(picked) > 1:
        print("\npicked centroid cosine (separation; lower=more distinct):")
        for i in range(len(picked)):
            for j in range(i + 1, len(picked)):
                cc = float(C[picked[i]['cluster']] @ C[picked[j]['cluster']])
                print(f"  c{picked[i]['cluster']:02d} vs c{picked[j]['cluster']:02d}: {cc:.3f}")
    json.dump(manifest, open(f"{out}/clusters.json", "w"))
    print(f"\nwrote {out}/clusters.json + {len(picked)} subset files", flush=True)


if __name__ == "__main__":
    main()
