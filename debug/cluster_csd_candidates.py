#!/usr/bin/env python3
"""cluster_csd_candidates.py — cluster a per-shard CSD cache into distinct style clusters, extract one
exemplar image per cluster (nearest centroid) from the shard tars, and build a contact sheet — for
broadening the style eval set. Generalizes train/lora/cluster_hot_styles.py to any --csd-dir + --shards-dir.

  train/.venv/bin/python debug/cluster_csd_candidates.py \
      --csd-dir /Volumes/2TBSSD/wikiart_csd --shards-dir /Volumes/2TBSSD/wikiart_pool_hot \
      --k 40 --pick 24 --out /Volumes/2TBSSD/sref_eval/wikiart_candidates
"""
import argparse, glob, os, tarfile
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csd-dir", required=True)
    ap.add_argument("--shards-dir", required=True)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--pick", type=int, default=24)
    ap.add_argument("--sep", type=float, default=0.55, help="max centroid cosine to count as distinct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ids, shard_of, vecs = [], {}, []
    for f in sorted(glob.glob(f"{args.csd_dir}/*.npz")):
        stem = os.path.basename(f)[:-4]
        z = np.load(f)
        for k in z.files:
            ids.append(k); shard_of[k] = stem; vecs.append(z[k].astype(np.float32))
    X = np.stack(vecs)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    print(f"loaded {len(ids)} CSD vectors from {args.csd_dir}")

    km = MiniBatchKMeans(n_clusters=args.k, random_state=args.seed, batch_size=4096, n_init=5)
    lab = km.fit_predict(X)
    C = km.cluster_centers_; C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)
    # rank clusters by tightness (mean cosine to centroid)
    ranked = []
    for c in range(args.k):
        m = np.where(lab == c)[0]
        if len(m) < 3: continue
        ranked.append((c, float((X[m] @ C[c]).mean()), m))
    ranked.sort(key=lambda r: -r[1])
    # greedily pick tight + mutually separated clusters
    picked = []
    for c, tight, m in ranked:
        if len(picked) >= args.pick: break
        if all(float(C[c] @ C[pc]) < args.sep for pc, _, _ in picked):
            picked.append((c, tight, m))
    print(f"picked {len(picked)} distinct clusters (sep<{args.sep})")

    saved = []
    for c, tight, m in picked:
        ex = ids[m[np.argmax(X[m] @ C[c])]]          # rec_id nearest centroid
        stem = shard_of[ex]
        tar = os.path.join(args.shards_dir, f"{stem}.tar")
        try:
            with tarfile.open(tar) as t:
                data = t.extractfile(f"{ex}.jpg").read()
        except Exception as e:
            print(f"  c{c}: extract {ex} failed ({e})"); continue
        fn = os.path.join(args.out, f"c{c:02d}_{ex}.jpg")
        open(fn, "wb").write(data); saved.append((c, tight, fn))
        print(f"  c{c:02d} tight={tight:.3f} {ex}")

    # contact sheet
    cols = 5; rows = (len(saved) + cols - 1) // cols; th = 200; pad = 24
    sheet = Image.new("RGB", (cols * th, rows * (th + pad)), (30, 30, 30))
    dr = ImageDraw.Draw(sheet)
    for i, (c, _, fn) in enumerate(saved):
        im = Image.open(fn).convert("RGB"); im.thumbnail((th, th))
        x = (i % cols) * th; y = (i // cols) * (th + pad)
        sheet.paste(im, (x + (th - im.width) // 2, y + pad))
        dr.text((x + 4, y + 6), f"c{c:02d}", fill=(255, 255, 0))
    cs = os.path.join(args.out, "contact_sheet.png"); sheet.save(cs)
    print(f"contact sheet -> {cs}  ({len(saved)} exemplars)")


if __name__ == "__main__":
    main()
