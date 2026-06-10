#!/usr/bin/env python
"""
style_cluster.py — style-cluster the SigLIP cache for style-paired training (SREF-1).

Why: cross_ref currently pairs a target with an ARBITRARY other image, which trains the
adapter to ignore the reference (BACKLOG SREF-1; ref_gap~0). Style-paired training needs
same-style/different-content references. This script builds that pairing substrate:

  1. Sample record IDs WITHOUT enumerating the cold cache (uses the VAE cache's
     _id_index sidecar + direct existence probes on the SigLIP dir).
  2. Per-image style descriptor from cached SigLIP tokens [729,1152]:
     concat(mean, std) over tokens, each half L2-normalised — first/second-moment
     statistics are far more content-invariant than pooled semantics alone (v1;
     a CSD-style learned descriptor can replace this later without changing the DB).
  3. MiniBatchKMeans (K clusters) on the sample; persist centroids + assignments.
  4. Quality report: cluster balance, silhouette (subsample), per-cluster examples.

Outputs (hot SSD — never write to the busy cold disk):
  <out>/centroids.npz          float32 [K, 2304] + descriptor/config metadata
  <out>/assignments.sqlite     rec_id -> cluster (sample only; full pool is a later pass)
  <out>/report.json            balance / silhouette / examples

Usage:
  style_cluster.py [--siglip-dir D] [--vae-index F] [--sample 20000] [--k 256]
                   [--throttle-ms 5] [--out /Volumes/2TBSSD/style_clusters]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from ip_adapter.dataset import _load_siglip_embed  # canonical 4-bit dequant


def build_descriptor(tokens: np.ndarray) -> np.ndarray:
    """[729, 1152] tokens -> 2304-d style descriptor (mean+std, each L2-normalised)."""
    mean = tokens.mean(axis=0)
    std = tokens.std(axis=0)
    mean /= (np.linalg.norm(mean) + 1e-8)
    std /= (np.linalg.norm(std) + 1e-8)
    return np.concatenate([mean, std]).astype(np.float32)


def sample_record_ids(vae_index: Path, siglip_dir: Path, n: int,
                      seed: int, throttle_ms: float) -> list[str]:
    """Random-sample IDs from the VAE id-index sidecar, keeping those that exist in
    the SigLIP cache (direct path probes — no directory enumeration)."""
    ids = vae_index.read_text().split()
    rng = random.Random(seed)
    rng.shuffle(ids)
    picked: list[str] = []
    probed = 0
    for rid in ids:
        probed += 1
        if (siglip_dir / f"{rid}.npz").exists():
            picked.append(rid)
            if len(picked) >= n:
                break
        if throttle_ms and probed % 200 == 0:
            time.sleep(throttle_ms / 1000.0)
    print(f"  sampled {len(picked)} ids ({probed} probed)", flush=True)
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--siglip-dir", default="/Volumes/16TBCold/precomputed/siglip/current")
    ap.add_argument("--vae-index", default=None,
                    help="VAE _id_index sidecar (default: auto-detect in vae/current)")
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--throttle-ms", type=float, default=5.0,
                    help="sleep between cold reads — politeness to concurrent disk users")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/Volumes/2TBSSD/style_clusters")
    args = ap.parse_args()

    siglip_dir = Path(args.siglip_dir).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.vae_index:
        vae_index = Path(args.vae_index)
    else:
        vae_cur = Path("/Volumes/16TBCold/precomputed/vae/current").resolve()
        cands = sorted(vae_cur.glob("_id_index_*.txt"))
        if not cands:
            print("ERROR: no VAE _id_index sidecar found — run a build_index first "
                  "(train_vae_proxy) or pass --vae-index", file=sys.stderr)
            return 1
        vae_index = cands[-1]
    print(f"id source: {vae_index.name}", flush=True)

    ids = sample_record_ids(vae_index, siglip_dir, args.sample, args.seed, args.throttle_ms)
    if len(ids) < args.k * 10:
        print(f"ERROR: only {len(ids)} usable records (< 10x K) — lower --k or raise --sample",
              file=sys.stderr)
        return 1

    # ── descriptors (the cold-read pass; throttled) ─────────────────────────────
    t0 = time.time()
    descs = np.empty((len(ids), 2304), dtype=np.float32)
    kept: list[str] = []
    for i, rid in enumerate(ids):
        toks = _load_siglip_embed(rid, str(siglip_dir))
        if toks is None:
            continue
        descs[len(kept)] = build_descriptor(toks.astype(np.float32))
        kept.append(rid)
        if args.throttle_ms:
            time.sleep(args.throttle_ms / 1000.0)
        if (i + 1) % 2000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  descriptors {i+1}/{len(ids)}  ({rate:.0f}/s)", flush=True)
    descs = descs[:len(kept)]
    print(f"  {len(kept)} descriptors in {time.time()-t0:.0f}s", flush=True)

    # ── cluster ────────────────────────────────────────────────────────────────
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score
    km = MiniBatchKMeans(n_clusters=args.k, random_state=args.seed,
                         batch_size=2048, n_init=10)
    labels = km.fit_predict(descs)
    print(f"  kmeans fit: inertia={km.inertia_:.1f}", flush=True)

    # ── quality report ─────────────────────────────────────────────────────────
    sizes = np.bincount(labels, minlength=args.k)
    sil_n = min(4000, len(kept))
    sil_idx = np.random.RandomState(args.seed).choice(len(kept), sil_n, replace=False)
    sil = float(silhouette_score(descs[sil_idx], labels[sil_idx]))
    # examples: 3 members from each of the 5 largest clusters
    examples = {}
    for c in sizes.argsort()[::-1][:5]:
        members = [kept[j] for j in np.where(labels == c)[0][:3]]
        examples[int(c)] = members
    report = {
        "n_records": len(kept), "k": args.k, "descriptor": "siglip_mean_std_l2_v1",
        "silhouette_4k": round(sil, 4),
        "cluster_sizes": {"min": int(sizes.min()), "median": int(np.median(sizes)),
                          "max": int(sizes.max()), "empty": int((sizes == 0).sum())},
        "largest_cluster_examples": examples,
        "siglip_dir": str(siglip_dir), "seed": args.seed,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))

    # ── persist ────────────────────────────────────────────────────────────────
    np.savez(out / "centroids.npz", centroids=km.cluster_centers_.astype(np.float32),
             k=args.k, descriptor="siglip_mean_std_l2_v1", seed=args.seed)
    db = sqlite3.connect(out / "assignments.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS assignments (rec_id TEXT PRIMARY KEY, cluster INTEGER)")
    db.executemany("INSERT OR REPLACE INTO assignments VALUES (?, ?)",
                   [(r, int(l)) for r, l in zip(kept, labels)])
    db.commit()
    db.close()

    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("largest_cluster_examples", "siglip_dir")}, indent=2))
    print(f"\nWrote: {out}/centroids.npz, assignments.sqlite, report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
