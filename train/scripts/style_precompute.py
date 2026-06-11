#!/usr/bin/env python
"""
style_precompute.py — CSD style-embedding precompute (SREF-1 encoder pass).

Encodes shard images with the MLX CSD style encoder (train/style_encoder/csd_mlx.py)
into per-SHARD npz bundles — one file per shard (keys = record IDs, values = [768]
f16 L2-normalised style embeddings), NOT one file per record: a million 3 KB files
is exactly the cold-HDD enumeration trap (PERF-IO / build_index lesson).

Output layout (PRECOMP-3-style identity):
  <out>/<shard_stem>.npz       per-shard bundle
  <out>/manifest.json          encoder identity + record/shard counts

Usage:
  style_precompute.py --shards DIR_OR_TAR [...] --out DIR
                      [--weights /Volumes/2TBSSD/models/csd_vit_l_style.safetensors]
                      [--batch 16] [--limit-shards N]
Resumable: existing complete bundles are skipped (count checked vs tar).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from style_encoder.csd_mlx import CSDStyleEncoder, preprocess

ENCODER_ID = "csd_vit_l_style_v1"


def encode_shard(enc: CSDStyleEncoder, tar_path: Path, batch_size: int,
                 subsample: int = 0) -> dict[str, np.ndarray]:
    """subsample>0: encode only the FIRST N valid images (deterministic prefix —
    same rule as precompute_all --subsample-per-shard, so neighbor records have
    SigLIP features in the cache)."""
    out: dict[str, np.ndarray] = {}
    batch_x, batch_ids = [], []

    def flush():
        nonlocal batch_x, batch_ids
        if batch_x:
            E = enc.encode(np.stack(batch_x))
            for rid, e in zip(batch_ids, E):
                out[rid] = e.astype(np.float16)
            batch_x, batch_ids = [], []

    with tarfile.open(tar_path) as tar:
        for m in tar:
            if not (m.isfile() and m.name.lower().endswith((".jpg", ".jpeg", ".png"))):
                continue
            rid = m.name.rsplit(".", 1)[0]
            try:
                img = Image.open(io.BytesIO(tar.extractfile(m).read()))
                batch_x.append(preprocess(img))
                batch_ids.append(rid)
            except Exception:
                continue
            if subsample and (len(out) + len(batch_x)) >= subsample:
                flush()
                break
            if len(batch_x) >= batch_size:
                flush()
    flush()
    if subsample and len(out) > subsample:
        out = dict(list(out.items())[:subsample])
    return out


def bundle_complete(dst: Path, n_target: int) -> bool:
    """An existing bundle satisfies this run only if it holds >= n_target keys —
    a plain exists() check would block upgrading a subsampled bundle to a fuller
    one (DP-2c monotonic-growth rule). Unreadable bundle -> re-encode."""
    if not dst.exists():
        return False
    try:
        return len(np.load(dst).files) >= n_target
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", nargs="+", required=True,
                    help="shard dirs and/or individual .tar files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="/Volumes/2TBSSD/models/csd_vit_l_style.safetensors")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit-shards", type=int, default=0)
    ap.add_argument("--subsample-per-shard", type=int, default=0,
                    help="encode only the first N valid images per shard "
                         "(deterministic prefix; 0 = full shard)")
    args = ap.parse_args()

    tars: list[Path] = []
    for s in args.shards:
        p = Path(s)
        tars.extend(sorted(p.glob("*.tar")) if p.is_dir() else [p])
    if args.limit_shards:
        tars = tars[:args.limit_shards]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    enc = CSDStyleEncoder(args.weights)
    n_rec = n_done = 0
    t0 = time.time()
    for i, tar_path in enumerate(tars):
        dst = out / f"{tar_path.stem}.npz"
        # Upgrade-aware skip: a subsampled bundle must not block a fuller pass.
        # n_target for a full run (subsample=0) is unknowable without reading the
        # tar, so full runs only trust bundles previously written by a full run
        # (marked via the manifest-free heuristic: >= subsample when set, else
        # re-encode unless the bundle came from a full pass — cheap tar-count).
        if args.subsample_per_shard > 0:
            if bundle_complete(dst, args.subsample_per_shard):
                n_done += 1
                continue
        elif dst.exists():
            with tarfile.open(tar_path) as _t:
                _n_imgs = sum(1 for m in _t
                              if m.isfile() and m.name.lower().endswith(
                                  (".jpg", ".jpeg", ".png")))
            # 2% tolerance: undecodable images never produce keys; without slack a
            # shard with one corrupt jpg would re-encode forever.
            if bundle_complete(dst, int(_n_imgs * 0.98)):
                n_done += 1
                continue
        embs = encode_shard(enc, tar_path, args.batch, args.subsample_per_shard)
        if not embs:
            print(f"  [{tar_path.stem}] no images — skipped", flush=True)
            continue
        tmp = dst.with_suffix(".tmp.npz")
        np.savez(tmp, **embs)
        tmp.rename(dst)
        n_rec += len(embs)
        rate = n_rec / max(1e-9, time.time() - t0)
        print(f"  [{i+1}/{len(tars)}] {tar_path.stem}: {len(embs)} embeds "
              f"({rate:.0f} img/s cumulative)", flush=True)

    manifest = {"encoder": ENCODER_ID, "weights": str(args.weights),
                "dim": 768, "dtype": "float16",
                "shard_count": len(list(out.glob("*.npz"))),
                "record_count": int(sum(len(np.load(f).files) for f in out.glob("*.npz"))),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
