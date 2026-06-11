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
                 subsample: int = 0,
                 existing_keys: int = 0) -> Optional[dict[str, np.ndarray]]:
    """Encode a shard's records into {rec_id: [768] f16}.

    Record eligibility and prefix MUST mirror precompute_all.iter_shard exactly
    (img AND txt member present; counted toward the subsample cap regardless of
    whether the image later decodes) — diverging rules would give style
    neighbors no SigLIP features in the hot cache.

    existing_keys: key count of an already-present bundle. Returns None (skip)
    when it covers >= 98% of this run's target — the tolerance absorbs
    undecodable images, which never produce keys (without it, a shard with one
    corrupt jpg would re-encode forever). A smaller bundle is re-encoded in
    full (DP-2c upgrade)."""
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
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        keys: dict[str, dict] = {}
        for name in members:
            stem, _, ext = name.rpartition(".")
            keys.setdefault(stem, {})[ext.lower()] = name
        eligible = []
        for stem, exts in keys.items():
            jpg_key = exts.get("jpg") or exts.get("jpeg") or exts.get("png")
            txt_key = exts.get("txt") or exts.get("caption")
            if jpg_key and txt_key:
                eligible.append((stem, jpg_key))
        n_target = min(subsample, len(eligible)) if subsample else len(eligible)
        if n_target > 0 and existing_keys >= int(n_target * 0.98):
            return None
        n_seen = 0
        for stem, jpg_key in eligible:
            n_seen += 1
            try:
                img = Image.open(io.BytesIO(tar.extractfile(members[jpg_key]).read()))
                batch_x.append(preprocess(img))
                batch_ids.append(stem)
            except Exception:
                pass
            if len(batch_x) >= batch_size:
                flush()
            if subsample and n_seen >= subsample:
                break
    flush()
    return out


def bundle_keys(dst: Path) -> int:
    """Key count of an existing bundle; 0 if absent or unreadable (-> re-encode)."""
    if not dst.exists():
        return 0
    try:
        return len(np.load(dst).files)
    except Exception:
        return 0


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

    # Clear stale temp files from interrupted runs (incl. the legacy *.tmp.npz
    # naming, which matched *.npz globs and could poison neighbors/publish).
    for stale in list(out.glob("*.npz.tmp")) + list(out.glob("*.tmp.npz")):
        stale.unlink(missing_ok=True)

    enc = CSDStyleEncoder(args.weights)
    n_rec = n_done = 0
    t0 = time.time()
    for i, tar_path in enumerate(tars):
        dst = out / f"{tar_path.stem}.npz"
        # Upgrade-aware skip happens inside encode_shard (it knows the eligible
        # count): an existing bundle covering >=98% of this run's target -> None.
        embs = encode_shard(enc, tar_path, args.batch, args.subsample_per_shard,
                            existing_keys=bundle_keys(dst))
        if embs is None:
            n_done += 1
            continue
        if not embs:
            print(f"  [{tar_path.stem}] no usable records — skipped", flush=True)
            continue
        # Atomic write. The temp name must NOT match *.npz (a crash-orphaned
        # temp would otherwise be loaded by style_neighbors / published to cold
        # as a junk bundle). np.savez appends .npz to bare paths, so write
        # through an open file handle to keep the exact .npz.tmp name.
        tmp = dst.with_name(dst.name + ".tmp")
        try:
            with open(tmp, "wb") as _f:
                np.savez(_f, **embs)
            tmp.rename(dst)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
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
