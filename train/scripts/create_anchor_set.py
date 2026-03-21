#!/usr/bin/env python3
"""
train/scripts/create_anchor_set.py — Sample a small fixed anchor set from the
merged shards for use during incremental/chunked training.

The anchor set is mixed into every subsequent training chunk at ~20% ratio to
prevent distribution shift when training on a single-source chunk (e.g. JourneyDB).

The anchor set is kept on local disk permanently and never deleted between chunks.

Usage:
    source train/.venv/bin/activate
    python train/scripts/create_anchor_set.py \
        --shards train/data/raw/laion train/data/raw/wikiart_wds \
        --output train/data/anchor_shards \
        --n 10000

    Pass only the source directories you want in the anchor set.
    Omit JourneyDB to exclude synthetic-prompt images.

Output: ~1.5 GB of webdataset tar shards in train/data/anchor_shards/
"""

import argparse
import io
import os
import random
import tarfile


def _iter_shard(path: str):
    try:
        with tarfile.open(path, "r") as tf:
            members = tf.getmembers()
            keys = {}
            for m in members:
                base, ext = os.path.splitext(m.name)
                keys.setdefault(base, {})[ext] = m
            for base, exts in keys.items():
                if ".jpg" in exts and ".txt" in exts:
                    jpg_f = tf.extractfile(exts[".jpg"])
                    txt_f = tf.extractfile(exts[".txt"])
                    if jpg_f and txt_f:
                        yield {
                            "id": base,
                            "jpg": jpg_f.read(),
                            "txt": txt_f.read().decode("utf-8", errors="replace"),
                        }
    except Exception as e:
        print(f"  Warning: skipping {os.path.basename(path)}: {e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards",  required=True, nargs="+",
                        help="One or more source shard directories (pass each source "
                             "you want to include; omit JourneyDB to exclude it)")
    parser.add_argument("--output",  required=True, help="Output anchor shard directory")
    parser.add_argument("--n",       type=int, default=10_000,
                        help="Number of anchor samples to keep (default 10000)")
    parser.add_argument("--shard-size", type=int, default=1000,
                        help="Records per output shard (default 1000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.output, exist_ok=True)

    # ── Collect candidate records (reservoir sample to avoid loading all into RAM) ─
    shard_paths = []
    for shards_dir in args.shards:
        if not os.path.isdir(shards_dir):
            print(f"  Warning: --shards directory not found: {shards_dir} — skipping")
            continue
        shard_paths.extend(sorted(
            os.path.join(shards_dir, f)
            for f in os.listdir(shards_dir)
            if f.endswith(".tar")
        ))
    if not shard_paths:
        print("No .tar shards found in any of the provided --shards directories.")
        return

    print(f"Scanning {len(shard_paths)} shards across {len(args.shards)} source dir(s)...")

    # Reservoir sampling: keep N records uniformly at random
    reservoir = []
    total_seen = 0

    for shard_path in shard_paths:
        for rec in _iter_shard(shard_path):
            total_seen += 1
            if len(reservoir) < args.n:
                reservoir.append(rec)
            else:
                j = rng.randint(0, total_seen - 1)
                if j < args.n:
                    reservoir[j] = rec

        if total_seen % 50_000 == 0 and total_seen > 0:
            print(f"  Scanned {total_seen:,} records, reservoir {len(reservoir):,}", flush=True)

    print(f"Done scanning. {total_seen:,} eligible records, keeping {len(reservoir):,}.")

    if not reservoir:
        print("ERROR: no records matched — check --shards path and --exclude-sources.")
        return

    rng.shuffle(reservoir)

    # ── Write output shards ───────────────────────────────────────────────────
    shard_idx = 0
    written = 0

    for i in range(0, len(reservoir), args.shard_size):
        batch = reservoir[i:i + args.shard_size]
        path = os.path.join(args.output, f"{shard_idx:06d}.tar")
        with tarfile.open(path, "w") as tf:
            for rec in batch:
                for ext, data in [(".jpg", rec["jpg"]), (".txt", rec["txt"].encode())]:
                    info = tarfile.TarInfo(name=f"anchor_{written:07d}{ext}")
                    info.size = len(data)
                    tf.addfile(info, io.BytesIO(data))
                    written += 1 if ext == ".jpg" else 0
        print(f"  Wrote anchor shard {shard_idx:06d}.tar ({len(batch)} records)", flush=True)
        shard_idx += 1

    size_mb = sum(
        os.path.getsize(os.path.join(args.output, f))
        for f in os.listdir(args.output)
        if f.endswith(".tar")
    ) / 1e6
    print(f"\nAnchor set: {len(reservoir):,} records, {shard_idx} shards, {size_mb:.0f} MB")
    print(f"Output: {args.output}")
    print(f"\nPass to training with: --anchor-shards {args.output}")


if __name__ == "__main__":
    main()
