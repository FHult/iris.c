#!/usr/bin/env python3
"""
train/scripts/convert_journeydb.py — Convert JourneyDB HuggingFace download
to WebDataset tar shards, ready for build_shards.py.

JourneyDB stores images as *.tgz archives inside data/train/imgs/.
Captions come from data/train/train_anno_realease_repath.jsonl.tgz.

Usage:
    source train/.venv/bin/activate
    python train/scripts/convert_journeydb.py \
        --input  train/data/raw/journeydb \
        --output train/data/raw/journeydb_wds \
        --shard-size 5000

Output: WebDataset .tar shards with {key}.jpg + {key}.txt pairs.
"""

import argparse
import glob
import io
import json
import os
import tarfile


def load_annotations(anno_tgz_path: str) -> dict:
    """
    Load train_anno_realease_repath.jsonl.tgz → dict mapping
    image filename (basename, no extension) to caption string.

    JourneyDB annotation format per line:
      {"img_path": "...", "prompt": "...", "caption": "..."}
    or
      {"Key": "...", "Prompt": "...", "Task2": {...}}

    We prefer 'caption' over 'prompt' when both exist.
    """
    print(f"Loading annotations from {os.path.basename(anno_tgz_path)} ...", flush=True)
    captions = {}

    with tarfile.open(anno_tgz_path, "r:gz") as tf:
        for member in tf.getmembers():
            if member.name.endswith(".jsonl"):
                f = tf.extractfile(member)
                if f is None:
                    continue
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Normalise key — support both formats
                    img_path = (obj.get("img_path") or obj.get("Key") or "").strip()
                    caption  = (obj.get("caption") or obj.get("Prompt") or "").strip()

                    if img_path and caption:
                        # Key by basename without extension
                        key = os.path.splitext(os.path.basename(img_path))[0]
                        captions[key] = caption

    print(f"  Loaded {len(captions):,} captions.", flush=True)
    return captions


def convert(input_dir: str, output_dir: str, shard_size: int, min_size: int):
    os.makedirs(output_dir, exist_ok=True)

    # ── Load annotations ──────────────────────────────────────────────────────
    anno_path = os.path.join(input_dir, "data", "train",
                             "train_anno_realease_repath.jsonl.tgz")
    if not os.path.exists(anno_path):
        # Fallback to plain jsonl.tgz
        anno_path = os.path.join(input_dir, "data", "train", "train_anno.jsonl.tgz")
    if not os.path.exists(anno_path):
        raise FileNotFoundError(
            f"Annotation file not found. Expected:\n  {anno_path}\n"
            "Run: python train/scripts/convert_journeydb.py after download completes."
        )

    captions = load_annotations(anno_path)

    # ── Find image tgz archives ───────────────────────────────────────────────
    imgs_dir = os.path.join(input_dir, "data", "train", "imgs")
    tgz_files = sorted(glob.glob(os.path.join(imgs_dir, "*.tgz")))
    if not tgz_files:
        raise FileNotFoundError(f"No .tgz files found in {imgs_dir}")

    print(f"Found {len(tgz_files)} image archives.", flush=True)

    # ── Convert ───────────────────────────────────────────────────────────────
    shard_idx    = 0
    buf          = []   # list of (key, jpg_bytes, caption)
    total_written = 0
    total_skipped = 0
    no_caption    = 0
    global_idx    = 0

    def flush_shard():
        nonlocal shard_idx, total_written
        path = os.path.join(output_dir, f"{shard_idx:06d}.tar")
        with tarfile.open(path, "w") as tf:
            for key, jpg_bytes, txt in buf:
                for ext, data in [(".jpg", jpg_bytes), (".txt", txt.encode())]:
                    info = tarfile.TarInfo(name=key + ext)
                    info.size = len(data)
                    tf.addfile(info, io.BytesIO(data))
        print(f"  Wrote shard {shard_idx:06d}.tar ({len(buf)} records)", flush=True)
        total_written += len(buf)
        shard_idx += 1
        buf.clear()

    for tgz_path in tgz_files:
        archive_name = os.path.basename(tgz_path)
        print(f"Processing {archive_name} ...", flush=True)

        try:
            with tarfile.open(tgz_path, "r:gz") as tf:
                for member in tf.getmembers():
                    if not member.name.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue

                    img_key = os.path.splitext(os.path.basename(member.name))[0]
                    caption = captions.get(img_key)
                    if not caption:
                        no_caption += 1
                        total_skipped += 1
                        continue

                    f = tf.extractfile(member)
                    if f is None:
                        total_skipped += 1
                        continue

                    jpg_bytes = f.read()
                    if len(jpg_bytes) < 1024:
                        total_skipped += 1
                        continue

                    if min_size > 0:
                        try:
                            from PIL import Image as _Img
                            import io as _io
                            img = _Img.open(_io.BytesIO(jpg_bytes))
                            if img.width < min_size or img.height < min_size:
                                total_skipped += 1
                                continue
                        except Exception:
                            total_skipped += 1
                            continue

                    key = f"journeydb_{global_idx:08d}"
                    buf.append((key, jpg_bytes, caption))
                    global_idx += 1

                    if len(buf) >= shard_size:
                        flush_shard()

        except Exception as e:
            print(f"  Warning: failed to read {archive_name}: {e}", flush=True)
            continue

    if buf:
        flush_shard()

    print(f"\nDone. Written: {total_written:,}  Skipped: {total_skipped:,} "
          f"(no caption: {no_caption:,})")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",      default="train/data/raw/journeydb")
    parser.add_argument("--output",     default="train/data/raw/journeydb_wds")
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--min-size",   type=int, default=256)
    args = parser.parse_args()
    convert(args.input, args.output, args.shard_size, args.min_size)


if __name__ == "__main__":
    main()
