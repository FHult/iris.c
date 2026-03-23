#!/usr/bin/env python3
"""
train/scripts/convert_wikiart.py — Convert WikiArt HuggingFace parquet to
WebDataset tar shards, ready for build_shards.py.

Images are already embedded as bytes in the parquet — no downloading needed.
Captions are generated as: "{style} painting: {description}"

Usage:
    source train/.venv/bin/activate
    python train/scripts/convert_wikiart.py \
        --input  train/data/raw/wikiart/data \
        --output train/data/raw/wikiart_wds \
        --shard-size 1000
"""

import argparse
import io
import json
import os
import tarfile

import pandas as pd
from PIL import Image


def make_caption(row) -> str:
    style = (getattr(row, "style", None) or "").strip()
    desc  = (getattr(row, "description", None) or getattr(row, "title", None) or "").strip()
    if style and desc:
        return f"{style} painting: {desc}"
    elif style:
        return f"An artwork painted in the {style} style"
    return desc


def write_shard(records: list, path: str):
    with tarfile.open(path, "w") as tf:
        for i, (key, jpg_bytes, caption) in enumerate(records):
            # jpg
            jpg_info = tarfile.TarInfo(name=f"{key}.jpg")
            jpg_info.size = len(jpg_bytes)
            tf.addfile(jpg_info, io.BytesIO(jpg_bytes))
            # txt
            txt_bytes = caption.encode("utf-8")
            txt_info = tarfile.TarInfo(name=f"{key}.txt")
            txt_info.size = len(txt_bytes)
            tf.addfile(txt_info, io.BytesIO(txt_bytes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",  default="train/data/raw/wikiart/data")
    parser.add_argument("--output", default="train/data/raw/wikiart_wds")
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--min-size", type=int, default=256,
                        help="Skip images smaller than this in either dimension")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    parquet_files = sorted(
        os.path.join(args.input, f)
        for f in os.listdir(args.input)
        if f.endswith(".parquet")
    )
    if not parquet_files:
        print(f"No parquet files found in {args.input}")
        return

    shard_idx = 0
    buf = []
    total_written = total_skipped = 0
    global_idx = 0

    for pq_path in parquet_files:
        print(f"Reading {os.path.basename(pq_path)} ...", flush=True)
        df = pd.read_parquet(pq_path, columns=["image", "style", "description", "title"])

        for row in df.itertuples(index=False):
            img_data = row.image
            raw_bytes = img_data["bytes"] if isinstance(img_data, dict) else img_data

            # Lazy open for size check (header only, no decode); convert only after passing
            try:
                img = Image.open(io.BytesIO(raw_bytes))
                if img.width < args.min_size or img.height < args.min_size:
                    total_skipped += 1
                    continue
                img = img.convert("RGB")
                out = io.BytesIO()
                img.save(out, format="JPEG", quality=90)
                jpg_bytes = out.getvalue()
            except Exception:
                total_skipped += 1
                continue

            caption = make_caption(row)
            if not caption:
                total_skipped += 1
                continue

            key = f"wikiart_{global_idx:07d}"
            buf.append((key, jpg_bytes, caption))
            global_idx += 1

            if len(buf) >= args.shard_size:
                shard_path = os.path.join(args.output, f"{shard_idx:06d}.tar")
                write_shard(buf, shard_path)
                print(f"  Wrote shard {shard_idx:06d}.tar ({len(buf)} records)", flush=True)
                total_written += len(buf)
                shard_idx += 1
                buf = []

    # flush remainder
    if buf:
        shard_path = os.path.join(args.output, f"{shard_idx:06d}.tar")
        write_shard(buf, shard_path)
        print(f"  Wrote shard {shard_idx:06d}.tar ({len(buf)} records)", flush=True)
        total_written += len(buf)

    print(f"\nDone. {total_written:,} images written to {args.output}, "
          f"{total_skipped:,} skipped.")


if __name__ == "__main__":
    main()
