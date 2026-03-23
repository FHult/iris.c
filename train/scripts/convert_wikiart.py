#!/usr/bin/env python3
"""
train/scripts/convert_wikiart.py — Convert WikiArt HuggingFace parquet to
WebDataset tar shards, ready for build_shards.py.

Supports the huggan/wikiart schema where style/artist/genre are integer
ClassLabel IDs. Label names are loaded from the dataset's feature metadata
and used to generate captions: "{style} painting" or
"A {genre} painted in the {style} style".

Usage:
    source train/.venv/bin/activate
    python train/scripts/convert_wikiart.py \
        --input  train/data/raw/wikiart/data \
        --output train/data/raw/wikiart_wds \
        --shard-size 1000
"""

import argparse
import io
import os
import tarfile

import pandas as pd
from PIL import Image


def _load_label_maps(input_dir: str) -> tuple[list, list, list]:
    """
    Load style/artist/genre label name lists from HuggingFace dataset metadata.
    Falls back to integer strings if metadata is unavailable.
    Returns (style_names, artist_names, genre_names).
    """
    # Walk up one level from the data/ subdir to find dataset_infos.json or
    # .huggingface/dataset_info.json written by snapshot_download.
    for candidate in [
        os.path.join(input_dir, "..", "dataset_infos.json"),
        os.path.join(input_dir, "..", ".huggingface", "dataset_info.json"),
    ]:
        path = os.path.normpath(candidate)
        if os.path.exists(path):
            import json
            with open(path) as f:
                info = json.load(f)
            # dataset_infos.json: {"default": {"features": {...}}}
            # dataset_info.json:  {"features": {...}}
            features = info.get("default", info).get("features", {})
            def _names(key):
                feat = features.get(key, {})
                return feat.get("names", [])
            return _names("style"), _names("artist"), _names("genre")

    # Fallback: use HuggingFace datasets library if available
    try:
        from datasets import load_dataset
        ds = load_dataset("huggan/wikiart", split="train", streaming=True)
        feat = ds.features
        style_names  = feat["style"].names  if hasattr(feat.get("style",  None), "names") else []
        artist_names = feat["artist"].names if hasattr(feat.get("artist", None), "names") else []
        genre_names  = feat["genre"].names  if hasattr(feat.get("genre",  None), "names") else []
        return style_names, artist_names, genre_names
    except Exception:
        pass

    return [], [], []


def _label(names: list, idx) -> str:
    """Return human-readable label, replacing underscores with spaces."""
    if idx is None:
        return ""
    try:
        return names[int(idx)].replace("_", " ")
    except (IndexError, TypeError, ValueError):
        return str(idx)


def make_caption(style: str, genre: str) -> str:
    style = (style or "").strip()
    genre = (genre or "").strip()
    if style and genre and genre.lower() not in ("unknown genre", ""):
        return f"A {genre} painted in the {style} style"
    elif style:
        return f"A painting in the {style} style"
    elif genre:
        return f"A {genre} painting"
    return ""


def write_shard(records: list, path: str):
    tmp_path = path + ".tmp"
    with tarfile.open(tmp_path, "w") as tf:
        for key, jpg_bytes, caption in records:
            jpg_info = tarfile.TarInfo(name=f"{key}.jpg")
            jpg_info.size = len(jpg_bytes)
            tf.addfile(jpg_info, io.BytesIO(jpg_bytes))
            txt_bytes = caption.encode("utf-8")
            txt_info = tarfile.TarInfo(name=f"{key}.txt")
            txt_info.size = len(txt_bytes)
            tf.addfile(txt_info, io.BytesIO(txt_bytes))
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",  default="train/data/raw/wikiart/data")
    parser.add_argument("--output", default="train/data/raw/wikiart_wds")
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--min-size", type=int, default=256,
                        help="Skip images smaller than this in either dimension")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    style_names, artist_names, genre_names = _load_label_maps(args.input)
    if style_names:
        print(f"Loaded {len(style_names)} style labels, {len(genre_names)} genre labels.")
    else:
        print("Warning: could not load label maps — captions will use integer IDs.")

    parquet_files = sorted(
        os.path.join(args.input, f)
        for f in os.listdir(args.input)
        if f.endswith(".parquet")
    )
    if not parquet_files:
        print(f"No parquet files found in {args.input}")
        return
    print(f"Found {len(parquet_files)} parquet files.")

    # Read only the columns that exist in the huggan/wikiart schema.
    # Legacy datasets with description/title will also work because
    # we fall back gracefully when those columns are absent.
    available_cols = None  # detect from first file

    shard_idx = 0
    buf = []
    total_written = total_skipped = 0
    global_idx = 0

    for pq_path in parquet_files:
        print(f"Reading {os.path.basename(pq_path)} ...", flush=True)

        if available_cols is None:
            import pyarrow.parquet as pq_mod
            schema = pq_mod.read_schema(pq_path)
            all_cols = schema.names
            want = ["image", "style", "genre", "description", "title", "artist"]
            available_cols = [c for c in want if c in all_cols]

        df = pd.read_parquet(pq_path, columns=available_cols)

        for row in df.itertuples(index=False):
            img_data = row.image
            raw_bytes = img_data["bytes"] if isinstance(img_data, dict) else img_data
            if not raw_bytes:
                total_skipped += 1
                continue

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

            # Build caption: prefer text fields (legacy schema); fall back to labels
            raw_style = getattr(row, "style", None)
            raw_genre = getattr(row, "genre", None)
            desc  = (getattr(row, "description", None) or "").strip() if hasattr(row, "description") else ""
            title = (getattr(row, "title",       None) or "").strip() if hasattr(row, "title")       else ""

            if style_names and isinstance(raw_style, (int, float)):
                style_str = _label(style_names, raw_style)
                genre_str = _label(genre_names, raw_genre)
            else:
                style_str = str(raw_style or "").strip()
                genre_str = str(raw_genre or "").strip()

            if desc or title:
                # Legacy schema with free-text description
                text = desc or title
                caption = f"{style_str} painting: {text}" if style_str else text
            else:
                caption = make_caption(style_str, genre_str)

            if not caption:
                total_skipped += 1
                continue

            key = f"wikiart_{global_idx:07d}"
            buf.append((key, jpg_bytes, caption))
            global_idx += 1

            if len(buf) >= args.shard_size:
                shard_path = os.path.join(args.output, f"{shard_idx:06d}.tar")
                if os.path.exists(shard_path):
                    print(f"  Shard {shard_idx:06d}.tar already exists — skipping", flush=True)
                else:
                    write_shard(buf, shard_path)
                    print(f"  Wrote shard {shard_idx:06d}.tar ({len(buf)} records)", flush=True)
                    total_written += len(buf)
                shard_idx += 1
                buf = []

    if buf:
        shard_path = os.path.join(args.output, f"{shard_idx:06d}.tar")
        if os.path.exists(shard_path):
            print(f"  Shard {shard_idx:06d}.tar already exists — skipping", flush=True)
        else:
            write_shard(buf, shard_path)
            print(f"  Wrote shard {shard_idx:06d}.tar ({len(buf)} records)", flush=True)
            total_written += len(buf)

    print(f"\nDone. {total_written:,} images written to {args.output}, "
          f"{total_skipped:,} skipped.")


if __name__ == "__main__":
    main()
