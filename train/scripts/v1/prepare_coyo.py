#!/usr/bin/env python3
"""
train/scripts/prepare_coyo.py — Pre-filter COYO-700M metadata.

Streams the COYO-700M dataset from HuggingFace and writes a parquet file
containing only the rows that pass quality filters, ready for img2dataset.

No images are downloaded here — only URLs and captions are kept.

Usage:
    source train/.venv/bin/activate
    python train/scripts/prepare_coyo.py \
        --output train/data/raw/coyo_filtered.parquet

Output columns: url, caption
Estimated output size: ~25-40 MB parquet (200k-400k rows)
"""

import argparse
import sys

import pandas as pd
from datasets import load_dataset


# ── Filter thresholds ─────────────────────────────────────────────────────────
MIN_AESTHETIC = 4.5      # aesthetic_score_laion_v2 (LAION CLIP aesthetic predictor)
MIN_SIZE      = 512      # min(width, height) in pixels
MIN_CLIP_SIM  = 0.30     # CLIP image-text similarity
MAX_NSFW      = 0.30     # nsfw_score_opennsfw2 (lower = safer)
MAX_WATERMARK = 0.30     # watermark_score (lower = cleaner)
MAX_ROWS      = 500_000  # cap to keep img2dataset runtime manageable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True,
                        help="Output parquet path, e.g. train/data/raw/coyo_filtered.parquet")
    parser.add_argument("--min-aesthetic", type=float, default=MIN_AESTHETIC)
    parser.add_argument("--min-size",      type=int,   default=MIN_SIZE)
    parser.add_argument("--min-clip-sim",  type=float, default=MIN_CLIP_SIM)
    parser.add_argument("--max-nsfw",      type=float, default=MAX_NSFW)
    parser.add_argument("--max-watermark", type=float, default=MAX_WATERMARK)
    parser.add_argument("--max-rows",      type=int,   default=MAX_ROWS)
    args = parser.parse_args()

    print("Downloading COYO-700M metadata (streaming, no images) ...")
    print(f"Filter: aesthetic >= {args.min_aesthetic}, size >= {args.min_size}px, "
          f"clip_sim >= {args.min_clip_sim}, nsfw <= {args.max_nsfw}, "
          f"watermark <= {args.max_watermark}, max {args.max_rows:,} rows")

    ds = load_dataset(
        "kakaobrain/coyo-700m",
        split="train",
        streaming=True,
    )

    rows = []
    seen = skipped = 0

    for row in ds:
        seen += 1

        if seen % 100_000 == 0:
            print(f"  Scanned {seen:,} | kept {len(rows):,} | skipped {skipped:,}", flush=True)

        if len(rows) >= args.max_rows:
            print(f"  Reached max_rows={args.max_rows:,}, stopping.", flush=True)
            break

        url     = row.get("url") or ""
        caption = row.get("text") or ""
        width   = row.get("width") or 0
        height  = row.get("height") or 0
        aesthetic  = row.get("aesthetic_score_laion_v2") or 0.0
        clip_sim   = row.get("clip_similarity_vitb32") or 0.0
        nsfw       = row.get("nsfw_score_opennsfw2") or 1.0
        watermark  = row.get("watermark_score") or 1.0

        if not url or not caption:
            skipped += 1
            continue
        if aesthetic < args.min_aesthetic:
            skipped += 1
            continue
        if min(width, height) < args.min_size:
            skipped += 1
            continue
        if clip_sim < args.min_clip_sim:
            skipped += 1
            continue
        if nsfw > args.max_nsfw:
            skipped += 1
            continue
        if watermark > args.max_watermark:
            skipped += 1
            continue

        rows.append({"url": url, "caption": caption})

    print(f"\nDone. Scanned {seen:,} rows, kept {len(rows):,}, skipped {skipped:,}.")

    if not rows:
        print("ERROR: no rows passed the filter — check column names or thresholds.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    df.to_parquet(args.output, index=False)
    print(f"Saved {len(df):,} rows to {args.output}")


if __name__ == "__main__":
    main()
