#!/usr/bin/env python3
"""
train/prepare_laion.py — Pre-filter the LAION-Aesthetics-v2 parquet index.

Downloads only the metadata (no images). Filters to aesthetic_score >= 5.5,
both dimensions >= 512px, English captions. Outputs a parquet file ready to
drive img2dataset.

This runs on internal storage; the parquet files are small (~500MB output).
Run this BEFORE the SSD arrives so the download can start immediately.

Usage:
    source train/.venv/bin/activate
    python train/prepare_laion.py --output laion_filtered.parquet

The output parquet is then passed to train/download_datasets.sh.
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Pre-filter LAION-Aesthetics-v2 parquet")
    parser.add_argument("--output", default="laion_filtered.parquet",
                        help="Output parquet path (default: laion_filtered.parquet)")
    parser.add_argument("--min_aesthetic", type=float, default=5.5,
                        help="Minimum aesthetic score (default: 5.5)")
    parser.add_argument("--min_size", type=int, default=512,
                        help="Minimum width AND height in pixels (default: 512)")
    parser.add_argument("--max_rows", type=int, default=1_500_000,
                        help="Maximum output rows (default: 1500000)")
    parser.add_argument("--cache_dir", default=None,
                        help="HuggingFace cache dir for dataset download")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        import pandas as pd
    except ImportError:
        print("Error: missing packages. Run: source train/.venv/bin/activate", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading LAION-Aesthetics-v2 metadata (streaming, no images) ...")
    print(f"Filter: aesthetic >= {args.min_aesthetic}, size >= {args.min_size}px, max {args.max_rows:,} rows")

    ds = load_dataset(
        "laion/laion2B-en-aesthetic",
        split="train",
        streaming=True,
        cache_dir=args.cache_dir,
    )

    rows = []
    skipped = 0
    seen = 0

    for row in ds:
        seen += 1
        if seen % 100_000 == 0:
            print(f"  Scanned {seen:,} | kept {len(rows):,} | skipped {skipped:,}", flush=True)

        score = row.get("aesthetic_score") or row.get("AESTHETIC_SCORE") or 0.0
        width = row.get("width") or row.get("WIDTH") or 0
        height = row.get("height") or row.get("HEIGHT") or 0
        url = row.get("url") or row.get("URL") or ""
        caption = row.get("text") or row.get("TEXT") or ""

        if score < args.min_aesthetic or width < args.min_size or height < args.min_size:
            skipped += 1
            continue
        if not url or not caption:
            skipped += 1
            continue

        rows.append({"url": url, "caption": caption})
        if len(rows) >= args.max_rows:
            print(f"  Reached max_rows limit ({args.max_rows:,})")
            break

    print(f"\nFiltered: {len(rows):,} rows kept from {seen:,} scanned ({skipped:,} skipped)")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_parquet(args.output, index=False)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Saved: {args.output} ({size_mb:.1f} MB, {len(df):,} rows)")
    print(f"\nNext step: bash train/download_datasets.sh --laion {args.output}")

if __name__ == "__main__":
    main()
