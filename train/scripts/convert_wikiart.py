#!/usr/bin/env python3
"""
train/scripts/convert_wikiart.py — Convert WikiArt HuggingFace parquet to
WebDataset tar shards, ready for build_shards.py.

Supports the huggan/wikiart schema where style/artist/genre are integer
ClassLabel IDs. Label names are loaded from the dataset's feature metadata
and used to generate captions: "{style} painting" or
"A {genre} painted in the {style} style".

Parallel design: each parquet file is processed by a separate worker.
Shard files are named {parquet_idx:04d}_{shard_local:04d}.tar so workers
never collide. build_shards.py picks them all up via glob("*.tar").

Usage:
    source train/.venv/bin/activate
    python train/scripts/convert_wikiart.py \
        --input  train/data/raw/wikiart/data \
        --output train/data/raw/wikiart_wds \
        --shard-size 1000
"""

import argparse
import io
import multiprocessing
import os
import sys
import tarfile

import pandas as pd

try:
    import subprocess
    _PERF_CORES = int(subprocess.check_output(
        ["sysctl", "-n", "hw.perflevel0.logicalcpu"], text=True).strip())
except Exception:
    _PERF_CORES = os.cpu_count() or 4

COMPUTE_WORKERS = max(1, _PERF_CORES - 2)

try:
    from turbojpeg import TurboJPEG
    _HAS_TURBOJPEG = True
except ImportError:
    _HAS_TURBOJPEG = False


def _load_label_maps(input_dir: str) -> tuple[list, list, list]:
    """
    Load style/artist/genre label name lists from HuggingFace dataset metadata.
    Falls back to integer strings if metadata is unavailable.
    Returns (style_names, artist_names, genre_names).
    """
    for candidate in [
        os.path.join(input_dir, "..", "dataset_infos.json"),
        os.path.join(input_dir, "..", ".huggingface", "dataset_info.json"),
    ]:
        path = os.path.normpath(candidate)
        if os.path.exists(path):
            import json
            with open(path) as f:
                info = json.load(f)
            features = info.get("default", info).get("features", {})
            def _names(key):
                feat = features.get(key, {})
                return feat.get("names", [])
            return _names("style"), _names("artist"), _names("genre")

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


def _process_parquet(args) -> dict:
    """
    Worker: process one parquet file → write WDS shards.

    Shards are named {parquet_idx:04d}_{shard_local:04d}.tar so parallel
    workers never collide. Returns counts for progress reporting.
    """
    pq_path, parquet_idx, output_dir, shard_size, min_size, \
        style_names, genre_names, available_cols = args

    if _HAS_TURBOJPEG:
        tj = TurboJPEG()

    pq_name = os.path.basename(pq_path)
    buf = []
    shard_local = 0
    written = 0
    skipped = 0
    global_idx = 0

    def flush_shard():
        nonlocal shard_local, written
        path = os.path.join(output_dir, f"{parquet_idx:04d}_{shard_local:04d}.tar")
        if os.path.exists(path):
            pass  # already written by a previous run
        else:
            tmp_path = path + ".tmp"
            with tarfile.open(tmp_path, "w") as tf:
                for key, jpg_bytes, caption in buf:
                    jpg_info = tarfile.TarInfo(name=f"{key}.jpg")
                    jpg_info.size = len(jpg_bytes)
                    tf.addfile(jpg_info, io.BytesIO(jpg_bytes))
                    txt_bytes = caption.encode("utf-8")
                    txt_info = tarfile.TarInfo(name=f"{key}.txt")
                    txt_info.size = len(txt_bytes)
                    tf.addfile(txt_info, io.BytesIO(txt_bytes))
            os.replace(tmp_path, path)
            written += len(buf)
        shard_local += 1
        buf.clear()

    try:
        df = pd.read_parquet(pq_path, columns=available_cols)
    except Exception as e:
        print(f"  Error reading {pq_name}: {e}", file=sys.stderr, flush=True)
        return {"pq": pq_name, "written": written, "skipped": skipped, "error": True}

    for row in df.itertuples(index=False):
        img_data = row.image
        raw_bytes = img_data["bytes"] if isinstance(img_data, dict) else img_data
        if not raw_bytes:
            skipped += 1
            continue

        try:
            if _HAS_TURBOJPEG:
                try:
                    w, h, _, _ = tj.decode_header(raw_bytes)
                except Exception:
                    # Not JPEG (e.g. PNG) — re-encode via PIL
                    from PIL import Image
                    img = Image.open(io.BytesIO(raw_bytes))
                    w, h = img.width, img.height
                    if w < min_size or h < min_size:
                        skipped += 1
                        continue
                    img = img.convert("RGB")
                    out = io.BytesIO()
                    img.save(out, format="JPEG", quality=90)
                    jpg_bytes = out.getvalue()
                else:
                    if w < min_size or h < min_size:
                        skipped += 1
                        continue
                    jpg_bytes = raw_bytes  # already JPEG, pass through
            else:
                from PIL import Image
                img = Image.open(io.BytesIO(raw_bytes))
                if img.width < min_size or img.height < min_size:
                    skipped += 1
                    continue
                if img.format == "JPEG" and img.mode == "RGB":
                    jpg_bytes = raw_bytes  # pass through
                else:
                    img = img.convert("RGB")
                    out = io.BytesIO()
                    img.save(out, format="JPEG", quality=90)
                    jpg_bytes = out.getvalue()
        except Exception:
            skipped += 1
            continue

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
            text = desc or title
            caption = f"{style_str} painting: {text}" if style_str else text
        else:
            caption = make_caption(style_str, genre_str)

        if not caption:
            skipped += 1
            continue

        key = f"wikiart_{parquet_idx:04d}_{global_idx:06d}"
        buf.append((key, jpg_bytes, caption))
        global_idx += 1

        if len(buf) >= shard_size:
            flush_shard()

    if buf:
        flush_shard()

    return {"pq": pq_name, "written": written, "skipped": skipped, "error": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",      default="train/data/raw/wikiart/data")
    parser.add_argument("--output",     default="train/data/raw/wikiart_wds")
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--min-size",   type=int, default=256,
                        help="Skip images smaller than this in either dimension")
    parser.add_argument("--workers",    type=int, default=COMPUTE_WORKERS,
                        help=f"Parallel workers (default {COMPUTE_WORKERS} = P-cores - 2)")
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

    # Detect available columns from the first parquet schema (all files share it)
    import pyarrow.parquet as pq_mod
    schema = pq_mod.read_schema(parquet_files[0])
    all_cols = schema.names
    want = ["image", "style", "genre", "description", "title", "artist"]
    available_cols = [c for c in want if c in all_cols]

    n_workers = min(args.workers, len(parquet_files))
    print(f"Found {len(parquet_files)} parquet files.")
    print(f"Workers: {n_workers}  turbojpeg: {'yes' if _HAS_TURBOJPEG else 'no (slower fallback)'}")

    work_items = [
        (pq_path, idx, args.output, args.shard_size, args.min_size,
         style_names, genre_names, available_cols)
        for idx, pq_path in enumerate(parquet_files)
    ]

    import time as _time
    results = []
    t_start = _time.time()
    with multiprocessing.Pool(processes=n_workers) as pool:
        for done, result in enumerate(
            pool.imap_unordered(_process_parquet, work_items, chunksize=1), 1
        ):
            results.append(result)
            total_written = sum(r["written"] for r in results)
            total_skipped = sum(r["skipped"] for r in results)
            errs = sum(1 for r in results if r["error"])
            elapsed = _time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(work_items) - done) / rate if rate > 0 else 0
            err_str = f"  errors={errs}" if errs else ""
            print(
                f"  [{done}/{len(work_items)}] {result['pq']}"
                f" → written={total_written:,}  skipped={total_skipped:,}"
                f"{err_str}  ETA {eta/60:.1f}m",
                flush=True,
            )

    total_written = sum(r["written"] for r in results)
    total_skipped = sum(r["skipped"] for r in results)
    errors = sum(1 for r in results if r["error"])
    shard_count = len([f for f in os.listdir(args.output) if f.endswith(".tar")])
    print(f"\nDone. {total_written:,} images written to {args.output} ({shard_count} shards), "
          f"{total_skipped:,} skipped.")
    if errors:
        print(f"  {errors} parquet files had errors (check stderr)", file=sys.stderr)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
