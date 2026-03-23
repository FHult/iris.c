#!/usr/bin/env python3
"""
train/scripts/convert_journeydb.py — Convert JourneyDB HuggingFace download
to WebDataset tar shards, ready for build_shards.py.

JourneyDB stores images as *.tgz archives inside data/train/imgs/.
Captions come from data/train/train_anno_realease_repath.jsonl.tgz.

Parallel design: each .tgz archive is processed by a separate worker.
Shard files are named {tgz_idx:04d}_{shard_local:04d}.tar so workers
never collide. build_shards.py picks them all up via glob("*.tar").

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
import multiprocessing
import os
import sys
import tarfile

try:
    import subprocess
    _PERF_CORES = int(subprocess.check_output(
        ["sysctl", "-n", "hw.perflevel0.logicalcpu"], text=True).strip())
except Exception:
    _PERF_CORES = os.cpu_count() or 4

COMPUTE_WORKERS = max(1, _PERF_CORES - 2)

try:
    from turbojpeg import TurboJPEG, TJPF_RGB
    _HAS_TURBOJPEG = True
except ImportError:
    _HAS_TURBOJPEG = False


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

                    img_path = (obj.get("img_path") or obj.get("Key") or "").strip()
                    task2    = obj.get("Task2") or {}
                    caption  = (obj.get("caption")
                                or task2.get("Caption")
                                or obj.get("prompt")
                                or obj.get("Prompt")
                                or "").strip()

                    if img_path and caption:
                        key = os.path.splitext(os.path.basename(img_path))[0]
                        captions[key] = caption

    print(f"  Loaded {len(captions):,} captions.", flush=True)
    return captions


def _process_tgz(args) -> dict:
    """
    Worker: process one .tgz archive → write WDS shards.

    Shards are named {tgz_idx:04d}_{shard_local:04d}.tar so parallel workers
    never collide. Returns counts for progress reporting.
    """
    tgz_path, captions, output_dir, shard_size, min_size, tgz_idx = args

    if _HAS_TURBOJPEG:
        tj = TurboJPEG()

    archive_name = os.path.basename(tgz_path)
    buf = []
    shard_local = 0
    written = 0
    skipped = 0
    no_caption = 0
    global_idx = 0

    def flush_shard():
        nonlocal shard_local, written
        path = os.path.join(output_dir, f"{tgz_idx:04d}_{shard_local:04d}.tar")
        if os.path.exists(path):
            print(f"  [{archive_name}] shard {shard_local} already exists — skipping", flush=True)
        else:
            tmp_path = path + ".tmp"
            with tarfile.open(tmp_path, "w") as tf:
                for key, jpg_bytes, txt in buf:
                    for ext, data in [(".jpg", jpg_bytes), (".txt", txt.encode())]:
                        info = tarfile.TarInfo(name=key + ext)
                        info.size = len(data)
                        tf.addfile(info, io.BytesIO(data))
            os.replace(tmp_path, path)
            written += len(buf)
        shard_local += 1
        buf.clear()

    try:
        with tarfile.open(tgz_path, "r:gz") as tf:
            for member in tf.getmembers():
                if not member.name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                img_key = os.path.splitext(os.path.basename(member.name))[0]
                caption = captions.get(img_key)
                if not caption:
                    no_caption += 1
                    skipped += 1
                    continue

                f = tf.extractfile(member)
                if f is None:
                    skipped += 1
                    continue

                raw = f.read()
                if len(raw) < 1024:
                    skipped += 1
                    continue

                if min_size > 0:
                    try:
                        if _HAS_TURBOJPEG:
                            w, h, _, _ = tj.decode_header(raw)
                        else:
                            from PIL import Image as _Img
                            img = _Img.open(io.BytesIO(raw))
                            w, h = img.size
                        if w < min_size or h < min_size:
                            skipped += 1
                            continue
                    except Exception:
                        skipped += 1
                        continue

                key = f"journeydb_{tgz_idx:04d}_{global_idx:06d}"
                buf.append((key, raw, caption))
                global_idx += 1

                if len(buf) >= shard_size:
                    flush_shard()

    except Exception as e:
        print(f"  Warning: failed to read {archive_name}: {e}", file=sys.stderr, flush=True)
        return {"tgz": archive_name, "written": written, "skipped": skipped,
                "no_caption": no_caption, "error": True}

    if buf:
        flush_shard()

    return {"tgz": archive_name, "written": written, "skipped": skipped,
            "no_caption": no_caption, "error": False}


def convert(input_dir: str, output_dir: str, shard_size: int, min_size: int,
            start_tgz: int | None = None, end_tgz: int | None = None,
            workers: int = COMPUTE_WORKERS):
    os.makedirs(output_dir, exist_ok=True)

    # ── Load annotations (main process — shared by all workers via pickle) ──────
    anno_path = os.path.join(input_dir, "data", "train",
                             "train_anno_realease_repath.jsonl.tgz")
    if not os.path.exists(anno_path):
        anno_path = os.path.join(input_dir, "data", "train", "train_anno.jsonl.tgz")
    if not os.path.exists(anno_path):
        raise FileNotFoundError(
            f"Annotation file not found. Expected:\n  {anno_path}\n"
            "Run after download completes."
        )

    captions = load_annotations(anno_path)

    # ── Find image tgz archives ───────────────────────────────────────────────
    imgs_dir = os.path.join(input_dir, "data", "train", "imgs")
    all_tgz = sorted(glob.glob(os.path.join(imgs_dir, "*.tgz")))

    def _tgz_num(p: str) -> int:
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            return int(stem)
        except ValueError:
            return -1

    tgz_files = [
        f for f in all_tgz
        if _tgz_num(f) >= 0
        and (start_tgz is None or _tgz_num(f) >= start_tgz)
        and (end_tgz   is None or _tgz_num(f) <= end_tgz)
    ]
    if not tgz_files:
        raise FileNotFoundError(
            f"No .tgz files found in {imgs_dir} "
            f"(start_tgz={start_tgz}, end_tgz={end_tgz})"
        )

    range_str = ""
    if start_tgz is not None or end_tgz is not None:
        range_str = f" (filtered: start={start_tgz}, end={end_tgz})"
    n_workers = min(workers, len(tgz_files))
    print(f"Found {len(tgz_files)} image archives{range_str}.")
    print(f"Workers: {n_workers}  turbojpeg: {'yes' if _HAS_TURBOJPEG else 'no (slower fallback)'}")

    # tgz_idx is the position in the full sorted list (not filtered), so that
    # shard names are stable across repeated runs with different start/end ranges.
    all_tgz_sorted = sorted(
        f for f in all_tgz if _tgz_num(f) >= 0
    )
    tgz_idx_map = {p: i for i, p in enumerate(all_tgz_sorted)}

    work_items = [
        (tgz_path, captions, output_dir, shard_size, min_size, tgz_idx_map[tgz_path])
        for tgz_path in tgz_files
    ]

    # ── Process in parallel ───────────────────────────────────────────────────
    import time as _time
    results = []
    t_start = _time.time()
    with multiprocessing.Pool(processes=n_workers) as pool:
        for done, result in enumerate(
            pool.imap_unordered(_process_tgz, work_items, chunksize=1), 1
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
                f"  [{done}/{len(work_items)}] {result['tgz']}"
                f" → written={total_written:,}  skipped={total_skipped:,}"
                f"{err_str}  ETA {eta/60:.0f}m",
                flush=True,
            )

    total_written = sum(r["written"] for r in results)
    total_skipped = sum(r["skipped"] for r in results)
    no_caption    = sum(r["no_caption"] for r in results)
    errors        = sum(1 for r in results if r["error"])
    shard_count   = len(glob.glob(os.path.join(output_dir, "*.tar")))
    print(f"\nDone. Written: {total_written:,}  Skipped: {total_skipped:,} "
          f"(no caption: {no_caption:,})")
    print(f"Output: {output_dir}  ({shard_count} shards)")
    if errors:
        print(f"  {errors} archives had errors (check stderr)", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",      default="train/data/raw/journeydb")
    parser.add_argument("--output",     default="train/data/raw/journeydb_wds")
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--min-size",   type=int, default=256)
    parser.add_argument("--workers",    type=int, default=COMPUTE_WORKERS,
                        help=f"Parallel workers (default {COMPUTE_WORKERS} = P-cores - 2)")
    parser.add_argument("--start-tgz",  type=int, default=None,
                        help="First tgz archive number to include (e.g. 50 for chunk 2)")
    parser.add_argument("--end-tgz",    type=int, default=None,
                        help="Last tgz archive number to include, inclusive (e.g. 99 for chunk 2)")
    args = parser.parse_args()
    convert(args.input, args.output, args.shard_size, args.min_size,
            start_tgz=args.start_tgz, end_tgz=args.end_tgz, workers=args.workers)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
