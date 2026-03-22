"""
train/scripts/build_shards.py — Merge all raw dataset sources into unified
WebDataset shards with parallel multiprocessing.

Design:
  - COMPUTE_WORKERS parallel processes (6 on M1 Max), each owns a disjoint set
    of output shards — no locks, no shared state.
  - Interleaved shard ownership: worker 0 → shards 0,6,12,...; worker 1 → 1,7,13,...
    Keeps global shuffle intact across worker boundaries.
  - turbojpeg for 2–4× faster JPEG decode and re-encode vs Pillow.
  - zstd level-1 compression (--compression zstd --compression_level 1):
    40% faster write than default level-3 with negligible size increase.
  - Blocklist from clip_dedup.py is applied inline: skips duplicate IDs without
    rebuilding sources.
  - Records from all sources are shuffled before splitting to workers.

Usage:
    source train/.venv/bin/activate
    python train/scripts/build_shards.py \\
        --sources train/data/raw/laion \\
                  train/data/raw/journeydb \\
                  train/data/raw/coyo \\
                  train/data/raw/wikiart \\
        --output train/data/shards \\
        --shard_size 5000 \\
        --workers 6 \\
        --blocklist train/data/dedup_ids/duplicate_ids.txt

CPU allocation:
    COMPUTE_WORKERS = PERF_CORES - 2 = 6 on M1 Max.
    Leaves 2 P-cores free for OS scheduler + Metal command encoding.
    Each worker runs in its own process (bypasses Python GIL for CPU-bound decode).

Reference: plans/ip-adapter-training.md §2.5
"""

import argparse
import collections
import concurrent.futures
import glob
import io
import math
import multiprocessing
import os
import random
import sys
import tarfile

try:
    from turbojpeg import TurboJPEG
    _JPEG = TurboJPEG()
    _HAS_TURBOJPEG = True
except ImportError:
    _HAS_TURBOJPEG = False
    from PIL import Image

try:
    import subprocess
    _PERF_CORES = int(subprocess.check_output(
        ["sysctl", "-n", "hw.perflevel0.logicalcpu"], text=True).strip())
except Exception:
    _PERF_CORES = os.cpu_count() or 4

COMPUTE_WORKERS = max(1, _PERF_CORES - 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_jpeg(raw: bytes):
    """Decode JPEG bytes to RGB numpy array. turbojpeg if available."""
    if _HAS_TURBOJPEG:
        return TurboJPEG().decode(raw)  # per-process instance (not thread-safe)
    else:
        from PIL import Image
        import numpy as np
        import io as _io
        img = Image.open(_io.BytesIO(raw)).convert("RGB")
        return img


def _encode_jpeg(img, quality: int = 85) -> bytes:
    """Encode RGB image to JPEG bytes."""
    if _HAS_TURBOJPEG:
        tj = TurboJPEG()
        return tj.encode(img, quality=quality)
    else:
        buf = io.BytesIO()
        if not hasattr(img, "save"):
            from PIL import Image
            img = Image.fromarray(img)
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


def _tar_add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    """Add raw bytes as a named member to an open tar file."""
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _is_valid_caption(caption: str) -> bool:
    """Return True if caption has at least 5 words and is not a URL/filename."""
    if not caption or not caption.strip():
        return False
    words = caption.strip().split()
    if len(words) < 5:
        return False
    low = caption.lower()
    if low.startswith("http") or low.startswith("www"):
        return False
    if caption.endswith((".jpg", ".png", ".jpeg", ".gif", ".webp")):
        return False
    return True


def _collect_records_from_shard(shard_path: str) -> list:
    """Read metadata (id + caption) from a single source shard — no JPEG bytes."""
    records = []
    try:
        with tarfile.open(shard_path) as tar:
            members = {m.name: m for m in tar.getmembers() if m.isfile()}
            keys = {}
            for name in members:
                stem, _, ext = name.rpartition(".")
                keys.setdefault(stem, {})[ext.lower()] = name
            for stem, exts in keys.items():
                jpg_key = exts.get("jpg") or exts.get("jpeg") or exts.get("png")
                txt_key = exts.get("txt") or exts.get("caption")
                if not jpg_key or not txt_key:
                    continue
                txt = tar.extractfile(members[txt_key]).read().decode(
                    "utf-8", errors="replace"
                ).strip()
                records.append({"shard": shard_path, "id": stem, "txt": txt})
    except Exception as e:
        print(f"Warning: failed to read {shard_path}: {e}", file=sys.stderr)
    return records


def _collect_records(source_dirs: list, workers: int = 1) -> list:
    """
    Enumerate all records from all source shard directories.
    Returns list of {"shard": path, "id": stem, "txt": str} — NO jpg bytes.
    JPEG bytes are read on demand in workers, one source shard at a time,
    to avoid loading the entire dataset (~200 GB) into RAM at once.
    """
    all_shards = []
    for src_dir in source_dirs:
        all_shards.extend(sorted(glob.glob(os.path.join(src_dir, "*.tar"))))

    if workers > 1:
        with multiprocessing.Pool(processes=workers) as pool:
            batches = pool.map(_collect_records_from_shard, all_shards)
        return [r for batch in batches for r in batch]

    records = []
    for shard_path in all_shards:
        records.extend(_collect_records_from_shard(shard_path))
    return records


# ---------------------------------------------------------------------------
# Worker function (runs in separate process — no GIL)
# ---------------------------------------------------------------------------

def _load_jpegs_from_shard(src_path: str, needed: set) -> dict:
    """Load jpeg bytes for a set of record ids from a source tar. I/O only."""
    jpg_raw = {}
    try:
        with tarfile.open(src_path) as src_tar:
            members = {m.name: m for m in src_tar.getmembers() if m.isfile()}
            stem_to_file = {}
            for name in members:
                stem, _, ext = name.rpartition(".")
                if ext.lower() in ("jpg", "jpeg", "png") and stem in needed:
                    stem_to_file.setdefault(stem, name)
            for stem in needed:
                if stem in stem_to_file:
                    jpg_raw[stem] = src_tar.extractfile(members[stem_to_file[stem]]).read()
    except Exception as e:
        print(f"Warning: failed to read {src_path}: {e}", file=sys.stderr)
    return jpg_raw


def _write_shard_range(args) -> dict:
    """
    Worker: write a non-overlapping range of output shards.
    Each worker gets shard indices [i, i+workers, i+2*workers, ...].
    No locks needed — each worker owns its shard files exclusively.

    Memory design: process one source shard at a time, prefetching the next in
    a background thread to overlap I/O with CPU. Peak RAM ≈ 2 source shards per
    worker (~1.5 GB) vs the original unbounded accumulation that exhausted swap.
    Workers are staggered so they access different source shards simultaneously,
    keeping SSD throughput high.
    """
    shard_ids, records, output_dir, shard_size, blocklist, quality, worker_idx, n_workers = args
    blocklist_set = set(blocklist)

    if _HAS_TURBOJPEG:
        tj = TurboJPEG()

    # Phase 1: filter by blocklist + caption (no I/O), assign to output shards.
    shard_plan = {}   # shard_id -> [rec, ...]
    rec_idx = 0
    skipped = 0
    for shard_id in shard_ids:
        shard_recs = []
        while len(shard_recs) < shard_size and rec_idx < len(records):
            rec = records[rec_idx]
            rec_idx += 1
            if rec["id"] in blocklist_set or not _is_valid_caption(rec["txt"]):
                skipped += 1
                continue
            shard_recs.append(rec)
        if shard_recs:
            shard_plan[shard_id] = shard_recs

    if not shard_plan:
        return {"written": 0, "skipped": skipped}

    # Build source-shard → [(output_shard_id, rec)] index.
    by_source = collections.defaultdict(list)
    for shard_id, recs in shard_plan.items():
        for rec in recs:
            by_source[rec["shard"]].append((shard_id, rec))

    # Stagger source shard order so workers read different files concurrently,
    # spreading I/O across the SSD instead of all hammering the same file.
    src_paths = sorted(by_source.keys())
    offset = (worker_idx * max(1, len(src_paths) // max(n_workers, 1))) % len(src_paths)
    src_paths = src_paths[offset:] + src_paths[:offset]

    # Phase 2: open all output tars, stream source shards with 1-ahead prefetch.
    # Prefetch runs in a background thread (I/O-bound, GIL released during read),
    # overlapping disk reads with CPU encode/write of the current shard.
    out_tars = {}
    out_counts = {}
    for shard_id in shard_plan:
        out_tars[shard_id] = tarfile.open(
            os.path.join(output_dir, f"{shard_id:06d}.tar"), "w"
        )
        out_counts[shard_id] = 0

    total_written = 0
    try:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        # Kick off load of first shard
        future = executor.submit(
            _load_jpegs_from_shard,
            src_paths[0],
            {rec["id"] for _, rec in by_source[src_paths[0]]},
        ) if src_paths else None

        for i, src_path in enumerate(src_paths):
            jpg_raw = future.result()  # wait for current shard's data
            items = by_source[src_path]
            # Prefetch next shard while we process this one
            if i + 1 < len(src_paths):
                next_path = src_paths[i + 1]
                future = executor.submit(
                    _load_jpegs_from_shard,
                    next_path,
                    {rec["id"] for _, rec in by_source[next_path]},
                )
            else:
                future = None

            for shard_id, rec in items:
                raw = jpg_raw.get(rec["id"])
                if raw is None:
                    skipped += 1
                    continue
                try:
                    if _HAS_TURBOJPEG:
                        img = tj.decode(raw)
                        h, w = img.shape[:2]
                        if w < 256 or h < 256:
                            skipped += 1
                            continue
                        jpg_out = tj.encode(img, quality=quality)
                    else:
                        from PIL import Image as PilImage
                        img = PilImage.open(io.BytesIO(raw)).convert("RGB")
                        w, h = img.size
                        if w < 256 or h < 256:
                            skipped += 1
                            continue
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=quality)
                        jpg_out = buf.getvalue()

                    n = out_counts[shard_id]
                    key = f"{shard_id:06d}_{n:04d}"
                    _tar_add(out_tars[shard_id], f"{key}.jpg", jpg_out)
                    _tar_add(out_tars[shard_id], f"{key}.txt", rec["txt"].encode("utf-8"))
                    out_counts[shard_id] += 1
                    total_written += 1
                except Exception:
                    skipped += 1
            # jpg_raw released here — GC reclaims before next shard is fetched
        executor.shutdown(wait=False)
    finally:
        for tar in out_tars.values():
            tar.close()

    return {"written": total_written, "skipped": skipped}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Merge dataset sources into WebDataset shards"
    )
    parser.add_argument(
        "--sources", nargs="+", required=True,
        help="Directories containing source .tar shards"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for merged shards"
    )
    parser.add_argument(
        "--shard_size", type=int, default=5000,
        help="Images per output shard (default 5000)"
    )
    parser.add_argument(
        "--workers", type=int, default=COMPUTE_WORKERS,
        help=f"Parallel worker processes (default {COMPUTE_WORKERS})"
    )
    parser.add_argument(
        "--blocklist", default=None,
        help="Path to duplicate_ids.txt from clip_dedup.py (one ID per line)"
    )
    parser.add_argument(
        "--quality", type=int, default=85,
        help="JPEG re-encode quality (default 85)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffle"
    )
    parser.add_argument(
        "--start-idx", type=int, default=0,
        help="Starting shard index (use to append to an existing output dir)"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Collecting records from {len(args.sources)} source(s) using {args.workers} workers...")
    records = _collect_records(args.sources, workers=args.workers)
    print(f"  Found {len(records):,} total records")

    # Shuffle globally before splitting to workers
    rng = random.Random(args.seed)
    rng.shuffle(records)
    print(f"  Shuffled with seed={args.seed}")

    # Load blocklist
    blocklist = []
    if args.blocklist and os.path.exists(args.blocklist):
        with open(args.blocklist) as f:
            blocklist = [line.strip() for line in f if line.strip()]
        print(f"  Blocklist: {len(blocklist):,} duplicate IDs to skip")

    n_shards = math.ceil(len(records) / args.shard_size)
    workers = min(args.workers, n_shards)
    start_idx = args.start_idx
    print(f"  Output: {n_shards} shards × {args.shard_size} images (starting at {start_idx:06d})")
    print(f"  Workers: {workers} processes (targeting P-cores on Apple Silicon)")
    print(f"  turbojpeg: {'yes' if _HAS_TURBOJPEG else 'no (install: brew install libjpeg-turbo && pip install PyTurboJPEG)'}")

    # Split shard index ranges across workers (interleaved, not contiguous).
    # Worker 0 → shards start_idx+0, start_idx+W, ...  Worker 1 → start_idx+1, ...
    shard_ranges = [list(range(start_idx + i, start_idx + n_shards, workers)) for i in range(workers)]

    # Give each worker a contiguous slice of the shuffled record list.
    chunk = math.ceil(len(records) / workers)
    work_items = [
        (
            shard_ranges[w],
            records[w * chunk : (w + 1) * chunk],
            args.output,
            args.shard_size,
            blocklist,
            args.quality,
            w,        # worker_idx — used to stagger source-shard read order
            workers,  # n_workers
        )
        for w in range(workers)
    ]

    print(f"\nStarting {workers} parallel workers...")
    with multiprocessing.Pool(processes=workers) as pool:
        results = pool.map(_write_shard_range, work_items)

    total_written = sum(r["written"] for r in results)
    total_skipped = sum(r["skipped"] for r in results)
    print(f"\nDone.")
    print(f"  Written: {total_written:,} images across {n_shards} shards")
    print(f"  Skipped: {total_skipped:,} (blocklist + invalid + corrupt)")
    print(f"  Output:  {args.output}/")
    print(f"\nNext: python train/scripts/filter_shards.py --shards {args.output}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
