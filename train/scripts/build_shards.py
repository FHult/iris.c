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
    _HAS_TURBOJPEG = True
except ImportError:
    _HAS_TURBOJPEG = False

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

def _is_jpeg(data: bytes) -> bool:
    return len(data) >= 3 and data[0] == 0xFF and data[1] == 0xD8 and data[2] == 0xFF


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


def _parse_source_arg(arg: str) -> tuple:
    """
    Parse a source argument into (directory_path, chunk_idx, total_chunks).

    Plain path:      "/some/dir"            → ("/some/dir", None, None)
    Sliced path:     "/some/dir:2/4"        → ("/some/dir", 2, 4)

    Sliced sources use a deterministic shuffle (seed=42) and take the
    records[start:end] slice corresponding to chunk_idx out of total_chunks.
    This ensures LAION/COYO are split consistently across all 4 training runs.
    Pre-staged sources (JDB, WikiArt) pass no slice spec — they already
    contain only the chunk's data.
    """
    if ":" in arg:
        path, spec = arg.rsplit(":", 1)
        parts = spec.split("/")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return (path, int(parts[0]), int(parts[1]))
    return (arg, None, None)


def _collect_records(source_args: list, workers: int = 1) -> list:
    """
    Enumerate all records from all source shard directories.
    Returns list of {"shard": path, "id": stem, "txt": str} — NO jpg bytes.
    JPEG bytes are read on demand in workers, one source shard at a time,
    to avoid loading the entire dataset (~200 GB) into RAM at once.

    source_args may include slice specs like "/path/to/laion:2/4" to take
    the second quarter of records from that source (for V2 per-chunk sampling).
    """
    all_records = []
    for arg in source_args:
        src_dir, chunk_idx, total_chunks = _parse_source_arg(arg)
        shards = sorted(glob.glob(os.path.join(src_dir, "*.tar")))

        if workers > 1:
            with multiprocessing.Pool(processes=workers) as pool:
                batches = pool.map(_collect_records_from_shard, shards)
            src_records = [r for batch in batches for r in batch]
        else:
            src_records = []
            for shard_path in shards:
                src_records.extend(_collect_records_from_shard(shard_path))

        if chunk_idx is not None and total_chunks and total_chunks > 1:
            # Deterministic slice: shuffle with fixed seed then take chunk's window.
            src_records.sort(key=lambda r: r["id"])  # stable sort before shuffle
            rng = random.Random(42)
            rng.shuffle(src_records)
            n = len(src_records)
            start = (chunk_idx - 1) * n // total_chunks
            end   = chunk_idx * n // total_chunks
            src_records = src_records[start:end]
            print(f"  Source slice {src_dir}: chunk {chunk_idx}/{total_chunks} "
                  f"→ records {start:,}–{end:,} ({len(src_records):,} total)",
                  flush=True)

        all_records.extend(src_records)
    return all_records


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
    shard_ids, records, output_dir, shard_size, blocklist, quality, worker_idx, n_workers, write_from = args
    blocklist_set = set(blocklist)

    if _HAS_TURBOJPEG:
        tj = TurboJPEG()

    # Phase 1: filter by blocklist + caption (no I/O), assign to output shards.
    # shard_ids covers the FULL range from 0 so that record consumption matches
    # a fresh run exactly. Shards before write_from are consumed but not written,
    # which ensures resumed workers pick up at the correct record position.
    shard_plan = {}   # shard_id -> [rec, ...]  (only shard_id >= write_from)
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
        if shard_recs and shard_id >= write_from:
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
    # Output tars are opened lazily on first write so that a crash mid-run
    # does not leave empty stub files that would be mistaken for completed shards.
    # Each shard is written to a .tar.tmp temp file and atomically renamed to .tar
    # only when it reaches shard_size records.  A re-run will skip any .tar that
    # already exists on disk (from a previous complete run) even if shard_id >=
    # write_from, preventing a restart from truncating finished shards.
    shard_plan = {sid: recs for sid, recs in shard_plan.items()
                  if not os.path.exists(os.path.join(output_dir, f"{sid:06d}.tar"))}
    if not shard_plan:
        return {"written": 0, "skipped": skipped}
    out_tars = {}
    out_counts = {shard_id: 0 for shard_id in shard_plan}
    completed_count = 0

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
                        # Full decode validates the entire JPEG bitstream, catching
                        # truncated files that pass a header-only check (DQ-002).
                        img = tj.decode(raw)
                        h, w = img.shape[:2]
                        if w < 256 or h < 256:
                            skipped += 1
                            continue
                        if _is_jpeg(raw):
                            jpg_out = raw  # pass through valid original bytes
                        else:
                            # PNG or other format: re-encode to JPEG
                            jpg_out = tj.encode(img, quality=quality)
                    else:
                        from PIL import Image as PilImage
                        pil_img = PilImage.open(io.BytesIO(raw))
                        pil_img.verify()  # catches truncated files
                        pil_img = PilImage.open(io.BytesIO(raw))  # reopen after verify
                        w, h = pil_img.size
                        if w < 256 or h < 256:
                            skipped += 1
                            continue
                        if _is_jpeg(raw):
                            jpg_out = raw  # pass through JPEG as-is
                        else:
                            _buf = io.BytesIO()
                            pil_img.convert("RGB").save(_buf, format="JPEG", quality=quality)
                            jpg_out = _buf.getvalue()

                    if shard_id not in out_tars:
                        tmp_path = os.path.join(output_dir, f"{shard_id:06d}.tar.tmp")
                        out_tars[shard_id] = tarfile.open(tmp_path, "w")
                    n = out_counts[shard_id]
                    key = f"{shard_id:06d}_{n:04d}"
                    _tar_add(out_tars[shard_id], f"{key}.jpg", jpg_out)
                    _tar_add(out_tars[shard_id], f"{key}.txt", rec["txt"].encode("utf-8"))
                    out_counts[shard_id] += 1
                    total_written += 1
                    # Atomically publish when the shard is full.
                    if out_counts[shard_id] >= shard_size:
                        out_tars[shard_id].close()
                        del out_tars[shard_id]
                        tmp_path = os.path.join(output_dir, f"{shard_id:06d}.tar.tmp")
                        final_path = os.path.join(output_dir, f"{shard_id:06d}.tar")
                        os.replace(tmp_path, final_path)
                        completed_count += 1
                except Exception as _exc:
                    print(
                        f"[worker {worker_idx}] skipping corrupt image "
                        f"rec={rec['id']} src={rec.get('src', '?')}: {_exc}",
                        flush=True,
                    )
                    skipped += 1
            # jpg_raw released here — GC reclaims before next shard is fetched

            # Heartbeat every 5 source shards so pipeline_status.sh shows progress.
            if (i + 1) % 5 == 0 or (i + 1) == len(src_paths):
                print(
                    f"[worker {worker_idx}] src {i+1}/{len(src_paths)} "
                    f"| written {total_written:,} records "
                    f"| shards {completed_count}/{len(shard_plan)} done",
                    flush=True,
                )
        executor.shutdown(wait=True)
    finally:
        # Close remaining open tars and publish any that have content.
        # Shards often don't reach shard_size exactly (images skipped due to
        # decode errors or blocklist), so we must publish partial shards too.
        # Only truly empty shards (0 records written) stay as .tar.tmp to be
        # cleaned up on the next run.
        for shard_id, tar in list(out_tars.items()):
            tar.close()
            if out_counts.get(shard_id, 0) > 0:
                tmp_path = os.path.join(output_dir, f"{shard_id:06d}.tar.tmp")
                final_path = os.path.join(output_dir, f"{shard_id:06d}.tar")
                if os.path.exists(tmp_path):
                    os.replace(tmp_path, final_path)
                    completed_count += 1

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
        help="Directories containing source .tar shards. Append :chunk/total to "
             "take a deterministic slice (e.g. /data/laion:2/4 = second quarter)"
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

    # Clean up any .tar.tmp files left by a previously interrupted run.
    stale_tmps = glob.glob(os.path.join(args.output, "*.tar.tmp"))
    if stale_tmps:
        print(f"Cleaning up {len(stale_tmps)} orphaned .tar.tmp files from previous run...")
        for p in stale_tmps:
            try:
                os.remove(p)
            except OSError:
                pass

    print(f"Collecting records from {len(args.sources)} source(s) using {args.workers} workers...")
    records = _collect_records(args.sources, workers=args.workers)  # args.sources may contain :chunk/total slices
    print(f"  Found {len(records):,} total records")

    # Shuffle globally before splitting to workers
    rng = random.Random(args.seed)
    rng.shuffle(records)
    print(f"  Shuffled with seed={args.seed}")

    # Load blocklist
    blocklist = []
    if args.blocklist:
        if not os.path.exists(args.blocklist):
            print(f"ERROR: blocklist file not found: {args.blocklist}", file=sys.stderr)
            sys.exit(1)
        with open(args.blocklist) as f:
            blocklist = [line.strip() for line in f if line.strip()]
        print(f"  Blocklist: {len(blocklist):,} duplicate IDs to skip")

    if not records:
        print("  No records found in source directories — nothing to shard.", flush=True)
        sys.exit(0)

    n_shards = math.ceil(len(records) / args.shard_size)
    workers = min(args.workers, n_shards)
    start_idx = args.start_idx
    total_shards = start_idx + n_shards
    print(f"  Output: {n_shards} new shards × {args.shard_size} images (writing {start_idx:06d}–{start_idx+n_shards-1:06d} of {total_shards} total)")
    print(f"  Workers: {workers} processes (targeting P-cores on Apple Silicon)")
    print(f"  turbojpeg: {'yes' if _HAS_TURBOJPEG else 'no (install: brew install libjpeg-turbo && pip install PyTurboJPEG)'}")
    # Each worker owns an interleaved subset of the NEW shards only (>= start_idx).
    # Previously workers started from shard 0 and simulated consuming records for
    # shards 0..start_idx-1 before writing — but each worker's record slice is too
    # small to survive that simulation when start_idx is large (e.g. 84 for chunk 3),
    # causing workers to exhaust their records before reaching writable shards → 0 output.
    full_shard_ranges = [list(range(start_idx + i, total_shards, workers)) for i in range(workers)]

    # Give each worker a contiguous slice of the shuffled record list.
    chunk = math.ceil(len(records) / workers)
    work_items = [
        (
            full_shard_ranges[w],
            records[w * chunk : (w + 1) * chunk],
            args.output,
            args.shard_size,
            blocklist,
            args.quality,
            w,          # worker_idx — used to stagger source-shard read order
            workers,    # n_workers
            start_idx,  # write_from — skip writing shards already on disk
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
