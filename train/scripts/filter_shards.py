"""
train/scripts/filter_shards.py — Validate and clean all shards in parallel.

Drops records with:
  - Corrupted or unreadable images
  - width < 256 or height < 256
  - Empty captions, captions < 5 words, captions that are filenames or URLs

Uses multiprocessing.Pool(processes=PERF_CORES) — pure CPU validation parallelises
identically to shard writing. Each worker rewrites its assigned shard in place.

Expected loss: ~3–5% → ~1.55M usable unique images from 1.6M input.

Usage:
    source train/.venv/bin/activate
    python train/scripts/filter_shards.py \\
        --shards train/data/shards

Reference: plans/ip-adapter-training.md §2.6
"""

import argparse
import glob
import io
import multiprocessing
import os
import sys
import tarfile
import tempfile

try:
    from turbojpeg import TurboJPEG, TJPF_RGB
    _HAS_TURBOJPEG = True
except ImportError:
    _HAS_TURBOJPEG = False

try:
    import subprocess
    _PERF_CORES = int(subprocess.check_output(
        ["sysctl", "-n", "hw.perflevel0.logicalcpu"], text=True).strip())
except Exception:
    _PERF_CORES = os.cpu_count() or 4

# Criteria used by the current run (set via Pool initializer from main args).
# Stored as module-level so worker processes can read them without passing per-shard.
_CRITERIA: dict = {"min_size": 256, "min_words": 5}

# Sentinels written by older versions of this script are empty files.
# Treat empty content as the historical defaults so existing filtered shards
# are not re-processed unless the user actually changes the criteria.
_LEGACY_FINGERPRINT = "min_size=256 min_words=5"


def _criteria_fingerprint(min_size: int, min_words: int) -> str:
    return f"min_size={min_size} min_words={min_words}"


def _sentinel_valid(shard_path: str, fingerprint: str) -> bool:
    """Return True iff shard_path + '.filtered' exists and matches fingerprint."""
    p = shard_path + ".filtered"
    if not os.path.exists(p):
        return False
    try:
        content = open(p).read().strip()
        stored = content if content else _LEGACY_FINGERPRINT
        return stored == fingerprint
    except OSError:
        return False


def _worker_init(min_size: int, min_words: int) -> None:
    global _CRITERIA
    _CRITERIA = {"min_size": min_size, "min_words": min_words}


def _is_valid_caption(caption: str, min_words: int) -> bool:
    if not caption or not caption.strip():
        return False
    words = caption.strip().split()
    if len(words) < min_words:
        return False
    low = caption.lower()
    if low.startswith("http") or low.startswith("www"):
        return False
    if caption.rstrip().endswith((".jpg", ".png", ".jpeg", ".gif", ".webp")):
        return False
    return True


def filter_shard(shard_path: str) -> dict:
    """
    Validate all records in one shard; rewrite in place keeping only valid ones.
    Returns dict with kept/dropped counts.
    """
    min_size = _CRITERIA["min_size"]
    min_words = _CRITERIA["min_words"]
    fingerprint = _criteria_fingerprint(min_size, min_words)
    if _HAS_TURBOJPEG:
        tj = TurboJPEG()

    kept_records = []
    original_count = 0

    try:
        with tarfile.open(shard_path) as tar:
            # Use iteration instead of getmembers() so that truncated tars
            # (missing end-of-archive block) are handled gracefully: we read
            # as many members as possible and silently stop at the truncation.
            members = {}
            try:
                for m in tar:
                    if m.isfile():
                        members[m.name] = m
            except Exception:
                pass  # truncated — use whatever was read before EOF
            keys = {}
            for name in members:
                stem, _, ext = name.rpartition(".")
                if stem not in keys:
                    keys[stem] = {}
                keys[stem][ext.lower()] = name

            for stem, exts in keys.items():
                jpg_key = exts.get("jpg") or exts.get("jpeg") or exts.get("png")
                txt_key = exts.get("txt") or exts.get("caption")
                if not jpg_key or not txt_key:
                    continue

                original_count += 1

                txt = tar.extractfile(members[txt_key]).read().decode(
                    "utf-8", errors="replace"
                ).strip()
                if not _is_valid_caption(txt, min_words):
                    continue

                jpg = tar.extractfile(members[jpg_key]).read()
                try:
                    if _HAS_TURBOJPEG:
                        # decode_header reads only the JPEG header (~0.01 ms vs ~5 ms
                        # for a full decode) — sufficient to get dimensions.
                        w, h, _, _ = tj.decode_header(jpg)
                    else:
                        from PIL import Image as PilImage
                        w, h = PilImage.open(io.BytesIO(jpg)).size
                    if w < min_size or h < min_size:
                        continue
                except Exception:
                    continue

                kept_records.append({"key": stem, "jpg": jpg, "txt": txt})
    except Exception as e:
        print(f"Error reading {shard_path}: {e}", file=sys.stderr)
        return {"shard": shard_path, "kept": 0, "dropped": 0, "error": True}

    done_path = shard_path + ".filtered"

    # Guard: treat 0-record shard as an error (truncated tar with no usable content).
    if original_count == 0:
        print(f"Error: {shard_path} has 0 records (truncated or empty tar)", file=sys.stderr)
        return {"shard": shard_path, "kept": 0, "dropped": 0, "error": True}

    # Fast path: nothing dropped — write sentinel without any I/O
    if len(kept_records) == original_count:
        open(done_path, "w").write(fingerprint + "\n")
        return {"shard": shard_path, "kept": original_count, "dropped": 0, "error": False}

    # Rewrite shard in a temp file in the same directory then atomically replace.
    # Using mkstemp (not shard_path + ".tmp") prevents concurrent build_shards.py
    # from accidentally unlinking a temp file it is still writing to.
    shard_dir = os.path.dirname(shard_path)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=shard_dir, suffix=".filter_tmp")
    try:
        os.close(tmp_fd)
        with tarfile.open(tmp_path, "w") as out_tar:
            for i, rec in enumerate(kept_records):
                key = f"{rec['key']}"
                for ext, data in [("jpg", rec["jpg"]), ("txt", rec["txt"].encode("utf-8"))]:
                    info = tarfile.TarInfo(name=f"{key}.{ext}")
                    info.size = len(data)
                    out_tar.addfile(info, io.BytesIO(data))
        os.replace(tmp_path, shard_path)
        open(done_path, "w").write(fingerprint + "\n")  # mark done after successful replace
    except Exception as e:
        print(f"Error rewriting {shard_path}: {e}", file=sys.stderr)
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return {"shard": shard_path, "kept": len(kept_records),
                "dropped": original_count - len(kept_records), "error": True}

    return {"shard": shard_path, "kept": len(kept_records),
            "dropped": original_count - len(kept_records), "error": False}


def main():
    parser = argparse.ArgumentParser(
        description="Validate and filter WebDataset shards in parallel"
    )
    parser.add_argument(
        "--shards", required=True,
        help="Directory containing .tar shards to filter"
    )
    parser.add_argument(
        "--workers", type=int, default=_PERF_CORES,
        help=f"Parallel workers (default {_PERF_CORES} = all P-cores)"
    )
    parser.add_argument(
        "--start-idx", type=int, default=0,
        help="Only filter shards whose numeric index >= this value (use to "
             "filter only newly appended shards without re-processing old ones)"
    )
    parser.add_argument(
        "--min-size", type=int, default=256,
        help="Drop images smaller than this in either dimension (default 256). "
             "Changing this value invalidates existing per-shard sentinels."
    )
    parser.add_argument(
        "--min-words", type=int, default=5,
        help="Drop captions with fewer than this many words (default 5). "
             "Changing this value invalidates existing per-shard sentinels."
    )
    args = parser.parse_args()

    # Clean up orphaned .tmp files left by a previously interrupted run.
    # Only remove files older than 5 minutes — younger files may be actively
    # written by a concurrent build_shards.py run, and deleting them would
    # unlink the directory entry while the fd stays open, causing build_shards
    # to silently produce zero .tar files (os.path.exists returns False in finally).
    import time as _time
    stale_cutoff = _time.time() - 300  # 5 minutes
    stale_tmps = [p for p in glob.glob(os.path.join(args.shards, "*.tar.tmp"))
                  if os.path.getmtime(p) < stale_cutoff]
    if stale_tmps:
        print(f"Cleaning up {len(stale_tmps)} orphaned .tmp files from previous run...")
        for p in stale_tmps:
            try:
                os.remove(p)
            except OSError:
                pass

    all_shards = sorted(glob.glob(os.path.join(args.shards, "*.tar")))
    if args.start_idx > 0:
        all_shards = [s for s in all_shards
                      if int(os.path.splitext(os.path.basename(s))[0]) >= args.start_idx]
        print(f"--start-idx {args.start_idx}: {len(all_shards)} shards in range")

    fingerprint = _criteria_fingerprint(args.min_size, args.min_words)

    # Skip shards whose sentinel matches current criteria fingerprint
    shards = [s for s in all_shards if not _sentinel_valid(s, fingerprint)]
    already_done = len(all_shards) - len(shards)
    if already_done:
        print(f"Skipping {already_done} already-filtered shards (resume)")
    if not shards:
        print(f"All shards already filtered.")
        return

    print(f"Filtering {len(shards)} shards with {args.workers} workers...")
    print(f"  criteria: min_size={args.min_size}  min_words={args.min_words}")
    print(f"  turbojpeg: {'yes' if _HAS_TURBOJPEG else 'no (slower fallback)'}")

    import time as _time
    results = []
    t_start = _time.time()
    t_last_hb = t_start
    interval_rates = []
    last_done = 0
    with multiprocessing.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(args.min_size, args.min_words),
    ) as pool:
        for done, result in enumerate(
            pool.imap_unordered(filter_shard, shards, chunksize=4), 1
        ):
            results.append(result)
            # Warn on unusually high drop rate (>15%) — may indicate a data quality
            # regression in the source.  Expected loss is 3–5%.
            if not result.get("error"):
                _total = result["kept"] + result["dropped"]
                if _total > 0 and result["dropped"] / _total > 0.15:
                    print(
                        f"WARNING: {os.path.basename(result['shard'])} dropped "
                        f"{result['dropped']}/{_total} records "
                        f"({result['dropped']/_total:.0%}) — unusually high",
                        file=sys.stderr, flush=True,
                    )
            if done % 10 == 0 or done == len(shards):
                kept_so_far = sum(r["kept"] for r in results)
                dropped_so_far = sum(r["dropped"] for r in results)
                errs_so_far = sum(1 for r in results if r["error"])
                t_now = _time.time()
                interval_time = t_now - t_last_hb
                if interval_time > 0:
                    interval_rates.append((done - last_done) / interval_time)
                avg_rate = sum(interval_rates) / len(interval_rates) if interval_rates else 0
                eta = (len(shards) - done) / avg_rate if avg_rate > 0 else 0
                t_last_hb = t_now
                last_done = done
                err_str = f"  errors={errs_so_far}" if errs_so_far else ""
                print(
                    f"  [{done}/{len(shards)}] kept={kept_so_far:,}"
                    f"  dropped={dropped_so_far:,}{err_str}"
                    f"  {1/avg_rate:.1f} s/shard  ETA {eta/60:.0f}m",
                    flush=True,
                )

    total_kept = sum(r["kept"] for r in results)
    errors = sum(1 for r in results if r["error"])
    print(f"\nDone.")
    print(f"  Kept:   {total_kept:,} valid records")
    print(f"  Shards: {len(shards)}")
    if errors:
        print(f"  Errors: {errors} shards had read/write errors", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
