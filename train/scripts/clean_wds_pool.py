#!/usr/bin/env python3
"""
train/scripts/clean_wds_pool.py — Retroactive pool cleaning script (DEDUP-1 Track 2).

Iterates over all *.tar files in the converted pool, deduplicates each one
using CLIP embeddings + a cumulative FAISS index, and rewrites the tars
in-place with duplicate records removed.  Idempotent: tars with a .deduped
sentinel are skipped on re-run.

Usage:
    train/.venv/bin/python train/scripts/clean_wds_pool.py \
        [--pool-dir PATH] [--index PATH] [--blocklist PATH] \
        [--tgz-range START END] [--threshold FLOAT] [--clip-backend STR] [--dry-run]

Defaults:
    --pool-dir   COLD_ROOT/converted/journeydb
    --index      COLD_METADATA_DIR/dedup_index.faiss
    --blocklist  COLD_METADATA_DIR/duplicate_ids.txt

The --index sidecar (.ids file) is derived from the --index path automatically
(same stem, .ids extension).
"""

import argparse
import os
import sys
import threading
import time
from pathlib import Path

_RETRY_ATTEMPTS = 3
_RETRY_DELAY    = 15  # seconds between retries on transient I/O errors

# Must be set before numpy/FAISS import on macOS to prevent libOMP crash.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_lib import (
    COLD_ROOT, COLD_METADATA_DIR,
    write_heartbeat, log_orch,
)
from clip_dedup import dedup_wds_tar, DUP_THRESHOLD


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Retroactive WDS pool deduplication (DEDUP-1 Track 2)"
    )
    ap.add_argument(
        "--pool-dir",
        default=str(COLD_ROOT / "converted" / "journeydb"),
        help="Directory containing *.tar pool files (default: COLD_ROOT/converted/journeydb)",
    )
    ap.add_argument(
        "--index",
        default=str(COLD_METADATA_DIR / "dedup_index.faiss"),
        help="Cumulative FAISS index path (default: COLD_METADATA_DIR/dedup_index.faiss)",
    )
    ap.add_argument(
        "--blocklist",
        default=str(COLD_METADATA_DIR / "duplicate_ids.txt"),
        help="Blocklist output path (default: COLD_METADATA_DIR/duplicate_ids.txt)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=DUP_THRESHOLD,
        help=f"Inner-product similarity threshold for duplicate detection (default: {DUP_THRESHOLD})",
    )
    ap.add_argument(
        "--clip-backend",
        dest="clip_backend",
        choices=("auto", "mlx", "open_clip", "transformers"),
        default="auto",
        help="CLIP backend (default: auto)",
    )
    ap.add_argument(
        "--tgz-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Process only tgz indices START..END inclusive (e.g. --tgz-range 0 49)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done but skip tar rewrite and sentinel creation",
    )
    args = ap.parse_args()

    pool_dir = Path(args.pool_dir)
    index_path = Path(args.index)
    ids_path = index_path.with_suffix(".ids")
    blocklist_path = Path(args.blocklist)

    if not pool_dir.exists():
        print(f"ERROR: pool-dir does not exist: {pool_dir}", file=sys.stderr)
        sys.exit(1)

    tars = sorted(pool_dir.glob("*.tar"))
    if args.tgz_range is not None:
        start, end = args.tgz_range
        tars = [t for t in tars
                if t.stem.isdigit() and start <= int(t.stem) <= end]
    if not tars:
        print(f"No *.tar files found in {pool_dir}")
        sys.exit(0)

    # Partition into pending (no .deduped sentinel) and already-done.
    pending = [t for t in tars if not t.with_suffix(".tar.deduped").exists()]
    already_done = len(tars) - len(pending)

    print(f"Pool: {len(tars)} tars total, {already_done} already deduped, "
          f"{len(pending)} to process")
    if not pending:
        print("All tars already deduped — nothing to do.")
        sys.exit(0)

    if args.dry_run:
        print("[dry-run] Would process:")
        for t in pending:
            print(f"  {t.name}")
        sys.exit(0)

    # Heartbeat daemon.
    _done = [0]
    _total = len(pending)
    _stop = threading.Event()

    def _heartbeat():
        while not _stop.wait(30):
            pct = round(_done[0] / _total * 100, 1) if _total else 100.0
            write_heartbeat("clean_wds_pool", None,
                            done=_done[0], total=_total, pct=pct)

    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()

    total_in = 0
    total_out = 0
    total_dups = 0
    run_start = time.time()

    def _index_size() -> int:
        """Count vectors currently in the cumulative index (lines in .ids file)."""
        try:
            return sum(1 for _ in ids_path.open() if True) if ids_path.exists() else 0
        except Exception:
            return -1

    try:
        for i, tar_path in enumerate(pending, 1):
            sentinel = tar_path.with_suffix(".tar.deduped")
            tar_size_mb = tar_path.stat().st_size / 1_048_576
            print(f"[{i}/{_total}] {tar_path.name} ({tar_size_mb:.0f} MB) ...",
                  end=" ", flush=True)
            t0 = time.time()
            last_err = None
            for attempt in range(1, _RETRY_ATTEMPTS + 1):
                try:
                    rec_in, rec_out = dedup_wds_tar(
                        tar_path=tar_path,
                        index_path=index_path,
                        ids_path=ids_path,
                        blocklist_path=blocklist_path,
                        threshold=args.threshold,
                        backend=args.clip_backend,
                    )
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if attempt < _RETRY_ATTEMPTS:
                        print(f"\n  transient error (attempt {attempt}/{_RETRY_ATTEMPTS}): {e}"
                              f" — retrying in {_RETRY_DELAY}s", flush=True)
                        time.sleep(_RETRY_DELAY)
                    else:
                        elapsed = time.time() - t0
                        print(f"FAILED ({elapsed:.0f}s): {e}", file=sys.stderr, flush=True)
                        log_orch(f"clean_wds_pool: failed {tar_path.name}: {e}", level="error")
            if last_err is not None:
                _done[0] += 1
                continue

            elapsed = time.time() - t0
            dups = rec_in - rec_out
            dup_pct = 100 * dups / rec_in if rec_in else 0.0
            total_in += rec_in
            total_out += rec_out
            total_dups += dups

            idx_sz = _index_size()
            cumulative_pct = 100 * total_dups / total_in if total_in else 0.0
            print(
                f"{rec_in:,} -> {rec_out:,}  ({dups:,} removed, {dup_pct:.1f}%)  "
                f"{elapsed:.0f}s  index={idx_sz:,}",
                flush=True,
            )
            print(
                f"  cumulative: {total_in:,} in  {total_out:,} out  "
                f"{total_dups:,} dups ({cumulative_pct:.1f}%)  "
                f"elapsed={time.time()-run_start:.0f}s",
                flush=True,
            )
            log_orch(
                f"clean_wds_pool: {tar_path.name}: {rec_in} -> {rec_out}"
                f" ({dups} removed, {dup_pct:.1f}%)  {elapsed:.0f}s  index={idx_sz}"
            )

            # Write sentinel to mark this tar as deduped.
            sentinel.touch()
            _done[0] += 1

    finally:
        _stop.set()

    total_elapsed = time.time() - run_start
    print(f"\nDone: {_done[0]}/{_total} tars processed in {total_elapsed:.0f}s")
    print(f"  Total records in:   {total_in:,}")
    print(f"  Total records out:  {total_out:,}")
    print(f"  Duplicates removed: {total_dups:,}")
    if total_in > 0:
        print(f"  Dedup rate: {100*total_dups/total_in:.1f}%")
    idx_sz = _index_size()
    if idx_sz >= 0:
        print(f"  Index size: {idx_sz:,} vectors")

    write_heartbeat("clean_wds_pool", None, done=_done[0], total=_total, pct=100)


if __name__ == "__main__":
    main()
