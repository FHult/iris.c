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
    write_heartbeat, log_orch, faiss_read_index_retry,
)
from clip_dedup import dedup_wds_tar, DUP_THRESHOLD


def _truncate_index(index, count: int):
    """Return an IndexFlatIP holding only the first ``count`` vectors of ``index``
    in insertion order. Used to roll a partially-indexed tar's vectors back out
    of the cumulative index (DEDUP-3). Returns ``index`` unchanged when it is None
    or already at/below ``count``."""
    import faiss
    if index is None or count >= index.ntotal:
        return index
    truncated = faiss.IndexFlatIP(index.d)
    if count > 0:
        truncated.add(index.reconstruct_n(0, count))
    return truncated


def _truncate_ids(ids_path: Path, count: int) -> None:
    """Truncate the ``.ids`` sidecar to its first ``count`` lines so it stays in
    1:1 correspondence with a truncated index (DEDUP-3)."""
    if not ids_path.exists():
        return
    lines = ids_path.read_text().splitlines()
    if len(lines) <= count:
        return
    with open(ids_path, "w") as f:
        for fid in lines[:count]:
            f.write(fid + "\n")


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

    # Load FAISS index once and keep it resident across all tars; persist every
    # _INDEX_FLUSH_EVERY tars and on exit. Avoids reloading multi-GB index per tar.
    _INDEX_FLUSH_EVERY = 10
    import faiss  # noqa: E402
    faiss.omp_set_num_threads(1)
    if index_path.exists():
        print(f"Loading FAISS index from {index_path} ...", flush=True)
        index = faiss_read_index_retry(index_path)
        print(f"  loaded: {index.ntotal:,} vectors", flush=True)
    else:
        index = None  # dedup_wds_tar will create IndexFlatIP on first call

    def _flush_index() -> None:
        if index is None:
            return
        index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = index_path.with_suffix(".faiss.tmp")
        faiss.write_index(index, str(tmp))
        with open(tmp, "rb") as _f:
            os.fsync(_f.fileno())
        os.replace(str(tmp), str(index_path))

    # DEDUP-3 recovery. If a previous run was interrupted after a tar's vectors
    # were added to the index but before the tar was marked .deduped, the
    # interrupt-time flush may have persisted those partial vectors to disk.
    # Reprocessing that tar would then search its own vectors and flag every
    # record as a self-duplicate (score ≈ 1.0 ≥ threshold). The .processing
    # marker records the index size *before* the in-flight tar's add, so we roll
    # the index (and .ids sidecar) back to that count and force the tar to be
    # reprocessed cleanly.
    processing_path = index_path.with_suffix(".processing")
    if processing_path.exists():
        lines = processing_path.read_text().splitlines()
        try:
            saved_n = int(lines[0].strip())
        except (IndexError, ValueError):
            saved_n = None
        saved_tar = lines[1].strip() if len(lines) > 1 else None
        if saved_n is not None:
            before_n = index.ntotal if index is not None else 0
            if index is not None and index.ntotal > saved_n:
                index = _truncate_index(index, saved_n)
                _flush_index()
            # Keep the sidecar aligned even when the index itself was not flushed
            # ahead (the .ids append happens before the periodic index flush).
            _truncate_ids(ids_path, saved_n)
            # Force the interrupted tar back into the pending set so its vectors
            # are re-added after the rollback.
            if saved_tar:
                stale_sentinel = pool_dir / (saved_tar + ".deduped")
                if stale_sentinel.exists():
                    stale_sentinel.unlink()
            print(f"DEDUP-3 recovery: rolled index back {before_n:,} -> {saved_n:,} "
                  f"vectors ({saved_tar or 'in-flight tar'} will be reprocessed)",
                  flush=True)
            log_orch(f"clean_wds_pool: DEDUP-3 recovery, index {before_n} -> "
                     f"{saved_n}, reprocess {saved_tar}")
        processing_path.unlink()
        # The pending set was computed before this rollback; recompute it so a
        # sentinel removed above is picked up.
        pending = [t for t in tars if not t.with_suffix(".tar.deduped").exists()]
        _total = len(pending)
        _done[0] = 0

    try:
        for i, tar_path in enumerate(pending, 1):
            sentinel = tar_path.with_suffix(".tar.deduped")
            tar_size_mb = tar_path.stat().st_size / 1_048_576
            print(f"[{i}/{_total}] {tar_path.name} ({tar_size_mb:.0f} MB) ...",
                  end=" ", flush=True)
            t0 = time.time()
            # DEDUP-3: record the index size before this tar's vectors are added.
            # If the run is interrupted after the add (and the partial vectors get
            # flushed to disk), startup recovery rolls the index back to this count
            # so the tar is not deduplicated against its own vectors on restart.
            prior_n = index.ntotal if index is not None else 0
            processing_path.write_text(f"{prior_n}\n{tar_path.name}\n")
            last_err = None
            for attempt in range(1, _RETRY_ATTEMPTS + 1):
                # Roll back any vectors a previous failed attempt added before
                # retrying, so the retry never searches this tar's own vectors.
                if index is not None and index.ntotal > prior_n:
                    index = _truncate_index(index, prior_n)
                    _truncate_ids(ids_path, prior_n)
                try:
                    rec_in, rec_out, index = dedup_wds_tar(
                        tar_path=tar_path,
                        index_path=index_path,
                        ids_path=ids_path,
                        blocklist_path=blocklist_path,
                        threshold=args.threshold,
                        backend=args.clip_backend,
                        index=index,
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
                # Drop any partial vectors the failed tar added so neither the next
                # tar nor a restart deduplicates against an un-rewritten tar.
                if index is not None and index.ntotal > prior_n:
                    index = _truncate_index(index, prior_n)
                    _truncate_ids(ids_path, prior_n)
                processing_path.unlink(missing_ok=True)
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

            # Write sentinel to mark this tar as deduped, then clear the in-flight
            # marker — this tar is now fully committed (DEDUP-3).
            sentinel.touch()
            processing_path.unlink(missing_ok=True)
            _done[0] += 1

            # Periodic FAISS index persistence (every _INDEX_FLUSH_EVERY tars).
            if _done[0] % _INDEX_FLUSH_EVERY == 0:
                _flush_index()
                print(f"  [index flushed to {index_path.name}]", flush=True)

    finally:
        try:
            _flush_index()
        except Exception as e:
            print(f"WARNING: final index flush failed: {e}", file=sys.stderr, flush=True)
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
