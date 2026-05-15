#!/usr/bin/env python3
"""
train/scripts/dedupe_filter.py — Quality filter + CLIP dedup of converted JDB tars.

Runs between convert and build_shards. Processes each converted WDS tar in-place:
  1. Quality filter: remove corrupt images, undersized images, and bad captions (CPU).
  2. CLIP dedup: embed survivors and remove near-duplicates against the cumulative
     FAISS index (GPU). Holds GPU_TOKEN while running — will not start while
     training is active.

Each tar gets a {tar}.deduped sentinel on completion so the step is idempotent.

Usage:
    train/.venv/bin/python train/scripts/dedupe_filter.py \
        --chunk N [--config PATH] [--conv-dir PATH] \
        [--threshold FLOAT] [--clip-backend STR] \
        [--min-size INT] [--min-caption-words INT]
"""

import argparse
import io
import os
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_lib import (
    STAGING_DIR, COLD_ROOT, TRAIN_DIR,
    write_heartbeat, log_orch, load_config,
)
from clip_dedup import dedup_wds_tar, DUP_THRESHOLD

try:
    from turbojpeg import TurboJPEG
    _HAS_TURBOJPEG = True
    _tj = TurboJPEG()
except ImportError:
    _HAS_TURBOJPEG = False
    _tj = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy src → dst atomically (tmp on dst's filesystem, then os.replace)."""
    tmp = dst.parent / (dst.name + ".copy_tmp")
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Quality filter helpers
# ---------------------------------------------------------------------------

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


def _quality_filter_tar(
    src_path: Path,
    dst_path: Path,
    min_size: int,
    min_words: int,
) -> tuple[int, int]:
    """
    Read src_path, write quality-passing records to dst_path.

    Returns (kept, removed).
    """
    kept = 0
    removed = 0

    try:
        with tarfile.open(src_path) as src_tar:
            members: dict = {}
            try:
                for m in src_tar:
                    if m.isfile():
                        members[m.name] = m
            except Exception:
                pass  # truncated tar — use what was read

            keys: dict = {}
            for name in members:
                stem, _, ext = name.rpartition(".")
                if stem not in keys:
                    keys[stem] = {}
                keys[stem][ext.lower()] = name

            kept_records = []
            for stem, exts in keys.items():
                jpg_key = exts.get("jpg") or exts.get("jpeg") or exts.get("png")
                txt_key = exts.get("txt") or exts.get("caption")
                if not jpg_key or not txt_key:
                    continue

                txt = src_tar.extractfile(members[txt_key]).read().decode(
                    "utf-8", errors="replace"
                ).strip()
                if not _is_valid_caption(txt, min_words):
                    removed += 1
                    continue

                jpg = src_tar.extractfile(members[jpg_key]).read()
                try:
                    if _HAS_TURBOJPEG:
                        w, h, _, _ = _tj.decode_header(jpg)
                    else:
                        from PIL import Image as _PilImage
                        w, h = _PilImage.open(io.BytesIO(jpg)).size
                    if w < min_size or h < min_size:
                        removed += 1
                        continue
                except Exception:
                    removed += 1
                    continue

                kept_records.append((stem, jpg, txt))
                kept += 1

    except Exception as e:
        print(f"Error reading {src_path.name}: {e}", file=sys.stderr, flush=True)
        return 0, 0

    # Write quality-passing records to dst_path
    try:
        with tarfile.open(dst_path, "w") as dst_tar:
            for stem, jpg, txt in kept_records:
                for ext, data in [("jpg", jpg), ("txt", txt.encode("utf-8"))]:
                    info = tarfile.TarInfo(name=f"{stem}.{ext}")
                    info.size = len(data)
                    dst_tar.addfile(info, io.BytesIO(data))
    except Exception as e:
        print(f"Error writing quality-filtered tar {dst_path.name}: {e}",
              file=sys.stderr, flush=True)
        dst_path.unlink(missing_ok=True)
        return 0, 0

    return kept, removed


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run_dedupe_filter(
    chunk: int,
    conv_dir: Path,
    index_path: Path,
    ids_path: Path,
    blocklist_path: Path,
    threshold: float,
    backend: str,
    min_size: int,
    min_words: int,
    pool_dir: Optional[Path] = None,
) -> None:
    tars = sorted(conv_dir.glob("*.tar"))
    total = len(tars)
    if total == 0:
        log_orch(f"Chunk {chunk}: dedupe_filter — no tars found in {conv_dir}")
        return

    log_orch(f"Chunk {chunk}: dedupe_filter — {total} tars in {conv_dir}")
    log_orch(f"  threshold={threshold}  min_size={min_size}  min_words={min_words}")
    log_orch(f"  FAISS index: {index_path}")

    done = 0
    total_in = 0
    total_quality_kept = 0
    total_after_dedup = 0
    t_hb = time.time()

    for tar_path in tars:
        real_path = tar_path.resolve()

        # Determine the canonical cold pool tar for this file (if any).
        # Sentinel lives on the pool tar so future conv pool hits skip correctly.
        # Three cases:
        #   A) symlink  (same-device): real_path IS the pool tar
        #   B) copy (diff-device): pool_dir known → pool_tar is pool_dir/name
        #   C) no pool info: sentinel stays on staging
        if real_path != tar_path:
            # Case A: staging path is a symlink; real_path is the cold pool tar
            pool_tar: Optional[Path] = real_path
        elif pool_dir is not None and (pool_dir / tar_path.name).exists():
            # Case B: staging path is a real copy; cold pool tar exists separately
            pool_tar = pool_dir / tar_path.name
        else:
            pool_tar = None  # Case C

        sentinel = Path(str(pool_tar if pool_tar is not None else real_path) + ".deduped")
        if sentinel.exists():
            done += 1
            write_heartbeat("dedupe_filter", chunk,
                            done=done, total=total,
                            pct=round(done / total * 100, 1))
            continue

        # work_path: the file we read/write in place.
        # Case A: work on cold pool tar directly (via real_path).
        # Cases B/C: work on staging copy.
        work_path = real_path
        tmp_fd, tmp_str = tempfile.mkstemp(
            dir=work_path.parent, suffix=".qf_tmp"
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_str)

        try:
            quality_kept, quality_removed = _quality_filter_tar(
                work_path, tmp_path, min_size, min_words
            )
            in_count = quality_kept + quality_removed

            if quality_kept == 0:
                # All records were quality-rejected; replace tar with empty tar
                tmp_path.unlink(missing_ok=True)
                with tarfile.open(work_path, "w"):
                    pass  # write empty tar
                # Case B: write empty tar back to cold pool atomically
                if pool_tar is not None and pool_tar != work_path:
                    _atomic_copy(work_path, pool_tar)
                sentinel.touch()
                done += 1
                total_in += in_count
                print(
                    f"  {tar_path.name}: {in_count} → 0 → 0 "
                    f"({quality_removed} quality-removed, all empty)",
                    flush=True,
                )
                continue

            # Step 2: rename quality-filtered tmp → work_path
            os.replace(tmp_path, work_path)

            # Step 3: CLIP dedup rewrites work_path in-place
            records_in, records_out = dedup_wds_tar(
                work_path, index_path, ids_path, blocklist_path,
                threshold=threshold, backend=backend,
            )
            dup_removed = records_in - records_out

            # Case B: write deduped staging tar back to cold pool atomically
            if pool_tar is not None and pool_tar != work_path:
                _atomic_copy(work_path, pool_tar)

        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            print(f"  ERROR processing {tar_path.name}: {e}",
                  file=sys.stderr, flush=True)
            raise

        # Step 4: write sentinel on cold pool tar (or staging if no pool)
        sentinel.touch()
        done += 1
        total_in += in_count
        total_quality_kept += quality_kept
        total_after_dedup += records_out

        print(
            f"  {tar_path.name}: {in_count} → {quality_kept} → {records_out} "
            f"({quality_removed} quality, {dup_removed} dups)",
            flush=True,
        )

        now = time.time()
        if now - t_hb >= 30:
            write_heartbeat("dedupe_filter", chunk,
                            done=done, total=total,
                            pct=round(done / total * 100, 1))
            t_hb = now

    write_heartbeat("dedupe_filter", chunk, done=done, total=total, pct=100)
    log_orch(
        f"Chunk {chunk}: dedupe_filter complete — "
        f"{done} tars processed, "
        f"{total_in} in → {total_quality_kept} quality → {total_after_dedup} final"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Quality filter + CLIP dedup of converted JDB tars"
    )
    ap.add_argument("--chunk",             type=int, required=True)
    ap.add_argument("--config",            default=str(TRAIN_DIR / "configs" / "v2_pipeline.yaml"))
    ap.add_argument("--conv-dir",          default=None,
                    help="Directory containing converted *.tar files "
                         "(default: STAGING_DIR/chunk{N}/converted/journeydb)")
    ap.add_argument("--cold-root",         default=None,
                    help="Cold root for FAISS index (default: from config or COLD_ROOT)")
    ap.add_argument("--threshold",         type=float, default=DUP_THRESHOLD)
    ap.add_argument("--clip-backend",      default="auto",
                    choices=["auto", "open_clip", "mlx", "transformers"])
    ap.add_argument("--min-size",          type=int, default=256)
    ap.add_argument("--min-caption-words", type=int, default=5)
    args = ap.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        config = {}

    chunk = args.chunk

    # Resolve conv_dir
    if args.conv_dir:
        conv_dir = Path(args.conv_dir)
    else:
        conv_dir = STAGING_DIR / f"chunk{chunk}" / "converted" / "journeydb"

    # Resolve cold_root → FAISS index paths + cold pool dir
    if args.cold_root:
        cold_root = Path(args.cold_root)
    else:
        storage = config.get("storage", {})
        cold_root = Path(storage.get("cold_root", str(COLD_ROOT)))

    storage = config.get("storage", {})
    index_path     = cold_root / "metadata" / "dedup_index.faiss"
    ids_path       = cold_root / "metadata" / "dedup_index.ids"
    blocklist_path = cold_root / "metadata" / "duplicate_ids.txt"

    # Cold converted pool dir — used to write the deduped tar back to cold when
    # staging holds a copy (different-device path) rather than a symlink.
    _pool_root = storage.get("converted_pool_root")
    pool_dir   = Path(_pool_root) if _pool_root else None

    log_orch(f"dedupe_filter starting for chunk {chunk}")
    log_orch(f"  conv_dir:   {conv_dir}")
    log_orch(f"  cold_root:  {cold_root}")
    log_orch(f"  pool_dir:   {pool_dir}")

    if not conv_dir.exists():
        print(f"ERROR: conv_dir does not exist: {conv_dir}", file=sys.stderr)
        sys.exit(1)

    run_dedupe_filter(
        chunk=chunk,
        conv_dir=conv_dir,
        index_path=index_path,
        ids_path=ids_path,
        blocklist_path=blocklist_path,
        threshold=args.threshold,
        backend=args.clip_backend,
        min_size=args.min_size,
        min_words=args.min_caption_words,
        pool_dir=pool_dir,
    )
    log_orch(f"dedupe_filter complete for chunk {chunk}")


if __name__ == "__main__":
    main()
