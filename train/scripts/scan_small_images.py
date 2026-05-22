#!/usr/bin/env python3
"""
train/scripts/scan_small_images.py — Scan WDS tar pool for images below a size threshold.

Two modes:

  scan     (default) — Report per-tar and aggregate counts of undersized images.
                       No files are modified.

  rewrite  — Rewrite each tar in-place, removing images below the threshold.
             Atomic: writes .tar.tmp then renames. .deduped sentinels are preserved.

Usage:
    train/.venv/bin/python train/scripts/scan_small_images.py \\
        --pool-dir /Volumes/16TBCold/converted/journeydb \\
        [--min-size 256] \\
        [--mode scan|rewrite] \\
        [--only-deduped] \\
        [--dry-run]

Defaults:
    --pool-dir   /Volumes/16TBCold/converted/journeydb
    --min-size   256
    --mode       scan
"""

import argparse
import io
import os
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_lib import COLD_ROOT

MIN_DIM_DEFAULT = 256


def _image_size(data: bytes):
    """Return (width, height) by reading the image header only (no full decode)."""
    from PIL import Image as _PIL
    img = _PIL.open(io.BytesIO(data))
    return img.size  # (width, height) — lazy, header-only for JPEG/PNG


def _scan_tar(tar_path: Path, min_dim: int) -> dict:
    """
    Read all image records from a tar and classify by size.

    Returns:
        {
            "total": int,
            "small": int,           # images below min_dim on either axis
            "small_ids": list[str], # record stems that are undersized
            "error": str|None,
        }
    """
    total = 0
    small = 0
    small_ids = []
    try:
        with tarfile.open(str(tar_path), "r") as tf:
            members = {m.name: m for m in tf.getmembers() if m.isfile()}
            keys: dict = {}
            for name in members:
                stem, _, ext = name.rpartition(".")
                if ext.lower() in ("jpg", "jpeg", "png"):
                    keys.setdefault(stem, name)

            for stem, img_name in sorted(keys.items()):
                total += 1
                try:
                    f = tf.extractfile(members[img_name])
                    if f is None:
                        continue
                    data = f.read()
                    w, h = _image_size(data)
                    if w < min_dim or h < min_dim:
                        small += 1
                        small_ids.append(stem)
                except Exception:
                    pass  # corrupt image — leave to build_shards to skip
    except Exception as e:
        return {"total": 0, "small": 0, "small_ids": [], "error": str(e)}

    return {"total": total, "small": small, "small_ids": small_ids, "error": None}


def _rewrite_tar(tar_path: Path, remove_ids: set, dry_run: bool) -> int:
    """
    Rewrite tar_path in-place removing records in remove_ids (stems).
    Returns number of records removed.
    """
    if not remove_ids:
        return 0
    if dry_run:
        return len(remove_ids)

    tmp_path = tar_path.with_suffix(".tar.tmp")
    removed = 0
    try:
        with tarfile.open(str(tar_path), "r") as src, \
             tarfile.open(str(tmp_path), "w") as dst:
            for member in src:
                if not member.isfile():
                    continue
                stem, _, _ = member.name.rpartition(".")
                if stem in remove_ids:
                    removed += 1
                    continue
                f = src.extractfile(member)
                if f is not None:
                    dst.addfile(member, f)
        os.replace(str(tmp_path), str(tar_path))
    except Exception as e:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise RuntimeError(f"rewrite failed for {tar_path.name}: {e}") from e
    return removed


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Scan WDS tar pool for undersized images"
    )
    ap.add_argument(
        "--pool-dir",
        default=str(COLD_ROOT / "converted" / "journeydb"),
        help="Directory containing *.tar pool files",
    )
    ap.add_argument(
        "--min-size",
        type=int,
        default=MIN_DIM_DEFAULT,
        metavar="PX",
        help=f"Minimum image dimension in pixels (default: {MIN_DIM_DEFAULT})",
    )
    ap.add_argument(
        "--mode",
        choices=("scan", "rewrite"),
        default="scan",
        help="scan: report only; rewrite: remove undersized images from tars",
    )
    ap.add_argument(
        "--only-deduped",
        action="store_true",
        help="Only process tars that have a .deduped sentinel",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="(rewrite mode) Print what would be removed but do not modify tars",
    )
    ap.add_argument(
        "--tgz-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Only process tars in this numeric range (inclusive)",
    )
    args = ap.parse_args()

    pool_dir = Path(args.pool_dir)
    if not pool_dir.exists():
        print(f"ERROR: pool-dir does not exist: {pool_dir}", file=sys.stderr)
        sys.exit(1)

    tars = sorted(pool_dir.glob("*.tar"))
    if not tars:
        print(f"No *.tar files found in {pool_dir}")
        sys.exit(0)

    if args.tgz_range:
        lo, hi = args.tgz_range
        tars = [t for t in tars if lo <= int(t.stem) <= hi]
        print(f"Filtering to tgz range {lo:03d}–{hi:03d}: {len(tars)} tars")

    if args.only_deduped:
        tars = [t for t in tars if t.with_suffix(".tar.deduped").exists()]
        print(f"Filtering to deduped tars: {len(tars)} found")

    min_dim = args.min_size
    mode    = args.mode

    print(f"Pool : {pool_dir}")
    print(f"Mode : {mode}" + (" [dry-run]" if args.dry_run and mode == "rewrite" else ""))
    print(f"Tars : {len(tars)}")
    print(f"Threshold: < {min_dim}px on any axis")
    print()

    # Header
    if mode == "rewrite":
        print(f"{'TAR':<30}  {'TOTAL':>7}  {'SMALL':>7}  {'PCT':>6}  {'ACTION'}")
    else:
        print(f"{'TAR':<30}  {'TOTAL':>7}  {'SMALL':>7}  {'PCT':>6}")
    print("-" * 70)

    total_images = 0
    total_small  = 0
    tars_with_small = 0
    errors = []

    for i, tar_path in enumerate(tars, 1):
        result = _scan_tar(tar_path, min_dim)

        if result["error"]:
            errors.append((tar_path.name, result["error"]))
            print(f"{tar_path.name:<30}  ERROR: {result['error']}", flush=True)
            continue

        n_total = result["total"]
        n_small = result["small"]
        pct     = 100 * n_small / n_total if n_total else 0.0

        total_images += n_total
        total_small  += n_small
        if n_small:
            tars_with_small += 1

        if mode == "rewrite" and n_small:
            removed = _rewrite_tar(
                tar_path,
                set(result["small_ids"]),
                dry_run=args.dry_run,
            )
            action = f"{'[dry] ' if args.dry_run else ''}removed {removed}"
            print(f"{tar_path.name:<30}  {n_total:>7,}  {n_small:>7,}  {pct:>5.1f}%  {action}",
                  flush=True)
        elif n_small or mode == "scan":
            # In scan mode print all tars; in rewrite mode only print tars with small images.
            print(f"{tar_path.name:<30}  {n_total:>7,}  {n_small:>7,}  {pct:>5.1f}%",
                  flush=True)

        if i % 10 == 0 or i == len(tars):
            print(f"  [{i}/{len(tars)}] running: {total_images:,} images, "
                  f"{total_small:,} small ({100*total_small/total_images:.1f}% so far)"
                  if total_images else f"  [{i}/{len(tars)}]",
                  flush=True)

    print()
    print("=" * 70)
    print(f"Total tars scanned : {len(tars)}")
    print(f"Tars with small    : {tars_with_small}")
    print(f"Total images       : {total_images:,}")
    print(f"Undersized images  : {total_small:,}  ({100*total_small/total_images:.2f}%)"
          if total_images else "Undersized images  : 0")
    if mode == "rewrite":
        verb = "Would remove" if args.dry_run else "Removed"
        print(f"{verb}              : {total_small:,} images")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for name, msg in errors:
            print(f"  {name}: {msg}")


if __name__ == "__main__":
    main()
