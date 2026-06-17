#!/usr/bin/env python3
"""rotate_logs.py — archive old pipeline logs (GROK-T-13).

Pipeline log/jsonl filenames are fixed (per-step / per-iter, not per-run), so old
logs persist across runs and bloat LOG_DIR. This is the non-destructive companion
to the doctor's `_check_log_disk` warning: instead of deleting, it gzips logs older
than --days and moves them into LOG_DIR/archive/, then optionally prunes archives
older than --max-archive-days.

Only top-level *.log and *.jsonl files in LOG_DIR are considered; the archive/
subdirectory itself is never rotated. "Old" is decided by file mtime.

Usage:
  rotate_logs.py                      # gzip logs older than 7 days into archive/
  rotate_logs.py --days 14            # keep two weeks of live logs
  rotate_logs.py --max-archive-days 90  # also delete archives older than 90 days
  rotate_logs.py --dry-run            # show what would happen, change nothing
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import time
from pathlib import Path

# ── bootstrap sys.path so we can import pipeline_lib ────────────────────────
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from pipeline_lib import LOG_DIR

ROTATE_SUFFIXES = (".log", ".jsonl")


def rotate_logs(log_dir: Path, days: float = 7.0, max_archive_days: float | None = None,
                dry_run: bool = False, now: float | None = None) -> dict:
    """Gzip+move logs older than `days` into log_dir/archive/.

    Returns a summary dict: {"archived": [...], "pruned": [...], "archive_dir": str}.
    `now` is injectable for tests; defaults to wall-clock time.
    """
    now = time.time() if now is None else now
    archive_dir = log_dir / "archive"
    cutoff = now - days * 86400.0
    archived: list[str] = []
    pruned: list[str] = []

    if log_dir.exists():
        for f in sorted(log_dir.iterdir()):
            if not f.is_file() or f.suffix not in ROTATE_SUFFIXES:
                continue
            try:
                if f.stat().st_mtime >= cutoff:
                    continue  # still recent — leave it live
            except OSError:
                continue
            archived.append(f.name)
            if dry_run:
                continue
            archive_dir.mkdir(parents=True, exist_ok=True)
            dest = archive_dir / (f.name + ".gz")
            # Disambiguate if an archive with this name already exists.
            i = 1
            while dest.exists():
                dest = archive_dir / f"{f.name}.{i}.gz"
                i += 1
            with open(f, "rb") as src, gzip.open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
            f.unlink()

    # Optional retention sweep over the archive itself.
    if max_archive_days is not None and archive_dir.exists():
        prune_cutoff = now - max_archive_days * 86400.0
        for f in sorted(archive_dir.iterdir()):
            if not f.is_file() or not f.name.endswith(".gz"):
                continue
            try:
                if f.stat().st_mtime >= prune_cutoff:
                    continue
            except OSError:
                continue
            pruned.append(f.name)
            if not dry_run:
                f.unlink()

    return {"archived": archived, "pruned": pruned, "archive_dir": str(archive_dir)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Archive old pipeline logs (gzip + move to logs/archive/).")
    ap.add_argument("--log-dir", type=Path, default=LOG_DIR,
                    help=f"Directory holding pipeline logs (default: {LOG_DIR})")
    ap.add_argument("--days", type=float, default=7.0,
                    help="Archive logs older than this many days (default: 7)")
    ap.add_argument("--max-archive-days", type=float, default=None,
                    help="Also delete archived .gz files older than this many days (default: keep forever)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be archived/pruned without changing anything")
    args = ap.parse_args()

    res = rotate_logs(args.log_dir, days=args.days,
                      max_archive_days=args.max_archive_days, dry_run=args.dry_run)
    tag = "[dry-run] would archive" if args.dry_run else "archived"
    print(f"{tag} {len(res['archived'])} log file(s) to {res['archive_dir']}")
    for name in res["archived"]:
        print(f"  {name}")
    if args.max_archive_days is not None:
        ptag = "[dry-run] would prune" if args.dry_run else "pruned"
        print(f"{ptag} {len(res['pruned'])} archived file(s) older than {args.max_archive_days:g} days")


if __name__ == "__main__":
    main()
