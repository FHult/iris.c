#!/usr/bin/env python3
"""
train/scripts/cache_inspect.py — Inspect and manage the versioned precompute cache.

Usage:
    python cache_inspect.py                         # list all versions for all encoders
    python cache_inspect.py --encoder vae           # list one encoder only
    python cache_inspect.py --clear-stale           # delete non-current versions
    python cache_inspect.py --clear-version v_a3f9c2
    python cache_inspect.py --migrate-legacy        # move flat .npz → v_legacy/ + symlink

Run from the train/scripts directory or any dir with pipeline_lib.py on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from pipeline_lib import DATA_ROOT      # noqa: E402
from cache_manager import PrecomputeCache, ENCODERS  # noqa: E402


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_BOLD  = "\033[1m"
_DIM   = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED   = "\033[31m"
_RESET = "\033[0m"


def _c(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}" if sys.stdout.isatty() else text


def _fmt_version(info: dict) -> str:
    tag   = _c(" [current]", _GREEN + _BOLD) if info["current"] else ""
    done  = _c("complete", _GREEN) if info["complete"] else _c("incomplete", _YELLOW)
    count = f"{info.get('record_count', 0):,}"
    date  = (info.get("completed_at") or info.get("created_at", "?"))[:19]
    cfg   = info.get("config", {})
    cfg_s = "  ".join(f"{k}={v}" for k, v in cfg.items())
    return f"  {_c(info['version'], _BOLD)}{tag}  {done}  {count} records  {date}  {cfg_s}"


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_list(precomp: Path, encoder: str | None) -> None:
    encs = (encoder,) if encoder else ENCODERS
    any_found = False
    for enc in encs:
        versions = PrecomputeCache.list_versions(precomp, enc)
        cur = PrecomputeCache.effective_dir(precomp, enc)
        if not versions and cur is None:
            print(f"{enc}: (no versioned cache)")
            continue
        any_found = True
        n_npz = sum(1 for f in cur.iterdir() if f.suffix == ".npz") if cur else 0
        label = f"{enc}: {len(versions)} version(s)"
        if cur:
            label += f"  →  current has {n_npz:,} records"
        print(_c(label, _BOLD))
        for v in versions:
            print(_fmt_version(v))
    if not any_found:
        print("No versioned cache found.  Run cache_inspect.py --migrate-legacy to "
              "convert an existing flat cache.")


def cmd_list_ai(precomp: Path, encoder: str | None) -> None:
    import json
    encs = (encoder,) if encoder else ENCODERS
    result = {}
    for enc in encs:
        versions = PrecomputeCache.list_versions(precomp, enc)
        cur = PrecomputeCache.effective_dir(precomp, enc)
        n_current = sum(1 for f in cur.iterdir() if f.suffix == ".npz") if cur else 0
        result[enc] = {
            "versions": versions,
            "current_dir": str(cur) if cur else None,
            "current_record_count": n_current,
        }
    print(json.dumps(result, default=str))


def cmd_clear_stale(precomp: Path, encoder: str | None) -> None:
    encs = (encoder,) if encoder else ENCODERS
    for enc in encs:
        deleted = PrecomputeCache.clear(precomp, enc, stale_only=True)
        if deleted:
            print(f"{enc}: deleted {deleted}")
        else:
            print(f"{enc}: nothing to delete")


def cmd_clear_version(precomp: Path, encoder: str | None, ver: str) -> None:
    encs = (encoder,) if encoder else ENCODERS
    for enc in encs:
        deleted = PrecomputeCache.clear(precomp, enc, version=ver)
        if deleted:
            print(f"{enc}: deleted {deleted}")
        else:
            print(f"{enc}: version {ver} not found")


def cmd_migrate(precomp: Path, encoder: str | None) -> None:
    encs = (encoder,) if encoder else ENCODERS
    for enc in encs:
        result = PrecomputeCache.migrate_legacy(precomp, enc)
        if result:
            print(f"{enc}: migrated legacy flat files → {result.name}/  (current symlink created)")
        else:
            print(f"{enc}: no flat .npz files found — skipping")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inspect and manage the versioned precompute cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--data-root", type=Path, default=DATA_ROOT,
        help=f"Pipeline data root (default: {DATA_ROOT})",
    )
    ap.add_argument(
        "--encoder", choices=ENCODERS, default=None,
        help="Restrict to one encoder (default: all)",
    )
    ap.add_argument(
        "--clear-stale", action="store_true",
        help="Delete all non-current version dirs",
    )
    ap.add_argument(
        "--clear-version", default=None, metavar="VERSION",
        help="Delete a specific version dir (e.g. v_a3f9c2)",
    )
    ap.add_argument(
        "--migrate-legacy", action="store_true",
        help="Move flat .npz files in enc_dir into v_legacy/ and create current symlink",
    )
    ap.add_argument(
        "--ai", action="store_true",
        help="Output compact JSON for AI consumption (list mode only)",
    )
    args = ap.parse_args()

    precomp = args.data_root / "precomputed"

    if args.clear_stale:
        cmd_clear_stale(precomp, args.encoder)
    elif args.clear_version:
        cmd_clear_version(precomp, args.encoder, args.clear_version)
    elif args.migrate_legacy:
        cmd_migrate(precomp, args.encoder)
    elif args.ai:
        cmd_list_ai(precomp, args.encoder)
    else:
        cmd_list(precomp, args.encoder)


if __name__ == "__main__":
    main()
