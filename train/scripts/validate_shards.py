#!/usr/bin/env python3
"""
train/scripts/validate_shards.py — Fast shard integrity scan (PIPELINE-8).

Opens each .tar in the production shards directory (header-only, no JPEG
decompression) and checks for:
  - Truncated archives (tarfile exception on getmembers)
  - Zero-byte members (partial writes)
  - Missing paired files (image without caption, or vice versa)
  - Empty archives (no members at all)

Prints a one-line summary per shard.  Any CRITICAL error exits 1.
Warnings (e.g. a small number of zero-byte members) exit 0 but are logged.

Usage:
    python train/scripts/validate_shards.py --shards /Volumes/2TBSSD/shards --chunk 1
    python train/scripts/validate_shards.py --shards /Volumes/2TBSSD/shards  # all shards
"""

import argparse
import json
import os
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_lib import (
    DATA_ROOT, LOG_DIR, SHARDS_DIR,
    mark_done, mark_error, log_orch, log_event, now_iso,
)

# Shard IDs reserved per chunk: chunk N owns [N-1)*200_000, N*200_000)
_SHARD_BLOCK = 200_000


def _chunk_shard_range(chunk: int) -> tuple[int, int]:
    return (chunk - 1) * _SHARD_BLOCK, chunk * _SHARD_BLOCK


def validate_shard(tar_path: Path) -> dict:
    """
    Header-only scan — opens the tar and reads all member metadata.
    Returns a dict with keys: path, ok, errors, warnings, n_members.
    """
    errors: list[str] = []
    warnings: list[str] = []
    n_members = 0

    try:
        with tarfile.open(tar_path) as tf:
            members = tf.getmembers()
    except tarfile.TruncatedHeaderError as e:
        return {"path": str(tar_path), "ok": False,
                "errors": [f"truncated archive: {e}"], "warnings": [], "n_members": 0}
    except Exception as e:
        return {"path": str(tar_path), "ok": False,
                "errors": [f"cannot open: {e}"], "warnings": [], "n_members": 0}

    n_members = len(members)
    if n_members == 0:
        errors.append("empty archive — no members")
    else:
        # Build stem → extensions mapping to detect unpaired files
        stem_exts: dict[str, set[str]] = {}
        for m in members:
            if not m.isfile():
                continue
            name = m.name
            # Zero-byte files (partial writes that weren't cleaned up)
            if m.size == 0:
                warnings.append(f"zero-byte member: {name}")
            stem, ext = os.path.splitext(name)
            stem_exts.setdefault(stem, set()).add(ext.lower())

        # Each record should have both an image (.jpg/.jpeg) and a caption (.txt/.json)
        img_exts  = {".jpg", ".jpeg", ".png"}
        text_exts = {".txt", ".json", ".caption"}
        for stem, exts in stem_exts.items():
            has_img  = bool(exts & img_exts)
            has_text = bool(exts & text_exts)
            if has_img and not has_text:
                warnings.append(f"image without caption: {stem}")
            elif has_text and not has_img:
                warnings.append(f"caption without image: {stem}")

    ok = len(errors) == 0
    return {"path": str(tar_path), "ok": ok,
            "errors": errors, "warnings": warnings, "n_members": n_members}


def main() -> None:
    ap = argparse.ArgumentParser(description="Shard integrity scan (PIPELINE-8)")
    ap.add_argument("--shards",  default=str(SHARDS_DIR),
                    help="Directory containing .tar shard files")
    ap.add_argument("--chunk",   type=int, default=None,
                    help="Restrict scan to shards belonging to this chunk "
                         "(by shard ID range)")
    ap.add_argument("--json",    dest="json_out", action="store_true",
                    help="Output JSON summary to stdout instead of human text")
    ap.add_argument("--ai",      action="store_true",
                    help="Emit compact JSON to stdout only; progress goes to stderr")
    ap.add_argument("--report",  default=None, metavar="PATH",
                    help="Write JSON report to this file")
    args = ap.parse_args()

    if args.ai:
        args.json_out = True  # suppress per-shard lines

    shards_dir = Path(args.shards)
    if not shards_dir.exists():
        print(f"ERROR: shards directory not found: {shards_dir}", file=sys.stderr)
        sys.exit(1)

    all_tars = sorted(shards_dir.glob("*.tar"))
    if not all_tars:
        print(f"ERROR: no .tar files in {shards_dir}", file=sys.stderr)
        sys.exit(1)

    # Filter to chunk's shard ID range if requested
    if args.chunk is not None:
        lo, hi = _chunk_shard_range(args.chunk)
        tars = [t for t in all_tars
                if t.stem.isdigit() and lo <= int(t.stem) < hi]
        if not tars:
            print(f"No shards for chunk {args.chunk} in range [{lo}, {hi})")
            sys.exit(0)
    else:
        tars = all_tars

    print(f"Scanning {len(tars)} shard(s) in {shards_dir} ...",
          file=sys.stderr if args.ai else sys.stdout, flush=True)

    results = []
    n_critical = 0
    n_warn = 0
    for i, tar_path in enumerate(tars, 1):
        r = validate_shard(tar_path)
        results.append(r)
        if not r["ok"]:
            n_critical += 1
            tag = "CRITICAL"
        elif r["warnings"]:
            n_warn += 1
            tag = f"WARN({len(r['warnings'])})"
        else:
            tag = "OK"
        if not args.json_out:
            errs_str  = ""
            if r["errors"]:
                errs_str = " | " + "; ".join(r["errors"][:2])
            warn_str = ""
            if r["warnings"] and not r["errors"]:
                warn_str = " | " + "; ".join(r["warnings"][:2])
            print(f"  [{i}/{len(tars)}] {tar_path.name}  n={r['n_members']}  {tag}{errs_str}{warn_str}",
                  flush=True)

    summary = {
        "ts":         now_iso(),
        "shards_dir": str(shards_dir),
        "chunk":      args.chunk,
        "n_scanned":  len(tars),
        "n_critical": n_critical,
        "n_warn":     n_warn,
        "results":    results,
    }

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(summary, indent=2))
        print(f"Report written → {args.report}",
              file=sys.stderr if args.ai else sys.stdout)

    if args.ai:
        corrupt_paths = [r["path"] for r in summary["results"] if not r["ok"]]
        ai_out = {
            "ok": n_critical == 0,
            "total": len(tars),
            "passed": len(tars) - n_critical,
            "failed": n_critical,
            "warnings": n_warn,
            "corrupt_paths": corrupt_paths,
        }
        print(json.dumps(ai_out))
    elif args.json_out:
        print(json.dumps(summary, indent=2))
    else:
        print(f"\nScan complete: {len(tars)} shards — "
              f"{n_critical} critical, {n_warn} warnings")

    if n_critical > 0:
        print(f"FAIL: {n_critical} shard(s) are corrupt. Investigate before training.",
              file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
