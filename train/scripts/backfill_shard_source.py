"""
backfill_shard_source.py — re-derive each shard's `source` from provenance.json
and fix the current data gap in both shard DBs.

Why: both source-population paths were wrong (now fixed in shard_index.py and
shard_selector.scan_shard_pool):
  - shard_index.db left every multi-source shard "unknown" (123/1280).
  - shard_scores.db stamped every shard "journeydb" via the ID-range heuristic.

The authoritative label is each shard's provenance.json (see shard_source.py):
pure shard -> single type ("journeydb"); mix -> combined tag
("coyo+laion+wikiart"). This script rewrites `source` (and, on the selector DB,
`manifest_source`) to that value.

Default is a DRY RUN. Pass --apply to write.

    train/.venv/bin/python train/scripts/backfill_shard_source.py            # dry run
    train/.venv/bin/python train/scripts/backfill_shard_source.py --apply
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_lib import COLD_ROOT, SHARD_INDEX_PATH, SHARD_SCORES_DB_PATH
from shard_source import source_for_tar

# The canonical shard pool (with provenance.json sidecars) lives on cold; the hot
# SHARDS_DIR only ever holds transiently-staged tars without provenance.
COLD_SHARDS_DIR = COLD_ROOT / "shards"


def backfill_db(db_path: Path, shards_dir: Path, set_manifest: bool, apply: bool) -> dict:
    """Re-derive source for every row in `db_path`.shards from provenance.json."""
    if not Path(db_path).exists():
        return {"db": str(db_path), "error": "missing"}
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cols = {r[1] for r in con.execute("pragma table_info(shards)")}
    rows = con.execute("SELECT shard_id, source FROM shards").fetchall()

    updates: list[tuple[str, str]] = []
    changes: Counter = Counter()
    new_dist: Counter = Counter()
    unresolved = 0
    for r in rows:
        new = source_for_tar(Path(shards_dir) / f"{r['shard_id']}.tar")
        if new is None:
            unresolved += 1
            new_dist[r["source"] or "unknown"] += 1   # leave as-is
            continue
        new_dist[new] += 1
        if new != (r["source"] or None):
            updates.append((r["shard_id"], new))
            changes[f"{r['source']} -> {new}"] += 1

    if apply and updates:
        has_manifest = set_manifest and "manifest_source" in cols
        for sid, new in updates:
            if has_manifest:
                con.execute(
                    "UPDATE shards SET source=?, manifest_source=? WHERE shard_id=?",
                    (new, new, sid))
            else:
                con.execute("UPDATE shards SET source=? WHERE shard_id=?", (new, sid))
        con.commit()
    con.close()
    return {
        "db": str(db_path), "rows": len(rows), "changed": len(updates),
        "unresolved": unresolved, "applied": bool(apply and updates),
        "top_changes": dict(changes.most_common(8)),
        "new_distribution": dict(new_dist.most_common()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", type=Path, default=COLD_SHARDS_DIR,
                    help=f"shard pool dir with provenance.json (default {COLD_SHARDS_DIR})")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = ap.parse_args(argv)

    # (db, also-set-manifest_source). The selector DB uses manifest_source as the
    # authoritative tag, so set both there; shard_index.db has only `source`.
    targets = [(SHARD_INDEX_PATH, False), (SHARD_SCORES_DB_PATH, True)]
    mode = "APPLY" if args.apply else "DRY-RUN"
    for db_path, set_manifest in targets:
        rep = backfill_db(Path(db_path), args.shards, set_manifest, args.apply)
        print(f"\n[{mode}] {rep.get('db')}")
        if rep.get("error"):
            print(f"  error: {rep['error']}")
            continue
        print(f"  rows={rep['rows']}  changed={rep['changed']}  "
              f"unresolved={rep['unresolved']}  applied={rep['applied']}")
        if rep["top_changes"]:
            print(f"  top changes: {rep['top_changes']}")
        print(f"  new distribution: {rep['new_distribution']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
