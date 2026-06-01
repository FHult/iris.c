"""
train/scripts/shard_index.py — SQLite shard pool index for fast campaign creation.

Caches per-shard metadata (scores, decisions, source, file size) in a SQLite database
so that campaign_manager.py can build manifests in under 1 second instead of reading
1280+ JSON sidecars from a spinning HDD (16–60× faster depending on storage speed).

Index location: {cold_root}/shard_index.db

Commands:
    build    — build or incrementally update the index
    update   — alias for build (incremental by default)
    stats    — per-source summary statistics and score distributions
    validate — verify indexed shards exist on disk; report unindexed shards
    stale    — list shards whose sidecar files are newer than their index entry

Usage:
    # Initial build (or full rebuild with --force):
    train/.venv/bin/python train/scripts/shard_index.py build \\
        --shards /Volumes/16TBCold/shards

    # Incremental update after new shards or rescoring:
    train/.venv/bin/python train/scripts/shard_index.py update \\
        --shards /Volumes/16TBCold/shards

    # Summary statistics:
    train/.venv/bin/python train/scripts/shard_index.py stats

    # Consistency check (index vs. disk):
    train/.venv/bin/python train/scripts/shard_index.py validate \\
        --shards /Volumes/16TBCold/shards

    # List stale entries (sidecar newer than indexed_at):
    train/.venv/bin/python train/scripts/shard_index.py stale \\
        --shards /Volumes/16TBCold/shards
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_lib import COLD_ROOT

# Index lives alongside the shard pool on cold storage.
INDEX_PATH = COLD_ROOT / "shard_index.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS shards (
    shard_id              TEXT PRIMARY KEY,
    path                  TEXT NOT NULL,
    source                TEXT DEFAULT 'unknown',
    file_size_bytes       INTEGER,
    has_light_score       INTEGER NOT NULL DEFAULT 0,
    light_score           REAL,
    light_decision        TEXT DEFAULT 'unknown',
    has_unified           INTEGER NOT NULL DEFAULT 0,
    final_value_score     REAL,
    training_loss_normalized REAL,
    diversity_score       REAL,
    diversity_source      TEXT,
    indexed_at            REAL NOT NULL,
    tar_mtime             REAL,
    light_mtime           REAL,
    unified_mtime         REAL
);
CREATE INDEX IF NOT EXISTS idx_source ON shards(source);
CREATE INDEX IF NOT EXISTS idx_fvs    ON shards(final_value_score);
CREATE INDEX IF NOT EXISTS idx_dec    ON shards(light_decision);
CREATE INDEX IF NOT EXISTS idx_div    ON shards(diversity_score);
"""


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add new columns to existing databases as the schema evolves."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(shards)").fetchall()}
    migrations = [
        ("diversity_score",  "ALTER TABLE shards ADD COLUMN diversity_score REAL"),
        ("diversity_source", "ALTER TABLE shards ADD COLUMN diversity_source TEXT"),
    ]
    for col, stmt in migrations:
        if col not in cols:
            conn.execute(stmt)
    conn.commit()


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _open_db(path: Path = INDEX_PATH, create: bool = True) -> sqlite3.Connection:
    if not path.exists() and not create:
        raise FileNotFoundError(
            f"Shard index not found: {path}\n"
            "Run: train/.venv/bin/python train/scripts/shard_index.py build --shards <DIR>"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    if create:
        conn.executescript(_SCHEMA)
        conn.commit()
        _migrate_schema(conn)
    return conn


def _mtime(p: Path) -> Optional[float]:
    try:
        return p.stat().st_mtime
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Reading one shard
# ---------------------------------------------------------------------------

def _read_shard(tar_path: Path) -> dict:
    """Build an indexable record from a shard's .tar and sidecar files."""
    entry: dict = {
        "shard_id":          tar_path.stem,
        "path":              str(tar_path),
        "source":            "unknown",
        "file_size_bytes":   None,
        "has_light_score":   0,
        "light_score":       None,
        "light_decision":    "unknown",
        "has_unified":       0,
        "final_value_score": None,
        "training_loss_normalized": None,
        "diversity_score":   None,
        "diversity_source":  None,
        "indexed_at":        time.time(),
        "tar_mtime":         _mtime(tar_path),
        "light_mtime":       None,
        "unified_mtime":     None,
    }

    try:
        entry["file_size_bytes"] = tar_path.stat().st_size
    except OSError:
        pass

    light_p = tar_path.with_suffix(".light_scores.json")
    if light_p.exists():
        entry["light_mtime"] = _mtime(light_p)
        try:
            ld = json.loads(light_p.read_text())
            sc = ld.get("light_scores", {})
            entry["has_light_score"] = 1
            entry["source"]          = ld.get("source", "unknown")
            entry["light_score"]     = float(sc.get("combined_score", 0.0))
            entry["light_decision"]  = ld.get("decision", "unknown")
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    unified_p = tar_path.with_suffix(".unified_score.json")
    if unified_p.exists():
        entry["unified_mtime"] = _mtime(unified_p)
        try:
            ud = json.loads(unified_p.read_text())
            entry["has_unified"]       = 1
            entry["source"]            = ud.get("source", entry["source"])
            entry["final_value_score"] = float(ud.get("final_value_score", 0.0))
            entry["training_loss_normalized"] = ud.get("training_loss_normalized")
            # Diversity signal — present only after unify_scores.py v3.16.0+
            div_comp = ud.get("components", {}).get("diversity", {})
            if div_comp.get("value") is not None:
                entry["diversity_score"]  = float(div_comp["value"])
                entry["diversity_source"] = div_comp.get("source", "source_rarity")
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    # Use light_score as final_value_score when no unified score exists yet.
    if not entry["has_unified"] and entry["has_light_score"]:
        entry["final_value_score"] = entry["light_score"]

    return entry


def _needs_reindex(row: sqlite3.Row, tar_path: Path) -> bool:
    """True when any sidecar file is newer than the row's indexed_at timestamp."""
    indexed_at = row["indexed_at"] or 0.0
    for suffix, col in [(".light_scores.json", "light_mtime"),
                        (".unified_score.json", "unified_mtime")]:
        p = tar_path.with_suffix(suffix)
        existed_before = row[col] is not None
        exists_now = p.exists()
        if existed_before != exists_now:
            return True  # sidecar appeared or disappeared
        if exists_now and (_mtime(p) or 0.0) > indexed_at:
            return True  # sidecar was updated
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build(shards_dir: Path, force: bool = False, quiet: bool = False) -> int:
    """
    Build or incrementally update the shard index.

    force=True  — wipe existing data and reindex every shard.
    force=False — only index new shards and shards whose sidecars have changed.
    Returns the number of rows written.
    """
    tars = sorted(p for p in shards_dir.glob("*.tar")
                  if not p.name.endswith(".tar.tmp"))
    if not tars:
        if not quiet:
            print(f"No .tar shards found in {shards_dir}", file=sys.stderr)
        return 0

    conn = _open_db()
    if force:
        conn.execute("DELETE FROM shards")
        conn.commit()

    existing: dict[str, sqlite3.Row] = {}
    if not force:
        for row in conn.execute("SELECT * FROM shards"):
            existing[row["shard_id"]] = row

    to_index = [p for p in tars
                if p.stem not in existing or _needs_reindex(existing[p.stem], p)]

    if not quiet:
        up_to_date = len(tars) - len(to_index)
        print(f"Shard index: {len(tars)} total, "
              f"{up_to_date} up-to-date, {len(to_index)} to index")

    n_written = 0
    for i, tar_path in enumerate(to_index):
        entry = _read_shard(tar_path)
        conn.execute("""
            INSERT OR REPLACE INTO shards
            (shard_id, path, source, file_size_bytes,
             has_light_score, light_score, light_decision,
             has_unified, final_value_score, training_loss_normalized,
             diversity_score, diversity_source,
             indexed_at, tar_mtime, light_mtime, unified_mtime)
            VALUES
            (:shard_id, :path, :source, :file_size_bytes,
             :has_light_score, :light_score, :light_decision,
             :has_unified, :final_value_score, :training_loss_normalized,
             :diversity_score, :diversity_source,
             :indexed_at, :tar_mtime, :light_mtime, :unified_mtime)
        """, entry)
        n_written += 1
        if (i + 1) % 100 == 0 or (i + 1) == len(to_index):
            conn.commit()
            if not quiet:
                print(f"  [{i+1}/{len(to_index)}]", flush=True)

    conn.commit()
    conn.close()

    if not quiet and n_written:
        print(f"Written {n_written} rows → {INDEX_PATH}")
    elif not quiet:
        print("Index is already up to date.")
    return n_written


def query_all(shards_dir: Optional[Path] = None) -> list[dict]:
    """
    Return all shard entries as dicts compatible with campaign_manager format.

    Raises FileNotFoundError if the index does not exist (caller falls back
    to directory scan).
    """
    conn = _open_db(create=False)
    rows = conn.execute("""
        SELECT shard_id, path, source,
               has_light_score, light_score, light_decision,
               has_unified, final_value_score, training_loss_normalized,
               diversity_score, diversity_source,
               file_size_bytes
        FROM shards ORDER BY shard_id
    """).fetchall()
    conn.close()

    result = []
    for row in rows:
        if shards_dir is not None and Path(row["path"]).parent != shards_dir:
            continue
        result.append({
            "path":                    row["path"],
            "shard_id":               row["shard_id"],
            "source":                  row["source"] or "unknown",
            "final_value_score":       row["final_value_score"] or 0.0,
            "light_score":             row["light_score"] or 0.0,
            "training_loss_normalized": row["training_loss_normalized"],
            "diversity_score":         row["diversity_score"],
            "diversity_source":        row["diversity_source"],
            "light_decision":         row["light_decision"] or "unknown",
            "has_unified":            bool(row["has_unified"]),
            "file_size_bytes":        row["file_size_bytes"],
        })
    return result


def stale_shard_ids(shards_dir: Path) -> list[str]:
    """Return shard_ids whose index entries are out-of-date."""
    try:
        conn = _open_db(create=False)
    except FileNotFoundError:
        return []
    rows = conn.execute("SELECT * FROM shards").fetchall()
    conn.close()
    return [row["shard_id"] for row in rows
            if _needs_reindex(row, Path(row["path"]))]


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_build(args) -> None:
    shards_dir = Path(args.shards)
    if not shards_dir.exists():
        print(f"ERROR: {shards_dir} not found", file=sys.stderr)
        sys.exit(1)
    build(shards_dir, force=args.force)


def cmd_update(args) -> None:
    shards_dir = Path(args.shards)
    if not shards_dir.exists():
        print(f"ERROR: {shards_dir} not found", file=sys.stderr)
        sys.exit(1)
    build(shards_dir, force=False)


def cmd_stats(args) -> None:
    try:
        conn = _open_db(create=False)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    n_total   = conn.execute("SELECT COUNT(*) FROM shards").fetchone()[0]
    n_light   = conn.execute("SELECT COUNT(*) FROM shards WHERE has_light_score=1").fetchone()[0]
    n_unified = conn.execute("SELECT COUNT(*) FROM shards WHERE has_unified=1").fetchone()[0]
    n_keep    = conn.execute("SELECT COUNT(*) FROM shards WHERE light_decision='keep'").fetchone()[0]
    n_discard = conn.execute("SELECT COUNT(*) FROM shards WHERE light_decision='discard'").fetchone()[0]
    total_bytes = conn.execute("SELECT SUM(file_size_bytes) FROM shards").fetchone()[0] or 0

    print(f"Shard index: {INDEX_PATH}")
    print(f"  Total shards:   {n_total:,}")
    print(f"  Light scored:   {n_light:,}  ({100*n_light//max(n_total,1)}%)")
    print(f"  Unified scored: {n_unified:,}  ({100*n_unified//max(n_total,1)}%)")
    print(f"  Keep / discard: {n_keep:,} / {n_discard:,}")
    print(f"  Total pool size: {total_bytes / 1e9:.1f} GB")
    print()

    rows = conn.execute("""
        SELECT source,
               COUNT(*) as n,
               SUM(file_size_bytes) as total_bytes,
               AVG(final_value_score) as avg_fvs,
               AVG(light_score) as avg_ls,
               AVG(diversity_score) as avg_div,
               SUM(CASE WHEN diversity_source='embedding' THEN 1 ELSE 0 END) as n_emb_div
        FROM shards GROUP BY source ORDER BY n DESC
    """).fetchall()
    print(f"  {'Source':<16} {'Count':>6}  {'GB':>6}  {'Avg score':>9}  "
          f"{'Avg light':>9}  {'Avg div':>8}  {'Emb div':>8}")
    print(f"  {'-'*73}")
    for row in rows:
        src     = (row["source"] or "unknown")
        gb      = f"{(row['total_bytes'] or 0) / 1e9:.1f}"
        fvs     = f"{row['avg_fvs']:.4f}"  if row["avg_fvs"]  is not None else "—"
        ls      = f"{row['avg_ls']:.4f}"   if row["avg_ls"]   is not None else "—"
        div     = f"{row['avg_div']:.4f}"  if row["avg_div"]  is not None else "—"
        n_emb   = row["n_emb_div"] or 0
        pct_emb = f"{100*n_emb//max(row['n'],1)}%" if n_emb > 0 else "—"
        print(f"  {src:<16} {row['n']:>6}  {gb:>6}  {fvs:>9}  "
              f"{ls:>9}  {div:>8}  {pct_emb:>8}")

    # Diversity distribution across all scored shards
    div_rows = conn.execute(
        "SELECT diversity_score FROM shards WHERE diversity_score IS NOT NULL "
        "ORDER BY diversity_score"
    ).fetchall()
    if div_rows:
        vals = [r[0] for r in div_rows]
        n    = len(vals)
        print(f"\n  Diversity score distribution ({n} shards with diversity):")
        print(f"    min={vals[0]:.4f}  p25={vals[n//4]:.4f}  "
              f"med={vals[n//2]:.4f}  p75={vals[3*n//4]:.4f}  max={vals[-1]:.4f}")

    conn.close()


def cmd_validate(args) -> None:
    shards_dir = Path(args.shards)
    try:
        conn = _open_db(create=False)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    rows = conn.execute("SELECT shard_id, path FROM shards").fetchall()
    indexed_ids = {row["shard_id"] for row in rows}
    conn.close()

    n_missing = 0
    for row in rows:
        if not Path(row["path"]).exists():
            print(f"  MISSING: {row['shard_id']}  {row['path']}")
            n_missing += 1

    on_disk = set()
    if shards_dir.exists():
        on_disk = {p.stem for p in shards_dir.glob("*.tar")
                   if not p.name.endswith(".tar.tmp")}
    unindexed = on_disk - indexed_ids

    print(f"Validated {len(rows)} indexed shards  ({shards_dir})")
    print(f"  Missing from disk:  {n_missing}")
    print(f"  On disk, unindexed: {len(unindexed)}")
    if unindexed:
        print("  Run: train/scripts/shard_index.py update --shards <DIR>")
    if n_missing:
        sys.exit(1)


def cmd_stale(args) -> None:
    shards_dir = Path(args.shards)
    stale = stale_shard_ids(shards_dir)
    if not stale:
        print("Index is up to date — no stale entries.")
        return
    print(f"{len(stale)} stale entries (sidecar newer than indexed_at):")
    for sid in stale[:50]:
        print(f"  {sid}")
    if len(stale) > 50:
        print(f"  ... and {len(stale) - 50} more")
    print("Run: train/scripts/shard_index.py update --shards <DIR>")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SQLite shard pool index — fast metadata cache for campaign creation"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_b = sub.add_parser("build", help="Build or rebuild the index from shards directory")
    p_b.add_argument("--shards", required=True, metavar="DIR")
    p_b.add_argument("--force", action="store_true", help="Wipe and rebuild from scratch")

    p_u = sub.add_parser("update", help="Incremental update (only re-index stale entries)")
    p_u.add_argument("--shards", required=True, metavar="DIR")

    sub.add_parser("stats", help="Print index summary and per-source statistics")

    p_v = sub.add_parser("validate", help="Check indexed shards exist on disk")
    p_v.add_argument("--shards", required=True, metavar="DIR")

    p_s = sub.add_parser("stale", help="List shards with stale index entries")
    p_s.add_argument("--shards", required=True, metavar="DIR")

    args = parser.parse_args()
    dispatch = {
        "build":    cmd_build,
        "update":   cmd_update,
        "stats":    cmd_stats,
        "validate": cmd_validate,
        "stale":    cmd_stale,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
