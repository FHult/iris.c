#!/usr/bin/env python3
"""
train/scripts/shard_scorer.py — Compute per-tgz quality scores from provenance + shard_scores.db.

Reads:
  - shard_scores.db (hot or cold): per-shard effective_score from the last flywheel run
  - SHARDS_DIR/shard-NNNNNN.provenance.json: source-to-shard mappings written by build_shards.py

Writes:
  - cold_root/metadata/tgz_scores.json: per-tgz quality score (mean of contributing shard scores)

Called by orchestrator.py after mine.done (lightweight, no GPU, <1 min).

Usage:
  python train/scripts/shard_scorer.py [--config PATH] [--dry-run]
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_lib import (
    DATA_ROOT, COLD_ROOT, COLD_METADATA_DIR, SHARDS_DIR,
    SHARD_SCORES_DB_PATH, load_config, now_iso,
)


def _open_ro_db(path: Path) -> Optional[sqlite3.Connection]:
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error:
        return None


def _load_shard_scores(db_path: Path) -> dict[str, float]:
    """Return {shard_id (stem, e.g. '000042'): effective_score}."""
    conn = _open_ro_db(db_path)
    if conn is None:
        return {}
    try:
        rows = conn.execute(
            "SELECT shard_id, effective_score FROM shards WHERE effective_score IS NOT NULL"
        ).fetchall()
        # shard_id in DB is e.g. "000007" (no "shard-" prefix) — match provenance format
        return {str(r["shard_id"]): float(r["effective_score"]) for r in rows}
    except sqlite3.Error:
        return {}
    finally:
        conn.close()


def _load_provenance(shards_dir: Path) -> dict[str, list[dict]]:
    """Return {shard_stem: [source_entry, ...]} from provenance JSON sidecars."""
    result: dict[str, list[dict]] = {}
    try:
        for prov_file in shards_dir.glob("*.provenance.json"):
            try:
                data = json.loads(prov_file.read_text())
            except (ValueError, OSError):
                continue
            # shard_id field is "shard-000042" — extract stem "000042"
            shard_full = data.get("shard_id", "")
            stem = shard_full.replace("shard-", "") if shard_full else prov_file.stem.replace(".provenance", "")
            result[stem] = data.get("sources", [])
    except OSError:
        pass
    return result


def compute_tgz_scores(
    shard_scores: dict[str, float],
    provenance: dict[str, list[dict]],
) -> dict[int, dict]:
    """
    Join shard scores with provenance to compute per-tgz quality scores.

    Returns {tgz_idx: {"score": float, "n_shards": int, "shard_ids": [str]}}
    for all JDB tgzs that appear in at least one scored shard's provenance.
    """
    # tgz_idx → list of shard scores
    tgz_shard_scores: dict[int, list[float]] = defaultdict(list)
    tgz_shard_ids:    dict[int, list[str]]   = defaultdict(list)

    for shard_stem, sources in provenance.items():
        score = shard_scores.get(shard_stem)
        if score is None:
            continue  # shard not yet scored
        for src in sources:
            if src.get("type") == "jdb" and "tgz" in src:
                tgz_idx = int(src["tgz"])
                tgz_shard_scores[tgz_idx].append(score)
                tgz_shard_ids[tgz_idx].append(shard_stem)

    result: dict[int, dict] = {}
    for tgz_idx, scores in tgz_shard_scores.items():
        result[tgz_idx] = {
            "score":    round(sum(scores) / len(scores), 6),
            "n_shards": len(scores),
            "shard_ids": tgz_shard_ids[tgz_idx],
        }
    return result


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Compute per-tgz quality scores")
    ap.add_argument("--config",  default=None, help="Pipeline YAML config path")
    ap.add_argument("--dry-run", action="store_true", help="Print results without writing")
    args = ap.parse_args()

    cfg: dict = {}
    if args.config:
        try:
            cfg = load_config(args.config)
        except FileNotFoundError:
            print(f"WARNING: config not found: {args.config} — using defaults", file=sys.stderr)

    storage   = cfg.get("storage", {})
    cold_root = Path(storage.get("cold_root", COLD_ROOT))
    hot_root  = Path(storage.get("hot_root",  DATA_ROOT))
    shards_dir = Path(storage.get("shards_dir", SHARDS_DIR))

    # Use cold metadata DB first, fall back to hot
    cold_db = cold_root / "metadata" / "shard_scores.db"
    db_path = cold_db if cold_db.exists() else SHARD_SCORES_DB_PATH
    if not db_path.exists():
        print("No shard_scores.db found — run flywheel first.")
        sys.exit(0)

    print(f"shard_scorer: loading scores from {db_path}")
    shard_scores = _load_shard_scores(db_path)
    print(f"  {len(shard_scores)} scored shards")

    if not shards_dir.exists():
        print(f"WARNING: shards dir not found: {shards_dir}")
        shards_dir = hot_root / "shards"

    print(f"shard_scorer: loading provenance from {shards_dir}")
    provenance = _load_provenance(shards_dir)
    print(f"  {len(provenance)} shards with provenance sidecars")

    if not provenance:
        print("No provenance sidecars found — run build_shards.py first (sidecars written automatically).")
        sys.exit(0)

    tgz_scores = compute_tgz_scores(shard_scores, provenance)
    print(f"  {len(tgz_scores)} JDB tgzs scored")

    if not tgz_scores:
        print("No JDB tgz scores computable (no JDB provenance in scored shards).")
        sys.exit(0)

    # Sort by score desc for readability
    output = {
        "generated_at": now_iso(),
        "db_path":       str(db_path),
        "shards_dir":    str(shards_dir),
        "n_tgzs":        len(tgz_scores),
        "tgz_scores":    {str(k): v for k, v in sorted(tgz_scores.items(),
                                                         key=lambda x: x[1]["score"], reverse=True)},
    }

    if args.dry_run:
        print("\nDry-run — tgz_scores.json not written. Top 10:")
        for tgz_idx, info in sorted(tgz_scores.items(), key=lambda x: x[1]["score"], reverse=True)[:10]:
            print(f"  tgz {tgz_idx:03d}: score={info['score']:.4f}  n_shards={info['n_shards']}")
        return

    out_dir = cold_root / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tgz_scores.json"
    tmp_path = out_path.with_suffix(".json.tmp")
    try:
        tmp_path.write_text(json.dumps(output, indent=2))
        tmp_path.rename(out_path)
        print(f"shard_scorer: wrote {out_path}")
    except OSError as e:
        print(f"ERROR writing {out_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Print top-10 summary
    print("\n  Top 10 tgzs by quality score:")
    for tgz_idx, info in sorted(tgz_scores.items(), key=lambda x: x[1]["score"], reverse=True)[:10]:
        print(f"    tgz {tgz_idx:03d}: score={info['score']:.4f}  n_shards={info['n_shards']}")


if __name__ == "__main__":
    main()
