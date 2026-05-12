#!/usr/bin/env python3
"""
train/scripts/shard_selector.py — Smart Shard Selection for Flywheel mode.

Maintains persistent per-shard scores in SQLite.  After each training
iteration the caller updates scores; the selector then applies a UCB-style
policy (performance bias + diversity floor + exploration) to choose the next
shard subset.

CLI (standalone — for scanning and inspecting):
    python train/scripts/shard_selector.py scan
    python train/scripts/shard_selector.py status
    python train/scripts/shard_selector.py select --n 80
"""

import argparse
import json
import math
import os
import random
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_lib import DATA_ROOT, SHARDS_DIR, SHARD_SCORES_DB_PATH, now_iso


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS shards (
    shard_id     TEXT PRIMARY KEY,
    path         TEXT NOT NULL,
    source       TEXT NOT NULL DEFAULT 'unknown',
    n_images     INTEGER DEFAULT 0,

    -- Cumulative quality signals (mean over all runs that used this shard)
    ref_gap_mean   REAL,
    cond_gap_mean  REAL,
    loss_mean      REAL,
    n_scored       INTEGER NOT NULL DEFAULT 0,

    -- Latest-run signals
    ref_gap_last   REAL,
    cond_gap_last  REAL,
    loss_last      REAL,

    -- Composite score (recomputed on update)
    composite_score REAL,

    -- Selection bookkeeping
    n_selected        INTEGER NOT NULL DEFAULT 0,
    last_selected_at  TEXT,

    ts_created TEXT NOT NULL,
    ts_updated TEXT
);

CREATE INDEX IF NOT EXISTS idx_score ON shards(composite_score DESC);
CREATE INDEX IF NOT EXISTS idx_source ON shards(source);

CREATE TABLE IF NOT EXISTS selection_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    flywheel_name TEXT NOT NULL,
    iteration    INTEGER NOT NULL,
    n_shards     INTEGER NOT NULL,
    shard_ids    TEXT NOT NULL,   -- JSON array
    config       TEXT,            -- JSON selection config snapshot
    ts           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS score_updates (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    shard_id     TEXT NOT NULL,
    flywheel_name TEXT,
    iteration    INTEGER,
    ref_gap      REAL,
    cond_gap     REAL,
    loss         REAL,
    ts           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS _meta (k TEXT PRIMARY KEY, v TEXT);
INSERT OR IGNORE INTO _meta VALUES ('schema_version', '1');
"""


# ---------------------------------------------------------------------------
# ShardScoreDB
# ---------------------------------------------------------------------------

class ShardScoreDB:
    """Persistent per-shard quality scores. Thread-safe."""

    def __init__(self, db_path: Path = SHARD_SCORES_DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Shard registration

    def upsert_shard(self, shard_id: str, path: str,
                     source: str = "unknown", n_images: int = 0) -> None:
        ts = now_iso()
        with self._lock:
            self._conn.execute("""
                INSERT INTO shards (shard_id, path, source, n_images, ts_created)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(shard_id) DO UPDATE SET
                    path=excluded.path,
                    source=excluded.source,
                    n_images=CASE WHEN excluded.n_images > 0
                                  THEN excluded.n_images ELSE shards.n_images END,
                    ts_updated=?
            """, (shard_id, path, source, n_images, ts, ts))
            self._conn.commit()

    def shard_exists(self, shard_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM shards WHERE shard_id=?", (shard_id,)
            ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Score updates

    def update_scores(
        self,
        shard_id: str,
        ref_gap: Optional[float],
        cond_gap: Optional[float],
        loss: Optional[float],
        flywheel_name: str = "",
        iteration: int = 0,
    ) -> None:
        """Update EMA scores for a shard after a training run."""
        ts = now_iso()
        with self._lock:
            row = self._conn.execute(
                "SELECT ref_gap_mean, cond_gap_mean, loss_mean, n_scored FROM shards WHERE shard_id=?",
                (shard_id,),
            ).fetchone()
            if row is None:
                return
            n = (row["n_scored"] or 0) + 1
            # Exponential moving average (α = 1/n, converging to simple mean)
            def _ema(old, new):
                if old is None:
                    return new
                if new is None:
                    return old
                alpha = 1.0 / n
                return (1 - alpha) * old + alpha * new

            new_ref  = _ema(row["ref_gap_mean"],  ref_gap)
            new_cond = _ema(row["cond_gap_mean"], cond_gap)
            new_loss = _ema(row["loss_mean"],      loss)
            composite = _compute_composite(new_ref, new_cond, new_loss)

            self._conn.execute("""
                UPDATE shards SET
                    ref_gap_mean=?, cond_gap_mean=?, loss_mean=?,
                    ref_gap_last=?, cond_gap_last=?, loss_last=?,
                    composite_score=?,
                    n_scored=?,
                    ts_updated=?
                WHERE shard_id=?
            """, (new_ref, new_cond, new_loss,
                  ref_gap, cond_gap, loss,
                  composite, n, ts, shard_id))

            # Append raw update record
            self._conn.execute("""
                INSERT INTO score_updates (shard_id, flywheel_name, iteration,
                                           ref_gap, cond_gap, loss, ts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (shard_id, flywheel_name, iteration, ref_gap, cond_gap, loss, ts))
            self._conn.commit()

    def mark_selected(self, shard_ids: list[str]) -> None:
        ts = now_iso()
        with self._lock:
            for sid in shard_ids:
                self._conn.execute("""
                    UPDATE shards SET n_selected = n_selected + 1, last_selected_at=?
                    WHERE shard_id=?
                """, (ts, sid))
            self._conn.commit()

    def log_selection(self, flywheel_name: str, iteration: int,
                      shard_ids: list[str], cfg: dict) -> None:
        ts = now_iso()
        with self._lock:
            self._conn.execute("""
                INSERT INTO selection_log (flywheel_name, iteration, n_shards,
                                           shard_ids, config, ts)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (flywheel_name, iteration, len(shard_ids),
                  json.dumps(shard_ids), json.dumps(cfg, default=str), ts))
            self._conn.commit()

    # ------------------------------------------------------------------
    # Queries

    def get_all_shards(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM shards ORDER BY shard_id"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_scored_shards(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM shards WHERE n_scored > 0 ORDER BY composite_score DESC NULLS LAST"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_top_shards(self, n: int) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM shards WHERE composite_score IS NOT NULL "
                "ORDER BY composite_score DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        with self._lock:
            total    = self._conn.execute("SELECT COUNT(*) FROM shards").fetchone()[0]
            scored   = self._conn.execute("SELECT COUNT(*) FROM shards WHERE n_scored>0").fetchone()[0]
            best_row = self._conn.execute(
                "SELECT shard_id, composite_score FROM shards ORDER BY composite_score DESC LIMIT 1"
            ).fetchone()
            worst_row = self._conn.execute(
                "SELECT shard_id, composite_score FROM shards "
                "WHERE composite_score IS NOT NULL ORDER BY composite_score ASC LIMIT 1"
            ).fetchone()
        return {
            "total":  total,
            "scored": scored,
            "best":   dict(best_row) if best_row else None,
            "worst":  dict(worst_row) if worst_row else None,
        }

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Composite scoring formula
# ---------------------------------------------------------------------------

def _compute_composite(ref_gap: Optional[float],
                        cond_gap: Optional[float],
                        loss: Optional[float]) -> Optional[float]:
    """
    Normalize each signal to ~[0,1] and return weighted composite.
    ref_gap:  cross−self style gap.  Typical range [−0.5, +0.5].  Higher=better.
    cond_gap: null−cond loss gap.    Typical range [−3,  +0.5].  Higher=better.
    loss:     smooth training loss.  Typical range [0.3,  3.0].  Lower=better.
    """
    if ref_gap is None and cond_gap is None and loss is None:
        return None
    score = 0.0
    w_total = 0.0
    if ref_gap is not None:
        norm = (ref_gap + 0.5) / 1.0          # [−0.5,+0.5] → [0,1]
        norm = max(0.0, min(1.0, norm))
        score  += 0.55 * norm
        w_total += 0.55
    if cond_gap is not None:
        norm = (cond_gap + 3.0) / 3.5         # [−3,+0.5] → [0,1]
        norm = max(0.0, min(1.0, norm))
        score  += 0.30 * norm
        w_total += 0.30
    if loss is not None:
        norm = 1.0 - min(1.0, max(0.0, (loss - 0.3) / 2.7))  # lower loss = higher score
        score  += 0.15 * norm
        w_total += 0.15
    return score / w_total if w_total > 0 else None


# ---------------------------------------------------------------------------
# Selection algorithm
# ---------------------------------------------------------------------------

def select_shards(
    db: ShardScoreDB,
    n_shards: int,
    cfg: dict,
    flywheel_name: str = "",
    iteration: int = 0,
) -> list[str]:
    """
    Select n_shards paths for the next training iteration.

    Policy (in order of assignment):
      1. Top performers (performance_weight fraction) — deterministic top-N by score.
      2. Diversity quota (min_diversity_pct fraction) — ensures source variety.
      3. Exploration budget (exploration_rate fraction) — unseen / least-seen shards.
      4. Fill remainder from weighted random among scored shards.

    Recency penalty discounts shards selected in the last recency_window iterations.
    Unscored shards have a UCB-style optimism bonus so they eventually get explored.
    """
    performance_weight = float(cfg.get("performance_weight", 0.60))
    exploration_rate   = float(cfg.get("exploration_rate",   0.15))
    min_diversity_pct  = float(cfg.get("min_diversity_pct",  0.20))
    recency_penalty    = float(cfg.get("recency_penalty",    0.30))
    recency_window     = int(cfg.get("recency_window_iters", 3))

    all_shards = db.get_all_shards()
    if not all_shards:
        print("WARNING: no shards in DB — returning empty selection", file=sys.stderr)
        return []
    if len(all_shards) <= n_shards:
        return [s["path"] for s in all_shards]

    selected_ids: set[str] = set()
    selected_paths: list[str] = []

    def _add_shard(s: dict) -> None:
        if s["shard_id"] not in selected_ids:
            selected_ids.add(s["shard_id"])
            selected_paths.append(s["path"])

    # ------------------------------------------------------------------
    # Step 1: top performers
    n_perf = max(1, int(n_shards * performance_weight))
    scored = [s for s in all_shards if s.get("composite_score") is not None]
    scored.sort(key=lambda s: s["composite_score"], reverse=True)
    for s in scored[:n_perf]:
        _add_shard(s)

    # ------------------------------------------------------------------
    # Step 2: diversity floor — source variety
    n_div = max(0, int(n_shards * min_diversity_pct))
    if n_div > 0:
        source_counts: dict[str, int] = {}
        for s in selected_paths:
            pass  # count sources of already-selected
        for shrd in all_shards:
            if shrd["shard_id"] in selected_ids:
                src = shrd.get("source", "unknown")
                source_counts[src] = source_counts.get(src, 0) + 1

        # Pick from under-represented sources
        remaining = [s for s in all_shards if s["shard_id"] not in selected_ids]
        remaining.sort(key=lambda s: source_counts.get(s.get("source", "unknown"), 0))
        for s in remaining[:n_div]:
            _add_shard(s)

    # ------------------------------------------------------------------
    # Step 3: exploration — least-selected shards
    n_explore = max(0, int(n_shards * exploration_rate))
    if n_explore > 0:
        unexplored = [s for s in all_shards
                      if s["shard_id"] not in selected_ids and s["n_selected"] == 0]
        if unexplored:
            random.shuffle(unexplored)
            for s in unexplored[:n_explore]:
                _add_shard(s)
        else:
            # All explored: pick least-recently-selected
            remaining = [s for s in all_shards if s["shard_id"] not in selected_ids]
            remaining.sort(key=lambda s: s.get("n_selected", 0))
            for s in remaining[:n_explore]:
                _add_shard(s)

    # ------------------------------------------------------------------
    # Step 4: fill remainder with weighted random
    needed = n_shards - len(selected_paths)
    if needed > 0:
        remaining = [s for s in all_shards if s["shard_id"] not in selected_ids]
        if remaining:
            weights = []
            for s in remaining:
                w = s.get("composite_score") or 0.3   # optimism for unscored
                # Recency penalty: discount shards selected recently
                n_sel = s.get("n_selected", 0)
                if n_sel > 0 and recency_window > 0:
                    # Simple proxy: assume last selection was ≤ recency_window iters ago
                    # if n_selected is high relative to iteration count
                    penalty = min(1.0, n_sel / max(1, iteration / max(1, recency_window)))
                    w = w * (1.0 - recency_penalty * penalty)
                weights.append(max(0.001, w))
            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            picks_idx = _weighted_sample_no_replace(probs, min(needed, len(remaining)))
            for i in picks_idx:
                _add_shard(remaining[i])

    # Log the selection
    db.mark_selected(list(selected_ids))
    db.log_selection(flywheel_name, iteration, list(selected_ids), cfg)

    return selected_paths


def _weighted_sample_no_replace(probs: list[float], k: int) -> list[int]:
    """Sample k indices without replacement using given probabilities."""
    indices = list(range(len(probs)))
    chosen = []
    remaining_probs = list(probs)
    for _ in range(min(k, len(indices))):
        total = sum(remaining_probs)
        if total <= 0:
            break
        r = random.random() * total
        cum = 0.0
        for idx, p in enumerate(remaining_probs):
            cum += p
            if r <= cum:
                chosen.append(indices[idx])
                remaining_probs[idx] = 0.0
                break
    return chosen


# ---------------------------------------------------------------------------
# Shard pool scanning
# ---------------------------------------------------------------------------

def scan_shard_pool(
    db: ShardScoreDB,
    shards_dir: Path = SHARDS_DIR,
    verbose: bool = True,
) -> int:
    """Scan shards_dir and register any new shards in the DB. Returns count added."""
    if not shards_dir.exists():
        print(f"WARNING: shards dir not found: {shards_dir}", file=sys.stderr)
        return 0

    tars = sorted(shards_dir.glob("*.tar"))
    added = 0
    for tar in tars:
        shard_id = tar.stem
        if not db.shard_exists(shard_id):
            source = _infer_source(shard_id)
            db.upsert_shard(shard_id, str(tar), source=source)
            added += 1

    if verbose:
        stats = db.get_stats()
        print(f"Shard pool scan: {len(tars)} total, {added} new "
              f"→ {stats['total']} in DB ({stats['scored']} scored)")
    return added


def _infer_source(shard_id: str) -> str:
    """Infer data source from shard ID conventions."""
    try:
        n = int(shard_id)
        # IDs 0..199999 = chunk 1, 200000..399999 = chunk 2, etc.
        # Source heuristic based on typical JDB vs LAION ID spacing.
        # This is approximate — the authoritative source is the build manifest.
        if n < 400_000:
            return "journeydb"
        return "laion_coyo"
    except ValueError:
        return "unknown"


# ---------------------------------------------------------------------------
# Shard symlink staging (temp directory for a training iteration)
# ---------------------------------------------------------------------------

def stage_shards_for_iteration(
    shard_paths: list[str],
    staging_dir: Path,
) -> Path:
    """
    Create a temporary directory of symlinks pointing to the selected shards.
    Returns the staging directory path.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    # Remove stale links
    for link in staging_dir.glob("*.tar"):
        try:
            link.unlink()
        except OSError:
            pass
    # Create new links
    for path_str in shard_paths:
        src = Path(path_str)
        dst = staging_dir / src.name
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)
        except OSError as e:
            print(f"WARNING: could not symlink {src.name}: {e}", file=sys.stderr)
    return staging_dir


# ---------------------------------------------------------------------------
# HTML shard-selection report
# ---------------------------------------------------------------------------

def render_shard_report(
    db: ShardScoreDB,
    selected_ids: list[str],
    iteration: int,
    flywheel_name: str,
) -> str:
    all_shards = db.get_all_shards()
    selected_set = set(selected_ids)
    scored = sorted(
        [s for s in all_shards if s.get("composite_score") is not None],
        key=lambda s: s["composite_score"],
        reverse=True,
    )
    stats = db.get_stats()

    def _fmt(v):
        return f"{v:.4f}" if v is not None else "—"

    rows_html = ""
    for rank, s in enumerate(scored[:100]):
        sid = s["shard_id"]
        sel = "✓" if sid in selected_set else ""
        sel_col = "color:#7f7" if sel else "color:#555"
        rows_html += (
            f"<tr>"
            f"<td style='color:#aaa'>{rank+1}</td>"
            f"<td><code>{sid}</code></td>"
            f"<td style='color:#888'>{s.get('source','?')}</td>"
            f"<td style='color:#7af'>{_fmt(s.get('composite_score'))}</td>"
            f"<td>{_fmt(s.get('ref_gap_mean'))}</td>"
            f"<td>{_fmt(s.get('cond_gap_mean'))}</td>"
            f"<td>{_fmt(s.get('loss_mean'))}</td>"
            f"<td>{s.get('n_scored',0)}</td>"
            f"<td>{s.get('n_selected',0)}</td>"
            f"<td style='{sel_col}'>{sel}</td>"
            f"</tr>\n"
        )

    ts = now_iso()
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8">
<title>Shard Selection — {flywheel_name} iteration {iteration}</title>
<style>
  body{{font-family:monospace;background:#111;color:#ddd;margin:20px;line-height:1.5}}
  h1{{color:#7df}} h2{{color:#adf;font-size:1em;margin-top:20px;border-bottom:1px solid #333}}
  .meta{{color:#777;font-size:0.82em;margin-bottom:14px}}
  table{{border-collapse:collapse;font-size:0.82em;width:100%}}
  th,td{{border:1px solid #333;padding:4px 8px;text-align:left}}
  th{{background:#1c1c1c;color:#adf}} td{{background:#161616}}
  .stat{{display:inline-block;background:#1a1a2e;border:1px solid #334;
         border-radius:4px;padding:6px 12px;margin:4px;font-size:0.85em}}
  .stat span{{color:#7af;font-weight:bold}}
</style>
</head>
<body>
<h1>Shard Selection Report</h1>
<div class="meta">flywheel={flywheel_name} &nbsp;|&nbsp; iteration={iteration}
  &nbsp;|&nbsp; {ts}</div>
<div>
  <div class="stat">total shards: <span>{stats['total']}</span></div>
  <div class="stat">scored: <span>{stats['scored']}</span></div>
  <div class="stat">selected this iter: <span>{len(selected_ids)}</span></div>
</div>
<h2>Top 100 Shards by Score</h2>
<table>
  <tr><th>Rank</th><th>Shard ID</th><th>Source</th>
    <th>Score ↓</th><th>ref_gap</th><th>cond_gap</th><th>loss</th>
    <th>Runs</th><th>Selected</th><th>✓</th></tr>
  {rows_html}
</table>
</body></html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_scan(args) -> None:
    db = ShardScoreDB(Path(args.db))
    n = scan_shard_pool(db, Path(args.shards_dir), verbose=True)
    print(f"Added {n} new shards.")
    db.close()


def _cmd_status(args) -> None:
    db = ShardScoreDB(Path(args.db))
    stats = db.get_stats()
    print(f"DB: {args.db}")
    print(f"  total shards : {stats['total']}")
    print(f"  scored shards: {stats['scored']}")
    if stats["best"]:
        print(f"  best  : {stats['best']['shard_id']}  score={stats['best']['composite_score']:.4f}")
    if stats["worst"]:
        print(f"  worst : {stats['worst']['shard_id']}  score={stats['worst']['composite_score']:.4f}")
    top = db.get_top_shards(10)
    if top:
        print("\n  Top 10:")
        for i, s in enumerate(top):
            print(f"    {i+1:2d}. {s['shard_id']}  score={s['composite_score']:.4f} "
                  f"ref={s.get('ref_gap_mean') or 0:.4f}  "
                  f"n_runs={s['n_scored']}")
    db.close()


def _cmd_select(args) -> None:
    db = ShardScoreDB(Path(args.db))
    scan_shard_pool(db, Path(args.shards_dir), verbose=False)
    cfg = {
        "performance_weight": args.performance_weight,
        "exploration_rate":   args.exploration_rate,
        "min_diversity_pct":  args.min_diversity_pct,
        "recency_penalty":    args.recency_penalty,
    }
    paths = select_shards(db, args.n, cfg)
    print(f"Selected {len(paths)} shards:")
    for p in paths:
        print(f"  {p}")
    db.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Shard selection CLI")
    ap.add_argument("--db", default=str(SHARD_SCORES_DB_PATH))
    ap.add_argument("--shards-dir", default=str(SHARDS_DIR))
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("scan",   help="Scan shard pool and populate DB")

    sub.add_parser("status", help="Show scoring statistics")

    p = sub.add_parser("select", help="Run selection and print chosen paths")
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--performance-weight", type=float, default=0.60)
    p.add_argument("--exploration-rate",   type=float, default=0.15)
    p.add_argument("--min-diversity-pct",  type=float, default=0.20)
    p.add_argument("--recency-penalty",    type=float, default=0.30)

    args = ap.parse_args()
    {"scan": _cmd_scan, "status": _cmd_status, "select": _cmd_select}[args.cmd](args)


if __name__ == "__main__":
    main()
