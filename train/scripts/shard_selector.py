#!/usr/bin/env python3
"""
train/scripts/shard_selector.py — Smart Shard Selection for Flywheel mode.

Maintains persistent per-shard scores in SQLite.  After each training
iteration the caller updates scores; the selector then applies a UCB-style
policy (performance bias + diversity floor + exploration) to choose the next
shard subset.

Attribution model
-----------------
Each shard maintains two running means of ref_gap/cond_gap:
  - included: mean across iterations where this shard was IN the training batch
  - excluded: mean across iterations where this shard was NOT in the batch

  attributed_ref_gap = ref_gap_incl_mean - ref_gap_excl_mean

This is an IPS (Inverse Propensity Score) estimator of marginal contribution.
It becomes reliable once both means have at least MIN_ATTR_OBS observations,
tracked via attr_confidence = min(1.0, hmean(n_included, n_excluded) / MIN_ATTR_OBS).

Selection uses effective_score = composite_score + attr_confidence * attributed_composite,
blending raw (correlation) and attributed (causal) signals as data accumulates.

Checkpoint versioning
---------------------
Every score_update row carries checkpoint_hash and checkpoint_iter so that
the full scoring history can be filtered or weighted per model epoch.

SigLIP mean embeddings
----------------------
scan_shard_pool() extracts a mean 1152-dim SigLIP embedding for each shard
(sampled from SIGLIP_SAMPLE_N records) and stores it as a BLOB.  The
diversity-floor selection step uses max-min cosine distance across these
embeddings when available, falling back to source-tag counting otherwise.

CLI (standalone — for scanning and inspecting):
    python train/scripts/shard_selector.py scan
    python train/scripts/shard_selector.py status
    python train/scripts/shard_selector.py select --n 80
    python train/scripts/shard_selector.py attribution
"""

import argparse
import json
import math
import os
import random
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_lib import DATA_ROOT, PRECOMP_DIR, SHARDS_DIR, SHARD_SCORES_DB_PATH, now_iso


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_ATTR_OBS   = 3     # min observations in BOTH included and excluded before using attribution
SIGLIP_SAMPLE_N = 50   # images to sample per shard when computing mean embedding
SIGLIP_DIM_PACKED = 576  # nibble-packed bytes per embedding
SIGLIP_DIM = 1152        # unpacked float32 dimensions


# ---------------------------------------------------------------------------
# Schema (v2)
# ---------------------------------------------------------------------------

_SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS shards (
    shard_id     TEXT PRIMARY KEY,
    path         TEXT NOT NULL,
    source       TEXT NOT NULL DEFAULT 'unknown',
    manifest_source TEXT,            -- authoritative source from shard manifest JSON
    n_images     INTEGER DEFAULT 0,

    -- Included EMA (runs where this shard WAS selected)
    ref_gap_mean   REAL,
    cond_gap_mean  REAL,
    loss_mean      REAL,
    n_scored       INTEGER NOT NULL DEFAULT 0,

    -- Latest-run signals
    ref_gap_last   REAL,
    cond_gap_last  REAL,
    loss_last      REAL,

    -- Excluded EMA (runs where this shard was NOT selected)
    n_excluded           INTEGER NOT NULL DEFAULT 0,
    ref_gap_excl_mean    REAL,
    cond_gap_excl_mean   REAL,

    -- Attribution (contrastive: incl_mean - excl_mean)
    attributed_ref_gap   REAL,
    attributed_cond_gap  REAL,
    attributed_composite REAL,
    attr_confidence      REAL,

    -- Raw composite score (incl-only; recomputed on each included update)
    composite_score REAL,

    -- Effective selection score (blend of raw and attributed)
    effective_score REAL,

    -- Selection bookkeeping
    n_selected        INTEGER NOT NULL DEFAULT 0,
    last_selected_at  TEXT,

    -- SigLIP mean embedding (SIGLIP_DIM float32, L2-normalised)
    siglip_mean_emb   BLOB,
    siglip_emb_ts     TEXT,

    -- Checkpoint version of last score update
    score_ckpt_hash   TEXT,
    score_ckpt_iter   INTEGER,

    ts_created TEXT NOT NULL,
    ts_updated TEXT
);

CREATE INDEX IF NOT EXISTS idx_score     ON shards(effective_score DESC);
CREATE INDEX IF NOT EXISTS idx_composite ON shards(composite_score DESC);
CREATE INDEX IF NOT EXISTS idx_source    ON shards(source);

CREATE TABLE IF NOT EXISTS selection_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    flywheel_name TEXT NOT NULL,
    iteration    INTEGER NOT NULL,
    n_shards     INTEGER NOT NULL,
    shard_ids    TEXT NOT NULL,      -- JSON array
    config       TEXT,               -- JSON selection config snapshot
    mean_attributed_score REAL,
    mean_raw_score        REAL,
    diversity_method      TEXT,
    n_unscored            INTEGER,
    ts           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS score_updates (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    shard_id      TEXT NOT NULL,
    flywheel_name TEXT,
    iteration     INTEGER,
    role          TEXT NOT NULL DEFAULT 'included',  -- 'included' | 'excluded'
    ref_gap       REAL,
    cond_gap      REAL,
    loss          REAL,
    checkpoint_hash  TEXT,
    checkpoint_iter  INTEGER,
    n_in_batch    INTEGER,
    ts            TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_updates_shard ON score_updates(shard_id, role);
CREATE INDEX IF NOT EXISTS idx_updates_iter  ON score_updates(flywheel_name, iteration);

CREATE TABLE IF NOT EXISTS _meta (k TEXT PRIMARY KEY, v TEXT);
INSERT OR IGNORE INTO _meta VALUES ('schema_version', '2');
"""

# Columns added in v2 (applied via ALTER TABLE on existing v1 DBs)
_V2_MIGRATIONS = [
    "ALTER TABLE shards ADD COLUMN manifest_source TEXT",
    "ALTER TABLE shards ADD COLUMN n_excluded INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE shards ADD COLUMN ref_gap_excl_mean REAL",
    "ALTER TABLE shards ADD COLUMN cond_gap_excl_mean REAL",
    "ALTER TABLE shards ADD COLUMN attributed_ref_gap REAL",
    "ALTER TABLE shards ADD COLUMN attributed_cond_gap REAL",
    "ALTER TABLE shards ADD COLUMN attributed_composite REAL",
    "ALTER TABLE shards ADD COLUMN attr_confidence REAL",
    "ALTER TABLE shards ADD COLUMN effective_score REAL",
    "ALTER TABLE shards ADD COLUMN siglip_mean_emb BLOB",
    "ALTER TABLE shards ADD COLUMN siglip_emb_ts TEXT",
    "ALTER TABLE shards ADD COLUMN score_ckpt_hash TEXT",
    "ALTER TABLE shards ADD COLUMN score_ckpt_iter INTEGER",
    "ALTER TABLE score_updates ADD COLUMN role TEXT NOT NULL DEFAULT 'included'",
    "ALTER TABLE score_updates ADD COLUMN checkpoint_hash TEXT",
    "ALTER TABLE score_updates ADD COLUMN checkpoint_iter INTEGER",
    "ALTER TABLE score_updates ADD COLUMN n_in_batch INTEGER",
    "ALTER TABLE selection_log ADD COLUMN mean_attributed_score REAL",
    "ALTER TABLE selection_log ADD COLUMN mean_raw_score REAL",
    "ALTER TABLE selection_log ADD COLUMN diversity_method TEXT",
    "ALTER TABLE selection_log ADD COLUMN n_unscored INTEGER",
]


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Apply v2 column additions to an existing v1 database. Idempotent."""
    ver = conn.execute("SELECT v FROM _meta WHERE k='schema_version'").fetchone()
    if ver and ver[0] == '2':
        return
    for stmt in _V2_MIGRATIONS:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.execute("INSERT OR REPLACE INTO _meta VALUES ('schema_version', '2')")
    conn.commit()


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
        self._conn.executescript(_SCHEMA_V2)
        _migrate_schema(self._conn)

    # ------------------------------------------------------------------
    # Shard registration

    def upsert_shard(self, shard_id: str, path: str,
                     source: str = "unknown", n_images: int = 0,
                     manifest_source: Optional[str] = None) -> None:
        ts = now_iso()
        with self._lock:
            self._conn.execute("""
                INSERT INTO shards (shard_id, path, source, manifest_source, n_images, ts_created)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(shard_id) DO UPDATE SET
                    path=excluded.path,
                    source=excluded.source,
                    manifest_source=COALESCE(excluded.manifest_source, shards.manifest_source),
                    n_images=CASE WHEN excluded.n_images > 0
                                  THEN excluded.n_images ELSE shards.n_images END,
                    ts_updated=?
            """, (shard_id, path, source, manifest_source, n_images, ts, ts))
            self._conn.commit()

    def shard_exists(self, shard_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM shards WHERE shard_id=?", (shard_id,)
            ).fetchone()
        return row is not None

    def store_siglip_embedding(self, shard_id: str, emb: np.ndarray) -> None:
        """Store L2-normalised float32 embedding blob for a shard."""
        blob = emb.astype(np.float32).tobytes()
        ts = now_iso()
        with self._lock:
            self._conn.execute("""
                UPDATE shards SET siglip_mean_emb=?, siglip_emb_ts=?, ts_updated=?
                WHERE shard_id=?
            """, (blob, ts, ts, shard_id))
            self._conn.commit()

    # ------------------------------------------------------------------
    # Score updates — included shards

    def update_scores(
        self,
        shard_id: str,
        ref_gap: Optional[float],
        cond_gap: Optional[float],
        loss: Optional[float],
        flywheel_name: str = "",
        iteration: int = 0,
        checkpoint_hash: str = "",
        checkpoint_iter: int = 0,
        n_in_batch: int = 0,
    ) -> None:
        """Update included EMA for a shard after a training run."""
        ts = now_iso()
        with self._lock:
            row = self._conn.execute(
                "SELECT ref_gap_mean, cond_gap_mean, loss_mean, n_scored FROM shards "
                "WHERE shard_id=?", (shard_id,),
            ).fetchone()
            if row is None:
                return
            n = (row["n_scored"] or 0) + 1
            new_ref  = _ema(row["ref_gap_mean"],  ref_gap,  n)
            new_cond = _ema(row["cond_gap_mean"], cond_gap, n)
            new_loss = _ema(row["loss_mean"],      loss,     n)
            raw_comp = _compute_raw_composite(new_ref, new_cond, new_loss)

            self._conn.execute("""
                UPDATE shards SET
                    ref_gap_mean=?, cond_gap_mean=?, loss_mean=?,
                    ref_gap_last=?, cond_gap_last=?, loss_last=?,
                    composite_score=?,
                    n_scored=?,
                    score_ckpt_hash=?, score_ckpt_iter=?,
                    ts_updated=?
                WHERE shard_id=?
            """, (new_ref, new_cond, new_loss,
                  ref_gap, cond_gap, loss,
                  raw_comp, n,
                  checkpoint_hash or None, checkpoint_iter or None,
                  ts, shard_id))

            self._conn.execute("""
                INSERT INTO score_updates
                    (shard_id, flywheel_name, iteration, role,
                     ref_gap, cond_gap, loss,
                     checkpoint_hash, checkpoint_iter, n_in_batch, ts)
                VALUES (?, ?, ?, 'included', ?, ?, ?, ?, ?, ?, ?)
            """, (shard_id, flywheel_name, iteration,
                  ref_gap, cond_gap, loss,
                  checkpoint_hash or None, checkpoint_iter or None,
                  n_in_batch or None, ts))
            self._conn.commit()

        # Recompute attributed + effective outside the lock (reads again)
        self._recompute_attributed([shard_id])

    # ------------------------------------------------------------------
    # Score updates — excluded shards

    def update_excluded_scores(
        self,
        all_shard_ids: list[str],
        selected_ids: set[str],
        ref_gap: Optional[float],
        cond_gap: Optional[float],
        flywheel_name: str = "",
        iteration: int = 0,
        checkpoint_hash: str = "",
        checkpoint_iter: int = 0,
        n_in_batch: int = 0,
    ) -> None:
        """
        Update excluded EMA for every shard NOT in selected_ids, then
        recompute attributed scores for all affected shards (selected + excluded).
        """
        excluded_ids = [sid for sid in all_shard_ids if sid not in selected_ids]
        ts = now_iso()
        with self._lock:
            for sid in excluded_ids:
                row = self._conn.execute(
                    "SELECT n_excluded, ref_gap_excl_mean, cond_gap_excl_mean "
                    "FROM shards WHERE shard_id=?", (sid,),
                ).fetchone()
                if row is None:
                    continue
                n = (row["n_excluded"] or 0) + 1
                new_ref_excl  = _ema(row["ref_gap_excl_mean"],  ref_gap,  n)
                new_cond_excl = _ema(row["cond_gap_excl_mean"], cond_gap, n)

                self._conn.execute("""
                    UPDATE shards SET
                        n_excluded=?,
                        ref_gap_excl_mean=?,
                        cond_gap_excl_mean=?,
                        ts_updated=?
                    WHERE shard_id=?
                """, (n, new_ref_excl, new_cond_excl, ts, sid))

                self._conn.execute("""
                    INSERT INTO score_updates
                        (shard_id, flywheel_name, iteration, role,
                         ref_gap, cond_gap,
                         checkpoint_hash, checkpoint_iter, n_in_batch, ts)
                    VALUES (?, ?, ?, 'excluded', ?, ?, ?, ?, ?, ?)
                """, (sid, flywheel_name, iteration,
                      ref_gap, cond_gap,
                      checkpoint_hash or None, checkpoint_iter or None,
                      n_in_batch or None, ts))
            self._conn.commit()

        # Recompute attributed + effective for all affected shards
        self._recompute_attributed(list(selected_ids) + excluded_ids)

    # ------------------------------------------------------------------
    # Attribution recomputation

    def _recompute_attributed(self, shard_ids: list[str]) -> None:
        """
        Recompute attributed_ref_gap, attributed_cond_gap, attributed_composite,
        attr_confidence, and effective_score for the given shards.
        Called after any score update.
        """
        ts = now_iso()
        with self._lock:
            for sid in shard_ids:
                row = self._conn.execute("""
                    SELECT ref_gap_mean, cond_gap_mean, loss_mean,
                           ref_gap_excl_mean, cond_gap_excl_mean,
                           n_scored, n_excluded,
                           composite_score
                    FROM shards WHERE shard_id=?
                """, (sid,)).fetchone()
                if row is None:
                    continue

                n_inc = row["n_scored"] or 0
                n_exc = row["n_excluded"] or 0
                raw_comp = row["composite_score"]

                # Attribution confidence: harmonic mean of observation counts,
                # normalised by MIN_ATTR_OBS, capped at 1.0
                if n_inc >= MIN_ATTR_OBS and n_exc >= MIN_ATTR_OBS:
                    hmean = 2 * n_inc * n_exc / (n_inc + n_exc)
                    conf = min(1.0, hmean / MIN_ATTR_OBS)
                else:
                    conf = 0.0

                attr_ref = attr_cond = attr_comp = None
                if (row["ref_gap_mean"] is not None and
                        row["ref_gap_excl_mean"] is not None):
                    attr_ref = row["ref_gap_mean"] - row["ref_gap_excl_mean"]
                if (row["cond_gap_mean"] is not None and
                        row["cond_gap_excl_mean"] is not None):
                    attr_cond = row["cond_gap_mean"] - row["cond_gap_excl_mean"]

                if attr_ref is not None or attr_cond is not None:
                    # Attributed composite: difference of raw composites
                    # (incl composite already stored; compute excl composite)
                    excl_comp = _compute_raw_composite(
                        row["ref_gap_excl_mean"],
                        row["cond_gap_excl_mean"],
                        None,  # loss not attributed to excluded
                    )
                    incl_comp = _compute_raw_composite(
                        row["ref_gap_mean"],
                        row["cond_gap_mean"],
                        None,
                    )
                    if incl_comp is not None and excl_comp is not None:
                        attr_comp = incl_comp - excl_comp  # ∈ (-1, 1)

                # effective_score: raw + confidence-weighted attributed boost
                if raw_comp is not None and attr_comp is not None and conf > 0:
                    eff = raw_comp + conf * attr_comp
                    eff = max(0.0, min(1.0, eff))
                else:
                    eff = raw_comp

                self._conn.execute("""
                    UPDATE shards SET
                        attributed_ref_gap=?,
                        attributed_cond_gap=?,
                        attributed_composite=?,
                        attr_confidence=?,
                        effective_score=?,
                        ts_updated=?
                    WHERE shard_id=?
                """, (attr_ref, attr_cond, attr_comp, conf if conf > 0 else None,
                      eff, ts, sid))
            self._conn.commit()

    # ------------------------------------------------------------------
    # Selection bookkeeping

    def mark_selected(self, shard_ids: list[str]) -> None:
        ts = now_iso()
        with self._lock:
            for sid in shard_ids:
                self._conn.execute("""
                    UPDATE shards SET n_selected = n_selected + 1, last_selected_at=?
                    WHERE shard_id=?
                """, (ts, sid))
            self._conn.commit()

    def log_selection(
        self,
        flywheel_name: str,
        iteration: int,
        shard_ids: list[str],
        cfg: dict,
        mean_attributed_score: Optional[float] = None,
        mean_raw_score: Optional[float] = None,
        diversity_method: str = "",
        n_unscored: int = 0,
    ) -> None:
        ts = now_iso()
        with self._lock:
            self._conn.execute("""
                INSERT INTO selection_log
                    (flywheel_name, iteration, n_shards, shard_ids, config,
                     mean_attributed_score, mean_raw_score,
                     diversity_method, n_unscored, ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (flywheel_name, iteration, len(shard_ids),
                  json.dumps(shard_ids), json.dumps(cfg, default=str),
                  mean_attributed_score, mean_raw_score,
                  diversity_method or None, n_unscored, ts))
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
                "SELECT * FROM shards WHERE n_scored > 0 "
                "ORDER BY effective_score DESC NULLS LAST"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_top_shards(self, n: int) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM shards WHERE composite_score IS NOT NULL "
                "ORDER BY effective_score DESC NULLS LAST LIMIT ?", (n,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        with self._lock:
            total  = self._conn.execute("SELECT COUNT(*) FROM shards").fetchone()[0]
            scored = self._conn.execute(
                "SELECT COUNT(*) FROM shards WHERE n_scored>0"
            ).fetchone()[0]
            attr_ready = self._conn.execute(
                "SELECT COUNT(*) FROM shards WHERE attr_confidence >= 1.0"
            ).fetchone()[0]
            best_row = self._conn.execute(
                "SELECT shard_id, effective_score FROM shards "
                "ORDER BY effective_score DESC LIMIT 1"
            ).fetchone()
            worst_row = self._conn.execute(
                "SELECT shard_id, effective_score FROM shards "
                "WHERE effective_score IS NOT NULL "
                "ORDER BY effective_score ASC LIMIT 1"
            ).fetchone()
        return {
            "total":      total,
            "scored":     scored,
            "attr_ready": attr_ready,
            "best":       dict(best_row) if best_row else None,
            "worst":      dict(worst_row) if worst_row else None,
        }

    def get_attribution_report(self) -> list[dict]:
        """
        Return per-shard attribution vs raw comparison, sorted by magnitude
        of the ranking flip (|effective - composite|), descending.
        Useful for debugging whether attribution is changing rankings.
        """
        with self._lock:
            rows = self._conn.execute("""
                SELECT shard_id, source, composite_score, effective_score,
                       attributed_composite, attr_confidence,
                       n_scored, n_excluded,
                       ref_gap_mean, ref_gap_excl_mean,
                       attributed_ref_gap
                FROM shards
                WHERE composite_score IS NOT NULL
                ORDER BY ABS(COALESCE(effective_score,0) - composite_score) DESC
            """).fetchall()
        report = []
        for r in rows:
            d = dict(r)
            raw = d.get("composite_score") or 0.0
            eff = d.get("effective_score") or raw
            d["rank_delta"] = eff - raw  # positive = attributed boosted it
            d["flip"] = (raw < 0.5) != (eff < 0.5) if raw is not None else False
            report.append(d)
        return report

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _ema(old: Optional[float], new: Optional[float], n: int) -> Optional[float]:
    """Exponential moving average with alpha=1/n (converges to simple mean)."""
    if old is None:
        return new
    if new is None:
        return old
    alpha = 1.0 / n
    return (1.0 - alpha) * old + alpha * new


def _compute_raw_composite(
    ref_gap: Optional[float],
    cond_gap: Optional[float],
    loss: Optional[float],
) -> Optional[float]:
    """
    Normalise each signal to ~[0,1] and return weighted composite.
    ref_gap:  cross−self style gap.  Typical range [−0.5, +0.5].  Higher=better.
    cond_gap: null−cond loss gap.    Typical range [−3,  +0.5].  Higher=better.
    loss:     smooth training loss.  Typical range [0.3,  3.0].  Lower=better.
    """
    if ref_gap is None and cond_gap is None and loss is None:
        return None
    score = 0.0
    w_total = 0.0
    if ref_gap is not None:
        norm = (ref_gap + 0.5) / 1.0
        norm = max(0.0, min(1.0, norm))
        score  += 0.55 * norm
        w_total += 0.55
    if cond_gap is not None:
        norm = (cond_gap + 3.0) / 3.5
        norm = max(0.0, min(1.0, norm))
        score  += 0.30 * norm
        w_total += 0.30
    if loss is not None:
        norm = 1.0 - min(1.0, max(0.0, (loss - 0.3) / 2.7))
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
      1. Top performers (performance_weight fraction) — deterministic top-N
         by effective_score (blend of raw and attributed).
      2. Diversity quota (min_diversity_pct fraction) — maximises min cosine
         distance via SigLIP embeddings; falls back to source-tag counting.
      3. Exploration budget (exploration_rate fraction) — unseen / least-seen.
      4. Fill remainder from weighted random among scored shards.

    Recency penalty discounts shards selected in the last recency_window iters.
    Unscored shards carry a UCB-style optimism bonus so they eventually get explored.
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
        selected_ids = [s["shard_id"] for s in all_shards]
        db.mark_selected(selected_ids)
        n_unscored = sum(1 for s in all_shards if s.get("n_scored", 0) == 0)
        db.log_selection(flywheel_name, iteration, selected_ids, cfg,
                         n_unscored=n_unscored, diversity_method="all")
        return [s["path"] for s in all_shards]

    selected_ids: set[str] = set()
    selected_paths: list[str] = []
    selected_shards: list[dict] = []

    def _score(s: dict) -> float:
        return s.get("effective_score") or s.get("composite_score") or 0.3

    def _add_shard(s: dict) -> None:
        if s["shard_id"] not in selected_ids:
            selected_ids.add(s["shard_id"])
            selected_paths.append(s["path"])
            selected_shards.append(s)

    # ------------------------------------------------------------------
    # Step 1: top performers by effective_score (raw + attribution blend)
    n_perf = max(1, int(n_shards * performance_weight))
    scored = [s for s in all_shards if s.get("effective_score") is not None
              or s.get("composite_score") is not None]
    scored.sort(key=_score, reverse=True)
    for s in scored[:n_perf]:
        _add_shard(s)

    # ------------------------------------------------------------------
    # Step 2: diversity floor — SigLIP max-min-distance or source-tag fallback
    n_div = max(0, int(n_shards * min_diversity_pct))
    diversity_method = "none"
    if n_div > 0:
        remaining = [s for s in all_shards if s["shard_id"] not in selected_ids]
        siglip_candidates = [s for s in remaining if s.get("siglip_mean_emb")]
        selected_embs = [
            np.frombuffer(s["siglip_mean_emb"], dtype=np.float32)
            for s in selected_shards if s.get("siglip_mean_emb")
        ]

        if len(siglip_candidates) >= n_div and len(selected_embs) > 0:
            # Max-min cosine distance selection
            diversity_method = "siglip"
            for _ in range(n_div):
                if not siglip_candidates:
                    break
                best = None
                best_dist = -1.0
                for s in siglip_candidates:
                    emb = np.frombuffer(s["siglip_mean_emb"], dtype=np.float32)
                    min_dist = min(
                        1.0 - float(np.dot(emb, sel))
                        for sel in selected_embs
                    ) if selected_embs else 1.0
                    if min_dist > best_dist:
                        best_dist = min_dist
                        best = s
                if best is None:
                    break
                _add_shard(best)
                selected_embs.append(
                    np.frombuffer(best["siglip_mean_emb"], dtype=np.float32)
                )
                siglip_candidates = [
                    s for s in siglip_candidates if s["shard_id"] != best["shard_id"]
                ]
        else:
            # Fallback: source-tag diversity (minimise source concentration)
            diversity_method = "source_tag"
            source_counts: dict[str, int] = {}
            for s in selected_shards:
                src = s.get("manifest_source") or s.get("source", "unknown")
                source_counts[src] = source_counts.get(src, 0) + 1
            remaining.sort(
                key=lambda s: source_counts.get(
                    s.get("manifest_source") or s.get("source", "unknown"), 0
                )
            )
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
            remaining = [s for s in all_shards if s["shard_id"] not in selected_ids]
            remaining.sort(key=lambda s: s.get("n_selected", 0))
            for s in remaining[:n_explore]:
                _add_shard(s)

    # ------------------------------------------------------------------
    # Step 4: fill remainder with recency-penalised weighted random
    needed = n_shards - len(selected_paths)
    if needed > 0:
        remaining = [s for s in all_shards if s["shard_id"] not in selected_ids]
        if remaining:
            weights = []
            for s in remaining:
                w = _score(s) if _score(s) > 0 else 0.3
                n_sel = s.get("n_selected", 0)
                if n_sel > 0 and recency_window > 0:
                    penalty = min(1.0, n_sel / max(1, iteration / max(1, recency_window)))
                    w = w * (1.0 - recency_penalty * penalty)
                weights.append(max(0.001, w))
            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            picks_idx = _weighted_sample_no_replace(probs, min(needed, len(remaining)))
            for i in picks_idx:
                _add_shard(remaining[i])

    # ------------------------------------------------------------------
    # Logging and bookkeeping
    n_unscored = sum(1 for sid in selected_ids
                     if next((s for s in all_shards if s["shard_id"] == sid), {})
                     .get("n_scored", 0) == 0)
    mean_raw   = _mean_field(selected_shards, "composite_score")
    mean_attr  = _mean_field(selected_shards, "effective_score")

    db.mark_selected(list(selected_ids))
    db.log_selection(
        flywheel_name, iteration, list(selected_ids), cfg,
        mean_attributed_score=mean_attr,
        mean_raw_score=mean_raw,
        diversity_method=diversity_method,
        n_unscored=n_unscored,
    )
    return selected_paths


def _mean_field(shards: list[dict], field: str) -> Optional[float]:
    vals = [s[field] for s in shards if s.get(field) is not None]
    return sum(vals) / len(vals) if vals else None


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
    siglip_dir: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    verbose: bool = True,
) -> int:
    """
    Scan shards_dir and register any new shards in the DB.
    Optionally load a shard manifest JSON for authoritative source tags.
    Optionally extract SigLIP mean embeddings for new shards.
    Returns count added.
    """
    if not shards_dir.exists():
        print(f"WARNING: shards dir not found: {shards_dir}", file=sys.stderr)
        return 0

    # Load manifest if provided: {shard_id: {source: str, ...}}
    manifest: dict[str, dict] = {}
    if manifest_path and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as e:
            print(f"WARNING: could not load shard manifest {manifest_path}: {e}",
                  file=sys.stderr)

    # Default SigLIP dir
    if siglip_dir is None:
        candidate = PRECOMP_DIR / "siglip"
        if candidate.exists():
            siglip_dir = candidate

    tars = sorted(shards_dir.glob("*.tar"))
    added = 0
    for tar in tars:
        shard_id = tar.stem
        meta = manifest.get(shard_id, {})
        manifest_src = meta.get("source") if meta else None
        if not db.shard_exists(shard_id):
            source = manifest_src or _infer_source(shard_id)
            db.upsert_shard(shard_id, str(tar), source=source,
                            manifest_source=manifest_src)
            added += 1
            # Extract SigLIP mean embedding for newly-discovered shard
            if siglip_dir:
                emb = _extract_siglip_mean(shard_id, siglip_dir)
                if emb is not None:
                    db.store_siglip_embedding(shard_id, emb)
        elif manifest_src:
            # Update manifest_source if manifest now covers this shard
            db.upsert_shard(shard_id, str(tar), manifest_source=manifest_src)

    if verbose:
        stats = db.get_stats()
        print(f"Shard pool scan: {len(tars)} total, {added} new "
              f"→ {stats['total']} in DB "
              f"({stats['scored']} scored, {stats['attr_ready']} attribution-ready)")
    return added


def _infer_source(shard_id: str) -> str:
    """Infer data source from shard ID conventions (heuristic fallback)."""
    try:
        n = int(shard_id)
        if n < 400_000:
            return "journeydb"
        return "laion_coyo"
    except ValueError:
        return "unknown"


# ---------------------------------------------------------------------------
# SigLIP mean embedding extraction
# ---------------------------------------------------------------------------

def _dequantize_4bit(q_packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    Unpack nibble-packed 4-bit signed integers and multiply by per-token scale.
    q_packed: [..., D//2] uint8
    scale:    [...,    1] float16
    Returns:  [...,    D] float32
    """
    lo = (q_packed & 0x0F).astype(np.int8)
    hi = ((q_packed >> 4) & 0x0F).astype(np.int8)
    # Sign-extend 4-bit 2s-complement: values > 7 → negative
    lo = np.where(lo > 7, lo.astype(np.int16) - 16, lo.astype(np.int16)).astype(np.int8)
    hi = np.where(hi > 7, hi.astype(np.int16) - 16, hi.astype(np.int16)).astype(np.int8)
    q = np.empty((*q_packed.shape[:-1], q_packed.shape[-1] * 2), dtype=np.float32)
    q[..., 0::2] = lo
    q[..., 1::2] = hi
    return q * scale.astype(np.float32)


def _extract_siglip_mean(
    shard_id: str,
    siglip_dir: Path,
    n_sample: int = SIGLIP_SAMPLE_N,
) -> Optional[np.ndarray]:
    """
    Load up to n_sample SigLIP records for shard_id, dequantize, mean-pool
    patches per image, then average across images.  Returns L2-normalised
    float32 array of shape [SIGLIP_DIM], or None if no records found.
    """
    # Records are named {shard_id}_{record_id:04d}.npz
    pattern = f"{shard_id}_*.npz"
    records = sorted(siglip_dir.glob(pattern))
    if not records:
        return None

    # Sample uniformly without replacement
    if len(records) > n_sample:
        step = len(records) // n_sample
        records = records[::step][:n_sample]

    image_embs = []
    for rec_path in records:
        try:
            data = np.load(str(rec_path))
            q     = data["q"]      # [N_patches, D//2] uint8
            scale = data["scale"]  # [N_patches,    1] float16
            emb = _dequantize_4bit(q, scale)   # [N_patches, D] float32
            image_embs.append(emb.mean(axis=0))  # mean-pool patches → [D]
        except Exception:
            continue

    if not image_embs:
        return None

    mean_emb = np.stack(image_embs).mean(axis=0)  # [D] float32
    norm = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm
    return mean_emb.astype(np.float32)


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
    for link in staging_dir.glob("*.tar"):
        try:
            link.unlink()
        except OSError:
            pass
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
        key=lambda s: s.get("effective_score") or s.get("composite_score") or 0,
        reverse=True,
    )
    stats = db.get_stats()

    def _fmt(v):
        return f"{v:.4f}" if v is not None else "—"

    def _conf_bar(v):
        if v is None:
            return "—"
        pct = int(v * 100)
        return f"<span style='color:{'#7f7' if pct>=100 else '#fa7'}'>{pct}%</span>"

    rows_html = ""
    for rank, s in enumerate(scored[:100]):
        sid = s["shard_id"]
        sel = "✓" if sid in selected_set else ""
        sel_col = "color:#7f7" if sel else "color:#555"
        delta = None
        if s.get("effective_score") is not None and s.get("composite_score") is not None:
            delta = s["effective_score"] - s["composite_score"]
        delta_str = (f"<span style='color:{'#7f7' if delta>=0 else '#f77'}'>"
                     f"{delta:+.4f}</span>") if delta is not None else "—"
        rows_html += (
            f"<tr>"
            f"<td style='color:#aaa'>{rank+1}</td>"
            f"<td><code>{sid}</code></td>"
            f"<td style='color:#888'>{s.get('manifest_source') or s.get('source','?')}</td>"
            f"<td style='color:#7af'>{_fmt(s.get('effective_score'))}</td>"
            f"<td>{_fmt(s.get('composite_score'))}</td>"
            f"<td>{delta_str}</td>"
            f"<td>{_conf_bar(s.get('attr_confidence'))}</td>"
            f"<td>{_fmt(s.get('ref_gap_mean'))}</td>"
            f"<td>{_fmt(s.get('ref_gap_excl_mean'))}</td>"
            f"<td>{_fmt(s.get('attributed_ref_gap'))}</td>"
            f"<td>{s.get('n_scored',0)}/{s.get('n_excluded',0)}</td>"
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
  table{{border-collapse:collapse;font-size:0.78em;width:100%}}
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
  <div class="stat">attribution-ready: <span>{stats['attr_ready']}</span></div>
  <div class="stat">selected: <span>{len(selected_ids)}</span></div>
</div>
<h2>Top 100 Shards by Effective Score</h2>
<table>
  <tr><th>Rank</th><th>Shard</th><th>Source</th>
    <th>Eff ↓</th><th>Raw</th><th>Δ (attr boost)</th><th>Attr conf</th>
    <th>ref_gap incl</th><th>ref_gap excl</th><th>attr ref_gap</th>
    <th>n incl/excl</th><th>×Sel</th><th>✓</th></tr>
  {rows_html}
</table>
</body></html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_scan(args) -> None:
    db = ShardScoreDB(Path(args.db))
    siglip_dir = Path(args.siglip_dir) if args.siglip_dir else None
    manifest   = Path(args.manifest)   if args.manifest   else None
    n = scan_shard_pool(db, Path(args.shards_dir), siglip_dir=siglip_dir,
                        manifest_path=manifest, verbose=True)
    print(f"Added {n} new shards.")
    db.close()


def _cmd_status(args) -> None:
    db = ShardScoreDB(Path(args.db))
    stats = db.get_stats()
    print(f"DB: {args.db}")
    print(f"  total shards     : {stats['total']}")
    print(f"  scored shards    : {stats['scored']}")
    print(f"  attribution-ready: {stats['attr_ready']}")
    if stats["best"]:
        print(f"  best  : {stats['best']['shard_id']}  eff={stats['best']['effective_score']:.4f}")
    if stats["worst"]:
        print(f"  worst : {stats['worst']['shard_id']}  eff={stats['worst']['effective_score']:.4f}")
    top = db.get_top_shards(10)
    if top:
        print("\n  Top 10 (by effective_score):")
        for i, s in enumerate(top):
            conf = s.get("attr_confidence") or 0.0
            print(f"    {i+1:2d}. {s['shard_id']}"
                  f"  eff={s.get('effective_score') or 0:.4f}"
                  f"  raw={s.get('composite_score') or 0:.4f}"
                  f"  attr_conf={conf:.0%}"
                  f"  n={s['n_scored']}/{s['n_excluded']}")
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


def _cmd_attribution(args) -> None:
    db = ShardScoreDB(Path(args.db))
    report = db.get_attribution_report()
    if not report:
        print("No scored shards yet.")
        db.close()
        return
    print(f"{'Shard':<18} {'Raw':>6} {'Eff':>6} {'Δ':>7} {'AttrRef':>8} "
          f"{'Conf':>6} {'n inc/exc':>10}  Flip?")
    print("-" * 80)
    for r in report[:30]:
        print(
            f"{r['shard_id']:<18}"
            f"  {(r.get('composite_score') or 0):.4f}"
            f"  {(r.get('effective_score') or 0):.4f}"
            f"  {r['rank_delta']:+.4f}"
            f"  {(r.get('attributed_ref_gap') or 0):+.4f}"
            f"  {(r.get('attr_confidence') or 0):5.0%}"
            f"  {r.get('n_scored',0):4}/{r.get('n_excluded',0):<4}"
            f"  {'FLIP' if r['flip'] else ''}"
        )
    db.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Shard selection CLI")
    ap.add_argument("--db", default=str(SHARD_SCORES_DB_PATH))
    ap.add_argument("--shards-dir", default=str(SHARDS_DIR))
    ap.add_argument("--siglip-dir", default=None,
                    help="SigLIP precomputed dir (default: DATA_ROOT/precomputed/siglip)")
    ap.add_argument("--manifest", default=None,
                    help="Shard manifest JSON {shard_id: {source: str}}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("scan",        help="Scan shard pool and populate DB")
    sub.add_parser("status",      help="Show scoring statistics")
    sub.add_parser("attribution", help="Show attribution vs raw ranking comparison")

    p = sub.add_parser("select", help="Run selection and print chosen paths")
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--performance-weight", type=float, default=0.60)
    p.add_argument("--exploration-rate",   type=float, default=0.15)
    p.add_argument("--min-diversity-pct",  type=float, default=0.20)
    p.add_argument("--recency-penalty",    type=float, default=0.30)

    args = ap.parse_args()
    {
        "scan":        _cmd_scan,
        "status":      _cmd_status,
        "select":      _cmd_select,
        "attribution": _cmd_attribution,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
