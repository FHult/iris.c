"""
train/experiments/preferences.py — preference + synthetic-data signals (v3.21.0 phase 4).

Closes the data flywheel: signals about *which outputs are good* feed back into
the dataset's unified scores so future campaigns train on better-curated data.

Two signal types, one store:
  - preference: a judgement on a dataset item (shard stem or record id).
      source = "human"  — explicit operator preference
               "self"   — model/self-preference (e.g. champion's CLIP-I on it)
               "auto"   — automated proxy (aesthetic, etc.)
  - synthetic: provenance for a generated image added back into the pool, so it
      can be scored and (optionally) trained on, while staying distinguishable
      from real data (source="synthetic", with the generating experiment + prompt).

The generation step itself (sampling images from the champion model) is GPU-bound
and run when the pipeline is idle; this module is the pure store + the function
that blends preference signal into a unified score. Tested in test_preferences.py.

Pure: explicit db_path, stdlib sqlite3, no GPU.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _default_db_path() -> Path:
    import os
    root = Path(os.environ.get("PIPELINE_DATA_ROOT", "/Volumes/2TBSSD"))
    return root / "preferences.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Per-source trust weights when aggregating preference signal. Human judgement
# outweighs self/automated signal.
SOURCE_WEIGHTS = {"human": 1.0, "self": 0.5, "auto": 0.3, "synthetic": 0.0}


_SCHEMA = """
CREATE TABLE IF NOT EXISTS preferences (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    item_id   TEXT NOT NULL,          -- shard stem or record id
    source    TEXT NOT NULL,          -- human | self | auto
    value     REAL NOT NULL,          -- normalised preference in [-1, 1]
    campaign  TEXT,
    meta      TEXT
);
CREATE INDEX IF NOT EXISTS idx_pref_item ON preferences(item_id);

CREATE TABLE IF NOT EXISTS synthetic_items (
    item_id     TEXT PRIMARY KEY,     -- id of the generated image record
    ts          TEXT NOT NULL,
    experiment  TEXT,                 -- generating experiment id
    prompt      TEXT,
    quality     REAL,                 -- score that gated its inclusion
    meta        TEXT
);
"""


class PreferenceStore:
    """Durable preference + synthetic-provenance signals."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._path = Path(db_path) if db_path is not None else _default_db_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ── Preferences ──────────────────────────────────────────────────────────

    def record_preference(self, item_id: str, source: str, value: float,
                          campaign: Optional[str] = None,
                          meta: Optional[dict] = None) -> None:
        """Record a preference signal. value is clamped to [-1, 1]."""
        if source not in SOURCE_WEIGHTS:
            raise ValueError(f"source must be one of {sorted(SOURCE_WEIGHTS)}")
        value = max(-1.0, min(1.0, float(value)))
        with self._lock:
            self._conn.execute(
                "INSERT INTO preferences (ts, item_id, source, value, campaign, meta) "
                "VALUES (?,?,?,?,?,?)",
                (_now(), item_id, source, value, campaign,
                 json.dumps(meta) if meta else None),
            )
            self._conn.commit()

    def aggregate(self, item_id: str) -> dict:
        """Source-weighted mean preference for an item.

        Returns {item_id, n, score, by_source}. score is in [-1, 1]: the weighted
        mean of per-source mean values, weighted by SOURCE_WEIGHTS. Items with no
        signal return n=0, score=0.0.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT source, value FROM preferences WHERE item_id=?",
                (item_id,)).fetchall()
        if not rows:
            return {"item_id": item_id, "n": 0, "score": 0.0, "by_source": {}}

        by_source: dict[str, list[float]] = {}
        for r in rows:
            by_source.setdefault(r["source"], []).append(r["value"])

        num = den = 0.0
        src_means = {}
        for src, vals in by_source.items():
            m = sum(vals) / len(vals)
            src_means[src] = round(m, 4)
            w = SOURCE_WEIGHTS.get(src, 0.0)
            num += w * m
            den += w
        score = round(num / den, 4) if den else 0.0
        return {"item_id": item_id, "n": len(rows), "score": score,
                "by_source": src_means}

    def all_items(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT item_id FROM preferences ORDER BY item_id").fetchall()
        return [r[0] for r in rows]

    # ── Synthetic provenance ─────────────────────────────────────────────────

    def record_synthetic(self, item_id: str, experiment: Optional[str],
                         prompt: Optional[str], quality: Optional[float],
                         meta: Optional[dict] = None) -> None:
        """Register a generated image added back into the pool (idempotent upsert)."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO synthetic_items "
                "(item_id, ts, experiment, prompt, quality, meta) VALUES (?,?,?,?,?,?)",
                (item_id, _now(), experiment, prompt, quality,
                 json.dumps(meta) if meta else None),
            )
            self._conn.commit()

    def is_synthetic(self, item_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM synthetic_items WHERE item_id=?", (item_id,)).fetchone()
        return row is not None

    def synthetic_count(self) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM synthetic_items").fetchone()[0]


# ---------------------------------------------------------------------------
# Score blending (pure) — how preference influences the unified score
# ---------------------------------------------------------------------------

def blend_preference(base_score: float, preference_score: float,
                     pref_weight: float = 0.10) -> float:
    """Blend a preference signal ([-1,1]) into a unified score ([0,1]).

    new = clamp01(base + pref_weight * preference_score)

    pref_weight is the maximum shift a unanimous strong preference can apply.
    A neutral/absent preference (0.0) leaves the score unchanged. Conservative by
    default (0.10) so preference nudges ranking without overriding static quality.
    """
    new = base_score + pref_weight * preference_score
    return max(0.0, min(1.0, round(new, 6)))


def apply_preferences(unified_scores: dict, store: PreferenceStore,
                      pref_weight: float = 0.10) -> dict:
    """Return a new {item_id: score} with preference signal blended in.

    Items without preference signal are returned unchanged.
    """
    out = {}
    for item_id, base in unified_scores.items():
        agg = store.aggregate(item_id)
        out[item_id] = (blend_preference(base, agg["score"], pref_weight)
                        if agg["n"] else base)
    return out
