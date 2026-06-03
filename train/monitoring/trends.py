"""
train/monitoring/trends.py — historical metric trend store (v3.21.0).

A lightweight SQLite time-series for the signals that matter over a long
flywheel campaign: proxy fallback rate, unified-score distribution, precompute
speed, training loss, champion quality, disk/memory. The collector records
points; the doctor / alerts read trends.

Why its own store (vs reading heartbeats/logs each time): heartbeats are
point-in-time and get overwritten; logs are fixed-name and rotate. To answer
"is the fallback rate creeping up over 30 days?" you need durable history.

Pure: explicit db_path, stdlib sqlite3, no GPU. Hermetically testable.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional


def _default_db_path() -> Path:
    import os
    root = Path(os.environ.get("PIPELINE_DATA_ROOT", "/Volumes/2TBSSD"))
    return root / "monitoring.db"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _parse(ts: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(ts)
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    except (ValueError, TypeError):
        return None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS trends (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       TEXT NOT NULL,
    metric   TEXT NOT NULL,
    value    REAL,
    campaign TEXT,
    meta     TEXT
);
CREATE INDEX IF NOT EXISTS idx_trends_metric_ts ON trends(metric, ts);
"""


class TrendStore:
    """Append-only metric time-series with trend/summary queries."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._path = Path(db_path) if db_path is not None else _default_db_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ── Write ────────────────────────────────────────────────────────────────

    def record(self, metric: str, value: Optional[float],
               campaign: Optional[str] = None, ts: Optional[str] = None,
               meta: Optional[dict] = None) -> None:
        """Append one metric observation (ts defaults to now, UTC ISO)."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO trends (ts, metric, value, campaign, meta) VALUES (?,?,?,?,?)",
                (ts or _iso(_now()), metric, value, campaign,
                 json.dumps(meta) if meta else None),
            )
            self._conn.commit()

    # ── Read ─────────────────────────────────────────────────────────────────

    def history(self, metric: str, since_days: Optional[float] = 30,
                campaign: Optional[str] = None) -> list[dict]:
        """All observations of `metric` within `since_days`, oldest→newest.

        campaign=None returns all (global + per-campaign); pass a name to filter.
        since_days=None returns the full history.
        """
        q = "SELECT ts, value, campaign, meta FROM trends WHERE metric=?"
        params: list = [metric]
        if campaign is not None:
            q += " AND campaign=?"; params.append(campaign)
        q += " ORDER BY ts"
        with self._lock:
            rows = [dict(r) for r in self._conn.execute(q, params).fetchall()]

        if since_days is not None:
            cutoff = _now() - timedelta(days=since_days)
            rows = [r for r in rows
                    if (_parse(r["ts"]) or _now()) >= cutoff]
        for r in rows:
            if r.get("meta"):
                try:
                    r["meta"] = json.loads(r["meta"])
                except (TypeError, json.JSONDecodeError):
                    pass
        return rows

    def latest(self, metric: str, campaign: Optional[str] = None) -> Optional[dict]:
        q = "SELECT ts, value, campaign, meta FROM trends WHERE metric=?"
        params: list = [metric]
        if campaign is not None:
            q += " AND campaign=?"; params.append(campaign)
        q += " ORDER BY ts DESC LIMIT 1"
        with self._lock:
            row = self._conn.execute(q, params).fetchone()
        return dict(row) if row else None

    def summary(self, metric: str, since_days: Optional[float] = 30,
                campaign: Optional[str] = None) -> dict:
        """Aggregate stats for a metric over the window.

        Returns n, mean, min, max, first, last, and slope (least-squares per
        observation; >0 rising, <0 falling). Empty window → n=0, others None.
        """
        rows = self.history(metric, since_days, campaign)
        vals = [r["value"] for r in rows if r["value"] is not None]
        if not vals:
            return {"metric": metric, "n": 0, "mean": None, "min": None,
                    "max": None, "first": None, "last": None, "slope": None}
        n = len(vals)
        mean = sum(vals) / n
        # Least-squares slope over integer index (robust enough for trend sign).
        xs = list(range(n))
        xbar = sum(xs) / n
        ybar = mean
        denom = sum((x - xbar) ** 2 for x in xs)
        slope = (sum((x - xbar) * (y - ybar) for x, y in zip(xs, vals)) / denom
                 if denom else 0.0)
        return {
            "metric": metric, "n": n, "mean": round(mean, 6),
            "min": min(vals), "max": max(vals),
            "first": vals[0], "last": vals[-1], "slope": round(slope, 6),
        }

    def metrics(self) -> list[str]:
        """Distinct metric names present in the store."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT metric FROM trends ORDER BY metric").fetchall()
        return [r[0] for r in rows]
