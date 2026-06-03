"""
train/experiments/registry.py — Experiment registry (v3.21.0).

A trained IP-Adapter model produced by a campaign is an *experiment*. The
registry is the durable record of every experiment: which campaign/manifest
produced it, the proxy VAE settings + observed fallback rate, the training
hyperparameters and metrics, the golden-set evaluation results, the final
weights path, and a Champion/Challenger ranking.

This is the backbone the rest of v3.21.0 attaches to:
  - golden-set eval writes its metrics here (attach_golden);
  - monitoring reads trends from here;
  - the Champion is the model promoted for serving / the next warm-start.

Design mirrors FlywheelDB: a single SQLite file with an explicit db_path so it
is trivially testable in isolation and never collides with live state. Pure —
no GPU, no network, stdlib sqlite3 only.

Status lifecycle:
    registered → evaluated → (champion | challenger) → superseded
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# Default location alongside the other pipeline DBs on the hot volume.
def _default_db_path() -> Path:
    import os
    root = Path(os.environ.get("PIPELINE_DATA_ROOT", "/Volumes/2TBSSD"))
    return root / "experiments.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Golden-set metrics that get their own indexed columns for fast ranking.
# Higher-is-better for all except lpips/fid (lower-is-better) — see _BETTER.
GOLDEN_METRICS = ("clip_i", "clip_t", "aesthetic", "lpips", "fid")
_LOWER_IS_BETTER = {"lpips", "fid"}


def metric_is_better(metric: str, a: float, b: float) -> bool:
    """True if value a is better than value b for the given metric."""
    if metric in _LOWER_IS_BETTER:
        return a < b
    return a > b


_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id              TEXT PRIMARY KEY,        -- exp_0001
    created_ts      TEXT NOT NULL,
    campaign        TEXT,
    manifest        TEXT,
    git_sha         TEXT,
    weights_path    TEXT,
    -- proxy VAE provenance
    proxy_enabled   INTEGER DEFAULT 0,
    proxy_mode      TEXT,
    proxy_fallback_rate REAL,
    -- training
    hyperparams     TEXT,                    -- JSON
    train_loss      REAL,
    cond_gap        REAL,
    ref_gap         REAL,
    total_steps     INTEGER,
    -- golden-set evaluation (full blob + indexed headline metrics)
    golden_results  TEXT,                    -- JSON: {arm: {metric: value}}
    golden_clip_i   REAL,
    golden_clip_t   REAL,
    golden_aesthetic REAL,
    golden_lpips    REAL,
    golden_fid      REAL,
    -- lifecycle
    status          TEXT DEFAULT 'registered',
    notes           TEXT
);
"""


class ExperimentRegistry:
    """SQLite-backed experiment store with Champion/Challenger ranking."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._path = Path(db_path) if db_path is not None else _default_db_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ── ID allocation ────────────────────────────────────────────────────────

    def _next_id(self) -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM experiments ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if not row:
            return "exp_0001"
        try:
            n = int(row["id"].split("_")[1])
        except (IndexError, ValueError):
            n = 0
        return f"exp_{n + 1:04d}"

    # ── Create / update ──────────────────────────────────────────────────────

    def register(
        self,
        campaign: str,
        weights_path: Optional[str] = None,
        manifest: Optional[str] = None,
        git_sha: str = "",
        proxy_enabled: bool = False,
        proxy_mode: Optional[str] = None,
        proxy_fallback_rate: Optional[float] = None,
        hyperparams: Optional[dict] = None,
        train_loss: Optional[float] = None,
        cond_gap: Optional[float] = None,
        ref_gap: Optional[float] = None,
        total_steps: Optional[int] = None,
        notes: str = "",
    ) -> str:
        """Insert a new experiment record; returns its id (exp_NNNN)."""
        eid = self._next_id()
        with self._lock:
            self._conn.execute(
                """INSERT INTO experiments
                   (id, created_ts, campaign, manifest, git_sha, weights_path,
                    proxy_enabled, proxy_mode, proxy_fallback_rate,
                    hyperparams, train_loss, cond_gap, ref_gap, total_steps,
                    status, notes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (eid, _now(), campaign, manifest, git_sha, weights_path,
                 1 if proxy_enabled else 0, proxy_mode, proxy_fallback_rate,
                 json.dumps(hyperparams or {}), train_loss, cond_gap, ref_gap,
                 total_steps, "registered", notes),
            )
            self._conn.commit()
        return eid

    def attach_golden(self, exp_id: str, arm_results: dict,
                      headline_arm: str = "proxy_fallback") -> None:
        """Attach golden-set evaluation results to an experiment.

        arm_results: {arm_name: {metric: value, ...}} where arm_name is one of
        real / proxy_fallback / proxy_forced and metric ∈ GOLDEN_METRICS (plus
        any extras). The headline_arm's metrics populate the indexed columns
        used for Champion ranking (default: the production proxy-with-fallback arm).
        """
        head = arm_results.get(headline_arm) or {}
        with self._lock:
            self._conn.execute(
                """UPDATE experiments SET
                     golden_results=?, golden_clip_i=?, golden_clip_t=?,
                     golden_aesthetic=?, golden_lpips=?, golden_fid=?,
                     status=CASE WHEN status='registered' THEN 'evaluated' ELSE status END
                   WHERE id=?""",
                (json.dumps(arm_results),
                 head.get("clip_i"), head.get("clip_t"), head.get("aesthetic"),
                 head.get("lpips"), head.get("fid"), exp_id),
            )
            self._conn.commit()

    def set_status(self, exp_id: str, status: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE experiments SET status=? WHERE id=?",
                               (status, exp_id))
            self._conn.commit()

    def update(self, exp_id: str, **fields) -> None:
        """Update arbitrary columns (hyperparams is JSON-encoded if a dict)."""
        if not fields:
            return
        if isinstance(fields.get("hyperparams"), dict):
            fields["hyperparams"] = json.dumps(fields["hyperparams"])
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self._conn.execute(f"UPDATE experiments SET {cols} WHERE id=?",
                               (*fields.values(), exp_id))
            self._conn.commit()

    # ── Query ────────────────────────────────────────────────────────────────

    def get(self, exp_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM experiments WHERE id=?", (exp_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def list(self, campaign: Optional[str] = None,
             status: Optional[str] = None) -> list[dict]:
        q = "SELECT * FROM experiments"
        clauses, params = [], []
        if campaign:
            clauses.append("campaign=?"); params.append(campaign)
        if status:
            clauses.append("status=?"); params.append(status)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY id"
        with self._lock:
            rows = self._conn.execute(q, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        for k in ("hyperparams", "golden_results"):
            if d.get(k):
                try:
                    d[k] = json.loads(d[k])
                except (TypeError, json.JSONDecodeError):
                    pass
        return d

    # ── Ranking / Champion-Challenger ────────────────────────────────────────

    def rank(self, metric: str = "clip_i") -> list[dict]:
        """Return evaluated experiments ordered best→worst by a golden metric.

        metric is one of GOLDEN_METRICS; lpips/fid rank ascending (lower better).
        Experiments missing that metric are excluded.
        """
        col = f"golden_{metric}"
        if metric not in GOLDEN_METRICS:
            raise ValueError(f"metric must be one of {GOLDEN_METRICS}")
        order = "ASC" if metric in _LOWER_IS_BETTER else "DESC"
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM experiments WHERE {col} IS NOT NULL "
                f"ORDER BY {col} {order}, id"
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def champion(self, metric: str = "clip_i") -> Optional[dict]:
        """The best evaluated experiment by `metric`, or None if none evaluated."""
        ranked = self.rank(metric)
        return ranked[0] if ranked else None

    def promote_champion(self, metric: str = "clip_i",
                         min_margin: float = 0.0) -> Optional[str]:
        """Recompute the Champion and update statuses.

        The best experiment by `metric` is marked 'champion'; all other
        evaluated experiments become 'challenger'; the prior champion (if a
        different one and the new best does not beat it by min_margin) is left
        as champion to avoid churn on noise.

        Returns the champion experiment id, or None if nothing is evaluated.
        """
        ranked = self.rank(metric)
        if not ranked:
            return None

        col = f"golden_{metric}"
        prior = next((e for e in ranked if e["status"] == "champion"), None)
        best  = ranked[0]

        # Hysteresis: only switch champion if the new best beats the incumbent
        # by at least min_margin (guards against promoting on metric noise).
        new_champ = best
        if prior is not None and prior["id"] != best["id"]:
            pv, bv = prior.get(col), best.get(col)
            if pv is not None and bv is not None:
                improvement = (pv - bv) if metric in _LOWER_IS_BETTER else (bv - pv)
                if improvement < min_margin:
                    new_champ = prior

        with self._lock:
            for e in ranked:
                st = "champion" if e["id"] == new_champ["id"] else "challenger"
                self._conn.execute("UPDATE experiments SET status=? WHERE id=?",
                                   (st, e["id"]))
            self._conn.commit()
        return new_champ["id"]

    # ── Comparison ───────────────────────────────────────────────────────────

    def compare(self, id_a: str, id_b: str) -> dict:
        """Structured comparison of two experiments across golden + train metrics.

        Returns {fields: {field: {a, b, delta, better}}, winner_by_metric: {...}}.
        delta = b - a; 'better' names which id wins for that field's direction.
        """
        a, b = self.get(id_a), self.get(id_b)
        if a is None or b is None:
            raise KeyError(f"unknown experiment(s): {id_a if a is None else ''} "
                           f"{id_b if b is None else ''}".strip())

        fields = {}
        winner_by_metric = {}
        # Golden metrics + training metrics. All higher-is-better except lpips/fid.
        directional = [(f"golden_{m}", m) for m in GOLDEN_METRICS]
        directional += [("cond_gap", "cond_gap"), ("ref_gap", "ref_gap")]

        for col, metric in directional:
            av, bv = a.get(col), b.get(col)
            entry = {"a": av, "b": bv, "delta": None, "better": None}
            if av is not None and bv is not None:
                entry["delta"] = round(bv - av, 6)
                if av == bv:
                    winner = "tie"
                elif metric_is_better(metric, av, bv):   # True when a is better
                    winner = id_a
                else:
                    winner = id_b
                entry["better"] = winner
                winner_by_metric[metric] = winner
            fields[col] = entry

        return {
            "id_a": id_a, "id_b": id_b,
            "campaign_a": a.get("campaign"), "campaign_b": b.get("campaign"),
            "fields": fields,
            "winner_by_metric": winner_by_metric,
        }
