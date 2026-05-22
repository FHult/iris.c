#!/usr/bin/env python3
"""
train/scripts/flywheel_lib.py — Flywheel support library.

Provides: FlywheelDB (SQLite iteration telemetry), config loading, and
HTML report rendering.  The actual iteration loop lives in orchestrator.py
(_run_flywheel_loop) so it can reuse the orchestrator's existing
heartbeat / dispatch / tmux / control-signal infrastructure.

This module has no runner class and no main loop.  The entry point for
starting a flywheel run is:

    pipeline_ctl start-flywheel train/configs/flywheel_sref_v1.yaml

which launches orchestrator.py with --flywheel-config.

Report-only regeneration:
    train/.venv/bin/python train/scripts/orchestrator.py \\
        --flywheel-config train/configs/flywheel_sref_v1.yaml --report-only
"""

import json
import re
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_lib import FLYWHEEL_DB_PATH, now_iso

# ---------------------------------------------------------------------------
# Metric log regexes (mirrors ablation_harness.py)
# ---------------------------------------------------------------------------

RE_STEP  = re.compile(r"^step\s+([\d,]+)/([\d,]+)\s+loss\s+([\d.]+)\s+\(avg\s+([\d.]+)\)")
RE_COND  = re.compile(r"loss_cond=([\d.]+)\s+loss_null=([\d.]+)\s+gap=([+-][\d.]+)")
RE_REF   = re.compile(r"loss_ref:.*?self=([\d.]+)(?:.*?cross=([\d.]+).*?gap=([+-][\d.]+))?")
RE_SCALE = re.compile(r"ip_scale:\s+mean=([\d.]+).*?double=([\d.]+).*?single=([\d.]+)")


# ---------------------------------------------------------------------------
# FlywheelDB — iteration telemetry
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS iterations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    flywheel_name TEXT    NOT NULL,
    iteration     INTEGER NOT NULL,
    n_shards      INTEGER,
    selected_shards TEXT,   -- JSON list of shard IDs
    hyperparams   TEXT,     -- JSON
    ablation_run  TEXT,     -- run_name if ablation was triggered
    steps         INTEGER,
    train_loss    REAL,
    ref_gap       REAL,
    cond_gap      REAL,
    status        TEXT NOT NULL DEFAULT 'running',
    exit_code     INTEGER,
    elapsed_secs  INTEGER,
    checkpoint    TEXT,
    checkpoint_hash TEXT,   -- first 12 chars of checkpoint stem (version tag)
    git_commit    TEXT,
    ts_start      TEXT NOT NULL,
    ts_end        TEXT
);
CREATE INDEX IF NOT EXISTS idx_fw_iter ON iterations(flywheel_name, iteration);

CREATE TABLE IF NOT EXISTS checkpoint_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    flywheel_name   TEXT    NOT NULL,
    iteration       INTEGER NOT NULL,
    checkpoint_path TEXT,
    checkpoint_hash TEXT,
    ref_gap         REAL,
    cond_gap        REAL,
    train_loss      REAL,
    is_best         INTEGER NOT NULL DEFAULT 0,
    ts              TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ckpt_fw ON checkpoint_log(flywheel_name, iteration);

CREATE TABLE IF NOT EXISTS campaign_summary (
    flywheel_name    TEXT PRIMARY KEY,
    n_iterations     INTEGER NOT NULL DEFAULT 0,
    total_elapsed_secs INTEGER,
    best_cond_gap    REAL,
    best_ref_gap     REAL,
    best_checkpoint  TEXT,
    final_cond_gap   REAL,
    final_ref_gap    REAL,
    status           TEXT NOT NULL DEFAULT 'active',
    ts_first         TEXT,
    ts_last          TEXT
);

CREATE TABLE IF NOT EXISTS _meta (k TEXT PRIMARY KEY, v TEXT);
INSERT OR IGNORE INTO _meta VALUES ('schema_version', '3');
"""

# v1→v2: add checkpoint_hash to iterations
_V2_MIGRATION = "ALTER TABLE iterations ADD COLUMN checkpoint_hash TEXT"
_V3_MIGRATION = """
CREATE TABLE IF NOT EXISTS campaign_summary (
    flywheel_name    TEXT PRIMARY KEY,
    n_iterations     INTEGER NOT NULL DEFAULT 0,
    total_elapsed_secs INTEGER,
    best_cond_gap    REAL,
    best_ref_gap     REAL,
    best_checkpoint  TEXT,
    final_cond_gap   REAL,
    final_ref_gap    REAL,
    status           TEXT NOT NULL DEFAULT 'active',
    ts_first         TEXT,
    ts_last          TEXT
)
"""

_V4_MIGRATIONS = [
    "ALTER TABLE campaign_summary ADD COLUMN clip_i_best REAL",
    "ALTER TABLE campaign_summary ADD COLUMN clip_i_final REAL",
    "ALTER TABLE campaign_summary ADD COLUMN total_steps INTEGER",
    "ALTER TABLE campaign_summary ADD COLUMN superseded_by TEXT",
    "INSERT OR IGNORE INTO _meta VALUES ('schema_version', '4')",
]

_V5_MIGRATIONS = [
    # Ablation-tuned hyperparams serialised as JSON; written after each successful
    # ablation burst and read back on orchestrator restart so restarts inherit tuning.
    "ALTER TABLE campaign_summary ADD COLUMN best_hyperparams TEXT",
    # Absolute path to the flywheel YAML config used to launch this campaign.
    # Written once at campaign start; lets the doctor find ablation_config without
    # parsing logs.
    "ALTER TABLE campaign_summary ADD COLUMN config_path TEXT",
    "INSERT OR REPLACE INTO _meta VALUES ('schema_version', '5')",
]

_V6_MIGRATIONS = [
    # Number of shards that gained their first-ever score observation in this iteration.
    # Tracks warm-up discovery rate; plateaus when pool coverage saturates.
    "ALTER TABLE iterations ADD COLUMN n_first_contact INTEGER",
    # Checkpoint the campaign started from (before any iterations ran).
    # Makes it easy to identify what Run N+1 should warm-start from.
    "ALTER TABLE campaign_summary ADD COLUMN base_checkpoint TEXT",
    "INSERT OR REPLACE INTO _meta VALUES ('schema_version', '6')",
]


def _checkpoint_hash(ckpt_path: Optional[str]) -> str:
    """Stable 12-char version tag derived from the checkpoint filename."""
    if not ckpt_path:
        return ""
    return Path(ckpt_path).stem[:12]


class FlywheelDB:
    """SQLite-backed iteration telemetry. Thread-safe."""

    def __init__(self, db_path: Path = FLYWHEEL_DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._conn.executescript(_SCHEMA)
        for _mig in (_V2_MIGRATION, _V3_MIGRATION):
            try:
                self._conn.executescript(_mig)
            except Exception:
                pass
        for _mig in _V4_MIGRATIONS:
            try:
                self._conn.execute(_mig)
            except Exception:
                pass
        for _mig in _V5_MIGRATIONS:
            try:
                self._conn.execute(_mig)
            except Exception:
                pass
        for _mig in _V6_MIGRATIONS:
            try:
                self._conn.execute(_mig)
            except Exception:
                pass
        self._conn.commit()

    def insert_iteration(
        self,
        name: str,
        iteration: int,
        n_shards: int,
        shard_ids: list[str],
        hyperparams: dict,
        steps: int,
        git_commit: str = "",
        checkpoint_hash: str = "",
    ) -> int:
        ts = now_iso()
        with self._lock:
            cur = self._conn.execute("""
                INSERT INTO iterations
                  (flywheel_name, iteration, n_shards, selected_shards,
                   hyperparams, steps, git_commit, checkpoint_hash, ts_start)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, iteration, n_shards, json.dumps(shard_ids),
                  json.dumps(hyperparams, default=str), steps, git_commit,
                  checkpoint_hash or None, ts))
            self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def update_iteration(
        self,
        row_id: int,
        status: str,
        exit_code: int,
        elapsed_secs: int,
        train_loss: Optional[float],
        ref_gap: Optional[float],
        cond_gap: Optional[float],
        checkpoint: Optional[str] = None,
        checkpoint_hash: str = "",
        ablation_run: Optional[str] = None,
    ) -> None:
        ts = now_iso()
        with self._lock:
            self._conn.execute("""
                UPDATE iterations SET
                  status=?, exit_code=?, elapsed_secs=?,
                  train_loss=?, ref_gap=?, cond_gap=?,
                  checkpoint=?, checkpoint_hash=?, ablation_run=?, ts_end=?
                WHERE id=?
            """, (status, exit_code, elapsed_secs,
                  train_loss, ref_gap, cond_gap,
                  checkpoint, checkpoint_hash or None, ablation_run, ts, row_id))
            self._conn.commit()

    # ------------------------------------------------------------------
    # Checkpoint log

    def upsert_checkpoint(
        self,
        name: str,
        iteration: int,
        checkpoint_path: str,
        checkpoint_hash: str,
        ref_gap: Optional[float],
        cond_gap: Optional[float],
        train_loss: Optional[float],
    ) -> None:
        """Record a checkpoint produced by this iteration."""
        ts = now_iso()
        with self._lock:
            self._conn.execute("""
                INSERT INTO checkpoint_log
                    (flywheel_name, iteration, checkpoint_path, checkpoint_hash,
                     ref_gap, cond_gap, train_loss, ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, iteration, checkpoint_path or None, checkpoint_hash or None,
                  ref_gap, cond_gap, train_loss, ts))
            self._conn.commit()

    def mark_best_checkpoint(self, name: str, iteration: int) -> None:
        """Mark the checkpoint at <iteration> as the current best; clear prior bests."""
        with self._lock:
            self._conn.execute(
                "UPDATE checkpoint_log SET is_best=0 WHERE flywheel_name=?", (name,)
            )
            self._conn.execute(
                "UPDATE checkpoint_log SET is_best=1 "
                "WHERE flywheel_name=? AND iteration=?", (name, iteration)
            )
            self._conn.commit()

    def get_checkpoint_history(self, name: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM checkpoint_log WHERE flywheel_name=? ORDER BY iteration",
                (name,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_iterations(self, name: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM iterations WHERE flywheel_name=? ORDER BY iteration",
                (name,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_best(self, name: str) -> Optional[dict]:
        # cond_gap is the primary quality signal: stable and monotonically informative
        # at 1000-step iteration budgets.  ref_gap is noisy at this scale.
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM iterations WHERE flywheel_name=? AND cond_gap IS NOT NULL "
                "ORDER BY cond_gap DESC LIMIT 1",
                (name,),
            ).fetchone()
        return dict(row) if row else None

    def get_all_run_names(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT flywheel_name FROM iterations ORDER BY flywheel_name"
            ).fetchall()
        return [r[0] for r in rows]

    def refresh_campaign_summary(self, name: str) -> None:
        """Recompute and upsert the campaign_summary row from the iterations table."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM iterations WHERE flywheel_name=? ORDER BY iteration",
                (name,),
            ).fetchall()
        if not rows:
            return
        iters = [dict(r) for r in rows]
        total_elapsed = sum(r.get("elapsed_secs") or 0 for r in iters)
        total_steps   = sum(r.get("steps") or 0 for r in iters if r.get("status") == "done")
        best = max(
            (r for r in iters if r.get("cond_gap") is not None),
            key=lambda r: r["cond_gap"],
            default=None,
        )
        last = iters[-1]
        ts_first = iters[0].get("ts_start")
        ts_last  = last.get("ts_end") or last.get("ts_start")
        with self._lock:
            self._conn.execute("""
                INSERT INTO campaign_summary
                    (flywheel_name, n_iterations, total_elapsed_secs, total_steps,
                     best_cond_gap, best_ref_gap, best_checkpoint,
                     final_cond_gap, final_ref_gap, ts_first, ts_last)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(flywheel_name) DO UPDATE SET
                    n_iterations=excluded.n_iterations,
                    total_elapsed_secs=excluded.total_elapsed_secs,
                    total_steps=excluded.total_steps,
                    best_cond_gap=excluded.best_cond_gap,
                    best_ref_gap=excluded.best_ref_gap,
                    best_checkpoint=excluded.best_checkpoint,
                    final_cond_gap=excluded.final_cond_gap,
                    final_ref_gap=excluded.final_ref_gap,
                    ts_last=excluded.ts_last
            """, (
                name, len(iters), total_elapsed or None, total_steps or None,
                best["cond_gap"] if best else None,
                best["ref_gap"]  if best else None,
                best["checkpoint"] if best else None,
                last.get("cond_gap"), last.get("ref_gap"),
                ts_first, ts_last,
            ))
            self._conn.commit()

    def get_campaign_summaries(self) -> list[dict]:
        """Return all campaign summary rows, most-recently-updated first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM campaign_summary ORDER BY ts_last DESC NULLS LAST"
            ).fetchall()
        return [dict(r) for r in rows]

    def set_campaign_status(self, name: str, status: str,
                            superseded_by: Optional[str] = None) -> None:
        """Write lifecycle status (active/plateau/completed/superseded) to campaign_summary."""
        ts = now_iso()
        with self._lock:
            # Ensure the row exists first (upsert with minimal fields if absent)
            self._conn.execute("""
                INSERT OR IGNORE INTO campaign_summary (flywheel_name, status, ts_first, ts_last)
                VALUES (?, ?, ?, ?)
            """, (name, status, ts, ts))
            if superseded_by:
                self._conn.execute("""
                    UPDATE campaign_summary SET status=?, superseded_by=?, ts_last=?
                    WHERE flywheel_name=?
                """, (status, superseded_by, ts, name))
            else:
                self._conn.execute("""
                    UPDATE campaign_summary SET status=?, ts_last=? WHERE flywheel_name=?
                """, (status, ts, name))
            self._conn.commit()

    def set_best_hyperparams(self, name: str, hyperparams: dict) -> None:
        """Persist ablation-tuned hyperparams so orchestrator restarts inherit them."""
        import json as _json
        ts = now_iso()
        encoded = _json.dumps(hyperparams, sort_keys=True)
        with self._lock:
            self._conn.execute("""
                INSERT OR IGNORE INTO campaign_summary (flywheel_name, status, ts_first, ts_last)
                VALUES (?, 'active', ?, ?)
            """, (name, ts, ts))
            self._conn.execute("""
                UPDATE campaign_summary SET best_hyperparams=?, ts_last=? WHERE flywheel_name=?
            """, (encoded, ts, name))
            self._conn.commit()

    def get_best_hyperparams(self, name: str) -> Optional[dict]:
        """Return persisted ablation-tuned hyperparams, or None if none recorded."""
        import json as _json
        with self._lock:
            row = self._conn.execute(
                "SELECT best_hyperparams FROM campaign_summary WHERE flywheel_name=?",
                (name,),
            ).fetchone()
        if row is None or row[0] is None:
            return None
        try:
            return _json.loads(row[0])
        except Exception:
            return None

    def set_config_path(self, name: str, config_path: str) -> None:
        """Record the absolute path of the flywheel YAML used to launch this campaign."""
        ts = now_iso()
        with self._lock:
            self._conn.execute("""
                INSERT OR IGNORE INTO campaign_summary (flywheel_name, status, ts_first, ts_last)
                VALUES (?, 'active', ?, ?)
            """, (name, ts, ts))
            self._conn.execute("""
                UPDATE campaign_summary SET config_path=? WHERE flywheel_name=?
            """, (config_path, name))
            self._conn.commit()

    def set_base_checkpoint(self, name: str, checkpoint_path: Optional[str]) -> None:
        """Record the checkpoint the campaign started from (written once at loop start)."""
        if not checkpoint_path:
            return
        ts = now_iso()
        with self._lock:
            self._conn.execute("""
                INSERT OR IGNORE INTO campaign_summary (flywheel_name, status, ts_first, ts_last)
                VALUES (?, 'active', ?, ?)
            """, (name, ts, ts))
            self._conn.execute("""
                UPDATE campaign_summary SET base_checkpoint=? WHERE flywheel_name=?
            """, (checkpoint_path, name))
            self._conn.commit()

    def set_n_first_contact(self, row_id: int, n: int) -> None:
        """Write the first-contact shard count for an iteration row."""
        with self._lock:
            self._conn.execute(
                "UPDATE iterations SET n_first_contact=? WHERE id=?", (n, row_id)
            )
            self._conn.commit()

    def get_non_superseded_campaigns(self) -> list[dict]:
        """Return campaign summaries excluding superseded ones, best cond_gap first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM campaign_summary WHERE status != 'superseded' "
                "ORDER BY best_cond_gap DESC NULLS LAST"
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_superseded_by(self, new_name: str) -> list[str]:
        """Mark older completed campaigns as superseded if new_name exceeds them on all metrics.

        A campaign is superseded when new_name has strictly higher best_cond_gap AND
        best_ref_gap (both non-null). Returns names of campaigns that were superseded.
        """
        new_row = None
        with self._lock:
            r = self._conn.execute(
                "SELECT best_cond_gap, best_ref_gap FROM campaign_summary WHERE flywheel_name=?",
                (new_name,),
            ).fetchone()
        if r is None:
            return []
        new_cg, new_rg = r[0], r[1]
        if new_cg is None:
            return []

        superseded = []
        with self._lock:
            candidates = self._conn.execute(
                "SELECT flywheel_name, best_cond_gap, best_ref_gap FROM campaign_summary "
                "WHERE flywheel_name != ? AND status IN ('completed', 'plateau')",
                (new_name,),
            ).fetchall()
        for row in candidates:
            old_name, old_cg, old_rg = row[0], row[1], row[2]
            if old_cg is None:
                continue
            cg_better = new_cg > old_cg
            rg_better = (new_rg is None or old_rg is None) or new_rg > old_rg
            if cg_better and rg_better:
                self.set_campaign_status(old_name, "superseded", superseded_by=new_name)
                superseded.append(old_name)
        return superseded

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Campaign summary export
# ---------------------------------------------------------------------------

def write_campaign_summary_json(name: str, db: "FlywheelDB", cold_root: "Path") -> list[str]:
    """Write campaign summary JSON to cold metadata + weight dir. Returns paths written."""
    import json as _json
    from pathlib import Path as _Path

    iters = db.get_iterations(name)
    summaries = {r["flywheel_name"]: r for r in db.get_campaign_summaries()}
    summary_row = summaries.get(name, {})

    done_iters = [r for r in iters if r.get("status") == "done"]
    cond_gap_trajectory = [r.get("cond_gap") for r in done_iters if r.get("cond_gap") is not None]
    ablation_runs = sorted({r["ablation_run"] for r in iters if r.get("ablation_run")})

    shards_consumed: list[str] = []
    seen: set = set()
    for r in iters:
        raw = r.get("selected_shards")
        try:
            sel = _json.loads(raw) if isinstance(raw, str) else (raw or [])
        except Exception:
            sel = []
        for sid in sel:
            if sid not in seen:
                seen.add(sid)
                shards_consumed.append(sid)

    git_commit = next((r.get("git_commit") for r in reversed(iters) if r.get("git_commit")), None)

    data = {
        "name":                name,
        "status":              summary_row.get("status", "active"),
        "n_iterations":        len(iters),
        "total_steps":         summary_row.get("total_steps"),
        "total_elapsed_secs":  summary_row.get("total_elapsed_secs"),
        "best_cond_gap":       summary_row.get("best_cond_gap"),
        "best_ref_gap":        summary_row.get("best_ref_gap"),
        "best_clip_i":         summary_row.get("clip_i_best"),
        "best_checkpoint":     summary_row.get("best_checkpoint"),
        "final_cond_gap":      summary_row.get("final_cond_gap"),
        "final_ref_gap":       summary_row.get("final_ref_gap"),
        "cond_gap_trajectory": cond_gap_trajectory,
        "ablation_runs":       ablation_runs,
        "shards_consumed":     shards_consumed,
        "ts_start":            summary_row.get("ts_first"),
        "ts_end":              summary_row.get("ts_last"),
        "git_commit":          git_commit,
    }

    written: list[str] = []
    cold = _Path(cold_root)

    # 1. cold/metadata/flywheel_logs/campaign-{name}.json
    logs_dir = cold / "metadata" / "flywheel_logs"
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        dest = logs_dir / f"campaign-{name}.json"
        dest.write_text(_json.dumps(data, indent=2))
        written.append(str(dest))
    except OSError as e:
        print(f"[flywheel] WARNING: could not write campaign log: {e}", file=sys.stderr)

    # 2. cold/weights/flywheel-{date}/summary.json (most recent campaign dir)
    weights_dir = cold / "weights"
    if weights_dir.exists():
        campaigns = sorted(weights_dir.glob("flywheel-*"), reverse=True)
        if campaigns:
            try:
                dest2 = campaigns[0] / "summary.json"
                dest2.write_text(_json.dumps(data, indent=2))
                written.append(str(dest2))
            except OSError as e:
                print(f"[flywheel] WARNING: could not write weight summary: {e}", file=sys.stderr)

    return written


# ---------------------------------------------------------------------------
# Metric parsing
# ---------------------------------------------------------------------------

def collect_metrics_from_log(log_path: Path) -> dict:
    """Parse trainer log; return the last-seen snapshot of each metric."""
    metrics: dict = {}
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return metrics

    for line in text.splitlines():
        m = RE_STEP.search(line)
        if m:
            metrics.update({
                "step": int(m.group(1).replace(",", "")),
                "loss": float(m.group(3)),
                "loss_smooth": float(m.group(4)),
            })
            continue
        m = RE_COND.search(line)
        if m:
            metrics.update({
                "loss_cond": float(m.group(1)),
                "loss_null": float(m.group(2)),
                "cond_gap":  float(m.group(3)),
            })
            continue
        m = RE_REF.search(line)
        if m:
            metrics["loss_self_ref"] = float(m.group(1))
            if m.group(2):
                metrics["loss_cross_ref"] = float(m.group(2))
                metrics["ref_gap"]        = float(m.group(3))
            continue
        m = RE_SCALE.search(line)
        if m:
            metrics.update({
                "ip_scale_mean":   float(m.group(1)),
                "ip_scale_double": float(m.group(2)),
                "ip_scale_single": float(m.group(3)),
            })
    return metrics


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def check_plateau(done_iters: list[dict], patience: int, threshold: float) -> Optional[str]:
    """Return a reason string if cond_gap has plateaued over the last `patience` iterations.

    Plateau is declared when the net improvement over the window is below `threshold`:
        max(window) - window[0] < threshold

    This asks "did we make meaningful progress from where the window started?"
    A rising trend with low noise correctly passes; a flat or regressing run fails.

    A secondary regression signal fires when the final value is strictly below
    the window-start value, regardless of threshold.

    Returns None when there is insufficient data or no plateau.
    """
    if patience <= 0 or len(done_iters) < patience:
        return None
    recent = done_iters[-patience:]
    vals   = [i.get("cond_gap") for i in recent if i.get("cond_gap") is not None]
    if len(vals) < patience:
        return None

    window_start = vals[0]
    window_max   = max(vals)
    window_mean  = sum(vals) / len(vals)
    improvement  = window_max - window_start

    if vals[-1] < window_start:
        return (
            f"cond_gap regression over last {patience} iterations "
            f"(final={vals[-1]:.4f} < start={window_start:.4f}, "
            f"mean={window_mean:.4f})"
        )
    if improvement < threshold:
        return (
            f"cond_gap plateau over last {patience} iterations "
            f"(improvement={improvement:.4f} < threshold={threshold:.4f}, "
            f"mean={window_mean:.4f})"
        )
    return None


def load_flywheel_config(path: Path) -> dict:
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = raw.get("flywheel", raw)
    if "name" not in cfg:
        print(f"ERROR: flywheel config missing 'name': {path}", file=sys.stderr)
        sys.exit(1)
    return cfg


# ---------------------------------------------------------------------------
# HTML reports
# ---------------------------------------------------------------------------

def render_flywheel_index(
    name: str,
    iterations: list[dict],
    shard_stats: dict,
    top_shards: list[dict],
    current_hyperparams: dict,
    plateau_reason: Optional[str] = None,
) -> str:
    ts     = now_iso()
    done   = [i for i in iterations if i["status"] == "done"]
    failed = [i for i in iterations if i["status"] == "failed"]

    def _fmt(v, f=".4f"):
        return f"{v:{f}}" if v is not None else "—"

    # ── Iteration table ───────────────────────────────────────────────────────
    _cond_by_iter = {it["iteration"]: it["cond_gap"] for it in done if it.get("cond_gap") is not None}
    best_iter = max(_cond_by_iter, key=_cond_by_iter.__getitem__) if _cond_by_iter else None

    rows = ""
    for it in iterations:
        st  = it["status"]
        col = {"done": "#7f7", "failed": "#f77", "running": "#fa7"}.get(st, "#888")
        e   = it.get("elapsed_secs") or 0
        et  = f"{e//3600}h{(e%3600)//60}m" if e > 3600 else f"{e//60}m{e%60:02d}s"
        abl = "✓" if it.get("ablation_run") else ""
        is_best = it["iteration"] == best_iter and st == "done"
        row_style = " style='background:#0d1a0d'" if is_best else ""
        nfc = it.get("n_first_contact")
        nfc_str = str(nfc) if nfc is not None else "—"
        rows += (
            f"<tr{row_style}>"
            f"<td style='color:#aaa'>{it['iteration']}"
            f"{'★' if is_best else ''}</td>"
            f"<td style='color:{col}'>{st}</td>"
            f"<td style='color:#7af;font-weight:bold'>{_fmt(it.get('ref_gap'))}</td>"
            f"<td style='color:#7df'>{_fmt(it.get('cond_gap'))}</td>"
            f"<td>{_fmt(it.get('train_loss'))}</td>"
            f"<td style='color:#888'>{it.get('n_shards',0)}</td>"
            f"<td style='color:#fa7'>{nfc_str}</td>"
            f"<td style='color:#888'>{et}</td>"
            f"<td style='color:#7f7'>{abl}</td>"
            f"<td style='color:#777;font-size:0.75em'>{str(it.get('ts_start',''))[:16]}</td>"
            f"</tr>\n"
        )

    # ── Trend data (cond_gap primary, ref_gap secondary) ─────────────────────
    trend_pts = [
        {"i": it["iteration"], "ref_gap": it.get("ref_gap"),
         "cond_gap": it.get("cond_gap"), "loss": it.get("train_loss")}
        for it in done
    ]

    # Rolling best for cond_gap
    cond_best_line: list = []
    best_val = None
    for pt in trend_pts:
        v = pt["cond_gap"]
        if v is not None and (best_val is None or v > best_val):
            best_val = v
        cond_best_line.append(best_val)

    # Confidence band: rolling mean ± std over sliding window of up to 5 done iters
    window = 5
    cond_band = []
    for i, pt in enumerate(trend_pts):
        w_pts = [trend_pts[j]["cond_gap"] for j in range(max(0, i - window + 1), i + 1)
                 if trend_pts[j]["cond_gap"] is not None]
        if len(w_pts) >= 2:
            mean = sum(w_pts) / len(w_pts)
            std  = (sum((v - mean) ** 2 for v in w_pts) / len(w_pts)) ** 0.5
            cond_band.append({"i": pt["i"], "mean": round(mean, 5), "std": round(std, 5)})
        else:
            cond_band.append({"i": pt["i"], "mean": pt["cond_gap"], "std": 0.0})

    # ── Pareto scatter: ref_gap vs cond_gap ───────────────────────────────────
    pareto_pts = []
    if len(done) >= 2:
        scored = [it for it in done
                  if it.get("ref_gap") is not None and it.get("cond_gap") is not None]
        for j, ej in enumerate(scored):
            dominated = any(
                ek["ref_gap"]  >= ej["ref_gap"]  and ek["cond_gap"] >= ej["cond_gap"] and
                (ek["ref_gap"] >  ej["ref_gap"]  or  ek["cond_gap"] >  ej["cond_gap"])
                for k, ek in enumerate(scored) if k != j
            )
            pareto_pts.append({
                "i":         ej["iteration"],
                "ref_gap":   ej.get("ref_gap"),
                "cond_gap":  ej.get("cond_gap"),
                "is_pareto": 0 if dominated else 1,
            })

    # ── Discovery rate: n_first_contact per done iteration ────────────────────
    discovery_pts = [
        {"i": it["iteration"], "n": it.get("n_first_contact")}
        for it in done
    ]

    # ── Shard heatmap ─────────────────────────────────────────────────────────
    heatmap_shard_ids = [s["shard_id"] for s in top_shards[:20]]
    heatmap_iters = []
    for it in iterations[-30:]:
        raw = it.get("selected_shards")
        try:
            sel = json.loads(raw) if isinstance(raw, str) else (raw or [])
        except (json.JSONDecodeError, TypeError):
            sel = []
        sel_set = set(sel)
        heatmap_iters.append({
            "iter":  it["iteration"],
            "cells": [1 if sid in sel_set else 0 for sid in heatmap_shard_ids],
        })

    # ── Shard table ───────────────────────────────────────────────────────────
    shard_rows = ""
    for rank, s in enumerate(top_shards):
        shard_rows += (
            f"<tr>"
            f"<td style='color:#aaa'>{rank+1}</td>"
            f"<td><code style='color:#7af'>{s['shard_id']}</code></td>"
            f"<td style='color:#888'>{s.get('source','?')}</td>"
            f"<td style='color:#7af'>{_fmt(s.get('composite_score'))}</td>"
            f"<td>{_fmt(s.get('ref_gap_mean'))}</td>"
            f"<td>{_fmt(s.get('cond_gap_mean'))}</td>"
            f"<td>{s.get('n_scored',0)}</td>"
            f"<td>{s.get('n_selected',0)}</td>"
            f"</tr>\n"
        )

    hp_rows = ""
    for k, v in current_hyperparams.items():
        hp_rows += f"<tr><td>{k}</td><td style='color:#7af'>{v}</td></tr>\n"
    if not hp_rows:
        hp_rows = "<tr><td colspan=2 style='color:#555'>defaults from base config</td></tr>\n"

    plateau_banner = ""
    if plateau_reason:
        plateau_banner = (
            f"<div style='background:#1a0a00;border:1px solid #f72;border-radius:4px;"
            f"padding:8px 14px;margin:10px 0;color:#fa7;font-size:0.88em'>"
            f"⚠ Plateau detected — flywheel paused: {plateau_reason}<br>"
            f"<span style='color:#888'>Resume: <code>pipeline_ctl resume-flywheel</code> "
            f"&nbsp;|&nbsp; Force continue: <code>pipeline_ctl force-continue-flywheel</code></span>"
            f"</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8">
<title>Flywheel — {name}</title>
<style>
  body{{font-family:monospace;background:#111;color:#ddd;margin:20px;line-height:1.5}}
  h1{{color:#7df;margin-bottom:4px}}
  h2{{color:#adf;font-size:1em;margin-top:24px;margin-bottom:6px;
      border-bottom:1px solid #333;padding-bottom:3px}}
  .meta{{color:#777;font-size:0.82em;margin-bottom:14px}}
  .stat{{display:inline-block;background:#1a1a2e;border:1px solid #334;
         border-radius:4px;padding:6px 12px;margin:4px;font-size:0.85em}}
  .stat span{{color:#7af;font-weight:bold}}
  table{{border-collapse:collapse;font-size:0.82em;margin-top:6px}}
  th,td{{border:1px solid #333;padding:4px 8px;text-align:left}}
  th{{background:#1c1c1c;color:#adf}} td{{background:#161616}}
  canvas{{background:#161616;border:1px solid #2a2a2a;border-radius:4px}}
  .chart-label{{color:#777;font-size:0.8em;margin-bottom:3px;margin-top:12px}}
  .charts{{display:flex;flex-wrap:wrap;gap:16px;margin-top:6px}}
</style>
</head>
<body>
<h1>Flywheel</h1>
<div class="meta">name={name} &nbsp;|&nbsp; {ts}</div>
{plateau_banner}
<div>
  <div class="stat">iterations done: <span>{len(done)}</span></div>
  <div class="stat">failed: <span>{len(failed)}</span></div>
  <div class="stat">shards scored: <span>{shard_stats.get('scored',0)}/{shard_stats.get('total',0)}</span></div>
  <div class="stat">coverage: <span>{round(100*shard_stats.get('scored',0)/max(1,shard_stats.get('total',1)))}%</span></div>
</div>

<h2>Quality Trends</h2>
<div class="charts">
  <div>
    <div class="chart-label">cond_gap (null − cond) ↑ primary quality signal — band = rolling ±σ (window 5)</div>
    <canvas id="condTrend" width="560" height="200"></canvas>
  </div>
  <div>
    <div class="chart-label">Pareto front: ref_gap vs cond_gap</div>
    <canvas id="refTrend" width="340" height="200"></canvas>
  </div>
  <div>
    <div class="chart-label">New shards discovered per iteration (warm-up coverage rate)</div>
    <canvas id="discoveryTrend" width="340" height="200"></canvas>
  </div>
</div>

<h2>Iterations</h2>
<table>
  <tr><th>#</th><th>Status</th><th>ref_gap ↑</th><th>cond_gap ↑</th>
    <th>loss</th><th>shards</th><th>new</th><th>time</th><th>abl</th><th>started</th></tr>
  {rows}
</table>

<h2>Active Hyperparams</h2>
<table style="max-width:440px">
  <tr><th>Parameter</th><th>Value</th></tr>
  {hp_rows}
</table>

<h2>Top Shards by Composite Score</h2>
<div class="charts">
  <table>
    <tr><th>Rank</th><th>Shard</th><th>Source</th><th>Score</th>
      <th>ref_gap</th><th>cond_gap</th><th>Runs</th><th>×Selected</th></tr>
    {shard_rows}
  </table>
  <div id="heatmapBox">
    <div class="chart-label">Shard selection heatmap (top shards × recent iterations)</div>
    <canvas id="heatmap" width="480" height="320"></canvas>
  </div>
</div>

<h2>Data</h2>
<p>Per-iteration logs: <code>flywheel_&lt;name&gt;_iter&lt;N&gt;.log</code></p>
<p>Shard reports: <code>reports/shard_selection_iter&lt;N&gt;.html</code></p>

<script>
const TREND_PTS      = {json.dumps(trend_pts)};
const COND_BAND      = {json.dumps(cond_band)};
const BEST_COND      = {json.dumps(cond_best_line)};
const PARETO_PTS     = {json.dumps(pareto_pts)};
const DISCOVERY_PTS  = {json.dumps(discovery_pts)};
const HM_SHARDS      = {json.dumps(heatmap_shard_ids)};
const HM_ITERS       = {json.dumps(heatmap_iters)};

function drawTrend(id, key, color, band, bestLine) {{
  const cv=document.getElementById(id); if(!cv||!TREND_PTS.length) return;
  const ctx=cv.getContext('2d');
  const W=cv.width,H=cv.height,pad={{t:16,r:16,b:28,l:52}};
  const cw=W-pad.l-pad.r,ch=H-pad.t-pad.b;
  const vals=TREND_PTS.map(p=>p[key]).filter(v=>v!=null);
  if(!vals.length) return;
  const bandVals=band?band.flatMap(b=>b.mean!=null?[b.mean+b.std,b.mean-b.std]:[]):[];
  let yMin=Math.min(...vals,...bandVals), yMax=Math.max(...vals,...bandVals);
  const yp=(yMax-yMin)*0.1||0.02; yMin-=yp; yMax+=yp;
  const n=TREND_PTS.length;
  const sx=i=>pad.l+cw*i/(n-1||1);
  const sy=y=>pad.t+ch-((yMax>yMin)?(y-yMin)/(yMax-yMin)*ch:ch/2);
  ctx.strokeStyle='#2a2a2a'; ctx.lineWidth=1;
  for(let k=0;k<=4;k++){{
    const y=yMin+(yMax-yMin)*k/4;
    ctx.beginPath();ctx.moveTo(pad.l,sy(y));ctx.lineTo(pad.l+cw,sy(y));ctx.stroke();
    ctx.fillStyle='#555';ctx.font='9px monospace';ctx.fillText(y.toFixed(3),2,sy(y)+3);
  }}
  if(yMin<0&&yMax>0){{
    ctx.strokeStyle='#444';ctx.lineWidth=1.5;ctx.setLineDash([4,4]);
    ctx.beginPath();ctx.moveTo(pad.l,sy(0));ctx.lineTo(pad.l+cw,sy(0));ctx.stroke();
    ctx.setLineDash([]);
  }}
  // Confidence band
  if(band&&band.length>1){{
    ctx.save();
    ctx.fillStyle='rgba(100,180,255,0.12)';
    ctx.beginPath();
    band.forEach((b,i)=>{{if(b.mean==null)return; i===0?ctx.moveTo(sx(i),sy(b.mean+b.std)):ctx.lineTo(sx(i),sy(b.mean+b.std));}});
    for(let i=band.length-1;i>=0;i--){{const b=band[i];if(b.mean==null)continue;ctx.lineTo(sx(i),sy(b.mean-b.std));}}
    ctx.closePath(); ctx.fill();
    ctx.restore();
    // Band mean line
    ctx.strokeStyle='rgba(100,180,255,0.5)'; ctx.lineWidth=1; ctx.setLineDash([3,3]);
    ctx.beginPath(); let fst=true;
    band.forEach((b,i)=>{{if(b.mean==null)return; fst?ctx.moveTo(sx(i),sy(b.mean)):ctx.lineTo(sx(i),sy(b.mean)); fst=false;}});
    ctx.stroke(); ctx.setLineDash([]);
  }}
  // Per-iteration dots
  TREND_PTS.forEach((p,i)=>{{
    const v=p[key]; if(v==null) return;
    ctx.fillStyle=color;
    ctx.beginPath();ctx.arc(sx(i),sy(v),4,0,2*Math.PI);ctx.fill();
  }});
  // Rolling best line
  if(bestLine){{
    ctx.strokeStyle='rgba(255,255,255,0.7)'; ctx.lineWidth=1.5;
    ctx.beginPath(); let first=true;
    bestLine.forEach((v,i)=>{{if(v==null)return; first?ctx.moveTo(sx(i),sy(v)):ctx.lineTo(sx(i),sy(v)); first=false;}});
    ctx.stroke();
  }}
  for(let k=0;k<=4;k++){{
    const i=Math.round(k*(n-1)/4);
    ctx.fillStyle='#555';ctx.font='9px monospace';ctx.fillText(i+1,sx(i)-4,pad.t+ch+18);
  }}
}}

function drawPareto(id) {{
  const cv=document.getElementById(id); if(!cv||!PARETO_PTS.length) return;
  const ctx=cv.getContext('2d');
  const W=cv.width,H=cv.height,pad={{t:16,r:16,b:28,l:52}};
  const cw=W-pad.l-pad.r,ch=H-pad.t-pad.b;
  const pts=PARETO_PTS.filter(p=>p.ref_gap!=null&&p.cond_gap!=null);
  if(!pts.length) return;
  const xs=pts.map(p=>p.ref_gap), ys=pts.map(p=>p.cond_gap);
  let xMin=Math.min(...xs),xMax=Math.max(...xs);
  let yMin=Math.min(...ys),yMax=Math.max(...ys);
  const xp=(xMax-xMin)*0.15||0.05, yp=(yMax-yMin)*0.15||0.05;
  xMin-=xp; xMax+=xp; yMin-=yp; yMax+=yp;
  const sx=x=>pad.l+(x-xMin)/(xMax-xMin)*cw;
  const sy=y=>pad.t+ch-(y-yMin)/(yMax-yMin)*ch;
  ctx.strokeStyle='#2a2a2a'; ctx.lineWidth=1;
  for(let k=0;k<=3;k++){{
    const y=yMin+(yMax-yMin)*k/3;
    ctx.beginPath();ctx.moveTo(pad.l,sy(y));ctx.lineTo(pad.l+cw,sy(y));ctx.stroke();
    ctx.fillStyle='#555';ctx.font='9px monospace';ctx.fillText(y.toFixed(3),2,sy(y)+3);
  }}
  if(xMin<0&&xMax>0){{ctx.strokeStyle='#444';ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(sx(0),pad.t);ctx.lineTo(sx(0),pad.t+ch);ctx.stroke();ctx.setLineDash([]);}}
  if(yMin<0&&yMax>0){{ctx.strokeStyle='#444';ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(pad.l,sy(0));ctx.lineTo(pad.l+cw,sy(0));ctx.stroke();ctx.setLineDash([]);}}
  ctx.fillStyle='#555';ctx.font='9px monospace';ctx.fillText('ref_gap →',pad.l+cw/2-20,pad.t+ch+18);
  pts.forEach(p=>{{
    const x=sx(p.ref_gap),y=sy(p.cond_gap), ip=p.is_pareto===1;
    const clr=ip?'#7df':'rgba(90,90,90,0.6)';
    ctx.fillStyle=clr; ctx.strokeStyle=ip?'rgba(255,255,255,0.7)':'transparent'; ctx.lineWidth=1.5;
    ctx.beginPath();ctx.arc(x,y,ip?5:3,0,2*Math.PI);ctx.fill();
    if(ip){{ctx.stroke(); ctx.fillStyle='#bbb';ctx.font='8px monospace';ctx.fillText('iter '+p.i,x+7,y+3);}}
  }});
}}

function drawDiscovery(id) {{
  const cv=document.getElementById(id); if(!cv||!DISCOVERY_PTS.length) return;
  const ctx=cv.getContext('2d');
  const W=cv.width,H=cv.height,pad={{t:16,r:16,b:28,l:40}};
  const cw=W-pad.l-pad.r,ch=H-pad.t-pad.b;
  const vals=DISCOVERY_PTS.map(p=>p.n).filter(v=>v!=null);
  if(!vals.length) return;
  let yMax=Math.max(...vals,1); yMax=yMax+Math.ceil(yMax*0.15);
  const n=DISCOVERY_PTS.length;
  const sx=i=>pad.l+cw*i/(n-1||1);
  const sy=y=>pad.t+ch*(1-y/yMax);
  ctx.strokeStyle='#2a2a2a'; ctx.lineWidth=1;
  for(let k=0;k<=4;k++){{
    const y=yMax*k/4;
    ctx.beginPath();ctx.moveTo(pad.l,sy(y));ctx.lineTo(pad.l+cw,sy(y));ctx.stroke();
    ctx.fillStyle='#555';ctx.font='9px monospace';ctx.fillText(Math.round(y),2,sy(y)+3);
  }}
  // Zero line
  ctx.strokeStyle='#444';ctx.lineWidth=1.5;ctx.setLineDash([4,4]);
  ctx.beginPath();ctx.moveTo(pad.l,sy(0));ctx.lineTo(pad.l+cw,sy(0));ctx.stroke();
  ctx.setLineDash([]);
  // Bar chart
  const bw=Math.max(2,Math.floor(cw/n)-2);
  DISCOVERY_PTS.forEach((p,i)=>{{
    if(p.n==null) return;
    const x=sx(i)-bw/2, barH=ch*p.n/yMax;
    ctx.fillStyle='#fa7';
    ctx.fillRect(x,pad.t+ch-barH,bw,barH);
  }});
  // X labels
  for(let k=0;k<=4;k++){{
    const i=Math.round(k*(n-1)/4);
    ctx.fillStyle='#555';ctx.font='9px monospace';
    ctx.fillText(DISCOVERY_PTS[i].i,sx(i)-4,pad.t+ch+18);
  }}
}}

function drawHeatmap(id) {{
  const cv=document.getElementById(id); if(!cv||!HM_ITERS.length||!HM_SHARDS.length) return;
  const ctx=cv.getContext('2d');
  const W=cv.width, H=cv.height;
  const leftPad=120, topPad=16, botPad=24;
  const nR=HM_SHARDS.length, nC=HM_ITERS.length;
  if(!nR||!nC) return;
  const cellW=Math.min(24,Math.floor((W-leftPad)/nC));
  const cellH=Math.min(14,Math.floor((H-topPad-botPad)/nR));
  const gridW=cellW*nC, gridH=cellH*nR;
  ctx.fillStyle='#888'; ctx.font='8px monospace';
  HM_SHARDS.forEach((sid,r)=>{{
    ctx.fillStyle='#777'; ctx.font='8px monospace';
    ctx.fillText(sid.slice(-6),leftPad-66,topPad+r*cellH+cellH*0.75);
  }});
  HM_ITERS.forEach((it,c)=>{{
    ctx.fillStyle='#555'; ctx.font='8px monospace';
    ctx.fillText(it.iter,leftPad+c*cellW+(cellW/2-6),topPad+gridH+16);
    it.cells.forEach((v,r)=>{{
      ctx.fillStyle=v?'#4af':'#1e1e1e';
      ctx.fillRect(leftPad+c*cellW+1,topPad+r*cellH+1,cellW-2,cellH-2);
    }});
  }});
}}

drawTrend('condTrend', 'cond_gap', '#7df', COND_BAND, BEST_COND);
drawPareto('refTrend');
drawDiscovery('discoveryTrend');
drawHeatmap('heatmap');
</script>
</body></html>"""
