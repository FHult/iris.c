#!/usr/bin/env python3
"""
train/scripts/ablation_harness.py — QUALITY-10+: Long-term autonomous style-feature ablation.

Two operating modes:

  BATCH (original): Fixed matrix of combos, sequential, HTML report.
    python train/scripts/ablation_harness.py \\
        --matrix small --steps 300 --log-every 50 \\
        --shards /Volumes/2TBSSD/shards --output-dir /Volumes/2TBSSD/ablation_test

  LONG-TERM: SQLite-backed, Bayesian/random/grid search, fire-and-forget.
    python train/scripts/ablation_harness.py \\
        --config train/configs/ablation_v2.yaml \\
        --output-dir /Volumes/2TBSSD/ablation_long

  Report only (re-render HTML from existing DB without running):
    python train/scripts/ablation_harness.py \\
        --config train/configs/ablation_v2.yaml \\
        --output-dir /Volumes/2TBSSD/ablation_long --report-only

Harness config YAML (long-term mode):

    ablation:
      name: "sref-quality-v2"
      max_total_runs: 300
      steps_per_run: 12000
      strategy: "bayesian"           # grid | random | bayesian
      n_initial: 10                  # bayesian only: random explorations before GP kicks in
      objective:
        clip_i_weight: 0.55          # proxy for CLIP-I style fidelity (ref_gap)
        cross_ref_gap_weight: 0.30   # cond_gap: adapter learning signal
        stability_weight: 0.15       # training stability (low final loss)
      variables:
        cross_ref_prob: [0.0, 0.2, 0.35, 0.5]
        patch_shuffle_prob: [0.0, 0.25, 0.4, 0.5]
        freeze_double_stream_scales: [true, false]
        style_loss_weight: [0.0, 0.03, 0.07, 0.12]
      conditions:                    # optional: filter invalid param combinations
        - "style_loss_weight > 0 or cross_ref_prob == 0"

Control signals (long-term mode):
    pipeline_ctl start-ablation train/configs/ablation_v2.yaml
    pipeline_ctl ablation-status
    pipeline_ctl stop-ablation
    pipeline_ctl pause-ablation
    pipeline_ctl resume-ablation

Batch-mode scoring:
    score = 100 * ref_gap + 200 * cond_gap - 3 * final_loss

Long-term scoring (configurable weights):
    score = 100 * (clip_i_w * ref_norm + gap_w * cond_norm + stab_w * stab_norm)

Batch mode matrix presets:
    small  (4  combos): cross_ref=[0.3,0.5] x patch=[0.0,0.5]
    medium (12 combos): cross_ref=[0.0,0.3,0.5] x patch=[0.0,0.5] x freeze=[T,F]
    full   (54 combos): cross_ref x patch x freeze x style_loss_weight
"""

import argparse
import hashlib
import itertools
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_lib import ABLATION_CONTROL_FILE as _ABLATION_CONTROL  # noqa: E402

# ── Optional ML deps ─────────────────────────────────────────────────────────
try:
    import numpy as np
    _NUMPY_OK = True
except ImportError:
    _NUMPY_OK = False

try:
    import scipy.optimize as _scipy_opt
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

try:
    import optuna as _optuna
    _optuna.logging.set_verbosity(_optuna.logging.WARNING)
    _OPTUNA_OK = True
except ImportError:
    _OPTUNA_OK = False

# ── Repo layout ───────────────────────────────────────────────────────────────
_SCRIPT_DIR  = Path(__file__).resolve().parent
_TRAIN_DIR   = _SCRIPT_DIR.parent
_REPO_ROOT   = _TRAIN_DIR.parent
_VENV_PYTHON = _TRAIN_DIR / ".venv" / "bin" / "python"
_BASE_CONFIG = _TRAIN_DIR / "configs" / "stage1_512px.yaml"
_TRAINER     = _TRAIN_DIR / "train_ip_adapter.py"
_DATA_ROOT   = Path(os.environ.get("PIPELINE_DATA_ROOT", "/Volumes/2TBSSD"))
_DEFAULT_SHARDS  = _DATA_ROOT / "shards"
_DEFAULT_QWEN3   = _DATA_ROOT / "precomputed" / "qwen3"
_DEFAULT_VAE     = _DATA_ROOT / "precomputed" / "vae"
_DEFAULT_SIGLIP  = _DATA_ROOT / "precomputed" / "siglip"

_ABLATION_HB_PATH = _DATA_ROOT / ".heartbeat" / "ablation.json"

# ── Batch-mode matrix presets ─────────────────────────────────────────────────
MATRIX_PRESETS: dict[str, dict] = {
    "small": {
        "variables": {
            "cross_ref_prob":             [0.3, 0.5],
            "patch_shuffle_prob":         [0.0, 0.5],
            "freeze_double_stream_scales": [True],
            "style_loss_weight":          [0.05],
        },
    },
    "medium": {
        "variables": {
            "cross_ref_prob":             [0.0, 0.3, 0.5],
            "patch_shuffle_prob":         [0.0, 0.5],
            "freeze_double_stream_scales": [True, False],
            "style_loss_weight":          [0.05],
        },
    },
    "full": {
        "variables": {
            "cross_ref_prob":             [0.0, 0.3, 0.5],
            "patch_shuffle_prob":         [0.0, 0.3, 0.5],
            "freeze_double_stream_scales": [True, False],
            "style_loss_weight":          [0.0, 0.05, 0.1],
        },
    },
}

# ── Metric log regexes ────────────────────────────────────────────────────────
_RE_STEP  = re.compile(r"^step\s+([\d,]+)/([\d,]+)\s+loss\s+([\d.]+)\s+\(avg\s+([\d.]+)\)")
_RE_COND  = re.compile(r"loss_cond=([\d.]+)\s+loss_null=([\d.]+)\s+gap=([+-][\d.]+)")
_RE_REF   = re.compile(r"loss_ref:.*?self=([\d.]+)(?:.*?cross=([\d.]+).*?gap=([+-][\d.]+))?")
_RE_GRAD  = re.compile(r"grad_norm\s+([\d.]+)\s+\(smooth\s+([\d.]+)\)")
_RE_SCALE = re.compile(r"ip_scale:\s+mean=([\d.]+).*?double=([\d.]+).*?single=([\d.]+)")

# ── Console colours ───────────────────────────────────────────────────────────
_IS_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _IS_TTY:
        return text
    codes = {"green": "32", "yellow": "33", "red": "31", "cyan": "36",
             "bold": "1", "reset": "0", "dim": "2", "magenta": "35"}
    return f"\033[{codes.get(code, '0')}m{text}\033[0m"


# ── Metric collector ──────────────────────────────────────────────────────────

class MetricCollector:
    """Parses train_ip_adapter.py log lines and accumulates per-log-interval snapshots."""

    def __init__(self) -> None:
        self.snapshots: list[dict] = []
        self._pending: dict = {}

    def feed(self, line: str) -> Optional[dict]:
        m = _RE_STEP.search(line)
        if m:
            self._pending = {
                "step":        int(m.group(1).replace(",", "")),
                "loss":        float(m.group(3)),
                "loss_smooth": float(m.group(4)),
            }
            return None
        if not self._pending:
            return None
        m = _RE_COND.search(line)
        if m:
            self._pending["loss_cond"] = float(m.group(1))
            self._pending["loss_null"] = float(m.group(2))
            self._pending["cond_gap"]  = float(m.group(3))
            return None
        m = _RE_REF.search(line)
        if m:
            self._pending["loss_self_ref"] = float(m.group(1))
            if m.group(2) is not None:
                self._pending["loss_cross_ref"] = float(m.group(2))
                self._pending["ref_gap"]        = float(m.group(3))
            return None
        m = _RE_GRAD.search(line)
        if m:
            self._pending["grad_norm"]        = float(m.group(1))
            self._pending["grad_norm_smooth"] = float(m.group(2))
            return None
        m = _RE_SCALE.search(line)
        if m:
            self._pending["ip_scale_mean"]   = float(m.group(1))
            self._pending["ip_scale_double"] = float(m.group(2))
            self._pending["ip_scale_single"] = float(m.group(3))
            snap = dict(self._pending)
            self._pending = {}
            self.snapshots.append(snap)
            return snap
        return None


# ── Persistent SQLite history ─────────────────────────────────────────────────

class AblationDB:
    """SQLite-backed experiment history. Never forgets a run, never repeats a config."""

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS experiments (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            run_name     TEXT    NOT NULL,
            config_hash  TEXT    NOT NULL,
            strategy     TEXT,
            params       TEXT    NOT NULL,
            score        REAL,
            verdict      TEXT,
            ref_gap      REAL,
            cond_gap     REAL,
            final_loss   REAL,
            elapsed_secs INTEGER,
            steps        INTEGER,
            n_snapshots  INTEGER,
            exit_code    INTEGER,
            git_commit   TEXT,
            ts           TEXT    NOT NULL,
            snapshots    TEXT,
            is_pareto    INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_run  ON experiments(run_name);
        CREATE INDEX IF NOT EXISTS idx_hash ON experiments(run_name, config_hash);
        CREATE TABLE IF NOT EXISTS _meta (k TEXT PRIMARY KEY, v TEXT);
        INSERT OR IGNORE INTO _meta VALUES ('schema_version', '2');
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._conn.executescript(self.SCHEMA)
        # v1 → v2: add is_pareto column
        try:
            self._conn.execute(
                "ALTER TABLE experiments ADD COLUMN is_pareto INTEGER NOT NULL DEFAULT 0"
            )
        except Exception:
            pass
        self._conn.commit()

    @staticmethod
    def params_hash(params: dict) -> str:
        return hashlib.sha256(
            json.dumps(params, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

    def is_duplicate(self, run_name: str, params: dict) -> bool:
        h = self.params_hash(params)
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM experiments WHERE run_name=? AND config_hash=?",
                (run_name, h),
            ).fetchone()
        return row is not None

    def insert_experiment(self, run_name: str, params: dict, strategy: str, steps: int) -> int:
        h = self.params_hash(params)
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        gc = _get_git_commit()
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO experiments
                   (run_name, config_hash, strategy, params, steps, git_commit, ts)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (run_name, h, strategy, json.dumps(params, default=str), steps, gc, ts),
            )
            self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def update_experiment(
        self,
        exp_id: int,
        score: Optional[float],
        verdict: str,
        ref_gap: Optional[float],
        cond_gap: Optional[float],
        final_loss: Optional[float],
        elapsed_secs: int,
        n_snapshots: int,
        exit_code: int,
        snapshots: list,
    ) -> None:
        snaps_json = json.dumps(snapshots)
        with self._lock:
            self._conn.execute(
                """UPDATE experiments SET
                   score=?, verdict=?, ref_gap=?, cond_gap=?, final_loss=?,
                   elapsed_secs=?, n_snapshots=?, exit_code=?, snapshots=?
                   WHERE id=?""",
                (score, verdict, ref_gap, cond_gap, final_loss,
                 elapsed_secs, n_snapshots, exit_code, snaps_json, exp_id),
            )
            self._conn.commit()

    def get_experiments(self, run_name: str, scored_only: bool = False) -> list[dict]:
        q = "SELECT * FROM experiments WHERE run_name=?"
        if scored_only:
            q += " AND score IS NOT NULL"
        q += " ORDER BY id"
        with self._lock:
            rows = self._conn.execute(q, (run_name,)).fetchall()
        return [self._decode_row(r) for r in rows]

    def get_best(self, run_name: str, n: int = 5) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM experiments WHERE run_name=? AND score IS NOT NULL "
                "ORDER BY score DESC LIMIT ?",
                (run_name, n),
            ).fetchall()
        return [self._decode_row(r) for r in rows]

    def get_all_run_names(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT run_name FROM experiments ORDER BY run_name"
            ).fetchall()
        return [r[0] for r in rows]

    def update_pareto_front(self, run_name: str) -> int:
        """Recompute Pareto-efficient experiments (3-objective) and update is_pareto flag.

        Objectives: maximise ref_gap, maximise cond_gap, minimise final_loss.
        Returns the number of experiments on the Pareto front.
        """
        exps = self.get_experiments(run_name, scored_only=True)
        pareto_ids = _pareto_efficient(exps)
        with self._lock:
            self._conn.execute(
                "UPDATE experiments SET is_pareto=0 WHERE run_name=?", (run_name,)
            )
            for exp_id in pareto_ids:
                self._conn.execute(
                    "UPDATE experiments SET is_pareto=1 WHERE id=?", (exp_id,)
                )
            self._conn.commit()
        return len(pareto_ids)

    @staticmethod
    def _decode_row(row: sqlite3.Row) -> dict:
        r = dict(row)
        r["params"] = json.loads(r["params"])
        snaps_raw = r.get("snapshots")
        r["snapshots"] = json.loads(snaps_raw) if snaps_raw else []
        r["combo_id"] = f"exp_{r['id']:04d}"
        # Normalise to the field names expected by _render_html / _print_final_ranking
        r["mean_ref_gap"]  = r.get("ref_gap")
        r["mean_cond_gap"] = r.get("cond_gap")
        return r

    def close(self) -> None:
        self._conn.close()


# ── Search strategies ─────────────────────────────────────────────────────────

def _params_key(params: dict) -> str:
    return json.dumps(params, sort_keys=True, default=str)


class SearchStrategy:
    """Returns the next parameter dict to try, or None when exhausted."""
    def next_candidate(self, tried_results: list[dict]) -> Optional[dict]:
        raise NotImplementedError


class GridSearch(SearchStrategy):
    """Exhaustive grid: tries every candidate in fixed order."""
    def __init__(self, candidates: list[dict]) -> None:
        self._candidates = candidates

    def next_candidate(self, tried_results: list[dict]) -> Optional[dict]:
        tried_keys = {_params_key(t["params"]) for t in tried_results}
        for c in self._candidates:
            if _params_key(c) not in tried_keys:
                return c
        return None


class RandomSearch(SearchStrategy):
    """Random permutation of the candidate grid."""
    def __init__(self, candidates: list[dict], seed: Optional[int] = None) -> None:
        import random
        self._candidates = list(candidates)
        random.seed(seed)
        random.shuffle(self._candidates)

    def next_candidate(self, tried_results: list[dict]) -> Optional[dict]:
        tried_keys = {_params_key(t["params"]) for t in tried_results}
        for c in self._candidates:
            if _params_key(c) not in tried_keys:
                return c
        return None


class _NumpyGP:
    """Pure numpy/scipy Gaussian Process with Matern-5/2 kernel.

    Log marginal likelihood is optimised via scipy L-BFGS-B over log-space
    hyperparameters (length_scale, signal_variance, noise_variance).

    Only used when scikit-learn is not available.  Requires numpy + scipy.
    """

    def __init__(self) -> None:
        self._l      = 1.0    # length scale
        self._sf2    = 1.0    # signal variance
        self._sn2    = 0.1    # noise variance
        self._alpha  = None   # (K + sn2 I)^{-1} y_norm
        self._L      = None   # Cholesky factor
        self._Xtrain = None
        self._ymean  = 0.0
        self._ystd   = 1.0

    @staticmethod
    def _k52(X: "np.ndarray", Y: "np.ndarray", l: float) -> "np.ndarray":
        """Matern-5/2 kernel matrix, shapes [n,d] × [m,d] → [n,m]."""
        diff = X[:, None, :] - Y[None, :, :]            # [n, m, d]
        r    = np.sqrt(np.maximum(np.sum(diff ** 2, axis=-1), 0.0)) / l
        return (1.0 + np.sqrt(5.0) * r + 5.0 * r ** 2 / 3.0) * np.exp(-np.sqrt(5.0) * r)

    def _neg_lml(self, log_p: "np.ndarray", X: "np.ndarray", y: "np.ndarray") -> float:
        l, sf2, sn2 = np.exp(log_p[0]), np.exp(log_p[1]), np.exp(log_p[2])
        n   = X.shape[0]
        K   = sf2 * self._k52(X, X, l) + sn2 * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e9
        a   = np.linalg.solve(L.T, np.linalg.solve(L, y))
        lml = -0.5 * y @ a - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2.0 * np.pi)
        return float(-lml)

    def fit(self, X: "np.ndarray", y: "np.ndarray") -> None:
        self._ymean = float(y.mean())
        self._ystd  = float(y.std()) + 1e-8
        yn = (y - self._ymean) / self._ystd
        x0 = np.array([0.0, 0.0, -2.0])
        try:
            res = _scipy_opt.minimize(
                self._neg_lml, x0, args=(X, yn), method="L-BFGS-B",
                bounds=[(-3.0, 3.0), (-3.0, 3.0), (-6.0, 2.0)],
                options={"maxiter": 100, "ftol": 1e-4},
            )
            self._l, self._sf2, self._sn2 = (
                float(np.exp(res.x[0])), float(np.exp(res.x[1])), float(np.exp(res.x[2]))
            )
        except Exception:
            pass  # keep defaults if optimizer fails
        n = X.shape[0]
        K = self._sf2 * self._k52(X, X, self._l) + self._sn2 * np.eye(n)
        for jitter in (0.0, 1e-6, 1e-4, 1e-2):
            try:
                self._L = np.linalg.cholesky(K + jitter * np.eye(n))
                break
            except np.linalg.LinAlgError:
                continue
        else:
            self._L = np.eye(n)
        self._alpha  = np.linalg.solve(self._L.T, np.linalg.solve(self._L, yn))
        self._Xtrain = X.copy()

    def predict(self, Xstar: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
        Ks  = self._sf2 * self._k52(Xstar, self._Xtrain, self._l)  # [m, n]
        mu_n = Ks @ self._alpha
        v    = np.linalg.solve(self._L, Ks.T)                       # [n, m]
        var_n = np.maximum(self._sf2 - np.sum(v ** 2, axis=0), 0.0)
        return mu_n * self._ystd + self._ymean, np.sqrt(var_n) * self._ystd


class BayesianSearch(SearchStrategy):
    """GP-UCB Bayesian optimisation over the discrete candidate grid.

    Uses sklearn GaussianProcessRegressor when available; falls back to a pure
    numpy/scipy GP (_NumpyGP, Matern-5/2 with log-MLL optimisation) otherwise.

    Key improvement: X normalisation uses the FULL candidate range (computed at
    construction time), not the observed range.  Using the observed range causes
    UCB variance to collapse when all observed points share the same region,
    forcing random fallback even after n_initial runs have been collected.
    """

    def __init__(
        self,
        candidates: list[dict],
        n_initial: int = 10,
        kappa: float = 2.0,
    ) -> None:
        self._candidates   = candidates
        self._n_initial    = n_initial
        self._kappa        = kappa
        self._feature_keys = list(candidates[0].keys()) if candidates else []
        self._sklearn_gp   = None  # lazy init

        # Pre-compute full-range normalisation constants (stable across iterations)
        if candidates and _NUMPY_OK:
            all_X = np.array(
                [[float(c.get(k, 0)) for k in self._feature_keys] for c in candidates],
                dtype=np.float64,
            )
            self._X_min   = all_X.min(axis=0)
            self._X_max   = all_X.max(axis=0)
            self._X_range = np.where(self._X_max > self._X_min,
                                     self._X_max - self._X_min, 1.0)
        else:
            self._X_min = self._X_max = self._X_range = None

    def _encode(self, params: dict) -> "np.ndarray":
        return np.array([float(params.get(k, 0)) for k in self._feature_keys],
                        dtype=np.float64)

    def _normalize(self, X: "np.ndarray") -> "np.ndarray":
        if self._X_range is None:
            return X
        return (X - self._X_min) / self._X_range

    def next_candidate(self, tried_results: list[dict]) -> Optional[dict]:
        import random
        tried_keys = {_params_key(t["params"]) for t in tried_results}
        remaining  = [c for c in self._candidates if _params_key(c) not in tried_keys]
        if not remaining:
            return None

        scored = [t for t in tried_results if t.get("score") is not None]
        if len(scored) < self._n_initial or not _NUMPY_OK:
            return random.choice(remaining)

        X_obs  = np.array([self._encode(t["params"]) for t in scored])
        y_obs  = np.array([float(t["score"]) for t in scored])
        X_norm = self._normalize(X_obs)

        X_cand      = np.array([self._encode(c) for c in remaining])
        X_cand_norm = self._normalize(X_cand)

        try:
            if _SKLEARN_OK:
                if self._sklearn_gp is None:
                    kernel = (Matern(nu=2.5, length_scale_bounds=(1e-2, 10.0))
                              + WhiteKernel(noise_level=0.1))
                    self._sklearn_gp = GaussianProcessRegressor(
                        kernel=kernel, normalize_y=True, n_restarts_optimizer=3
                    )
                self._sklearn_gp.fit(X_norm, y_obs)
                mu, sigma = self._sklearn_gp.predict(X_cand_norm, return_std=True)
            else:
                gp = _NumpyGP()
                gp.fit(X_norm, y_obs)
                mu, sigma = gp.predict(X_cand_norm)

            ucb = mu + self._kappa * sigma
            return remaining[int(np.argmax(ucb))]
        except Exception:
            return random.choice(remaining)


class OptunaSearch(SearchStrategy):
    """Optuna TPE Bayesian search over a discrete candidate grid.

    Rebuilds the study from scored history on each call (stateless between calls).
    CategoricalDistribution is used for all variables so booleans and float lists
    work without encoding assumptions.

    `candidates` is the conditions-filtered list (same as BayesianSearch receives).
    `variables` is the raw dict used only for building Optuna distributions.

    Falls back to random choice when optuna is not installed or when fewer than
    n_initial scored results exist.
    """

    def __init__(self, variables: dict, candidates: list[dict],
                 n_initial: int = 10) -> None:
        self._variables  = variables
        self._candidates = candidates   # conditions-filtered; used for remaining list
        self._n_initial  = n_initial
        self._dists: dict = {}
        if _OPTUNA_OK:
            for k, vals in variables.items():
                self._dists[k] = _optuna.distributions.CategoricalDistribution(
                    [str(v) for v in vals]
                )

    def _decode(self, optuna_params: dict) -> dict:
        result = {}
        for k, vals in self._variables.items():
            sv = optuna_params.get(k, str(vals[0]))
            for v in vals:
                if str(v) == sv:
                    result[k] = v
                    break
            else:
                result[k] = vals[0]
        return result

    def next_candidate(self, tried_results: list[dict]) -> Optional[dict]:
        import random
        tried_keys = {_params_key(t["params"]) for t in tried_results}
        # Use the pre-filtered candidate list (respects conditions)
        remaining = [c for c in self._candidates if _params_key(c) not in tried_keys]
        if not remaining:
            return None

        scored = [r for r in tried_results if r.get("score") is not None]
        if len(scored) < self._n_initial or not _OPTUNA_OK:
            return random.choice(remaining)

        sampler = _optuna.samplers.TPESampler(n_startup_trials=self._n_initial, seed=42)
        study   = _optuna.create_study(direction="maximize", sampler=sampler)

        for r in scored:
            p = r["params"]
            try:
                t = _optuna.trial.create_trial(
                    params={k: str(p.get(k, vals[0])) for k, vals in self._variables.items()},
                    distributions=self._dists,
                    value=float(r["score"]),
                )
                study.add_trial(t)
            except Exception:
                continue

        try:
            trial  = study.ask(fixed_distributions=self._dists)
            params = self._decode(trial.params)
            # If Optuna suggests a filtered-out or already-tried combo, fall back to random
            if _params_key(params) in tried_keys or params not in remaining:
                return random.choice(remaining)
            return params
        except Exception:
            return random.choice(remaining)


# ── Heartbeat ─────────────────────────────────────────────────────────────────

class HeartbeatWriter:
    """Writes a JSON heartbeat file every 60 s from a background thread."""

    def __init__(self, run_name: str, path: Path = _ABLATION_HB_PATH) -> None:
        self._path = path
        self._run_name = run_name
        self._state: dict = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def update(self, **fields) -> None:
        with self._lock:
            self._state.update(fields)
        self._write()

    def _write(self) -> None:
        with self._lock:
            payload = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "process": "ablation",
                "run_name": self._run_name,
                **self._state,
            }
        tmp = self._path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.rename(self._path)
        except OSError:
            pass

    def _loop(self) -> None:
        while not self._stop_event.wait(60):
            self._write()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._path.unlink(missing_ok=True)
        except OSError:
            pass


# ── Control signals ───────────────────────────────────────────────────────────

def _read_control() -> str:
    """Returns 'run', 'pause', or 'stop'."""
    try:
        ctrl = json.loads(_ABLATION_CONTROL.read_text())
        return ctrl.get("action", "run")
    except (OSError, json.JSONDecodeError):
        return "run"


def _wait_if_paused(hb: HeartbeatWriter) -> bool:
    """Block while paused. Returns False if a stop signal arrives."""
    first = True
    while True:
        action = _read_control()
        if action == "stop":
            return False
        if action == "run":
            return True
        if first:
            print("  [paused — waiting for resume signal]", flush=True)
            hb.update(status="paused")
            first = False
        time.sleep(15)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_REPO_ROOT), stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _eval_conditions(params: dict, conditions: list[str]) -> bool:
    """Return True if all conditions pass for this param set. Fails open on errors."""
    for cond in conditions:
        try:
            if not eval(cond, {"__builtins__": {}}, dict(params)):  # noqa: S307
                return False
        except Exception:
            pass  # malformed condition — don't filter this candidate
    return True


# ── Matrix generation ─────────────────────────────────────────────────────────

def _generate_combos(matrix_def: dict) -> list[dict]:
    variables = matrix_def.get("variables", {})
    if not variables:
        return []
    keys = list(variables.keys())
    values_list = [variables[k] for k in keys]
    combos = []
    for i, vals in enumerate(itertools.product(*values_list)):
        combos.append({
            "combo_id": f"combo_{i + 1:03d}",
            "params":   dict(zip(keys, vals)),
        })
    return combos


def _load_matrix(args) -> dict:
    if args.matrix_file:
        p = Path(args.matrix_file)
        if not p.exists():
            print(f"ERROR: --matrix-file not found: {p}", file=sys.stderr)
            sys.exit(1)
        with open(p) as f:
            raw = yaml.safe_load(f)
        return raw.get("ablation", raw)
    name = args.matrix or "small"
    if name not in MATRIX_PRESETS:
        print(f"ERROR: unknown matrix preset '{name}'. "
              f"Available: {list(MATRIX_PRESETS)}", file=sys.stderr)
        sys.exit(1)
    return MATRIX_PRESETS[name]


# ── Config builder ────────────────────────────────────────────────────────────

def _build_run_config(
    base_config_path: Path,
    shards: str,
    qwen3_cache: Optional[str],
    vae_cache: Optional[str],
    siglip_cache: Optional[str],
    checkpoint_dir: str,
    steps: int,
    log_every: int,
    params: dict,
) -> dict:
    with open(base_config_path) as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("data", {})
    cfg["data"]["shard_path"]       = shards
    cfg["data"]["qwen3_cache_dir"]  = qwen3_cache
    cfg["data"]["vae_cache_dir"]    = vae_cache
    cfg["data"]["siglip_cache_dir"] = siglip_cache
    cfg["data"]["anchor_shard_dir"] = None
    cfg["data"]["hard_example_dir"] = None
    cfg["data"]["prefetch_batches"] = 4
    cfg["data"]["num_prefetch_threads"] = 1

    cfg.setdefault("training", {})
    cfg["training"]["num_steps"]     = steps
    cfg["training"]["warmup_steps"]  = min(cfg["training"].get("warmup_steps", 1000), steps // 5)
    cfg["training"]["style_loss_every"] = 1

    cfg.setdefault("adapter", {})

    for key, val in params.items():
        if key == "freeze_double_stream_scales":
            cfg["adapter"]["freeze_double_stream_scales"] = val
        elif key in ("cross_ref_prob", "patch_shuffle_prob",
                     "style_loss_weight", "learning_rate"):
            cfg["training"][key] = val
        else:
            cfg["training"][key] = val

    cfg.setdefault("output", {})
    cfg["output"]["checkpoint_dir"]       = checkpoint_dir
    cfg["output"]["log_every"]            = log_every
    cfg["output"]["checkpoint_every"]     = steps * 100
    cfg["output"]["keep_last_n"]          = 1
    cfg["output"]["skip_checkpoint_save"] = True

    cfg.setdefault("eval", {})
    cfg["eval"]["enabled"] = False

    return cfg


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score(snapshots: list[dict], exit_code: int) -> float:
    """Batch-mode fixed scoring: 100*ref_gap + 200*cond_gap - 3*final_loss."""
    if exit_code != 0 or not snapshots:
        return float("-inf")
    n_skip = max(0, len(snapshots) * 2 // 5)
    tail = snapshots[n_skip:] or snapshots
    ref_gaps  = [s["ref_gap"]    for s in tail if "ref_gap"    in s]
    cond_gaps = [s["cond_gap"]   for s in tail if "cond_gap"   in s]
    loss_vals = [s["loss_smooth"] for s in tail if "loss_smooth" in s]
    mean_ref   = sum(ref_gaps)  / len(ref_gaps)  if ref_gaps  else 0.0
    mean_cond  = sum(cond_gaps) / len(cond_gaps) if cond_gaps else 0.0
    final_loss = loss_vals[-1] if loss_vals else 9.9
    return 100.0 * mean_ref + 200.0 * mean_cond - 3.0 * final_loss


def _score_weighted(snapshots: list[dict], exit_code: int, objective: dict) -> float:
    """Long-term mode scoring with configurable objective weights.

    Each component is normalised to approximately [-1, +1] then weighted:
      clip_i_w    → mean_ref_gap  (style fidelity proxy; typical ±0.5 → ×2)
      gap_w       → mean_cond_gap (adapter learning; typical [-2, 0.5] → /2.5)
      stability_w → -(final_loss - 1) / 4  (stability; typical 0.5–5.0)
    """
    if exit_code != 0 or not snapshots:
        return float("-inf")

    clip_i_w    = float(objective.get("clip_i_weight", 0.55))
    gap_w       = float(objective.get("cross_ref_gap_weight", 0.30))
    stability_w = float(objective.get("stability_weight", 0.15))

    n_skip = max(0, len(snapshots) * 2 // 5)
    tail = snapshots[n_skip:] or snapshots

    ref_gaps  = [s["ref_gap"]    for s in tail if "ref_gap"    in s]
    cond_gaps = [s["cond_gap"]   for s in tail if "cond_gap"   in s]
    losses    = [s["loss_smooth"] for s in tail if "loss_smooth" in s]

    mean_ref   = sum(ref_gaps)  / len(ref_gaps)  if ref_gaps  else 0.0
    mean_cond  = sum(cond_gaps) / len(cond_gaps) if cond_gaps else 0.0
    final_loss = losses[-1] if losses else 9.9

    ref_norm  = max(-1.0, min(1.0, mean_ref  * 2.0))
    cond_norm = max(-1.0, min(1.0, mean_cond / 2.5))
    stab_norm = max(-1.0, min(0.5, -(final_loss - 1.0) / 4.0))

    return (clip_i_w * ref_norm + gap_w * cond_norm + stability_w * stab_norm) * 100.0


def _verdict(snapshots: list[dict], exit_code: int) -> str:
    if exit_code != 0:
        return "CRASH"
    if not snapshots:
        return "NO_DATA"
    tail = snapshots[max(0, len(snapshots) * 2 // 5):]
    ref_gaps  = [s["ref_gap"]  for s in tail if "ref_gap"  in s]
    cond_gaps = [s["cond_gap"] for s in tail if "cond_gap" in s]
    loss_vals = [s.get("loss_smooth", 0) for s in tail]
    if loss_vals and loss_vals[-1] > 5.0:
        return "UNSTABLE"
    if ref_gaps and sum(r > 0 for r in ref_gaps) > len(ref_gaps) * 0.5:
        if cond_gaps and sum(c > 0 for c in cond_gaps) > len(cond_gaps) * 0.5:
            return "PASS"
        return "WARN"
    return "WARN"


# ── Best-config export ────────────────────────────────────────────────────────

def _export_best_config(result: dict, output_dir: Path, run_name: str = "") -> None:
    """Write a ready-to-use training config YAML for the best result."""
    params = result.get("params", {})
    out = {
        "ablation_source": {
            "run_name":  run_name,
            "combo_id":  result.get("combo_id"),
            "score":     result.get("score"),
            "ref_gap":   result.get("mean_ref_gap"),
            "cond_gap":  result.get("mean_cond_gap"),
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "training": {},
        "adapter":  {},
    }
    for k, v in params.items():
        if k == "freeze_double_stream_scales":
            out["adapter"][k] = v
        else:
            out["training"][k] = v

    path = output_dir / "best_config.yaml"
    try:
        with open(path, "w") as f:
            yaml.dump(out, f, default_flow_style=False, sort_keys=False)
        print(f"  Best config exported: {path}")
    except OSError as e:
        print(f"WARNING: could not write best_config.yaml: {e}", file=sys.stderr)


# ── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopper:
    """Monitor training snapshots and SIGTERM the subprocess when cond_gap is persistently bad.

    Activates only after `min_snapshots` snapshots have been collected (warmup guard).
    After `patience` consecutive snapshots with cond_gap < min_cond_gap, the process
    receives SIGTERM and the run is recorded as an early-stopped result.
    """

    def __init__(self, min_cond_gap: float, patience: int, min_snapshots: int) -> None:
        self.min_cond_gap  = min_cond_gap
        self.patience      = patience
        self.min_snapshots = min_snapshots
        self._proc: Optional[subprocess.Popen] = None
        self._n_snapshots  = 0
        self._bad_streak   = 0
        self._triggered    = False

    def attach(self, proc: subprocess.Popen) -> None:
        """Wire up immediately after Popen()."""
        self._proc        = proc
        self._n_snapshots = 0
        self._bad_streak  = 0
        self._triggered   = False

    def feed_snapshot(self, snap: dict) -> bool:
        """Returns True if early stopping was triggered this call or previously."""
        if self._triggered:
            return True
        self._n_snapshots += 1
        if self._n_snapshots <= self.min_snapshots:
            return False
        cond_gap = snap.get("cond_gap")
        if cond_gap is None:
            return False
        if cond_gap < self.min_cond_gap:
            self._bad_streak += 1
        else:
            self._bad_streak = 0
        if self._bad_streak >= self.patience:
            self._triggered = True
            if self._proc is not None and self._proc.poll() is None:
                try:
                    import signal as _sig
                    self._proc.send_signal(_sig.SIGTERM)
                except Exception:
                    pass
            return True
        return False


# ── Campaign-level plateau detection ─────────────────────────────────────────

class CampaignPlateau:
    """Stop the whole campaign when the best score has not improved for N runs.

    Distinct from EarlyStopper (which kills a single bad run within that run).
    This watches the *campaign-level* best score across completed experiments.
    Activates only after min_runs to avoid reacting to noisy early exploration.
    """

    def __init__(self, patience: int, min_delta: float = 0.01, min_runs: int = 5) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.min_runs  = min_runs
        self._best:  Optional[float] = None
        self._stale  = 0
        self._n_runs = 0

    def update(self, score: Optional[float]) -> bool:
        """Feed latest run score. Returns True when plateau is detected."""
        self._n_runs += 1
        if score is None:
            return False
        improved = self._best is None or score > self._best + self.min_delta
        if improved:
            self._best  = score
            self._stale = 0
        else:
            self._stale += 1
        if self._n_runs < self.min_runs:
            return False
        return self._stale >= self.patience

    @property
    def stale_count(self) -> int:
        return self._stale

    @property
    def best_score(self) -> Optional[float]:
        return self._best

    def status(self) -> str:
        if self._best is None:
            return "no data"
        return f"best={self._best:.3f}  stale={self._stale}/{self.patience}"


# ── Single-run execution ──────────────────────────────────────────────────────

def _run_one(
    combo: dict,
    run_dir: Path,
    args,
    log_every: int,
    quiet: bool = False,
    hb: Optional[HeartbeatWriter] = None,
    early_stopper: Optional[EarlyStopper] = None,
) -> dict:
    combo_id = combo["combo_id"]
    params   = combo["params"]

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    log_path = run_dir / "training.log"

    cfg = _build_run_config(
        base_config_path=Path(args.base_config),
        shards=args.shards,
        qwen3_cache=getattr(args, "qwen3_cache", None),
        vae_cache=getattr(args, "vae_cache", None),
        siglip_cache=getattr(args, "siglip_cache", None),
        checkpoint_dir=str(ckpt_dir),
        steps=args.steps,
        log_every=log_every,
        params=params,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False,
                                     prefix=f"ablation_{combo_id}_") as tf:
        yaml.dump(cfg, tf)
        tmp_cfg = tf.name

    try:
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(cfg, f)
    except OSError:
        pass

    cmd = [
        str(_VENV_PYTHON), "-u", str(_TRAINER),
        "--config", tmp_cfg,
        "--max-steps", str(args.steps),
        "--log-every", str(log_every),
    ]

    collector = MetricCollector()
    t_start = time.time()
    exit_code = -1

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(_REPO_ROOT),
        )
        if early_stopper is not None:
            early_stopper.attach(proc)
        with open(log_path, "w") as log_f:
            for raw_line in proc.stdout:  # type: ignore[union-attr]
                line = raw_line.rstrip()
                log_f.write(raw_line)
                log_f.flush()
                if not quiet:
                    print(f"  {line}", flush=True)
                snap = collector.feed(line)
                if snap is not None:
                    ref_gap  = snap.get("ref_gap", 0.0)
                    cond_gap = snap.get("cond_gap", 0.0)
                    loss     = snap.get("loss_smooth", snap.get("loss", 0.0))
                    flag = "✓" if ref_gap > 0 else "○"
                    if not quiet:
                        print(_c("dim", f"  ↳ step {snap['step']:>6}  "
                                  f"loss={loss:.4f}  "
                                  f"ref_gap={ref_gap:+.4f}{flag}  "
                                  f"cond_gap={cond_gap:+.4f}"), flush=True)
                    if hb is not None:
                        hb.update(
                            current_step=snap["step"],
                            current_loss=round(loss, 4),
                            current_ref_gap=round(ref_gap, 4),
                        )
                    if early_stopper is not None and early_stopper.feed_snapshot(snap):
                        print(_c("yellow",
                                 f"  ↳ early stopping triggered at step {snap['step']} "
                                 f"(cond_gap={cond_gap:+.4f})"), flush=True)
        proc.wait()
        exit_code = proc.returncode
    except KeyboardInterrupt:
        print(f"\n  [interrupted — saving partial results for {combo_id}]")
        try:
            proc.terminate(); proc.wait(timeout=5)
        except Exception:
            pass
        exit_code = -2
    except Exception as exc:
        print(f"  FATAL: failed to launch trainer: {exc}", file=sys.stderr)
        exit_code = -1
    finally:
        try:
            os.unlink(tmp_cfg)
        except OSError:
            pass

    elapsed = time.time() - t_start
    batch_score = _score(collector.snapshots, exit_code)
    verdict = _verdict(collector.snapshots, exit_code)

    n_skip = max(0, len(collector.snapshots) * 2 // 5)
    tail = collector.snapshots[n_skip:] or collector.snapshots
    last = collector.snapshots[-1] if collector.snapshots else {}

    result = {
        "combo_id":       combo_id,
        "params":         params,
        "score":          round(batch_score, 4) if batch_score != float("-inf") else None,
        "verdict":        verdict,
        "exit_code":      exit_code,
        "elapsed_secs":   round(elapsed),
        "n_snapshots":    len(collector.snapshots),
        "final_loss":     last.get("loss_smooth"),
        "final_ref_gap":  last.get("ref_gap"),
        "final_cond_gap": last.get("cond_gap"),
        "mean_ref_gap":   round(sum(s["ref_gap"] for s in tail if "ref_gap" in s) /
                                max(1, sum(1 for s in tail if "ref_gap" in s)), 4)
                          if any("ref_gap" in s for s in tail) else None,
        "mean_cond_gap":  round(sum(s["cond_gap"] for s in tail if "cond_gap" in s) /
                                max(1, sum(1 for s in tail if "cond_gap" in s)), 4)
                          if any("cond_gap" in s for s in tail) else None,
        "snapshots":      collector.snapshots,
        "log_path":       str(log_path),
    }

    try:
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)
    except OSError:
        pass

    keep_ckpts = getattr(args, "keep_checkpoints", False)
    if not keep_ckpts and ckpt_dir.exists():
        try:
            shutil.rmtree(ckpt_dir)
        except OSError:
            pass

    return result


# ── Long-term harness config loading ─────────────────────────────────────────

def _load_harness_config(path: Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = raw.get("ablation", raw)
    # Validate required keys
    if "variables" not in cfg:
        print(f"ERROR: harness config missing 'variables' section: {path}", file=sys.stderr)
        sys.exit(1)
    return cfg


def _build_search_strategy(cfg: dict, candidates: list[dict]) -> SearchStrategy:
    strategy  = cfg.get("strategy", "random").lower()
    n_initial = int(cfg.get("n_initial", max(8, len(candidates) // 5)))
    variables = cfg.get("variables", {})
    if strategy == "grid":
        return GridSearch(candidates)
    if strategy == "random":
        return RandomSearch(candidates, seed=cfg.get("seed"))
    if strategy == "bayesian":
        # Prefer Optuna TPE (handles discrete/categorical better than GP)
        if _OPTUNA_OK and variables:
            print("  Bayesian backend: Optuna TPE", flush=True)
            return OptunaSearch(variables, candidates=candidates, n_initial=n_initial)
        # Fall back to GP-UCB
        if not _NUMPY_OK:
            print("WARNING: numpy not available — falling back to random search",
                  file=sys.stderr)
            return RandomSearch(candidates)
        if not (_SKLEARN_OK or _SCIPY_OK):
            print("WARNING: neither scikit-learn nor scipy available — "
                  "falling back to random search", file=sys.stderr)
            return RandomSearch(candidates)
        if not _SKLEARN_OK:
            print("INFO: scikit-learn not available — using pure numpy/scipy GP",
                  file=sys.stderr)
        print("  Bayesian backend: GP-UCB (sklearn)", flush=True)
        return BayesianSearch(candidates, n_initial=n_initial)
    print(f"ERROR: unknown strategy '{strategy}'. Use: grid, random, bayesian", file=sys.stderr)
    sys.exit(1)


# ── Warm-start helper ────────────────────────────────────────────────────────

def _warm_start_candidate(warm_start_dir: Path, run_name: str,
                           current_db: "AblationDB") -> Optional[dict]:
    """Load the best params from a prior campaign's DB as a forced first candidate.

    Searches `warm_start_dir/ablation_history.db` for the highest-scored experiment
    matching `run_name`, or any run if no match.  Returns None if the params are
    already in the current DB or the prior DB cannot be read.
    """
    db_path = warm_start_dir / "ablation_history.db"
    if not db_path.exists():
        print(f"WARNING: --warm-start-from DB not found: {db_path}", file=sys.stderr)
        return None
    try:
        prior = AblationDB(db_path)
        best_list = prior.get_best(run_name, 1)
        if not best_list:
            for name in prior.get_all_run_names():
                best_list = prior.get_best(name, 1)
                if best_list:
                    break
        prior.close()
        if not best_list:
            print("WARNING: --warm-start-from DB has no scored experiments", file=sys.stderr)
            return None
        exp    = best_list[0]
        params = exp["params"]
        if current_db.is_duplicate(run_name, params):
            print(f"  Warm-start: best prior params already in current DB — skipping injection")
            return None
        print(f"  Warm-start: injecting best params from prior campaign "
              f"(score={exp.get('score'):.3f}  run={exp.get('run_name')})")
        return params
    except Exception as exc:
        print(f"WARNING: --warm-start-from failed: {exc}", file=sys.stderr)
        return None


# ── Long-term run loop ────────────────────────────────────────────────────────

def run_long_term(harness_cfg: dict, db: AblationDB, cli_args) -> None:
    """Fire-and-forget long-term loop.  Runs until max_total_runs or stop signal."""
    run_name   = harness_cfg.get("name", "default")
    max_runs   = int(harness_cfg.get("max_total_runs", 100))
    steps      = int(harness_cfg.get("steps_per_run", 8000))
    strategy   = harness_cfg.get("strategy", "random")
    objective  = harness_cfg.get("objective", {})
    variables  = harness_cfg.get("variables", {})
    conditions = harness_cfg.get("conditions", [])
    log_every  = max(50, steps // 80)
    output_dir = Path(cli_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build candidate pool
    raw_combos = _generate_combos({"variables": variables})
    candidates = [c["params"] for c in raw_combos]
    if conditions:
        before = len(candidates)
        candidates = [c for c in candidates if _eval_conditions(c, conditions)]
        print(f"  Conditions filtered: {before} → {len(candidates)} candidates")

    if not candidates:
        print("ERROR: all candidates filtered by conditions", file=sys.stderr)
        sys.exit(1)

    searcher = _build_search_strategy(harness_cfg, candidates)

    # Per-run early stopping
    es_cfg = harness_cfg.get("early_stopping", {})
    early_stopper: Optional[EarlyStopper] = None
    if es_cfg.get("enabled", False):
        early_stopper = EarlyStopper(
            min_cond_gap=float(es_cfg.get("min_cond_gap", -0.3)),
            patience=int(es_cfg.get("patience", 4)),
            min_snapshots=int(es_cfg.get("min_snapshots", 5)),
        )

    # Campaign-level plateau detection
    pd_cfg = harness_cfg.get("plateau_detection", {})
    plateau: Optional[CampaignPlateau] = None
    if pd_cfg.get("enabled", False):
        plateau = CampaignPlateau(
            patience=int(pd_cfg.get("patience", 8)),
            min_delta=float(pd_cfg.get("min_delta", 0.01)),
            min_runs=int(pd_cfg.get("min_runs", 5)),
        )

    hb = HeartbeatWriter(run_name)
    hb.start()

    # Load what's already in the DB (tried = all experiments, scored or not)
    all_experiments = db.get_experiments(run_name)
    tried = list(all_experiments)
    n_done = len(all_experiments)

    # Pre-feed existing scores into plateau detector so it resumes correctly
    if plateau is not None:
        for exp in all_experiments:
            plateau.update(exp.get("score"))

    # Warm-start: inject best params from a prior campaign as the first candidate
    _ws_params: Optional[dict] = None
    ws_dir = getattr(cli_args, "warm_start_from", None)
    if ws_dir:
        _ws_params = _warm_start_candidate(Path(ws_dir), run_name, db)

    print()
    print(_c("cyan", f"{'═'*64}"))
    print(_c("bold", f"  Long-Term Ablation — {run_name}"))
    print(_c("cyan", f"{'═'*64}"))
    print(f"  strategy={strategy}  steps={steps}  max_runs={max_runs}")
    print(f"  candidates={len(candidates)}  already_done={n_done}")
    print(f"  db: {db._db_path}")
    print(f"  output: {output_dir}")
    if early_stopper is not None:
        print(f"  early_stopping: min_cond_gap={early_stopper.min_cond_gap}  "
              f"patience={early_stopper.patience}  "
              f"min_snapshots={early_stopper.min_snapshots}")
    if plateau is not None:
        print(f"  plateau_detection: patience={plateau.patience}  "
              f"min_delta={plateau.min_delta}  min_runs={plateau.min_runs}")
    if _ws_params is not None:
        print(f"  warm_start: {_ws_params}")
    print(_c("cyan", f"{'═'*64}\n"))

    hb.update(status="running", strategy=strategy, n_candidates=len(candidates),
               n_done=n_done, n_max=max_runs)
    force_continue = getattr(cli_args, "force_continue", False)

    while n_done < max_runs:
        if not _wait_if_paused(hb):
            print("\n  Stop signal received — exiting")
            break

        # Warm-start injection: override searcher for the first untried candidate
        if _ws_params is not None:
            params     = _ws_params
            _ws_params = None
        else:
            params = searcher.next_candidate(tried)
        if params is None:
            print("\n  Candidate pool exhausted — all combinations tried")
            break

        # Double-check DB (another process could have run this)
        if db.is_duplicate(run_name, params):
            tried.append({"params": params, "score": None})
            continue

        n_done += 1
        combo_id = f"exp_{n_done:04d}"
        combo = {"combo_id": combo_id, "params": params}

        hb.update(status="running", current_combo=combo_id,
                  n_done=n_done, n_max=max_runs, params=params)

        exp_id = db.insert_experiment(run_name, params, strategy, steps)

        run_dir = output_dir / "runs" / run_name / combo_id
        run_args = argparse.Namespace(
            base_config=getattr(cli_args, "base_config", str(_BASE_CONFIG)),
            shards=cli_args.shards,
            qwen3_cache=getattr(cli_args, "qwen3_cache", None),
            vae_cache=getattr(cli_args, "vae_cache", None),
            siglip_cache=getattr(cli_args, "siglip_cache", None),
            steps=steps,
            keep_checkpoints=False,
        )

        _print_combo_header(combo, n_done, max_runs, steps)
        result = _run_one(combo, run_dir, run_args, log_every, quiet=False, hb=hb,
                          early_stopper=early_stopper)

        # Re-score with configurable objective
        weighted_score = _score_weighted(result.get("snapshots", []), result["exit_code"], objective)
        result["score"] = round(weighted_score, 4) if weighted_score != float("-inf") else None

        db.update_experiment(
            exp_id=exp_id,
            score=result["score"],
            verdict=result["verdict"],
            ref_gap=result.get("mean_ref_gap"),
            cond_gap=result.get("mean_cond_gap"),
            final_loss=result.get("final_loss"),
            elapsed_secs=result.get("elapsed_secs", 0),
            n_snapshots=result.get("n_snapshots", 0),
            exit_code=result["exit_code"],
            snapshots=result.get("snapshots", []),
        )
        db.update_pareto_front(run_name)

        tried.append({"params": params, "score": result["score"]})

        _print_result_line(result)

        # Campaign plateau check
        if plateau is not None:
            if plateau.update(result.get("score")) and not force_continue:
                print(_c("yellow",
                         f"\n  Campaign plateau detected — {plateau.status()}"))
                print("  No improvement for the last "
                      f"{plateau.patience} runs.  Use --force-continue to override.")
                hb.update(status="plateau_stopped", plateau=plateau.status())
                break
            elif plateau.stale_count > 0:
                print(_c("dim",
                         f"  plateau: {plateau.status()}"), flush=True)

        # Regenerate report after every run
        all_experiments = db.get_experiments(run_name)
        _generate_lt_report(all_experiments, output_dir, run_name, steps, objective)

        if result.get("exit_code") == -2:  # KeyboardInterrupt inside _run_one
            print("\n  Interrupted — exiting")
            break

    # Final
    all_experiments = db.get_experiments(run_name)
    _print_final_ranking(all_experiments)
    _generate_lt_report(all_experiments, output_dir, run_name, steps, objective)

    best_list = db.get_best(run_name, 1)
    if best_list:
        _export_best_config(best_list[0], output_dir, run_name)

    _stop_reason = "done"
    if plateau is not None and plateau.stale_count >= plateau.patience and not force_continue:
        _stop_reason = f"plateau ({plateau.status()})"
    hb.update(status=_stop_reason, n_done=n_done,
              plateau=plateau.status() if plateau else None)
    hb.stop()


def _pareto_efficient(experiments: list[dict]) -> set:
    """Return set of experiment IDs on the 3-objective Pareto front.

    Objectives: maximise ref_gap, maximise cond_gap, minimise final_loss.
    An experiment is Pareto-efficient if no other scored experiment dominates
    it on all three objectives simultaneously.
    """
    valid = [e for e in experiments
             if e.get("ref_gap") is not None
             and e.get("cond_gap") is not None
             and e.get("final_loss") is not None]
    if not valid:
        return set()

    pareto_ids: set = set()
    for j, ej in enumerate(valid):
        dominated = False
        for k, ek in enumerate(valid):
            if k == j:
                continue
            # ek dominates ej if ek is ≥ ej on all objectives AND strictly > on at least one
            if (ek["ref_gap"]    >= ej["ref_gap"]    and
                    ek["cond_gap"]   >= ej["cond_gap"]   and
                    ek["final_loss"] <= ej["final_loss"]  and
                    (ek["ref_gap"]    > ej["ref_gap"]    or
                     ek["cond_gap"]   > ej["cond_gap"]   or
                     ek["final_loss"] < ej["final_loss"])):
                dominated = True
                break
        if not dominated:
            pareto_ids.add(ej["id"])

    return pareto_ids


def _generate_lt_report(
    experiments: list[dict],
    output_dir: Path,
    run_name: str,
    steps: int,
    objective: dict,
) -> None:
    """Write index.html from DB experiments for the long-term run."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ranked = sorted(
        [e for e in experiments if e.get("score") is not None],
        key=lambda e: e["score"],
        reverse=True,
    )
    total_elapsed = sum(e.get("elapsed_secs", 0) for e in experiments)
    html = _render_html(
        results=experiments,
        matrix_name=run_name,
        steps=steps,
        ts=ts,
        total_elapsed=total_elapsed,
        run_dir_name=run_name,
        objective=objective,
    )
    try:
        with open(output_dir / "index.html", "w") as f:
            f.write(html)
    except OSError as e:
        print(f"WARNING: could not write report: {e}", file=sys.stderr)


# ── HTML report ───────────────────────────────────────────────────────────────

_PALETTE = [
    "#7af", "#7f7", "#f77", "#fa7", "#c7f", "#7fc", "#ff7",
    "#f7c", "#7cf", "#fc7", "#a7f", "#7fa",
]


def _html_color(rank: int) -> str:
    if rank == 1:
        return "#7f7"
    if rank == 2:
        return "#7af"
    if rank == 3:
        return "#fa7"
    idx = rank - 1
    if idx < len(_PALETTE):
        return _PALETTE[idx]
    hue = (idx * 137) % 360
    return f"hsl({hue},55%,60%)"


def _render_html(
    results: list[dict],
    matrix_name: str,
    steps: int,
    ts: str,
    total_elapsed: int,
    run_dir_name: str,
    objective: Optional[dict] = None,
) -> str:
    ranked = sorted(
        [r for r in results if r.get("score") is not None],
        key=lambda r: r["score"],
        reverse=True,
    )
    crashed = [r for r in results if r.get("score") is None]
    all_sorted = ranked + crashed

    def _fmt(v, fmt=".4f"):
        return f"{v:{fmt}}" if v is not None else "—"

    def _verdict_style(v):
        return {"PASS": "color:#7f7", "WARN": "color:#fa7",
                "CRASH": "color:#f77", "UNSTABLE": "color:#f77",
                "NO_DATA": "color:#888"}.get(v, "")

    rows_html = ""
    for rank_i, r in enumerate(all_sorted):
        rank_disp = rank_i + 1 if r.get("score") is not None else "—"
        col = _html_color(rank_i + 1 if isinstance(rank_disp, int) else 99)
        p = r["params"]
        params_str = "  ".join(
            f"{k.replace('freeze_double_stream_scales','freeze').replace('_prob','').replace('_weight','_w')}="
            f"{'T' if v is True else 'F' if v is False else v}"
            for k, v in p.items()
        )
        elapsed = r.get("elapsed_secs", 0)
        elapsed_str = f"{elapsed // 60}m{elapsed % 60:02d}s" if elapsed else "—"
        vstyle = _verdict_style(r.get("verdict", ""))
        ts_str = str(r.get("ts", ""))[:16]
        rows_html += (
            f"<tr>"
            f"<td style='color:{col};font-weight:bold'>{rank_disp}</td>"
            f"<td style='color:{col}'>{r['combo_id']}</td>"
            f"<td style='font-size:0.8em;color:#ccc'>{params_str}</td>"
            f"<td style='color:{col};font-weight:bold'>{_fmt(r.get('score'),'.2f')}</td>"
            f"<td>{_fmt(r.get('mean_ref_gap'))}</td>"
            f"<td>{_fmt(r.get('mean_cond_gap'))}</td>"
            f"<td>{_fmt(r.get('final_loss'))}</td>"
            f"<td>{elapsed_str}</td>"
            f"<td style='{vstyle}'>{r.get('verdict','?')}</td>"
            f"<td style='color:#666;font-size:0.78em'>{ts_str}</td>"
            f"</tr>\n"
        )

    # Best-config box + YAML download
    best = ranked[0] if ranked else None
    best_html = ""
    best_config_yaml = ""
    if best:
        p = best["params"]
        lines = [f"<b style='color:#7f7'>#{1}: {best['combo_id']}</b>  "
                 f"score={best['score']:.2f}  "
                 f"ref_gap={best.get('mean_ref_gap') or 0:.4f}  "
                 f"cond_gap={best.get('mean_cond_gap') or 0:.4f}"]
        lines.append("")
        for k, v in p.items():
            lines.append(f"  <b>{k}</b>: {v}")
        lines.append("")
        lines.append(
            "<button onclick=\"downloadBestConfig()\" "
            "style='background:#1a3a1a;border:1px solid #2a5;color:#7f7;"
            "padding:3px 10px;cursor:pointer;font-family:monospace;border-radius:3px'>"
            "⬇ Download best_config.yaml</button>"
        )
        best_html = "<br>".join(lines)
        # Build YAML string for in-browser download
        _yaml_lines = [
            f"# Best config from ablation run: {matrix_name}",
            f"# score={best['score']:.4f}  run={best['combo_id']}",
            "training:",
        ]
        for k, v in p.items():
            if k != "freeze_double_stream_scales":
                _yaml_lines.append(f"  {k}: {v}")
        _yaml_lines.append("adapter:")
        if "freeze_double_stream_scales" in p:
            _yaml_lines.append(f"  freeze_double_stream_scales: {str(p['freeze_double_stream_scales']).lower()}")
        best_config_yaml = "\n".join(_yaml_lines)

    # Summary stats for stats bar
    n_pareto = sum(1 for r in results if r.get("is_pareto") == 1)
    first_score = next((r["score"] for r in results if r.get("score") is not None), None)
    best_score  = ranked[0]["score"] if ranked else None
    if first_score is not None and best_score is not None and len(ranked) > 1:
        improvement_rate = f"+{(best_score - first_score):.2f} over {len(ranked)} runs"
    else:
        improvement_rate = "—"

    # Objective display
    obj_str = ""
    if objective:
        parts = [f"{k}={v}" for k, v in objective.items()]
        obj_str = "  objective: " + "  ".join(parts)

    # JS series for line charts (capped at 20 combos for readability)
    js_series = []
    for rank_i, r in enumerate(all_sorted[:20]):
        col = _html_color(rank_i + 1 if r.get("score") is not None else 99)
        snaps = r.get("snapshots", [])
        js_series.append({
            "label":    r["combo_id"],
            "color":    col,
            "rank":     rank_i + 1,
            "ref_gap":  [[s["step"], s.get("ref_gap")]  for s in snaps],
            "cond_gap": [[s["step"], s.get("cond_gap")] for s in snaps],
            "loss":     [[s["step"], s.get("loss_smooth")] for s in snaps],
            "scale":    [[s["step"], s.get("ip_scale_mean")] for s in snaps],
        })

    # Score bar chart
    bar_data = [
        {"label": r["combo_id"], "score": r.get("score") or 0,
         "color": _html_color(i + 1)}
        for i, r in enumerate(ranked)
    ]

    # Object-identity rank lookup — works for both DB-sourced (has "id") and
    # batch-mode results (no "id").  Must be built before trend_data.
    ranked_rank = {id(r): i + 1 for i, r in enumerate(ranked)}

    # Trend chart: score by experiment order (for long-term runs)
    trend_data = [
        {"i": idx + 1, "score": r.get("score"), "combo_id": r.get("combo_id", ""),
         "strategy": r.get("strategy", ""),
         "color": _html_color(ranked_rank.get(id(r), 99)) if r.get("score") is not None else "#444"}
        for idx, r in enumerate(results)
    ]
    # Rolling best
    best_so_far: list = []
    running_best = None
    for pt in trend_data:
        s = pt.get("score")
        if s is not None and (running_best is None or s > running_best):
            running_best = s
        best_so_far.append(running_best)

    # Pareto scatter: ref_gap vs cond_gap coloured by rank, Pareto front highlighted
    pareto_data = [
        {
            "id":         r.get("id", idx),
            "combo_id":   r.get("combo_id", ""),
            "ref_gap":    r.get("ref_gap"),
            "cond_gap":   r.get("cond_gap"),
            "final_loss": r.get("final_loss"),
            "score":      r.get("score"),
            "is_pareto":  r.get("is_pareto", 0),
            "color":      _html_color(ranked_rank.get(id(r), 99)),
        }
        for idx, r in enumerate(results)
        if r.get("ref_gap") is not None and r.get("cond_gap") is not None
    ]

    return _HTML_TEMPLATE.format(
        ts=ts,
        matrix_name=matrix_name,
        steps=steps,
        n_combos=len(results),
        n_ranked=len(ranked),
        n_crashed=len(crashed),
        total_elapsed=f"{total_elapsed // 3600}h {(total_elapsed % 3600) // 60}m",
        rows_html=rows_html,
        best_html=best_html,
        obj_str=obj_str,
        js_series=json.dumps(js_series),
        bar_data=json.dumps(bar_data),
        trend_data=json.dumps(trend_data),
        best_so_far=json.dumps(best_so_far),
        pareto_data=json.dumps(pareto_data),
        show_pareto="true" if len(pareto_data) >= 3 else "false",
        run_dir_name=run_dir_name,
        show_trend="true" if len(results) >= 5 else "false",
        best_score_str=f"{best_score:.3f}" if best_score is not None else "—",
        n_pareto=n_pareto,
        improvement_rate=improvement_rate,
        best_config_yaml=json.dumps(best_config_yaml),
    )


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Ablation Harness — {matrix_name}</title>
<style>
  body {{font-family:monospace;background:#111;color:#ddd;margin:20px;line-height:1.5}}
  h1 {{color:#7df;margin-bottom:4px}}
  h2 {{color:#adf;font-size:1em;margin-top:24px;margin-bottom:6px;
       border-bottom:1px solid #333;padding-bottom:3px}}
  .meta {{color:#777;font-size:0.82em;margin-bottom:10px}}
  .stats-bar {{display:flex;gap:24px;background:#161e16;border:1px solid #2a3a2a;
               border-radius:6px;padding:10px 16px;margin-bottom:14px;flex-wrap:wrap}}
  .stat {{display:flex;flex-direction:column;min-width:100px}}
  .stat-label {{color:#666;font-size:0.72em;text-transform:uppercase;letter-spacing:0.04em}}
  .stat-value {{color:#aef;font-size:1.05em;font-weight:bold}}
  .best-box {{background:#0d1a0d;border:1px solid #2a5;border-radius:6px;
              padding:12px 16px;margin-bottom:16px;font-size:0.88em;max-width:780px}}
  table {{border-collapse:collapse;font-size:0.82em;margin-top:6px;width:100%}}
  th,td {{border:1px solid #333;padding:4px 8px;text-align:left}}
  th {{background:#1c1c1c;color:#adf}}
  td {{background:#161616}}
  .charts {{display:flex;flex-wrap:wrap;gap:16px;margin-top:12px}}
  canvas {{background:#161616;border:1px solid #2a2a2a;border-radius:4px}}
  .chart-label {{color:#777;font-size:0.8em;margin-bottom:3px}}
  code {{background:#1c1c1c;padding:2px 6px;border-radius:3px}}
</style>
</head>
<body>
<h1>Ablation Harness</h1>
<div class="meta">
  run={matrix_name} &nbsp;|&nbsp; steps/run={steps} &nbsp;|&nbsp;
  {n_combos} experiments ({n_ranked} scored, {n_crashed} crashed) &nbsp;|&nbsp;
  total wall-clock: {total_elapsed} &nbsp;|&nbsp; {ts}<br>
  {obj_str}
</div>

<div class="stats-bar">
  <div class="stat">
    <span class="stat-label">Best Score</span>
    <span class="stat-value" style="color:#7f7">{best_score_str}</span>
  </div>
  <div class="stat">
    <span class="stat-label">Pareto Front</span>
    <span class="stat-value" style="color:#7af">{n_pareto} exps</span>
  </div>
  <div class="stat">
    <span class="stat-label">Improvement</span>
    <span class="stat-value">{improvement_rate}</span>
  </div>
  <div class="stat">
    <span class="stat-label">Scored / Total</span>
    <span class="stat-value">{n_ranked} / {n_combos}</span>
  </div>
</div>

<h2>Recommended Config</h2>
<div class="best-box">{best_html}</div>

<h2>Ranked Results</h2>
<table>
  <tr>
    <th>Rank</th><th>ID</th><th>Parameters</th>
    <th>Score ↓</th><th>ref_gap</th><th>cond_gap</th><th>final_loss</th>
    <th>Elapsed</th><th>Verdict</th><th>Timestamp</th>
  </tr>
  {rows_html}
</table>

<h2>Charts</h2>
<div class="charts">
  <div><div class="chart-label">Ref gap (cross − self) — higher = better style separation</div>
       <canvas id="refChart" width="480" height="220"></canvas></div>
  <div><div class="chart-label">Cond gap (null − cond) — higher = adapter learning</div>
       <canvas id="condChart" width="480" height="220"></canvas></div>
  <div><div class="chart-label">Loss (smooth)</div>
       <canvas id="lossChart" width="480" height="220"></canvas></div>
  <div><div class="chart-label">IP scale mean</div>
       <canvas id="scaleChart" width="480" height="220"></canvas></div>
  <div><div class="chart-label">Score ranking (all experiments)</div>
       <canvas id="barChart" width="480" height="220"></canvas></div>
  <div id="trendBox" style="display:none">
    <div class="chart-label">Score over time — learning progression (rolling best in white)</div>
    <canvas id="trendChart" width="480" height="220"></canvas>
  </div>
  <div id="paretoBox" style="display:none">
    <div class="chart-label">Pareto front — ref_gap vs cond_gap (★ = Pareto-efficient)</div>
    <canvas id="paretoChart" width="480" height="220"></canvas>
  </div>
</div>

<h2>Data</h2>
<p><a href="results.json" style="color:#7af">results.json</a> — full metric data for further analysis</p>
<p>Per-run logs: <code>runs/{run_dir_name}/&lt;combo_id&gt;/training.log</code></p>
<p>Best config: <code>best_config.yaml</code></p>

<script>
const SERIES           = {js_series};
const BAR              = {bar_data};
const TREND            = {trend_data};
const BEST_LINE        = {best_so_far};
const PARETO           = {pareto_data};
const SHOW_TREND       = {show_trend};
const SHOW_PARETO      = {show_pareto};
const BEST_CONFIG_YAML = {best_config_yaml};

function downloadBestConfig() {{
  if (!BEST_CONFIG_YAML) return;
  const blob = new Blob([BEST_CONFIG_YAML], {{type: 'text/yaml'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'best_config.yaml';
  document.body.appendChild(a); a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(a.href);
}}

function drawChart(id, key, zeroLine) {{
  const cv = document.getElementById(id); if (!cv) return;
  const ctx = cv.getContext('2d');
  const W=cv.width, H=cv.height, pad={{t:16,r:16,b:28,l:52}};
  const cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  const allX=[], allY=[];
  for (const s of SERIES)
    for (const [x,y] of s[key]) {{ allX.push(x); if(y!=null) allY.push(y); }}
  if (!allX.length) {{
    ctx.fillStyle='#555'; ctx.font='11px monospace';
    ctx.fillText('no data',W/2-25,H/2); return;
  }}
  const xMin=Math.min(...allX), xMax=Math.max(...allX)||1;
  let yMin=Math.min(...allY), yMax=Math.max(...allY);
  if (zeroLine) {{ yMin=Math.min(yMin,0); yMax=Math.max(yMax,0.01); }}
  yMin*=0.97; yMax=yMax*1.03+1e-9;
  const sx=x=>pad.l+(xMax>xMin?(x-xMin)/(xMax-xMin)*cw:cw/2);
  const sy=y=>pad.t+ch-(yMax>yMin?(y-yMin)/(yMax-yMin)*ch:ch/2);
  ctx.strokeStyle='#2a2a2a'; ctx.lineWidth=1;
  for(let i=0;i<=4;i++) {{
    const y=yMin+(yMax-yMin)*i/4;
    ctx.beginPath(); ctx.moveTo(pad.l,sy(y)); ctx.lineTo(pad.l+cw,sy(y)); ctx.stroke();
    ctx.fillStyle='#555'; ctx.font='9px monospace';
    ctx.fillText(y.toPrecision(3),2,sy(y)+3);
  }}
  if (zeroLine && yMin<0 && yMax>0) {{
    ctx.strokeStyle='#444'; ctx.lineWidth=1.5; ctx.setLineDash([4,4]);
    ctx.beginPath(); ctx.moveTo(pad.l,sy(0)); ctx.lineTo(pad.l+cw,sy(0)); ctx.stroke();
    ctx.setLineDash([]);
  }}
  ctx.fillStyle='#555'; ctx.font='9px monospace';
  for(let i=0;i<=4;i++) {{
    const x=xMin+(xMax-xMin)*i/4;
    ctx.fillText(Math.round(x/1000)+'k',sx(x)-8,pad.t+ch+18);
  }}
  for(const s of SERIES) {{
    const pts=s[key].filter(([,y])=>y!=null);
    if(!pts.length) continue;
    ctx.strokeStyle=s.color; ctx.lineWidth=s.rank<=3?2:1;
    ctx.globalAlpha=s.rank<=5?1.0:0.55;
    ctx.beginPath();
    pts.forEach(([x,y],i)=>{{
      const cx=sx(x),cy=sy(y); i===0?ctx.moveTo(cx,cy):ctx.lineTo(cx,cy);
    }});
    ctx.stroke(); ctx.globalAlpha=1;
  }}
  let lx=pad.l, ly=pad.t+10;
  for(const s of SERIES.slice(0,5)) {{
    ctx.fillStyle=s.color; ctx.fillRect(lx,ly-6,16,2);
    ctx.fillStyle='#ccc'; ctx.font='8px monospace';
    ctx.fillText(s.label,lx+20,ly);
    lx+=ctx.measureText(s.label).width+34;
    if(lx>W-60){{lx=pad.l;ly+=12;}}
  }}
  if(SERIES.length>5) {{
    ctx.fillStyle='#777'; ctx.font='8px monospace';
    ctx.fillText(`+${{SERIES.length-5}} others`,lx,ly);
  }}
}}

function drawBar(id) {{
  const cv = document.getElementById(id); if (!cv) return;
  const ctx = cv.getContext('2d');
  const W=cv.width, H=cv.height, pad={{t:16,r:60,b:16,l:72}};
  const cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  if(!BAR.length) return;
  const scores=BAR.map(b=>b.score);
  const maxS=Math.max(...scores), minS=Math.min(0,...scores);
  const barH=Math.max(2,Math.floor((ch-4*BAR.length)/Math.max(BAR.length,1)));
  BAR.forEach((b,i) => {{
    const y=pad.t+i*(barH+4);
    const w=Math.max(2,(b.score-minS)/(maxS-minS+1e-9)*cw);
    ctx.fillStyle=b.color; ctx.fillRect(pad.l,y,w,Math.max(barH,2));
    ctx.fillStyle='#888'; ctx.font='8px monospace';
    ctx.fillText(b.label,2,y+barH/2+3);
    ctx.fillStyle='#ccc'; ctx.font='8px monospace';
    ctx.fillText(b.score.toFixed(2),pad.l+w+4,y+barH/2+3);
  }});
}}

function drawTrend(id) {{
  const cv = document.getElementById(id); if (!cv) return;
  if(!SHOW_TREND || !TREND.length) return;
  document.getElementById('trendBox').style.display='block';
  const ctx = cv.getContext('2d');
  const W=cv.width, H=cv.height, pad={{t:16,r:16,b:28,l:52}};
  const cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  const pts=TREND.filter(p=>p.score!=null);
  if(!pts.length) return;
  const allY=pts.map(p=>p.score).concat(BEST_LINE.filter(v=>v!=null));
  let yMin=Math.min(...allY), yMax=Math.max(...allY);
  yMin=yMin*1.03-0.01; yMax=yMax*1.03+0.01;
  const n=TREND.length;
  const sx=i=>pad.l+cw*i/(n-1||1);
  const sy=y=>pad.t+ch-(yMax>yMin?(y-yMin)/(yMax-yMin)*ch:ch/2);
  ctx.strokeStyle='#2a2a2a'; ctx.lineWidth=1;
  for(let k=0;k<=4;k++) {{
    const y=yMin+(yMax-yMin)*k/4;
    ctx.beginPath(); ctx.moveTo(pad.l,sy(y)); ctx.lineTo(pad.l+cw,sy(y)); ctx.stroke();
    ctx.fillStyle='#555'; ctx.font='9px monospace';
    ctx.fillText(y.toFixed(1),2,sy(y)+3);
  }}
  if(yMin<0&&yMax>0) {{
    ctx.strokeStyle='#444'; ctx.lineWidth=1.5; ctx.setLineDash([4,4]);
    ctx.beginPath(); ctx.moveTo(pad.l,sy(0)); ctx.lineTo(pad.l+cw,sy(0)); ctx.stroke();
    ctx.setLineDash([]);
  }}
  // dots per experiment
  TREND.forEach((p,i)=>{{
    if(p.score==null) return;
    ctx.fillStyle=p.color;
    ctx.beginPath(); ctx.arc(sx(i),sy(p.score),4,0,2*Math.PI); ctx.fill();
  }});
  // rolling best line (white)
  ctx.strokeStyle='rgba(255,255,255,0.6)'; ctx.lineWidth=1.5;
  ctx.beginPath();
  let first=true;
  BEST_LINE.forEach((v,i)=>{{
    if(v==null) return;
    first?ctx.moveTo(sx(i),sy(v)):ctx.lineTo(sx(i),sy(v)); first=false;
  }});
  ctx.stroke();
  for(let k=0;k<=4;k++) {{
    const i=Math.round(k*(n-1)/4);
    ctx.fillStyle='#555'; ctx.font='9px monospace';
    ctx.fillText(i+1,sx(i)-4,pad.t+ch+18);
  }}
}}

function drawPareto(id) {{
  const cv = document.getElementById(id); if (!cv) return;
  if (!SHOW_PARETO || !PARETO.length) return;
  document.getElementById('paretoBox').style.display = 'block';
  const ctx = cv.getContext('2d');
  const W=cv.width, H=cv.height, pad={{t:16,r:16,b:32,l:52}};
  const cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  const pts = PARETO.filter(p=>p.ref_gap!=null&&p.cond_gap!=null);
  if (!pts.length) return;
  const xs=pts.map(p=>p.ref_gap), ys=pts.map(p=>p.cond_gap);
  let xMin=Math.min(...xs), xMax=Math.max(...xs);
  let yMin=Math.min(...ys), yMax=Math.max(...ys);
  const xp=(xMax-xMin)*0.12||0.05, yp=(yMax-yMin)*0.12||0.05;
  xMin-=xp; xMax+=xp; yMin-=yp; yMax+=yp;
  const sx=x=>pad.l+(x-xMin)/(xMax-xMin)*cw;
  const sy=y=>pad.t+ch-(y-yMin)/(yMax-yMin)*ch;
  ctx.strokeStyle='#2a2a2a'; ctx.lineWidth=1;
  for(let i=0;i<=4;i++) {{
    const y=yMin+(yMax-yMin)*i/4;
    ctx.beginPath(); ctx.moveTo(pad.l,sy(y)); ctx.lineTo(pad.l+cw,sy(y)); ctx.stroke();
    ctx.fillStyle='#555'; ctx.font='9px monospace'; ctx.fillText(y.toFixed(3),2,sy(y)+3);
  }}
  for(let i=0;i<=4;i++) {{
    const x=xMin+(xMax-xMin)*i/4;
    ctx.beginPath(); ctx.moveTo(sx(x),pad.t); ctx.lineTo(sx(x),pad.t+ch); ctx.stroke();
    ctx.fillStyle='#555'; ctx.font='9px monospace'; ctx.fillText(x.toFixed(3),sx(x)-16,pad.t+ch+20);
  }}
  if(xMin<0&&xMax>0){{
    ctx.strokeStyle='#444'; ctx.setLineDash([4,4]);
    ctx.beginPath(); ctx.moveTo(sx(0),pad.t); ctx.lineTo(sx(0),pad.t+ch); ctx.stroke();
    ctx.setLineDash([]);
  }}
  if(yMin<0&&yMax>0){{
    ctx.strokeStyle='#444'; ctx.setLineDash([4,4]);
    ctx.beginPath(); ctx.moveTo(pad.l,sy(0)); ctx.lineTo(pad.l+cw,sy(0)); ctx.stroke();
    ctx.setLineDash([]);
  }}
  ctx.fillStyle='#666'; ctx.font='9px monospace';
  ctx.fillText('cond_gap ↑',2,pad.t+ch/2);
  ctx.fillText('ref_gap →',pad.l+cw/2-25,pad.t+ch+28);
  // Find best-compromise: Pareto point closest to normalised utopia (1,1)
  const paretoEff = pts.filter(p=>p.is_pareto===1);
  let bestComp = null;
  if (paretoEff.length > 1) {{
    const prMax=Math.max(...paretoEff.map(p=>p.ref_gap));
    const prMin=Math.min(...paretoEff.map(p=>p.ref_gap));
    const pcMax=Math.max(...paretoEff.map(p=>p.cond_gap));
    const pcMin=Math.min(...paretoEff.map(p=>p.cond_gap));
    let bestD = Infinity;
    for (const p of paretoEff) {{
      const nr = prMax>prMin ? (p.ref_gap-prMin)/(prMax-prMin) : 0.5;
      const nc = pcMax>pcMin ? (p.cond_gap-pcMin)/(pcMax-pcMin) : 0.5;
      const d  = Math.sqrt((1-nr)**2+(1-nc)**2);
      if (d<bestD) {{ bestD=d; bestComp=p; }}
    }}
  }} else if (paretoEff.length===1) {{ bestComp=paretoEff[0]; }}

  pts.forEach(p=>{{
    const x=sx(p.ref_gap), y=sy(p.cond_gap);
    const ip=p.is_pareto===1;
    const bc=bestComp&&p.id===bestComp.id;
    ctx.fillStyle=bc?'#ff7':ip?p.color:'rgba(90,90,90,0.6)';
    ctx.strokeStyle=bc?'#fff':ip?'rgba(255,255,255,0.6)':'transparent';
    ctx.lineWidth=bc?2:1.5;
    ctx.beginPath(); ctx.arc(x,y,bc?8:ip?6:3,0,2*Math.PI); ctx.fill();
    if(ip||bc){{ ctx.stroke(); }}
    if(bc){{ ctx.fillStyle='#ff7'; ctx.font='bold 8px monospace'; ctx.fillText('⊕ '+p.combo_id+' (best)',x+10,y+3); }}
    else if(ip){{ ctx.fillStyle='#ccc'; ctx.font='8px monospace'; ctx.fillText('★ '+p.combo_id,x+8,y+3); }}
  }});
}}

drawChart('refChart',  'ref_gap',  true);
drawChart('condChart', 'cond_gap', true);
drawChart('lossChart', 'loss',     false);
drawChart('scaleChart','scale',    false);
drawBar('barChart');
drawTrend('trendChart');
drawPareto('paretoChart');
</script>
</body>
</html>
"""


# ── Human-readable progress ───────────────────────────────────────────────────

def _print_combo_header(combo: dict, idx: int, total: int, steps: int) -> None:
    p = combo["params"]
    params_str = "  ".join(f"{k}={v}" for k, v in p.items())
    print()
    print(_c("cyan", f"{'─'*64}"))
    print(_c("bold", f"  [{idx}/{total}] {combo['combo_id']}  ({steps} steps)"))
    print(f"  {params_str}")
    print(_c("cyan", f"{'─'*64}"))


def _print_result_line(r: dict) -> None:
    v = r.get("verdict", "?")
    score = r.get("score")
    ref   = r.get("mean_ref_gap")
    cond  = r.get("mean_cond_gap")
    score_str = f"{score:.2f}" if score is not None else "—"
    ref_str   = f"{ref:+.4f}"  if ref  is not None else "—"
    cond_str  = f"{cond:+.4f}" if cond is not None else "—"
    vcol = {"PASS": "green", "WARN": "yellow", "CRASH": "red",
            "UNSTABLE": "red", "NO_DATA": "dim"}.get(v, "reset")
    print(f"  {_c(vcol, v):<12}  score={score_str}  "
          f"ref_gap={ref_str}  cond_gap={cond_str}  "
          f"elapsed={r.get('elapsed_secs', 0)}s")


def _print_final_ranking(results: list[dict]) -> None:
    ranked = sorted(
        [r for r in results if r.get("score") is not None],
        key=lambda r: r["score"],
        reverse=True,
    )
    print()
    print(_c("cyan", "═" * 64))
    print(_c("bold", f"  Final ranking ({len(ranked)} scored, "
             f"{len(results)-len(ranked)} crashed)"))
    print(_c("cyan", "═" * 64))
    for i, r in enumerate(ranked):
        rank = i + 1
        col = "green" if rank == 1 else "cyan" if rank == 2 else "yellow" if rank == 3 else "reset"
        p = r["params"]
        p_str = "  ".join(
            f"{k.replace('freeze_double_stream_scales','freeze').replace('_prob','').replace('_weight','_w')}="
            f"{'T' if v is True else 'F' if v is False else v}"
            for k, v in p.items()
        )
        ref  = r.get("mean_ref_gap")
        cond = r.get("mean_cond_gap")
        score = r.get("score", 0)
        line = f"  {_c(col, f'#{rank}')}  {r['combo_id']}  score={score:.2f}"
        if ref is not None and cond is not None:
            line += f"  ref_gap={ref:+.4f}  cond_gap={cond:+.4f}"
        print(line)
        print(f"     {p_str}")
    if ranked:
        best = ranked[0]
        print()
        print(_c("green", "  ★ Best config:"))
        for k, v in best["params"].items():
            print(f"    {k}: {v}")
    print(_c("cyan", "═" * 64))


# ── Batch-mode results persistence ────────────────────────────────────────────

def _load_results(output_dir: Path) -> list[dict]:
    p = output_dir / "results.json"
    if not p.exists():
        return []
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []


def _save_results(output_dir: Path, results: list[dict]) -> None:
    slim = [{k: v for k, v in r.items() if k != "snapshots"} for r in results]
    try:
        with open(output_dir / "results.json", "w") as f:
            json.dump(slim, f, indent=2)
    except OSError as e:
        print(f"WARNING: could not save results.json: {e}", file=sys.stderr)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="QUALITY-10+: Automated style feature ablation harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Long-term mode
    ap.add_argument("--config", default=None, metavar="PATH",
                    help="Harness config YAML (enables long-term mode)")
    ap.add_argument("--db", default=None, metavar="PATH",
                    help="SQLite DB path (long-term mode; default: output-dir/ablation_history.db)")
    ap.add_argument("--report-only", action="store_true",
                    help="Regenerate HTML report from existing DB without running new experiments")
    ap.add_argument("--warm-start-from", default=None, metavar="DIR",
                    help="Output dir of a prior campaign; injects its best params as first candidate")
    ap.add_argument("--force-continue", action="store_true",
                    help="Keep running even after campaign plateau is detected")

    # Batch mode
    ap.add_argument("--matrix", default="small", metavar="PRESET",
                    help=f"Built-in matrix: {list(MATRIX_PRESETS)} (default: small)")
    ap.add_argument("--matrix-file", default=None, metavar="PATH",
                    help="Custom matrix YAML (overrides --matrix)")
    ap.add_argument("--steps", type=int, default=8000,
                    help="Training steps per combo (default: 8000)")
    ap.add_argument("--log-every", type=int, default=None,
                    help="Log interval in steps (default: auto)")

    # Shared
    ap.add_argument("--output-dir", default="train/reports/ablation_run",
                    help="Output directory (default: %(default)s)")
    ap.add_argument("--base-config", default=str(_BASE_CONFIG),
                    help="Base training config YAML")
    ap.add_argument("--shards", default=str(_DEFAULT_SHARDS))
    ap.add_argument("--qwen3-cache", default=str(_DEFAULT_QWEN3) if _DEFAULT_QWEN3.exists() else None)
    ap.add_argument("--vae-cache",   default=str(_DEFAULT_VAE)   if _DEFAULT_VAE.exists()   else None)
    ap.add_argument("--siglip-cache",default=str(_DEFAULT_SIGLIP) if _DEFAULT_SIGLIP.exists() else None)
    ap.add_argument("--max-runs", type=int, default=None,
                    help="Run at most N combos then stop")
    ap.add_argument("--resume", action="store_true",
                    help="Skip combos already in results.json (batch mode)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the combo matrix without training")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-step training output")
    ap.add_argument("--keep-checkpoints", action="store_true",
                    help="Keep checkpoint files after each run")
    ap.add_argument("--ai", action="store_true",
                    help="Emit compact JSON summary on completion")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Long-term mode ────────────────────────────────────────────────────────
    if args.config:
        harness_cfg = _load_harness_config(Path(args.config))
        db_path = Path(args.db) if args.db else output_dir / "ablation_history.db"
        db = AblationDB(db_path)
        run_name = harness_cfg.get("name", "default")

        if args.report_only:
            exps = db.get_experiments(run_name)
            if not exps:
                print(f"No experiments found for run '{run_name}' in {db_path}")
                sys.exit(1)
            steps = harness_cfg.get("steps_per_run", args.steps)
            objective = harness_cfg.get("objective", {})
            _generate_lt_report(exps, output_dir, run_name, steps, objective)
            best_list = db.get_best(run_name, 1)
            if best_list:
                _export_best_config(best_list[0], output_dir, run_name)
            _print_final_ranking(exps)
            print(f"\n  Report: {output_dir / 'index.html'}")
            db.close()
            return

        try:
            run_long_term(harness_cfg, db, args)
        finally:
            db.close()
        return

    # ── Batch mode ────────────────────────────────────────────────────────────
    matrix_def  = _load_matrix(args)
    all_combos  = _generate_combos(matrix_def)
    matrix_name = args.matrix or "custom"

    if not all_combos:
        print("ERROR: empty matrix — no variables defined", file=sys.stderr)
        sys.exit(1)

    log_every = args.log_every or max(50, args.steps // 80)

    runs_dir = output_dir / "runs" / matrix_name
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")

    shards_dir = Path(args.shards)
    if not args.dry_run:
        if not shards_dir.exists() or not list(shards_dir.glob("*.tar")):
            print(f"ERROR: no .tar shards in {shards_dir}", file=sys.stderr)
            sys.exit(1)
        if not Path(args.base_config).exists():
            print(f"ERROR: base config not found: {args.base_config}", file=sys.stderr)
            sys.exit(1)
        if not _VENV_PYTHON.exists():
            print(f"ERROR: venv Python not found: {_VENV_PYTHON}", file=sys.stderr)
            sys.exit(1)
        if not args.siglip_cache or not Path(args.siglip_cache).exists():
            print(_c("yellow", "WARNING: no siglip-cache — ref_gap signal unavailable"))

    if args.dry_run:
        print(_c("bold", f"\nAblation matrix — {matrix_name}"))
        print(f"  {len(all_combos)} combos × {args.steps} steps")
        for c in all_combos:
            print(f"  {c['combo_id']}:  " +
                  "  ".join(f"{k}={v}" for k, v in c["params"].items()))
        print()
        return

    done_ids: set[str] = set()
    all_results: list[dict] = []
    if args.resume:
        all_results = _load_results(output_dir)
        done_ids = {r["combo_id"] for r in all_results}
        if done_ids:
            print(f"  Resuming: {len(done_ids)} done, {len(all_combos)-len(done_ids)} remaining")

    pending = [c for c in all_combos if c["combo_id"] not in done_ids]
    if args.max_runs is not None:
        pending = pending[:args.max_runs]

    print(_c("cyan", f"\n{'═'*64}"))
    print(_c("bold", f"  Ablation Harness — matrix={matrix_name}"))
    print(_c("cyan", f"{'═'*64}"))
    print(f"  {len(all_combos)} total combos  {len(pending)} to run  "
          f"{args.steps} steps each  log_every={log_every}")
    print(f"  output:  {output_dir}")
    print(f"  shards:  {args.shards}")
    print(f"  siglip:  {args.siglip_cache or '⚠ not set — ref_gap unavailable'}")
    if not args.qwen3_cache:
        print(_c("yellow", "  WARNING: no qwen3-cache — live Qwen3 encoding (slow)"))
    print(_c("cyan", f"{'═'*64}\n"))

    run_start = time.time()
    for idx, combo in enumerate(pending, start=len(done_ids) + 1):
        _print_combo_header(combo, idx, len(all_combos), args.steps)
        run_dir = runs_dir / combo["combo_id"]
        result = _run_one(combo, run_dir, args, log_every, quiet=args.quiet)
        all_results.append(result)
        _save_results(output_dir, all_results)
        _print_result_line(result)
        if result.get("exit_code") == -2:
            print(_c("yellow", "\n  Run interrupted — generating report with partial results"))
            break

    total_elapsed = int(time.time() - run_start)
    _print_final_ranking(all_results)

    results_with_snaps = []
    for r in all_results:
        run_dir = runs_dir / r["combo_id"]
        mf = run_dir / "metrics.json"
        if mf.exists():
            try:
                with open(mf) as f:
                    full = json.load(f)
                r_copy = dict(r)
                r_copy["snapshots"] = full.get("snapshots", [])
                results_with_snaps.append(r_copy)
                continue
            except (OSError, json.JSONDecodeError):
                pass
        results_with_snaps.append(r)

    html = _render_html(
        results=results_with_snaps,
        matrix_name=matrix_name,
        steps=args.steps,
        ts=ts,
        total_elapsed=total_elapsed,
        run_dir_name=matrix_name,
    )
    report_path = output_dir / "index.html"
    try:
        with open(report_path, "w") as f:
            f.write(html)
        print(f"\n  Report: {report_path}")
    except OSError as e:
        print(f"WARNING: could not write report: {e}", file=sys.stderr)

    ranked = sorted(
        [r for r in all_results if r.get("score") is not None],
        key=lambda r: r["score"],
        reverse=True,
    )
    if ranked:
        _export_best_config(ranked[0], output_dir)

    if args.ai:
        ai_out = {
            "matrix":             matrix_name,
            "n_combos":           len(all_results),
            "n_scored":           len(ranked),
            "n_crashed":          len(all_results) - len(ranked),
            "total_elapsed_secs": total_elapsed,
            "best": ranked[0] if ranked else None,
            "top5": [{"combo_id": r["combo_id"], "params": r["params"],
                      "score": r.get("score"), "mean_ref_gap": r.get("mean_ref_gap"),
                      "mean_cond_gap": r.get("mean_cond_gap")}
                     for r in ranked[:5]],
            "report": str(report_path),
        }
        print(json.dumps(ai_out, indent=2))

    has_any = any(r.get("score") is not None for r in all_results)
    sys.exit(0 if has_any else 1)


if __name__ == "__main__":
    main()
