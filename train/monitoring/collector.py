"""
train/monitoring/collector.py — record live snapshots into the trend store (v3.21.0).

Thin recorders that the orchestrator / flywheel / precompute call to append the
current value of each tracked signal. Kept deliberately simple: each takes the
already-computed value (pure, testable) so the call sites stay decoupled from
the store internals. collect_system() is the one convenience that reads the
environment (disk free, available memory).
"""

from __future__ import annotations

from typing import Optional

from .trends import TrendStore


def record_proxy_fallback(store: TrendStore, rate: float,
                          campaign: Optional[str] = None) -> None:
    """Record the proxy VAE fallback rate (both per-campaign and global)."""
    store.record("proxy_fallback_rate", rate, campaign=campaign)
    if campaign is not None:
        store.record("proxy_fallback_rate", rate, campaign=None)  # global series too


def record_precompute_speed(store: TrendStore, s_per_shard: float,
                            cache_hit_rate: Optional[float] = None,
                            campaign: Optional[str] = None) -> None:
    store.record("precompute_s_per_shard", s_per_shard, campaign=campaign)
    if cache_hit_rate is not None:
        store.record("precompute_cache_hit_rate", cache_hit_rate, campaign=campaign)


def record_train_loss(store: TrendStore, loss: float,
                      campaign: Optional[str] = None,
                      step: Optional[int] = None) -> None:
    store.record("train_loss", loss, campaign=campaign,
                 meta={"step": step} if step is not None else None)


def record_unified_score(store: TrendStore, mean: float,
                         p10: Optional[float] = None,
                         p90: Optional[float] = None) -> None:
    store.record("unified_score_mean", mean,
                 meta={"p10": p10, "p90": p90} if p10 is not None else None)


def record_champion(store: TrendStore, registry, metric: str = "clip_i") -> None:
    """Record the current champion's headline golden metric for trend tracking."""
    champ = registry.champion(metric)
    if champ is not None:
        val = champ.get(f"golden_{metric}")
        if val is not None:
            store.record(f"champion_{metric}", val,
                         meta={"experiment": champ.get("id"),
                               "campaign": champ.get("campaign")})


def collect_system(store: TrendStore) -> None:
    """Record disk-free and available-memory (best-effort; reads environment)."""
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from pipeline_lib import free_gb, DATA_ROOT
        store.record("disk_free_gb", float(free_gb(DATA_ROOT)))
    except Exception:
        pass
    try:
        import psutil
        store.record("mem_available_gb",
                     round(psutil.virtual_memory().available / 1e9, 2))
    except Exception:
        pass
