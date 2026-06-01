"""
train/scripts/unify_scores.py — Unified per-shard scoring.

Merges static quality signals (from .light_scores.json sidecars) with
optional dynamic training signals (from shard_scores.db) into a single
final_value_score per shard.  Intended to run after score_shards_light.py
and before campaign_manager.py creates training manifests.

Signals and default weights (configurable via v2_pipeline.yaml):
  light_score      0.45 — combined_score from score_shards_light.py (static, always)
  diversity        0.20 — source rarity OR SigLIP embedding diversity (see below)
  training_loss    0.25 — mean loss during training (dynamic, requires flywheel DB)
  contribution     0.10 — model improvement per shard (dynamic, requires flywheel DB)

Embedding diversity (v3.16.0):
  When unified_scoring.enable_embedding_diversity: true in v2_pipeline.yaml and
  SigLIP features have been precomputed (precompute_all.py --siglip), the diversity
  signal is replaced by a richer two-component score:
    intra-shard diversity  — 1 - mean pairwise cosine similarity within the shard
    inter-shard rarity     — 1 - mean cosine sim to the 10 most similar shard centroids
  Centroids are cached in {cold_root}/diversity_centroids.npz for fast incremental runs.

When dynamic signals are absent (cold start), their weight is redistributed
proportionally to available signals — same approach as aesthetic weight
redistribution in score_shards_light.py.

Output per shard:
  {shard_stem}.unified_score.json  — full scoring result and metadata
  {shard_stem}.unified             — sentinel (skip on re-run)

Usage:
    train/.venv/bin/python train/scripts/unify_scores.py \\
        --shards /Volumes/16TBCold/shards \\
        --config train/configs/v2_pipeline.yaml

    # Force re-score all shards (also rebuilds diversity cache):
    train/.venv/bin/python train/scripts/unify_scores.py \\
        --shards /Volumes/16TBCold/shards --force

    # Dry run (print score preview without writing files):
    train/.venv/bin/python train/scripts/unify_scores.py \\
        --shards /Volumes/16TBCold/shards --dry-run

    # Force-rebuild the diversity centroid cache only, then exit:
    train/.venv/bin/python train/scripts/unify_scores.py \\
        --shards /Volumes/16TBCold/shards --rebuild-diversity-cache
"""

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_lib import (
    now_iso, write_heartbeat, log_event,
    SHARD_SCORES_DB_PATH,
)

VERSION = "v2"

_DIVERSITY_CACHE_VERSION = 1  # bump to invalidate all cached centroids

DEFAULT_WEIGHTS: dict[str, float] = {
    "light_score":   0.45,
    "diversity":     0.20,  # raised in v3.17.0: embedding diversity now production-ready
    "training_loss": 0.25,  # inverted: lower loss → higher score (more to learn)
    "contribution":  0.10,
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config_weights(config_path: str) -> dict[str, float]:
    """Load unified_scoring.weights from pipeline YAML. Falls back to defaults."""
    try:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        w = cfg.get("unified_scoring", {}).get("weights", {})
        if w:
            return {
                "light_score":   float(w.get("light_score",   DEFAULT_WEIGHTS["light_score"])),
                "diversity":     float(w.get("diversity",     DEFAULT_WEIGHTS["diversity"])),
                "training_loss": float(w.get("training_loss", DEFAULT_WEIGHTS["training_loss"])),
                "contribution":  float(w.get("contribution",  DEFAULT_WEIGHTS["contribution"])),
            }
    except Exception as e:
        print(f"  WARNING: could not load config {config_path}: {e}")
    return dict(DEFAULT_WEIGHTS)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_light_scores(shards_dir: Path) -> dict[str, dict]:
    """
    Read all .light_scores.json sidecars in shards_dir.
    Returns {shard_stem: light_data}.
    """
    result: dict[str, dict] = {}
    for p in shards_dir.glob("*.light_scores.json"):
        # Stem of "000042.light_scores.json" is "000042.light_scores"; strip suffix
        stem = p.name[: -len(".light_scores.json")]
        try:
            result[stem] = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    return result


def read_training_signals(shard_stems: list[str]) -> dict[str, dict]:
    """
    Read per-shard training signals from shard_scores.db (flywheel DB).
    Returns {shard_stem: {"avg_loss": float, "contribution": float}}.
    Returns {} when the DB is absent or has no matching schema.

    Columns are detected defensively — only fields present in the schema are read.
    """
    if not SHARD_SCORES_DB_PATH.exists():
        return {}
    try:
        import sqlite3
        conn = sqlite3.connect(str(SHARD_SCORES_DB_PATH))
        cols = {row[1] for row in conn.execute("PRAGMA table_info(shards)").fetchall()}
        wanted: list[str] = []
        if "avg_loss" in cols:
            wanted.append("avg_loss")
        if "contribution_score" in cols:
            wanted.append("contribution_score")
        if not wanted:
            conn.close()
            return {}
        select = ", ".join(["shard_id"] + wanted)
        rows = conn.execute(
            f"SELECT {select} FROM shards WHERE n_scored > 0"
        ).fetchall()
        conn.close()
    except Exception as e:
        print(f"  WARNING: could not read shard_scores.db: {e}")
        return {}

    result: dict[str, dict] = {}
    for row in rows:
        raw_id = str(row[0])
        entry: dict = {}
        if "avg_loss" in wanted:
            val = row[wanted.index("avg_loss") + 1]
            if val is not None:
                entry["avg_loss"] = float(val)
        if "contribution_score" in wanted:
            val = row[wanted.index("contribution_score") + 1]
            if val is not None:
                entry["contribution"] = float(val)
        if not entry:
            continue
        # Store under the raw DB key. Also store under the 6-digit zero-padded form
        # used on the filesystem (e.g. "42" → "000042") so lookups by shard stem
        # succeed regardless of whether the DB stores integers or padded strings.
        result[raw_id] = entry
        try:
            padded = f"{int(raw_id):06d}"
            if padded != raw_id:
                result[padded] = entry
        except ValueError:
            pass
    return result


# ---------------------------------------------------------------------------
# Signal derivation
# ---------------------------------------------------------------------------

def compute_source_fractions(light_data: dict[str, dict]) -> dict[str, float]:
    """Fraction of the pool occupied by each source dataset."""
    counts: dict[str, int] = defaultdict(int)
    for data in light_data.values():
        counts[data.get("source", "unknown")] += 1
    total = max(sum(counts.values()), 1)
    return {src: count / total for src, count in counts.items()}


def normalize_losses(training_signals: dict[str, dict]) -> dict[str, float]:
    """
    Normalise raw avg_loss values to [0, 1] with an inverted scale:
      low raw loss (already learned) → low normalized score
      high raw loss (hard/novel) → high normalized score

    Returns {shard_stem: normalized_score}.
    """
    losses = {
        sid: d["avg_loss"]
        for sid, d in training_signals.items()
        if "avg_loss" in d
    }
    if not losses:
        return {}
    vals = list(losses.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {sid: 0.5 for sid in losses}
    return {sid: float((v - lo) / (hi - lo)) for sid, v in losses.items()}


# ---------------------------------------------------------------------------
# SigLIP embedding diversity (v3.16.0)
# ---------------------------------------------------------------------------

def _dequantize_4bit(q_packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    Inverse of precompute_all._quantize_4bit: nibble-packed int8 + per-token scale.

    Input:  q_packed [..., dim//2] uint8,  scale [..., 1] float16.
    Output: float32 array shape [..., dim].
    """
    lo = (q_packed & 0x0F).astype(np.int8)
    hi = ((q_packed >> 4) & 0x0F).astype(np.int8)
    # sign-extend from 4-bit two's complement (values 8..15 → -8..-1)
    lo = np.where(lo > 7, lo - 16, lo).astype(np.float32)
    hi = np.where(hi > 7, hi - 16, hi).astype(np.float32)
    out = np.empty((*q_packed.shape[:-1], q_packed.shape[-1] * 2), dtype=np.float32)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out * scale.astype(np.float32)


def _load_siglip_pooled(npz_path: Path) -> Optional[np.ndarray]:
    """
    Load one SigLIP .npz, dequantize, mean-pool over patch tokens, L2-normalise.
    Returns float32 [dim] or None on any error.
    """
    try:
        f = np.load(str(npz_path))
        feat = _dequantize_4bit(f["q"], f["scale"])   # [seq_len, dim] or [dim]
        if feat.ndim == 2:
            feat = feat.mean(axis=0)                   # mean-pool patches → [dim]
        norm = float(np.linalg.norm(feat))
        if norm > 1e-8:
            feat = feat / norm
        return feat.astype(np.float32)
    except Exception:
        return None


def _shard_siglip_vecs(
    tar_path: Path, siglip_dir: Path, sample_k: int
) -> list[np.ndarray]:
    """
    Return up to sample_k L2-normalised pooled embeddings for samples in a shard.

    Discovers record IDs by listing tar member stems; looks up {rec_id}.npz
    in siglip_dir; subsamples randomly when more than sample_k are available.
    """
    import tarfile as _tar
    import random as _rnd
    try:
        with _tar.open(str(tar_path), "r:") as tf:
            stems: set[str] = set()
            for m in tf.getmembers():
                if m.isfile():
                    stem, _, _ = m.name.rpartition(".")
                    stems.add(stem)
    except Exception:
        return []

    available = [s for s in stems if (siglip_dir / f"{s}.npz").exists()]
    if not available:
        return []
    if len(available) > sample_k:
        available = _rnd.sample(available, sample_k)

    vecs: list[np.ndarray] = []
    for rid in available:
        v = _load_siglip_pooled(siglip_dir / f"{rid}.npz")
        if v is not None:
            vecs.append(v)
    return vecs


class DiversityCache:
    """
    On-disk cache of per-shard SigLIP centroids and intra-diversity scores.

    Stored at {cold_root}/diversity_centroids.npz.  Invalidated per-shard
    by tar mtime: if the shard's .tar has been replaced since the centroid
    was computed, it is recomputed on the next run.

    Arrays in the .npz:
      shard_ids   object  [N]         — shard stem strings
      centroids   float32 [N, dim]    — L2-normalised mean embedding per shard
      intra       float32 [N]         — intra-shard diversity score ∈ [0, 1]
      tar_mtimes  float64 [N]         — tar mtime at compute time (for invalidation)
      version     int32  scalar       — cache format version
    """

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            f = np.load(str(self.cache_path), allow_pickle=True)
            if int(f["version"]) != _DIVERSITY_CACHE_VERSION:
                return
            for i, sid in enumerate(f["shard_ids"]):
                self._data[str(sid)] = {
                    "centroid":  f["centroids"][i],
                    "intra":     float(f["intra"][i]),
                    "tar_mtime": float(f["tar_mtimes"][i]),
                }
        except Exception:
            pass  # corrupt or version mismatch — start fresh

    def is_stale(self, shard_id: str, tar_mtime: float) -> bool:
        entry = self._data.get(shard_id)
        return entry is None or tar_mtime > entry["tar_mtime"]

    def update(self, shard_id: str, centroid: np.ndarray, intra: float,
               tar_mtime: float) -> None:
        self._data[shard_id] = {
            "centroid":  centroid.astype(np.float32),
            "intra":     float(intra),
            "tar_mtime": float(tar_mtime),
        }

    def save(self) -> None:
        if not self._data:
            return
        ids    = np.array(list(self._data.keys()), dtype=object)
        cens   = np.stack([v["centroid"]  for v in self._data.values()]).astype(np.float32)
        intra  = np.array([v["intra"]     for v in self._data.values()], dtype=np.float32)
        mtimes = np.array([v["tar_mtime"] for v in self._data.values()], dtype=np.float64)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".npz.tmp")
        np.savez(str(tmp),
                 shard_ids=ids, centroids=cens, intra=intra, tar_mtimes=mtimes,
                 version=np.array(_DIVERSITY_CACHE_VERSION, dtype=np.int32))
        # rename the file np.savez actually creates (appends .npz to the stem)
        real_tmp = Path(str(tmp) + ".npz") if not str(tmp).endswith(".npz") else tmp
        if real_tmp.exists() and real_tmp != self.cache_path:
            real_tmp.rename(self.cache_path)
        elif tmp.exists():
            tmp.rename(self.cache_path)

    def centroid(self, shard_id: str) -> Optional[np.ndarray]:
        e = self._data.get(shard_id)
        return e["centroid"] if e else None

    def intra_score(self, shard_id: str) -> Optional[float]:
        e = self._data.get(shard_id)
        return e["intra"] if e else None


def _compute_shard_centroid(
    shard_path: str, siglip_dir: Path, sample_k: int
) -> tuple[str, Optional[np.ndarray], float, float]:
    """
    Compute centroid + intra-diversity for one shard.

    Returns (stem, centroid_or_None, intra, tar_mtime).
    Designed to be called from a thread pool — no shared mutable state.
    """
    stem = Path(shard_path).stem
    try:
        tar_mtime = Path(shard_path).stat().st_mtime
    except OSError:
        tar_mtime = 0.0

    vecs = _shard_siglip_vecs(Path(shard_path), siglip_dir, sample_k)
    if not vecs:
        return (stem, None, 0.5, tar_mtime)

    mat      = np.stack(vecs)
    centroid = mat.mean(axis=0)
    c_norm   = float(np.linalg.norm(centroid))
    if c_norm > 1e-8:
        centroid = centroid / c_norm

    if len(vecs) >= 2:
        sim  = mat @ mat.T
        K    = len(vecs)
        mask = np.triu(np.ones((K, K), dtype=bool), k=1)
        intra = float(np.clip(1.0 - sim[mask].mean(), 0.0, 1.0))
    else:
        intra = 0.5

    return (stem, centroid.astype(np.float32), intra, tar_mtime)


def build_embedding_diversity(
    all_shards: list[str],
    siglip_dir: Path,
    cache_path: Path,
    sample_k: int = 256,
    intra_weight: float = 0.5,
    force: bool = False,
    n_workers: int = 8,
) -> dict[str, float]:
    """
    Compute combined embedding-diversity scores for all shards.

    Two signals are blended:
      intra_weight      — intra-shard diversity: 1 - mean pairwise cosine sim
                          among sample_k images within the shard.
      1 - intra_weight  — inter-shard rarity: 1 - mean cosine sim to the 10
                          most similar shard centroids in the pool.

    Returns {shard_stem: score ∈ [0, 1]}.  Shards without any SigLIP features
    are omitted; the caller substitutes source-rarity diversity for those.

    Centroids and intra scores are cached in cache_path (an .npz file).
    Only shards whose .tar mtime is newer than the cached entry are recomputed,
    making incremental runs fast even on 3 TB pools.

    Pass 1 (centroid computation) is parallelised across n_workers threads,
    significantly reducing wall-clock time for the initial build on 1000+ shards.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache = DiversityCache(cache_path)
    cache_lock = threading.Lock()
    n_computed = n_cached = n_missing = 0

    # Determine which shards need (re)computation
    stale: list[str] = []
    for shard_path in all_shards:
        stem = Path(shard_path).stem
        try:
            tar_mtime = Path(shard_path).stat().st_mtime
        except OSError:
            tar_mtime = 0.0
        if force or cache.is_stale(stem, tar_mtime):
            stale.append(shard_path)
        else:
            n_cached += 1

    # Pass 1 — per-shard centroid + intra-diversity (parallel)
    if stale:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = {
                pool.submit(_compute_shard_centroid, s, siglip_dir, sample_k): s
                for s in stale
            }
            for fut in as_completed(futs):
                try:
                    stem, centroid, intra, tar_mtime = fut.result()
                except Exception as exc:
                    shard_path = futs[fut]
                    print(f"  WARNING: error processing {Path(shard_path).stem}: {exc}")
                    n_missing += 1
                    continue
                if centroid is None:
                    n_missing += 1
                    if force:
                        # Remove any stale entry so deleted SigLIP features don't persist.
                        with cache_lock:
                            cache._data.pop(stem, None)
                    continue
                with cache_lock:
                    cache.update(stem, centroid, intra, tar_mtime)
                n_computed += 1
                if n_computed % 50 == 0:
                    print(f"  [{n_computed} computed, {n_cached} cached] ...", flush=True)

    if n_computed > 0 or (force and n_missing > 0):
        # Save when new centroids were computed OR when force-evictions happened
        # (to persist removal of stale entries for shards that lost SigLIP features).
        cache.save()
    print(f"  Embedding diversity: {n_computed} computed, "
          f"{n_cached} from cache, {n_missing} without SigLIP features")

    # Collect shards that have valid centroids (computed or cached)
    valid = [(Path(s).stem, cache.centroid(Path(s).stem))
             for s in all_shards if cache.centroid(Path(s).stem) is not None]
    if not valid:
        return {}

    stems     = [v[0] for v in valid]
    centroids = np.stack([v[1] for v in valid])        # [N, dim]

    # Pass 2 — inter-shard rarity via pairwise cosine similarity matrix
    sim_matrix = centroids @ centroids.T               # [N, N], already normalised
    np.fill_diagonal(sim_matrix, 0.0)

    top_k = min(10, len(stems) - 1)
    if top_k > 0:
        top_sims = np.sort(sim_matrix, axis=1)[:, -top_k:].mean(axis=1)
        inter    = np.clip(1.0 - top_sims, 0.0, 1.0)
    else:
        inter = np.full(len(stems), 0.5, dtype=np.float32)

    # Normalise inter to [0, 1] across the pool
    lo, hi = inter.min(), inter.max()
    if hi > lo:
        inter = (inter - lo) / (hi - lo)

    result: dict[str, float] = {}
    for i, stem in enumerate(stems):
        intra_v  = cache.intra_score(stem) or 0.5
        combined = intra_weight * intra_v + (1.0 - intra_weight) * float(inter[i])
        result[stem] = float(np.clip(combined, 0.0, 1.0))
    return result


def _load_embedding_diversity_config(
    config_path: str,
) -> tuple[Optional[Path], int, float, bool, Path]:
    """
    Parse embedding diversity settings from the pipeline YAML.

    Returns (siglip_dir, sample_k, intra_weight, enabled, cache_path).
    """
    try:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        us       = cfg.get("unified_scoring", {})
        enabled  = bool(us.get("enable_embedding_diversity", False))
        raw_dir  = us.get("siglip_precompute_dir")
        cold_str = cfg.get("storage", {}).get("cold_root", "/Volumes/16TBCold")
        cold_root = Path(cold_str)
        siglip_dir: Optional[Path] = (
            Path(raw_dir) if raw_dir
            else cold_root / "precomputed" / "siglip"
        )
        sample_k     = int(us.get("embedding_diversity_sample_k", 256))
        intra_weight = float(us.get("embedding_diversity_alpha",  0.5))
        cache_path   = cold_root / "diversity_centroids.npz"
        return siglip_dir, sample_k, intra_weight, enabled, cache_path
    except Exception as e:
        print(f"  WARNING: could not load embedding diversity config: {e}")
        return None, 256, 0.5, False, Path("/tmp/diversity_centroids.npz")


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def compute_unified_score(
    shard_path: str,
    source_fractions: dict[str, float],
    light_data: dict,
    training_signals: Optional[dict],
    loss_normalized: Optional[float],
    weights: dict[str, float],
    embedding_diversity: Optional[float] = None,
) -> dict:
    """
    Compute final_value_score for a single shard.

    Weights for absent signals are redistributed proportionally to whichever
    signals are available — ensures the final score stays on a [0, 1] scale
    regardless of which dynamic signals have been collected.

    embedding_diversity: pre-computed SigLIP-based diversity score (v3.16.0).
      When provided, replaces the source-rarity fallback for the diversity
      signal.  When None, falls back to 1 - source_fraction.
    """
    shard_id = Path(shard_path).stem
    source = light_data.get("source", "unknown")
    light_combined = float(light_data.get("light_scores", {}).get("combined_score", 0.0))

    # Diversity: SigLIP-based when available, else rarer source → higher value.
    if embedding_diversity is not None:
        diversity_score  = float(np.clip(embedding_diversity, 0.0, 1.0))
        diversity_source = "embedding"
    else:
        diversity_score  = float(max(0.0, 1.0 - source_fractions.get(source, 1.0)))
        diversity_source = "source_rarity"

    has_loss  = loss_normalized is not None
    has_contr = training_signals is not None and "contribution" in training_signals
    training_loss_raw = training_signals.get("avg_loss") if training_signals else None
    contribution_val  = float(training_signals["contribution"]) if has_contr else None

    # Build effective weights, redistributing absent signals proportionally
    w_l  = weights["light_score"]
    w_d  = weights["diversity"]
    w_lo = weights["training_loss"] if has_loss  else 0.0
    w_co = weights["contribution"]  if has_contr else 0.0
    total_avail = w_l + w_d + w_lo + w_co
    total_all   = sum(weights.values())

    if total_avail > 0:
        scale    = total_all / total_avail
        w_l_eff  = w_l  * scale
        w_d_eff  = w_d  * scale
        w_lo_eff = w_lo * scale
        w_co_eff = w_co * scale
    else:
        w_l_eff = 1.0
        w_d_eff = w_lo_eff = w_co_eff = 0.0

    final_value_score = float(np.clip(
        w_l_eff  * light_combined +
        w_d_eff  * diversity_score +
        (w_lo_eff * loss_normalized  if has_loss  else 0.0) +
        (w_co_eff * contribution_val if has_contr else 0.0),
        0.0, 1.0,
    ))

    signals_used = ["light_score", "diversity"]
    if has_loss:
        signals_used.append("training_loss")
    if has_contr:
        signals_used.append("contribution")

    return {
        "shard_id":                  shard_id,
        "source":                    source,
        "light_score":               round(light_combined, 4),
        "training_loss_avg":         round(training_loss_raw, 4) if training_loss_raw is not None else None,
        "training_loss_normalized":  round(loss_normalized, 4)   if loss_normalized is not None else None,
        "contribution_score":        round(contribution_val, 4)  if contribution_val is not None else None,
        "diversity_score":           round(diversity_score, 4),
        "final_value_score":         round(final_value_score, 4),
        "components": {
            "light_score":   {"value": round(light_combined, 4), "weight": round(w_l_eff, 4)},
            "diversity":     {"value": round(diversity_score, 4), "weight": round(w_d_eff, 4),
                              "source": diversity_source},
            "training_loss": {
                "value":  round(loss_normalized, 4) if loss_normalized is not None else None,
                "weight": round(w_lo_eff, 4),
            },
            "contribution":  {
                "value":  round(contribution_val, 4) if contribution_val is not None else None,
                "weight": round(w_co_eff, 4),
            },
        },
        "signals_used": signals_used,
        "version":      VERSION,
        "timestamp":    now_iso(),
    }


# ---------------------------------------------------------------------------
# UnifiedScorer class — extensible scoring engine
# ---------------------------------------------------------------------------

class UnifiedScorer:
    """
    Computes final_value_score by combining quality signals for a shard.

    Signals:
      light_score      — static quality from score_shards_light.py (always available)
      diversity        — SigLIP embedding diversity when precomputed, else source-rarity
      training_loss    — mean training loss on shard samples (dynamic, flywheel DB)
      contribution     — per-shard model improvement (dynamic, flywheel DB)

    Embedding diversity (v3.16.0):
      Pre-computed in from_data() when enable_embedding_diversity: true in
      v2_pipeline.yaml and SigLIP features are present in siglip_precompute_dir.
      Two components are blended:
        intra — 1 - mean pairwise cosine sim within the shard (varied content)
        inter — 1 - mean cosine sim to the 10 nearest shard centroids (rarity)
      Centroids are cached in {cold_root}/diversity_centroids.npz.
    """

    def __init__(
        self,
        weights: dict[str, float],
        source_fractions: dict[str, float],
        loss_normalized_by_stem: dict[str, float],
        embedding_diversity_by_stem: Optional[dict[str, float]] = None,
    ):
        self.weights = weights
        self.source_fractions = source_fractions
        self.loss_normalized_by_stem = loss_normalized_by_stem
        self.embedding_diversity_by_stem: dict[str, float] = (
            embedding_diversity_by_stem or {}
        )

    @classmethod
    def from_data(
        cls,
        config_path: Optional[str],
        light_by_stem: dict[str, dict],
        training_by_stem: dict[str, dict],
        all_shards: Optional[list[str]] = None,
        force_diversity: bool = False,
        n_diversity_workers: int = 8,
    ) -> "UnifiedScorer":
        """
        Construct a scorer from already-loaded light and training data.

        all_shards: list of shard .tar paths — required to compute embedding
          diversity.  When None or when the config disables it, diversity falls
          back to source-rarity.
        force_diversity: pass True to rebuild the centroid cache even for shards
          that have not changed (useful after a full --siglip re-precompute).
        n_diversity_workers: thread-pool size for parallel centroid computation.
        """
        weights              = _load_config_weights(config_path) if config_path else dict(DEFAULT_WEIGHTS)
        source_fractions     = compute_source_fractions(light_by_stem)
        loss_normalized_by_stem = normalize_losses(training_by_stem)

        embedding_diversity_by_stem: dict[str, float] = {}
        if all_shards and config_path:
            siglip_dir, sample_k, intra_weight, enabled, cache_path = \
                _load_embedding_diversity_config(config_path)
            if enabled and siglip_dir and siglip_dir.exists():
                print(f"  Computing SigLIP embedding diversity "
                      f"(siglip_dir={siglip_dir}, workers={n_diversity_workers}) ...",
                      flush=True)
                embedding_diversity_by_stem = build_embedding_diversity(
                    all_shards, siglip_dir, cache_path,
                    sample_k=sample_k, intra_weight=intra_weight,
                    force=force_diversity, n_workers=n_diversity_workers,
                )
                print(f"  Embedding diversity ready for "
                      f"{len(embedding_diversity_by_stem)} shards")
            elif enabled:
                print(f"  WARNING: enable_embedding_diversity=true but "
                      f"siglip_dir not found: {siglip_dir}")

        return cls(weights, source_fractions, loss_normalized_by_stem,
                   embedding_diversity_by_stem)

    def score(self, shard_path: str, light_data: dict,
              training_signals: Optional[dict]) -> dict:
        """Compute and return the full scoring result dict for one shard."""
        stem = Path(shard_path).stem
        return compute_unified_score(
            shard_path=shard_path,
            source_fractions=self.source_fractions,
            light_data=light_data,
            training_signals=training_signals,
            loss_normalized=self.loss_normalized_by_stem.get(stem),
            weights=self.weights,
            embedding_diversity=self.embedding_diversity_by_stem.get(stem),
        )


# ---------------------------------------------------------------------------
# Sidecar / sentinel helpers
# ---------------------------------------------------------------------------

def _sidecar_path(shard_path: str) -> Path:
    return Path(shard_path).with_suffix(".unified_score.json")


def _sentinel_path(shard_path: str) -> Path:
    return Path(shard_path).with_suffix(".unified")


def is_unified(shard_path: str) -> bool:
    return _sentinel_path(shard_path).exists()


def write_result(shard_path: str, result: dict) -> None:
    sidecar = _sidecar_path(shard_path)
    sidecar.write_text(json.dumps(result, indent=2))
    _sentinel_path(shard_path).touch()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified per-shard scoring (light scores + optional training signals)"
    )
    parser.add_argument("--shards", required=True,
                        help="Directory containing shard .tar files")
    parser.add_argument("--config", default=None, metavar="PATH",
                        help="Pipeline YAML config to load unified_scoring.weights from")
    parser.add_argument("--force", action="store_true",
                        help="Re-score shards that already have a .unified sentinel")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print scores without writing sidecars or sentinels")
    parser.add_argument("--shard-pattern", default="*.tar",
                        help="Glob pattern for shards within --shards (default: *.tar)")
    parser.add_argument("--rebuild-diversity-cache", action="store_true",
                        help="Rebuild the SigLIP centroid cache for all shards and exit "
                             "(does not write .unified sidecars)")
    parser.add_argument("--diversity-weight", type=float, default=None, metavar="FLOAT",
                        help="Override the diversity signal weight (e.g. 0.20); "
                             "the remaining weight budget is redistributed proportionally "
                             "to other signals at scoring time")
    parser.add_argument("--diversity-workers", type=int, default=8, metavar="N",
                        help="Thread-pool size for parallel centroid computation "
                             "(default: 8; set to 1 to disable parallelism)")
    args = parser.parse_args()

    shards_dir = Path(args.shards)
    if not shards_dir.exists():
        print(f"ERROR: --shards dir not found: {shards_dir}", file=sys.stderr)
        sys.exit(1)

    all_shards = sorted(glob.glob(str(shards_dir / args.shard_pattern)))
    all_shards = [s for s in all_shards if s.endswith(".tar") and not s.endswith(".tar.tmp")]
    if not all_shards:
        print(f"No shards found in {shards_dir}")
        sys.exit(0)

    # --rebuild-diversity-cache: compute/refresh centroids only, no sidecar writes
    if getattr(args, "rebuild_diversity_cache", False):
        if not args.config:
            print("ERROR: --rebuild-diversity-cache requires --config", file=sys.stderr)
            sys.exit(1)
        siglip_dir, sample_k, intra_weight, enabled, cache_path = \
            _load_embedding_diversity_config(args.config)
        if not enabled:
            print("WARNING: enable_embedding_diversity is false in config — nothing to do")
            sys.exit(0)
        if not siglip_dir or not siglip_dir.exists():
            print(f"ERROR: siglip_dir not found: {siglip_dir}", file=sys.stderr)
            sys.exit(1)
        n_workers = getattr(args, "diversity_workers", 8)
        print(f"Rebuilding diversity centroid cache ({len(all_shards)} shards, "
              f"{n_workers} workers) ...")
        scores = build_embedding_diversity(
            all_shards, siglip_dir, cache_path,
            sample_k=sample_k, intra_weight=intra_weight, force=True,
            n_workers=n_workers,
        )
        print(f"Done. {len(scores)} shards with diversity scores → {cache_path}")
        sys.exit(0)

    # Load all light scores (fast — small JSON, sequential)
    print(f"\nReading light scores ...", flush=True)
    light_by_stem = read_light_scores(shards_dir)
    n_light = len(light_by_stem)
    print(f"  {n_light}/{len(all_shards)} shards have .light_scores.json")

    if n_light == 0:
        print("ERROR: No light scores found. Run score_shards_light.py first.", file=sys.stderr)
        sys.exit(1)

    # Load training signals from flywheel DB (optional)
    print("Reading training signals from shard_scores.db ...", flush=True)
    training_by_stem = read_training_signals([Path(s).stem for s in all_shards])
    n_training = len(training_by_stem)
    if n_training > 0:
        print(f"  {n_training} shards have training signals (loss, contribution)")
    else:
        print("  No training signals — cold start: static signals only")

    # Build the scorer (handles weight loading, source fractions, loss normalization,
    # and optional SigLIP embedding diversity when configured)
    n_workers = getattr(args, "diversity_workers", 8)
    scorer = UnifiedScorer.from_data(
        args.config, light_by_stem, training_by_stem,
        all_shards=all_shards,
        force_diversity=args.force,
        n_diversity_workers=n_workers,
    )

    # Apply --diversity-weight CLI override after scorer construction
    if args.diversity_weight is not None:
        dw = float(np.clip(args.diversity_weight, 0.0, 1.0))
        old = scorer.weights.get("diversity", 0.0)
        if dw != old:
            delta = dw - old
            total_others = sum(v for k, v in scorer.weights.items() if k != "diversity")
            for k in scorer.weights:
                if k != "diversity":
                    scorer.weights[k] = max(0.0, scorer.weights[k]
                                            - delta * scorer.weights[k] / max(total_others, 1e-9))
            scorer.weights["diversity"] = dw
            print(f"  --diversity-weight override: {old:.3f} → {dw:.3f}")
    print(f"Weights: " + "  ".join(f"{k}={v:.2f}" for k, v in scorer.weights.items()))
    print("  Source distribution: " +
          ", ".join(f"{src}={frac:.1%}"
                    for src, frac in sorted(scorer.source_fractions.items())))

    # Determine pending shards
    pending = [
        s for s in all_shards
        if (args.force or not is_unified(s)) and Path(s).stem in light_by_stem
    ]
    already_done = sum(1 for s in all_shards if is_unified(s) and not args.force)
    no_light     = len(all_shards) - n_light

    print(f"\nunify_scores: {len(all_shards)} total, "
          f"{already_done} already scored, {no_light} without light scores, "
          f"{len(pending)} to process")

    if not pending:
        print("All shards already unified. Use --force to re-score.")
        sys.exit(0)

    # Score
    n_written = 0
    for i, shard_path in enumerate(pending):
        stem    = Path(shard_path).stem
        light_d = light_by_stem[stem]
        train_d = training_by_stem.get(stem)

        try:
            result = scorer.score(shard_path, light_d, train_d)
        except Exception as e:
            import traceback
            print(f"ERROR scoring {stem}: {e}")
            traceback.print_exc()
            log_event("unify_scores", "shard_error", shard=stem, error=str(e))
            continue

        if args.dry_run:
            sig_str = ", ".join(result["signals_used"])
            print(f"  [dry-run] {stem}: final_value_score={result['final_value_score']:.4f}"
                  f"  signals=[{sig_str}]  source={result['source']}")
        else:
            write_result(shard_path, result)
            n_written += 1

        if (i + 1) % 100 == 0 or (i + 1) == len(pending):
            pct = round(100 * (i + 1) / len(pending))
            print(f"  [{i+1}/{len(pending)}] {pct}% ...", flush=True)
            write_heartbeat(
                "unify_scores", chunk=None,
                done=i + 1, total=len(pending), pct=pct,
                n_with_training=n_training,
                n_with_embedding_diversity=len(scorer.embedding_diversity_by_stem),
            )

    if args.dry_run:
        print(f"\nDry run complete — would write {len(pending)} unified score files.")
    else:
        print(f"\nDone. Written: {n_written}  Sidecars: {shards_dir}/*.unified_score.json")


if __name__ == "__main__":
    main()
