"""
train/scripts/unify_scores.py — Unified per-shard scoring.

Merges static quality signals (from .light_scores.json sidecars) with
optional dynamic training signals (from shard_scores.db) into a single
final_value_score per shard.  Intended to run after score_shards_light.py
and before campaign_manager.py creates training manifests.

Signals and default weights (configurable via v2_pipeline.yaml):
  light_score      0.50 — combined_score from score_shards_light.py (static, always)
  diversity        0.10 — source rarity in pool (computed here, always available)
  training_loss    0.25 — mean loss during training (dynamic, requires flywheel DB)
  contribution     0.15 — model improvement per shard (dynamic, requires flywheel DB)

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

    # Force re-score all shards:
    train/.venv/bin/python train/scripts/unify_scores.py \\
        --shards /Volumes/16TBCold/shards --force

    # Dry run (print score preview without writing files):
    train/.venv/bin/python train/scripts/unify_scores.py \\
        --shards /Volumes/16TBCold/shards --dry-run
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

DEFAULT_WEIGHTS: dict[str, float] = {
    "light_score":   0.50,
    "diversity":     0.10,
    "training_loss": 0.25,  # inverted: lower loss → higher score (more to learn)
    "contribution":  0.15,
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
# Core scoring
# ---------------------------------------------------------------------------

def compute_unified_score(
    shard_path: str,
    source_fractions: dict[str, float],
    light_data: dict,
    training_signals: Optional[dict],
    loss_normalized: Optional[float],
    weights: dict[str, float],
) -> dict:
    """
    Compute final_value_score for a single shard.

    Weights for absent signals are redistributed proportionally to whichever
    signals are available — ensures the final score stays on a [0, 1] scale
    regardless of which dynamic signals have been collected.
    """
    shard_id = Path(shard_path).stem
    source = light_data.get("source", "unknown")
    light_combined = float(light_data.get("light_scores", {}).get("combined_score", 0.0))

    # Diversity: rarer source → higher diversity value (more unique content per shard)
    diversity_score = float(max(0.0, 1.0 - source_fractions.get(source, 1.0)))

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
            "diversity":     {"value": round(diversity_score, 4), "weight": round(w_d_eff, 4)},
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

    Current signals:
      light_score      — static quality from score_shards_light.py (always available)
      diversity        — source rarity in the pool (always computed)
      training_loss    — mean training loss on shard samples (dynamic, flywheel DB)
      contribution     — per-shard model improvement (dynamic, flywheel DB)

    Future signals can be added by:
      1. Adding a weight key to DEFAULT_WEIGHTS
      2. Implementing the signal computation in a subclass or optional method
      3. Passing the value as an extra kwarg to compute_unified_score()

    Embedding-diversity signal (inter/intra-shard SigLIP diversity):
      Scaffold is present via _compute_embedding_diversity() below.
      Returns None until precompute_all.py has run with --siglip.
    """

    def __init__(
        self,
        weights: dict[str, float],
        source_fractions: dict[str, float],
        loss_normalized_by_stem: dict[str, float],
    ):
        self.weights = weights
        self.source_fractions = source_fractions
        self.loss_normalized_by_stem = loss_normalized_by_stem

    @classmethod
    def from_data(
        cls,
        config_path: Optional[str],
        light_by_stem: dict[str, dict],
        training_by_stem: dict[str, dict],
    ) -> "UnifiedScorer":
        """Construct a scorer from already-loaded light and training data."""
        weights = _load_config_weights(config_path) if config_path else dict(DEFAULT_WEIGHTS)
        source_fractions = compute_source_fractions(light_by_stem)
        loss_normalized_by_stem = normalize_losses(training_by_stem)
        return cls(weights, source_fractions, loss_normalized_by_stem)

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
        )

    def _compute_embedding_diversity(self, shard_path: str) -> Optional[float]:
        """
        Intra-shard embedding diversity from precomputed SigLIP features.

        Returns None when SigLIP features have not yet been precomputed for this shard.
        To enable: run precompute_all.py with --siglip then set
        unified_scoring.enable_embedding_diversity: true in v2_pipeline.yaml.

        When implemented:
          1. Load per-sample SigLIP embeddings from the precompute directory.
          2. Compute the mean cosine similarity between all pairs of samples.
          3. Return 1 - mean_similarity (higher = more diverse).
        This provides an inter-sample diversity signal beyond the pool-level
        source-rarity measure already captured by the diversity signal.
        """
        return None  # Not yet implemented — precomputed features required


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

    # Build the scorer (handles weight loading, source fractions, loss normalization)
    scorer = UnifiedScorer.from_data(args.config, light_by_stem, training_by_stem)
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
    n_training = len(training_by_stem)
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
            )

    if args.dry_run:
        print(f"\nDry run complete — would write {len(pending)} unified score files.")
    else:
        print(f"\nDone. Written: {n_written}  Sidecars: {shards_dir}/*.unified_score.json")


if __name__ == "__main__":
    main()
