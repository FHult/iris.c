"""
config_schema.py — lightweight validation for pipeline / flywheel / training YAML
configs (GROK-T-7).

There is no schema today: configs are loaded with a bare yaml.safe_load, so a
typo'd key fails silently (the consumer's `.get(key, default)` quietly uses the
default) or blows up deep in the loop. This catches the common mistakes cheaply:

  - unknown top-level sections (likely typos) → WARNING
  - structural problems (config not a mapping, a section that must be a mapping
    isn't) → ERROR
  - unknown keys inside `flywheel_health` — the doctor's knobs, where a typo
    silently reverts to a default — → WARNING

It is intentionally conservative: it only flags keys it is certain about, so it
never false-positives on a valid config. Run standalone:

    python train/scripts/config_schema.py train/configs/v2_pipeline.yaml

Exits non-zero if any ERROR is found (usable as a pre-flight gate).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Union of every top-level key that appears across the real configs (pipeline,
# flywheel, and per-stage training configs), plus flywheel_health (the doctor's
# knob section). A top-level key outside this set is almost certainly a typo.
KNOWN_TOP_LEVEL = {
    "ablation", "adapter", "anomaly", "campaigns", "chunks", "data",
    "data_sources", "data_volume", "download", "eval", "experiments",
    "flywheel", "flywheel_health", "golden_set", "jdb", "light_scoring",
    "logging", "losses", "model", "monitoring", "output", "poll_interval",
    "precompute", "proxy_vae", "recipe", "scale", "shard_index", "skip_dedup",
    "storage", "student", "teacher", "training", "training_config",
    "unified_scoring", "validation", "weight_registry",
}

# Sections whose value must be a mapping when present.
MUST_BE_MAPPING = {
    "anomaly", "flywheel", "flywheel_health", "model", "training", "data",
    "output", "storage", "proxy_vae", "monitoring", "experiments", "golden_set",
}

# Fully-enumerable nested key sets we are certain about. A key here that isn't
# listed is a typo that would otherwise silently use a default.
NESTED_KNOWN = {
    "flywheel_health": {
        "stall_precompute_secs", "stall_train_secs", "logs_max_gb",
        "max_campaign_days", "consecutive_fail_warn", "consecutive_fail_crit",
    },
    # Every key the orchestrator's flywheel loop reads (fw_cfg.get/[]). The
    # SREF chain is fail-open by design, so a typo here (style_paring,
    # precompute_subsample_pershard, ...) would otherwise silently run as a
    # legacy campaign with nothing downstream complaining.
    "flywheel": {
        "ablation_config", "ablation_every_n", "ablation_max_runs",
        "ablation_warmstart_arms", "archive_best_to_cold",
        "base_checkpoint", "data_root",
        "from_scratch_each_iter", "hyperparams", "max_iterations",
        "min_attribution_obs", "min_free_gb", "n_shards", "name",
        "pipeline_config",
        "plateau_ablation_runs", "plateau_patience", "plateau_threshold",
        "poll_interval", "precomp_trigger_shards",
        "precompute_subsample_per_shard", "quality_gate",
        "resume_from_champion", "shard_manifest", "shard_selection",
        "shards_dir", "source_holdout", "steps_per_iteration", "storage",
        "style_cold_dir", "style_pairing", "temporal_decay", "training_config",
        "ephemeral_scores",
    },
}


def validate_config(cfg) -> list[dict]:
    """Return a list of {severity, path, message} issues (possibly empty)."""
    issues: list[dict] = []

    if not isinstance(cfg, dict):
        return [{"severity": "ERROR", "path": "<root>",
                 "message": f"config is not a mapping (got {type(cfg).__name__})"}]

    for key, val in cfg.items():
        if key not in KNOWN_TOP_LEVEL:
            issues.append({"severity": "WARNING", "path": key,
                           "message": f"unknown top-level key '{key}' — likely a typo"})
        if key in MUST_BE_MAPPING and val is not None and not isinstance(val, dict):
            issues.append({"severity": "ERROR", "path": key,
                           "message": f"'{key}' must be a mapping (got {type(val).__name__})"})

    # Nested known-key checks for the fully-enumerable sections.
    for section, known in NESTED_KNOWN.items():
        sec = cfg.get(section)
        if isinstance(sec, dict):
            for k in sec:
                if k not in known:
                    issues.append({"severity": "WARNING", "path": f"{section}.{k}",
                                   "message": f"unknown key '{k}' in '{section}' — "
                                              f"typo? (would silently use the default)"})

    # Light structural sanity for the two main kinds.
    fw = cfg.get("flywheel")
    if isinstance(fw, dict):
        for req in ("max_iterations", "n_shards"):
            if req not in fw:
                issues.append({"severity": "WARNING", "path": f"flywheel.{req}",
                               "message": f"flywheel.{req} not set — orchestrator will use its default"})
    tr = cfg.get("training")
    if isinstance(tr, dict) and "scale" in cfg and "steps" not in tr:
        issues.append({"severity": "WARNING", "path": "training.steps",
                       "message": "pipeline config has no training.steps map"})

    return issues


def format_issues(issues: list[dict], source: str = "") -> str:
    if not issues:
        return f"OK — {source or 'config'} passed validation."
    lines = [f"{len(issues)} issue(s) in {source or 'config'}:"]
    for i in sorted(issues, key=lambda x: (x["severity"] != "ERROR", x["path"])):
        lines.append(f"  [{i['severity']}] {i['path']}: {i['message']}")
    return "\n".join(lines)


def validate_file(path: str) -> list[dict]:
    import yaml
    try:
        cfg = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as e:
        return [{"severity": "ERROR", "path": "<file>", "message": f"could not parse {path}: {e}"}]
    return validate_config(cfg)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: config_schema.py <config.yaml> [more.yaml ...]", file=sys.stderr)
        return 2
    had_error = False
    for path in sys.argv[1:]:
        issues = validate_file(path)
        print(format_issues(issues, source=path))
        had_error = had_error or any(i["severity"] == "ERROR" for i in issues)
    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
