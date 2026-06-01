"""
train/scripts/campaign_manager.py — Training campaign and shard manifest management.

A campaign is a named, reproducible shard selection strategy.  Each campaign
produces a manifest JSON that lists which shards to include in a training run,
ranked by final_value_score from unify_scores.py (or combined_score from light
scores if unified scores are not yet available).

Built-in strategies:
  all_keep        — all shards where light scoring decision == keep
  top_pct N       — top N% of scored shards by final_value_score
  source_boost    — multiply per-source scores by a factor before ranking
  balanced        — equal (per-source-proportional) representation
  hard_focus      — rank by training_loss_normalized; prefer harder examples

Manifests are written to:
    {cold_root}/campaigns/{name}/manifest.json

They can be consumed by ip_adapter/dataset.py via make_prefetch_loader_from_manifest()
or read by training scripts to construct the shard_paths list.

Usage:
    python train/scripts/campaign_manager.py create large_baseline
    python train/scripts/campaign_manager.py create high_signal_pruned --top-pct 60
    python train/scripts/campaign_manager.py create wikiart_focused \\
        --source-boost wikiart:2.0 --top-pct 80
    python train/scripts/campaign_manager.py create hard_example_heavy --strategy hard_focus
    python train/scripts/campaign_manager.py list
    python train/scripts/campaign_manager.py show large_baseline
    python train/scripts/campaign_manager.py stats
    python train/scripts/campaign_manager.py diff large_baseline high_signal_pruned
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_lib import (
    now_iso, COLD_ROOT,
    load_config,
)

# Campaign manifests live on cold storage next to the shard pool they reference
CAMPAIGNS_DIR = COLD_ROOT / "campaigns"

MANIFEST_VERSION = "v1"

# Default pre-defined campaigns (matches v2_pipeline.yaml campaigns section)
DEFAULT_CAMPAIGNS: dict[str, dict] = {
    "large_baseline": {
        "description": "All keep-rated shards, balanced across sources",
        "strategy": "all_keep",
    },
    "high_signal_pruned": {
        "description": "Top 80% by final_value_score — removes lowest-quality shards",
        "strategy": "top_pct",
        "top_pct": 80,
    },
    "wikiart_focused": {
        "description": "WikiArt score-boosted (2×), top 80% — emphasises artistic style",
        "strategy": "source_boost",
        "source_boosts": {"wikiart": 2.0},
        "top_pct": 80,
    },
    "hard_example_heavy": {
        "description": "Ranked by training difficulty — prefers shards the model finds hard",
        "strategy": "hard_focus",
        "top_pct": 60,
    },
}


# ---------------------------------------------------------------------------
# Score loading helpers
# ---------------------------------------------------------------------------

def _load_shard_scores(shards_dir: Path) -> list[dict]:
    """
    Load scoring data for every shard in shards_dir.

    Prefers .unified_score.json (final_value_score).
    Falls back to .light_scores.json (combined_score) when unified score absent.
    Shards with no score sidecar are included with score=0.0 and decision=unknown.

    Returns list of dicts, one per .tar shard:
      path, shard_id, source, final_value_score, light_score,
      training_loss_normalized, decision, has_unified
    """
    entries: list[dict] = []

    for tar_path in sorted(shards_dir.glob("*.tar")):
        if tar_path.name.endswith(".tar.tmp"):
            continue
        stem = tar_path.stem

        # Try unified score first
        unified_p = tar_path.with_suffix(".unified_score.json")
        light_p   = tar_path.with_suffix(".light_scores.json")

        has_unified = unified_p.exists()
        entry: dict = {
            "path":                    str(tar_path),
            "shard_id":               stem,
            "source":                  "unknown",
            "final_value_score":       0.0,
            "light_score":             0.0,
            "training_loss_normalized": None,
            "decision":               "unknown",
            "has_unified":            has_unified,
        }

        if has_unified:
            try:
                data = json.loads(unified_p.read_text())
                entry["source"]                   = data.get("source", "unknown")
                entry["final_value_score"]        = float(data.get("final_value_score", 0.0))
                entry["light_score"]              = float(data.get("light_score", 0.0))
                entry["training_loss_normalized"] = data.get("training_loss_normalized")
            except (OSError, json.JSONDecodeError):
                pass
        elif light_p.exists():
            try:
                data = json.loads(light_p.read_text())
                scores = data.get("light_scores", {})
                entry["source"]            = data.get("source", "unknown")
                entry["final_value_score"] = float(scores.get("combined_score", 0.0))
                entry["light_score"]       = float(scores.get("combined_score", 0.0))
                # Use light score decision as proxy when no unified score
                entry["decision"] = data.get("decision", "unknown")
            except (OSError, json.JSONDecodeError):
                pass

        # When unified score present, decision is always "include" (filter happens at strategy)
        if has_unified:
            entry["decision"] = "scored"

        entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------

def _apply_strategy(entries: list[dict], strategy: str, params: dict) -> list[dict]:
    """
    Return the subset of entries to include based on the named strategy.
    Each returned entry gets a "decision" field: "include" or "exclude".

    Entries are annotated in-place and the full list is returned (not filtered)
    so the manifest records both decisions.
    """
    for e in entries:
        e["decision"] = "exclude"

    if strategy == "all_keep":
        # Include all shards that passed the light score filter
        for e in entries:
            src_decision = e.get("decision")
            # When unified score is present (decision="scored"), include all
            # When only light score, use its keep/discard decision
            if e["has_unified"] or src_decision not in ("discard",):
                e["decision"] = "include"

    elif strategy == "top_pct":
        pct = float(params.get("top_pct", 80))
        eligible = [e for e in entries if e["final_value_score"] > 0.0]
        eligible.sort(key=lambda x: x["final_value_score"], reverse=True)
        cutoff = max(1, int(len(eligible) * pct / 100))
        for e in eligible[:cutoff]:
            e["decision"] = "include"

    elif strategy == "source_boost":
        # Multiply each source's score by its boost factor, then rank
        boosts: dict[str, float] = params.get("source_boosts", {})
        pct = float(params.get("top_pct", 80))
        for e in entries:
            boost = boosts.get(e["source"], 1.0)
            e["_boosted_score"] = e["final_value_score"] * boost
        eligible = [e for e in entries if e["_boosted_score"] > 0.0]
        eligible.sort(key=lambda x: x["_boosted_score"], reverse=True)
        cutoff = max(1, int(len(eligible) * pct / 100))
        for e in eligible[:cutoff]:
            e["decision"] = "include"
        # Clean up temp field
        for e in entries:
            e.pop("_boosted_score", None)

    elif strategy == "balanced":
        # Proportional per-source caps: each source contributes at most its
        # natural pool fraction of the total included count.
        pct = float(params.get("top_pct", 100))
        from collections import defaultdict
        by_source: dict[str, list] = defaultdict(list)
        for e in entries:
            if e["final_value_score"] > 0.0:
                by_source[e["source"]].append(e)
        total_eligible = sum(len(v) for v in by_source.values())
        for src, src_entries in by_source.items():
            src_entries.sort(key=lambda x: x["final_value_score"], reverse=True)
            # Source cap = source fraction × total_eligible × pct%
            src_frac = len(src_entries) / max(total_eligible, 1)
            n_include = max(1, int(len(src_entries) * pct / 100))
            for e in src_entries[:n_include]:
                e["decision"] = "include"

    elif strategy == "hard_focus":
        # Rank by training_loss_normalized descending (prefers hard/novel shards).
        # Falls back to inverted light_score for shards without training signal.
        pct = float(params.get("top_pct", 60))
        def _hard_score(e: dict) -> float:
            tl = e.get("training_loss_normalized")
            if tl is not None:
                return float(tl)
            # Invert light score: harder shards have lower CLIP alignment scores
            return 1.0 - e["light_score"]
        eligible = [e for e in entries if e["final_value_score"] > 0.0]
        eligible.sort(key=_hard_score, reverse=True)
        cutoff = max(1, int(len(eligible) * pct / 100))
        for e in eligible[:cutoff]:
            e["decision"] = "include"

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. "
                         "Choose from: all_keep, top_pct, source_boost, balanced, hard_focus")

    # Honour force-include / force-exclude overrides (set externally)
    for e in entries:
        if e.get("force_include"):
            e["decision"] = "include"
        elif e.get("force_exclude"):
            e["decision"] = "exclude"

    return entries


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def save_manifest(manifest: dict, path: Path) -> None:
    """Atomically write manifest JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.rename(path)


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def list_campaigns() -> list[dict]:
    """Return summary dicts for all campaign manifests under CAMPAIGNS_DIR."""
    if not CAMPAIGNS_DIR.exists():
        return []
    summaries = []
    for manifest_path in sorted(CAMPAIGNS_DIR.glob("*/manifest.json")):
        try:
            m = json.loads(manifest_path.read_text())
            summaries.append({
                "name":         manifest_path.parent.name,
                "campaign":     m.get("campaign", "?"),
                "strategy":     m.get("strategy", "?"),
                "n_include":    m.get("n_include", 0),
                "n_exclude":    m.get("n_exclude", 0),
                "total_samples":m.get("total_samples", 0),
                "created_at":   m.get("created_at", ""),
                "has_unified":  m.get("has_unified_scores", False),
                "path":         str(manifest_path),
            })
        except (OSError, json.JSONDecodeError):
            pass
    return summaries


# ---------------------------------------------------------------------------
# Subcommand: create
# ---------------------------------------------------------------------------

def cmd_create(args, cfg: dict) -> None:
    name     = args.name
    strategy = args.strategy
    shards_dir = Path(args.shards or
                      cfg.get("storage", {}).get("cold_root", "/Volumes/16TBCold") + "/shards")

    if not shards_dir.exists():
        print(f"ERROR: shards dir not found: {shards_dir}", file=sys.stderr)
        sys.exit(1)

    # Resolve strategy params from CLI args or built-in defaults
    if name in DEFAULT_CAMPAIGNS and strategy == "all_keep":
        # Use predefined strategy unless caller explicitly passed --strategy
        template = DEFAULT_CAMPAIGNS[name]
        if not args.strategy_explicit:
            strategy = template["strategy"]

    strategy_params: dict = {}
    if args.top_pct is not None:
        strategy_params["top_pct"] = args.top_pct
    elif name in DEFAULT_CAMPAIGNS and "top_pct" in DEFAULT_CAMPAIGNS[name]:
        strategy_params["top_pct"] = DEFAULT_CAMPAIGNS[name]["top_pct"]

    if args.source_boosts:
        boosts = {}
        for item in args.source_boosts:
            src, _, mult = item.partition(":")
            if src and mult:
                boosts[src.strip()] = float(mult.strip())
        if boosts:
            strategy_params["source_boosts"] = boosts
    elif name in DEFAULT_CAMPAIGNS and "source_boosts" in DEFAULT_CAMPAIGNS[name]:
        strategy_params["source_boosts"] = DEFAULT_CAMPAIGNS[name]["source_boosts"]

    # Load and score shards
    print(f"Loading shard scores from {shards_dir} ...", flush=True)
    entries = _load_shard_scores(shards_dir)
    if not entries:
        print("ERROR: no shards found.", file=sys.stderr)
        sys.exit(1)

    n_with_unified = sum(1 for e in entries if e["has_unified"])
    n_with_light   = sum(1 for e in entries
                         if not e["has_unified"] and
                         Path(e["path"]).with_suffix(".light_scores.json").exists())
    print(f"  {len(entries)} shards: {n_with_unified} with unified scores, "
          f"{n_with_light} with light scores only, "
          f"{len(entries) - n_with_unified - n_with_light} unscored")

    print(f"Applying strategy '{strategy}' ...", flush=True)
    entries = _apply_strategy(entries, strategy, strategy_params)

    included = [e for e in entries if e["decision"] == "include"]
    excluded = [e for e in entries if e["decision"] != "include"]

    # Per-source stats
    from collections import defaultdict
    src_stats: dict[str, dict] = defaultdict(lambda: {"include": 0, "exclude": 0, "scores": []})
    for e in entries:
        src = e["source"]
        src_stats[src][e["decision"]] += 1
        if e["final_value_score"] > 0.0:
            src_stats[src]["scores"].append(e["final_value_score"])

    # Approximate sample count (5000 images/shard is the standard from build_shards)
    # Actual count would require opening each shard tar — too expensive here
    total_samples_approx = len(included) * 5000

    manifest: dict = {
        "manifest_id":       f"{name}_v{now_iso()[:10].replace('-','')}",
        "campaign":          name,
        "created_at":        now_iso(),
        "strategy":          strategy,
        "strategy_params":   strategy_params,
        "shards_dir":        str(shards_dir),
        "n_include":         len(included),
        "n_exclude":         len(excluded),
        "total_shards":      len(entries),
        "total_samples":     total_samples_approx,
        "has_unified_scores": n_with_unified > 0,
        "avg_final_value_score": round(
            sum(e["final_value_score"] for e in included) / max(len(included), 1), 4
        ),
        "source_stats": {
            src: {
                "include": s["include"],
                "exclude": s["exclude"],
                "avg_score": round(sum(s["scores"]) / max(len(s["scores"]), 1), 4),
            }
            for src, s in sorted(src_stats.items())
        },
        "entries": [
            {
                "path":              e["path"],
                "shard_id":          e["shard_id"],
                "source":            e["source"],
                "final_value_score": e["final_value_score"],
                "light_score":       e["light_score"],
                "decision":          e["decision"],
                "force_include":     e.get("force_include", False),
                "force_exclude":     e.get("force_exclude", False),
            }
            for e in entries
        ],
        "version": MANIFEST_VERSION,
    }

    if args.description:
        manifest["description"] = args.description
    elif name in DEFAULT_CAMPAIGNS:
        manifest["description"] = DEFAULT_CAMPAIGNS[name]["description"]

    out_path = CAMPAIGNS_DIR / name / "manifest.json"
    if args.dry_run:
        print(f"\n[dry-run] Would write manifest to {out_path}")
        print(f"  include={len(included)}/{len(entries)}  "
              f"avg_score={manifest['avg_final_value_score']:.4f}")
        for src, s in manifest["source_stats"].items():
            print(f"  {src}: include={s['include']} exclude={s['exclude']} "
                  f"avg={s['avg_score']:.4f}")
        return

    save_manifest(manifest, out_path)
    print(f"\nCreated campaign '{name}':")
    print(f"  Include: {len(included):,} / {len(entries):,} shards  "
          f"({100 * len(included) / max(len(entries), 1):.0f}%)")
    print(f"  Avg final_value_score: {manifest['avg_final_value_score']:.4f}")
    print(f"  Approx samples: {total_samples_approx:,}")
    for src, s in manifest["source_stats"].items():
        print(f"  {src}: include={s['include']} exclude={s['exclude']} avg={s['avg_score']:.4f}")
    print(f"  Manifest: {out_path}")


# ---------------------------------------------------------------------------
# Subcommand: list
# ---------------------------------------------------------------------------

def cmd_list(args, cfg: dict) -> None:
    campaigns = list_campaigns()
    if not campaigns:
        print(f"No campaigns found under {CAMPAIGNS_DIR}")
        print("Run: python train/scripts/campaign_manager.py create <name>")
        return

    print(f"{'Campaign':<28} {'Strategy':<16} {'Include':>8} {'Samples':>10}  {'Score':>6}  Created")
    print("-" * 84)
    for c in campaigns:
        print(f"{c['name']:<28} {c['strategy']:<16} {c['n_include']:>8,} "
              f"{c['total_samples']:>10,}  "
              f"{'unified' if c['has_unified'] else 'light  ':>6}  "
              f"{c['created_at'][:10]}")


# ---------------------------------------------------------------------------
# Subcommand: show
# ---------------------------------------------------------------------------

def cmd_show(args, cfg: dict) -> None:
    manifest_path = CAMPAIGNS_DIR / args.name / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: campaign '{args.name}' not found at {manifest_path}", file=sys.stderr)
        sys.exit(1)

    m = load_manifest(manifest_path)
    print(f"Campaign: {m['campaign']}")
    if m.get("description"):
        print(f"  {m['description']}")
    print(f"  Strategy:      {m['strategy']}  params={m.get('strategy_params', {})}")
    print(f"  Shards dir:    {m['shards_dir']}")
    print(f"  Include:       {m['n_include']:,} / {m['total_shards']:,}")
    print(f"  Samples (est): {m['total_samples']:,}")
    print(f"  Avg score:     {m.get('avg_final_value_score', 0):.4f}")
    print(f"  Score type:    {'unified' if m.get('has_unified_scores') else 'light only'}")
    print(f"  Created:       {m['created_at']}")
    print()
    print("  Per-source breakdown:")
    for src, s in sorted(m.get("source_stats", {}).items()):
        pct = 100 * s["include"] / max(s["include"] + s["exclude"], 1)
        print(f"    {src:<16} include={s['include']:>5}  "
              f"exclude={s['exclude']:>5}  ({pct:.0f}% kept)  "
              f"avg_score={s['avg_score']:.4f}")

    if args.verbose:
        print("\n  Top 10 included shards by score:")
        included = sorted(
            [e for e in m["entries"] if e["decision"] == "include"],
            key=lambda x: x["final_value_score"], reverse=True,
        )
        for e in included[:10]:
            print(f"    {e['shard_id']}  {e['source']:<12}  "
                  f"final_value={e['final_value_score']:.4f}  light={e['light_score']:.4f}")
        print("\n  Bottom 10 included shards by score:")
        for e in included[-10:]:
            print(f"    {e['shard_id']}  {e['source']:<12}  "
                  f"final_value={e['final_value_score']:.4f}  light={e['light_score']:.4f}")


# ---------------------------------------------------------------------------
# Subcommand: stats
# ---------------------------------------------------------------------------

def cmd_stats(args, cfg: dict) -> None:
    campaigns = list_campaigns()
    if not campaigns:
        print("No campaigns found.")
        return

    print(f"Campaign score distribution summary ({len(campaigns)} campaigns)\n")
    for c in campaigns:
        try:
            m = load_manifest(Path(c["path"]))
        except Exception:
            continue
        scores = [e["final_value_score"] for e in m["entries"] if e["decision"] == "include"]
        if not scores:
            continue
        import statistics
        mn  = min(scores)
        mx  = max(scores)
        med = statistics.median(scores)
        avg = sum(scores) / len(scores)
        p10 = sorted(scores)[int(len(scores) * 0.10)]
        p90 = sorted(scores)[int(len(scores) * 0.90)]
        print(f"  {c['name']}")
        print(f"    n={len(scores):,}  min={mn:.4f}  p10={p10:.4f}  med={med:.4f}  "
              f"avg={avg:.4f}  p90={p90:.4f}  max={mx:.4f}")


# ---------------------------------------------------------------------------
# Subcommand: diff
# ---------------------------------------------------------------------------

def cmd_diff(args, cfg: dict) -> None:
    def _load(name: str) -> dict:
        p = CAMPAIGNS_DIR / name / "manifest.json"
        if not p.exists():
            print(f"ERROR: campaign '{name}' not found at {p}", file=sys.stderr)
            sys.exit(1)
        return load_manifest(p)

    m1 = _load(args.name1)
    m2 = _load(args.name2)

    set1 = {e["shard_id"] for e in m1["entries"] if e["decision"] == "include"}
    set2 = {e["shard_id"] for e in m2["entries"] if e["decision"] == "include"}

    only1 = set1 - set2
    only2 = set2 - set1
    both  = set1 & set2

    print(f"Diff: '{args.name1}' vs '{args.name2}'")
    print(f"  {args.name1}: {len(set1):,} shards")
    print(f"  {args.name2}: {len(set2):,} shards")
    print(f"  In both:      {len(both):,}")
    print(f"  Only in {args.name1}: {len(only1):,}")
    print(f"  Only in {args.name2}: {len(only2):,}")

    if args.verbose and (only1 or only2):
        score_by_id = {e["shard_id"]: e["final_value_score"]
                       for e in m1["entries"] + m2["entries"]}
        if only1:
            top = sorted(only1, key=lambda sid: score_by_id.get(sid, 0.0), reverse=True)[:5]
            print(f"\n  Top 5 unique to {args.name1}:")
            for sid in top:
                print(f"    {sid}  score={score_by_id.get(sid, 0.0):.4f}")
        if only2:
            top = sorted(only2, key=lambda sid: score_by_id.get(sid, 0.0), reverse=True)[:5]
            print(f"\n  Top 5 unique to {args.name2}:")
            for sid in top:
                print(f"    {sid}  score={score_by_id.get(sid, 0.0):.4f}")


# ---------------------------------------------------------------------------
# Subcommand: create-defaults
# ---------------------------------------------------------------------------

def cmd_create_defaults(args, cfg: dict) -> None:
    """Create all four default campaigns in one shot."""
    shards_dir_str = (args.shards or
                      cfg.get("storage", {}).get("cold_root", "/Volumes/16TBCold") + "/shards")

    class _FakeArgs:
        pass

    for name, template in DEFAULT_CAMPAIGNS.items():
        fa = _FakeArgs()
        fa.name = name
        fa.shards = shards_dir_str
        fa.strategy = template["strategy"]
        fa.strategy_explicit = True
        fa.top_pct = template.get("top_pct")
        fa.source_boosts = (
            [f"{src}:{mult}" for src, mult in template.get("source_boosts", {}).items()]
            or None
        )
        fa.description = template.get("description")
        fa.dry_run = getattr(args, "dry_run", False)
        print(f"\n{'='*60}")
        print(f"Creating default campaign: {name}")
        print(f"{'='*60}")
        cmd_create(fa, cfg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Training campaign and shard manifest management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default=None, metavar="PATH",
                        help="Pipeline YAML config (auto-detected if omitted)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # create
    p_create = sub.add_parser("create", help="Create a campaign manifest")
    p_create.add_argument("name", help="Campaign name")
    p_create.add_argument("--shards", default=None, metavar="DIR",
                          help="Shard pool directory (default from config storage.cold_root/shards)")
    p_create.add_argument("--strategy", default="all_keep",
                          choices=["all_keep", "top_pct", "source_boost", "balanced", "hard_focus"],
                          help="Selection strategy (default: all_keep)")
    p_create.add_argument("--top-pct", type=float, default=None, metavar="N",
                          help="Top N%% of shards to include (used by top_pct, source_boost, balanced)")
    p_create.add_argument("--source-boost", dest="source_boosts", action="append", metavar="SOURCE:MULT",
                          help="Score multiplier for a source (e.g. wikiart:2.0); repeatable")
    p_create.add_argument("--description", default=None,
                          help="Human-readable campaign description")
    p_create.add_argument("--dry-run", action="store_true",
                          help="Print manifest stats without writing to disk")

    # create-defaults
    p_defaults = sub.add_parser("create-defaults",
                                 help="Create all four built-in default campaigns")
    p_defaults.add_argument("--shards", default=None, metavar="DIR")
    p_defaults.add_argument("--dry-run", action="store_true")

    # list
    sub.add_parser("list", help="List all campaigns")

    # show
    p_show = sub.add_parser("show", help="Show campaign details")
    p_show.add_argument("name", help="Campaign name")
    p_show.add_argument("-v", "--verbose", action="store_true",
                         help="Show top/bottom shards by score")

    # stats
    sub.add_parser("stats", help="Score distribution across all campaigns")

    # diff
    p_diff = sub.add_parser("diff", help="Compare two campaign manifests")
    p_diff.add_argument("name1")
    p_diff.add_argument("name2")
    p_diff.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    cfg = load_config(args.config)

    # Mark whether --strategy was explicitly passed (vs default)
    if hasattr(args, "strategy"):
        args.strategy_explicit = "--strategy" in sys.argv

    dispatch = {
        "create":          cmd_create,
        "create-defaults": cmd_create_defaults,
        "list":            cmd_list,
        "show":            cmd_show,
        "stats":           cmd_stats,
        "diff":            cmd_diff,
    }
    dispatch[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
