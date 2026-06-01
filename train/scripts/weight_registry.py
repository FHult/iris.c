"""
train/scripts/weight_registry.py — Lightweight experiment weight registry.

Records trained adapter weight snapshots with metadata for reproducibility
and cross-experiment comparison.  Each snapshot gets a JSON manifest
capturing the campaign, training config, validation metrics, and weights path.

Registry layout:
  {cold_root}/weights/registry/
    index.json                   — all snapshots (latest entry per ID wins)
    {snapshot_id}.json           — individual snapshot manifests

Snapshot naming convention (recommended):
  exp_{YYYYMMDD}_{campaign}_{version}
  e.g. exp_20260601_large_baseline_v1
       exp_20260601_wikiart_focused_v2

Usage:
    python train/scripts/weight_registry.py register exp_20260601_large_v1 \\
        --campaign large_baseline \\
        --weights-path /Volumes/16TBCold/weights/step_200000.safetensors \\
        --notes "First large run, no hard examples"

    python train/scripts/weight_registry.py list
    python train/scripts/weight_registry.py show exp_20260601_large_v1
    python train/scripts/weight_registry.py compare exp_20260601_large_v1 exp_20260601_wikiart_v1
    python train/scripts/weight_registry.py tag exp_20260601_large_v1 best
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_lib import now_iso, COLD_ROOT, load_config

REGISTRY_DIR = COLD_ROOT / "weights" / "registry"
REGISTRY_VERSION = "v1"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _registry_path(snapshot_id: str) -> Path:
    return REGISTRY_DIR / f"{snapshot_id}.json"


def _index_path() -> Path:
    return REGISTRY_DIR / "index.json"


def _read_index() -> list[dict]:
    p = _index_path()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return []


def _write_index(entries: list[dict]) -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _index_path().with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entries, indent=2))
    tmp.rename(_index_path())


def _save_snapshot(snapshot: dict) -> Path:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    path = _registry_path(snapshot["snapshot_id"])
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(snapshot, indent=2))
    tmp.rename(path)
    return path


def load_snapshot(snapshot_id: str) -> Optional[dict]:
    p = _registry_path(snapshot_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Registry operations
# ---------------------------------------------------------------------------

def register_snapshot(
    snapshot_id: str,
    campaign: str,
    weights_path: str,
    manifest_used: Optional[str] = None,
    shards_included: Optional[int] = None,
    total_samples: Optional[int] = None,
    final_value_score_threshold: Optional[float] = None,
    training_config: Optional[dict] = None,
    validation_metrics: Optional[dict] = None,
    notes: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> Path:
    """
    Register a new weight snapshot.  Returns the path of the written manifest.

    If a snapshot with the same ID already exists it is overwritten — re-running
    a registration after updating validation metrics is intentional.
    """
    snapshot: dict = {
        "snapshot_id":               snapshot_id,
        "timestamp":                 now_iso(),
        "campaign":                  campaign,
        "weights_path":              str(weights_path),
        "version":                   REGISTRY_VERSION,
    }
    if manifest_used is not None:
        snapshot["manifest_used"] = str(manifest_used)
    if shards_included is not None:
        snapshot["shards_included"] = shards_included
    if total_samples is not None:
        snapshot["total_samples"] = total_samples
    if final_value_score_threshold is not None:
        snapshot["final_value_score_threshold"] = final_value_score_threshold
    if training_config:
        snapshot["training_config"] = training_config
    if validation_metrics:
        snapshot["validation_metrics"] = validation_metrics
    if notes:
        snapshot["notes"] = notes
    if tags:
        snapshot["tags"] = tags

    path = _save_snapshot(snapshot)

    # Update the index (upsert by snapshot_id)
    index = _read_index()
    summary = {
        "snapshot_id":  snapshot_id,
        "campaign":     campaign,
        "timestamp":    snapshot["timestamp"],
        "weights_path": str(weights_path),
        "tags":         tags or [],
    }
    if validation_metrics:
        summary["val_loss"] = validation_metrics.get("val_loss")
        summary["cond_gap"] = validation_metrics.get("cond_gap")
    index = [e for e in index if e.get("snapshot_id") != snapshot_id]
    index.append(summary)
    index.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    _write_index(index)

    return path


def tag_snapshot(snapshot_id: str, tag: str) -> None:
    """Add a tag to an existing snapshot."""
    snap = load_snapshot(snapshot_id)
    if snap is None:
        raise FileNotFoundError(f"Snapshot not found: {snapshot_id}")
    tags = snap.get("tags", [])
    if tag not in tags:
        tags.append(tag)
    snap["tags"] = tags
    _save_snapshot(snap)
    # Refresh index entry
    index = _read_index()
    for e in index:
        if e.get("snapshot_id") == snapshot_id:
            e["tags"] = tags
            break
    _write_index(index)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_register(args, cfg: dict) -> None:
    training_config: Optional[dict] = None
    if args.training_config:
        try:
            import yaml
            with open(args.training_config) as f:
                training_config = yaml.safe_load(f)
        except Exception as e:
            print(f"  WARNING: could not read --training-config: {e}")

    validation_metrics: Optional[dict] = None
    if args.val_loss is not None or args.cond_gap is not None:
        validation_metrics = {}
        if args.val_loss is not None:
            validation_metrics["val_loss"] = args.val_loss
        if args.cond_gap is not None:
            validation_metrics["cond_gap"] = args.cond_gap

    path = register_snapshot(
        snapshot_id=args.snapshot_id,
        campaign=args.campaign,
        weights_path=args.weights_path,
        manifest_used=args.manifest,
        shards_included=args.shards_included,
        total_samples=args.total_samples,
        final_value_score_threshold=args.score_threshold,
        training_config=training_config,
        validation_metrics=validation_metrics,
        notes=args.notes,
        tags=args.tags,
    )
    print(f"Registered snapshot '{args.snapshot_id}' → {path}")
    if validation_metrics:
        print(f"  Metrics: " +
              "  ".join(f"{k}={v}" for k, v in validation_metrics.items()))


def cmd_list(args, cfg: dict) -> None:
    index = _read_index()
    if not index:
        print(f"No snapshots in registry ({REGISTRY_DIR})")
        return

    print(f"{'Snapshot ID':<40} {'Campaign':<20} {'Tags':<16} {'Val loss':>9}  {'Cond gap':>9}  Created")
    print("-" * 108)
    for e in index:
        tags_str  = ",".join(e.get("tags", []))[:14]
        val_loss  = f"{e['val_loss']:.4f}" if e.get("val_loss") is not None else "—"
        cond_gap  = f"{e['cond_gap']:.4f}" if e.get("cond_gap") is not None else "—"
        ts = e.get("timestamp", "")[:10]
        print(f"{e['snapshot_id']:<40} {e.get('campaign',''):<20} {tags_str:<16} "
              f"{val_loss:>9}  {cond_gap:>9}  {ts}")


def cmd_show(args, cfg: dict) -> None:
    snap = load_snapshot(args.snapshot_id)
    if snap is None:
        print(f"ERROR: snapshot '{args.snapshot_id}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Snapshot:  {snap['snapshot_id']}")
    print(f"  Campaign:    {snap.get('campaign', '—')}")
    print(f"  Timestamp:   {snap.get('timestamp', '—')}")
    print(f"  Tags:        {', '.join(snap.get('tags', [])) or '(none)'}")
    print(f"  Weights:     {snap.get('weights_path', '—')}")
    if snap.get("manifest_used"):
        print(f"  Manifest:    {snap['manifest_used']}")
    if snap.get("shards_included"):
        print(f"  Shards:      {snap['shards_included']:,}")
    if snap.get("total_samples"):
        print(f"  Samples:     {snap['total_samples']:,}")
    if snap.get("final_value_score_threshold") is not None:
        print(f"  Score thr:   {snap['final_value_score_threshold']:.4f}")
    if snap.get("validation_metrics"):
        print(f"  Validation:")
        for k, v in snap["validation_metrics"].items():
            print(f"    {k}: {v}")
    if snap.get("training_config"):
        import json as _j
        print(f"  Training config (excerpt):")
        for k in ("learning_rate", "batch_size", "steps", "hard_mix_ratio"):
            if k in snap["training_config"]:
                print(f"    {k}: {snap['training_config'][k]}")
    if snap.get("notes"):
        print(f"  Notes: {snap['notes']}")


def cmd_compare(args, cfg: dict) -> None:
    s1 = load_snapshot(args.id1)
    s2 = load_snapshot(args.id2)
    if s1 is None:
        print(f"ERROR: snapshot '{args.id1}' not found.", file=sys.stderr)
        sys.exit(1)
    if s2 is None:
        print(f"ERROR: snapshot '{args.id2}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Comparing snapshots:")
    print(f"  A: {s1['snapshot_id']}")
    print(f"  B: {s2['snapshot_id']}\n")
    print(f"  {'Field':<28} {'A':>12}  {'B':>12}  {'Δ (B-A)':>12}")
    print(f"  {'-'*70}")

    def _row(label: str, a, b, fmt: str = "") -> None:
        if a is None and b is None:
            return
        a_s = f"{a:{fmt}}" if a is not None and fmt else (str(a) if a is not None else "—")
        b_s = f"{b:{fmt}}" if b is not None and fmt else (str(b) if b is not None else "—")
        if a is not None and b is not None and isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta = b - a
            d_s   = f"{delta:+.4f}" if fmt else f"{delta:+}"
        else:
            d_s = ""
        print(f"  {label:<28} {a_s:>12}  {b_s:>12}  {d_s:>12}")

    _row("Campaign", s1.get("campaign"), s2.get("campaign"))
    _row("Shards included", s1.get("shards_included"), s2.get("shards_included"), fmt=",")
    _row("Score threshold",
         s1.get("final_value_score_threshold"), s2.get("final_value_score_threshold"), fmt=".4f")

    m1 = s1.get("validation_metrics", {}) or {}
    m2 = s2.get("validation_metrics", {}) or {}
    for key in sorted(set(m1) | set(m2)):
        _row(f"  val.{key}", m1.get(key), m2.get(key), fmt=".4f")


def cmd_tag(args, cfg: dict) -> None:
    try:
        tag_snapshot(args.snapshot_id, args.tag)
        print(f"Tagged '{args.snapshot_id}' as '{args.tag}'")
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weight snapshot registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default=None, metavar="PATH")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # register
    p_reg = sub.add_parser("register", help="Register a weight snapshot")
    p_reg.add_argument("snapshot_id")
    p_reg.add_argument("--campaign",         required=True)
    p_reg.add_argument("--weights-path",     required=True, metavar="PATH")
    p_reg.add_argument("--manifest",         default=None, metavar="PATH",
                        help="Campaign manifest JSON used for this run")
    p_reg.add_argument("--shards-included",  type=int, default=None)
    p_reg.add_argument("--total-samples",    type=int, default=None)
    p_reg.add_argument("--score-threshold",  type=float, default=None)
    p_reg.add_argument("--training-config",  default=None, metavar="PATH",
                        help="Training YAML config path")
    p_reg.add_argument("--val-loss",  type=float, default=None)
    p_reg.add_argument("--cond-gap",  type=float, default=None,
                        help="loss_null - loss_cond gap (adapter conditioning quality signal)")
    p_reg.add_argument("--notes",  default=None)
    p_reg.add_argument("--tag", dest="tags", action="append", default=None,
                        metavar="TAG", help="Tag (repeatable)")

    # list
    sub.add_parser("list", help="List all snapshots")

    # show
    p_show = sub.add_parser("show", help="Show snapshot details")
    p_show.add_argument("snapshot_id")

    # compare
    p_cmp = sub.add_parser("compare", help="Compare two snapshots")
    p_cmp.add_argument("id1")
    p_cmp.add_argument("id2")

    # tag
    p_tag = sub.add_parser("tag", help="Add a tag to a snapshot")
    p_tag.add_argument("snapshot_id")
    p_tag.add_argument("tag")

    args = parser.parse_args()
    cfg = load_config(args.config)

    dispatch = {
        "register": cmd_register,
        "list":     cmd_list,
        "show":     cmd_show,
        "compare":  cmd_compare,
        "tag":      cmd_tag,
    }
    dispatch[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
