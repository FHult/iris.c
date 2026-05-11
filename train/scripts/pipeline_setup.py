#!/usr/bin/env python3
"""
train/scripts/pipeline_setup.py — Interactive pipeline configuration wizard.

First-run script: validates the environment, creates required directories,
generates a pipeline config, and prints the exact commands to start the
orchestrator safely.

Usage:
    python train/scripts/pipeline_setup.py              # interactive wizard
    python train/scripts/pipeline_setup.py --ai         # JSON output for automation
    python train/scripts/pipeline_setup.py --check      # check only, no writes
    python train/scripts/pipeline_setup.py \\
        --scale small --data-root /Volumes/2TBSSD       # skip interactive prompts
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TRAIN_DIR    = Path(__file__).resolve().parent.parent
SCRIPTS_DIR  = TRAIN_DIR / "scripts"
CONFIGS_DIR  = TRAIN_DIR / "configs"
VENV_PYTHON  = TRAIN_DIR / ".venv" / "bin" / "python"
VENV_ACTIVATE = TRAIN_DIR / ".venv" / "bin" / "activate"

# ---------------------------------------------------------------------------
# Run-size catalogue
# ---------------------------------------------------------------------------

RUN_SIZES = {
    "dev": {
        "title": "Integration / binary testing",
        "description": (
            "Produces a valid checkpoint in the shortest possible time.\n"
            "  No quality features are enabled.  Intended for iris.c binary\n"
            "  integration testing: checkpoint format, LoRA loading, inference\n"
            "  path correctness.  NOT suitable for judging model quality."
        ),
        "chunks": 1,
        "steps_display": "200 steps",
        "jdb_display": "1 tgz",
        "laion_display": "none (skipped)",
        "disk_gb": 15,
        "time_display": "~1–2 h",
        "quality_defaults": {"siglip": False, "mine": False, "dedup": False, "mine_use_ema": False},
        "standard_config": "v2_pipeline_dev.yaml",
    },
    "smoke": {
        "title": "Full-pipeline validation",
        "description": (
            "Exercises every pipeline step end-to-end on minimal data.\n"
            "  All quality features enabled: SigLIP precompute, hard-example\n"
            "  mining, CLIP dedup, EMA checkpoint scoring.\n"
            "  Run this before starting a real training run to confirm the\n"
            "  environment, model weights, and disk layout are all correct."
        ),
        "chunks": 1,
        "steps_display": "100 steps",
        "jdb_display": "1 tgz",
        "laion_display": "none (skipped)",
        "disk_gb": 40,
        "time_display": "~6 h",
        "quality_defaults": {"siglip": True, "mine": True, "dedup": True, "mine_use_ema": True},
        "standard_config": "v2_pipeline_smoke.yaml",
    },
    "small": {
        "title": "Quick quality checkpoint",
        "description": (
            "First meaningful quality run — enough data to produce a style-\n"
            "  transfer adapter worth evaluating, short enough to iterate.\n"
            "  Good for confirming hyperparameters before a longer run."
        ),
        "chunks": 4,
        "steps_display": "50 000 + 3 × 15 000 steps",
        "jdb_display": "5 tgzs per chunk",
        "laion_display": "10 %",
        "disk_gb": 250,
        "time_display": "~3–4 days",
        "quality_defaults": {"siglip": True, "mine": True, "dedup": True, "mine_use_ema": True},
        "standard_config": None,
    },
    "medium": {
        "title": "Production quality",
        "description": (
            "Recommended for deployment-quality adapters.  Covers a wide\n"
            "  style distribution with hard-example curriculum learning.\n"
            "  Suitable for sharing or production use."
        ),
        "chunks": 4,
        "steps_display": "105 000 + 3 × 40 000 steps",
        "jdb_display": "13 tgzs per chunk",
        "laion_display": "25 %",
        "disk_gb": 600,
        "time_display": "~10–14 days",
        "quality_defaults": {"siglip": True, "mine": True, "dedup": True, "mine_use_ema": True},
        "standard_config": None,
    },
    "large": {
        "title": "High-quality adapter",
        "description": (
            "Highest quality achievable on M1 Max in ~2 weeks.\n"
            "  50 % of the LAION dataset and a full JourneyDB half.\n"
            "  Requires ~1.2 TB free space."
        ),
        "chunks": 4,
        "steps_display": "200 000 + 3 × 60 000 steps",
        "jdb_display": "25 tgzs per chunk",
        "laion_display": "50 %",
        "disk_gb": 1200,
        "time_display": "~2 weeks",
        "quality_defaults": {"siglip": True, "mine": True, "dedup": True, "mine_use_ema": True},
        "standard_config": None,
    },
    "all-in": {
        "title": "Maximum quality (full dataset)",
        "description": (
            "Full JourneyDB + all LAION/COYO data.  Only practical as a\n"
            "  long-running unattended job.  Requires ~2.2 TB free space."
        ),
        "chunks": 4,
        "steps_display": "540 000 + 3 × 200 000 steps",
        "jdb_display": "50 tgzs per chunk",
        "laion_display": "100 %",
        "disk_gb": 2200,
        "time_display": "~2–3 months",
        "quality_defaults": {"siglip": True, "mine": True, "dedup": True, "mine_use_ema": True},
        "standard_config": None,
    },
}

QUALITY_FLAGS = {
    "siglip": {
        "label": "SigLIP visual conditioning",
        "description": (
            "Precomputes SigLIP-SO400M image features and uses them for\n"
            "  IP-adapter visual conditioning during training AND mining.\n"
            "  Improves style fidelity.  Adds ~3 h precompute per chunk\n"
            "  and ~120 GB disk per full run."
        ),
    },
    "mine": {
        "label": "Hard-example mining",
        "description": (
            "After each training chunk, evaluates per-sample loss across the\n"
            "  dataset and extracts the highest-loss records for replay in\n"
            "  later chunks.  Improves training on difficult compositions\n"
            "  and rare styles.  Adds ~1–3 h GPU time after each chunk."
        ),
    },
    "mine_use_ema": {
        "label": "EMA weights for mining",
        "description": (
            "Uses the exponential moving-average checkpoint for mining loss\n"
            "  evaluation rather than the raw checkpoint.  EMA weights are\n"
            "  smoother and select more representative hard examples.\n"
            "  No extra cost — strongly recommended."
        ),
    },
    "dedup": {
        "label": "CLIP deduplication",
        "description": (
            "Embeds each chunk's images with ViT-L-14 CLIP and removes\n"
            "  near-duplicates (cosine similarity > 0.95) before training.\n"
            "  Prevents overfitting on repeated content.  Adds ~20–40 min\n"
            "  per chunk; safe to disable only for dev or smoke runs."
        ),
    },
}

# Directories that must exist under DATA_ROOT before the orchestrator starts.
REQUIRED_DIRS = [
    "logs",
    "pipeline",
    "staging",
    "shards",
    "precomputed",
    "hard_examples",
    "anchor_shards",
    "dedup_ids",
    "checkpoints/stage1",
    "checkpoints/stage1/archive",
    ".heartbeat",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BOLD  = "\033[1m"
_DIM   = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED   = "\033[31m"
_CYAN  = "\033[36m"
_RESET = "\033[0m"

def _c(text: str, code: str) -> str:
    """Apply ANSI colour only when stdout is a tty."""
    if sys.stdout.isatty():
        return f"{code}{text}{_RESET}"
    return text

def bold(t): return _c(t, _BOLD)
def dim(t):  return _c(t, _DIM)
def ok(t):   return _c(t, _GREEN)
def warn(t): return _c(t, _YELLOW)
def err(t):  return _c(t, _RED)
def hi(t):   return _c(t, _CYAN)


def _ask(prompt: str, default: str = "") -> str:
    """Prompt user for input with a default value."""
    disp = f" [{default}]" if default else ""
    try:
        answer = input(f"{prompt}{disp}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return answer if answer else default


def _ask_bool(prompt: str, default: bool = True) -> bool:
    default_str = "Y/n" if default else "y/N"
    answer = _ask(f"{prompt} [{default_str}]", "").lower()
    if not answer:
        return default
    return answer in ("y", "yes", "1", "true")


def _free_gb(path: Path) -> Optional[float]:
    try:
        stat = shutil.disk_usage(path if path.exists() else path.parent)
        return round(stat.free / 1e9, 1)
    except Exception:
        return None


def _tmux_available() -> bool:
    return shutil.which("tmux") is not None


def _venv_ok() -> bool:
    return VENV_PYTHON.exists()


def _model_path(cfg_model: str) -> Optional[Path]:
    """Resolve model path relative to TRAIN_DIR.parent, as orchestrator does."""
    p = Path(cfg_model)
    if not p.is_absolute():
        p = TRAIN_DIR.parent / cfg_model
    return p if p.exists() else None


def _find_checkpoints(data_root: Path) -> list:
    """Return sorted list of step_NNNNNNN.safetensors paths in checkpoints/stage1/."""
    ckpt_dir = data_root / "checkpoints" / "stage1"
    if not ckpt_dir.exists():
        return []
    return sorted(
        p for p in ckpt_dir.glob("step_*.safetensors")
        if p.is_file() and p.stat().st_size > 0
    )


def _archive_checkpoints(data_root: Path,
                          archive_to: Optional[Path] = None) -> tuple:
    """
    Copy all files from checkpoints/stage1/ (not the archive/ subdir) to a
    timestamped archive directory.  Returns (archive_path, bytes_copied).
    """
    if archive_to is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_to = data_root / "checkpoints" / "stage1" / "archive" / f"run_{ts}"
    archive_to.mkdir(parents=True, exist_ok=True)
    ckpt_dir = data_root / "checkpoints" / "stage1"
    total = 0
    if ckpt_dir.exists():
        for p in ckpt_dir.iterdir():
            if p.is_file():
                dst = archive_to / p.name
                shutil.copy2(p, dst)
                total += dst.stat().st_size
    return archive_to, total


def _purge_pipeline_state(data_root: Path, mode: str) -> int:
    """
    Delete pipeline state.  Returns total bytes freed.

    partial — delete shards, precomputed, hard_examples, dedup_ids, staging,
              pipeline sentinels, logs, heartbeats.  Keeps raw/ downloads and
              checkpoints/ so a re-run skips slow downloads and weights survive.
    full    — same as partial, plus checkpoints/stage1/* (non-archive files).
              checkpoints/stage1/archive/ is never touched by any reset mode.
    """
    total = 0

    def _rmtree(p: Path) -> int:
        if not p.exists():
            return 0
        n = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        shutil.rmtree(p)
        return n

    def _rm_glob(parent: Path, pattern: str) -> int:
        if not parent.exists():
            return 0
        n = 0
        for f in parent.glob(pattern):
            if f.is_file():
                n += f.stat().st_size
                f.unlink()
        return n

    # Common to both modes
    for rel in ["shards", "precomputed", "hard_examples", "dedup_ids",
                "staging", "pipeline"]:
        total += _rmtree(data_root / rel)
    total += _rm_glob(data_root / "logs", "*.log")
    total += _rm_glob(data_root / "logs", "*.jsonl")
    total += _rm_glob(data_root / ".heartbeat", "*.json")

    if mode == "full":
        ckpt_dir = data_root / "checkpoints" / "stage1"
        if ckpt_dir.exists():
            for p in ckpt_dir.iterdir():
                if p.is_file():
                    total += p.stat().st_size
                    p.unlink()
                elif p.is_dir() and p.name != "archive":
                    total += _rmtree(p)

    return total


def _interactive_reset_wizard(data_root: Path, existing: dict) -> str:
    """
    Display existing state and ask the user to choose a reset mode.
    Returns 'full', 'partial', or 'resume'.
    """
    print(bold("Existing pipeline state detected"))
    print()
    done_chunks = [k for k, steps in existing["chunks"].items()
                   if "validate" in steps or len(steps) >= 12]
    if done_chunks:
        print(f"  Completed chunks: {', '.join(str(c) for c in sorted(done_chunks))}")
    in_progress = {k: steps for k, steps in existing["chunks"].items()
                   if k not in done_chunks and steps}
    for cnum, steps in in_progress.items():
        last = steps[-1] if steps else "—"
        print(f"  Chunk {cnum}: {len(steps)} steps completed, last = {last}")
    checkpoints = _find_checkpoints(data_root)
    if checkpoints:
        print(f"  Checkpoints: {len(checkpoints)} found, latest = {checkpoints[-1].name}")
    cache_state = existing.get("precompute_cache", {})
    if cache_state:
        for enc, info in cache_state.items():
            if info.get("layout") == "versioned":
                cur  = info.get("current", "?")
                recs = info.get("record_count", 0)
                done = "complete" if info.get("complete") else "in-progress"
                stale = info.get("versions", 1) - 1
                stale_s = f"  ({stale} stale version(s))" if stale else ""
                print(f"  Cache {enc}: {cur}  {recs:,} records  {done}{stale_s}")
            elif info.get("layout") == "legacy_flat":
                recs = info.get("record_count", 0)
                print(f"  Cache {enc}: legacy flat layout  {recs:,} records"
                      f"  (run cache_inspect.py --migrate-legacy to version)")
    print()
    print(bold("How do you want to proceed?"))
    print()
    print(f"  {hi('[1]')}  {bold('Resume')}        — continue from where it left off (default)")
    print(f"  {hi('[2]')}  {bold('Partial reset')} — delete processed data, keep raw downloads")
    print(f"         {dim('   Removes: shards, precomputed, sentinels, logs, heartbeats, checkpoints')}")
    print(f"         {dim('   Keeps:   raw/ (JDB tgzs), anchor_shards/, archive/ weights')}")
    print(f"  {hi('[3]')}  {bold('Full reset')}    — delete everything except archived weights")
    print(f"         {dim('   Removes all of the above plus checkpoints/stage1/* (not archive/)')}")
    print()
    raw = _ask("  Choose [1-3]", "1")
    try:
        choice = int(raw)
    except ValueError:
        choice = 1
    return {1: "resume", 2: "partial", 3: "full"}.get(choice, "resume")


def _detect_existing_state(data_root: Path) -> dict:
    """Read sentinel dirs to summarise existing pipeline state."""
    sentinel_dir = data_root / "pipeline"
    state: dict = {"found": False, "chunks": {}}
    if not sentinel_dir.exists():
        return state
    state["found"] = True
    for chunk_dir in sorted(sentinel_dir.iterdir()):
        if not chunk_dir.is_dir() or not chunk_dir.name.startswith("chunk"):
            continue
        try:
            cnum = int(chunk_dir.name[5:])
        except ValueError:
            continue
        done = sorted(p.stem for p in chunk_dir.glob("*.done"))
        state["chunks"][cnum] = done

    # Versioned cache state
    state["precompute_cache"] = _scan_cache_versions(data_root / "precomputed")
    return state


def _scan_cache_versions(precomp_root: Path) -> dict:
    """Return a summary of versioned precompute cache state per encoder."""
    try:
        import sys as _sys
        _sys.path.insert(0, str(precomp_root.parent.parent / "train" / "scripts"))
        from cache_manager import PrecomputeCache, ENCODERS
    except ImportError:
        return {}

    result = {}
    for enc in ENCODERS:
        versions = PrecomputeCache.list_versions(precomp_root, enc)
        if not versions:
            # Check for flat legacy layout
            flat_count = sum(1 for f in (precomp_root / enc).glob("*.npz")) \
                if (precomp_root / enc).is_dir() else 0
            if flat_count:
                result[enc] = {"layout": "legacy_flat", "record_count": flat_count}
            continue
        cur = next((v for v in versions if v["current"]), None)
        result[enc] = {
            "layout":        "versioned",
            "versions":      len(versions),
            "current":       cur["version"] if cur else None,
            "record_count":  cur.get("record_count", 0) if cur else 0,
            "complete":      cur.get("complete", False) if cur else False,
        }
    return result


def _run_yaml_load(path: Path) -> Optional[dict]:
    """Load YAML via the venv python to avoid importing yaml in system python."""
    if not VENV_PYTHON.exists():
        return None
    try:
        out = subprocess.check_output(
            [str(VENV_PYTHON), "-c",
             f"import yaml, sys; print(__import__('json').dumps(yaml.safe_load(open('{path}'))))"],
            stderr=subprocess.DEVNULL, text=True, timeout=10
        )
        return json.loads(out.strip())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def _make_config(scale: str, quality: dict, data_root: Path, model_path: str) -> tuple[Path, bool]:
    """
    Return (config_path, was_generated).

    For dev/smoke, return the standard config file directly.
    For other scales, generate train/configs/v2_pipeline_active.yaml from
    the base template with the chosen scale and quality flags baked in.
    """
    sz = RUN_SIZES[scale]

    # Dev / smoke: use existing dedicated configs unchanged.
    if sz["standard_config"]:
        return CONFIGS_DIR / sz["standard_config"], False

    # Load the base template.
    template_path = CONFIGS_DIR / "v2_pipeline.yaml"
    base = _run_yaml_load(template_path)
    if base is None:
        # Fallback: return the template and tell the caller to set --scale manually.
        return template_path, False

    # Bake in scale.
    base["scale"] = scale

    # Quality overrides relative to defaults.
    tcfg = base.setdefault("training", {})
    tcfg["siglip"]       = quality.get("siglip", True)
    tcfg["mine"]         = quality.get("mine", True)
    tcfg["mine_use_ema"] = quality.get("mine_use_ema", True)

    # skip_dedup at top level (orchestrator checks this key).
    base["skip_dedup"] = not quality.get("dedup", True)

    out_path = CONFIGS_DIR / "v2_pipeline_active.yaml"
    _dump_yaml(base, out_path)
    return out_path, True


def _dump_yaml(data: dict, path: Path) -> None:
    """Dump dict to YAML via the venv python."""
    payload = json.dumps(data)
    script = (
        "import yaml, json, sys; "
        f"d = json.loads({repr(payload)}); "
        f"open({repr(str(path))}, 'w').write(yaml.dump(d, default_flow_style=False, sort_keys=False))"
    )
    subprocess.run([str(VENV_PYTHON), "-c", script], check=True, timeout=15)


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------

def _setup_dirs(data_root: Path, dry_run: bool = False) -> tuple[list, list]:
    """Create required directories.  Returns (created, existing)."""
    created, existing = [], []
    for rel in REQUIRED_DIRS:
        p = data_root / rel
        if p.exists():
            existing.append(str(p))
        else:
            if not dry_run:
                p.mkdir(parents=True, exist_ok=True)
            created.append(str(p))
    return created, existing


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def _preflight(data_root: Path, disk_needed_gb: int, model_str: str) -> list[dict]:
    """Run all environment checks.  Returns list of {key, ok, detail} dicts."""
    checks = []

    # Disk space
    free = _free_gb(data_root)
    if free is None:
        checks.append({"key": "disk", "ok": False,
                        "detail": f"{data_root} not reachable — mount or create it first"})
    elif free < disk_needed_gb:
        checks.append({"key": "disk", "ok": False,
                        "detail": f"{free} GB free on {data_root}, need ~{disk_needed_gb} GB"})
    else:
        checks.append({"key": "disk", "ok": True,
                        "detail": f"{free} GB free on {data_root} (need ~{disk_needed_gb} GB)"})

    # tmux
    if _tmux_available():
        checks.append({"key": "tmux", "ok": True, "detail": shutil.which("tmux")})
    else:
        checks.append({"key": "tmux", "ok": False,
                        "detail": "tmux not found — install with: brew install tmux"})

    # Python venv
    if _venv_ok():
        checks.append({"key": "venv", "ok": True, "detail": str(VENV_PYTHON)})
    else:
        checks.append({"key": "venv", "ok": False,
                        "detail": f"venv not found at {VENV_PYTHON} — run: cd train && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"})

    # Model weights
    resolved = _model_path(model_str)
    if resolved:
        checks.append({"key": "model", "ok": True, "detail": str(resolved)})
    else:
        p = TRAIN_DIR.parent / model_str
        checks.append({"key": "model", "ok": False,
                        "detail": (
                            f"Model not found at {p}\n"
                            f"  Download or symlink it: ln -s /path/to/model {p}"
                        )})

    # caffeinate (macOS sleep-prevention)
    if shutil.which("caffeinate"):
        checks.append({"key": "caffeinate", "ok": True,
                        "detail": "available (prevents sleep during training)"})
    else:
        checks.append({"key": "caffeinate", "ok": False,
                        "detail": "caffeinate not found — macOS only; training may pause on sleep"})

    return checks


# ---------------------------------------------------------------------------
# Command generation
# ---------------------------------------------------------------------------

def _build_commands(data_root: Path, config_path: Path) -> dict:
    venv_py  = str(VENV_PYTHON)
    ctl      = str(SCRIPTS_DIR / "pipeline_ctl.py")
    data_env = f"PIPELINE_DATA_ROOT='{data_root}'"
    cfg_arg  = f"--config '{config_path}'"

    return {
        "start": (
            f"{data_env} \\\n"
            f"  {venv_py} {ctl} restart-orchestrator {cfg_arg}"
        ),
        "status": (
            f"{data_env} \\\n"
            f"  {venv_py} {ctl} status"
        ),
        "status_brief": (
            f"{data_env} \\\n"
            f"  {venv_py} {ctl} status --brief"
        ),
        "doctor": (
            f"{data_env} \\\n"
            f"  {venv_py} {str(SCRIPTS_DIR / 'pipeline_doctor.py')} --ai"
        ),
        "pause": (
            f"{data_env} \\\n"
            f"  {venv_py} {ctl} pause"
        ),
        "resume": (
            f"{data_env} \\\n"
            f"  {venv_py} {ctl} resume"
        ),
        "abort": (
            f"{data_env} \\\n"
            f"  {venv_py} {ctl} abort"
        ),
    }


# ---------------------------------------------------------------------------
# Interactive wizard
# ---------------------------------------------------------------------------

def _print_header():
    print()
    print(bold("=" * 62))
    print(bold("  Iris Pipeline Setup"))
    print(bold("=" * 62))
    print()


def _print_run_sizes():
    sizes = list(RUN_SIZES.items())
    print(bold("Step 1 — Choose a run scale"))
    print()
    for i, (key, sz) in enumerate(sizes, 1):
        tag = hi(f"[{i}]")
        name = bold(f"{key:8s}")
        title = sz["title"]
        print(f"  {tag}  {name}  {title}")
        # indent description lines
        for line in sz["description"].splitlines():
            print(f"           {dim(line)}")
        print(f"           {dim('Chunks:')}    {sz['chunks']}   "
              f"{dim('Steps:')} {sz['steps_display']}")
        print(f"           {dim('JDB:')}       {sz['jdb_display']}")
        print(f"           {dim('LAION:')}     {sz['laion_display']}   "
              f"{dim('Disk:')} ~{sz['disk_gb']} GB   "
              f"{dim('Time:')} {sz['time_display']}")
        print()


def _print_quality_review(quality: dict, scale: str):
    print(bold("Step 2 — Quality features"))
    print()
    print(f"  Default configuration for {bold(scale)}:")
    print()
    for key, finfo in QUALITY_FLAGS.items():
        if key == "mine_use_ema" and not quality.get("mine", True):
            continue  # no point showing EMA if mining is off
        state_str = ok("✓ ON ") if quality.get(key, False) else dim("✗ off")
        print(f"  {state_str}  {bold(finfo['label'])}")
        for line in finfo["description"].splitlines():
            print(f"             {dim(line)}")
        print()


def _interactive_quality(defaults: dict) -> dict:
    quality = dict(defaults)
    modify = _ask_bool("  Modify quality settings?", default=False)
    if not modify:
        return quality

    print()
    for key in ["siglip", "mine", "dedup"]:
        finfo = QUALITY_FLAGS[key]
        current = quality.get(key, True)
        answer = _ask_bool(f"  Enable {finfo['label']}?", default=current)
        quality[key] = answer

    if quality.get("mine", True):
        current_ema = quality.get("mine_use_ema", True)
        quality["mine_use_ema"] = _ask_bool(
            "  Use EMA checkpoint for mining?", default=current_ema)
    else:
        quality["mine_use_ema"] = False

    return quality


def _print_preflight_results(checks: list):
    all_ok = True
    for c in checks:
        symbol = ok("✓") if c["ok"] else err("✗")
        label  = f"{c['key']:<14}"
        detail = c["detail"]
        if c["ok"]:
            print(f"  {symbol}  {label}  {detail}")
        else:
            all_ok = False
            print(f"  {symbol}  {label}  {warn(detail)}")
    return all_ok


def run_interactive(args) -> int:
    _print_header()

    # ── Step 1: scale ──────────────────────────────────────────────────────
    _print_run_sizes()
    sizes = list(RUN_SIZES.keys())

    # Honour --scale CLI flag to skip the prompt.
    if getattr(args, "scale", None) and args.scale in sizes:
        scale = args.scale
        print(f"  {dim('Scale pre-selected via --scale:')} {bold(scale)}")
        print()
    else:
        default_idx = sizes.index("large") + 1
        while True:
            raw = _ask(f"  Choose scale [1–{len(sizes)}]", str(default_idx))
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(sizes):
                    scale = sizes[idx]
                    break
            except ValueError:
                if raw in sizes:
                    scale = raw
                    break
            print(f"  {warn('Enter a number 1–' + str(len(sizes)) + ' or a scale name')}")

    sz = RUN_SIZES[scale]
    print()
    print(f"  Selected: {bold(scale)} — {sz['title']}")
    print()

    # ── Step 2: quality ────────────────────────────────────────────────────
    # Apply any quality overrides from CLI flags, then let user review/edit.
    defaults = dict(sz["quality_defaults"])
    for key in ("siglip", "mine", "dedup", "mine_use_ema"):
        override = getattr(args, key, None)
        if override is not None:
            defaults[key] = override

    _print_quality_review(defaults, scale)
    quality = _interactive_quality(defaults)
    print()

    # ── Step 3: data root ──────────────────────────────────────────────────
    print(bold("Step 3 — Data root"))
    print()
    print("  All pipeline data (shards, precomputed features, checkpoints)")
    print("  will be stored under this directory.  Use a fast NVMe volume.")
    print()
    if getattr(args, "data_root", None):
        default_root = args.data_root
    elif scale in ("dev", "smoke"):
        default_root = f"/Volumes/2TBSSD/{scale}"
    else:
        default_root = "/Volumes/2TBSSD"
    data_root = Path(_ask("  Data root", default_root))
    print()

    free = _free_gb(data_root)
    if free is not None:
        needed = sz["disk_gb"]
        if free < needed:
            print(f"  {warn(f'Only {free} GB free — this scale needs ~{needed} GB.')}")
            if not _ask_bool("  Continue anyway?", default=False):
                return 1
        else:
            disk_needed = sz["disk_gb"]
            print(f"  {ok(f'{free} GB free')}  (need ~{disk_needed} GB)")
    print()

    # ── Step 4: model ──────────────────────────────────────────────────────
    print(bold("Step 4 — Flux model"))
    print()
    print("  The IP-adapter is trained on top of the Flux Klein 4B base model.")
    print(f"  Relative paths are resolved from {dim(str(TRAIN_DIR.parent))}")
    print()
    model_str = _ask("  Flux model path", "flux-klein-model")
    resolved  = _model_path(model_str)
    if resolved:
        print(f"  {ok('✓')}  Found: {resolved}")
    else:
        expected = TRAIN_DIR.parent / model_str
        print(f"  {warn('✗')}  Not found at: {expected}")
        print(f"     Download or symlink: {dim(f'ln -s /path/to/model {expected}')}")
    print()

    # ── Pre-flight checks ──────────────────────────────────────────────────
    print(bold("Pre-flight checks"))
    print()
    checks = _preflight(data_root, sz["disk_gb"], model_str)
    checks_ok = _print_preflight_results(checks)
    print()

    if not checks_ok:
        blockers = [c for c in checks if not c["ok"] and c["key"] in ("tmux", "venv", "disk")]
        if blockers:
            print(err("  Blocking issues found.  Resolve them before starting the pipeline."))
            return 1
        print(warn("  Non-blocking warnings above.  The pipeline may still run."))
        if not _ask_bool("  Proceed anyway?", default=False):
            return 1

    if getattr(args, "check", False):
        print("  (--check mode: no changes made)")
        return 0

    # ── Existing state ─────────────────────────────────────────────────────
    existing = _detect_existing_state(data_root)

    # --reset flag skips the interactive menu and uses the chosen mode directly.
    reset_mode = getattr(args, "reset", None)

    if existing["found"] or reset_mode:
        if reset_mode is None:
            reset_mode = _interactive_reset_wizard(data_root, existing)
        else:
            # Brief status display even when --reset was passed non-interactively.
            checkpoints = _find_checkpoints(data_root)
            print(bold("Existing state detected"))
            print()
            for cnum, steps in sorted(existing.get("chunks", {}).items()):
                last = steps[-1] if steps else "—"
                print(f"  Chunk {cnum}: {len(steps)} steps completed, last = {last}")
            if checkpoints:
                print(f"  Checkpoints: {len(checkpoints)} found, latest = {checkpoints[-1].name}")
            print(f"  Reset mode: {bold(reset_mode)} (from --reset flag)")
            print()

        if reset_mode in ("full", "partial"):
            checkpoints = _find_checkpoints(data_root)
            if checkpoints and not getattr(args, "check", False):
                print(f"  Found {len(checkpoints)} checkpoint(s), latest: {bold(checkpoints[-1].name)}")
                do_archive = _ask_bool("  Archive checkpoints before purging?", default=True)
                if do_archive:
                    archive_path, arc_bytes = _archive_checkpoints(data_root)
                    print(f"  {ok('+')}  Archived to: {archive_path}  ({arc_bytes // 1024 // 1024} MB)")
                else:
                    print(f"  {warn('Skipping archive — checkpoint files will be deleted.')}")
                print()
            if not getattr(args, "check", False):
                deleted = _purge_pipeline_state(data_root, reset_mode)
                print(f"  {ok('✓')}  Purged {deleted // 1024 // 1024} MB  (mode={reset_mode})")
                print()
            else:
                print(f"  {dim('(--check mode: skipping purge)')}")
                print()
        else:
            # Resume — show summary and continue
            print("  The orchestrator will automatically resume from where it left off.")
            print()

    # ── Setup ──────────────────────────────────────────────────────────────
    print(bold("Summary"))
    print()
    quality_summary = "  ".join(
        f"{ok('✓') if v else dim('✗')} {k}" for k, v in quality.items()
    )
    print(f"  Scale:    {bold(scale)}  ({sz['steps_display']})")
    print(f"  Quality:  {quality_summary}")
    print(f"  Data:     {data_root}")
    print(f"  Disk:     ~{sz['disk_gb']} GB needed")
    print()

    if not _ask_bool("  Apply setup and create directories?", default=True):
        return 0

    # Directories
    created, existing_dirs = _setup_dirs(data_root)
    if created:
        print(f"  Created {len(created)} director{'ies' if len(created) != 1 else 'y'}:")
        for d in created:
            print(f"    {ok('+')} {d}")
    else:
        print(f"  {ok('✓')}  All directories already exist")

    # Config
    config_path, was_generated = _make_config(scale, quality, data_root, model_str)
    if was_generated:
        print(f"  {ok('+')}  Config generated: {config_path}")
    else:
        print(f"  {ok('✓')}  Using config: {config_path}")
    print()

    # ── Commands ───────────────────────────────────────────────────────────
    cmds = _build_commands(data_root, config_path)
    _print_commands(cmds, scale)
    return 0


def _print_commands(cmds: dict, scale: str):
    print(bold("=" * 62))
    print(bold("  Ready — run these commands"))
    print(bold("=" * 62))
    print()

    print(bold("  Start (or restart) the orchestrator:"))
    print()
    print(f"    {hi(cmds['start'])}")
    print()
    print(f"  {dim('This opens (or replaces) the iris-orch tmux window.')}")
    print(f"  {dim('The orchestrator is fully resumable — safe to Ctrl-C and re-run.')}")
    print()

    print(bold("  Monitor:"))
    print()
    print(f"    {cmds['status_brief']}     {dim('# one-line summary')}")
    print(f"    {cmds['status']}           {dim('# full view + doctor')}")
    print(f"    {cmds['doctor']}           {dim('# machine-readable JSON for Claude Code')}")
    print()
    print(f"    tmux attach -t iris        {dim('# watch the live windows')}")
    print()

    print(bold("  Control:"))
    print()
    print(f"    {cmds['pause']}   {dim('# pause after current step completes')}")
    print(f"    {cmds['resume']}  {dim('# clear pause signal')}")
    print(f"    {cmds['abort']}   {dim('# stop orchestrator + prep (training keeps running)')}")
    print()

    if scale in ("dev", "smoke"):
        print(f"  {dim('Tip: for an isolated run use a separate DATA_ROOT (e.g. /Volumes/2TBSSD/smoke)')}")
        print(f"  {dim('     so it does not interfere with a production run on the same volume.')}")
        print()


# ---------------------------------------------------------------------------
# AI / JSON mode
# ---------------------------------------------------------------------------

def run_ai(args) -> int:
    """
    Non-interactive mode — returns a JSON report.

    If --scale is given, runs a dry-run setup and returns what would be done.
    If no --scale given, detects the current environment and returns its state.
    """
    scale      = getattr(args, "scale", None)
    data_root  = Path(getattr(args, "data_root", None) or "/Volumes/2TBSSD")
    model_str  = getattr(args, "model", "flux-klein-model")
    check_only = getattr(args, "check", False)
    reset_mode = getattr(args, "reset", None)

    if scale and scale not in RUN_SIZES:
        print(json.dumps({"error": f"Unknown scale '{scale}'. "
                          f"Valid: {list(RUN_SIZES.keys())}"}))
        return 1

    result: dict = {
        "ready": False,
        "scale": scale,
        "data_root": str(data_root),
    }

    # Choose defaults based on scale or fall back to current prod config.
    if scale:
        sz      = RUN_SIZES[scale]
        quality = dict(sz["quality_defaults"])
        # Apply any explicit quality overrides from args.
        for key in ("siglip", "mine", "dedup", "mine_use_ema"):
            override = getattr(args, key, None)
            if override is not None:
                quality[key] = override
    else:
        scale   = "small"
        sz      = RUN_SIZES[scale]
        quality = dict(sz["quality_defaults"])

    # Pre-flight checks.
    checks    = _preflight(data_root, sz["disk_gb"], model_str)
    check_map = {c["key"]: {"ok": c["ok"], "detail": c["detail"]} for c in checks}
    blockers  = [c["key"] for c in checks
                 if not c["ok"] and c["key"] in ("tmux", "venv", "disk")]

    # Dir audit (dry-run: don't create).
    created, existing_dirs = _setup_dirs(data_root, dry_run=True)

    # Existing state.
    state     = _detect_existing_state(data_root)
    done_chunks = [k for k, steps in state.get("chunks", {}).items()
                   if "validate" in steps or len(steps) >= 12]

    # Execute reset immediately in --ai mode when --reset is specified.
    if reset_mode and reset_mode in ("full", "partial") and not check_only:
        checkpoints = _find_checkpoints(data_root)
        archive_path = None
        arc_bytes = 0
        if checkpoints:
            archive_path, arc_bytes = _archive_checkpoints(data_root)
        deleted_bytes = _purge_pipeline_state(data_root, reset_mode)
        _setup_dirs(data_root)  # recreate required dirs that were just deleted
        print(json.dumps({
            "action": "purge",
            "mode": reset_mode,
            "archived_to": str(archive_path) if archive_path else None,
            "archived_bytes": arc_bytes,
            "deleted_bytes": deleted_bytes,
        }))
        return 0

    # Default suggested action when existing state found and no --reset given.
    if state["found"] and reset_mode is None:
        suggested_reset = "partial"
    else:
        suggested_reset = None

    # Config path (no generation in --ai mode — just report the path).
    if RUN_SIZES[scale]["standard_config"]:
        config_path = CONFIGS_DIR / RUN_SIZES[scale]["standard_config"]
    else:
        config_path = CONFIGS_DIR / "v2_pipeline_active.yaml"
    config_exists = config_path.exists()

    # Build commands.
    cmds = _build_commands(data_root, config_path)

    issues:   list[str] = []
    warnings: list[str] = []

    for c in checks:
        if not c["ok"]:
            if c["key"] in ("tmux", "venv", "disk"):
                issues.append(f"{c['key']}: {c['detail']}")
            else:
                warnings.append(f"{c['key']}: {c['detail']}")

    if not config_exists:
        warnings.append(f"Config not yet generated — run pipeline_setup.py first: {config_path}")

    if created:
        warnings.append(f"Directories not yet created ({len(created)} missing) — run pipeline_setup.py to create them")

    result.update({
        "ready":       not blockers and not issues,
        "scale":       scale,
        "config_path": str(config_path),
        "config_exists": config_exists,
        "data_root":   str(data_root),
        "quality":     quality,
        "run_profile": {
            "chunks":        sz["chunks"],
            "steps_display": sz["steps_display"],
            "disk_gb_needed": sz["disk_gb"],
            "time_estimate": sz["time_display"],
        },
        "checks": check_map,
        "dirs": {
            "needs_creation": created,
            "existing_count": len(existing_dirs),
        },
        "existing_pipeline_state": {
            "found":          state["found"],
            "chunks_done":    done_chunks,
            "chunk_detail":   {str(k): v for k, v in state.get("chunks", {}).items()},
            "suggested_reset": suggested_reset,
            "precompute_cache": state.get("precompute_cache", {}),
        },
        "issues":   issues,
        "warnings": warnings,
        "commands": cmds,
    })

    print(json.dumps(result, indent=2))
    return 0 if result["ready"] else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Interactive pipeline setup wizard — run this first.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ai",         action="store_true",
                    help="Output JSON report instead of interactive wizard")
    ap.add_argument("--check",      action="store_true",
                    help="Check environment only; do not write files or create directories")
    ap.add_argument("--scale",      choices=list(RUN_SIZES.keys()),
                    help="Skip the scale selection prompt")
    ap.add_argument("--data-root",  metavar="PATH",
                    help="Data root directory (default: /Volumes/2TBSSD)")
    ap.add_argument("--model",      metavar="PATH", default="flux-klein-model",
                    help="Flux model path (relative to repo root or absolute)")
    # Quality flag overrides (useful with --ai --scale small --no-siglip etc.)
    ap.add_argument("--siglip",     dest="siglip",     action="store_true", default=None)
    ap.add_argument("--no-siglip",  dest="siglip",     action="store_false")
    ap.add_argument("--mine",       dest="mine",       action="store_true", default=None)
    ap.add_argument("--no-mine",    dest="mine",       action="store_false")
    ap.add_argument("--dedup",      dest="dedup",      action="store_true", default=None)
    ap.add_argument("--no-dedup",   dest="dedup",      action="store_false")
    ap.add_argument("--ema",        dest="mine_use_ema", action="store_true", default=None)
    ap.add_argument("--no-ema",     dest="mine_use_ema", action="store_false")
    ap.add_argument("--reset",      choices=["full", "partial", "resume"], default=None,
                    help="Reset mode: skip interactive menu and use this mode directly. "
                         "'partial' keeps raw downloads; 'full' also removes checkpoints "
                         "(archive/ is never deleted). Only acts when existing state is found.")

    args = ap.parse_args()

    if args.ai:
        return run_ai(args)

    # Apply any CLI flags to pre-fill the interactive wizard.
    return run_interactive(args)


if __name__ == "__main__":
    sys.exit(main())
