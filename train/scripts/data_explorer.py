#!/usr/bin/env python3
"""
train/scripts/data_explorer.py — Data intelligence layer for iris flywheel.

Commands:
  --overview              Disk, shard stats, flywheel summary, precompute versions
  --top-shards [N]        Ranked shard table from shard_scores.db
  --weights [--best]      Checkpoint listing with quality metrics
  --warm-start            Emit warm-start YAML snippet (use --apply to write it)
  --validate-coverage     Precompute coverage per encoder vs. shard pool
  --html PATH             Write self-contained HTML dashboard
  --ai                    JSON output for machine consumption

Usage:
  train/.venv/bin/python train/scripts/data_explorer.py --overview
  train/.venv/bin/python train/scripts/data_explorer.py --top-shards 20
  train/.venv/bin/python train/scripts/data_explorer.py --warm-start --from-flywheel sref-v1
  train/.venv/bin/python train/scripts/data_explorer.py --html /tmp/explorer.html
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Path constants (mirrors pipeline_lib)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline_lib import (
    CKPT_DIR,
    DATA_ROOT,
    FLYWHEEL_DB_PATH,
    PRECOMP_DIR,
    SHARD_SCORES_DB_PATH,
    SHARDS_DIR,
    free_gb,
    load_config,
    COLD_ROOT,
    COLD_PRECOMPUTE_DIR,
    COLD_WEIGHTS_DIR,
    COLD_METADATA_DIR,
    ABLATION_DB_PATH,
)

# Lazy imports — these modules may be absent in minimal environments
try:
    from flywheel_lib import FlywheelDB
    _HAS_FLYWHEEL_DB = True
except ImportError:
    _HAS_FLYWHEEL_DB = False

try:
    from shard_selector import ShardScoreDB
    _HAS_SHARD_DB = True
except ImportError:
    _HAS_SHARD_DB = False

try:
    from cache_manager import PrecomputeCache
    ENCODERS = ("qwen3", "vae", "siglip")
    _HAS_CACHE_MGR = True
except ImportError:
    _HAS_CACHE_MGR = False
    ENCODERS = ("qwen3", "vae", "siglip")


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _fmt_gb(gb: float) -> str:
    return f"{gb:.1f} GB"


def _opt_f(v, fmt=".4f") -> str:
    return f"{v:{fmt}}" if v is not None else "—"


def collect_overview() -> dict:
    out: dict = {}

    # Disk
    try:
        out["disk_free_gb"] = free_gb(DATA_ROOT)
    except Exception:
        out["disk_free_gb"] = None
    try:
        usage = shutil.disk_usage(str(DATA_ROOT))
        out["disk_total_gb"] = usage.total / 1e9
        out["disk_used_gb"]  = usage.used  / 1e9
    except Exception:
        out["disk_total_gb"] = out["disk_used_gb"] = None

    # Shards
    try:
        shard_paths = list(SHARDS_DIR.glob("*.tar")) + list(SHARDS_DIR.glob("*.tgz"))
        out["shard_count"] = len(shard_paths)
    except Exception:
        out["shard_count"] = None

    # Shard scores DB
    out["shard_stats"] = None
    if _HAS_SHARD_DB and SHARD_SCORES_DB_PATH.exists():
        try:
            db = ShardScoreDB(SHARD_SCORES_DB_PATH)
            out["shard_stats"] = db.get_stats()
            db.close()
        except Exception:
            pass

    # Checkpoints
    try:
        ckpts = sorted(CKPT_DIR.glob("step_*.safetensors"))
        out["checkpoint_count"] = len(ckpts)
        out["latest_checkpoint"] = str(ckpts[-1]) if ckpts else None
    except Exception:
        out["checkpoint_count"] = 0
        out["latest_checkpoint"] = None

    # Flywheel runs
    out["flywheel_runs"] = []
    if _HAS_FLYWHEEL_DB and FLYWHEEL_DB_PATH.exists():
        try:
            fw = FlywheelDB(FLYWHEEL_DB_PATH)
            names = fw.get_all_run_names()
            for name in names:
                iters = fw.get_iterations(name)
                done  = [i for i in iters if i.get("status") == "done"]
                best  = fw.get_best(name)
                out["flywheel_runs"].append({
                    "name":          name,
                    "total_iters":   len(iters),
                    "done_iters":    len(done),
                    "best_cond_gap": best.get("cond_gap") if best else None,
                    "best_iter":     best.get("iteration") if best else None,
                    "best_ckpt":     best.get("checkpoint") if best else None,
                })
            fw.close()
        except Exception:
            pass

    # Precompute versions
    out["precompute"] = {}
    if _HAS_CACHE_MGR:
        for enc in ENCODERS:
            try:
                versions = PrecomputeCache.list_versions(PRECOMP_DIR, enc)
                cur = PrecomputeCache.current_dir(PRECOMP_DIR, enc)
                out["precompute"][enc] = {
                    "versions":     versions,
                    "current":      str(cur) if cur else None,
                    "n_versions":   len(versions),
                    "current_records": 0,
                }
                for v in versions:
                    if v.get("current"):
                        out["precompute"][enc]["current_records"] = v.get("record_count", 0)
            except Exception:
                out["precompute"][enc] = {"versions": [], "current": None, "n_versions": 0}

    return out


def collect_top_shards(limit: int = 30, source_filter: Optional[str] = None) -> list[dict]:
    if not _HAS_SHARD_DB or not SHARD_SCORES_DB_PATH.exists():
        return []
    try:
        db = ShardScoreDB(SHARD_SCORES_DB_PATH)
        shards = db.get_scored_shards()
        db.close()
    except Exception:
        return []

    if source_filter:
        shards = [s for s in shards if s.get("source", "") == source_filter]

    return shards[:limit]


def collect_weights(best_only: bool = False, flywheel_name: Optional[str] = None) -> list[dict]:
    """Return checkpoint list with quality metrics from checkpoint_log."""
    rows: list[dict] = []

    if _HAS_FLYWHEEL_DB and FLYWHEEL_DB_PATH.exists():
        try:
            fw = FlywheelDB(FLYWHEEL_DB_PATH)
            names = [flywheel_name] if flywheel_name else fw.get_all_run_names()

            # Determine actual best iteration per flywheel by cond_gap (not DB is_best flag,
            # which may be stale if the best-checkpoint criterion changed).
            best_iters: dict[str, Optional[int]] = {}
            for name in names:
                b = fw.get_best(name)
                best_iters[name] = b.get("iteration") if b else None

            for name in names:
                history = fw.get_checkpoint_history(name)
                for r in history:
                    is_best = (r.get("iteration") == best_iters.get(name))
                    if best_only and not is_best:
                        continue
                    rows.append({
                        "flywheel":   name,
                        "iteration":  r.get("iteration"),
                        "path":       r.get("checkpoint_path"),
                        "hash":       r.get("checkpoint_hash"),
                        "cond_gap":   r.get("cond_gap"),
                        "ref_gap":    r.get("ref_gap"),
                        "train_loss": r.get("train_loss"),
                        "is_best":    is_best,
                        "ts":         r.get("ts"),
                    })
            fw.close()
        except Exception:
            pass
    else:
        # Fall back to scanning CKPT_DIR directly
        try:
            for p in sorted(CKPT_DIR.glob("step_*.safetensors")):
                rows.append({
                    "flywheel":   None,
                    "iteration":  None,
                    "path":       str(p),
                    "hash":       p.stem[:12],
                    "cond_gap":   None,
                    "ref_gap":    None,
                    "train_loss": None,
                    "is_best":    False,
                    "ts":         None,
                })
        except Exception:
            pass

    return rows


def validate_coverage() -> dict:
    """Check precompute coverage vs. shard pool."""
    result: dict = {}

    # Count shards in pool
    try:
        pool_shards = set(p.stem for p in SHARDS_DIR.glob("*.tar"))
        pool_shards |= set(p.stem for p in SHARDS_DIR.glob("*.tgz"))
        result["pool_shards"] = len(pool_shards)
    except Exception:
        pool_shards = set()
        result["pool_shards"] = 0

    if not _HAS_CACHE_MGR:
        return result

    result["encoders"] = {}
    for enc in ENCODERS:
        try:
            cur = PrecomputeCache.effective_dir(PRECOMP_DIR, enc)
            if cur is None:
                result["encoders"][enc] = {"status": "missing", "records": 0, "coverage_pct": 0}
                continue
            records = sum(1 for f in cur.iterdir() if f.suffix == ".npz")
            # Coverage: compare record shard IDs against pool
            cached_shards = set(f.stem.split("_")[0] for f in cur.iterdir() if f.suffix == ".npz")
            covered = len(cached_shards & pool_shards) if pool_shards else 0
            pct = 100.0 * covered / len(pool_shards) if pool_shards else 0.0
            result["encoders"][enc] = {
                "status":       "ok",
                "path":         str(cur),
                "records":      records,
                "cached_shards": len(cached_shards),
                "covered_pool": covered,
                "coverage_pct": pct,
            }
        except Exception as e:
            result["encoders"][enc] = {"status": "error", "error": str(e)}

    return result


def generate_warm_start(
    flywheel_name: str,
    iteration: Optional[int] = None,
) -> dict:
    """Return warm-start info: config snippet and best checkpoint path."""
    out: dict = {"flywheel_name": flywheel_name, "iteration": iteration}

    if not _HAS_FLYWHEEL_DB or not FLYWHEEL_DB_PATH.exists():
        out["error"] = "flywheel_history.db not found"
        return out

    try:
        fw = FlywheelDB(FLYWHEEL_DB_PATH)

        if iteration is not None:
            iters = fw.get_iterations(flywheel_name)
            match = [i for i in iters if i.get("iteration") == iteration]
            row = match[0] if match else None
        else:
            row = fw.get_best(flywheel_name)

        if row is None:
            out["error"] = f"no completed iteration found for flywheel '{flywheel_name}'"
            fw.close()
            return out

        ckpt = row.get("checkpoint")
        if ckpt and not Path(ckpt).exists():
            out["warning"] = f"checkpoint path does not exist: {ckpt}"

        out["checkpoint"]  = ckpt
        out["iteration"]   = row.get("iteration")
        out["cond_gap"]    = row.get("cond_gap")
        out["ref_gap"]     = row.get("ref_gap")
        out["train_loss"]  = row.get("train_loss")

        # YAML snippet to paste into flywheel config
        if ckpt:
            out["yaml_snippet"] = f"base_checkpoint: \"{ckpt}\""
        else:
            out["yaml_snippet"] = "# no checkpoint found"

        fw.close()
    except Exception as e:
        out["error"] = str(e)

    return out


def _apply_warm_start(config_path: Path, checkpoint: str) -> str:
    """Patch base_checkpoint in flywheel YAML config. Returns message."""
    try:
        text = config_path.read_text()
    except Exception as e:
        return f"error reading {config_path}: {e}"

    # Replace existing base_checkpoint line, preserving leading whitespace.
    # Capture leading spaces so the key stays in its YAML section.
    pattern = re.compile(r'^(\s*)base_checkpoint\s*:.*$', re.MULTILINE)
    esc = checkpoint.replace('"', '\\"')
    if pattern.search(text):
        new_text = pattern.sub(lambda m: f'{m.group(1)}base_checkpoint: "{esc}"', text)
    else:
        # Append under model: section header with 2-space indent
        new_text = re.sub(
            r'(^model\s*:.*$)',
            lambda m: m.group(0) + f'\n  base_checkpoint: "{esc}"',
            text,
            count=1,
            flags=re.MULTILINE,
        )

    try:
        config_path.write_text(new_text)
        return f"patched {config_path}"
    except Exception as e:
        return f"error writing {config_path}: {e}"


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def _bar(pct: float, width: int = 20) -> str:
    filled = int(width * pct / 100)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {pct:.0f}%"


def print_overview(ov: dict) -> None:
    free  = ov.get("disk_free_gb")
    total = ov.get("disk_total_gb")
    used  = ov.get("disk_used_gb")
    print("\n=== Overview ===")
    if total:
        pct = 100.0 * (used or 0) / total
        print(f"  Disk  {_bar(pct)}  {_fmt_gb(used or 0)} used / {_fmt_gb(total)} total  ({_fmt_gb(free or 0)} free)")
    else:
        print(f"  Disk  free: {_fmt_gb(free or 0)}")

    ss = ov.get("shard_stats")
    sc = ov.get("shard_count", 0)
    if ss:
        print(f"  Shards  pool={sc}  registered={ss.get('total',0)}  "
              f"scored={ss.get('scored',0)}  attr_ready={ss.get('attr_ready',0)}")
    else:
        print(f"  Shards  pool={sc}")

    print(f"  Checkpoints  {ov.get('checkpoint_count', 0)} in CKPT_DIR")

    runs = ov.get("flywheel_runs", [])
    if runs:
        print(f"  Flywheel runs: {len(runs)}")
        for r in runs:
            best_cg = _opt_f(r.get("best_cond_gap"))
            print(f"    {r['name']}  iters={r['done_iters']}/{r['total_iters']}  "
                  f"best_cond_gap={best_cg}  iter={r.get('best_iter','—')}")
    else:
        print("  Flywheel runs: none")

    pc = ov.get("precompute", {})
    if pc:
        print("  Precompute:")
        for enc, info in pc.items():
            cur_rec = info.get("current_records", 0)
            n_ver   = info.get("n_versions", 0)
            print(f"    {enc:<8}  versions={n_ver}  current_records={cur_rec:,}")


def print_top_shards(shards: list[dict], limit: int) -> None:
    print(f"\n=== Top {min(limit, len(shards))} Shards (by effective_score) ===")
    if not shards:
        print("  (no scored shards)")
        return
    hdr = f"  {'shard_id':<14}  {'source':<6}  {'eff_score':>9}  {'comp':>8}  "
    hdr += f"{'cond_gap_mean':>13}  {'n_scored':>8}  {'n_sel':>5}  {'attr_conf':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for s in shards:
        sid  = (s.get("shard_id") or "")[:14]
        src  = (s.get("source") or "")[:6]
        eff  = _opt_f(s.get("effective_score"), ".4f")
        comp = _opt_f(s.get("composite_score"), ".4f")
        cgm  = _opt_f(s.get("cond_gap_mean"), ".4f")
        ns   = s.get("n_scored", 0)
        nsel = s.get("n_selected", 0)
        ac   = _opt_f(s.get("attr_confidence"), ".3f")
        print(f"  {sid:<14}  {src:<6}  {eff:>9}  {comp:>8}  {cgm:>13}  {ns:>8}  {nsel:>5}  {ac:>9}")


def print_weights(rows: list[dict], best_only: bool) -> None:
    label = "Best Checkpoint" if best_only else "All Checkpoints"
    print(f"\n=== {label} ===")
    if not rows:
        print("  (none found)")
        return
    hdr = f"  {'flywheel':<12}  {'iter':>4}  {'cond_gap':>9}  {'ref_gap':>8}  {'loss':>7}  {'best':>5}  path"
    print(hdr)
    print("  " + "-" * 70)
    for r in rows:
        fw   = (r.get("flywheel") or "—")[:12]
        it   = str(r.get("iteration") or "—")
        cg   = _opt_f(r.get("cond_gap"), ".4f")
        rg   = _opt_f(r.get("ref_gap"),  ".4f")
        lo   = _opt_f(r.get("train_loss"),".4f")
        best = "★" if r.get("is_best") else " "
        path = r.get("path") or "—"
        print(f"  {fw:<12}  {it:>4}  {cg:>9}  {rg:>8}  {lo:>7}  {best:>5}  {path}")


def print_warm_start(ws: dict) -> None:
    print("\n=== Warm-Start Recommendation ===")
    if "error" in ws:
        print(f"  Error: {ws['error']}")
        return
    if "warning" in ws:
        print(f"  Warning: {ws['warning']}")
    print(f"  Flywheel : {ws.get('flywheel_name')}")
    print(f"  Iteration: {ws.get('iteration')}")
    print(f"  cond_gap : {_opt_f(ws.get('cond_gap'))}")
    print(f"  Checkpoint: {ws.get('checkpoint') or '—'}")
    print()
    print("  Paste into flywheel config:")
    print(f"    {ws.get('yaml_snippet', '')}")


def print_validate(cov: dict) -> None:
    print("\n=== Precompute Coverage ===")
    print(f"  Pool shards: {cov.get('pool_shards', 0)}")
    enc = cov.get("encoders", {})
    if not enc:
        print("  (cache_manager not available)")
        return
    for name, info in enc.items():
        status = info.get("status", "?")
        if status == "ok":
            pct = info.get("coverage_pct", 0)
            print(f"  {name:<8}  {_bar(pct, 16)}  "
                  f"{info.get('covered_pool',0)}/{cov.get('pool_shards',0)} shards  "
                  f"{info.get('records',0):,} records")
        else:
            print(f"  {name:<8}  {status}  {info.get('error', '')}")


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_HTML_STYLE = """
*{box-sizing:border-box}
body{font-family:system-ui,sans-serif;margin:0;background:#111827;color:#e5e7eb;font-size:13px}
.hdr{background:#0f172a;color:#f1f5f9;padding:14px 20px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #1e293b}
.hdr h1{margin:0;font-size:16px;font-weight:700;color:#f8fafc}.ts{font-size:11px;color:#94a3b8}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;padding:16px 20px}
.card{background:#1e293b;border-radius:8px;padding:14px 18px;border:1px solid #334155}
.card-full{background:#1e293b;border-radius:8px;padding:14px 18px;margin:0 20px 16px;border:1px solid #334155}
h2{margin:0 0 10px;font-size:10px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.8px}
.stat-val{font-size:24px;font-weight:700;font-family:monospace;color:#f8fafc}
.stat-sub{font-size:11px;color:#94a3b8;margin-top:2px}
table{border-collapse:collapse;width:100%;font-size:12px}
td,th{padding:6px 10px;border-bottom:1px solid #1e293b;text-align:left;vertical-align:top;white-space:nowrap}
th{background:#0f172a;font-weight:600;color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:.5px}
tr:hover td{background:#243044}
tr:last-child td{border-bottom:none}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700}
.ok{background:#14532d;color:#86efac}.warn{background:#713f12;color:#fde68a}
.err{background:#7f1d1d;color:#fca5a5}.info{background:#0c4a6e;color:#7dd3fc}
.best{color:#fbbf24;font-weight:700}
.mono{font-family:monospace;font-size:11px;color:#94a3b8}
.path{font-family:monospace;font-size:10px;color:#64748b;max-width:360px;overflow:hidden;text-overflow:ellipsis}
.pbar{height:6px;background:#334155;border-radius:3px;overflow:hidden;margin:4px 0;min-width:80px}
.pbar-fill{height:100%;border-radius:3px;background:#3b82f6}
.pbar-warn{background:#f59e0b}
.pbar-crit{background:#ef4444}
code{background:#0f172a;padding:2px 6px;border-radius:3px;font-size:11px;color:#7dd3fc}
pre{background:#0f172a;color:#a5f3fc;padding:12px;border-radius:6px;font-size:11px;overflow-x:auto;margin:8px 0}
"""


def _h(s) -> str:
    import html as _html_mod
    return _html_mod.escape(str(s)) if s is not None else ""


def _pbar_html(pct: float) -> str:
    cls = "pbar-fill"
    if pct < 50:
        cls = "pbar-fill pbar-crit"
    elif pct < 80:
        cls = "pbar-fill pbar-warn"
    return f'<div class="pbar"><div class="{cls}" style="width:{min(100,pct):.0f}%"></div></div>'


def _stat_card(title: str, value, sub: str = "") -> str:
    return (
        f'<div class="card"><h2>{_h(title)}</h2>'
        f'<div class="stat-val">{_h(value)}</div>'
        + (f'<div class="stat-sub">{_h(sub)}</div>' if sub else "")
        + "</div>"
    )


def _table(headers: list[str], rows: list[list]) -> str:
    ths = "".join(f"<th>{_h(h)}</th>" for h in headers)
    trs = ""
    for r in rows:
        trs += "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    return f"<table><tr>{ths}</tr>{trs}</table>"


def render_html(ov: dict, top_shards: list[dict], weights: list[dict], cov: dict) -> str:
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Summary cards
    disk_free = ov.get("disk_free_gb")
    disk_total = ov.get("disk_total_gb")
    ss = ov.get("shard_stats") or {}
    disk_str = f"{disk_free:.1f} GB" if disk_free is not None else "?"
    disk_sub = f"{ov.get('disk_used_gb', 0):.1f} / {disk_total:.1f} GB" if disk_total else ""
    cards = [
        _stat_card("Disk Free", disk_str, disk_sub),
        _stat_card("Shards (pool)", ov.get("shard_count", 0)),
        _stat_card("Scored Shards", ss.get("scored", "?"), f"of {ss.get('total','?')} registered"),
        _stat_card("Checkpoints", ov.get("checkpoint_count", 0)),
    ]
    for run in ov.get("flywheel_runs", []):
        cg = run.get("best_cond_gap")
        cg_str = f"{cg:.4f}" if cg is not None else "?"
        cards.append(_stat_card(
            f"Flywheel: {run['name']}",
            cg_str,
            f"{run['done_iters']}/{run['total_iters']} iters  best=iter {run.get('best_iter','?')}",
        ))
    grid = '<div class="grid">' + "".join(cards) + "</div>"

    # Top shards table
    shard_rows = []
    for s in top_shards:
        eff  = _opt_f(s.get("effective_score"))
        comp = _opt_f(s.get("composite_score"))
        cgm  = _opt_f(s.get("cond_gap_mean"))
        ac   = _opt_f(s.get("attr_confidence"), ".3f")
        ns   = s.get("n_scored", 0)
        nsel = s.get("n_selected", 0)
        shard_rows.append([
            f'<span class="mono">{_h(s.get("shard_id",""))}</span>',
            _h(s.get("source", "")),
            f'<span class="mono">{eff}</span>',
            f'<span class="mono">{comp}</span>',
            f'<span class="mono">{cgm}</span>',
            str(ns), str(nsel),
            f'<span class="mono">{ac}</span>',
        ])
    shards_section = (
        '<div class="card-full"><h2>Top Shards (by effective_score)</h2>'
        + (_table(
            ["Shard ID", "Source", "Eff Score", "Composite", "cond_gap_mean",
             "N Scored", "N Sel", "Attr Conf"],
            shard_rows,
        ) if shard_rows else "<p style='color:#64748b'>No scored shards found.</p>")
        + "</div>"
    )

    # Weights / checkpoint table
    ckpt_rows = []
    for r in weights:
        cg   = _opt_f(r.get("cond_gap"))
        rg   = _opt_f(r.get("ref_gap"))
        lo   = _opt_f(r.get("train_loss"))
        best = '<span class="best">★</span>' if r.get("is_best") else ""
        path = r.get("path") or "—"
        ckpt_rows.append([
            _h(r.get("flywheel") or "—"),
            str(r.get("iteration") or "—"),
            f'<span class="mono">{cg}</span>',
            f'<span class="mono">{rg}</span>',
            f'<span class="mono">{lo}</span>',
            best,
            f'<span class="path" title="{_h(path)}">{_h(Path(path).name if path != "—" else "—")}</span>',
        ])
    weights_section = (
        '<div class="card-full"><h2>Checkpoints</h2>'
        + (_table(
            ["Flywheel", "Iter", "cond_gap", "ref_gap", "Loss", "Best", "File"],
            ckpt_rows,
        ) if ckpt_rows else "<p style='color:#64748b'>No checkpoints found.</p>")
        + "</div>"
    )

    # Precompute coverage
    cov_rows = []
    pool_n = cov.get("pool_shards", 0)
    for enc, info in cov.get("encoders", {}).items():
        if info.get("status") == "ok":
            pct = info.get("coverage_pct", 0)
            pbar = _pbar_html(pct)
            cov_rows.append([
                _h(enc),
                f'{pbar}<span class="mono">{pct:.0f}%</span>',
                f'{info.get("covered_pool",0)}/{pool_n}',
                f'{info.get("records", 0):,}',
                f'<span class="path">{_h(info.get("path",""))}</span>',
            ])
        else:
            cov_rows.append([_h(enc), '<span class="err badge">missing</span>', "—", "—", ""])
    cov_section = (
        '<div class="card-full"><h2>Precompute Coverage</h2>'
        + (_table(["Encoder", "Coverage", "Shards", "Records", "Path"], cov_rows)
           if cov_rows else "<p style='color:#64748b'>cache_manager not available.</p>")
        + "</div>"
    )

    body = (
        f'<div class="hdr"><h1>iris.c — Data Explorer</h1>'
        f'<span class="ts">{_h(ts)}</span></div>'
        + grid + shards_section + weights_section + cov_section
    )

    return (
        '<!DOCTYPE html><html lang="en"><head>'
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<style>{_HTML_STYLE}</style>'
        f'</head><body>{body}</body></html>'
    )


# ---------------------------------------------------------------------------
# Cold storage subcommand interface (PIPELINE-28)
# Subcommands: status, shards, weights, suggest-warmstart,
#              ablation, compare, maintenance
# ---------------------------------------------------------------------------

def _free_gb_cold(path: Path) -> Optional[float]:
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize / (1024 ** 3)
    except OSError:
        return None


def _col_table(rows: list[list[str]], headers: list[str], min_width: int = 6) -> str:
    """Simple fixed-width column formatter — no external deps."""
    all_rows = [headers] + rows
    widths = [max(len(str(cell)) for cell in col) for col in zip(*all_rows)]
    widths = [max(w, min_width) for w in widths]
    sep = "  ".join("-" * w for w in widths)
    lines = []
    for i, row in enumerate(all_rows):
        lines.append("  ".join(str(cell).ljust(w) for cell, w in zip(row, widths)))
        if i == 0:
            lines.append(sep)
    return "\n".join(lines)


def _open_ro_db(path: Path) -> Optional[sqlite3.Connection]:
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error:
        return None


def _list_campaigns(weights_dir: Path) -> list[Path]:
    if not weights_dir.exists():
        return []
    return sorted(p for p in weights_dir.iterdir()
                  if p.is_dir() and p.name.startswith("flywheel-"))


def _json_print(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


# ── status ─────────────────────────────────────────────────────────────────

def _cmd_status(args: argparse.Namespace) -> int:
    cfg: dict = {}
    if args.config:
        try:
            cfg = load_config(args.config)
        except FileNotFoundError:
            pass
    storage  = cfg.get("storage", {})
    cold_root = Path(storage.get("cold_root", COLD_ROOT))
    hot_root  = Path(storage.get("hot_root",  DATA_ROOT))

    cold_free = _free_gb_cold(cold_root) if cold_root.exists() else None
    hot_free  = _free_gb_cold(hot_root)  if hot_root.exists()  else None

    # Pools
    def _pool_count(root: Optional[str], sub: str) -> Optional[int]:
        if not root:
            return None
        p = Path(root) / sub
        return sum(1 for _ in p.iterdir()) if p.exists() else 0

    raw_pool_root  = storage.get("raw_pool_root")
    conv_pool_root = storage.get("converted_pool_root")

    # Precompute (cold)
    cold_precomp: dict = {}
    cold_precomp_dir = cold_root / "precomputed"
    if cold_root.exists():
        for enc in ("qwen3", "vae", "siglip"):
            cur = cold_precomp_dir / enc / "current"
            if cur.is_symlink():
                ver = os.path.basename(os.readlink(str(cur)))
                mf_path = cold_precomp_dir / enc / ver / "manifest.json"
                entry: dict[str, Any] = {"version": ver}
                if mf_path.exists():
                    try:
                        m = json.loads(mf_path.read_text())
                        entry["complete"] = m.get("complete", False)
                        entry["records"]  = m.get("records",  0)
                    except (ValueError, OSError):
                        pass
                cold_precomp[enc] = entry

    # Weights
    cold_weights = cold_root / "weights"
    campaigns = _list_campaigns(cold_weights)
    latest_steps = 0
    if campaigns:
        latest_steps = sum(
            1 for f in campaigns[-1].iterdir()
            if f.suffix == ".safetensors" and f.stem != "final"
        )

    # DBs
    def _db_info(path: Path, table: str) -> dict:
        if not path.exists():
            return {"rows": None, "age_h": None}
        conn = _open_ro_db(path)
        rows = None
        if conn:
            try:
                rows = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except sqlite3.Error:
                pass
            conn.close()
        age = round((datetime.now(timezone.utc).timestamp() - path.stat().st_mtime) / 3600, 1)
        return {"rows": rows, "age_h": age}

    result: dict[str, Any] = {
        "ok": True,
        "disk": {
            "cold_free_gb": round(cold_free, 1) if cold_free is not None else None,
            "hot_free_gb":  round(hot_free,  1) if hot_free  is not None else None,
        },
        "pools": {
            "raw_pool":       {"root": raw_pool_root,  "downloaded": _pool_count(raw_pool_root,  ".downloaded")},
            "converted_pool": {"root": conv_pool_root, "converted":  _pool_count(conv_pool_root, ".converted")},
        },
        "cold_precompute": cold_precomp,
        "weights": {
            "campaign_count":  len(campaigns),
            "latest_campaign": campaigns[-1].name.replace("flywheel-", "") if campaigns else None,
            "latest_steps":    latest_steps,
        },
        "databases": {
            "shard_scores":     _db_info(COLD_METADATA_DIR / "shard_scores.db"     if (COLD_METADATA_DIR / "shard_scores.db").exists()     else SHARD_SCORES_DB_PATH, "shards"),
            "ablation_history": _db_info(COLD_METADATA_DIR / "ablation_history.db" if (COLD_METADATA_DIR / "ablation_history.db").exists() else ABLATION_DB_PATH,      "experiments"),
        },
    }

    if args.json:
        _json_print(result)
        return 0

    print(f"\n{'─'*52}")
    print(" data_explorer — pipeline cold storage status")
    print(f"{'─'*52}")
    d = result["disk"]
    cold_str = f"{d['cold_free_gb']:.1f} GB free" if d["cold_free_gb"] is not None else "not mounted"
    hot_str  = f"{d['hot_free_gb']:.1f} GB free"  if d["hot_free_gb"]  is not None else "unavailable"
    print(f"\n  Disk       cold={cold_str}  hot={hot_str}")

    rp = result["pools"]["raw_pool"]
    cp = result["pools"]["converted_pool"]
    rp_str = f"{rp['downloaded']} downloaded" if rp["downloaded"] is not None else "not configured"
    cp_str = f"{cp['converted']} converted"   if cp["converted"]  is not None else "not configured"
    print(f"  Pools      raw={rp_str}  converted={cp_str}")

    if cold_precomp:
        print(f"\n  Precompute (cold):")
        for enc, info in cold_precomp.items():
            ver  = info.get("version", "?")
            comp = "complete" if info.get("complete") else "partial"
            rec  = f"{info.get('records', 0):,}"
            print(f"    {enc:<8}  {ver}  {comp}  {rec} records")
    else:
        print("\n  Precompute (cold): none archived yet")

    w = result["weights"]
    print(f"\n  Weights    {w['campaign_count']} campaign(s)", end="")
    if w["latest_campaign"]:
        print(f"  latest={w['latest_campaign']}  steps={w['latest_steps']}", end="")
    print()

    print(f"\n  Databases:")
    for name, info in result["databases"].items():
        rows = info["rows"]
        age  = info["age_h"]
        rs = f"{rows:,}" if rows is not None else "—"
        ag = f"{age}h ago" if age is not None else "—"
        print(f"    {name:<20}  {rs:>8} rows  updated {ag}")
    print()
    return 0


# ── shards ──────────────────────────────────────────────────────────────────

def _cmd_shards(args: argparse.Namespace) -> int:
    cold_db = COLD_METADATA_DIR / "shard_scores.db"
    db_path = cold_db if cold_db.exists() else SHARD_SCORES_DB_PATH
    conn = _open_ro_db(db_path)
    if conn is None:
        if args.json:
            _json_print({"ok": False, "error": "shard_scores.db not found", "shards": []})
        else:
            print("shard_scores.db not found — run flywheel first")
        return 0

    valid = {"effective_score", "composite_score", "n_scored"}
    sort_col = args.sort if args.sort in valid else "effective_score"
    top_n = args.top if args.top > 0 else 20

    try:
        rows = conn.execute(f"""
            SELECT shard_id, effective_score, composite_score, n_scored,
                   score_ckpt_iter, cond_gap_mean, ref_gap_mean
            FROM shards
            ORDER BY {sort_col} DESC NULLS LAST
            LIMIT ?
        """, (top_n,)).fetchall()
    except sqlite3.Error as e:
        conn.close()
        if args.json:
            _json_print({"ok": False, "error": str(e), "shards": []})
        else:
            print(f"DB error: {e}")
        return 1
    conn.close()

    data = [dict(r) for r in rows]
    if args.json:
        _json_print({"ok": True, "sort": sort_col, "db": str(db_path), "shards": data})
        return 0

    if not data:
        print("No shards in database.")
        return 0

    table_rows = []
    for r in data:
        eff   = f"{r['effective_score']:.4f}" if r["effective_score"] is not None else "—"
        comp  = f"{r['composite_score']:.4f}" if r["composite_score"] is not None else "—"
        n     = str(r["n_scored"] or 0)
        ckpt  = str(r["score_ckpt_iter"]  or "—")
        cg    = f"{r['cond_gap_mean']:.4f}" if r["cond_gap_mean"] is not None else "—"
        rg    = f"{r['ref_gap_mean']:.4f}"  if r["ref_gap_mean"]  is not None else "—"
        table_rows.append([r["shard_id"], eff, comp, cg, rg, n, ckpt])

    print(f"\n  {db_path}  (sorted by {sort_col}, top {len(data)})\n")
    print(_col_table(table_rows,
                     ["shard_id", "eff_score", "comp_score", "cond_gap_mean", "ref_gap_mean", "n_scored", "ckpt_iter"]))
    print()
    return 0


# ── weights ─────────────────────────────────────────────────────────────────

def _cmd_weights(args: argparse.Namespace) -> int:
    cfg: dict = {}
    if args.config:
        try:
            cfg = load_config(args.config)
        except FileNotFoundError:
            pass
    cold_root   = Path(cfg.get("storage", {}).get("cold_root", COLD_ROOT))
    cold_weights = cold_root / "weights"
    campaigns   = _list_campaigns(cold_weights)

    if args.campaign:
        campaigns = [c for c in campaigns if c.name == f"flywheel-{args.campaign}"]
        if not campaigns:
            if args.json:
                _json_print({"ok": False, "error": f"campaign {args.campaign} not found"})
            else:
                print(f"Campaign flywheel-{args.campaign} not found.")
            return 1

    result_list = []
    for camp in campaigns:
        safetensors = sorted(f for f in camp.iterdir()
                             if f.suffix == ".safetensors" and f.stem != "final")
        steps = []
        for sf in safetensors:
            j = sf.with_suffix(".json")
            meta: dict = {}
            if j.exists():
                try:
                    meta = json.loads(j.read_text())
                except (ValueError, OSError):
                    pass
            steps.append({"name": sf.name, **meta})

        best_links: dict = {}
        best_dir = cold_weights / "best"
        if best_dir.exists():
            for lnk in best_dir.iterdir():
                if lnk.suffix == ".safetensors" and lnk.is_symlink():
                    try:
                        resolved = (best_dir / os.readlink(str(lnk))).resolve()
                        if resolved.parent == camp.resolve():
                            best_links[lnk.stem] = resolved.name
                    except OSError:
                        pass

        result_list.append({
            "campaign":   camp.name,
            "date":       camp.name.replace("flywheel-", ""),
            "step_count": len(steps),
            "steps":      steps if args.campaign else [],
            "best_for":   best_links,
        })

    if args.json:
        _json_print({"ok": True, "campaigns": result_list})
        return 0

    if not result_list:
        print("No weight campaigns found in cold storage.")
        return 0

    if args.campaign and result_list:
        camp = result_list[0]
        print(f"\n  Campaign: {camp['campaign']}  steps={camp['step_count']}")
        if camp["best_for"]:
            print(f"  Best-for: {', '.join(f'{k}={v}' for k,v in camp['best_for'].items())}")
        if camp["steps"]:
            rows = []
            for s in camp["steps"]:
                cg = f"{s['cond_gap']:.4f}" if "cond_gap" in s else "—"
                rg = f"{s['ref_gap']:.4f}"  if "ref_gap"  in s else "—"
                ci = f"{s['clip_i']:.4f}"   if "clip_i"   in s else "—"
                rows.append([s["name"], cg, rg, ci])
            print()
            print(_col_table(rows, ["checkpoint", "cond_gap", "ref_gap", "clip_i"]))
        print()
    else:
        rows = [[c["campaign"], str(c["step_count"]),
                 ", ".join(c["best_for"].keys()) or "—"] for c in result_list]
        print(f"\n  Weight archive — {cold_weights}\n")
        print(_col_table(rows, ["campaign", "steps", "best_for"]))
        print()
    return 0


# ── suggest-warmstart ────────────────────────────────────────────────────────

def _cmd_suggest_warmstart(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        if args.json:
            _json_print({"ok": False, "error": f"config not found: {args.config}"})
        else:
            print(f"Config not found: {args.config}")
        return 1

    cold_root    = Path(cfg.get("storage", {}).get("cold_root", COLD_ROOT))
    cold_weights = cold_root / "weights"
    campaigns    = _list_campaigns(cold_weights)

    if not campaigns:
        r = {"ok": True, "recommendation": "train_from_scratch",
             "reason": "No archived weights found."}
        _json_print(r) if args.json else print("Recommendation: train from scratch (no archived weights).")
        return 0

    best: Optional[dict] = None
    for camp in reversed(campaigns):
        for sf in sorted(camp.iterdir()):
            if sf.suffix != ".safetensors" or sf.stem == "final":
                continue
            j = sf.with_suffix(".json")
            if not j.exists():
                continue
            try:
                meta = json.loads(j.read_text())
            except (ValueError, OSError):
                continue
            cg = meta.get("cond_gap")
            if cg is None:
                continue
            if best is None or cg > best["cond_gap"]:
                best = {"cond_gap": cg, "path": str(sf),
                        "campaign": camp.name, "meta": meta}

    if best is None:
        r = {"ok": True, "recommendation": "train_from_scratch",
             "reason": "No checkpoints with cond_gap metrics found."}
        _json_print(r) if args.json else print("Recommendation: train from scratch (no scored checkpoints).")
        return 0

    result = {
        "ok":             True,
        "recommendation": "warmstart",
        "warmstart_path": best["path"],
        "campaign":       best["campaign"],
        "cond_gap":       best["cond_gap"],
        "cli_flags":      f"--warmstart \"{best['path']}\"",
    }
    if args.json:
        _json_print(result)
        return 0

    print(f"\n  Recommended warmstart:")
    print(f"    campaign  : {best['campaign']}")
    print(f"    checkpoint: {best['path']}")
    print(f"    cond_gap  : {best['cond_gap']:.4f}")
    print(f"\n  Use: {result['cli_flags']}\n")
    return 0


# ── ablation ────────────────────────────────────────────────────────────────

def _cmd_ablation(args: argparse.Namespace) -> int:
    cold_db = COLD_METADATA_DIR / "ablation_history.db"
    db_path = cold_db if cold_db.exists() else ABLATION_DB_PATH
    conn = _open_ro_db(db_path)
    if conn is None:
        if args.json:
            _json_print({"ok": False, "error": "ablation_history.db not found", "experiments": []})
        else:
            print("ablation_history.db not found — run ablation harness first.")
        return 0

    try:
        rows = conn.execute("""
            SELECT id, params, score, verdict, ref_gap, cond_gap, final_loss, steps, ts
            FROM experiments ORDER BY ts DESC
        """).fetchall()
    except sqlite3.Error as e:
        conn.close()
        if args.json:
            _json_print({"ok": False, "error": str(e), "experiments": []})
        else:
            print(f"DB error: {e}")
        return 1
    conn.close()

    experiments = [dict(r) for r in rows]
    if args.campaign:
        experiments = [e for e in experiments
                       if args.campaign in str(e.get("ts", ""))]

    if args.pareto:
        dominated: set[int] = set()
        for i, a in enumerate(experiments):
            for j, b in enumerate(experiments):
                if i == j or j in dominated:
                    continue
                a_cg = a.get("cond_gap") or 0.0
                b_cg = b.get("cond_gap") or 0.0
                a_rg = a.get("ref_gap")  or 0.0
                b_rg = b.get("ref_gap")  or 0.0
                if b_cg >= a_cg and b_rg >= a_rg and (b_cg > a_cg or b_rg > a_rg):
                    dominated.add(i)
                    break
        experiments = [e for i, e in enumerate(experiments) if i not in dominated]

    if args.json:
        _json_print({"ok": True, "db": str(db_path), "count": len(experiments),
                     "experiments": experiments})
        return 0

    if not experiments:
        print("No experiments found.")
        return 0

    table_rows = []
    for e in experiments:
        cg   = f"{e['cond_gap']:.4f}"   if e["cond_gap"]   is not None else "—"
        rg   = f"{e['ref_gap']:.4f}"    if e["ref_gap"]    is not None else "—"
        sc   = f"{e['score']:.1f}"      if e["score"]      is not None else "—"
        loss = f"{e['final_loss']:.4f}" if e["final_loss"] is not None else "—"
        ns   = str(e["steps"] or "—")
        vrd  = str(e["verdict"] or "—")[:12]
        table_rows.append([str(e["id"]), cg, rg, sc, loss, ns, vrd])

    label = "(pareto)" if args.pareto else f"({len(experiments)} total)"
    print(f"\n  {db_path}  {label}\n")
    print(_col_table(table_rows,
                     ["id", "cond_gap", "ref_gap", "score", "loss", "n_steps", "verdict"]))
    print()
    return 0


# ── compare ─────────────────────────────────────────────────────────────────

def _cmd_compare(args: argparse.Namespace) -> int:
    cfg: dict = {}
    if args.config:
        try:
            cfg = load_config(args.config)
        except FileNotFoundError:
            pass
    cold_root    = Path(cfg.get("storage", {}).get("cold_root", COLD_ROOT))
    cold_weights = cold_root / "weights"

    def _load(date: str) -> Optional[dict]:
        d = cold_weights / f"flywheel-{date}"
        if not d.exists():
            return None
        steps: list[dict] = []
        for sf in sorted(d.iterdir()):
            if sf.suffix != ".safetensors" or sf.stem == "final":
                continue
            j = sf.with_suffix(".json")
            if j.exists():
                try:
                    steps.append(json.loads(j.read_text()))
                except (ValueError, OSError):
                    pass
        return {"name": d.name, "steps": steps}

    a, b = _load(args.campaign_a), _load(args.campaign_b)
    missing = [d for d, v in ((args.campaign_a, a), (args.campaign_b, b)) if v is None]
    if missing:
        msg = f"campaigns not found: {', '.join(missing)}"
        if args.json:
            _json_print({"ok": False, "error": msg})
        else:
            print(f"ERROR: {msg}")
        return 1

    def _ms(steps: list[dict], key: str) -> dict:
        vals = [s[key] for s in steps if key in s and s[key] is not None]
        if not vals:
            return {"count": 0, "min": None, "max": None, "last": None}
        return {"count": len(vals), "min": round(min(vals), 4),
                "max": round(max(vals), 4), "last": round(vals[-1], 4)}

    result: dict[str, Any] = {
        "ok": True,
        "campaign_a": args.campaign_a, "campaign_b": args.campaign_b,
    }
    for data, key in ((a, "a"), (b, "b")):
        result[f"steps_{key}"]    = len(data["steps"])       # type: ignore[index]
        for metric in ("cond_gap", "ref_gap", "clip_i"):
            result[f"{metric}_{key}"] = _ms(data["steps"], metric)  # type: ignore[index]

    if args.json:
        _json_print(result)
        return 0

    def _fmt(s: dict) -> str:
        return "—" if s["count"] == 0 else f"min={s['min']}  max={s['max']}  last={s['last']}"

    w = 34
    print(f"\n  Comparing: {args.campaign_a} vs {args.campaign_b}")
    print(f"  {'metric':<14}  {'flywheel-' + args.campaign_a:<{w}}  {'flywheel-' + args.campaign_b:<{w}}")
    print(f"  {'─'*14}  {'─'*w}  {'─'*w}")
    for m in ("cond_gap", "ref_gap", "clip_i"):
        print(f"  {m:<14}  {_fmt(result[f'{m}_a']):<{w}}  {_fmt(result[f'{m}_b']):<{w}}")
    print(f"  {'steps':<14}  {result['steps_a']:<{w}}  {result['steps_b']:<{w}}")
    print()
    return 0


# ── maintenance ──────────────────────────────────────────────────────────────

def _cmd_maintenance(args: argparse.Namespace) -> int:
    cfg: dict = {}
    if args.config:
        try:
            cfg = load_config(args.config)
        except FileNotFoundError:
            pass
    cold_root    = Path(cfg.get("storage", {}).get("cold_root", COLD_ROOT))
    hot_root     = Path(cfg.get("storage", {}).get("hot_root",  DATA_ROOT))
    keep_vers    = int(cfg.get("storage", {}).get("keep_versions", 3))
    cold_weights = cold_root / "weights"
    cold_precomp = cold_root / "precomputed"
    issues: list[str] = []

    # 1. Broken best/ symlinks
    broken_links: list[str] = []
    best_dir = cold_weights / "best"
    if best_dir.exists():
        for lnk in best_dir.iterdir():
            if lnk.is_symlink() and not lnk.exists():
                broken_links.append(str(lnk))
                issues.append(f"BROKEN symlink: {lnk}")

    # 2. Orphaned .npz files not in manifest
    orphaned_npz: list[str] = []
    if cold_precomp.exists():
        for enc_dir in cold_precomp.iterdir():
            if not enc_dir.is_dir():
                continue
            for ver_dir in enc_dir.iterdir():
                if not ver_dir.is_dir() or ver_dir.name == "current":
                    continue
                mf_p = ver_dir / "manifest.json"
                try:
                    referenced = set(json.loads(mf_p.read_text()).get("files", {}).keys()) if mf_p.exists() else set()
                except (ValueError, OSError):
                    referenced = set()
                for npz in ver_dir.glob("*.npz"):
                    if npz.name not in referenced:
                        orphaned_npz.append(str(npz))
                        issues.append(f"ORPHANED .npz: {npz}")

    # 3. Precompute versions in cold-only (not in hot)
    hot_precomp = hot_root / "precomputed"
    cold_only: list[str] = []
    if cold_precomp.exists() and hot_precomp.exists():
        for enc_dir in cold_precomp.iterdir():
            if not enc_dir.is_dir():
                continue
            for ver_dir in enc_dir.iterdir():
                if ver_dir.is_dir() and not (hot_precomp / enc_dir.name / ver_dir.name).exists():
                    cold_only.append(f"{enc_dir.name}/{ver_dir.name}")

    # 4. Prunable versions beyond keep_versions
    prunable: list[Path] = []
    if cold_precomp.exists():
        for enc_dir in cold_precomp.iterdir():
            if not enc_dir.is_dir():
                continue
            cur_link = enc_dir / "current"
            cur_ver = os.path.basename(os.readlink(str(cur_link))) if cur_link.is_symlink() else None
            versions = sorted(v for v in enc_dir.iterdir() if v.is_dir() and v.name != "current")
            for v in versions[:-keep_vers] if len(versions) > keep_vers else []:
                if v.name != cur_ver:
                    prunable.append(v)

    result: dict[str, Any] = {
        "ok":                 len(issues) == 0,
        "broken_symlinks":    broken_links,
        "orphaned_npz":       orphaned_npz,
        "cold_only_versions": cold_only,
        "prunable_versions":  [str(p) for p in prunable],
        "issues":             issues,
    }

    if args.prune:
        if not args.confirm:
            print("ERROR: --prune requires --confirm", file=sys.stderr)
            return 1
        pruned = []
        for p in prunable:
            try:
                shutil.rmtree(p)
                pruned.append(str(p))
            except OSError as e:
                print(f"  WARNING: could not remove {p}: {e}", file=sys.stderr)
        result["pruned"] = pruned

    if args.json:
        _json_print(result)
        return 0

    print(f"\n  maintenance — cold root: {cold_root}")
    print()
    if not issues and not cold_only and not prunable:
        print("  Everything looks clean.\n")
        return 0

    if issues:
        print(f"  ISSUES ({len(issues)}):")
        for iss in issues:
            print(f"    {iss}")
        print()
    if cold_only:
        print(f"  Cold-only versions (safe to remove from hot if space needed):")
        for v in cold_only:
            print(f"    {v}")
        print()
    if prunable:
        print(f"  Prunable (beyond keep_versions={keep_vers}):")
        for p in prunable:
            print(f"    {p}")
        print(f"  Run with --prune --confirm to delete.\n")
    return 0


# ── subcommand router ────────────────────────────────────────────────────────

_SUBCMDS = {"status", "shards", "weights", "suggest-warmstart",
            "ablation", "compare", "maintenance"}


def _main_subcmd() -> None:
    ap = argparse.ArgumentParser(
        description="Cold storage inspection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", default=None)
    ap.add_argument("--json",   action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _sp(name: str, help: str, with_config: bool = True) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help)
        p.add_argument("--json", action="store_true")
        if with_config:
            p.add_argument("--config", default=None)
        return p

    p_status = _sp("status", "Disk, pool, precompute, weights, DB overview")
    p_status.set_defaults(func=_cmd_status)

    p_shards = _sp("shards", "Shard quality scores")
    p_shards.add_argument("--top",  type=int, default=20)
    p_shards.add_argument("--sort", default="effective_score",
                          choices=["effective_score", "composite_score", "n_scored"])
    p_shards.set_defaults(func=_cmd_shards)

    p_weights = _sp("weights", "Archived weight campaigns")
    p_weights.add_argument("--campaign", default=None)
    p_weights.set_defaults(func=_cmd_weights)

    p_ws = _sp("suggest-warmstart", "Recommend warmstart checkpoint", with_config=False)
    p_ws.add_argument("--config", required=True)
    p_ws.set_defaults(func=_cmd_suggest_warmstart)

    p_abl = _sp("ablation", "Ablation experiment history")
    p_abl.add_argument("--campaign", default=None)
    p_abl.add_argument("--pareto",   action="store_true")
    p_abl.set_defaults(func=_cmd_ablation)

    p_cmp = _sp("compare", "Side-by-side campaign metric comparison")
    p_cmp.add_argument("campaign_a")
    p_cmp.add_argument("campaign_b")
    p_cmp.set_defaults(func=_cmd_compare)

    p_maint = _sp("maintenance", "Validate cold storage, GC orphans")
    p_maint.add_argument("--prune",   action="store_true")
    p_maint.add_argument("--confirm", action="store_true")
    p_maint.set_defaults(func=_cmd_maintenance)

    args = ap.parse_args()
    # Inherit top-level --json/--config when subparser doesn't override
    if not hasattr(args, "json") or args.json is False:
        args.json = ap.parse_known_args()[0].json
    if not getattr(args, "config", None):
        args.config = ap.parse_known_args()[0].config

    sys.exit(args.func(args))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Route to cold-storage subcommand interface when first positional arg is a known subcommand.
    if len(sys.argv) > 1 and sys.argv[1] in _SUBCMDS:
        _main_subcmd()
        return

    ap = argparse.ArgumentParser(
        description="Data intelligence layer for iris flywheel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--overview", action="store_true",
                    help="Disk, shard stats, flywheel summary, precompute versions")
    ap.add_argument("--top-shards", metavar="N", nargs="?", const=30, type=int,
                    help="Show top N shards by effective_score (default 30)")
    ap.add_argument("--source", metavar="SOURCE",
                    help="Filter shards by source (used with --top-shards)")
    ap.add_argument("--weights", action="store_true",
                    help="Show checkpoint listing with quality metrics")
    ap.add_argument("--best", action="store_true",
                    help="Show only best checkpoint (used with --weights)")
    ap.add_argument("--warm-start", action="store_true",
                    help="Emit warm-start YAML snippet for a flywheel config")
    ap.add_argument("--from-flywheel", metavar="NAME",
                    help="Flywheel name to pull warm-start checkpoint from")
    ap.add_argument("--iteration", metavar="N", type=int,
                    help="Specific iteration to use for warm-start (default: best)")
    ap.add_argument("--apply", metavar="CONFIG_PATH",
                    help="Apply warm-start patch to this flywheel YAML config file")
    ap.add_argument("--validate-coverage", action="store_true",
                    help="Check precompute coverage vs. shard pool")
    ap.add_argument("--html", metavar="PATH",
                    help="Write self-contained HTML dashboard to PATH")
    ap.add_argument("--ai", action="store_true",
                    help="JSON output for machine consumption")

    args = ap.parse_args()

    # Default: show overview if no command given
    if not any([args.overview, args.top_shards is not None, args.weights,
                args.warm_start, args.validate_coverage, args.html, args.ai]):
        args.overview = True

    # Collect data
    ov = collect_overview() if (args.overview or args.html or args.ai) else {}
    top_shards = collect_top_shards(args.top_shards or 30, args.source) \
        if (args.top_shards is not None or args.html or args.ai) else []
    weights = collect_weights(best_only=args.best) \
        if (args.weights or args.html or args.ai) else []
    cov = validate_coverage() if (args.validate_coverage or args.html or args.ai) else {}

    ws: Optional[dict] = None
    if args.warm_start:
        name = args.from_flywheel
        if not name:
            # Try first available flywheel
            if _HAS_FLYWHEEL_DB and FLYWHEEL_DB_PATH.exists():
                try:
                    fw = FlywheelDB(FLYWHEEL_DB_PATH)
                    names = fw.get_all_run_names()
                    fw.close()
                    name = names[0] if names else None
                except Exception:
                    pass
        if name:
            ws = generate_warm_start(name, args.iteration)
        else:
            ws = {"error": "no flywheel found; use --from-flywheel NAME"}

    # Output
    if args.ai:
        data = {
            "overview":   ov,
            "top_shards": top_shards,
            "weights":    weights,
            "coverage":   cov,
        }
        if ws:
            data["warm_start"] = ws
        print(json.dumps(data, indent=2, default=str))
        return

    if args.overview:
        print_overview(ov)

    if args.top_shards is not None:
        print_top_shards(top_shards, args.top_shards)

    if args.weights:
        print_weights(weights, args.best)

    if args.warm_start and ws:
        print_warm_start(ws)
        if args.apply and ws.get("checkpoint"):
            msg = _apply_warm_start(Path(args.apply), ws["checkpoint"])
            print(f"\n  Applied: {msg}")

    if args.validate_coverage:
        print_validate(cov)

    if args.html:
        html = render_html(ov, top_shards, weights, cov)
        try:
            Path(args.html).write_text(html)
            print(f"\nHTML dashboard written to {args.html}")
        except Exception as e:
            print(f"\nError writing HTML: {e}", file=sys.stderr)

    if not any([args.overview, args.top_shards is not None, args.weights,
                args.warm_start, args.validate_coverage, args.html]):
        ap.print_help()


if __name__ == "__main__":
    main()
