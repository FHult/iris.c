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
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
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
