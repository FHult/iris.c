#!/usr/bin/env python3
"""
train/scripts/ablation_harness.py — QUALITY-10: Automated style feature ablation harness.

Runs a matrix of short training experiments with different QUALITY hyperparameter
combinations and produces a ranked HTML report identifying the best settings for
style fidelity (measured by cross-ref vs self-ref loss gap and adapter learning).

Usage:
    # Quick 4-combo exploration (default 'small' matrix, 8000 steps each):
    python train/scripts/ablation_harness.py \\
        --shards /Volumes/2TBSSD/shards \\
        --output-dir train/reports/ablation_run1

    # Medium matrix (12 combos), 5000 steps each:
    python train/scripts/ablation_harness.py --matrix medium --steps 5000 \\
        --shards /Volumes/2TBSSD/shards \\
        --output-dir train/reports/ablation_run2

    # Full 54-combo matrix:
    python train/scripts/ablation_harness.py --matrix full --steps 8000 \\
        --shards /Volumes/2TBSSD/shards \\
        --output-dir train/reports/ablation_full

    # Resume an interrupted run:
    python train/scripts/ablation_harness.py \\
        --output-dir train/reports/ablation_run1 --resume

    # Dry run — print the combo matrix without training:
    python train/scripts/ablation_harness.py --matrix medium --dry-run

    # Custom matrix from YAML file:
    python train/scripts/ablation_harness.py \\
        --matrix-file my_matrix.yaml --output-dir train/reports/ablation_custom

Custom matrix YAML format:
    ablation:
      variables:
        cross_ref_prob: [0.0, 0.3, 0.5]
        patch_shuffle_prob: [0.0, 0.3, 0.5]
        freeze_double_stream_scales: [true, false]
        style_loss_weight: [0.0, 0.05, 0.1]
      steps_per_run: 8000          # optional, overridden by --steps

Matrix presets:
    small  (default): 4  combos — cross_ref=[0.3,0.5] × patch=[0.0,0.5]
    medium:           12 combos — cross_ref=[0.0,0.3,0.5] × patch=[0.0,0.5] × freeze=[T,F]
    full:             54 combos — cross_ref=[0.0,0.3,0.5] × patch=[0.0,0.3,0.5] × freeze=[T,F] × slw=[0.0,0.05,0.1]

Scoring:
    Each combo is ranked by a composite score derived entirely from training logs
    (no inference pass required):
      - ref_gap:   mean(loss_cross_ref - loss_self_ref) over the final 40% of steps
                   Positive = style-aware; negative = adapter ignores SigLIP features
      - cond_gap:  mean(loss_null - loss_cond) over the final 40% of steps
                   Positive = adapter is learning; near-zero = collapsed
      - loss_pen:  small penalty for high final loss (instability signal)

    score = 100 × ref_gap + 200 × cond_gap - 3 × final_loss
"""

import argparse
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import yaml

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

# ── Matrix presets ────────────────────────────────────────────────────────────

MATRIX_PRESETS: dict[str, dict] = {
    "small": {
        "variables": {
            "cross_ref_prob":            [0.3, 0.5],
            "patch_shuffle_prob":        [0.0, 0.5],
            "freeze_double_stream_scales": [True],
            "style_loss_weight":         [0.05],
        },
    },
    "medium": {
        "variables": {
            "cross_ref_prob":            [0.0, 0.3, 0.5],
            "patch_shuffle_prob":        [0.0, 0.5],
            "freeze_double_stream_scales": [True, False],
            "style_loss_weight":         [0.05],
        },
    },
    "full": {
        "variables": {
            "cross_ref_prob":            [0.0, 0.3, 0.5],
            "patch_shuffle_prob":        [0.0, 0.3, 0.5],
            "freeze_double_stream_scales": [True, False],
            "style_loss_weight":         [0.0, 0.05, 0.1],
        },
    },
}

# ── Metric log regexes (match train_ip_adapter.py output format) ──────────────
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
        """Parse one output line. Returns a completed snapshot when the ip_scale
        line is seen (which closes a log interval), else None."""
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


# ── Matrix generation ─────────────────────────────────────────────────────────

def _generate_combos(matrix_def: dict) -> list[dict]:
    """Return list of {combo_id, params} dicts from a matrix definition."""
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
    """Resolve the ablation matrix from --matrix-file or --matrix preset."""
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
    """Load base config and apply ablation overrides for one combo."""
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
    cfg["training"]["num_steps"]    = steps
    cfg["training"]["warmup_steps"] = min(cfg["training"].get("warmup_steps", 1000), steps // 5)
    cfg["training"]["style_loss_every"] = 1

    cfg.setdefault("adapter", {})

    # Apply per-combo parameters
    for key, val in params.items():
        if key == "freeze_double_stream_scales":
            cfg["adapter"]["freeze_double_stream_scales"] = val
        elif key in ("cross_ref_prob", "patch_shuffle_prob",
                     "style_loss_weight", "learning_rate"):
            cfg["training"][key] = val
        else:
            # Best-effort: try training section first, then adapter
            cfg["training"][key] = val

    cfg.setdefault("output", {})
    cfg["output"]["checkpoint_dir"]      = checkpoint_dir
    cfg["output"]["log_every"]           = log_every
    cfg["output"]["checkpoint_every"]    = steps * 100  # prevent periodic saves
    cfg["output"]["keep_last_n"]         = 1
    cfg["output"]["skip_checkpoint_save"] = True  # skip final ~8 GB write; not needed for ranking

    cfg.setdefault("eval", {})
    cfg["eval"]["enabled"] = False

    return cfg


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score(snapshots: list[dict], exit_code: int) -> float:
    """Composite score for ranking: higher is better style conditioning.

    Primary signal: ref_gap (cross_ref - self_ref) — positive = style-aware.
    Secondary: cond_gap (loss_null - loss_cond) — positive = adapter learning.
    Penalty: high final loss (instability).
    Returns -inf for crashed runs.
    """
    if exit_code != 0 or not snapshots:
        return float("-inf")

    # Skip warmup (first 40% of log intervals) — signal is noisy there
    n_skip = max(0, len(snapshots) * 2 // 5)
    tail = snapshots[n_skip:] or snapshots

    ref_gaps   = [s["ref_gap"]   for s in tail if "ref_gap"   in s]
    cond_gaps  = [s["cond_gap"]  for s in tail if "cond_gap"  in s]
    loss_vals  = [s["loss_smooth"] for s in tail if "loss_smooth" in s]

    mean_ref  = sum(ref_gaps)  / len(ref_gaps)  if ref_gaps  else 0.0
    mean_cond = sum(cond_gaps) / len(cond_gaps) if cond_gaps else 0.0
    final_loss = loss_vals[-1] if loss_vals else 9.9

    return 100.0 * mean_ref + 200.0 * mean_cond - 3.0 * final_loss


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


# ── Single-run execution ──────────────────────────────────────────────────────

def _run_one(
    combo: dict,
    run_dir: Path,
    args,
    log_every: int,
    quiet: bool = False,
) -> dict:
    """Execute one training combo. Returns result dict."""
    combo_id = combo["combo_id"]
    params   = combo["params"]

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    log_path = run_dir / "training.log"

    cfg = _build_run_config(
        base_config_path=Path(args.base_config),
        shards=args.shards,
        qwen3_cache=args.qwen3_cache,
        vae_cache=args.vae_cache,
        siglip_cache=args.siglip_cache,
        checkpoint_dir=str(ckpt_dir),
        steps=args.steps,
        log_every=log_every,
        params=params,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False,
                                     prefix=f"ablation_{combo_id}_") as tf:
        yaml.dump(cfg, tf)
        tmp_cfg = tf.name

    # Save the config used for reproducibility
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
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(_REPO_ROOT),
        )
        with open(log_path, "w") as log_f:
            for raw_line in proc.stdout:  # type: ignore[union-attr]
                line = raw_line.rstrip()
                log_f.write(raw_line)
                log_f.flush()
                if not quiet:
                    print(f"  {line}", flush=True)
                snap = collector.feed(line)
                if snap is not None and not quiet:
                    ref_gap  = snap.get("ref_gap", 0.0)
                    cond_gap = snap.get("cond_gap", 0.0)
                    loss     = snap.get("loss_smooth", snap.get("loss", 0.0))
                    flag = "✓" if ref_gap > 0 else "○"
                    print(_c("dim", f"  ↳ step {snap['step']:>6}  "
                              f"loss={loss:.4f}  "
                              f"ref_gap={ref_gap:+.4f}{flag}  "
                              f"cond_gap={cond_gap:+.4f}"), flush=True)
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
    score = _score(collector.snapshots, exit_code)
    verdict = _verdict(collector.snapshots, exit_code)

    # Summarise tail metrics for the result record
    tail = collector.snapshots[max(0, len(collector.snapshots) * 2 // 5):] or collector.snapshots
    last = collector.snapshots[-1] if collector.snapshots else {}

    result = {
        "combo_id":    combo_id,
        "params":      params,
        "score":       round(score, 4) if score != float("-inf") else None,
        "verdict":     verdict,
        "exit_code":   exit_code,
        "elapsed_secs": round(elapsed),
        "n_snapshots": len(collector.snapshots),
        "final_loss":  last.get("loss_smooth"),
        "final_ref_gap":  last.get("ref_gap"),
        "final_cond_gap": last.get("cond_gap"),
        "mean_ref_gap":   round(sum(s["ref_gap"] for s in tail if "ref_gap" in s) /
                                max(1, sum(1 for s in tail if "ref_gap" in s)), 4)
                          if any("ref_gap" in s for s in tail) else None,
        "mean_cond_gap":  round(sum(s["cond_gap"] for s in tail if "cond_gap" in s) /
                                max(1, sum(1 for s in tail if "cond_gap" in s)), 4)
                          if any("cond_gap" in s for s in tail) else None,
        "snapshots":   collector.snapshots,
        "log_path":    str(log_path),
    }

    # Save per-run metrics JSON
    try:
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)
    except OSError:
        pass

    # Clean up checkpoints unless requested
    if not args.keep_checkpoints and ckpt_dir.exists():
        try:
            shutil.rmtree(ckpt_dir)
        except OSError:
            pass

    return result


# ── HTML report ───────────────────────────────────────────────────────────────

_PALETTE = [
    "#7af", "#7f7", "#f77", "#fa7", "#c7f", "#7fc", "#ff7",
    "#f7c", "#7cf", "#fc7", "#a7f", "#7fa",
]


def _html_color(i: int, n: int, rank: int) -> str:
    """Color for combo i (by rank). Top 3 get saturated, rest greyed."""
    if rank == 1:
        return "#7f7"
    if rank == 2:
        return "#7af"
    if rank == 3:
        return "#fa7"
    if i < len(_PALETTE):
        return _PALETTE[i]
    hue = (i * 137) % 360
    return f"hsl({hue},55%,60%)"


def _render_html(
    results: list[dict],
    matrix_name: str,
    steps: int,
    ts: str,
    total_elapsed: int,
    run_dir_name: str,
) -> str:
    ranked = sorted(
        [r for r in results if r.get("score") is not None],
        key=lambda r: r["score"],
        reverse=True,
    )
    crashed = [r for r in results if r.get("score") is None]
    all_sorted = ranked + crashed

    # Build per-row score (None → "—")
    def _fmt(v, fmt=".4f"):
        return f"{v:{fmt}}" if v is not None else "—"

    def _verdict_style(v):
        return {"PASS": "color:#7f7", "WARN": "color:#fa7",
                "CRASH": "color:#f77", "UNSTABLE": "color:#f77",
                "NO_DATA": "color:#888"}.get(v, "")

    # Table rows
    rows_html = ""
    for rank_i, r in enumerate(all_sorted):
        rank_disp = rank_i + 1 if r.get("score") is not None else "—"
        col = _html_color(rank_i, len(all_sorted), rank_disp if isinstance(rank_disp, int) else 99)
        p = r["params"]
        score_str = _fmt(r.get("score"), ".2f")
        ref_str   = _fmt(r.get("mean_ref_gap"))
        cond_str  = _fmt(r.get("mean_cond_gap"))
        loss_str  = _fmt(r.get("final_loss"))
        elapsed   = r.get("elapsed_secs", 0)
        elapsed_str = f"{elapsed // 60}m{elapsed % 60:02d}s" if elapsed else "—"
        params_str = "  ".join(
            f"{k.replace('freeze_double_stream_scales','freeze').replace('_prob','').replace('_weight','_w')}="
            f"{'T' if v is True else 'F' if v is False else v}"
            for k, v in p.items()
        )
        vstyle = _verdict_style(r.get("verdict", ""))
        rows_html += (
            f"<tr>"
            f"<td style='color:{col};font-weight:bold'>{rank_disp}</td>"
            f"<td style='color:{col}'>{r['combo_id']}</td>"
            f"<td style='font-size:0.8em;color:#ccc'>{params_str}</td>"
            f"<td style='color:{col};font-weight:bold'>{score_str}</td>"
            f"<td>{ref_str}</td><td>{cond_str}</td><td>{loss_str}</td>"
            f"<td>{elapsed_str}</td>"
            f"<td style='{vstyle}'>{r.get('verdict','?')}</td>"
            f"</tr>\n"
        )

    # Best config box
    best = ranked[0] if ranked else None
    best_html = ""
    if best:
        p = best["params"]
        lines = [f"<b style='color:#7f7'>#{1}: {best['combo_id']}</b>  "
                 f"score={best['score']:.2f}  "
                 f"ref_gap={best.get('mean_ref_gap') or 0:.4f}  "
                 f"cond_gap={best.get('mean_cond_gap') or 0:.4f}"]
        lines.append("")
        for k, v in p.items():
            label = k
            lines.append(f"  <b>{label}</b>: {v}")
        lines.append("")
        lines.append("<b>Command to use this config:</b>")
        overrides = []
        for k, v in p.items():
            if k == "freeze_double_stream_scales":
                overrides.append("" if v else "--no-freeze-double")
            elif k == "cross_ref_prob":
                overrides.append(f"--cross-ref-prob {v}")
            elif k == "patch_shuffle_prob":
                overrides.append(f"--patch-shuffle-prob {v}")
            elif k == "style_loss_weight":
                overrides.append(f"--style-loss-weight {v}")
        cmd = (f"python train/scripts/test_quality_features.py "
               f"{' '.join(o for o in overrides if o)} --steps {steps * 2}")
        lines.append(f"  <code style='color:#7af'>{cmd}</code>")
        best_html = "<br>".join(lines)

    # JS data for charts: per-combo series
    js_series = []
    for rank_i, r in enumerate(all_sorted[:20]):  # cap at 20 for readability
        col = _html_color(rank_i, len(all_sorted), rank_i + 1 if r.get("score") is not None else 99)
        snaps = r.get("snapshots", [])
        label = r["combo_id"]
        js_series.append({
            "label":    label,
            "color":    col,
            "rank":     rank_i + 1,
            "ref_gap":  [[s["step"], s.get("ref_gap")]  for s in snaps],
            "cond_gap": [[s["step"], s.get("cond_gap")] for s in snaps],
            "loss":     [[s["step"], s.get("loss_smooth")] for s in snaps],
            "scale":    [[s["step"], s.get("ip_scale_mean")] for s in snaps],
        })

    # Score bar data (for all ranked combos)
    bar_data = [
        {"label": r["combo_id"], "score": r.get("score") or 0,
         "color": _html_color(i, len(ranked), i + 1)}
        for i, r in enumerate(ranked)
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
        js_series=json.dumps(js_series),
        bar_data=json.dumps(bar_data),
        run_dir_name=run_dir_name,
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
  .meta {{color:#777;font-size:0.82em;margin-bottom:14px}}
  .best-box {{background:#0d1a0d;border:1px solid #2a5;border-radius:6px;
              padding:12px 16px;margin-bottom:16px;font-size:0.88em;max-width:700px}}
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
  matrix={matrix_name} &nbsp;|&nbsp; steps/run={steps} &nbsp;|&nbsp;
  {n_combos} combos ({n_ranked} scored, {n_crashed} crashed) &nbsp;|&nbsp;
  total wall-clock: {total_elapsed} &nbsp;|&nbsp; {ts}
</div>

<h2>Recommended Config</h2>
<div class="best-box">{best_html}</div>

<h2>Ranked Results</h2>
<table>
  <tr>
    <th>Rank</th><th>Combo</th><th>Parameters</th>
    <th>Score ↓</th><th>ref_gap</th><th>cond_gap</th><th>final_loss</th>
    <th>Elapsed</th><th>Verdict</th>
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
  <div><div class="chart-label">IP scale (mean) — adapter weight magnitude</div>
       <canvas id="scaleChart" width="480" height="220"></canvas></div>
  <div><div class="chart-label">Score ranking</div>
       <canvas id="barChart" width="480" height="220"></canvas></div>
</div>

<h2>Data</h2>
<p><a href="results.json" style="color:#7af">results.json</a> — full metric data for further analysis</p>
<p>Per-run logs and configs: <code>runs/{run_dir_name}/combo_NNN/</code></p>

<script>
const SERIES = {js_series};
const BAR    = {bar_data};

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
    ctx.strokeStyle='#444'; ctx.lineWidth=1.5;
    ctx.setLineDash([4,4]);
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
  // Legend (top 5 + "others")
  let lx=pad.l; let ly=pad.t+10;
  const show = SERIES.slice(0,5);
  for(const s of show) {{
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
  const W=cv.width, H=cv.height;
  const pad={{t:16,r:16,b:16,l:72}};
  const cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  if(!BAR.length) return;
  const maxScore=Math.max(...BAR.map(b=>b.score));
  const minScore=Math.min(0,...BAR.map(b=>b.score));
  const barH=Math.floor((ch-4*BAR.length)/Math.max(BAR.length,1));
  BAR.forEach((b,i) => {{
    const y=pad.t+i*(barH+4);
    const w=Math.max(2,(b.score-minScore)/(maxScore-minScore+1e-9)*cw);
    ctx.fillStyle=b.color;
    ctx.fillRect(pad.l,y,w,Math.max(barH,2));
    ctx.fillStyle='#888'; ctx.font='8px monospace';
    ctx.fillText(b.label,2,y+barH/2+3);
    ctx.fillStyle='#ccc'; ctx.font='8px monospace';
    ctx.fillText(b.score.toFixed(2),pad.l+w+4,y+barH/2+3);
  }});
}}

drawChart('refChart',  'ref_gap',  true);
drawChart('condChart', 'cond_gap', true);
drawChart('lossChart', 'loss',     false);
drawChart('scaleChart','scale',    false);
drawBar('barChart');
</script>
</body>
</html>
"""


# ── Human-readable progress ───────────────────────────────────────────────────

def _print_combo_header(combo: dict, idx: int, total: int, steps: int) -> None:
    p = combo["params"]
    cid = combo["combo_id"]
    params_str = "  ".join(f"{k}={v}" for k, v in p.items())
    print()
    print(_c("cyan", f"{'─'*64}"))
    print(_c("bold", f"  [{idx}/{total}] {cid}  ({steps} steps)"))
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
    vcol = {"PASS": "green", "WARN": "yellow", "CRASH": "red", "UNSTABLE": "red",
             "NO_DATA": "dim"}.get(v, "reset")
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
        p_str = "  ".join(
            f"{k.replace('freeze_double_stream_scales','freeze').replace('_prob','').replace('_weight','_w')}="
            f"{'T' if v is True else 'F' if v is False else v}"
            for k, v in r["params"].items()
        )
        ref  = r.get("mean_ref_gap")
        cond = r.get("mean_cond_gap")
        score = r.get("score", 0)
        print(f"  {_c(col, f'#{rank}')}  {r['combo_id']}  score={score:.2f}  "
              f"ref_gap={ref:+.4f}  cond_gap={cond:+.4f}"
              if ref is not None and cond is not None else
              f"  {_c(col, f'#{rank}')}  {r['combo_id']}  score={score:.2f}")
        print(f"     {p_str}")
    if ranked:
        best = ranked[0]
        print()
        print(_c("green", "  ★ Best config:"))
        for k, v in best["params"].items():
            print(f"    {k}: {v}")
    print(_c("cyan", "═" * 64))


# ── Results persistence ────────────────────────────────────────────────────────

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
    # Write metrics JSON without snapshots (too large for results.json)
    slim = [{k: v for k, v in r.items() if k != "snapshots"} for r in results]
    try:
        with open(output_dir / "results.json", "w") as f:
            json.dump(slim, f, indent=2)
    except OSError as e:
        print(f"WARNING: could not save results.json: {e}", file=sys.stderr)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="QUALITY-10: Automated style feature ablation harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Custom matrix YAML format:")[0].strip(),
    )
    ap.add_argument("--matrix", default="small", metavar="PRESET",
                    help=f"Built-in matrix preset: {list(MATRIX_PRESETS)} (default: small)")
    ap.add_argument("--matrix-file", default=None, metavar="PATH",
                    help="Custom matrix YAML file (overrides --matrix)")
    ap.add_argument("--steps", type=int, default=8000,
                    help="Training steps per combo (default: 8000)")
    ap.add_argument("--log-every", type=int, default=None,
                    help="Log interval in steps (default: auto — steps//80, min 50)")
    ap.add_argument("--output-dir", default="train/reports/ablation_run",
                    help="Output directory for report and results (default: %(default)s)")
    ap.add_argument("--base-config", default=str(_BASE_CONFIG),
                    help="Base training config YAML (default: stage1_512px.yaml)")
    ap.add_argument("--shards", default=str(_DEFAULT_SHARDS),
                    help="Shard directory (default: %(default)s)")
    ap.add_argument("--qwen3-cache", default=str(_DEFAULT_QWEN3) if _DEFAULT_QWEN3.exists() else None,
                    help="Precomputed Qwen3 cache dir")
    ap.add_argument("--vae-cache",   default=str(_DEFAULT_VAE)   if _DEFAULT_VAE.exists()   else None,
                    help="Precomputed VAE cache dir")
    ap.add_argument("--siglip-cache",default=str(_DEFAULT_SIGLIP) if _DEFAULT_SIGLIP.exists() else None,
                    help="Precomputed SigLIP cache dir (required for ref_gap signal)")
    ap.add_argument("--max-runs", type=int, default=None,
                    help="Run at most N combos then stop (useful for testing the harness)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip combos whose combo_id already appears in output-dir/results.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the combo matrix without training")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-step training output (show only progress summaries)")
    ap.add_argument("--keep-checkpoints", action="store_true",
                    help="Keep checkpoint files after each run (disk-intensive for many combos)")
    ap.add_argument("--ai", action="store_true",
                    help="Emit compact JSON summary to stdout when done")
    args = ap.parse_args()

    # ── Resolve matrix and combos ─────────────────────────────────────────────
    matrix_def  = _load_matrix(args)
    all_combos  = _generate_combos(matrix_def)
    matrix_name = args.matrix or "custom"

    if not all_combos:
        print("ERROR: empty matrix — no variables defined", file=sys.stderr)
        sys.exit(1)

    # Auto log_every: aim for ~80 log lines per run, but not less than 50
    log_every = args.log_every or max(50, args.steps // 80)

    # ── Output directory setup ────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs" / matrix_name
    runs_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y-%m-%dT%H:%M:%S")

    # ── Pre-flight checks ─────────────────────────────────────────────────────
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
            print(_c("yellow", "WARNING: no siglip-cache — ref_gap signal will be absent. "
                     "Run precompute_all.py --siglip first for meaningful ablation."))

    # ── Dry run: just print the matrix ───────────────────────────────────────
    if args.dry_run:
        print(_c("bold", f"\nAblation matrix — {matrix_name}"))
        print(f"  {len(all_combos)} combos × {args.steps} steps = "
              f"~{len(all_combos) * args.steps // 1000}k total steps")
        print(f"  log_every={log_every}  output-dir={output_dir}\n")
        for c in all_combos:
            params_str = "  ".join(f"{k}={v}" for k, v in c["params"].items())
            print(f"  {c['combo_id']}:  {params_str}")
        print()
        return

    # ── Resume: load existing results ─────────────────────────────────────────
    done_ids: set[str] = set()
    all_results: list[dict] = []
    if args.resume:
        all_results = _load_results(output_dir)
        done_ids = {r["combo_id"] for r in all_results}
        if done_ids:
            print(f"  Resuming: {len(done_ids)} combos already done, "
                  f"{len(all_combos) - len(done_ids)} remaining")

    pending = [c for c in all_combos if c["combo_id"] not in done_ids]
    if args.max_runs is not None:
        pending = pending[:args.max_runs]

    # ── Banner ────────────────────────────────────────────────────────────────
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
    if not args.vae_cache:
        print(_c("yellow", "  WARNING: no vae-cache — live VAE encoding (slow)"))
    print(_c("cyan", f"{'═'*64}\n"))

    # ── Run loop ──────────────────────────────────────────────────────────────
    run_start = time.time()
    for idx, combo in enumerate(pending, start=len(done_ids) + 1):
        _print_combo_header(combo, idx, len(all_combos), args.steps)
        run_dir = runs_dir / combo["combo_id"]
        result = _run_one(combo, run_dir, args, log_every, quiet=args.quiet)
        all_results.append(result)
        _save_results(output_dir, all_results)
        _print_result_line(result)

        if result.get("exit_code") == -2:  # KeyboardInterrupt
            print(_c("yellow", "\n  Run interrupted — generating report with partial results"))
            break

    total_elapsed = int(time.time() - run_start)

    # ── Final ranking ─────────────────────────────────────────────────────────
    _print_final_ranking(all_results)

    # ── HTML report ───────────────────────────────────────────────────────────
    # Include snapshots from per-run metrics.json files for charts
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

    # ── AI JSON output ────────────────────────────────────────────────────────
    if args.ai:
        ranked = sorted(
            [r for r in all_results if r.get("score") is not None],
            key=lambda r: r["score"],
            reverse=True,
        )
        ai_out = {
            "matrix":       matrix_name,
            "n_combos":     len(all_results),
            "n_scored":     len(ranked),
            "n_crashed":    len(all_results) - len(ranked),
            "total_elapsed_secs": total_elapsed,
            "best": ranked[0] if ranked else None,
            "top5": [
                {"combo_id": r["combo_id"], "params": r["params"],
                 "score": r.get("score"), "mean_ref_gap": r.get("mean_ref_gap"),
                 "mean_cond_gap": r.get("mean_cond_gap")}
                for r in ranked[:5]
            ],
            "report": str(report_path),
        }
        print(json.dumps(ai_out, indent=2))

    has_any = any(r.get("score") is not None for r in all_results)
    sys.exit(0 if has_any else 1)


if __name__ == "__main__":
    main()
