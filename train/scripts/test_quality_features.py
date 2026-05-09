#!/usr/bin/env python3
"""
test_quality_features.py — Safety smoke-test for QUALITY-1/2/3/6/8/9 style-separation
features before committing to a full training chunk.

Runs a short training session (default 500 steps) with conservative or aggressive
settings and monitors:
  - loss stability (no explosion or collapse)
  - cross-ref vs self-ref loss gap (style/content separation working)
  - double-stream scales frozen at zero (QUALITY-2)
  - adapter learning (loss_cond < loss_null)
  - grad norm stability

Produces a PASS / WARN / FAIL verdict, an HTML report with inline charts, and
an --ai JSON summary.

Usage:
    # Conservative mode (cross_ref_prob=0.3, patch_shuffle_prob=0.3 — safe defaults)
    python train/scripts/test_quality_features.py \\
        --shards /Volumes/2TBSSD/shards \\
        --output-dir /tmp/quality_test

    # Production settings (cross_ref_prob=0.5, patch_shuffle_prob=0.5)
    python train/scripts/test_quality_features.py --aggressive \\
        --shards /Volumes/2TBSSD/shards \\
        --output-dir /tmp/quality_test

    # Just the AI verdict (JSON to stdout):
    python train/scripts/test_quality_features.py --ai \\
        --shards /Volumes/2TBSSD/shards

Prerequisites:
    - At least 1 shard .tar file in --shards dir
    - Precomputed Qwen3/VAE/SigLIP caches are optional but strongly recommended;
      without them the test runs live encoding which is 5-10x slower
    - No other training or GPU process running
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import yaml  # available in train/.venv

# ── Repo layout ───────────────────────────────────────────────────────────────
_SCRIPT_DIR  = Path(__file__).resolve().parent
_TRAIN_DIR   = _SCRIPT_DIR.parent
_VENV_PYTHON = _TRAIN_DIR / ".venv" / "bin" / "python"
_BASE_CONFIG = _TRAIN_DIR / "configs" / "stage1_512px.yaml"
_TRAINER     = _TRAIN_DIR / "train_ip_adapter.py"

# ── Production default paths ──────────────────────────────────────────────────
_DATA_ROOT   = Path(os.environ.get("PIPELINE_DATA_ROOT", "/Volumes/2TBSSD"))
_DEFAULT_SHARDS     = _DATA_ROOT / "shards"
_DEFAULT_QWEN3      = _DATA_ROOT / "precomputed" / "qwen3"
_DEFAULT_VAE        = _DATA_ROOT / "precomputed" / "vae"
_DEFAULT_SIGLIP     = _DATA_ROOT / "precomputed" / "siglip"

# ── Log-line regexes (match train_ip_adapter.py output format) ────────────────
_RE_STEP    = re.compile(
    r"^step\s+(\d+)[,/]\d+\s+loss\s+([\d.]+)\s+\(avg\s+([\d.]+)\)")
_RE_COND    = re.compile(
    r"loss_cond=([\d.]+)\s+loss_null=([\d.]+)\s+gap=([+-][\d.]+)")
_RE_REF     = re.compile(
    r"loss_ref:.*?self=([\d.]+).*?cross=([\d.]+).*?gap=([+-][\d.]+)")
_RE_GRAD    = re.compile(
    r"grad_norm\s+([\d.]+)\s+\(smooth\s+([\d.]+)\)")
_RE_SCALE   = re.compile(
    r"ip_scale:\s+mean=([\d.]+).*?double=([\d.]+).*?single=([\d.]+)")
_RE_WARNING = re.compile(r"\bWARNING\b")

# ── Console colours (suppress when not a tty) ─────────────────────────────────
_IS_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _IS_TTY:
        return text
    codes = {"green": "32", "yellow": "33", "red": "31", "cyan": "36",
             "bold": "1", "reset": "0", "dim": "2"}
    return f"\033[{codes.get(code, '0')}m{text}\033[0m"


def _banner(msg: str, width: int = 64) -> None:
    print(_c("cyan", "═" * width))
    print(_c("bold", f"  {msg}"))
    print(_c("cyan", "═" * width))


# ── Config builder ────────────────────────────────────────────────────────────

def build_test_config(
    base_config_path: Path,
    shards: str,
    qwen3_cache: Optional[str],
    vae_cache: Optional[str],
    siglip_cache: Optional[str],
    checkpoint_dir: str,
    steps: int,
    log_every: int,
    cross_ref_prob: float,
    patch_shuffle_prob: float,
    style_loss_weight: float,
    freeze_double: bool,
) -> dict:
    """Load base config and apply quality-test overrides."""
    with open(base_config_path) as f:
        cfg = yaml.safe_load(f)

    # ── Data paths ────────────────────────────────────────────────────────────
    cfg.setdefault("data", {})
    cfg["data"]["shard_path"]      = shards
    cfg["data"]["qwen3_cache_dir"] = qwen3_cache
    cfg["data"]["vae_cache_dir"]   = vae_cache
    cfg["data"]["siglip_cache_dir"] = siglip_cache
    cfg["data"]["anchor_shard_dir"] = None   # no anchor shards for quality test
    cfg["data"]["hard_example_dir"] = None   # no hard examples
    cfg["data"]["prefetch_batches"]  = 4     # small prefetch for short run
    cfg["data"]["num_prefetch_threads"] = 1

    # ── Training overrides ────────────────────────────────────────────────────
    cfg.setdefault("training", {})
    cfg["training"]["num_steps"]          = steps
    cfg["training"]["cross_ref_prob"]     = cross_ref_prob
    cfg["training"]["patch_shuffle_prob"] = patch_shuffle_prob
    cfg["training"]["style_loss_weight"]  = style_loss_weight
    cfg["training"]["style_loss_every"]   = 1
    # Keep dropout at production defaults so we exercise all code paths.
    cfg["training"].setdefault("image_dropout_prob", 0.30)
    cfg["training"].setdefault("text_dropout_prob", 0.10)
    # Shorter warmup so adapter learning shows within the test window.
    cfg["training"]["warmup_steps"] = min(cfg["training"].get("warmup_steps", 1000), steps // 5)

    # ── Adapter overrides ─────────────────────────────────────────────────────
    cfg.setdefault("adapter", {})
    cfg["adapter"]["freeze_double_stream_scales"] = freeze_double

    # ── Output ────────────────────────────────────────────────────────────────
    cfg.setdefault("output", {})
    cfg["output"]["checkpoint_dir"]  = checkpoint_dir
    cfg["output"]["log_every"]       = log_every
    cfg["output"]["checkpoint_every"] = max(steps, 500)  # one checkpoint at the end
    cfg["output"]["keep_last_n"]     = 1

    # ── Eval: disable for speed ───────────────────────────────────────────────
    cfg.setdefault("eval", {})
    cfg["eval"]["enabled"] = False

    return cfg


# ── Log line parser ───────────────────────────────────────────────────────────

class MetricCollector:
    """Parses train_ip_adapter.py log lines and accumulates metric snapshots."""

    def __init__(self) -> None:
        self.snapshots: list[dict] = []  # one per log interval
        self._pending: dict = {}         # fields being accumulated for current interval

    def feed(self, line: str) -> Optional[dict]:
        """Parse one output line. Returns a completed snapshot dict when a log
        interval is closed (i.e. after the ip_scale line is seen), else None."""
        m = _RE_STEP.search(line)
        if m:
            self._pending = {
                "step": int(m.group(1)),
                "loss": float(m.group(2)),
                "loss_smooth": float(m.group(3)),
            }
            return None

        if not self._pending:
            return None

        m = _RE_COND.search(line)
        if m:
            self._pending["loss_cond"]    = float(m.group(1))
            self._pending["loss_null"]    = float(m.group(2))
            self._pending["cond_gap"]     = float(m.group(3))
            return None

        m = _RE_REF.search(line)
        if m:
            self._pending["loss_self_ref"]  = float(m.group(1))
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


# ── Inline progress display ───────────────────────────────────────────────────

def _format_snapshot(snap: dict, freeze_double: bool) -> str:
    step = snap.get("step", "?")
    loss = snap.get("loss_smooth", snap.get("loss"))
    loss_str = f"loss={loss:.4f}" if loss is not None else "loss=?"

    cond_str = ""
    lc = snap.get("loss_cond")
    ln = snap.get("loss_null")
    if lc is not None and ln is not None and ln > 0:
        gap_pct = 100 * (ln - lc) / ln
        ok = gap_pct >= 1.0
        sym = _c("green", "✓") if ok else _c("yellow", "!")
        cond_str = f"  cond_gap={gap_pct:+.1f}%{sym}"

    ref_str = ""
    sr = snap.get("loss_self_ref")
    cr = snap.get("loss_cross_ref")
    if sr is not None and cr is not None:
        ref_gap = cr - sr
        ok = ref_gap >= -0.01
        sym = _c("green", "✓") if ok else _c("red", "✗")
        cross_high = cr > sr * 2.0 and cr > 0.1
        if cross_high:
            sym = _c("yellow", "△")
        ref_str = f"  self={sr:.4f} cross={cr:.4f} gap={ref_gap:+.4f}{sym}"

    scale_str = ""
    sm = snap.get("ip_scale_mean")
    sd = snap.get("ip_scale_double")
    if sm is not None:
        frozen_ok = freeze_double and sd is not None and sd < 0.001
        frozen_sym = _c("green", "✓") if frozen_ok else ""
        scale_str = f"  scale={sm:.4f}{frozen_sym}"

    return f"[step {step:>5}]  {loss_str}{cond_str}{ref_str}{scale_str}"


# ── Assessment logic ──────────────────────────────────────────────────────────

def _assess(
    snapshots: list[dict],
    exit_code: int,
    steps: int,
    aggressive: bool,
    freeze_double: bool,
    cross_ref_prob: float,
    patch_shuffle_prob: float,
    siglip_available: bool,
) -> dict:
    """Return assessment dict: verdict, issues[], warnings[], recommendations[]."""
    issues: list[str] = []
    warnings: list[str] = []
    recommendations: list[str] = []

    # 1. Process exit
    if exit_code != 0:
        issues.append(f"Training process exited with code {exit_code} (crash or OOM)")
        return {
            "verdict": "FAIL",
            "issues": issues,
            "warnings": warnings,
            "recommendations": ["Check stderr/log for exception traceback",
                                 "If OOM: reduce prefetch_batches or batch_size"],
        }

    if not snapshots:
        issues.append("No metric snapshots collected — training may have failed silently")
        return {"verdict": "FAIL", "issues": issues, "warnings": warnings,
                "recommendations": ["Run without --ai to see raw training output"]}

    # Metrics after warm-up only (skip first 20% of log intervals)
    warmup_n = max(1, len(snapshots) // 5)
    post_warmup = snapshots[warmup_n:]

    # 2. Loss stability
    loss_vals = [s["loss_smooth"] for s in post_warmup if "loss_smooth" in s]
    if loss_vals:
        max_loss = max(loss_vals)
        final_loss = loss_vals[-1]
        if max_loss > 5.0:
            issues.append(f"Loss exploded to {max_loss:.3f} — style_loss_weight may be too high")
            recommendations.append("Lower style_loss_weight from 0.05 to 0.01 and retry")
        elif final_loss > 3.0:
            warnings.append(f"Loss is high at end of run ({final_loss:.3f}); "
                            "may improve with more steps but check for instability")
        # Check monotone divergence
        if len(loss_vals) >= 4:
            trend_bad = all(loss_vals[i] < loss_vals[i+1] for i in range(len(loss_vals)-3, len(loss_vals)-1))
            if trend_bad and loss_vals[-1] > loss_vals[0] * 1.5:
                warnings.append("Loss is trending upward at end of run — possible instability")
                recommendations.append("Run for more steps to confirm; if trend continues, "
                                       "reduce style_loss_weight or cross_ref_prob")

    # 3. Cross-ref vs self-ref
    ref_snaps = [s for s in post_warmup if "loss_self_ref" in s and "loss_cross_ref" in s]
    if not ref_snaps:
        if not siglip_available:
            warnings.append("SigLIP cache not found — cross-ref permutation not exercised "
                            "(adapter saw null conditioning)")
            recommendations.append("Run precompute_all.py --siglip first to populate SigLIP cache, "
                                   "then re-run this test")
        else:
            warnings.append("No cross-ref/self-ref data collected — "
                            "check that cross_ref_prob > 0 and siglip_cache is populated")
    else:
        n_inverted = sum(1 for s in ref_snaps
                         if s["loss_cross_ref"] < s["loss_self_ref"] - 0.01)
        n_too_high  = sum(1 for s in ref_snaps
                          if s["loss_cross_ref"] > s["loss_self_ref"] * 2.2 and
                          s["loss_cross_ref"] > 0.1)
        inv_frac   = n_inverted / len(ref_snaps)
        high_frac  = n_too_high  / len(ref_snaps)

        if inv_frac > 0.5:
            warnings.append(
                f"Cross-ref loss was below self-ref in {inv_frac:.0%} of log intervals — "
                "adapter may be ignoring SigLIP features (treating them as noise)")
            recommendations.append(
                "Verify SigLIP cache matches the shards; re-run precompute if unsure")
        elif n_inverted > 0:
            warnings.append(
                f"Cross-ref loss dipped below self-ref in {n_inverted}/{len(ref_snaps)} "
                "intervals — monitor in the full run")

        if high_frac > 0.5:
            warnings.append(
                f"Cross-ref loss was >2× self-ref in {high_frac:.0%} of log intervals — "
                "cross-ref task may be too hard at these probabilities")
            if aggressive:
                recommendations.append(
                    "Consider switching from --aggressive to conservative mode "
                    "(cross_ref_prob=0.3, patch_shuffle_prob=0.3)")
            else:
                recommendations.append(
                    "Consider lowering cross_ref_prob to 0.2 for the first chunk")

        last_self = ref_snaps[-1]["loss_self_ref"]
        last_cross = ref_snaps[-1]["loss_cross_ref"]
        last_gap = last_cross - last_self
        if last_gap > 0 and last_gap < last_self * 0.5:
            # gap is there but not too extreme — desired state
            pass

    # 4. Adapter learning (cond < null after warmup)
    cond_snaps = [s for s in post_warmup if "loss_cond" in s and "loss_null" in s]
    if cond_snaps:
        n_not_learning = sum(1 for s in cond_snaps
                             if s["loss_null"] > 0 and
                             100 * (s["loss_null"] - s["loss_cond"]) / s["loss_null"] < 1.0)
        if n_not_learning > len(cond_snaps) * 0.7:
            warnings.append(
                "IP adapter showing very small loss_cond/loss_null gap in most intervals — "
                "adapter may not have started learning yet (normal in very short runs, "
                "check at 2000+ steps)")
            if len(snapshots) < 10:
                recommendations.append("Run with --steps 2000 for a more reliable assessment")

    # 5. Double-stream scale freezing
    if freeze_double:
        scale_snaps = [s for s in snapshots if "ip_scale_double" in s]
        if scale_snaps:
            max_double = max(s["ip_scale_double"] for s in scale_snaps)
            if max_double > 0.01:
                issues.append(
                    f"Double-stream scales not frozen: max={max_double:.4f} "
                    "(should be 0.0000 with freeze_double_stream_scales=true)")
                recommendations.append(
                    "Check adapter config freeze_double_stream_scales: true in stage1_512px.yaml")
        else:
            warnings.append("No ip_scale_double data collected — cannot verify freeze")

    # 6. Grad norm
    gn_snaps = [s for s in post_warmup if "grad_norm_smooth" in s]
    if gn_snaps:
        max_gn = max(s["grad_norm_smooth"] for s in gn_snaps)
        if max_gn > 10.0:
            warnings.append(f"Grad norm peaked at {max_gn:.2f} — monitor in full run")
            recommendations.append("If grad norm stays high: lower learning_rate by 2×")

    # ── Verdict ───────────────────────────────────────────────────────────────
    if issues:
        verdict = "FAIL"
    elif warnings:
        verdict = "WARN"
    else:
        verdict = "PASS"

    if verdict == "PASS":
        if aggressive:
            recommendations.append(
                "Production settings stable — safe to proceed to full chunk training.")
        else:
            recommendations.append(
                "Conservative settings stable — consider re-running with --aggressive "
                "to validate production probabilities before the full chunk.")

    return {
        "verdict": verdict,
        "issues": issues,
        "warnings": warnings,
        "recommendations": recommendations,
    }


# ── HTML report ───────────────────────────────────────────────────────────────

_HTML_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Quality Feature Test — {verdict}</title>
<style>
  body {{font-family:monospace;background:#111;color:#eee;margin:20px}}
  h1 {{color:#7df;margin-bottom:4px}}
  h2 {{color:#adf;font-size:1em;margin-top:20px;margin-bottom:4px}}
  .verdict-PASS {{color:#7f7;font-size:1.4em;font-weight:bold}}
  .verdict-WARN {{color:#fa7;font-size:1.4em;font-weight:bold}}
  .verdict-FAIL {{color:#f77;font-size:1.4em;font-weight:bold}}
  .meta {{color:#888;font-size:0.85em;margin-bottom:12px}}
  ul {{margin:4px 0;padding-left:20px}}
  li {{margin:2px 0;font-size:0.9em}}
  li.issue {{color:#f88}}
  li.warn  {{color:#fb8}}
  li.rec   {{color:#8df}}
  table {{border-collapse:collapse;max-width:600px;font-size:0.85em;margin-top:8px}}
  th,td {{border:1px solid #444;padding:4px 8px;text-align:left}}
  th {{background:#222;color:#adf}}
  td {{background:#181818}}
  .charts {{display:flex;flex-wrap:wrap;gap:14px;margin-top:12px}}
  canvas {{background:#181818;border:1px solid #333;border-radius:4px}}
</style>
</head>
<body>
<h1>Quality Feature Smoke Test</h1>
<div class="meta">Run: {run_name} &nbsp;|&nbsp; {ts} &nbsp;|&nbsp;
  mode={mode} &nbsp;|&nbsp; steps={steps}</div>
<div class="verdict-{verdict}">{verdict}</div>

<h2>Configuration</h2>
<table>
{cfg_rows}
</table>

<h2>Assessment</h2>
{assessment_html}

<h2>Metric Charts</h2>
<div class="charts">
  <div><div class="meta">Loss</div><canvas id="lossChart" width="440" height="200"></canvas></div>
  <div><div class="meta">Ref separation</div><canvas id="refChart" width="440" height="200"></canvas></div>
  <div><div class="meta">Adapter scales</div><canvas id="scaleChart" width="440" height="200"></canvas></div>
  <div><div class="meta">Grad norm (smooth)</div><canvas id="gradChart" width="440" height="200"></canvas></div>
</div>

<script>
const SNAPS = {snaps_json};

function chart(id, series, yMin) {{
  const cv = document.getElementById(id); if (!cv) return;
  const ctx = cv.getContext('2d');
  const W=cv.width, H=cv.height, pad={{t:12,r:12,b:28,l:48}};
  const cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  let allX=[], allY=[];
  for (const s of series) for (const [x,y] of s.pts) {{allX.push(x); if(y!=null) allY.push(y);}}
  if(!allX.length){{ctx.fillStyle='#555';ctx.font='11px monospace';ctx.fillText('no data',W/2-20,H/2);return;}}
  const xMin=Math.min(...allX), xMax=Math.max(...allX);
  const rawYMin=yMin!=null?yMin:Math.min(...allY)*0.97;
  const yMax=Math.max(...allY)*1.03+1e-9;
  const sx=x=>pad.l+(xMax>xMin?(x-xMin)/(xMax-xMin)*cw:cw/2);
  const sy=y=>pad.t+ch-(yMax>rawYMin?(y-rawYMin)/(yMax-rawYMin)*ch:ch/2);
  ctx.strokeStyle='#333';ctx.lineWidth=1;
  for(let i=0;i<=4;i++){{
    const y=rawYMin+(yMax-rawYMin)*i/4;
    ctx.beginPath();ctx.moveTo(pad.l,sy(y));ctx.lineTo(pad.l+cw,sy(y));ctx.stroke();
    ctx.fillStyle='#666';ctx.font='9px monospace';ctx.fillText(y.toPrecision(3),2,sy(y)+3);
  }}
  ctx.fillStyle='#666';ctx.font='9px monospace';
  for(let i=0;i<=4;i++){{
    const x=xMin+(xMax-xMin)*i/4;
    ctx.fillText(Math.round(x/1000)+'k',sx(x)-8,pad.t+ch+18);
  }}
  for(const s of series){{
    const pts=s.pts.filter(([,y])=>y!=null);
    if(!pts.length)continue;
    ctx.strokeStyle=s.c;ctx.lineWidth=2;
    ctx.beginPath();
    pts.forEach(([x,y],i)=>{{const cx=sx(x),cy=sy(y);i===0?ctx.moveTo(cx,cy):ctx.lineTo(cx,cy);}});
    ctx.stroke();
    ctx.fillStyle=s.c;
    for(const[x,y]of pts){{ctx.beginPath();ctx.arc(sx(x),sy(y),2.5,0,Math.PI*2);ctx.fill();}}
  }}
  let lx=pad.l+4;
  for(const s of series){{
    ctx.fillStyle=s.c;ctx.fillRect(lx,pad.t+2,10,2);
    ctx.fillStyle='#ccc';ctx.font='9px monospace';
    ctx.fillText(s.label,lx+14,pad.t+9);
    lx+=ctx.measureText(s.label).width+28;
  }}
}}

const steps=SNAPS.map(s=>s.step);
chart('lossChart',[
  {{label:'loss_smooth',c:'#7af',pts:SNAPS.map(s=>[s.step,s.loss_smooth??null])}},
  {{label:'loss_cond',  c:'#7f7',pts:SNAPS.map(s=>[s.step,s.loss_cond??null])}},
  {{label:'loss_null',  c:'#fa7',pts:SNAPS.map(s=>[s.step,s.loss_null??null])}},
],0);
chart('refChart',[
  {{label:'self_ref', c:'#7f7',pts:SNAPS.map(s=>[s.step,s.loss_self_ref??null])}},
  {{label:'cross_ref',c:'#f77',pts:SNAPS.map(s=>[s.step,s.loss_cross_ref??null])}},
],0);
chart('scaleChart',[
  {{label:'mean',  c:'#7af',pts:SNAPS.map(s=>[s.step,s.ip_scale_mean??null])}},
  {{label:'double',c:'#f77',pts:SNAPS.map(s=>[s.step,s.ip_scale_double??null])}},
  {{label:'single',c:'#7f7',pts:SNAPS.map(s=>[s.step,s.ip_scale_single??null])}},
],0);
chart('gradChart',[
  {{label:'grad_norm_smooth',c:'#fa7',pts:SNAPS.map(s=>[s.step,s.grad_norm_smooth??null])}},
],0);
</script>
</body>
</html>
"""


def render_html(
    run_name: str,
    mode: str,
    steps: int,
    cfg_params: dict,
    assessment: dict,
    snapshots: list[dict],
    ts: str,
) -> str:
    verdict = assessment["verdict"]

    # Config table
    cfg_rows = "\n".join(
        f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in cfg_params.items()
    )

    # Assessment block
    parts = []
    for item in assessment["issues"]:
        parts.append(f'<li class="issue">✗ {item}</li>')
    for item in assessment["warnings"]:
        parts.append(f'<li class="warn">⚠ {item}</li>')
    if verdict == "PASS":
        parts.append('<li style="color:#7f7">✓ All checks passed</li>')
    if assessment["recommendations"]:
        parts.append('<li style="color:#aaa;margin-top:6px;list-style:none"><b>Recommendations:</b></li>')
        for r in assessment["recommendations"]:
            parts.append(f'<li class="rec">→ {r}</li>')
    assessment_html = f"<ul>{''.join(parts)}</ul>"

    snaps_json = json.dumps(snapshots)
    return _HTML_TMPL.format(
        verdict=verdict,
        run_name=run_name,
        ts=ts,
        mode=mode,
        steps=steps,
        cfg_rows=cfg_rows,
        assessment_html=assessment_html,
        snaps_json=snaps_json,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Smoke-test QUALITY-1/2/3/6/8/9 features before full training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Prerequisites:")[0].strip(),
    )
    ap.add_argument("--shards",      default=str(_DEFAULT_SHARDS),
                    help="Shard directory (default: %(default)s)")
    ap.add_argument("--qwen3-cache", default=str(_DEFAULT_QWEN3) if _DEFAULT_QWEN3.exists() else None,
                    help="Precomputed Qwen3 cache dir")
    ap.add_argument("--vae-cache",   default=str(_DEFAULT_VAE)   if _DEFAULT_VAE.exists()   else None,
                    help="Precomputed VAE cache dir")
    ap.add_argument("--siglip-cache",default=str(_DEFAULT_SIGLIP) if _DEFAULT_SIGLIP.exists() else None,
                    help="Precomputed SigLIP cache dir (required for meaningful cross-ref test)")
    ap.add_argument("--config",      default=str(_BASE_CONFIG),
                    help="Base training config YAML (default: stage1_512px.yaml)")
    ap.add_argument("--steps",       type=int, default=500,
                    help="Number of training steps (default: 500; use 2000 for a stronger signal)")
    ap.add_argument("--log-every",   type=int, default=50,
                    help="Log interval in steps (default: 50)")
    ap.add_argument("--output-dir",  default="/tmp/quality_test",
                    help="Output directory for checkpoint + report (default: %(default)s)")
    ap.add_argument("--run-name",    default=None,
                    help="Label for the report (default: auto-generated timestamp)")
    ap.add_argument("--aggressive",  action="store_true",
                    help="Use production settings: cross_ref_prob=0.5, patch_shuffle_prob=0.5")
    ap.add_argument("--cross-ref-prob",    type=float, default=None,
                    help="Override cross_ref_prob (default: 0.3 conservative / 0.5 aggressive)")
    ap.add_argument("--patch-shuffle-prob",type=float, default=None,
                    help="Override patch_shuffle_prob (default: 0.3 / 0.5)")
    ap.add_argument("--style-loss-weight", type=float, default=0.05,
                    help="style_loss_weight (default: %(default)s)")
    ap.add_argument("--no-freeze-double",  action="store_true",
                    help="Disable double-stream scale freezing (to test QUALITY-2 separately)")
    ap.add_argument("--ai",          action="store_true",
                    help="Emit compact JSON verdict to stdout only; all training output goes to stderr")
    ap.add_argument("--report",      default=None,
                    help="HTML report output path (default: <output-dir>/quality_test_report.html)")
    args = ap.parse_args()

    # ── Resolve settings ──────────────────────────────────────────────────────
    if args.aggressive:
        cross_ref_prob     = args.cross_ref_prob     or 0.5
        patch_shuffle_prob = args.patch_shuffle_prob or 0.5
        mode = "aggressive"
    else:
        cross_ref_prob     = args.cross_ref_prob     or 0.3
        patch_shuffle_prob = args.patch_shuffle_prob or 0.3
        mode = "conservative"

    freeze_double = not args.no_freeze_double
    run_name = args.run_name or f"quality_test_{time.strftime('%Y%m%dT%H%M%S')}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = str(output_dir / "checkpoints")
    report_path = args.report or str(output_dir / "quality_test_report.html")

    ts_str = time.strftime("%Y-%m-%dT%H:%M:%S")
    _out = sys.stderr if args.ai else sys.stdout

    siglip_available = args.siglip_cache is not None and Path(args.siglip_cache).exists()

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    shards_dir = Path(args.shards)
    if not shards_dir.exists() or not list(shards_dir.glob("*.tar")):
        msg = f"No shard .tar files found in {shards_dir}"
        if args.ai:
            print(json.dumps({"ok": False, "error": msg, "verdict": "FAIL"}))
        else:
            print(f"ERROR: {msg}", file=sys.stderr)
            print(f"       Pass --shards pointing to a directory with .tar shards.")
        sys.exit(1)

    if not Path(args.config).exists():
        msg = f"Base config not found: {args.config}"
        if args.ai:
            print(json.dumps({"ok": False, "error": msg, "verdict": "FAIL"}))
        else:
            print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(1)

    if not _VENV_PYTHON.exists():
        msg = f"venv Python not found: {_VENV_PYTHON}  (run: cd train && python -m venv .venv)"
        if args.ai:
            print(json.dumps({"ok": False, "error": msg, "verdict": "FAIL"}))
        else:
            print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(1)

    # ── Build config ──────────────────────────────────────────────────────────
    cfg = build_test_config(
        base_config_path=Path(args.config),
        shards=str(shards_dir),
        qwen3_cache=args.qwen3_cache,
        vae_cache=args.vae_cache,
        siglip_cache=args.siglip_cache if siglip_available else None,
        checkpoint_dir=ckpt_dir,
        steps=args.steps,
        log_every=args.log_every,
        cross_ref_prob=cross_ref_prob,
        patch_shuffle_prob=patch_shuffle_prob,
        style_loss_weight=args.style_loss_weight,
        freeze_double=freeze_double,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False,
                                     prefix="quality_test_cfg_") as tf:
        yaml.dump(cfg, tf)
        temp_config = tf.name

    # ── Console header ────────────────────────────────────────────────────────
    print(_c("cyan", "═" * 64), file=_out)
    print(_c("bold", f"  Quality Feature Smoke Test — {run_name}"), file=_out)
    print(_c("cyan", "═" * 64), file=_out)
    print(f"  mode={mode}  steps={args.steps}  log_every={args.log_every}", file=_out)
    print(f"  cross_ref_prob={cross_ref_prob}  patch_shuffle_prob={patch_shuffle_prob}", file=_out)
    print(f"  style_loss_weight={args.style_loss_weight}"
          f"  freeze_double={freeze_double}", file=_out)
    print(f"  shards={args.shards}", file=_out)
    print(f"  siglip={'✓ ' + args.siglip_cache if siglip_available else '✗ not available (cross-ref test limited)'}", file=_out)
    if not args.qwen3_cache:
        print(_c("yellow", "  WARNING: no qwen3_cache — live encoding will be much slower"), file=_out)
    if not args.vae_cache:
        print(_c("yellow", "  WARNING: no vae_cache — live VAE encoding will be much slower"), file=_out)
    print(_c("cyan", "─" * 64), file=_out)
    print(f"  Config written to: {temp_config}", file=_out)
    print(f"  Checkpoints:       {ckpt_dir}", file=_out)
    print(f"  Report:            {report_path}", file=_out)
    print(_c("cyan", "─" * 64), file=_out)
    print("", file=_out)

    # ── Launch training subprocess ─────────────────────────────────────────────
    cmd = [
        str(_VENV_PYTHON), "-u", str(_TRAINER),
        "--config", temp_config,
        "--max-steps", str(args.steps),
        "--log-every", str(args.log_every),
    ]

    collector = MetricCollector()
    training_warnings: list[str] = []
    t_start = time.time()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(_TRAIN_DIR),
        )

        for raw_line in proc.stdout:  # type: ignore[union-attr]
            line = raw_line.rstrip()
            # Echo to appropriate stream
            print(line, file=sys.stderr if args.ai else sys.stdout, flush=True)

            snap = collector.feed(line)
            if snap is not None and not args.ai:
                display = _format_snapshot(snap, freeze_double)
                print(_c("dim", f"  ↳ {display}"), flush=True)

            if _RE_WARNING.search(line):
                training_warnings.append(line.strip())

        proc.wait()
        exit_code = proc.returncode

    except KeyboardInterrupt:
        print("\n\n[interrupted — assessing partial results]", file=_out)
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            pass
        exit_code = -1
    except Exception as exc:
        print(f"\nFATAL: failed to launch training: {exc}", file=sys.stderr)
        if args.ai:
            print(json.dumps({"ok": False, "error": str(exc), "verdict": "FAIL"}))
        sys.exit(1)
    finally:
        try:
            os.unlink(temp_config)
        except OSError:
            pass

    elapsed = time.time() - t_start

    # ── Assess ────────────────────────────────────────────────────────────────
    assessment = _assess(
        snapshots=collector.snapshots,
        exit_code=exit_code,
        steps=args.steps,
        aggressive=args.aggressive,
        freeze_double=freeze_double,
        cross_ref_prob=cross_ref_prob,
        patch_shuffle_prob=patch_shuffle_prob,
        siglip_available=siglip_available,
    )
    verdict = assessment["verdict"]

    # ── Console verdict ───────────────────────────────────────────────────────
    print("", file=_out)
    print(_c("cyan", "═" * 64), file=_out)
    _vcol = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}[verdict]
    print(_c(_vcol, f"  VERDICT: {verdict}"), file=_out)
    print("", file=_out)
    for item in assessment["issues"]:
        print(_c("red", f"  ✗ {item}"), file=_out)
    for item in assessment["warnings"]:
        print(_c("yellow", f"  ⚠ {item}"), file=_out)
    if verdict == "PASS":
        print(_c("green", "  ✓ All checks passed"), file=_out)
    if assessment["recommendations"]:
        print("", file=_out)
        for r in assessment["recommendations"]:
            print(f"  → {r}", file=_out)
    print("", file=_out)

    # Summary stats
    if collector.snapshots:
        last = collector.snapshots[-1]
        print(f"  steps_collected={len(collector.snapshots)}"
              f"  elapsed={elapsed:.0f}s"
              f"  final_loss_smooth={last.get('loss_smooth', '?')}",
              file=_out)
        sr = last.get("loss_self_ref")
        cr = last.get("loss_cross_ref")
        if sr is not None and cr is not None:
            print(f"  final: self_ref={sr:.4f}  cross_ref={cr:.4f}"
                  f"  gap={cr - sr:+.4f}", file=_out)
        sd = last.get("ip_scale_double")
        ss = last.get("ip_scale_single")
        sm = last.get("ip_scale_mean")
        if sm is not None:
            print(f"  ip_scale: mean={sm:.4f}"
                  + (f"  double={sd:.4f}" if sd is not None else "")
                  + (f"  single={ss:.4f}" if ss is not None else ""), file=_out)

    print(_c("cyan", "─" * 64), file=_out)
    print(f"  Report: {report_path}", file=_out)
    print(_c("cyan", "═" * 64), file=_out)

    # ── HTML report ────────────────────────────────────────────────────────────
    cfg_params = {
        "mode":               mode,
        "steps":              args.steps,
        "cross_ref_prob":     cross_ref_prob,
        "patch_shuffle_prob": patch_shuffle_prob,
        "style_loss_weight":  args.style_loss_weight,
        "freeze_double_stream_scales": freeze_double,
        "shards":             args.shards,
        "siglip_cache":       args.siglip_cache or "— not available",
        "elapsed_secs":       round(elapsed),
        "exit_code":          exit_code,
    }
    html = render_html(
        run_name=run_name,
        mode=mode,
        steps=args.steps,
        cfg_params=cfg_params,
        assessment=assessment,
        snapshots=collector.snapshots,
        ts=ts_str,
    )
    try:
        with open(report_path, "w") as f:
            f.write(html)
        print(f"\nReport written: {report_path}", file=_out)
    except OSError as e:
        print(f"WARNING: could not write report: {e}", file=sys.stderr)

    # ── AI JSON output ─────────────────────────────────────────────────────────
    if args.ai:
        last_snap = collector.snapshots[-1] if collector.snapshots else {}
        ai_out = {
            "ok": verdict == "PASS",
            "verdict": verdict,
            "mode": mode,
            "steps_completed": last_snap.get("step"),
            "final_loss_smooth": last_snap.get("loss_smooth"),
            "final_self_ref": last_snap.get("loss_self_ref"),
            "final_cross_ref": last_snap.get("loss_cross_ref"),
            "final_ip_scale_double": last_snap.get("ip_scale_double"),
            "issues": assessment["issues"],
            "warnings": assessment["warnings"],
            "recommendations": assessment["recommendations"],
            "elapsed_secs": round(elapsed),
            "report": report_path,
        }
        print(json.dumps(ai_out))

    sys.exit(0 if verdict != "FAIL" else 1)


if __name__ == "__main__":
    main()
