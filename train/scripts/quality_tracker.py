#!/usr/bin/env python3
"""
quality_tracker.py — aggregates per-checkpoint quality signals and produces
an HTML report with inline charts or a compact --ai JSON summary.

Data sources (all optional — script degrades gracefully when absent):
  <checkpoint_dir>/eval/step_NNNNNNN/eval_results.json   (from eval.py)
  <checkpoint_dir>/val_loss.jsonl
  <data_root>/.heartbeat/trainer_chunk*.json             (latest snapshot)

Usage:
  python train/scripts/quality_tracker.py \\
      --checkpoint-dir /Volumes/2TBSSD/checkpoints/stage1 \\
      --output /tmp/quality_report.html

  python train/scripts/quality_tracker.py \\
      --checkpoint-dir /Volumes/2TBSSD/checkpoints/stage1 \\
      --ai
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Optional

DATA_ROOT_DEFAULT = "/Volumes/2TBSSD"


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_eval_results(checkpoint_dir: str) -> list[dict]:
    """Load all eval_results.json files from checkpoint_dir/eval/step_*/."""
    points = []
    pattern = os.path.join(checkpoint_dir, "eval", "step_*", "eval_results.json")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path) as f:
                data = json.load(f)
            step_dir = os.path.basename(os.path.dirname(path))
            step = int(step_dir.replace("step_", ""))
            clip_i = data.get("clip_i_mean") or data.get("clip_i")
            clip_t = data.get("clip_t_mean") or data.get("clip_t")
            points.append({"step": step, "clip_i": clip_i, "clip_t": clip_t, "raw": data})
        except Exception:
            pass
    points.sort(key=lambda x: x["step"])
    return points


def _load_val_loss(checkpoint_dir: str) -> list[dict]:
    """Load val_loss.jsonl from checkpoint_dir."""
    path = os.path.join(checkpoint_dir, "val_loss.jsonl")
    points = []
    if not os.path.exists(path):
        return points
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if "step" in rec and "val_loss" in rec:
                        points.append({"step": int(rec["step"]), "val_loss": float(rec["val_loss"])})
                except Exception:
                    pass
    except Exception:
        pass
    points.sort(key=lambda x: x["step"])
    return points


def _load_heartbeat(data_root: str) -> Optional[dict]:
    """Load the most recent trainer heartbeat file."""
    pattern = os.path.join(data_root, ".heartbeat", "trainer_chunk*.json")
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    for path in candidates:
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


# ── Analysis ──────────────────────────────────────────────────────────────────

def _trend(values: list[float]) -> str:
    """Simple linear trend from the last 3 values."""
    if len(values) < 2:
        return "unknown"
    recent = values[-3:]
    delta = recent[-1] - recent[0]
    threshold = abs(recent[0]) * 0.02 if recent[0] != 0 else 0.001
    if delta < -threshold:
        return "improving"
    if delta > threshold:
        return "degrading"
    return "flat"


def _build_summary(eval_pts: list, val_pts: list, hb: Optional[dict]) -> dict:
    best_clip_i = None
    best_clip_t = None
    for pt in eval_pts:
        if pt["clip_i"] is not None:
            if best_clip_i is None or pt["clip_i"] > best_clip_i["value"]:
                best_clip_i = {"step": pt["step"], "value": round(pt["clip_i"], 4)}
        if pt["clip_t"] is not None:
            if best_clip_t is None or pt["clip_t"] > best_clip_t["value"]:
                best_clip_t = {"step": pt["step"], "value": round(pt["clip_t"], 4)}

    latest_val_loss = None
    if val_pts:
        vp = val_pts[-1]
        latest_val_loss = {"step": vp["step"], "value": round(vp["val_loss"], 6)}

    clip_i_vals = [p["clip_i"] for p in eval_pts if p["clip_i"] is not None]
    clip_t_vals = [p["clip_t"] for p in eval_pts if p["clip_t"] is not None]
    val_vals    = [p["val_loss"] for p in val_pts]

    trend_clip_i   = _trend(clip_i_vals)
    trend_clip_t   = _trend(clip_t_vals)
    trend_val_loss = _trend(val_vals) if val_vals else "unknown"
    # val_loss improving = decreasing
    if trend_val_loss == "improving":
        trend_val_loss = "improving"
    elif trend_val_loss == "degrading":
        trend_val_loss = "degrading"

    # top_action heuristic
    top_action = None
    if hb:
        hb_step = hb.get("step", 0)
        last_eval_step = eval_pts[-1]["step"] if eval_pts else 0
        steps_since_eval = hb_step - last_eval_step
        if steps_since_eval >= 10000:
            top_action = f"Run eval at step {hb_step} — {steps_since_eval:,} steps since last eval point."
    if top_action is None:
        if trend_val_loss == "degrading":
            top_action = "Validation loss is degrading — review recent checkpoints for overfitting."
        elif not eval_pts:
            top_action = "No eval data found — run eval.py to populate eval_results.json."
        else:
            top_action = "Training progressing normally."

    return {
        "steps_with_eval": len(eval_pts),
        "best_clip_i": best_clip_i,
        "best_clip_t": best_clip_t,
        "latest_val_loss": latest_val_loss,
        "trend_clip_i": trend_clip_i,
        "trend_clip_t": trend_clip_t,
        "trend_val_loss": trend_val_loss,
    }, top_action


# ── HTML report ───────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Quality Tracker</title>
<style>
  body {{ font-family: monospace; background: #111; color: #eee; margin: 20px; }}
  h1 {{ color: #7df; margin-bottom: 4px; }}
  h2 {{ color: #adf; margin-top: 24px; margin-bottom: 4px; font-size: 1em; }}
  .meta {{ color: #888; font-size: 0.85em; margin-bottom: 16px; }}
  table {{ border-collapse: collapse; width: 100%; max-width: 600px; font-size: 0.85em; }}
  th, td {{ border: 1px solid #444; padding: 4px 8px; text-align: left; }}
  th {{ background: #222; color: #adf; }}
  td {{ background: #181818; }}
  .charts {{ display: flex; flex-wrap: wrap; gap: 16px; margin-top: 12px; }}
  canvas {{ background: #181818; border: 1px solid #333; border-radius: 4px; }}
  .action {{ background: #1a2a1a; border: 1px solid #3a6a3a; padding: 8px 12px;
             border-radius: 4px; color: #afa; margin-top: 12px; }}
  .heartbeat {{ background: #1a1a2a; border: 1px solid #3a3a6a; padding: 8px 12px;
                border-radius: 4px; font-size: 0.85em; margin-top: 8px; }}
</style>
</head>
<body>
<h1>IP-Adapter Quality Tracker</h1>
<div class="meta">Generated: {generated_at}</div>

<div class="action">&#9654; {top_action}</div>

{heartbeat_section}

<h2>Summary</h2>
<table>
{summary_rows}
</table>

<h2>Charts</h2>
<div class="charts">
  <div>
    <div style="color:#888;font-size:0.8em;margin-bottom:4px">Loss curves</div>
    <canvas id="lossChart" width="480" height="240"></canvas>
  </div>
  <div>
    <div style="color:#888;font-size:0.8em;margin-bottom:4px">CLIP scores</div>
    <canvas id="clipChart" width="480" height="240"></canvas>
  </div>
  <div>
    <div style="color:#888;font-size:0.8em;margin-bottom:4px">IP scale (adapter)</div>
    <canvas id="scaleChart" width="480" height="240"></canvas>
  </div>
</div>

<script>
const DATA = {data_json};

function drawChart(canvasId, series, opts) {{
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const pad = {{top: 16, right: 16, bottom: 32, left: 56}};
  const cw = W - pad.left - pad.right;
  const ch = H - pad.top - pad.bottom;

  ctx.clearRect(0, 0, W, H);

  // collect all points
  let allX = [], allY = [];
  for (const s of series) {{
    for (const [x, y] of s.pts) {{ allX.push(x); if (y != null) allY.push(y); }}
  }}
  if (allX.length === 0) {{
    ctx.fillStyle = '#555'; ctx.font = '12px monospace';
    ctx.fillText('no data', pad.left + cw/2 - 20, pad.top + ch/2);
    return;
  }}

  const xMin = Math.min(...allX), xMax = Math.max(...allX);
  const yMin = opts.yMin != null ? opts.yMin : Math.min(...allY) * 0.98;
  const yMax = opts.yMax != null ? opts.yMax : Math.max(...allY) * 1.02 + 1e-9;

  const sx = x => pad.left + (xMax > xMin ? (x - xMin) / (xMax - xMin) * cw : cw/2);
  const sy = y => pad.top  + ch - (yMax > yMin ? (y - yMin) / (yMax - yMin) * ch : ch/2);

  // grid
  ctx.strokeStyle = '#333'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {{
    const y = yMin + (yMax - yMin) * i / 4;
    const cy = sy(y);
    ctx.beginPath(); ctx.moveTo(pad.left, cy); ctx.lineTo(pad.left + cw, cy); ctx.stroke();
    ctx.fillStyle = '#666'; ctx.font = '10px monospace';
    ctx.fillText(y.toPrecision(3), 2, cy + 3);
  }}

  // x-axis labels
  ctx.fillStyle = '#666'; ctx.font = '10px monospace';
  for (let i = 0; i <= 4; i++) {{
    const x = xMin + (xMax - xMin) * i / 4;
    const cx = sx(x);
    ctx.fillText((x/1000).toFixed(0)+'k', cx - 10, pad.top + ch + 18);
  }}

  // series
  for (const s of series) {{
    const pts = s.pts.filter(([, y]) => y != null);
    if (pts.length === 0) continue;
    ctx.strokeStyle = s.color; ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach(([x, y], i) => {{
      const cx = sx(x), cy = sy(y);
      if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
    }});
    ctx.stroke();
    // dots
    ctx.fillStyle = s.color;
    for (const [x, y] of pts) {{
      ctx.beginPath(); ctx.arc(sx(x), sy(y), 3, 0, Math.PI*2); ctx.fill();
    }}
    // legend
  }}

  // legend
  let lx = pad.left + 4;
  for (const s of series) {{
    ctx.fillStyle = s.color; ctx.fillRect(lx, pad.top + 4, 12, 3);
    ctx.fillStyle = '#ccc'; ctx.font = '10px monospace';
    ctx.fillText(s.label, lx + 16, pad.top + 10);
    lx += ctx.measureText(s.label).width + 36;
  }}
}}

// loss curves: val_loss, loss_smooth from heartbeat series
const lossData = DATA.val_loss || [];
const hbStep = DATA.hb_step;
const hbLoss = DATA.hb_loss_smooth;
const valPts = lossData.map(d => [d.step, d.val_loss]);
const hbPts  = hbStep != null ? [[hbStep, hbLoss]] : [];
drawChart('lossChart', [
  {{label:'val_loss', color:'#7af', pts: valPts}},
  {{label:'loss_smooth', color:'#fa7', pts: hbPts}},
], {{yMin: null, yMax: null}});

// CLIP scores
const evalPts = DATA.eval_pts || [];
drawChart('clipChart', [
  {{label:'CLIP-I', color:'#7fa', pts: evalPts.map(d => [d.step, d.clip_i])}},
  {{label:'CLIP-T', color:'#fa7', pts: evalPts.map(d => [d.step, d.clip_t])}},
], {{yMin: 0, yMax: null}});

// IP scale
const scalePts = DATA.scale_pts || [];
drawChart('scaleChart', [
  {{label:'scale_mean',   color:'#7af', pts: scalePts.map(d => [d.step, d.mean])}},
  {{label:'scale_double', color:'#f77', pts: scalePts.map(d => [d.step, d.double])}},
  {{label:'scale_single', color:'#7f7', pts: scalePts.map(d => [d.step, d.single])}},
], {{yMin: 0, yMax: null}});
</script>
</body>
</html>
"""


def _render_html(checkpoint_dir: str, data_root: str, output_path: str,
                 eval_pts: list, val_pts: list, hb: Optional[dict],
                 summary: dict, top_action: str) -> None:
    import datetime

    generated_at = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # Heartbeat section
    if hb:
        hb_step = hb.get("step", "?")
        hb_loss = hb.get("loss_smooth") or hb.get("loss", "?")
        hb_lr   = hb.get("lr", "?")
        hb_sps  = hb.get("steps_per_sec", "?")
        hb_eta  = hb.get("eta_sec")
        eta_str = ""
        if hb_eta:
            h, m = divmod(int(hb_eta) // 60, 60)
            eta_str = f"  ETA {h}h{m:02d}m"
        heartbeat_section = (
            f'<div class="heartbeat">'
            f'Live: step={hb_step}  loss_smooth={hb_loss:.4f}  lr={hb_lr:.2e}  '
            f'{hb_sps:.3f} steps/s{eta_str}'
            f'</div>'
            if isinstance(hb_loss, float) and isinstance(hb_lr, float)
            else f'<div class="heartbeat">Live heartbeat: step={hb_step}</div>'
        )
    else:
        heartbeat_section = ""

    # Summary table
    rows = [
        ("steps_with_eval", summary["steps_with_eval"]),
        ("best_clip_i", summary["best_clip_i"] or "—"),
        ("best_clip_t", summary["best_clip_t"] or "—"),
        ("latest_val_loss", summary["latest_val_loss"] or "—"),
        ("trend_clip_i", summary["trend_clip_i"]),
        ("trend_clip_t", summary["trend_clip_t"]),
        ("trend_val_loss", summary["trend_val_loss"]),
    ]
    summary_rows = "\n".join(
        f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows
    )

    # Chart data JSON
    hb_step_val = hb.get("step") if hb else None
    hb_loss_smooth = hb.get("loss_smooth") if hb else None
    scale_pts = []
    if hb and hb.get("ip_scale_mean") is not None:
        scale_pts = [{"step": hb.get("step", 0),
                      "mean": hb.get("ip_scale_mean"),
                      "double": hb.get("ip_scale_double"),
                      "single": hb.get("ip_scale_single")}]

    data_json = json.dumps({
        "eval_pts": eval_pts,
        "val_loss": val_pts,
        "hb_step": hb_step_val,
        "hb_loss_smooth": hb_loss_smooth,
        "scale_pts": scale_pts,
    })

    html = _HTML_TEMPLATE.format(
        generated_at=generated_at,
        top_action=top_action,
        heartbeat_section=heartbeat_section,
        summary_rows=summary_rows,
        data_json=data_json,
    )

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Report written to {output_path}", file=sys.stderr)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="IP-Adapter quality tracker")
    parser.add_argument("--checkpoint-dir", default="/Volumes/2TBSSD/checkpoints/stage1",
                        help="Checkpoint directory (contains eval/ and val_loss.jsonl)")
    parser.add_argument("--data-root", default=DATA_ROOT_DEFAULT,
                        help="Pipeline data root (for heartbeat files)")
    parser.add_argument("--output", default="/tmp/quality_report.html",
                        help="Output HTML path (ignored with --ai)")
    parser.add_argument("--ai", action="store_true",
                        help="Emit compact JSON to stdout instead of HTML")
    args = parser.parse_args()

    eval_pts = _load_eval_results(args.checkpoint_dir)
    val_pts  = _load_val_loss(args.checkpoint_dir)
    hb       = _load_heartbeat(args.data_root)

    summary, top_action = _build_summary(eval_pts, val_pts, hb)

    if args.ai:
        out = {
            "summary": summary,
            "top_action": top_action,
            "data_points": {
                "eval": eval_pts,
                "val_loss": val_pts,
                "heartbeat": hb,
            },
        }
        print(json.dumps(out))
        return

    _render_html(args.checkpoint_dir, args.data_root, args.output,
                 eval_pts, val_pts, hb, summary, top_action)


if __name__ == "__main__":
    main()
