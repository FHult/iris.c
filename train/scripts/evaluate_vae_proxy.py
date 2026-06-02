#!/usr/bin/env python3
"""
train/scripts/evaluate_vae_proxy.py — Comprehensive proxy VAE evaluation.

Runs a three-tier evaluation suite and writes a self-contained HTML report.

Tier 1  Latent quality     — MSE, cosine similarity, per-channel stats, FFT correlation
Tier 2  Decoded quality    — Reconstruction PSNR, MSE, channel std ratio (needs teacher)
Tier 3  Failure analysis   — Error distribution, fallback rate, quality mode comparison

Usage:
    # Full evaluation against a specific shard pool:
    python train/scripts/evaluate_vae_proxy.py \\
        --proxy /Volumes/2TBSSD/checkpoints/vae_proxy/proxy_final.safetensors \\
        --shards /Volumes/16TBCold/shards \\
        --vae-cache /Volumes/16TBCold/precomputed/vae/current \\
        --flux-model flux-klein-model \\
        --n-images 2000 \\
        --report /tmp/proxy_eval.html

    # Evaluate on a specific campaign shard list:
    python train/scripts/evaluate_vae_proxy.py \\
        --proxy proxy_final.safetensors \\
        --campaign warmup-run1 \\
        --n-images 500 \\
        --report /tmp/eval.html

    # Fast tier-1 only (no teacher needed):
    python train/scripts/evaluate_vae_proxy.py \\
        --proxy proxy_final.safetensors \\
        --vae-cache /path/to/vae/current \\
        --tier 1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_proxy(proxy_path: str, quality_mode: str, flux_model: str | None):
    import mlx.core as mx
    from vae_distill.proxy import ProxyVAE, BALANCED
    from vae_distill.teacher import TeacherEncoder

    teacher = None
    if flux_model:
        print("Loading teacher VAE ...")
        teacher = TeacherEncoder.load(flux_model)

    mode = quality_mode or BALANCED
    print(f"Loading proxy ({mode} mode) from {proxy_path} ...")
    proxy = ProxyVAE.load(proxy_path, teacher=teacher,
                          quality_mode=mode, fallback_threshold=0.75)
    return proxy, teacher


def _build_dataset(shards_dir: str | None, vae_cache: str,
                   campaign: str | None, n_images: int):
    from vae_distill.dataset import build_index, VAEDistillDataset

    shards = [shards_dir] if shards_dir else []

    if campaign:
        # Look up shards used in this flywheel campaign from the DB
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from flywheel_lib import FlywheelDB
            db = FlywheelDB()
            iters = db.get_iterations(campaign)
            campaign_shards: set[str] = set()
            for row in iters:
                for sid in (row.get("shard_ids") or []):
                    # shard_ids are stems like "000042" — find matching tars
                    campaign_shards.add(sid)
            if campaign_shards and shards:
                print(f"  Campaign {campaign}: {len(campaign_shards)} shards")
        except Exception as e:
            print(f"  Warning: could not read campaign {campaign}: {e}", file=sys.stderr)

    if not shards:
        # Derive shard dirs from cache path
        # Fall back to cold shards
        cold = Path(vae_cache).parent.parent.parent
        for candidate in [cold / "shards", Path("/Volumes/16TBCold/shards")]:
            if candidate.exists():
                shards = [str(candidate)]
                break

    if not shards:
        raise RuntimeError("No shard directories found; pass --shards explicitly")

    index = build_index(shards, vae_cache)
    if not index:
        raise RuntimeError(f"No training pairs found in {shards}")

    np.random.seed(42)
    np.random.shuffle(index)
    subset = index[:n_images]

    dataset = VAEDistillDataset(
        index=subset, vae_cache=vae_cache, image_size=512,
        batch_size=8, hflip_prob=0.0, seed=0,
    )
    return dataset, len(subset)


# ---------------------------------------------------------------------------
# Tier 1: Latent quality
# ---------------------------------------------------------------------------

def tier1_latent(proxy, dataset, n_images: int) -> dict:
    """MSE, cosine similarity, channel statistics, FFT correlation."""
    import mlx.core as mx

    ch_std_arr = np.array(proxy._ch_std).reshape(1, -1, 1, 1) + 1e-6

    proxy_lats, teacher_lats = [], []
    seen = 0

    for images_np, latents_np in dataset:
        if seen >= n_images:
            break
        images_mx  = mx.array(images_np)
        proxy_lat  = proxy.encode(images_mx, check_confidence=False)
        mx.eval(proxy_lat)
        proxy_lats.append(np.array(proxy_lat))
        teacher_lats.append(latents_np)
        seen += len(images_np)

    if not proxy_lats:
        return {"error": "dataset yielded no images"}

    P = np.concatenate(proxy_lats,  axis=0)   # [N, 32, H, W]
    T = np.concatenate(teacher_lats, axis=0)

    # Channel-normalised MSE
    diff_norm   = (P - T) / ch_std_arr
    latent_mse_norm = float(np.mean(diff_norm ** 2))

    # Per-channel cosine similarity
    Pf = P.reshape(P.shape[0], 32, -1)          # [N, 32, HW]
    Tf = T.reshape(T.shape[0], 32, -1)
    dot  = np.sum(Pf * Tf, axis=2)
    norm = np.linalg.norm(Pf, axis=2) * np.linalg.norm(Tf, axis=2) + 1e-8
    cosine_sim = float(np.mean(dot / norm))

    # Per-channel mean/std ratio (distribution matching)
    P_mean = P.mean(axis=(0, 2, 3))   # [32]
    T_mean = T.mean(axis=(0, 2, 3))
    P_std  = P.std(axis=(0, 2, 3))
    T_std  = T.std(axis=(0, 2, 3))
    ch_mean_mae  = float(np.mean(np.abs(P_mean - T_mean)))
    ch_std_ratio = float(np.mean(P_std / (T_std + 1e-8)))

    # FFT magnitude correlation (spatial frequency structure)
    # rfft2 for efficiency; compare magnitude spectra averaged over channels
    P_fft = np.fft.rfft2(P.astype(np.float32))   # [N, 32, H, W//2+1]
    T_fft = np.fft.rfft2(T.astype(np.float32))
    P_mag = np.abs(P_fft).flatten()
    T_mag = np.abs(T_fft).flatten()
    # Pearson correlation of frequency magnitudes
    corr_mat = np.corrcoef(P_mag, T_mag)
    fft_corr = float(corr_mat[0, 1]) if corr_mat.shape == (2, 2) else 0.0

    result = {
        "n_images":         seen,
        "latent_mse_norm":  round(latent_mse_norm, 6),
        "cosine_sim":       round(cosine_sim, 4),
        "ch_mean_mae":      round(ch_mean_mae, 6),
        "ch_std_ratio":     round(ch_std_ratio, 4),
        "fft_corr":         round(fft_corr, 4),
        "pass_cosine":      cosine_sim > 0.95,
        "pass_ch_std":      0.95 <= ch_std_ratio <= 1.05,
        "pass_fft":         fft_corr > 0.98,
        # Per-channel data for HTML report
        "_P_mean": P_mean.tolist(),
        "_T_mean": T_mean.tolist(),
        "_P_std":  P_std.tolist(),
        "_T_std":  T_std.tolist(),
    }

    _print_tier("Tier 1 — Latent Quality", [
        (f"latent_mse_norm  = {latent_mse_norm:.6f}", True),
        (f"cosine_sim       = {cosine_sim:.4f}",
         cosine_sim > 0.95, "target >0.95"),
        (f"ch_std_ratio     = {ch_std_ratio:.4f}",
         0.95 <= ch_std_ratio <= 1.05, "target 0.95–1.05"),
        (f"fft_corr         = {fft_corr:.4f}",
         fft_corr > 0.98, "target >0.98"),
    ])
    return result


# ---------------------------------------------------------------------------
# Tier 2: Decoded quality (requires teacher decoder)
# ---------------------------------------------------------------------------

def tier2_decoded(proxy, teacher, dataset, n_images: int) -> dict:
    """PSNR and pixel MSE after decoding proxy vs teacher latents."""
    import mlx.core as mx

    proxy_decs, teacher_decs = [], []
    seen = 0

    for images_np, latents_np in dataset:
        if seen >= n_images:
            break
        images_mx   = mx.array(images_np)
        proxy_lat   = proxy.encode(images_mx, check_confidence=False)
        teacher_lat = mx.array(latents_np)
        proxy_dec   = teacher.decode(proxy_lat)
        teacher_dec = teacher.decode(teacher_lat)
        mx.eval(proxy_dec, teacher_dec)
        proxy_decs.append(np.array(proxy_dec))
        teacher_decs.append(np.array(teacher_dec))
        seen += len(images_np)

    if not proxy_decs:
        return {"error": "dataset yielded no images"}

    P = np.concatenate(proxy_decs,   axis=0)
    T = np.concatenate(teacher_decs, axis=0)
    mse  = float(np.mean((P - T) ** 2))
    psnr = float(10 * np.log10(4.0 / (mse + 1e-10)))  # image range [-1,1] → peak=2

    # Per-channel decoded stats
    P_std = P.std(axis=(0, 2, 3))
    T_std = T.std(axis=(0, 2, 3))
    decoded_std_ratio = float(np.mean(P_std / (T_std + 1e-8)))

    result = {
        "n_images":           seen,
        "decoded_mse":        round(mse, 6),
        "decoded_psnr_db":    round(psnr, 2),
        "decoded_std_ratio":  round(decoded_std_ratio, 4),
        "pass_psnr":          psnr > 35.0,
        "pass_std_ratio":     0.95 <= decoded_std_ratio <= 1.05,
    }

    _print_tier("Tier 2 — Decoded Quality", [
        (f"decoded_psnr      = {psnr:.1f} dB",
         psnr > 35.0, "target >35 dB"),
        (f"decoded_std_ratio = {decoded_std_ratio:.4f}",
         0.95 <= decoded_std_ratio <= 1.05, "target 0.95–1.05"),
    ])
    return result


# ---------------------------------------------------------------------------
# Tier 3: Failure analysis and quality mode comparison
# ---------------------------------------------------------------------------

def tier3_analysis(proxy, dataset, n_images: int) -> dict:
    """Per-image error distribution and quality mode comparison."""
    import mlx.core as mx
    from vae_distill.proxy import SPEED, BALANCED, HIGH_FIDELITY

    ch_std = np.array(proxy._ch_std).reshape(1, -1, 1, 1) + 1e-6
    all_mse: list[float] = []
    all_conf: list[float] = []
    seen = 0

    for images_np, latents_np in dataset:
        if seen >= n_images:
            break
        images_mx  = mx.array(images_np)
        proxy_lat  = proxy.encode(images_mx, check_confidence=False)
        mx.eval(proxy_lat)

        p = np.array(proxy_lat)
        t = latents_np
        mse_per  = np.mean(((p - t) / ch_std) ** 2, axis=(1, 2, 3))
        all_mse.extend(mse_per.tolist())

        # Compute confidence for each image in batch
        from vae_distill.proxy import _mahalanobis_confidence
        for i in range(p.shape[0]):
            conf, _ = _mahalanobis_confidence(
                mx.array(p[i:i+1]),
                proxy._ch_mean, proxy._ch_std,
            )
            all_conf.append(conf)

        seen += len(images_np)

    if not all_mse:
        return {"error": "dataset yielded no images"}

    # Error distribution percentiles
    pcts = [50, 90, 95, 99]
    mse_pcts = {f"p{p}": round(float(np.percentile(all_mse, p)), 4) for p in pcts}

    # Fallback rate simulation for each mode and threshold
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]
    mode_sim = {}
    for thr in thresholds:
        n_fallback = sum(1 for c in all_conf if c < thr)
        mode_sim[f"thr_{int(thr*100)}"] = {
            "fallback_rate": round(n_fallback / max(1, len(all_conf)), 4),
            "n_fallback":    n_fallback,
        }

    result = {
        "n_images":       seen,
        "mean_norm_mse":  round(float(np.mean(all_mse)), 4),
        "mse_percentiles": mse_pcts,
        "mean_confidence": round(float(np.mean(all_conf)), 4),
        "proxy_stats":     proxy.stats(),
        "fallback_simulation": mode_sim,
        "_mse_hist":  _histogram(all_mse, bins=20),
        "_conf_hist": _histogram(all_conf, bins=20),
    }

    _print_tier("Tier 3 — Failure Analysis", [
        (f"mean_norm_mse = {result['mean_norm_mse']:.4f}", True),
        (f"p95_norm_mse  = {mse_pcts['p95']:.4f}", True),
        (f"mean_conf     = {result['mean_confidence']:.3f}", True),
        (f"fallback@0.80 = {mode_sim['thr_80']['fallback_rate']:.1%}", True),
    ])
    return result


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _histogram(values: list[float], bins: int = 20) -> dict:
    """Return histogram data as {bins: [...], counts: [...]}."""
    if not values:
        return {"bins": [], "counts": []}
    arr = np.array(values, dtype=np.float32)
    counts, edges = np.histogram(arr, bins=bins)
    return {
        "bins":   [round(float(e), 4) for e in edges[:-1]],
        "counts": counts.tolist(),
    }


def _bar_chart_js(canvas_id: str, labels: list, values: list,
                  title: str, color: str = "#4e79a7") -> str:
    return f"""
<canvas id="{canvas_id}" width="600" height="200"></canvas>
<script>
new Chart(document.getElementById('{canvas_id}'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps([f"{v:.3f}" for v in labels])},
    datasets: [{{ data: {json.dumps(values)}, backgroundColor: '{color}',
                  borderWidth: 0 }}]
  }},
  options: {{
    plugins: {{ legend: {{ display: false }},
                title: {{ display: true, text: '{title}' }} }},
    scales: {{ y: {{ beginAtZero: true }} }}
  }}
}});
</script>"""


def generate_html_report(results: dict, proxy_path: str) -> str:
    t1 = results.get("tier1", {})
    t2 = results.get("tier2", {})
    t3 = results.get("tier3", {})

    def _badge(ok: bool | None, label: str = "") -> str:
        if ok is None:
            return f'<span class="badge info">{label or "N/A"}</span>'
        color = "pass" if ok else "fail"
        text  = label or ("PASS" if ok else "FAIL")
        return f'<span class="badge {color}">{text}</span>'

    ch_labels = list(range(32))
    t1_pmean = t1.get("_P_mean", [0]*32)
    t1_tmean = t1.get("_T_mean", [0]*32)
    t1_pstd  = t1.get("_P_std",  [0]*32)
    t1_tstd  = t1.get("_T_std",  [0]*32)

    mse_hist  = t3.get("_mse_hist",  {"bins": [], "counts": []})
    conf_hist = t3.get("_conf_hist", {"bins": [], "counts": []})

    # Precompute nested lookups — dict defaults inside f-string replacement
    # fields are parsed as set literals ({{}} → set containing dict), which fails.
    _fb_sim = t3.get("fallback_simulation", {})
    _fb80   = _fb_sim.get("thr_80", {}).get("fallback_rate", "—")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Proxy VAE Evaluation Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         max-width:1100px; margin:0 auto; padding:2rem; background:#f8f9fa; color:#212529; }}
  h1   {{ font-size:1.6rem; border-bottom:3px solid #4e79a7; padding-bottom:.5rem; }}
  h2   {{ font-size:1.2rem; margin-top:2rem; color:#4e79a7; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
           gap:1rem; margin:1rem 0; }}
  .card {{ background:#fff; border-radius:8px; padding:1rem;
           box-shadow:0 1px 3px rgba(0,0,0,.1); }}
  .metric {{ font-size:1.8rem; font-weight:700; color:#212529; }}
  .label  {{ font-size:.8rem; color:#6c757d; text-transform:uppercase; letter-spacing:.05em; }}
  .badge  {{ display:inline-block; padding:.2em .6em; border-radius:.25em;
             font-size:.8rem; font-weight:600; }}
  .badge.pass {{ background:#d4edda; color:#155724; }}
  .badge.fail {{ background:#f8d7da; color:#721c24; }}
  .badge.info {{ background:#d1ecf1; color:#0c5460; }}
  table   {{ width:100%; border-collapse:collapse; background:#fff; border-radius:8px;
             overflow:hidden; box-shadow:0 1px 3px rgba(0,0,0,.1); }}
  th,td   {{ text-align:left; padding:.6rem 1rem; border-bottom:1px solid #dee2e6; }}
  th      {{ background:#4e79a7; color:#fff; font-weight:600; }}
  tr:last-child td {{ border-bottom:none; }}
  .chart-wrap {{ background:#fff; border-radius:8px; padding:1rem;
                 box-shadow:0 1px 3px rgba(0,0,0,.1); margin:.5rem 0; }}
  code {{ font-size:.85rem; background:#e9ecef; padding:.1em .4em; border-radius:.25em; }}
</style>
</head>
<body>
<h1>Proxy VAE Evaluation Report</h1>
<p style="color:#6c757d;font-size:.9rem">
  Proxy: <code>{proxy_path}</code> &nbsp;|&nbsp; Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}
</p>

<h2>Summary</h2>
<div class="grid">
  <div class="card">
    <div class="label">Cosine Similarity</div>
    <div class="metric">{t1.get('cosine_sim', '—')}</div>
    {_badge(t1.get('pass_cosine'), 'target >0.95')}
  </div>
  <div class="card">
    <div class="label">Channel Std Ratio</div>
    <div class="metric">{t1.get('ch_std_ratio', '—')}</div>
    {_badge(t1.get('pass_ch_std'), 'target 0.95–1.05')}
  </div>
  <div class="card">
    <div class="label">FFT Correlation</div>
    <div class="metric">{t1.get('fft_corr', '—')}</div>
    {_badge(t1.get('pass_fft'), 'target >0.98')}
  </div>
  <div class="card">
    <div class="label">Decoded PSNR</div>
    <div class="metric">{t2.get('decoded_psnr_db', '—')} dB</div>
    {_badge(t2.get('pass_psnr'), 'target >35 dB') if 'pass_psnr' in t2 else _badge(None, 'no teacher')}
  </div>
  <div class="card">
    <div class="label">Mean Confidence</div>
    <div class="metric">{t3.get('mean_confidence', '—')}</div>
    {_badge(t3.get('mean_confidence', 0) > 0.80, 'target >0.80') if t3 else _badge(None)}
  </div>
  <div class="card">
    <div class="label">Fallback Rate @0.80</div>
    <div class="metric">{_fb80}</div>
    {_badge(None)}
  </div>
</div>

<h2>Tier 1 — Latent Quality ({t1.get('n_images', 0):,} images)</h2>
<table>
  <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
  <tr><td>Latent MSE (norm)</td><td>{t1.get('latent_mse_norm', '—')}</td><td>—</td></tr>
  <tr><td>Cosine similarity</td><td>{t1.get('cosine_sim', '—')}</td>
      <td>{_badge(t1.get('pass_cosine'))}</td></tr>
  <tr><td>Channel mean MAE</td><td>{t1.get('ch_mean_mae', '—')}</td><td>—</td></tr>
  <tr><td>Channel std ratio</td><td>{t1.get('ch_std_ratio', '—')}</td>
      <td>{_badge(t1.get('pass_ch_std'))}</td></tr>
  <tr><td>FFT magnitude correlation</td><td>{t1.get('fft_corr', '—')}</td>
      <td>{_badge(t1.get('pass_fft'))}</td></tr>
</table>
"""

    if t1_pmean:
        html += """<div class="chart-wrap">"""
        html += _bar_chart_js("ch_mean_chart", ch_labels,
                              [p - t for p, t in zip(t1_pmean, t1_tmean)],
                              "Per-channel mean error (proxy − teacher)", "#e76f51")
        html += "</div>"
        html += """<div class="chart-wrap">"""
        html += _bar_chart_js("ch_std_chart", ch_labels,
                              [p / (t + 1e-8) for p, t in zip(t1_pstd, t1_tstd)],
                              "Per-channel std ratio (proxy / teacher)", "#4e79a7")
        html += "</div>"

    if t2:
        html += f"""
<h2>Tier 2 — Decoded Quality ({t2.get('n_images', 0):,} images)</h2>
<table>
  <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
  <tr><td>Decoded MSE</td><td>{t2.get('decoded_mse', '—')}</td><td>—</td></tr>
  <tr><td>Decoded PSNR</td><td>{t2.get('decoded_psnr_db', '—')} dB</td>
      <td>{_badge(t2.get('pass_psnr'))}</td></tr>
  <tr><td>Decoded std ratio</td><td>{t2.get('decoded_std_ratio', '—')}</td>
      <td>{_badge(t2.get('pass_std_ratio'))}</td></tr>
</table>"""

    if t3:
        mse_pcts = t3.get("mse_percentiles", {})
        fb_sim   = t3.get("fallback_simulation", {})
        pct_rows = "".join(
            f'<tr><td>p{p}</td><td>{mse_pcts.get(f"p{p}", "—")}</td></tr>'
            for p in [50, 90, 95, 99]
        )
        fb_rows = "".join(
            f'<tr><td>{thr}</td><td>'
            f'{fb_sim.get(f"thr_{int(float(thr)*100)}", {}).get("fallback_rate", "—")}'
            f'</td></tr>'
            for thr in ["0.70", "0.75", "0.80", "0.85", "0.90"]
        )
        html += f"""
<h2>Tier 3 — Failure Analysis ({t3.get('n_images', 0):,} images)</h2>
<div class="grid">
  <table>
    <tr><th colspan="2">MSE Percentiles (norm)</th></tr>
    {pct_rows}
  </table>
  <table>
    <tr><th>Threshold</th><th>Fallback Rate</th></tr>
    {fb_rows}
  </table>
</div>"""

        if mse_hist["bins"]:
            html += """<div class="chart-wrap">"""
            html += _bar_chart_js("mse_hist", mse_hist["bins"], mse_hist["counts"],
                                  "Normalised MSE distribution (lower=better)")
            html += "</div>"

        if conf_hist["bins"]:
            html += """<div class="chart-wrap">"""
            html += _bar_chart_js("conf_hist", conf_hist["bins"], conf_hist["counts"],
                                  "Confidence score distribution (higher=better)", "#76b041")
            html += "</div>"

    html += """
<h2>Quality Mode Recommendation</h2>
<table>
  <tr><th>Mode</th><th>When to use</th><th>Overhead</th></tr>
  <tr><td><code>speed</code></td>
      <td>Cold-start warmup iterations; data with known good distribution</td>
      <td>None</td></tr>
  <tr><td><code>balanced</code></td>
      <td>Default for most production precompute runs</td>
      <td>~1ms/batch (channel stats)</td></tr>
  <tr><td><code>high_fidelity</code></td>
      <td>Final campaign runs; wikiart / OOD sources; validation generation</td>
      <td>~20ms/batch (partial decode)</td></tr>
</table>
</body></html>"""
    return html


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _print_tier(title: str, rows: list) -> None:
    print(f"\n[{title}]")
    for row in rows:
        if len(row) == 3:
            val, ok, note = row
            mark = "✓" if ok else "✗"
            print(f"  {mark} {val}  ({note})")
        else:
            print(f"    {row[0]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate proxy VAE quality")
    ap.add_argument("--proxy",        required=True)
    ap.add_argument("--shards",       default=None)
    ap.add_argument("--vae-cache",    default="/Volumes/16TBCold/precomputed/vae/current")
    ap.add_argument("--flux-model",   default=None)
    ap.add_argument("--campaign",     default=None,
                    help="Flywheel campaign name — evaluate on its shard pool")
    ap.add_argument("--quality-mode", default=None,
                    choices=["speed", "balanced", "high_fidelity"])
    ap.add_argument("--n-images",     type=int, default=2000)
    ap.add_argument("--tier",         type=int, default=0,
                    help="0=all, 1=tier1 only, 2=tier1+2, 3=all+failure")
    ap.add_argument("--report",       default=None,
                    help="Write HTML report to this path")
    ap.add_argument("--out",          default=None,
                    help="Write JSON results to this path")
    args = ap.parse_args()

    proxy, teacher = _load_proxy(args.proxy, args.quality_mode, args.flux_model)
    dataset, n_pairs = _build_dataset(
        args.shards, args.vae_cache, args.campaign, args.n_images)
    print(f"Dataset: {n_pairs:,} pairs available")

    results: dict = {}
    t0 = time.time()

    results["tier1"] = tier1_latent(proxy, dataset, args.n_images)

    if args.tier in (0, 2) and teacher is not None:
        results["tier2"] = tier2_decoded(proxy, teacher, dataset, args.n_images)

    if args.tier in (0, 3):
        results["tier3"] = tier3_analysis(proxy, dataset, args.n_images)

    print(f"\nTotal elapsed: {time.time()-t0:.0f}s")

    # Strip internal histogram keys from JSON output
    json_results = {
        tier: {k: v for k, v in data.items() if not k.startswith("_")}
        for tier, data in results.items()
    }
    if args.out:
        Path(args.out).write_text(json.dumps(json_results, indent=2))
        print(f"JSON results: {args.out}")

    if args.report:
        html = generate_html_report(results, args.proxy)
        Path(args.report).write_text(html)
        print(f"HTML report:  {args.report}")

    if not args.out and not args.report:
        print(json.dumps(json_results, indent=2))
