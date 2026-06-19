#!/usr/bin/env python
"""
sref_dataset_campaign.py — direct-trainer dataset-quantity campaign (wedge-robust).

Trains N from-scratch IP-Adapter arms on nested shard subsets of a cached pool
(data-quantity ladder), exports each, and scores each as an ip-scale FRONTIER
(sref_sweep_eval.py). The point: see whether MORE data improves CSD style_sim,
and whether the cheap held-out cond_gap (the training-internal surrogate) tracks
the real style metric (BACKLOG SREF-METRIC-1 validity question).

Runs the DIRECT single-process trainer (not the flywheel) to avoid the MLX wedge
(BUGS MLX-2, flywheel ~5/5 wedge vs direct ~3/4 clean). Each arm is monitored:
no step progress for --stall-secs ⇒ treat as wedge ⇒ kill + retry (≤ --retries).

Resumable: an arm whose frontier.json already exists is skipped; a finished
checkpoint is reused (no retrain) if present.

Outputs under /Volumes/2TBSSD/sref_eval/campaign/:
  pool_<n>/                nested shard-subset symlink dir
  arm_<n>/config.yaml      generated train config
  arm_<n>/train.log        trainer stdout
  arm_<n>/ckpt/            checkpoints (best.safetensors = EMA)
  arm_<n>/bundle/          exported iris bundle
  ../<name>_arm<n>/...      sref_sweep_eval run dir (gens, scores, frontier)
  campaign_report.json     per-arm cond_gap + frontier summary

Usage:
  caffeinate -i train/.venv/bin/python train/scripts/sref_dataset_campaign.py \
      --arms 4,22,12,8,16 --steps 3000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
VENV = REPO / "train" / ".venv" / "bin" / "python"
BASE_CFG = REPO / "train" / "configs" / "stage1_512px.yaml"
TRAINER = REPO / "train" / "train_ip_adapter.py"
EXPORTER = REPO / "train" / "export" / "export_adapter.py"
SWEEP = REPO / "train" / "scripts" / "sref_sweep_eval.py"
POOL = Path("/Volumes/2TBSSD/baseline_pool_hot")  # tars MUST be on hot SSD: the loader
# enumerates every shard tar before the first sample, and a cold-storage scan is ~31s/shard
# (vs ~0.8s hot). >3 cold shards blows past the loader's 120s sample_q timeout (BUGS/anchor).
CACHE = {
    "vae": "/Volumes/2TBSSD/precomputed/vae/v_2232c1",
    "qwen3": "/Volumes/2TBSSD/precomputed/qwen3/v_059443",
    "siglip": "/Volumes/2TBSSD/precomputed/siglip/v_336c6e",
}
ROOT = Path("/Volumes/2TBSSD/sref_eval/campaign")
PERCEIVER_HEADS = 16


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def build_pool(n: int, shards: list[str]) -> Path:
    d = ROOT / f"pool_{n}"
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.tar"):
        f.unlink()
    for s in shards[:n]:
        (d / os.path.basename(s)).symlink_to(s)
    return d


def build_cfg(n: int, pool: Path, steps: int) -> tuple[Path, Path]:
    cfg = yaml.safe_load(BASE_CFG.read_text())
    cfg["data"]["shard_path"] = str(pool)
    cfg["data"]["bucket"] = [512, 512]
    for e, p in CACHE.items():
        cfg["data"][f"{e}_cache_dir"] = p
    cfg["data"]["anchor_shard_dir"] = None
    cfg["data"]["hard_example_dir"] = None
    cfg["data"]["hard_mix_ratio"] = 0.0
    cfg["training"]["num_steps"] = steps
    cfg["training"]["warmup_steps"] = max(1, min(cfg["training"].get("warmup_steps", 1000), steps // 10))
    cfg["training"]["mlx_memory_pct"] = 0.6
    cfg["training"]["val_every"] = steps  # skip mid-training val; final cond_gap computed at end
    ck = ROOT / f"arm_{n}" / "ckpt"
    ck.mkdir(parents=True, exist_ok=True)
    cfg["output"]["checkpoint_dir"] = str(ck)
    cfg["output"]["checkpoint_every"] = steps
    cfg["output"]["keep_last_n"] = 1
    cfg["output"]["log_every"] = 100
    cfg.setdefault("eval", {})["enabled"] = False
    p = ROOT / f"arm_{n}" / "config.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.dump(cfg, default_flow_style=False))
    return p, ck


def train_arm(n: int, cfg: Path, steps: int, stall_secs: int, retries: int) -> bool:
    log_f = ROOT / f"arm_{n}" / "train.log"
    for attempt in range(retries + 1):
        log(f"arm {n}: train attempt {attempt + 1}/{retries + 1}  (steps={steps}) -> {log_f}")
        fh = open(log_f, "w")
        proc = subprocess.Popen(
            [str(VENV), "-u", str(TRAINER), "--config", str(cfg),
             "--max-steps", str(steps), "--run-name", f"campaign_arm{n}"],
            stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO),
            preexec_fn=os.setsid,
        )
        last_step, last_prog, ok = -1, time.time(), False
        while True:
            rc = proc.poll()
            txt = log_f.read_text(errors="ignore") if log_f.exists() else ""
            if "Training complete" in txt:
                ok = True
                break
            m = re.findall(r"step\s+(\d+)/", txt)
            cur = int(m[-1]) if m else 0
            if cur > last_step:
                last_step, last_prog = cur, time.time()
            if rc is not None:
                ok = "Training complete" in txt
                break
            if time.time() - last_prog > stall_secs:
                log(f"arm {n}: STALL — {int(time.time()-last_prog)}s no progress at step {last_step}; kill+retry")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=30)
                break
            time.sleep(15)
        fh.close()
        if ok:
            log(f"arm {n}: training complete at step {last_step}")
            return True
        log(f"arm {n}: attempt {attempt + 1} failed (last step {last_step})")
        time.sleep(5)
    log(f"arm {n}: GAVE UP after {retries + 1} attempts")
    return False


def parse_cond_gap(n: int) -> float | None:
    log_f = ROOT / f"arm_{n}" / "train.log"
    if not log_f.exists():
        return None
    m = re.findall(r"cond_gap=([+-]?\d+\.\d+)", log_f.read_text(errors="ignore"))
    return float(m[-1]) if m else None


def export_arm(n: int, ck: Path) -> Path | None:
    bundle = ROOT / f"arm_{n}" / "bundle"
    # Try best.safetensors (bare EMA keys) first; fall back to the final step
    # checkpoint with --use-ema (loads ema.-prefixed weights). Either yields the EMA.
    attempts: list[tuple[Path, list[str]]] = []
    best = ck / "best.safetensors"
    if best.exists():
        attempts.append((best, []))
    steps = sorted(ck.glob("step_*.safetensors"))
    if steps:
        attempts.append((steps[-1], ["--use-ema"]))
    if not attempts:
        log(f"arm {n}: no checkpoint to export")
        return None
    logf = ROOT / f"arm_{n}" / "export.log"
    for ckpt, extra in attempts:
        r = subprocess.run(
            [str(VENV), str(EXPORTER), "--checkpoint", str(ckpt), "--output", str(bundle),
             "--perceiver-heads", str(PERCEIVER_HEADS), "--validate", *extra],
            cwd=str(REPO), capture_output=True, text=True,
        )
        with open(logf, "a") as fh:
            fh.write(f"\n=== export {ckpt.name} {extra} rc={r.returncode} ===\n{r.stdout}\n{r.stderr}\n")
        if r.returncode == 0 and (bundle / "adapter_meta.json").exists():
            log(f"arm {n}: exported bundle from {ckpt.name} {extra}")
            return bundle
        log(f"arm {n}: export from {ckpt.name} {extra} failed (rc={r.returncode}) — trying next")
    log(f"arm {n}: export FAILED on all checkpoints — see export.log")
    return None


def eval_arm(n: int, bundle: Path, name: str, scales: str, seeds: str) -> dict | None:
    run = f"{name}_arm{n}"
    r = subprocess.run(
        [str(VENV), "-u", str(SWEEP), "--bundle", str(bundle), "--name", run,
         "--scales", scales, "--seeds", seeds],
        cwd=str(REPO),
    )
    fr = Path("/Volumes/2TBSSD/sref_eval") / run / "frontier.json"
    if r.returncode != 0 or not fr.exists():
        log(f"arm {n}: eval FAILED")
        return None
    return json.loads(fr.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=str(POOL), help="shard pool dir (tars MUST be on hot SSD)")
    ap.add_argument("--arms", default="4,22,12,8,16", help="shard counts, evaluated in this order")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--scales", default="0.3,0.5,0.7")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--name", default="campaign")
    ap.add_argument("--stall-secs", type=int, default=900)
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)
    pool_dir = Path(args.pool)
    all_shards = sorted(str(p) for p in pool_dir.glob("*.tar"))
    if not all_shards:
        log(f"FATAL: no shards in {pool_dir}")
        return 1
    arms = [int(x) for x in args.arms.split(",") if x.strip()]
    arms = [a for a in arms if a <= len(all_shards)]
    log(f"campaign start: arms={arms} steps={args.steps} pool={len(all_shards)} shards")

    report_path = ROOT / "campaign_report.json"
    results: dict = json.loads(report_path.read_text()) if report_path.exists() else {}

    for n in arms:
        key = f"arm_{n}"
        fr_existing = Path("/Volumes/2TBSSD/sref_eval") / f"{args.name}_arm{n}" / "frontier.json"
        if fr_existing.exists():
            log(f"arm {n}: frontier exists — skipping (resumable)")
            results[key] = {**results.get(key, {}), "n_shards": n,
                            "frontier": json.loads(fr_existing.read_text()),
                            "cond_gap": parse_cond_gap(n)}
            report_path.write_text(json.dumps(results, indent=2))
            continue

        t0 = time.time()
        pool = build_pool(n, all_shards)
        cfg, ck = build_cfg(n, pool, args.steps)
        log(f"arm {n}: {n} shards -> {pool}")

        if (ck / "best.safetensors").exists():
            log(f"arm {n}: best.safetensors present — reusing (no retrain)")
            trained = True
        else:
            trained = train_arm(n, cfg, args.steps, args.stall_secs, args.retries)

        rec: dict = {"n_shards": n, "trained": trained,
                     "cond_gap": parse_cond_gap(n) if trained else None,
                     "train_secs": round(time.time() - t0)}
        if trained:
            bundle = export_arm(n, ck)
            if bundle:
                fr = eval_arm(n, bundle, args.name, args.scales, args.seeds)
                rec["frontier"] = fr
        results[key] = rec
        report_path.write_text(json.dumps(results, indent=2))
        log(f"arm {n}: done in {rec['train_secs']}s  cond_gap={rec.get('cond_gap')}")

    # Summary table
    log("=== CAMPAIGN SUMMARY ===")
    log(f"{'arm':>5} {'shards':>6} {'cond_gap':>9}  best-scale style_sim / leak / sref")
    for n in arms:
        r = results.get(f"arm_{n}", {})
        fr = r.get("frontier") or {}
        best = None
        for sc, a in fr.items():
            if a.get("sref_score") is not None and (best is None or a["sref_score"] > best[1].get("sref_score", -9)):
                best = (sc, a)
        cg = r.get("cond_gap")
        cg_s = f"{cg:+.3f}" if isinstance(cg, (int, float)) else "  —  "
        if best:
            sc, a = best
            log(f"{n:>5} {r.get('n_shards', n):>6} {cg_s:>9}  @{sc}: "
                f"style {a.get('style_sim')} / leak {a.get('content_leak')} / sref {a.get('sref_score')}")
        else:
            log(f"{n:>5} {r.get('n_shards', n):>6} {cg_s:>9}  (no frontier)")
    log(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
