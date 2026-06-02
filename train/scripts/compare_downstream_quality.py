#!/usr/bin/env python3
"""
train/scripts/compare_downstream_quality.py — Downstream IP-Adapter A/B harness.

The definitive proxy VAE quality test (Tier 2 of plans/precomp2-proxy-vae-design.md):
train two IP-Adapter runs that differ ONLY in their VAE latents — one set encoded
by the real Flux VAE, one by the proxy — on identical shards, seed, and steps,
then compare final training quality.

This answers the question that latent-space metrics cannot: does a model trained
on proxy latents reach the same quality as one trained on real latents?

Procedure:
  1. Pick N shards (from --shards or a --campaign's shard pool).
  2. Ensure REAL VAE latents exist for them (precompute_all without --proxy-vae).
  3. Compute PROXY VAE latents into a separate cache (precompute_all --proxy-vae).
  4. Run train_ip_adapter.py twice with identical config + seed, differing only in
     data.vae_cache_dir (real vs proxy). Qwen3/SigLIP caches are shared.
  5. Parse final metrics (loss, cond_gap, ref_gap) from both logs and compare.

GPU exclusivity:
  This launches GPU-heavy training. It refuses to run if the GPU lock is held
  (e.g. by a live flywheel/pipeline) unless --force is passed. Run it when the
  GPU is free, or on a machine dedicated to evaluation.

Usage:
    python train/scripts/compare_downstream_quality.py \\
        --proxy /Volumes/2TBSSD/checkpoints/vae_proxy/proxy_final.safetensors \\
        --shards /Volumes/16TBCold/shards \\
        --base-config train/configs/stage1_512px.yaml \\
        --flux-model flux-klein-model \\
        --n-shards 4 --steps 500 --seed 1234 \\
        --workdir /Volumes/2TBSSD/proxy_ab \\
        --out /Volumes/2TBSSD/proxy_ab/result.json

A "pass" means proxy-run final cond_gap is within --tolerance of the real-run
cond_gap (default 5% relative). Larger cond_gap is better.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml

from flywheel_lib import collect_metrics_from_log
from pipeline_lib import gpu_lock_holder, TRAIN_DIR, SCRIPTS_DIR


def _select_shards(shards_dir: str, campaign: str | None, n: int) -> list[str]:
    """Pick n shard tar paths, preferring a campaign's pool if given."""
    all_tars = sorted(Path(shards_dir).glob("*.tar"))
    if not all_tars:
        raise RuntimeError(f"No .tar shards in {shards_dir}")

    if campaign:
        try:
            from flywheel_lib import FlywheelDB
            db = FlywheelDB()
            wanted: set[str] = set()
            for row in db.get_iterations(campaign):
                for sid in (row.get("shard_ids") or []):
                    wanted.add(str(sid))
            picked = [t for t in all_tars if t.stem in wanted][:n]
            if picked:
                print(f"  Selected {len(picked)} shards from campaign '{campaign}'")
                return [str(p) for p in picked]
            print(f"  Campaign '{campaign}' shards not found in pool; using first {n}")
        except Exception as e:
            print(f"  Campaign lookup failed ({e}); using first {n} shards")

    return [str(p) for p in all_tars[:n]]


def _stage_shards(shard_paths: list[str], dest: Path) -> None:
    """Symlink selected shards into a flat dir for precompute_all --shards."""
    dest.mkdir(parents=True, exist_ok=True)
    for sp in shard_paths:
        link = dest / Path(sp).name
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(sp, link)


def _run(cmd: list[str], log_path: Path, env: dict) -> int:
    """Run a subprocess, tee to log_path, append EXIT_CODE, return it."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    with open(log_path, "a") as logf:
        logf.write(f"\nEXIT_CODE={proc.returncode}\n")
    return proc.returncode


def _precompute(shard_dir: Path, q_out: Path, v_out: Path, s_out: Path,
                flux_model: str, proxy_path: str | None, log_path: Path,
                venv_py: str) -> int:
    """Run precompute_all for one VAE arm (real if proxy_path None, else proxy)."""
    for d in (q_out, v_out, s_out):
        d.mkdir(parents=True, exist_ok=True)
    cmd = [
        venv_py, "-u", str(SCRIPTS_DIR / "precompute_all.py"),
        "--shards", str(shard_dir),
        "--qwen3-output", str(q_out),
        "--vae-output",   str(v_out),
        "--siglip-output", str(s_out),
        "--siglip",
    ]
    if Path(flux_model).exists():
        cmd += ["--flux-model", flux_model]
    if proxy_path:
        cmd += ["--proxy-vae", proxy_path, "--proxy-mode", "speed"]
    env = {**os.environ, "PIPELINE_ORCHESTRATED": "1"}
    return _run(cmd, log_path, env)


def _train_arm(base_config: dict, shard_dir: Path, q_dir: Path, v_dir: Path,
               s_dir: Path, steps: int, seed: int, ckpt_dir: Path,
               cfg_path: Path, log_path: Path, venv_py: str,
               data_root: str) -> int:
    """Run one train_ip_adapter arm with the given VAE cache dir."""
    cfg = dict(base_config)
    data = dict(cfg.get("data", {}))
    data["shard_path"]      = str(shard_dir)
    data["qwen3_cache_dir"] = str(q_dir)
    data["vae_cache_dir"]   = str(v_dir)
    data["siglip_cache_dir"] = str(s_dir)
    data["anchor_shard_dir"] = None
    data["hard_example_dir"] = None
    cfg["data"] = data

    out = dict(cfg.get("output", {}))
    out["checkpoint_dir"] = str(ckpt_dir)
    out["log_every"]      = max(10, steps // 10)
    out["checkpoint_every"] = steps          # one checkpoint at the end
    cfg["output"] = out

    train = dict(cfg.get("training", {}))
    train["num_steps"]    = steps
    train["warmup_steps"] = min(train.get("warmup_steps", 100), max(1, steps // 10))
    # train_ip_adapter reads training.seed (no --seed CLI flag); set it here so
    # both arms share identical RNG state — the only difference is the VAE cache.
    train["seed"]         = seed
    cfg["training"] = train

    cfg_path.write_text(yaml.dump(cfg, default_flow_style=False))

    cmd = [
        venv_py, "-u", str(TRAIN_DIR / "train_ip_adapter.py"),
        "--config", str(cfg_path),
        "--max-steps", str(steps),
        "--data-root", data_root,
    ]
    env = {**os.environ, "PIPELINE_ORCHESTRATED": "1"}
    return _run(cmd, log_path, env)


def main():
    ap = argparse.ArgumentParser(description="Downstream IP-Adapter proxy-vs-real A/B")
    ap.add_argument("--proxy",       required=True, help="Trained proxy checkpoint")
    ap.add_argument("--shards",      default="/Volumes/16TBCold/shards")
    ap.add_argument("--campaign",    default=None, help="Restrict to a campaign's shard pool")
    ap.add_argument("--base-config", default="train/configs/stage1_512px.yaml")
    ap.add_argument("--flux-model",  default="flux-klein-model")
    ap.add_argument("--n-shards",    type=int, default=4)
    ap.add_argument("--steps",       type=int, default=500)
    ap.add_argument("--seed",        type=int, default=1234)
    ap.add_argument("--workdir",     default="/Volumes/2TBSSD/proxy_ab")
    ap.add_argument("--data-root",   default="/Volumes/2TBSSD")
    ap.add_argument("--tolerance",   type=float, default=0.05,
                    help="Max relative cond_gap shortfall to still PASS (default 5%%)")
    ap.add_argument("--out",         default=None)
    ap.add_argument("--force",       action="store_true",
                    help="Run even if the GPU lock is held (NOT recommended)")
    ap.add_argument("--skip-precompute", action="store_true",
                    help="Reuse existing real/proxy caches in workdir (resume)")
    args = ap.parse_args()

    # GPU exclusivity guard
    holder = gpu_lock_holder()
    if holder and not args.force:
        print(f"ERROR: GPU lock held by '{holder.get('label','?')}' "
              f"(PID {holder.get('pid','?')}). This harness needs the GPU. "
              f"Wait for it to free, or pass --force.", file=sys.stderr)
        sys.exit(1)

    if not Path(args.proxy).exists():
        print(f"ERROR: proxy checkpoint not found: {args.proxy}", file=sys.stderr)
        sys.exit(1)

    venv_py = str(TRAIN_DIR / ".venv" / "bin" / "python")
    work = Path(args.workdir)
    shard_dir = work / "shards"
    real = work / "real"       # real VAE latents + shared qwen3/siglip
    prox = work / "proxy"      # proxy VAE latents
    base_config = yaml.safe_load(open(args.base_config))

    print(f"=== Downstream A/B: proxy vs real VAE ===")
    print(f"  proxy:   {args.proxy}")
    print(f"  shards:  {args.n_shards}  steps: {args.steps}  seed: {args.seed}")
    print(f"  workdir: {work}\n")

    # 1. Select + stage shards
    shard_paths = _select_shards(args.shards, args.campaign, args.n_shards)
    _stage_shards(shard_paths, shard_dir)

    # 2. + 3. Precompute both arms.
    #    Real arm computes qwen3 + siglip + real VAE.
    #    Proxy arm reuses the shared qwen3 + siglip, computes only proxy VAE.
    q_dir = real / "qwen3"
    s_dir = real / "siglip"
    real_v = real / "vae"
    prox_v = prox / "vae"

    if not args.skip_precompute:
        print("[1/4] Precomputing REAL VAE + qwen3 + siglip ...")
        rc = _precompute(shard_dir, q_dir, real_v, s_dir,
                         args.flux_model, None, work / "precompute_real.log", venv_py)
        if rc != 0:
            print(f"ERROR: real precompute failed (exit {rc})", file=sys.stderr)
            sys.exit(1)

        print("[2/4] Precomputing PROXY VAE (reusing qwen3/siglip) ...")
        # Point qwen3/siglip outputs at the same dirs so they are skipped (already present)
        rc = _precompute(shard_dir, q_dir, prox_v, s_dir,
                         args.flux_model, args.proxy, work / "precompute_proxy.log", venv_py)
        if rc != 0:
            print(f"ERROR: proxy precompute failed (exit {rc})", file=sys.stderr)
            sys.exit(1)
    else:
        print("[1-2/4] Skipping precompute (--skip-precompute)")

    # 4. Train both arms with identical config + seed.
    print(f"[3/4] Training REAL arm ({args.steps} steps) ...")
    t0 = time.time()
    rc_real = _train_arm(base_config, shard_dir, q_dir, real_v, s_dir,
                         args.steps, args.seed, real / "ckpt",
                         work / "train_real.yaml", work / "train_real.log",
                         venv_py, args.data_root)
    real_secs = time.time() - t0

    print(f"[4/4] Training PROXY arm ({args.steps} steps) ...")
    t0 = time.time()
    rc_prox = _train_arm(base_config, shard_dir, q_dir, prox_v, s_dir,
                         args.steps, args.seed, prox / "ckpt",
                         work / "train_proxy.yaml", work / "train_proxy.log",
                         venv_py, args.data_root)
    prox_secs = time.time() - t0

    # Parse + compare
    real_m = collect_metrics_from_log(work / "train_real.log")
    prox_m = collect_metrics_from_log(work / "train_proxy.log")

    def _cmp(key: str) -> dict:
        rv, pv = real_m.get(key), prox_m.get(key)
        rel = None
        if rv is not None and pv is not None and rv != 0:
            rel = round((pv - rv) / abs(rv), 4)
        return {"real": rv, "proxy": pv, "rel_delta": rel}

    cond = _cmp("cond_gap")
    # PASS if proxy cond_gap within tolerance below real (larger cond_gap = better).
    verdict = "UNKNOWN"
    if cond["real"] is not None and cond["proxy"] is not None:
        shortfall = (cond["real"] - cond["proxy"]) / abs(cond["real"]) if cond["real"] else 0.0
        verdict = "PASS" if shortfall <= args.tolerance else "FAIL"

    result = {
        "proxy_checkpoint": args.proxy,
        "n_shards": args.n_shards, "steps": args.steps, "seed": args.seed,
        "exit_codes": {"real": rc_real, "proxy": rc_prox},
        "train_secs": {"real": round(real_secs), "proxy": round(prox_secs)},
        "metrics": {
            "cond_gap":    cond,
            "ref_gap":     _cmp("ref_gap"),
            "loss_smooth": _cmp("loss_smooth"),
            "loss_cond":   _cmp("loss_cond"),
        },
        "tolerance": args.tolerance,
        "verdict": verdict,
    }

    print("\n=== Result ===")
    print(f"  cond_gap   real={cond['real']}  proxy={cond['proxy']}  "
          f"rel_delta={cond['rel_delta']}")
    print(f"  ref_gap    {result['metrics']['ref_gap']}")
    print(f"  VERDICT: {verdict}  (tolerance={args.tolerance:.0%})")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"  JSON: {args.out}")

    sys.exit(0 if verdict in ("PASS", "UNKNOWN") else 2)


if __name__ == "__main__":
    main()
