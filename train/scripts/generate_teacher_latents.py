#!/usr/bin/env python3
"""
train/scripts/generate_teacher_latents.py — Corpus preparation for proxy VAE training.

Selects high-quality shards from the cold shard pool (filtered by unified_score),
verifies that teacher VAE latents exist for them, and writes a training manifest
JSON that train_vae_proxy.py can use directly.

If a shard has no pre-computed latents yet, the script can optionally launch
precompute_all.py on just the VAE pass for those shards (--run-precompute flag).
This is useful for bootstrapping the first proxy training run before the full
flywheel precompute has completed.

Usage:
    # Just emit a manifest of ready shards (dry run, no compute):
    python train/scripts/generate_teacher_latents.py \\
        --shards /Volumes/16TBCold/shards \\
        --vae-cache /Volumes/16TBCold/precomputed/vae/current \\
        --out /Volumes/2TBSSD/vae_distill_manifest.json \\
        --min-score 0.5 \\
        --target-images 150000

    # Also compute missing latents for selected shards:
    python train/scripts/generate_teacher_latents.py \\
        --shards /Volumes/16TBCold/shards \\
        --vae-cache /Volumes/16TBCold/precomputed/vae/current \\
        --out /Volumes/2TBSSD/vae_distill_manifest.json \\
        --run-precompute \\
        --flux-model flux-klein-model
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional


def _load_unified_score(shard: Path) -> Optional[float]:
    score_file = shard.with_suffix(".unified_score.json")
    if score_file.exists():
        try:
            return float(json.loads(score_file.read_text()).get("final_value_score", 0.0))
        except Exception:
            pass
    light_file = shard.with_suffix(".light_scores.json")
    if light_file.exists():
        try:
            return float(json.loads(light_file.read_text()).get("combined_score", 0.0))
        except Exception:
            pass
    return None


def _count_cached_latents(shard_stem: str, vae_cache: Path) -> int:
    """Count how many .npz files in vae_cache belong to this shard."""
    return len(list(vae_cache.glob(f"{shard_stem}_*.npz")))


def generate_manifest(
    shards_dir: str,
    vae_cache_dir: str,
    out_path: str,
    min_score: float = 0.5,
    target_images: int = 150_000,
    run_precompute: bool = False,
    flux_model: Optional[str] = None,
) -> dict:
    shards_p    = Path(shards_dir)
    vae_cache_p = Path(vae_cache_dir)
    vae_cache_p.mkdir(parents=True, exist_ok=True)

    all_tars = sorted(shards_p.glob("*.tar"))
    print(f"Found {len(all_tars):,} shards in {shards_dir}")

    # Score and sort shards
    scored: list[tuple[float, Path]] = []
    unscored: list[Path] = []
    for t in all_tars:
        s = _load_unified_score(t)
        if s is not None:
            scored.append((s, t))
        else:
            unscored.append(t)

    scored.sort(reverse=True)
    # Include unscored after scored (unknown quality — still useful for diversity)
    ranked = [p for _, p in scored] + unscored

    selected: list[dict] = []
    total_images = 0
    precompute_needed: list[Path] = []

    for shard in ranked:
        if total_images >= target_images:
            break
        if min_score > 0:
            score = _load_unified_score(shard)
            if score is not None and score < min_score:
                continue

        n_cached = _count_cached_latents(shard.stem, vae_cache_p)
        has_latents = n_cached > 0

        if not has_latents:
            if run_precompute:
                precompute_needed.append(shard)
            else:
                continue  # skip shards without cached latents

        selected.append({
            "shard":        str(shard),
            "stem":         shard.stem,
            "score":        _load_unified_score(shard),
            "cached_npz":   n_cached,
        })
        total_images += max(n_cached, 5000)  # estimate 5000 if precompute pending

    print(f"Selected {len(selected):,} shards "
          f"({total_images:,} images estimated), "
          f"{len(precompute_needed):,} need precompute")

    # Run precompute_all.py (VAE-only) for shards that need it
    if precompute_needed and run_precompute:
        if not flux_model:
            print("ERROR: --flux-model required with --run-precompute", file=sys.stderr)
            sys.exit(1)
        _run_precompute_vae(precompute_needed, vae_cache_p, flux_model)

    # Write manifest
    manifest = {
        "version":        1,
        "shards_dir":     shards_dir,
        "vae_cache_dir":  vae_cache_dir,
        "min_score":      min_score,
        "target_images":  target_images,
        "n_shards":       len(selected),
        "shards":         selected,
    }
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest written: {out_path}")
    return manifest


def _run_precompute_vae(
    shards: list[Path],
    vae_cache: Path,
    flux_model: str,
) -> None:
    """Run precompute_all.py in VAE-only mode on a list of shards."""
    scripts_dir = Path(__file__).parent
    venv_python = scripts_dir.parent / ".venv" / "bin" / "python"

    # Write a temporary shards dir with symlinks for just these shards
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir, \
         tempfile.TemporaryDirectory() as qwen3_tmp, \
         tempfile.TemporaryDirectory() as siglip_tmp:
        for s in shards:
            link = Path(tmpdir) / s.name
            os.symlink(s, link)

        # Pass real temp dirs for qwen3/siglip — precompute_all.py calls
        # os.makedirs() on all output paths, so /dev/null would crash it.
        # The outputs are discarded; only vae_cache receives keeper files.
        cmd = [
            str(venv_python), "-u",
            str(scripts_dir / "precompute_all.py"),
            "--shards",       tmpdir,
            "--vae-output",   str(vae_cache),
            "--qwen3-output", qwen3_tmp,
            "--siglip-output", siglip_tmp,
        ]
        if (Path(flux_model)).exists():
            cmd += ["--flux-model", flux_model]

        print(f"  Running VAE-only precompute for {len(shards)} shards ...")
        env = {**os.environ, "PIPELINE_ORCHESTRATED": "1"}
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"  WARNING: precompute exited {result.returncode}", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prepare proxy VAE training corpus")
    ap.add_argument("--shards",        required=True)
    ap.add_argument("--vae-cache",     required=True)
    ap.add_argument("--out",           required=True, help="Output manifest path")
    ap.add_argument("--min-score",     type=float, default=0.5)
    ap.add_argument("--target-images", type=int, default=150_000)
    ap.add_argument("--run-precompute",action="store_true",
                    help="Run VAE precompute for shards missing latents")
    ap.add_argument("--flux-model",    default="flux-klein-model",
                    help="Flux Klein model path (required with --run-precompute)")
    args = ap.parse_args()

    generate_manifest(
        shards_dir     = args.shards,
        vae_cache_dir  = args.vae_cache,
        out_path       = args.out,
        min_score      = args.min_score,
        target_images  = args.target_images,
        run_precompute = args.run_precompute,
        flux_model     = args.flux_model,
    )
