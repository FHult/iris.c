#!/usr/bin/env python3
"""
train/scripts/benchmark_vae_proxy.py — Proxy vs teacher VAE speed benchmark.

Measures encode latency (ms/image) for the proxy student encoder and, optionally,
the real Flux VAE, then reports the speedup ratio.

Why the ratio is contention-robust:
  Absolute ms/image is inflated when another process shares the GPU (e.g. a live
  flywheel precompute).  But if proxy and teacher are timed back-to-back under the
  same load, the *ratio* between them stays meaningful.  Always read the speedup
  number, not the absolute latency, when the GPU is busy.

Memory note:
  --with-teacher loads the full Flux VAE (~0.5 GB weights + conv activation peaks).
  Do NOT run --with-teacher concurrently with a flywheel/pipeline precompute on a
  32 GB machine — the combined VAE + Qwen3 + SigLIP residency can trigger jetsam.
  Run proxy-only during contention; add --with-teacher only when the GPU is idle.

Usage:
    # Proxy-only (safe during precompute) — measures each variant's latency:
    python train/scripts/benchmark_vae_proxy.py --variants small,default,medium

    # Full comparison (run only when GPU is idle):
    python train/scripts/benchmark_vae_proxy.py \\
        --with-teacher --flux-model flux-klein-model \\
        --variants default --batch 4 --iters 20

    # Benchmark a trained checkpoint (uses its real architecture + bf16):
    python train/scripts/benchmark_vae_proxy.py \\
        --proxy /Volumes/2TBSSD/checkpoints/vae_proxy/proxy_final.safetensors \\
        --with-teacher --flux-model flux-klein-model
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _gpu_busy_hint() -> str:
    """Best-effort note on whether another process likely holds the GPU."""
    try:
        from pipeline_lib import gpu_lock_holder
        holder = gpu_lock_holder()
        if holder:
            return (f"GPU lock held by '{holder.get('label','?')}' "
                    f"(PID {holder.get('pid','?')}) — absolute latencies are "
                    f"inflated; trust the speedup RATIO, not ms/img.")
    except Exception:
        pass
    return ""


def _time_encode(encode_fn, x, iters: int, warmup: int) -> float:
    """Return mean ms/image over `iters` calls after `warmup` warmup calls."""
    import mlx.core as mx
    batch = x.shape[0]
    for _ in range(warmup):
        mx.eval(encode_fn(x))
    t0 = time.time()
    for _ in range(iters):
        mx.eval(encode_fn(x))
    elapsed = time.time() - t0
    return (elapsed / iters / batch) * 1000.0


def _build_proxy_encoder(variant: str, proxy_path: str | None, bf16: bool, compile_it: bool):
    """Return (encode_fn, label) for a proxy variant or a loaded checkpoint."""
    import mlx.core as mx
    from vae_distill.student import build_student, PRESETS

    if proxy_path:
        from vae_distill.proxy import ProxyVAE
        proxy = ProxyVAE.load(proxy_path, quality_mode="speed")
        student = proxy._student
        label = f"proxy[{Path(proxy_path).name}]"
    else:
        student = build_student({"student": {"variant": variant}})
        label = f"proxy[{variant}:{student.param_count()/1e6:.1f}M]"

    student.eval()
    if bf16:
        student.to_bfloat16()
        label += "+bf16"
    mx.eval(student.parameters())

    if compile_it:
        enc = student.make_compiled()
        label += "+compiled"
    else:
        enc = lambda x: student(x)

    dtype = mx.bfloat16 if bf16 else mx.float32
    return enc, label, dtype


def main():
    ap = argparse.ArgumentParser(description="Benchmark proxy vs teacher VAE speed")
    ap.add_argument("--variants", default="default",
                    help="Comma-separated: small,default,medium (ignored if --proxy set)")
    ap.add_argument("--proxy", default=None,
                    help="Path to a trained proxy checkpoint (overrides --variants)")
    ap.add_argument("--with-teacher", action="store_true",
                    help="Also benchmark the real Flux VAE (needs idle GPU — see header)")
    ap.add_argument("--flux-model", default="flux-klein-model")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--no-bf16", action="store_true", help="Disable bf16 (use float32)")
    ap.add_argument("--no-compile", action="store_true", help="Disable mx.compile")
    ap.add_argument("--out", default=None, help="Write JSON results to this path")
    args = ap.parse_args()

    import mlx.core as mx

    hint = _gpu_busy_hint()
    if hint:
        print(f"⚠  {hint}\n")

    bf16     = not args.no_bf16
    compile_it = not args.no_compile
    B, S     = args.batch, args.image_size

    results = {
        "batch": B, "image_size": S, "iters": args.iters,
        "bf16": bf16, "compiled": compile_it,
        "gpu_contention": bool(hint),
        "proxy": {}, "teacher": None,
    }

    # ── Proxy variants ───────────────────────────────────────────────────────
    variants = [args.proxy] if args.proxy else args.variants.split(",")
    proxy_latencies: dict[str, float] = {}

    for v in variants:
        enc, label, dtype = _build_proxy_encoder(
            v if not args.proxy else None, args.proxy, bf16, compile_it)
        x = mx.zeros((B, 3, S, S)).astype(dtype)
        ms = _time_encode(enc, x, args.iters, args.warmup)
        proxy_latencies[label] = ms
        results["proxy"][label] = round(ms, 3)
        print(f"  {label:42s} {ms:7.2f} ms/img")
        if args.proxy:
            break

    # ── Teacher VAE ──────────────────────────────────────────────────────────
    if args.with_teacher:
        print("\nLoading teacher Flux VAE (this competes for GPU memory) ...")
        from vae_distill.teacher import TeacherEncoder
        teacher = TeacherEncoder.load(args.flux_model)
        # Teacher encode runs in float32 (matches precompute_all fast path)
        x_t = mx.zeros((B, 3, S, S)).astype(mx.float32)
        teacher_ms = _time_encode(lambda x: teacher.encode(x), x_t,
                                   args.iters, args.warmup)
        results["teacher"] = round(teacher_ms, 3)
        print(f"  {'teacher[Flux VAE fp32]':42s} {teacher_ms:7.2f} ms/img")

        print("\n── Speedup (teacher / proxy) ──")
        for label, ms in proxy_latencies.items():
            speedup = teacher_ms / ms if ms > 0 else 0.0
            target = "✓" if speedup >= 5.0 else "✗ (target ≥5×)"
            print(f"  {label:42s} {speedup:5.1f}×  {target}")
            results.setdefault("speedup", {})[label] = round(speedup, 2)
    else:
        print("\n(teacher not benchmarked — pass --with-teacher on an idle GPU "
              "for the speedup ratio)")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nJSON: {args.out}")

    return results


if __name__ == "__main__":
    main()
