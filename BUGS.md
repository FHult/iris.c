# Bugs and Anomalies

## C Inference Bugs (Open)

- **VAE-1: generic (non-BLAS) build VAE encode is catastrophically wrong.** Found via the
  C-1 parity harness (`debug/vae_parity.c`) on a real image (000004_3398) against the
  mflux teacher latent. Same source + inputs:
  - **BLAS/Accelerate build**: per-channel correlation **0.99906** vs the teacher encoder
    (encoders agree; the ~1.76× magnitude gap is the packed-BN vs VAE-latent convention,
    not an error).
  - **Generic `-O2` build (no `-DUSE_BLAS`)**: correlation **≈ −0.04** — the latent is
    essentially uncorrelated garbage.
  The pure-C conv/GEMM fallback path in `iris_vae.c` produces wrong results for VAE encode.
  Low operational priority (generic is the "very slow fallback"; production uses MPS/BLAS),
  but it means generic-build image quality is silently broken and the generic path is not
  a trustworthy reference. Repro: `debug/gen_vae_parity_fixture.py` then `debug/vae_parity.c`
  built each way. Needs root-cause in the non-BLAS conv/GEMM (candidate: an accumulation or
  layout assumption that only holds under the BLAS path).

## Training Bugs (Fixed)

- **VAE-Q1: IP-adapter trained in the wrong latent space (raw VAE-latent, not the
  BN'd packed space C inference uses). FIXED 2026-06-05.**
  - **Root cause (confirmed by 4 independent code paths):** precompute stores the mflux
    **VAE-latent** space (`encode()`, no BN, std≈1.72). Training used it directly
    (`latents = mx.array(vae_np)`; `_vae_encode → vae.encode()`) and packed via **patchify
    only, no BatchNorm**. But C inference operates the frozen Flux.2 transformer in the
    **BN'd packed space** (std≈1): txt2img noise init std‑1 at 128ch (iris.c:900); img2img
    uses BN'd `iris_vae_encode`; decode "denormalize (batch denorm) → unpatchify" — identical
    to mflux `decode_packed_latents`. So the adapter was trained against the frozen base
    receiving std≈1.72 latents while inference feeds it std≈1 — the trained adapter would not
    transfer to C inference. (Training loss/ref_gap still improved because the loss measures
    self-consistency within the wrong space, not transfer.)
  - **Fix:** BN-pack the loaded latent on load to match C exactly. `_load_vae_bn_stats` reads
    the VAE `bn.running_mean/running_var`; `_bn_pack_latents` applies per-128-feature BN with
    feature(c,h,w)=c*4+(h%2)*2+(w%2) (the patchify channel order) in [32,Lh,Lw] space, so the
    trainer's existing patchify pack yields exactly C `iris_vae_encode`'s output. Applied to
    cached, live, and validation latent paths. **No re-precompute needed** — raw latents are
    fine; the BN is applied on load.
  - **Validation:** `_bn_pack_latents(teacher)` vs C's real packed output (`debug/vae_parity.c`
    dump) → **corr 0.9995, std 0.972 vs 0.973** (through the real bf16 code path).
  - **Impact:** warmup-run2 (and any prior IP-adapter training) was in the wrong space and was
    restarted after this fix.

## Pipeline Bugs (Fixed)

- **PIPE-1: Multi-chunk precomputed file collision** — When chunks 2–4 ran `build_shards.py`
  without `--start-idx`, all chunks started shard numbering from 000000. Staging shards for chunk 2
  produced the same internal record IDs (e.g. `000000_0000`) as chunk 1. On promotion, chunk 2's
  `precomputed/000000_0000.npz` overwrote chunk 1's file in the shared production `precomputed/`
  directory, corrupting training cache. **Fix**: orchestrator now passes `--start-idx (chunk-1)*200000`
  to `build_shards.py` so each chunk occupies a disjoint shard ID space (chunk 1: 0–199999,
  chunk 2: 200000–399999, etc.), and shards are promoted without renaming to preserve the stem↔npz match.

## Training Anomalies (Chunk 1 — Observed, Not Actionable Now)

- **ANOMALY-1: Shard-boundary stalls** — Two blocking stalls observed at step ~19,900 (55 min) and ~24,900 (2.6h). Root cause: epoch boundary + simultaneous JDB chunk 2 conversion competing for 2TBSSD I/O. Data% jumps to 100% in timing log. Both resolved automatically. Structural until pixel data is pre-cached to disk. See BACKLOG PIPELINE-3.

- **ANOMALY-2: Optimizer step spikes** — Isolated step-time spikes at steps 19,300 (2.4s vs normal 0.1s) and 25,900 (3.5s). Likely: gradient norm exceeded clip threshold (1.0) triggering full-parameter norm computation, or MLX lazy evaluator GC. No functional impact.
