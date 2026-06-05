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

## Open Questions (Train/Inference latent-convention — needs confirmation)

- **VAE-Q1: does IP-adapter training pack latents in the same BN convention the frozen
  Flux.2 transformer / C inference expect?** The precompute stores the mflux **VAE-latent**
  space (`encode()` = `(mean-shift)*scale`, no BN, std≈1.72). Training packs it to the
  transformer via **patchify only, no BatchNorm** (`train_ip_adapter.py:2508-2511`), whereas
  mflux's `decode_packed_latents` defines the transformer/packed space as BN'd
  (`packed*bn_std + bn_mean`, std≈1) and C's `iris_vae_encode` applies that BN. If the frozen
  base transformer expects BN'd packed latents, training feeds it std≈1.72 (un-BN'd) latents —
  a train/inference latent-scale mismatch. Could equally be correct if the base variant
  operates on un-BN'd latents. Resolve by tracing the C txt2img denoising convention + the
  frozen transformer's expected input space before treating as a bug. This is the genuine
  C-1-class risk the parity harness surfaced.

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
