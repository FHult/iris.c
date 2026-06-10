# Bugs and Anomalies

## C Inference Bugs (Fixed)

- **VAE-1: generic (non-BLAS) build VAE encode was catastrophically wrong. FIXED 2026-06-06.**
  - **Symptom:** generic `-O2` build's VAE encode correlated **≈ −0.04** with the teacher,
    while the BLAS build correlated **0.99906** (same source + inputs). Found via the parity
    harness (`debug/vae_parity.c`) on image 000004_3398.
  - **Root cause (buffer aliasing, not a kernel bug):** in `resblock_forward`, `conv1_out`
    was placed at `work + in_ch*spatial`. After `norm2` writes the `out_ch`-channel result
    back to `work[0..out_ch*spatial)`, `conv2` reads that while writing `conv1_out`. For a
    channel-*increasing* resblock (`in_ch < out_ch`, e.g. the encoder's 128→256 blocks) the
    input and output **overlap**. The BLAS `iris_conv2d` tolerates it (im2col copies the
    input first); the naive CPU conv reads partially-overwritten input → corruption. Localized
    by per-stage checksums to L1_resblk0 → conv2. Decoder resblocks (`in_ch > out_ch`) were
    unaffected.
  - **Fix:** place `conv1_out` at `work + max(in_ch,out_ch)*spatial` so neither conv1 nor
    conv2 ever aliases its output. Buffer bound unchanged (mid block's in==out==512 already
    needs 2·512·spatial); identical for `in_ch==out_ch` and for the decoder. After the fix,
    generic and BLAS encode are **bit-identical (corr 1.000000)** and BLAS↔teacher stays
    0.99906. Regression guard: `test_encode_golden` in `debug/test_vae.c` (pins the
    channel-increasing encode path; fails by diff ~118 if the aliasing returns).

- **QWEN-1: text-embedding pad rows were zeroed, regressing every generation. FIXED 2026-06-10.**
  - **Symptom:** `make test` golden comparisons failed broadly (txt2img mean_diff 43–52 vs
    threshold 20; img2img 57.5). Bisect: green at `76564f8`, red at `ee039d6` — culprit
    `ffecfcc` (2026-05-16, INFER-H-001), which zeroed all `qwen3_encode_text_ex` embedding
    rows beyond the real tokens "to match training convention."
  - **Why it's wrong:** the reference implementation (mflux `Flux2PromptEncoder` →
    `Qwen3TextEncoder.get_prompt_embeds`) returns the stacked hidden states **unmodified** —
    pad queries attend to real tokens under the encoder's padding mask, so pad rows carry
    non-zero prompt-derived state that flows into the transformer. Zeroed rows are NOT a
    mask (a zero K-row still takes softmax weight e^0), so the change altered text
    conditioning for every prompt < 512 tokens. It is the **training** zero-pad convention
    that diverges from the reference, not inference (tracked in BACKLOG TRAIN-PAD-1).
  - **Why it shipped:** the mps build was simultaneously broken (`cBuffers` ARC error, fixed
    only on 2026-06-05 by `ee75739`), so `make test` could not run between 2026-05-15 and
    2026-06-05 — the change was never validated. **Process lesson: never land engine-numerics
    changes while `make test` cannot run; a broken build masks regressions.**
  - **Fix:** revert the zeroing (pad rows returned as-is); `ffecfcc`'s other hunk (strict-
    aliasing `memcpy` in `f16_to_f32`) kept — behavior-neutral and correct. `make test`
    7/7 with restored margins (e.g. img2img 57.5 → 8.25).

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

- **PIPE-2: Flywheel warm-start from LATEST checkpoint compounds regressions → over-training. FIXED 2026-06-08.**
  - **Symptom (warmup-run2):** every iteration trained cleanly, but cond_gap declined
    *monotonically* while train_loss fell — iter1 `+0.0273` (champion) → iter2 `-0.0054`
    → iter3 `-0.0275`, with train_loss `1.0043 → 0.5328 → 0.3877`. The adapter was fitting
    the flow objective harder while *losing* reference conditioning (cond_gap went negative).
    Not stinker shards: three different shard mixes, all worse in lockstep with step count.
  - **Root cause:** the flywheel resumed each iteration from the **latest** `step_*.safetensors`
    via `--resume` (orchestrator `resume_ckpt = str(ckpts[-1])` for the initial pick and
    `resume_ckpt = ckpt_path` for the per-iter carry-forward). `--resume` continues the step
    counter, so steps *accumulated* across iterations (1000→2000→3000). A regressed iteration
    then became the base for the next → compounding + over-training. The DB champion pointer
    (`get_best`, kept at iter-1) was preserved for *shipping* but never used as the *training*
    base.
  - **Fix:** opt-in `resume_from_champion: true` (flywheel config). Each iteration warm-starts
    from the campaign champion (best cond_gap) with a **fresh schedule** via
    `--warmstart-weights` (start_step=0, fresh warmup/optimizer) instead of continuing the
    latest. Each iteration is an independent re-roll on fresh data that cannot drag the working
    checkpoint below the champion. Three gated touch points (initial pick, launch flag, carry-
    forward). warmup-run2 was superseded and relaunched as **warmup-run3** with the flag on,
    warm-starting from run2's preserved champion. Detector: the doctor cond_gap-stall check now
    flags the over-training signature (cond_gap declining while train_loss falls) distinctly.
    See plans/warmup-campaign-runbook.md §3.

## Training Anomalies (Chunk 1 — Observed, Not Actionable Now)

- **ANOMALY-1: Shard-boundary stalls** — Two blocking stalls observed at step ~19,900 (55 min) and ~24,900 (2.6h). Root cause: epoch boundary + simultaneous JDB chunk 2 conversion competing for 2TBSSD I/O. Data% jumps to 100% in timing log. Both resolved automatically. Structural until pixel data is pre-cached to disk. See BACKLOG PIPELINE-3.

- **ANOMALY-2: Optimizer step spikes** — Isolated step-time spikes at steps 19,300 (2.4s vs normal 0.1s) and 25,900 (3.5s). Likely: gradient norm exceeded clip threshold (1.0) triggering full-parameter norm computation, or MLX lazy evaluator GC. No functional impact.
