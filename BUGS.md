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

## Known Anomalies (Open)

- **IP-ADAPTER-INFER-1: `iris --ip` C inference degenerates into a patch-periodic GRID
  at ip-scale > 0 (2026-06-14, found by the first visual sref sweep).**
  - **Signature:** base generation perfect; `--ip-scale 0.0` bit-identical to base
    (inertness correct); but with the adapter ACTIVE the output degrades monotonically
    with scale — 0.4: coherent image + colour tint + faint patch-striping; 0.8:
    checkerboard; 1.2: a clean regular grid of dots, NO subject, in the REFERENCE's
    dominant colour (e.g. sage-green ref → green-dot grid). All 18 sweep generations
    affected; sref_eval scores on them are therefore meaningless.
  - **It is an INFERENCE bug, not a bad/undertrained adapter — three independent tells:**
    (1) the artifact is a crisp PATCH-PERIODIC grid (structural), not the blur an
    undertrained adapter produces; (2) the held-out **cond_gap is POSITIVE (+0.0727)** —
    the very weights that grid in C inference IMPROVE the loss in the Python training
    forward, so the weights are fine and the C forward differs; (3) the grid carries the
    reference's pooled COLOUR, i.e. the IP contribution is being injected with wrong
    spatial structure (patch-aligned broadcast) that grows with scale.
  - **Why G-1 Phase 2 missed it:** it validated adapter MATH (k/v projection, perceive)
    bit-exact vs Python goldens on synthetic fixtures, and the inertness gate (scale 0).
    It NEVER generated a full image with a real trained adapter at scale > 0 — necessary
    but not sufficient. (Lesson mirrors the "smoke before a long run" rule: unit-parity ≠
    end-to-end coherence.)
  - **Impact:** blocks the SREF-METRIC-1 validity gate (cannot visually assess whether
    cond_gap tracks style until generation works), AND the app style feature uses the same
    `iris --ip` path → it does not produce usable styled images yet. Does NOT affect
    training (run5/flywheel only train adapters; cond_gap is a latent-space loss metric).
  - **Disambiguation / next step (GPU):** generate with the SAME adapter+features via the
    Python/mflux inference path. Clean styled image there + grid in C ⇒ confirmed C
    injection bug (then audit the IP contribution's spatial mapping in
    iris_transformer_flux.c — patch alignment of the 128 image tokens / k_ip,v_ip add).
    Grid in both ⇒ adapter pathology (unlikely given the positive cond_gap). Repro:
    `./iris -d flux-klein-model -p "a cat on a windowsill" --ip
    /Volumes/2TBSSD/sref_sweep/bundle_iter0024 --ip-features
    /Volumes/2TBSSD/sref_sweep/refs/000181_0002.bin --ip-scale 1.2 --seed 42 --steps 4
    -W 512 -H 512 -o /tmp/grid.png`.

- **MLX-1: SIGSEGV in MLX compiled-kernel GPU eval on the trainer's ONLINE-ENCODE path
  (2026-06-10, observed during TRAIN-7 memory probes).**
  - **Crash signature:** `EXC_BAD_ACCESS KERN_INVALID_ADDRESS at 0x0` in
    `mlx::core::metal::CommandEncoder::set_input_array` ← `Compiled::eval_gpu` ←
    `gpu::eval` ← `eval_impl` (incident 7736CCA2). Python 3.14.5 / current libmlx.
    Reproduced twice within ~1-2 training steps at 768px with live encoders.
  - **Trigger:** the trainer's online-encode path only — per step-window it loads
    VAE/Qwen3 (and SigLIP), encodes, then "releases all live encoders + clears GPU
    caches". The crash fires in the next compiled-kernel eval after that release/clear:
    a compiled kernel's cached input buffer appears to be invalidated by the cache clear
    (NULL input array in the command encoder). A prefetch thread doing PIL resize was
    concurrently active.
  - **NOT hit in production:** flywheel/production training runs fully CACHED (no live
    encoders), which is why this never fired in weeks of campaign training. The TRAIN-7
    probes initially used live encode for worst-case memory and hit it immediately.
  - **Workaround:** run probes (and anything else) with precomputed caches — also more
    production-representative. The online-encode path should be treated as broken until
    the encoder-release/cache-clear sequence is fixed or MLX is upgraded past the bug;
    investigate `mx.clear_cache()` placement in the trainer's `[online-encode]` release
    step relative to pending compiled evals.
  - **Scope correction (same day):** the proxy-VAE trainer's apparent "MLX livelock"
    (eval_impl spin, no steps for hours) was NOT this bug — it was a legitimately
    long-running graph (see PROXY-1 below). MLX-1 remains real only for the SIGSEGV on
    the online-encode path. Don't conflate the two: an `eval_impl` busy-spin is how the
    MLX main thread waits on any long eval; it is not by itself evidence of a hang.

- **PROXY-1: decoded-MSE loss term costs ~75 s/step, not the documented ~20 ms (2026-06-10).**
  - `vae_proxy_512px.yaml` shipped with `decoded_mse_weight: 0.10` and the comment "adds
    perceptual signal; ~20ms overhead/batch" — that figure costed the decoder FORWARD only.
    As a loss term, the backward retains the frozen 512px teacher-decoder activations at
    batch 8: measured **74–79 s/step** vs **1.7 s/step** student-only (45×). 50K steps ⇒
    ~46 days. Compounded by `log_every: 100`: the first progress line would take 2+ hours,
    so three launch attempts were misread as hangs and killed (one of them minutes before
    its first log line).
  - **Resolution:** train latent-space-only (`decoded_mse_weight: 0.0`) — decoded quality
    is still MEASURED at Tier-1 eval (LPIPS/PSNR), just not trained against; `log_every: 10`.
    If Tier-1 LPIPS fails, revisit with a periodic (every-Nth-step) or low-res decoded term,
    never per-step at 512px. **Lesson: cost loss terms with their backward, and never set
    log_every so high that the first signal takes hours.**

## Training Anomalies (Chunk 1 — Observed, Not Actionable Now)

- **ANOMALY-1: Shard-boundary stalls** — Two blocking stalls observed at step ~19,900 (55 min) and ~24,900 (2.6h). Root cause: epoch boundary + simultaneous JDB chunk 2 conversion competing for 2TBSSD I/O. Data% jumps to 100% in timing log. Both resolved automatically. Structural until pixel data is pre-cached to disk. See BACKLOG PIPELINE-3.

- **ANOMALY-2: Optimizer step spikes** — Isolated step-time spikes at steps 19,300 (2.4s vs normal 0.1s) and 25,900 (3.5s). Likely: gradient norm exceeded clip threshold (1.0) triggering full-parameter norm computation, or MLX lazy evaluator GC. No functional impact.
