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
  - **FALSE LEAD (2026-06-17), superseded:** the `perceiver_heads` grouping in
    `iris_ip_adapter_perceive` (hardcoded `hidden/128` vs the trained `perceiver_heads`)
    was *hypothesised* to be the root cause and was fixed. It is a legitimate
    correctness improvement (kept), but it is NOT what caused the grid: the 2026-06-18
    visual confirmation still produced a clean grid at scale 1.2, and the Python/MLX
    perceiver collapses IDENTICALLY at both 16 and 24 heads (`ip_embeds` cross-token
    ratio 0.0068 vs 0.0062). The parity fixture passed because it only exercised
    synthetic math, never the real feature distribution.
  - **SECOND LEAD, REAL BUG but NOT the grid's cause (2026-06-18):** a 4-bit nibble
    DEQUANTISATION SIGN BUG in the training cache loaders. `precompute_all._quantize_4bit`
    stores TWO'S-COMPLEMENT signed nibbles (`clip(round(x/scale), -8, 7)` then
    `(q & 0x0F) | ((q & 0x0F) << 4)`), but `ip_adapter.dataset._load_siglip_embed` and
    `_load_qwen3_embed` unpacked with a plain `& 0x0F` and NO sign-extension — so every
    negative quantised value `[-8,-1]` was read back as a large positive `[8,15]`
    (round-trip corr **−0.52**; sign-extended fix **+0.85**). This is a genuine bug worth
    fixing (every prior adapter trained on sign-corrupted SigLIP+Qwen3 conditioning),
    fixed in `_unpack_signed_nibbles` (commit 6333e24, guard `test_nibble_dequant.py`;
    stored caches byte-correct → no re-precompute). BUT it is **NOT** what causes the
    grid: a confirmation adapter trained 1000 steps on the CORRECTED cache STILL collapses
    (`ip_embeds` cross-token ratio 0.0033) and STILL generates a degenerate image (a brown
    texture, no subject) — so feature-sign correctness does not prevent the collapse. The
    earlier "TRUE ROOT CAUSE" claim was premature. `iris_ip_adapter.c` is still faithful to
    the MLX model and needs no change.
  - **ACTUAL ROOT CAUSE (2026-06-18, confirmed mechanistically):** the `PerceiverResampler`
    (train/ip_adapter/model.py:59) feeds RAW SigLIP features into cross-attention with **NO
    input normalization** — `out = cross_attn(q, siglip_features, siglip_features)`; only
    the OUTPUT is LayerNorm'd. SigLIP has a few "massive-activation" feature DIMENSIONS
    whose magnitude dominates the Q·K dot products, so every learned query attends to the
    same dominant SigLIP token → near-identical output tokens → `k_ip`/`v_ip` carry a
    pooled, spatially-constant signal → constant per-patch injection → the grid/texture.
    **Mechanistic proof** (`/tmp/perceive_diag.py`): on the confirm-fix adapter all **128
    queries peak on the same key (token 352)** — "1 distinct top-key across 128 queries";
    and per-dim z-score standardization of the SigLIP input BREAKS the collapse — `ip_embeds`
    ratio jumps 0.0068→**1.6171** for iter0024 (238×) and 0.0033→0.0135 for the undertrained
    confirm-fix. Per-TOKEN LayerNorm does NOT help (it's per-DIMENSION outliers). The standard
    IP-Adapter Resampler normalizes its input features before attention; ours omits it.
    **FIX IMPLEMENTED + VALIDATED (2026-06-18):** input normalization added to the
    PerceiverResampler — `_norm_input` standardizes each SigLIP DIMENSION across the token
    axis (per-image), then a learned affine (`in_gamma`/`in_beta`), before `cross_attn`
    (train/ip_adapter/model.py). Mirrored in C `iris_ip_adapter_perceive` (in_gamma/in_beta
    loaded optionally → legacy bundles without them keep the old no-norm path, enabling
    same-binary A/B). Exporter carries the two new tensors (optional for legacy checkpoints).
    NOTE: it is per-DIM-across-tokens standardization, NOT a per-token LayerNorm (that doesn't
    help). **Validated end-to-end:** a 600-step retrain on the corrected cache lifts `ip_embeds`
    cross-token ratio 0.0033→**0.4308** (130×, collapse broken), and `iris --ip` A/B (same
    binary) shows legacy iter0024 = the dot grid while the new bundle at ip-scale 0.6 generates
    a COHERENT cat-on-windowsill with the reference's style applied (subject survives, style
    transfers). scale 1.2 over-saturates (high strength) and the 600-step result is hazy —
    undertraining, not a mechanism flaw; a full style-paired campaign will sharpen it.
    Independent of the dequant fix. SREF grid is FIXED at the root-cause level.

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

- **MLX-2: non-deterministic trainer wedge in `mlx::core::eval` — likely an allocator
  reclaim livelock from the MLX memory-limit sitting BELOW the working set (2026-06-18).**
  - **Signature:** during a from-scratch flywheel iteration (cached path, 512px) the
    trainer froze at ~step 10 — process alive at ~1.2 cores, tmux window open, ZERO step
    progress for 40 min. `sample` of the main thread: stuck in `mlx::core::eval_impl`,
    the majority of samples in `std::mutex::lock`/`pthread_mutex_lock` plus
    `get_memory_limit` / `get_active_memory` / `metal::allocator` — i.e. **lock-bound on
    the allocator, not compute-bound on kernels.** No memory pressure (61% free), no
    external GPU contention.
  - **NOT the dequant fix, NOT numerical instability:** an identical-config diagnostic
    run (log_every=1) completed all steps cleanly — early steps finite (volatile
    from-scratch loss spikes to 291 but recover; grad-clip fires; converges), no NaN/Inf,
    EXIT_CODE=0. Same regime, ran fine ⇒ the wedge is non-deterministic. The prior 48
    flywheel iterations (run5/source-probe/iter24-replay, the last ~2 h earlier) all
    completed at ~7–9 s/step, so it is genuinely intermittent, not a constant slowdown.
  - **Distinguished from PROXY-1 / the MLX-1 scope-correction caveat:** an `eval_impl`
    spin alone is NOT proof of a hang — it is also how MLX waits on a long graph (PROXY-1
    was a real 75 s/step graph misread as a hang). The differentiator here is the profile
    is **lock/allocator-bound, not kernel-bound**, AND the same graph runs in ~1–9 s in
    the diag. That is consistent with contention, not honest compute. (Residual
    uncertainty remains until an all-threads dump confirms it — see capture below.)
  - **Leading (avoidable) hypothesis — memory limit below working set:** the trainer caps
    MLX at `training.mlx_memory_pct` (default 0.44 → **14 GB** on 32 GB) with
    `set_cache_limit` ≈ 2 GB, but the measured 512px backward **working set is 20.44 GB**
    (TRAIN-7). MLX runs ~6 GB *above* its own soft limit on every heavy step, forcing
    continuous cache reclaim with little buffer cache to reuse → exactly the
    allocator-mutex churn the sample shows. Usually just overhead; occasionally the
    reclaim path + Metal stream thread livelock. **Lever to test:** raise
    `training.mlx_memory_pct` to ~0.66–0.70 (≈ 21–22 GB, above the 20.4 GB working set,
    below the ~jetsam threshold) and/or raise the cache limit — now SAFE to try because
    the stall-recovery turns a residual jetsam into a clean restart. Tradeoff (why it was
    capped low): without the cap, jetsam kills the process during `mx.compile`'s
    first-backward transient peaks. Alternatives if memory isn't it: MLX 0.31.2 →
    upstream version bump; the trainer also pre-compiles graphs for ALL aspect buckets at
    startup though `data.bucket` is pinned to [512,512] — a reducible compile surface.
  - **Mitigation IN PLACE (commit pending):** the flywheel monitor had NO trainer-stale
    detection (orchestrator.py — only polled the control file + tmux-window-open), so a
    wedge hung the iteration forever. Added: stale-trainer-heartbeat (>`stall_restart_secs`,
    default 2× HEARTBEAT_STALE_SECS = 1800 s) ⇒ `_capture_stall_diagnostics()` (all-threads
    `sample` of every train_ip_adapter.py PID + trainer log tail → `logs/stall_<name>_iter
    <N>_*.txt`) then kill + re-run the iteration, bounded by `max_stall_restarts` (default
    2) then record failed + dispatch an alert. **The capture is the root-cause path:** the
    all-threads dump on the NEXT wedge settles allocator-livelock (main thread in
    eval_impl+allocator) vs a GPU command-buffer hang (a Metal stream/driver thread in
    `waitUntilCompleted`). Until then the memory-limit hypothesis is leading but unproven.
  - **INVESTIGATION (2026-06-18, capture working).** The forensic dump fired: main thread
    blocked in `eval_impl → condition_variable::wait` (waiting, not spinning); the GPU/stream
    thread NOT in `waitUntilCompleted` but actively grinding a flood of tiny ops
    (`binary_op_gpu`/`concatenate_gpu`/`copy_gpu_inplace`/`reshape_gpu` + Metal buffer
    malloc/dealloc + `IOGPUMetalFence` create). So it is **a pathologically huge graph eval,
    not a deadlock or GPU-driver hang** — one `mx.eval` explodes into millions of tiny ops and
    the per-op allocator/fence overhead makes it effectively never finish (the PROXY-1 class,
    but extreme). Fires at the FIRST step (after compile + "Validation held-out", before step 1).
  - **RULED OUT:** style-pairing (wedges with `style_pairing:false` too); cache-miss/live-encode
    (MLX-1) — the wedged run logged "skipping model load … using precomputed cache", i.e. cache
    HIT, and it still wedged; memory-limit 0.6 (wedges at 0.6). **Correlation:** flywheel wedges
    ~4/4 at the first step; single-process DIRECT runs (same model/config family) ~3/4 clean —
    so it is non-deterministic with the flywheel context raising the hit rate, NOT a deterministic
    config bug. A bug fixed along the way (commit bdd7e00): the flywheel trainer config pointed
    cache_dir at the stale standard tree, not the per-iter staged cache — real (live-encode risk +
    unused stager symlinks) but NOT this wedge. **Open.** Leading remaining theory: a
    non-deterministic MLX graph-eval/compile pathology. **MLX version RULED OUT (2026-06-19):**
    0.31.2 is already the latest on PyPI (no forward bump), and a DOWNGRADE test to 0.30.6 STILL
    wedged in the flywheel → not a 0.31 regression. So the only remaining lever is **reducing the
    graph** (lower seq/ops on the IP `correct_forward_q` path so one `mx.eval` can't explode), or
    pinning down the exact flywheel-vs-direct difference (orchestrator-subprocess context — the
    last unruled-out axis; direct-run-of-flywheel-config tests were inconclusive/killed early).
    Workaround for now: stall-recovery retries; run experiments via direct single-process trainer
    (hits it far less, ~3/4 clean vs flywheel ~5/5 wedge).
  - **MITIGATION (2026-06-19, lever-1 partial): startup warmup graph-surface reduction.**
    The three startup warmup loops in `train_ip_adapter.py` (Flux-forward, correct-forward-Q,
    and the `--warmup-only` path) compiled the Flux graph for ALL 6 BUCKETS even though
    training is hard-pinned to a single `data.bucket` ([512,512]; every other bucket is a
    100% cache-miss skip). That built 5 unused large graphs and inflated the Metal PSO/
    allocator state ~6x at startup — the fragile moment right before the wedge fires. Fixed:
    `_warmup_buckets = [_fixed_bucket] if _fixed_bucket else BUCKETS`, used by all three loops.
    Validated by a 5-step direct smoke (warmup 6→1 shapes, ~2 s each; ran clean to completion,
    peak 21.52 GB > the 19 GB limit — i.e. the allocator-pressure condition was present and it
    still did not wedge, one data point). NOT a proven wedge fix — the per-step
    `_flux_forward_with_ip_collect_q` graph (the one the forensic dump showed exploding) is
    unchanged — but a clear, safe surface reduction and faster startup regardless. Next levers
    unchanged: per-step IP-forward graph reduction, or pinning the orchestrator-vs-direct axis.

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

## Inference / Server Anomalies

- **SREF-DAEMON-1 (2026-06-28): per-request IP-Adapter attach in `iris --server` produces CORRUPT
  (pure-noise) output on a warm daemon.** The SREF-3-A path (server-mode `ip_bundle`/`ip_features`/
  `ip_scale` → `iris_load_ip_adapter` in `server_worker` → generate → `iris_unload_ip_adapter`) loads
  the model once and avoids the per-gen reload (verified timings: sref#1 98s cold, sref#2 42s warm). BUT
  the image is NOISE — the denoise never converges, i.e. the transformer forward is corrupted while the
  adapter is attached in the daemon. ISOLATION (same bundle/features/model throughout):
  • daemon, NO adapter (normal gen): CORRECT (clean fox).
  • daemon + per-request attach (sref): NOISE — even on the FIRST request (so NOT pointer-reuse cache
    aliasing).
  • one-shot `iris --ip` (adapter attached at startup, before warmup/any gen): CORRECT (clean styled).
  ⇒ Bug is specific to attaching the adapter AFTER the model is loaded + bf16-warmed (and/or attaching
  on the worker thread). Hypotheses to check: (a) `flux_metal_warmup_bf16` primes fused-bf16 GPU
  graphs/caches that the per-block path (taken when tf->ip set) then misuses — the one-shot may skip
  warmup since tf->ip is set first; (b) `iris_ip_adapter_perceive`/`get_kv` GPU matmuls hit a warm
  static-weight GPU cache and get a wrong/stale buffer (cf. the historical MPS B-cache bug + the
  VAE-K/V-temporaries audit note in MEMORY); (c) worker-thread GPU context/autorelease state at attach.
  IMPACT: SREF-3-A (daemon model-reuse for sref) is BLOCKED. WORKAROUND IN PLACE: web/server.py routes
  sref through the working one-shot `run_generation_sref` (correct, but reloads the model per gen). The
  C server-mode IP support (commit 3872f90) stays as the foundation. `make test` is green — the
  server-mode smoke does NOT check sref image correctness, so it missed this; ADD a golden-pixel sref
  check to the server smoke to catch this class.
  CONFIRMED at MATCHED resolution (2026-06-28, after an initial mis-isolation that compared daemon@256
  vs one-shot@576): at 576px same bundle/features — ONE-SHOT = coherent, DAEMON-attach = PURE BLACK
  (all-zero) output; attach log says "IP-Adapter: attached" but the forward is corrupted. So the daemon
  attach genuinely breaks generation. (At 256px both produce noise — see SREF-LOWRES below.)
  PRECISE LOCALIZATION (2026-06-28, env-gated diagnostics, since removed): the corruption is in the
  PER-BLOCK transformer path that an attached adapter forces — NOT in perceive (ip_embeds byte-identical
  to one-shot, no NaN) and NOT in the inject math. With the adapter attached but ip_inject_block made a
  no-op, the daemon gen is CLEAN — so the trigger is the inject running. Per-block trace of the image-Q
  (tf->ip_q_scratch) at step 1, daemon@576 vs one-shot@576: IDENTICAL for blocks 0–12, then at BLOCK 13
  the daemon's image-Q rows are ALL ZERO (|q|=0.00, while k_ip is still fine) and stay zero for blocks
  13–24. A zero block-Q also breaks that block's own self-attention → propagates → black. So a block's
  fused-QKV GPU projection returns ZERO for the image rows at ~block 13 in the WARM daemon only. RULED
  OUT: LRU cache eviction (bumping MAX_LINEAR_GRAPH_CACHE 64→256 + WEIGHT/BF16 caches 512/1024→2048 did
  NOT change it). Suspected: a warm-daemon GPU-state / GEMM-row (M-dim) / graph-reuse interaction
  specific to running iris_metal_sgemm (IP get_kv) interleaved with the bf16-native per-block projections
  after the fused-path warmup. NEXT: instrument the single-block fused-QKV GEMM output rows per block
  (which block's projection first returns zero image rows), and/or run under a heap/UB checker that works
  with the large model (libgmalloc rejects the multi-GB allocs). The one-shot path (fresh process) does
  the identical compute correctly, so it is layout/warm-state dependent (heisenbug).
  DEEPER LOCALIZATION (2026-06-28, round 2): traced the actual NaN onset by logging every
  iris_linear_nobias_bf16 call (M/K/N + first/last output-row L1 + input max). The FIRST NaN is the
  SINGLE BLOCK's OUTPUT projection (proj_mlp, K=12288 N=3072) of single-block idx 7 (~global block 12),
  and its INPUT is clean and small (in_max=16.37, no NaN) while q/attn_out/mlp for that block are all
  NaN-free — i.e. iris_metal_sgemm_bf16 produces NaN from a finite, small input. The earlier "block-13
  image-Q = 0" was the DOWNSTREAM symptom (block 12's NaN proj output → block 13 input → the max-based
  |q| diag reported NaN as 0). So the defect is in the bf16 GPU GEMM (iris_metal_sgemm_bf16) path under
  warm-daemon state, NOT the inputs. RULED OUT FURTHER: (1) beta=0 reading stale pooled bufferC and
  0*NaN=NaN — zeroing bufferC when beta==0 did NOT fix it; (2) batch_get_input_buffer stale contents —
  g_batch_input_count is cleared every iris_metal_end_batch and the per-block batches are small, so no
  cross-block stale reuse. CAVEAT for the next debugger: in BATCH mode (g_in_batch) the matmul is
  DEFERRED (encoded to g_batch_cmd, copied back at end_batch), so reading the CPU output right after an
  iris_metal_sgemm_bf16 call is UNRELIABLE — instrument by forcing a synchronous commit, or check buffer
  contents on-GPU. Prime remaining suspects: an uninitialized/garbage GPU buffer (bufferA from the pool,
  or the cached f16 weight from get_cached_bf16_as_f16_buffer) read by MPSMatrixMultiplication on a warm
  command queue; or an MPS matmul descriptor/buffer-size mismatch that surfaces only after N prior batched
  ops. Recommended next probe: add a synchronous NaN check on bufferA and bufferC contents (GPU) inside
  iris_metal_sgemm_bf16 right after the matmul (force commit), for the failing call, in the daemon.
  ROUND 3 — PRECISE A/B + EXHAUSTIVE ELIMINATION (2026-06-29). Instrumented iris_metal_sgemm_bf16's
  non-batch path (forced synchronous via a temporary global batch-disable) to log per-call, image-rows-only,
  the two single-block GEMMs: fusedQKV (K=3072 N=27648) and proj_mlp (K=12288 N=3072) — input max,
  weight-nonzero count, and text-row vs image-row output max. Ran the SAME request (576px, 1 step, seed 7,
  same bundle/features, ip_scale 0.45) through the WARM DAEMON and a FRESH ONE-SHOT and diffed:
    * The two traces are BYTE-IDENTICAL through single-block 7 (global block 12). Clean CLIFF at
      single-block 8 (global block 13): that block's proj_mlp outputs ALL ZERO (text AND image rows,
      txtC 19.7→0, imgC 6.4→0) in the daemon, while the one-shot outputs the correct 4.188. The fusedQKV
      of the SAME block is fine in both. The zero then propagates (block 14+ inputs are zero).
    * At the failing call: INPUT bufferA is nonzero (imgA_max 6.443, identical to one-shot), WEIGHT bufferB
      is fully populated (Bnz 37.6M/37.7M nonzero — read on CPU), yet OUTPUT is exactly zero. cmdBuffer
      .status == Completed and .error == nil. So the matmul "succeeds" but writes nothing — a SILENT
      zero-write of a completed MPSMatrixMultiplication, from valid CPU-visible inputs, warm-process-only,
      at a fixed cumulative call count. The earlier "NaN" framing (round 2) was a BATCH-DEFERRAL artifact;
      forced synchronously the true symptom is ZERO output, not NaN.
    * RULED OUT (each tested by build+run, zero-count unchanged ~24/25): F16_WEIGHT_CACHE_SIZE 512→4096
      (weight stays populated regardless); iris_metal batch mode (option-2: global batch-disable while
      adapter attached → all GEMMs synchronous → STILL zero, and ~4x slower); MTLResidencySet on the
      non-batch command buffer (added useResidencySet to the non-batch path → no change; and blocks 5–12
      use the same non-set pool buffers fine); beta=0 stale bufferC (memset → no change); ACTIVATION_POOL_SIZE
      64→256 (no change); the load-time warmup (IRIS_NO_WARMUP → zero-count got WORSE 25→33, so warmup helps).
    * CONCLUSION: a deep MPS/Metal warm-process-state defect — one specific MPSMatrixMultiplication silently
      produces a zero result despite resident, valid inputs, only after enough prior GPU work in the process
      (the daemon's startup warmup/Qwen3 pushes the gen past the threshold; the fresh one-shot stays under it).
      Not reachable via the buffer/cache/batch/residency/pool knobs above. NEXT: Metal frame capture / Xcode
      GPU debugger on the block-13 proj_mlp command in the warm daemon (inspect the actual GPU-side matrix
      inputs/outputs and the MPS kernel selection), or A/B the f32 GPU path (iris_metal_sgemm — get_kv uses it
      and works warm) vs bf16 for the per-block linears. The WORKING one-shot remains shipped; daemon
      model-reuse (BACKLOG SREF-3 part A) stays blocked on this. All round-3 instrumentation reverted; C tree
      unchanged from the committed working state.
  ROUND 4 — f32-PATH A/B CONFIRMS THE CULPRIT (2026-06-29). Added a global iris_force_f32_linear that, while
  an IP-Adapter is attached, routes iris_linear_nobias_bf16 through the f32 GEMM (iris_metal_sgemm) instead
  of iris_metal_sgemm_bf16. RESULT: the warm-daemon sref render is CORRECT (coherent, no black, byte-for-byte
  the same subject as the one-shot) — so the defect is SPECIFICALLY the bf16 (f32×f16 mixed-precision) GPU
  GEMM; the f32×f32 path is immune. Two sub-findings:
    (a) The f32 fallback in iris_linear_nobias_bf16 was itself BUGGY: it converted bf16→f32 into a per-call
        malloc'd temporary and called iris_linear_nobias → iris_metal_sgemm_CACHED, which keys the weight
        buffer by POINTER. A freed+reused W_f32 address served a stale GPU buffer → NOISE (same class as the
        Z-Image B-cache pitfall). Fix: route the temporary through the NON-cached iris_matmul_t. With that, the
        f32 path renders correctly in BOTH the fresh one-shot and the warm daemon.
    (b) The f16 residency-eviction theory was tested and DISPROVEN: adding the f16 weight buffers to
        g_residency_set (addAllocation+commit) + bumping F16 cache so all weights cache → daemon STILL black.
        So the bf16 GEMM zero-write is NOT f16-weight non-residency.
  PERF VERDICT — f32 workaround is correct but NOT a net win as-is: warm-daemon f32 sref = ~287 s vs the
  one-shot's ~98 s (reload ~56 s + bf16 gen ~42 s; bf16 no-IP one-shot @576/4steps measured 36.9 s). The ~245 s
  overhead is per-call bf16→f32 weight conversion + re-upload (the fused-QKV weight alone is 340 MB, re-done
  every call). A FAST correct daemon needs the f32 weights CACHED in a GPU buffer keyed by the STABLE bf16
  source pointer (mirror get_cached_bf16_as_f16_buffer but f32 + an f32×f32 sgemm that uses it) — est. ~+8 GB
  unified memory (f32 vs f16 weight cache) and est. ~70 s/gen daemon (~28 s/image win over the one-shot).
  DECISION PENDING (memory cost vs modest win). All round-4 instrumentation reverted EXCEPT none committed yet;
  C tree clean, one-shot fast (bf16) and unaffected. The genuine fallback bug fix (a) is worth landing
  regardless if the cached-f32 path is pursued.
  ROUND 5 — MPSGraph PATH: CORRECT + NO MEMORY, BUT SLOW (2026-06-29). Built iris_metal_sgemm_bf16_mpsgraph:
  a per-block bf16 linear via MPSGraph (f32 x placeholder, the STABLE cached bf16 weight from
  get_cached_bf16_buffer, transpose+cast-to-f32, matmul → f32 out) — a DIFFERENT GPU code path than
  iris_metal_sgemm_bf16's MPSMatrixMultiplication, with NO extra memory (weight stays bf16, cast happens on
  GPU each call). Routed iris_linear_nobias_bf16 through it while an adapter is attached. RESULT: warm-daemon
  sref render is CORRECT (coherent cat, matches one-shot) AND fresh one-shot correct — so MPSGraph dodges the
  MPSMatrixMultiplication warm-state defect, confirming the defect is specific to MPS's MPSMatrixMultiplication
  (mixed f32×f16) object. PERF: warm-daemon MPSGraph gen = 211 s, one-shot MPSGraph = 218 s — ~5× the bf16
  gen (~42 s). Root cause of the slowness: the SINGLE-BLOCK path (single_block_forward) is CPU-SYNCHRONOUS
  (each linear's output feeds CPU ops — RMSNorm, attention, SwiGLU — so it can't batch), and is NOT wrapped in
  begin_batch (only double_block_forward batches). bf16 MPSMatrixMultiplication is cheap per-call (~0.12 s)
  but MPSGraph's per-call encode/MPSCommandBuffer/MPSGraphTensorData setup is ~5× heavier and can't be
  amortized in a per-call-synchronous loop. The FAST MPSGraph linears live only in the FUSED pipeline
  (single_block_forward_bf16: on-GPU iris_gpu_tensor_t, g_tensor_batch_mode deferred, no per-call commit+wait),
  which is gated OFF when tf->ip is set. CONCLUSION + REAL FIX: the genuinely fast + correct + zero-memory
  daemon = wire the IP-Adapter inject INTO the fused bf16 pipeline (BACKLOG SREF-3 part B2). That pipeline
  already (i) uses batched MPSGraph linears that dodge this bug and (ii) keeps activations on-GPU; injecting
  to_k_ip/to_v_ip there via SDPA gives bf16-speed correct sref with the model resident. Larger but well-scoped
  work. Until then: SHIP THE ONE-SHOT (bf16, ~98 s, correct). All round-5 code reverted; C tree clean.

- **SREF-LOWRES (2026-06-28): sref at 256px produces NOISE even on the working one-shot path.** Distinct
  from SREF-DAEMON-1: one-shot `iris --ip` at 256x256 = noise, at 576x576 = coherent (same bundle). The
  no-adapter path at 256px is fine (clean image), so it's the IP injection at low res. Likely the
  img_seq at 256px (16x16=256 image tokens) interacting with the 256 perceiver image-tokens, or the
  low-res latent + strong injection. NOT normally hit (sref is used at 512+); flagged so a future
  low-res sref path is validated. Workaround: use >=512px for style reference.
