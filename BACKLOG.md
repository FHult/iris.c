# FLUX.2 / iris.c — Improvement Backlog

Completed items are archived in [COMPLETED_BACKLOG.md](COMPLETED_BACKLOG.md).

---

## Foundation Quality — Phase A progress

**A1 — VAE precompute CLOSED (2026-07-29).** The doctor's long-standing "VAE `v_2232c1` incomplete"
(the roadmap's one real data-foundation gap) was NOT missing data. The cold cache held **1,154,314
npz latents / 936 shards** — one MORE than the complete qwen3 cache (1,154,313), same precompute run
(git_sha `6333e24f`), identical filenames in identical order — but the build was interrupted before
`mark_complete`, so the manifest stayed `complete:false` with no `record_count` (hence the doctor's
`records:null` — a manifest artifact, never a data count). Fixed with
`PrecomputeCache.mark_complete(record_count=1154314, shard_count=936)` behind guards (version hash
reconstructs to `v_2232c1`; count > 1.1M sanity). **No GPU, no re-encode, no shard staging.** Verified:
`cache_manager.py list vae` → `complete=True records=1154314`; doctor `cold_precompute.vae` →
`complete:true`. Notes: (a) there is **no `cache_manager.py --verify`** (only `list`/`consolidate`) —
the roadmap's stated verify command was wrong, now corrected. (b) `mark_complete` overwrites
`created_at` with now — but qwen3/siglip show the same (created_at ≈ completed_at), so it's the
convention, not a regression. (c) The remaining 3 "current version incomplete" doctor warnings are the
HOT ~522k-record staged **subsets** (Phase A2, cosmetic) — intentional partial working copies, NOT the
full 1.15M universe; do not blindly mark them complete.

**Coverage caveat (do NOT read A1 as "the corpus is precomputed").** `mark_complete` means "this run
finished the work it was configured to do," NOT "covers all shards/images." All three encoders cover
the SAME subset: ~935–945 of the **1,280**-shard corpus (~344 shards have NO precompute) and ~1.15M
records — roughly **18% of the ~6.4M documented corpus images** (~1,233 records/covered-shard vs
~5,000 nominal ⇒ the covered shards are also SUBSAMPLED, consistent with the `--subsample-per-shard`
lever, DP-2c). So A1 fixed a manifest bug and made VAE consistent with qwen3/siglip; it did NOT do
full-corpus precompute. Full-shard / full-image (and the 768/1024px re-encode) is the separate
data-SCALE work — roadmap Phase 3 / A3 — a real multi-day GPU + hot-staging job for ALL THREE encoders,
still open.

**ZIMAGE-SCHED-1 (open, needs authoritative golden) — Z-Image scheduler shift resolution-dependence.**
Review 2026-07-30 (H2): the C FlowMatch scheduler uses a resolution-blind static `shift=3.0`
(`iris_sample.c:201-222`). The only reference implementation on this machine, mflux, uses a
resolution-DEPENDENT shift for Z-Image-Turbo (`base_shift=0.5, max_shift=1.15, base/max_seq_len
256/4096`; effective ≈3.16 @1024², ≈1.88 @512²) — never a flat 3.0; symptom is a wasted near-zero
final step at high res (penultimate sigma 0.0089 vs ~0.31 @1024²/8-step). CLAUDE.md asserts static is
the OFFICIAL-diffusers behavior, but official diffusers + the model's `scheduler_config.json` are
ABSENT from the machine (confirmed by a machine-wide scan), so it cannot be arbitrated. **Not changed**
(regressing a working path on an unconfirmed reference is worse); a `IRIS_ZIMAGE_SHIFT` env override was
added (default 3.0, behavior unchanged). **To close:** obtain the official Z-Image-Turbo diffusers
scheduler config / model card, confirm `use_dynamic_shifting`, and only then switch + add a Z-Image
scheduler golden fixture (MISSING — the whole Z-Image path is unguarded by `make test`). Also fixes the
documented default-steps off-by-one (M1, 9→8) shipped in the same change. Cross-ref: review-2026-07-30.

**M3 (VERIFIED FAITHFUL + GUARDED, 2026-07-31) — IP-adapter per-block injection train↔infer parity.**
Review 2026-07-30 flagged that the DEFAULT trainer path `_pred_from_embeds`
(`train/train_ip_adapter.py`, `use_block_injection=False`) is an APPROXIMATION of C inference: it sums
all blocks' IP-attention outputs into `h_final` ONCE before `norm_out`/`proj_out`, deriving every
block's image-Q from the IP-FREE hidden — whereas C (`iris_transformer_flux.c`) injects k_ip/v_ip PER
BLOCK into the post-block hidden, so block i+1's Q sees block i's injection. **Source audit of the
alternative `use_block_injection=True` path (`_flux_forward_with_ip`) vs C: CONFIRMED FAITHFUL** on all
five axes — (1) injection point = post-block hidden (double: end of block; single: image rows of
[txt|img]); (2) propagation = each block's input carries prior injections; (3) k_ip/v_ip = `ip_embeds @
to_{k,v}_ip_stacked[i]` (einsum `btd,de`), reshaped to heads=hidden/128; (4) image-Q = post-QK-norm,
PRE-RoPE (both sides skip RoPE — SigLIP K/V are position-free); (5) per-block scale `scale[i]`, flat
index (double `0..nd-1`, single `nd+j`). So **enabling `use_block_injection=True` is train↔infer-CORRECT**
(safe from a parity standpoint) — its only cost is the ~4.7× slower training already documented, a
separate paid decision. Left OFF as instructed. New hermetic guard: `bundle_blockprop` +
`gold_blockprop.bin` in `debug/gen_ip_adapter_fixture.py` and `run_block_prop()` in
`debug/test_ip_adapter.c` (wired via `make test-unit`) reproduce the per-block-injected forward over 5
blocks with randomised per-block scale and a per-head-RMSNorm derive_q (so Q depends on accumulated
injections). Parity corr=1.000000 / max_abs=6e-5 (5e-5 under production `-ffast-math -O3 -flto` flags —
noise, not mismatch). The generator asserts the per-block vs end-sum forwards differ by 20.5% (>1%), so
a regression to the end-sum approximation would fail the fixture. NOTE: the `_pred_from_embeds` default
itself is unchanged — it remains the fast (approximate) training path; this work only VERIFIED the
correct path and added the regression guard. Cross-ref: review-2026-07-30, IP-ADAPTER-INFER-1.

---

## Training Development Lessons

Lessons crystallised from the TRAIN-6 / Option C development cycle (2026-05-13).

**1. Always smoke before a long run.**
Run 100 steps from the current checkpoint before enabling any new training code path. Validates: Metal graph warmup (catches 10–30 min compilation stalls), step timing, memory peak, NaN propagation, and style-loss / EMA / cross-ref interactions. The TRAIN-6 smoke discovered a 45 GB OOM peak (2.25× over budget) that was invisible in profiler benchmarks. Without the smoke, the first production run would have swapped and stalled indefinitely.

**2. Profiler step time ≠ production step time.**
The TRAIN-6 profiler gave 8.38 s/step; the smoke measured 14.2 s/step (1.69× gap). The gap comes from style loss, EMA update, data prefetch, cross-ref permutation, and checkpoint I/O — none of which appear in a synthetic-batch profiler. Always derive production timeline estimates from smoke-measured step time, not profiler numbers. Scale the profiler number by 1.6–1.7× for realistic planning.

**3. Memory peaks are only visible in smoke, not profiler.**
The synthetic profiler batch skips style loss (no reference latent), EMA (no ema_params state), and data prefetch (no concurrent threads). These add ~2–4 GB above the profiler peak. `memory_profile: true` in a real 100-step smoke is the only reliable way to confirm a new backward path fits in 32 GB.

**4. Warmstart comparison understates gradient path quality differences.**
The TRAIN-6 vs old-path 500-step comparison from step 108500 showed mean cond_gap +0.076 (TRAIN-6) vs +0.334 (old path). This gap is largely a warmstart artifact: switching gradient direction after 108K steps on the old path creates a distribution mismatch — the adapter's K/V weights are optimised for a different loss landscape, so the first ~250 steps are spent adjusting (first-half mean: +0.008, near noise). Only a from-scratch comparison or a multi-thousand-step continued run is a fair quality-ceiling test. Treat short warmstart comparisons as efficiency signals, not quality-ceiling signals.

**5. New gradient paths need end-to-end smoke validation, not just unit tests.**
`_flux_forward_with_ip_collect_q` was unit-tested (correct shapes, non-zero Q delta at later blocks). But the full training loop — warmup compilation for all 6 bucket shapes, `adapter.get_image_embeds` called outside `value_and_grad` with real weights, memory peak during the extra forward pass — was not validated until smoke. Unit tests confirm the function is correct; smoke tests confirm it integrates correctly with the rest of the training stack.

---

## Pipeline / Storage Development Lessons

Lessons crystallised from the PIPELINE-26/27/28/29 implementation and two-pass code review (2026-05-14).

**6. `Path.suffix` truncates at the last dot — watch for multi-extension files.**
`Path("chunk1_final.ema.safetensors").suffix` returns `".safetensors"`, not `".ema.safetensors"`. A glob + suffix-based copy loop for `chunk{N}_final.*` would map both `chunk1_final.safetensors` and `chunk1_final.ema.safetensors` to the same destination name. Whichever glob returns first "wins" — the EMA checkpoint could silently replace the real weights in the cold archive. Fix: enumerate extensions explicitly rather than deriving them from the source filename. Use `Path.suffixes` or name-based filtering (e.g. `".ema." in f.name`) when multi-dot extensions are in scope.

**7. Match manifest keys exactly — don't assume snake_case consistency.**
`cache_manager.py` writes `"record_count"` to `manifest.json`; an early version of `data_explorer.py status` read `"records"` and silently returned 0 for all precompute record counts. The doctor code had already established the correct `m.get("record_count", m.get("records"))` fallback pattern — new code that reads the same manifests must use the same key and fallback. When in doubt: grep for the write site, not for the read site.

**8. Use config-derived paths everywhere, not just for writes.**
The stager's `archive_chunk()` correctly derives `cold_root / "metadata"` from the config. An early version of `shard_scorer.py` used the hardcoded `COLD_METADATA_DIR` constant for its input DB path but the config-derived `cold_root / "metadata"` for its output path. If `cold_root` differs from the default constant, the scorer reads from the wrong place. Rule: derive all storage paths from the same source (config > constant), and be consistent across both read and write paths in the same function.

**9. Two-pass review is worth it for storage-touching code.**
The first review pass (immediately after implementation) caught 4 bugs: missing `load_config` import, wrong fix-command chunk number, absolute symlinks, wrong DB source path. The second pass (after a day) caught 3 more: EMA file collision (`Path.suffix` truncation), wrong manifest key (`records` vs `record_count`), and path-source inconsistency in `shard_scorer.py`. Storage code modifies persistent state — bugs here can corrupt cold archives silently for many chunks before detection. Budget for at least two review passes before first production use.

---

## Platform Vision & Long-term Architecture

**Goal:** evolve iris.c from a fast inference engine into a fully autonomous, self-improving `--sref` optimization platform — running continuous flywheel campaigns (days/weeks/months) that automatically improve both training data and hyperparameters, culminating in open-weight release of a high-quality IP-Adapter.

### Dual Flywheel System

```
┌──────────────────────────────────────────────────────────────┐
│  Meta / Optimization Flywheel  (slower cadence)              │
│  Smart Shard Selection  +  Ablation Harness                  │
│  → Which data to train on  +  Which hyperparameters to use   │
│  ← shard_scores.db + ablation_history.db (persistent)        │
└────────────────────┬─────────────────────────────────────────┘
                     │ curated shards + best config
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  Main Training Flywheel  (frequent)                          │
│  IP-Adapter training  →  eval metrics  →  shard scores       │
│  → cond_gap / CLIP-I / style loss feed back to meta          │
│  ← warm-started from best archived checkpoint                │
└────────────────────┬─────────────────────────────────────────┘
                     │ weights, embeddings, metrics
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  Cold Storage — Long-term Knowledge Base                     │
│  Weights archive  +  Versioned precompute  +  Shard scores   │
│  +  Ablation history  +  Raw data pool                       │
│  Every campaign leaves a richer foundation for the next.     │
└──────────────────────────────────────────────────────────────┘
```

The meta flywheel decides what to train on and with what config. The main flywheel executes and feeds metrics back. Cold storage accumulates the knowledge: each campaign extends shard score history, adds scored configs to ablation history, and archives weights. Every new campaign starts smarter than the last.

### Warm-Start as a First-Class Principle

Starting fresh is the expensive fallback, not the default. Every layer of the system must support warm-starting from prior state:

- **Training:** new campaigns warm-start from the best archived checkpoint for the target config. `data_explorer.py --suggest-warmstart` emits the exact `--warmstart` + `--precompute-version` flags.
- **Ablation harness:** on startup, loads `ablation_history.db` and pre-seeds the Optuna TPE study with all prior scored runs before suggesting new candidates. A new campaign instantly inherits the full Pareto frontier from prior work — no redundant exploration.
- **Precompute:** `cache_manager.py --warm-start-precompute <old_version>` copies embeddings for shards whose encoder did not change, skipping full recompute on partial updates. 6-month-old embeddings remain valid if the encoder is unchanged.
- **Shard selection:** `shard_scores.db` accumulates cond_gap and CLIP-I contributions across all campaigns; scores improve with each run. The meta flywheel never starts from zero.

This compounds: the 10th campaign benefits from 9 campaigns of shard intelligence, hyperparameter Pareto history, and weight lineage — dramatically narrowing the search space and reducing time-to-quality.

### Cold Storage as Long-term Knowledge Base

Cold storage is not a backup or overflow — it is the primary accumulator of system intelligence:

- **`metadata/shard_scores.db`** — never truncated; score history grows with every campaign. The meta flywheel scores shards from the full history, not just the most recent run.
- **`metadata/ablation_history.db`** — every ablation run ever recorded, across all campaigns. The Optuna study is rebuilt from this on each new run; the Pareto frontier only improves.
- **`weights/flywheel-*/`** — full checkpoint lineage. Enables warm-starting any future experiment, bisecting quality regressions, and comparing approaches tried months apart.
- **`precompute/v*/`** — versioned encoder outputs that remain valid indefinitely for unchanged encoders.

**Rule:** cold storage is append-only except for explicit operator-triggered garbage collection. Pipeline operations never touch the raw pool, weight archives, or metadata databases.

### Storage Architecture

Three-tier design (target state — current system is hot + cold only):

- **Ultrahot tier** (internal NVMe SSD, ~2–8 TB) — lowest-latency path for the live inference server and web app. Holds only the active weights and embeddings needed to serve requests. Populated by the stager from hot; never written to by training.
- **Hot storage** (`/Volumes/2TBSSD`, 2 TB TB5 SSD) — fast working area for the active + next compute window only. Pipeline reads shards, precompute, and weights from here during training.
- **Cold storage** (`/Volumes/16TBCold`, 16 TB spinning disk) — source of truth and long-term knowledge base. Never auto-deleted by pipeline operations.

**JIT Data Stager** manages all directions:
- **Cold → Hot (staging):** before a compute window, stages raw data, precompute, and weights from cold to hot. Uses symlinks when on the same filesystem (near-instant); atomic copies across filesystems. `_check_hot_space()` enforces `staging_margin_gb` before any transfer.
- **Hot → Cold (archiving):** after a successful run, archives newly generated precompute embeddings, weight checkpoints, and per-campaign telemetry to cold. This is the write path — without it, cold never grows and warm-starts never improve.
- **Hot → Inference (promote):** after archiving, copies or symlinks the active checkpoint + its precompute version to the Ultrahot tier so the web app picks up new weights without restarting.

All directions are first-class operations. Staging populates the working set; archiving accumulates the knowledge; promotion makes results live. See **PIPELINE-30** for implementation.

See also **PIPELINE-32** and **PIPELINE-33** (Hot ↔ Ultrahot movement helper + policy) for the remaining tier-transition convenience layer that is still largely manual today.

### Proposed cold storage layout

```
/Volumes/16TBCold/
├── raw/
│   ├── journeydb/          # persistent tgz pool — never auto-deleted
│   └── journeydb_anno/     # annotation index — downloaded once, kept
├── precompute/             # versioned encoder caches (managed by cache_manager.py)
│   ├── v1/
│   ├── v2/ …
│   ├── current/            # symlink → active version
│   └── manifests/          # per-version coverage manifests
├── weights/                # archived IP-Adapter weights + checkpoints
│   ├── flywheel-YYYYMMDD/  # one dir per campaign
│   └── best/               # symlinks → current best weights per metric
├── metadata/               # persistent telemetry — never reset between campaigns
│   ├── shard_scores.db     # scored shard history (feeds meta flywheel)
│   ├── ablation_history.db # all ablation runs ever (feeds Optuna warm-start)
│   └── flywheel_logs/      # structured per-campaign JSON logs
├── reports/                # all HTML reports (flywheel, ablation, shard selection)
├── temp/                   # staging area for in-progress transfers
└── logs/                   # operational logs (pipeline, orchestrator)
```

This layout is the target state. Current hot-storage paths under `/Volumes/2TBSSD/` remain unchanged during the transition; the stager will progressively migrate source-of-truth data to cold as PIPELINE-25/26/29 land.

### Hardware scaling roadmap

Current: M1 Max, 32 GB unified memory, ~400 GB/s bandwidth, 2 TB hot + 16 TB cold.
Future: M5 Max Mac Studio. The dual-flywheel architecture, cold storage layout, and versioned precompute design are all intended to scale without structural changes — only config and scale parameters change. The accumulated knowledge base (shard scores, ablation history, weight archive) carries forward directly to any new hardware.

**Key M5 Max facts (researched 2026-06-17; Apple M5 Pro/Max announced 2026-03-03).**
The HW-M5 items below are grounded in these — re-verify on the actual device.
- **GPU: 40 cores, each with a dedicated Neural Accelerator** (matrix-multiply hardware,
  Apple's tensor-core equivalent) → **"more than 4× peak GPU AI compute vs M4 Pro/Max"**,
  i.e. roughly **~6–8× vs this M1 Max** for matmul-heavy work (transformer GEMM/attention).
  This — not the memory — is the headline lever.
- **Memory: up to 128 GB unified.** NB the trainer's MLX cap ALREADY auto-scales with RAM
  (`mx.set_memory_limit(_ram_bytes * 0.44)`, train_ip_adapter.py:713-721 → 56 GB on 128 GB);
  the only hard-pinned bottleneck is `batch_size: 1`.
- **Bandwidth: ~614 GB/s (40-core) — only ~1.5× M1 Max's ~400.** So I/O-bound model LOAD
  improves only modestly → warm-resident models matter MORE than faster RAM (corrects an
  earlier assumption that load would speed up a lot).
- **Metal 4 era; native `bfloat` + simdgroup matrix intrinsics** (Apple9+, present since M3).
- Refs: en.wikipedia.org/wiki/Apple_M5 ; apple.com/newsroom/2026/03 (M5 Pro/Max debut) ;
  appleinsider.com/articles/26/03/03 (>4× GPU AI compute vs M4) ; notebookcheck.net M5 Max.

- **HW-M5-1: precompute model-load + per-iteration reload cost (defer to M5 Max 128 GB).**
  Observation (M1 Max, 2026-06): during the flywheel precompute's MODEL-LOAD phase the 2
  efficiency cores saturate while the 8 performance cores sit idle. Two causes with
  opposite fixability: (a) weight loading (~10 GB Qwen3+VAE+SigLIP) is I/O-bound → E-cores
  issue reads, P-cores idle, NOT CPU-limited → core choice irrelevant; (b) MLX/MPSGraph
  kernel COMPILE is CPU-bound and may run at a low QoS (E-cores) → a `taskpolicy -c
  userinitiated` launch *might* speed it (uncertain — MLX threads may self-tag; needs an
  A/B). The flywheel precompute is NOT throttled by us (plain `caffeinate`; the
  `nice -n 10 taskpolicy -d throttle` wrap is chunk-pipeline only). Payoff on M1 is small:
  load is partly masked (pass-1 pipelines during staging) and is a minor fraction of the
  ~2.87h iteration.
  **The real lever, unblocked by 128 GB:** the flywheel relaunches a FRESH precompute
  process every iteration → reloads ~10 GB of models + recompiles kernels 30× per campaign.
  Today precompute and training are SPLIT into separate processes precisely to fit the
  32 GB budget (precompute ~10 GB encoders + training ~20 GB Flux/adapter would blow it).
  With 128 GB they can COEXIST → a persistent precompute server keeps encoders warm across
  iterations, eliminating per-iteration load+compile entirely (far more than the E/P-core
  tweak). Revisit the whole item on M5 Max; the QoS A/B is only worth it if a persistent
  server isn't adopted. (See HW-M5-5 for the warm-server-vs-cache-free decision.)

- **HW-M5-2: bigger-batch + longer training → narrow the warmup-vs-PRODUCTION validity gap
  (highest training-quality lever).** Every stage config pins `batch_size: 1`
  (stage1/2/3_*.yaml) — NOT for I/O but for the **20.44 GB activation peak** at 512px
  batch-1 (train_ip_adapter.py:1911 fence comment; 1024px = 21.32 GB per TRAIN-7), which is
  right at the M1 ~21.5 GB ceiling. Activations scale ~linearly with batch (the FROZEN base's
  full forward is retained to backprop into the adapter), so 128 GB enables **batch ~8–12 at
  512px**, and 6–8× compute fits more steps in the same wall-clock. **The deep point (not
  just "better adapters"):** the flywheel ranks every shard mix by the cond_gap of a CRUDELY
  undertrained batch-1×1000-step from-scratch adapter — a proxy (cond_gap, see SREF-METRIC-1)
  measured on a proxy (undertrained adapter) for the production regime. Training the warmup at
  near-production batch/steps makes the **entire data-selection + ablation search more
  PREDICTIVE of the real outcome** — it shrinks the doubly-removed-proxy gap. Effort: loader/
  sampler for batch>1 (today's batch emit + style-pair logic assume batch 1) + LR re-scaling.
  Cross-refs: SREF-METRIC-1, SREF-DATA-1 (the style-signal sampler compounds at batch>1),
  TRAIN-7 (memory numbers), HW-M5-6 (the flywheel rethink this feeds).

- **HW-M5-3: production foundation run at M5 scale — the time-to-SHIP lever.** DP-5 (the
  ~12-day 512px foundation run on M1, then Stage 2 ~2.1d + Stage 3 ~2.1d per TRAIN-7) drops
  to **~2 days at 512px** at 6–8× compute. This is arguably the single biggest M5 benefit for
  actually SHIPPING — it converts "a 2-week run done once on faith" into "a 2-day run you can
  iterate." Re-derive the chunk/step budget from M5 throughput (PROD-2). Gate unchanged: only
  start once SREF-1 (style pairing validated), SREF-2 (style eval), and the data recipe are
  settled — M5 makes the run cheap, not the prerequisites optional. Cross-refs: DP-5, PROD-2,
  TRAIN-7, HW-M5-2 (batch/res it runs at).

- **HW-M5-4: GPU-native bf16 + matrix-accelerated IP-adapter inject — the app --sref latency.**
  The bf16 single-block inject is HARD-DISABLED today (`if (0 && …)` at
  iris_transformer_flux.c:3853) because on M1 "MPS GEMM doesn't support bf16, custom bf16
  kernel ~4× slower" → the adapter forces the slow CPU-block path. On M5 this inverts twice:
  (1) native `bfloat` (Metal 4) removes the penalty; (2) the per-core Neural Accelerators are
  PURPOSE-BUILT for this GEMM/attention. Re-enable the bf16 GPU inject and bring BL-004
  (simdgroup_matrix for attention GEMM tiles) + BL-005 (native `bfloat` MSL type) — both filed
  "P3, M3+ only, NOT for M1 Max" (MEMORY.md:69-70,129) — into the matmul-heavy paths. **High
  PRODUCT value** (the app's core path runs GPU-native). DEPENDENCY: fix IP-ADAPTER-INFER-1
  FIRST (BUGS.md — `iris --ip` currently grids at scale>0); do not speed up broken generation.
  Must be developed/validated on real M5 (Metal kernels). Cross-refs: IP-ADAPTER-INFER-1,
  BL-004, BL-005, G-1 Phase 3.

- **HW-M5-5: precompute architecture DECISION — warm-resident server vs cache-free online
  (pick one; they're competing designs).** Today precompute and training are SPLIT into
  separate processes purely to fit 32 GB (encoders ~10 GB + Flux/adapter ~20 GB). 128 GB lets
  them coexist, opening two mutually-exclusive paths:
  - **(A) Warm-encoder precompute server** (extends HW-M5-1): keep Qwen3/VAE/SigLIP resident
    across iterations, kill the per-step `_ensure/_release_live_encoders` churn
    (train_ip_adapter.py:1376) and the 30×-per-campaign model reload. KEEPS the cache (and its
    cross-iteration reuse of re-staged shards) but removes reload/compile overhead.
  - **(B) Cache-free online-encode**: encode in the training loop, DELETE the precompute cache
    layer entirely — eliminating the whole cache-convention-bug class (VAE-Q1 BN-pack, the
    shard-score cross-campaign contamination, version/identity management, the
    precompute↔train resolution contract). HONEST TRADE: this is a SIMPLICITY/CORRECTNESS play,
    NOT a speed one — it MOVES encode into training and LOSES the cache's cross-iteration reuse
    (the flywheel re-stages high-scoring shards repeatedly; online re-encodes each time), so net
    compute can be WORSE for high-overlap campaigns. M5's 6–8× compute is what makes the
    re-encode affordable enough to consider. DEPENDENCY: MLX-1 (the online-encode SIGSEGV,
    BUGS.md) must be fixed — the online path is currently a known crash.
  - **Decision criterion:** if the cache-bug class keeps costing engineering time, (B) for
    correctness; if throughput dominates, (A). Don't build both. Cross-refs: HW-M5-1, MLX-1,
    VAE-Q1, PRECOMP-3, the "precompute↔train resolution contract" invariant.

- **HW-M5-6: rethink the flywheel DESIGN for the cheap-iteration regime (architecture, not
  scaling).** The flywheel's defining choices — subsample (DP-2c), from-scratch each iter,
  many short cheap iterations — are ADAPTATIONS to expensive iterations on constrained
  hardware (see memory `flywheel_throughput_strategy`). When iterations are no longer
  expensive-by-necessity (6–8× compute, batch>1 via HW-M5-2), the optimal shape may differ:
  fewer/larger/more-REPRESENTATIVE iterations, warm-start instead of from-scratch (the
  from-scratch regime exists partly because short runs warm-start poorly — re-examine with
  longer runs), or a more direct curate-then-train-once flow. This is a deliberate
  re-evaluation of whether the meta-flywheel's assumptions still hold, NOT a parameter bump.
  Do it AFTER HW-M5-2 lands (need the batch>1 regime to judge). Cross-refs: HW-M5-2, DP-2c,
  SREF-OPT-1 (the optimisation framework this restructures), from_scratch_each_iter.

- **HW-M5-7: retire the M1-era throughput workarounds (proxy-VAE + subsampling) — partly a
  DON'T-BUILD.** The proxy-VAE (PRECOMP-2; small variant FAILED Tier-1 on capacity, medium
  retrain PENDING) and subsampled precompute (DP-2c, `precompute_subsample_per_shard`) exist
  ONLY to make precompute affordable on M1. At 6–8× compute, full real-VAE precompute at full
  coverage is cheap → the proxy is likely MOOT. **Actionable now: do NOT sink a GPU-night into
  the medium proxy if M5 is near** — M5 obsoletes it. On M5: drop subsampling (full signal every
  iteration, removing the prefix-coverage bookkeeping) and re-evaluate whether the proxy path
  is ever worth its parity risk. Cross-refs: PRECOMP-2 (incl. the C-1/C-2 pre-trust gates that
  also become moot if the proxy is dropped), DP-2c, SREF-DATA-1 (subsampling interacts with
  curation).

- **HW-M5-8: parallel campaigns/ablation + multi-model residency (incremental memory wins).**
  A single GPU lock serializes everything today (one campaign/ablation arm at a time;
  ablation_harness.py acquires the lock). 128 GB fits 2+ concurrent ~20 GB runs → run multiple
  ablation arms / a source-probe + a quality campaign concurrently, multiplying the
  hyperparameter axis (SREF-OPT-1) throughput; needs GPU-lock redesign + contention handling.
  Also: keep multiple models resident (4B + 9B both, several adapters for ensemble/A-B,
  teacher-VAE + proxy together) for multi-adapter serving and inference A/B. Lowest priority of
  the set. Cross-refs: SREF-OPT-1, ablation_harness GPU lock, the dual-flywheel design.

---

## SREF Objective — Style-Reference Model (Midjourney --sref for the app)

> **Canonical usage guide: [docs/sref.md](docs/sref.md)** — which of the four shipped modes
> (Learned Style adapter / band-control / Style Library / in-context) to use when, the conventions
> (base-not-distilled, the collapse gate), and a status index of this section's `plans/sref-*` trail.

### 🟡 STYLE-LIBRARY-BROADEN attempt (2026-07-23) — NEGATIVE: tight CSD cluster ≠ transferable style LoRA
Tried to grow the retrieval library from 3 → 5 styles by CSD-clustering the hot pool (k=50), visually
curating two BOLD, non-duplicate candidates (sticker/kawaii c26; B&W ink/gothic c3), seed-curating tight
subsets (mean cos 0.638 / 0.674, well above the failed painterly 0.45) and training rank-16 base LoRAs
(400 steps, clean convergence, correct 74.7 MB export). BOTH FAILED the render quality gate and were NOT
shipped (release stays at 3 styles):
- **sticker** — die-cut stickers are characters on a WHITE BACKGROUND, so the LoRA learned "blank
  background": invisible at scale 1.2, wipes the image to a flat fill at 2.5. No usable range.
- **ink** — learned TONE (dark/desaturated), not the pen-and-ink LINEWORK; at 2.5 just a dark moody
  photo portrait, not the illustration look.
**Lesson (durable):** a tight CSD cluster does NOT imply a transferable style LoRA. Styles defined by
COMPOSITION/BACKGROUND (die-cut) or fine LINEWORK don't imprint as usable LoRAs on the frozen base; the
shipped 3 (cyberpunk/fantasy/graphic) work because they're bold COLOR/TEXTURE transforms that apply to
any subject. Trained LoRAs kept as artifacts on
`/Volumes/2TBSSD/sref_eval/lora_lib/{sticker_c26,ink_c3}_base.safetensors` (unreleased).

**WATERCOLOUR (2026-07-24) — also NEGATIVE, and it corrected the evaluation method.** Trained a
watercolour LoRA on a caption-curated, subject-diverse "watercolour" subset (the right kind of
color/texture style). Two evaluation lessons:
(1) **My visual render gate was MISCALIBRATED.** The SHIPPED, working cyberpunk LoRA at scale 1.2 on an
off-domain prompt ("a fox in a forest") ALSO renders fully photorealistic — the retrieval library LoRAs
are subtle and prompt-congruence-dependent, so "photorealistic at 1.2" is NOT evidence of failure. Pushing
scale to force visibility just produces ARTIFACTS (chaotic streaks), not clean style — a false-negative trap.
(2) **The PROPER gate is styleCSD Δ**, exactly as the library was originally validated. Measured:
watercolour LoRA styleCSD Δ = **−0.11** (output is LESS watercolour than base), vs the shipped cyberpunk's
**+0.176**. So it genuinely does not transfer. **Recipe is IDENTICAL to the shipped ones** (rank 16, 400
steps, ema 0.9999; configs `lora_cluster{8,23}_base_v1.yaml`), so this is NOT a recipe bug — my chosen
clusters (sticker/ink/watercolour) simply don't imprint a usable style DIRECTION while c8/c23 do. **Any
future attempt MUST gate on styleCSD Δ (target ≳ +0.10 vs base), not on eyeballing a render.** Watercolour
LoRA kept unreleased. Library stays at 3.

**CLUSTER COMPARISON — WHY c8/c23 work and mine don't (2026-07-24).** Compared the shipped training subsets
(`hot_style_clusters/cluster{08,23,09}_subset.json`) vs the failed ones on size / tightness / far-from-prior,
and re-ran styleCSD Δ on all three failures. Findings:
- **NOT the recipe** (identical), **NOT subset size** (all 250), **NOT far-from-prior** — REFUTED: the failed
  sticker (cos-to-prior 0.53) / ink (0.50) are FARTHER from the global CSD prior than the working
  cyberpunk 0.68 / graphic 0.62 / fantasy 0.75.
- **Two necessary conditions distinguish them:**
  1. **Cluster tightness (intra-cos of the trained 250) ≥ ~0.74.** Working: cyberpunk 0.861, fantasy 0.791,
     graphic 0.741. Failed: watercolour **0.524** (caption-matching gathered stylistically-diverse
     "watercolour" → too loose). This alone kills watercolour.
  2. **Style must be a content-agnostic COLOR/TEXTURE transform** (recolors/retextures ANY subject).
     sticker (0.730) and ink (0.723) are tight ENOUGH but the WRONG TYPE — sticker is a COMPOSITION style
     (die-cut character on white), ink is a TONE style (desaturation); neither imprints a direction that
     applies to arbitrary content. styleCSD Δ confirms: sticker **−0.038**, ink **+0.029**, watercolour
     **−0.114** (vs shipped cyberpunk +0.176; all < the +0.10 bar).
- **The real obstacle for right-type styles: CSD conflates MEDIUM with CONTENT.** A CSD-tight "watercolour"
  cluster (seed-curated) pulls FANTASY PORTRAITS (content match), not the watercolour medium; the
  caption-clean watercolour set has the right medium but is too loose (0.52). Landing a subset that is BOTH
  tight AND genuinely the medium is the crux. **Untried path:** intersect the two — take the caption-matched
  watercolour records (right medium), then keep only the CSD-tightest ~250 (raises tightness toward 0.74
  while staying watercolour). That is the one experiment that could still land a 4th style; gate on styleCSD Δ.

**INTERSECTION EXPERIMENT RUN (2026-07-24) — improved but still FAILS.** Built it: 969 caption-matched
watercolour records ∩ CSD-tightest 250 → intra **0.682** (up from the loose 0.507; montage confirmed clean,
VARIED watercolour medium — not collapsed to fantasy). Trained the identical recipe, gated on styleCSD Δ
(base vs LoRA on robot + fisherman @1.2): mean Δ **−0.0125** (robot +0.023, fisherman −0.048); the fisherman
render is photorealistic, no watercolour. So tightening moved the metric the RIGHT way as predicted
(v1 −0.114 → intersection −0.0125) but did NOT cross the +0.10 bar. **Conclusion: even the right MEDIUM +
tighter cluster (0.68) doesn't imprint** — landing a 4th style needs tightness ≥ ~0.74 AND the CSD-content
conflation makes a genuinely-watercolour cluster that tight hard to build. **Library-broadening effort
CONCLUDED for now** (3 style types tried, all metric-confirmed negative; the shipped 3 remain the library).
Reopening this needs a fundamentally tighter clean-medium subset (e.g. tightest-150, or a non-CSD style
signal) — not more of the same. All failed LoRAs kept unreleased on `lora_lib/`.

The end goal: a user uploads a reference image; generations adopt its STYLE (not
content) via the IP-adapter on Flux.2 Klein, served by the iris engine. Gap analysis
2026-06-10 (post Phase-2 / TRAIN-7 / held-out-cond_gap session):

### 🟢 SREF-EVAL-BROADENED (2026-07-13) — eval 11→23 refs; refreshed baseline OVERTURNS the "semi_real fails" conclusion (it was n=1 noise) and CONFIRMS generalization.
Broadened the style eval from 11 (5 graphic / 5 painterly / 1 semi_real) to 23 (10 / 8 / 5): CSD
farthest-cluster (`cluster_hot_styles.py --k 60`) + hand-curation from a contact sheet (dropped generic
photos + a desert content-confound), labeled by `type` + `source` (held_out fine-art vs hot_cluster
in-distribution). Backup `debug/sref_eval_set_v1.json`. Fixes SREF-EVAL-COVERAGE-GAP.
REFRESHED BASELINE — shipped C style-CFG adapter @ α=0.4, styleCSD Δ:
  graphic   n=10  +0.257  (held-out 0.299 / in-dist 0.216)
  painterly n=8   +0.105  (0.120 / 0.081)
  semi_real n=5   +0.102  (the OLD n=1 anime = −0.029; in-dist n=4 = +0.135)
  overall   n=23  +0.171
TWO CORRECTIONS: (1) "semi_real FAILS (−0.03)" (SREF-JOINT-STAGE0.5 / -V2 / -C-PORT caveats) was a
SINGLE-SAMPLE ARTIFACT — at n=5 it's +0.102; the adapter transfers semi_real fine. The thin eval had
produced a false negative. (2) GENERALIZATION CONFIRMED: held-out ≥ in-distribution on graphic AND
painterly → the adapter is not merely recalling trained styles (if it were, in-dist would win). Ranking:
graphic strongest, painterly/semi_real ~equal. Caveats: promptAdh prints 0 in the C-render scorecard runs
(scoring-path artifact; content is legible visually); leak Δ ~0.10 moderate. Now the per-type GROUND TRUTH
for SREF-STYLE-ROUTER step 1 + the train-more gate. Painterly breadth still data-limited (in-dist painterly
is DIGITAL-painterly; true fine-art painterly needs the WikiArt build).

### 🔴 SREF-STYLE-ROUTER — CLOSED 2026-07-21 (three independent negatives; the GENERIC v5.3.0 stands).
No current method beats the generic per-type (step 1). A pure-WikiArt painterly specialist v2 got strong
style (styleCSD 0.157) but overfit → destroyed prompt-following (promptAdh 0.084). The mixed-data
hypothesis (WikiArt + diverse to keep style while restoring content) was the last untried lever and
FAILED: v3 40:60 gated at painterly styleCSD Δ **0.0495** (< generic-painterly ~0.058 — the mix
over-diluted the style) with cross-ref corr max 0.98 (reference-inertness; in-loop `loss_b` discrimination
did not transfer to inference). promptAdh recovered (0.115) but style collapsed. ⇒ the specialist is NOT
the lever at any mix that preserves content; the router direction is closed. Full trail:
`plans/sref-mixed-painterly-specialist.md`. Original proposal (kept for the trail):

### 🔵 SREF-STYLE-ROUTER (PROPOSED, 2026-07-13) — classify the style reference (CSD) → route to the best method/expert per reference-type; confidence-gated with the generic adapter as the floor.
IDEA (operator): style-transfer methods have LARGE, MEASURED per-reference-type effectiveness gaps, so
classify the reference's style at inference (CSD — already computed) and route to the best method /
specialist LoRA for that type, instead of applying one method to everything. A mixture-of-experts over
style-transfer methods, gated by a cheap CSD classifier.

WHY IT'S REAL — this session's scorecards, styleCSD Δ per type, SAME eval set:
| method                                  | graphic | painterly |
|-----------------------------------------|---------|-----------|
| band-control (training-free)            | strong  | FAILS (SREF-STYLE-CEILING) |
| joint adapter, GUIDANCE-EMBED (Python)  | 0.05    | 0.22      |
| joint adapter, STYLE-CFG (the C ship)   | 0.30    | 0.12      |
| retrieval Style Library                 | strong where a trained LoRA exists | — |
Load-bearing datum: the SAME adapter, two inference MODES, has OPPOSITE per-type strengths → the routing
axis is METHOD/MODE, not only "which model." Band-control's painterly failure vs the adapter's painterly
strength is the other half.

SKELETON ALREADY EXISTS: the retrieval Style Library IS a CSD→per-style-LoRA router (`resolve_style_lora`,
plans/sref-retrieval-hybrid-project.md). This generalizes it to route across METHODS/experts. Routing is
CHEAP: CSD(ref) is already computed at inference (library + adapter); style-type classification on CSD is
a tiny model (we already cluster by CSD — `train/lora/cluster_hot_styles.py`). No new encoder, no
per-image cost.

KEY DESIGN — CONFIDENCE-GATED, generic adapter as the FLOOR: route to a specialist only when the
classifier is confident; else fall back to the v5.3.0 generic adapter (SREF-JOINT-C-PORT — handles
anything). Then the router is NEVER worse than what shipped → bounded misroute risk + universal coverage.

CAVEATS: (1) do NOT build N specialists up front — each is a training run + tight look-coherent data
(DATA-SELECTION PRINCIPLE); the obvious first specialist, painterly, needs the WikiArt build (the deferred
painterly lever, thin today). (2) eval set is only 11 refs (5 graphic / 5 painterly / 1 semi-real) — too
thin for rigorous per-type routing; broaden first (SREF-EVAL-COVERAGE-GAP). (3) router quality caps the
whole thing, but CSD retrieval already discriminated held-out refs 3/3 → low risk.

MAP STEP-1 RESULT (2026-07-13, 23-ref set) — REFRAMES the router: no current method beats the adapter.
styleCSD Δ, DEPLOYABLE methods with comparable neutral baselines: graphic — band-control 0.108 vs
style-CFG adapter **0.257**; painterly — 0.036 vs **0.105** (band-control fails painterly, SREF-STYLE-CEILING);
semi_real — 0.104 vs 0.102 (TIE). So the shipped adapter is already the BEST deployable method on every
type; band-control only ties semi_real (its edge = free). ⇒ a router OVER EXISTING methods gains ~nothing;
the router's value is ENTIRELY in FUTURE SPECIALISTS that beat the generic floor (→ build the painterly
specialist FIRST; a router earns its keep only once a specialist wins its type). METHOD CAVEATS: (a) drop
guidance-embed's numbers — its "baseline" was a styled painterly image (not neutral) → inflated Δ, AND it
destroys content (Python-only, not deployable). (b) styleCSD Δ ALONE is misleading (rewards over-styling);
a valid map needs a COMMON baseline + a working CONTENT metric (promptAdh printed 0 — scoring-path artifact
to fix). Tools: `debug/cluster_csd_candidates.py` (generalized clustering for eval broadening).

SPECIALIST ATTEMPT 1 (2026-07-15) — FAILED (corrupted generation), but CONFOUNDED by a reused projector →
INCONCLUSIVE, + a real lesson. Built a painterly specialist on-device: 12 WikiArt shards → VAE256/qwen3/CSD
+ within-movement pairs (22,945; NN/random 0.58); trained the winning recipe 8000 steps. In-loop
discrimination was the CLEANEST yet (foreign-row acc pinned 1.000, pair→0.04). BUT the render gate on the 12
fine-art painterly refs: specialist styleCSD Δ **0.028** vs generic **0.058** — generic WINS on its own turf,
AND the specialist outputs are GLITCHY GARBAGE (blocky speckle), broken at every α (0.15/0.25/0.4) AND with
NO csd (base+specialist-LoRA baseline is already corrupt) → the LoRA itself broke generation, not the
csd_mod/amplification. ROOT CAUSE: the specialist reused the GENERIC's journeydb-fit latent→CSD projector on
WikiArt latents (flagged in the config). At high-t (recon weak) the contrastive dominates, so a mis-specified
projector target pushed the LoRA to produce latents that score in the WRONG space but decode to noise.
**LESSON (banked): the latent→CSD projector is DATA-SPECIFIC — retrain it per corpus; do NOT reuse across
data.** So the router's specialist premise is UNTESTED, not disproven. But combined with the MAP result (no
current method beats the generic) the working hypothesis is that the GENERIC adapter is the robust answer and
specialists are high-effort/uncertain. A clean re-test = retrain latent_csd projector on WikiArt latents →
retrain specialist → re-gate (~1.5 days). Artifacts: /Volumes/2TBSSD/checkpoints/sref_painterly_specialist,
/Volumes/2TBSSD/precomputed/vae_wikiart256 + qwen3_wikiart, wikiart_csd, wikiart_neighbors.sqlite.

SPECIALIST ATTEMPT 2 (2026-07-16) — clean re-test with a WikiArt-specific projector. The projector fix
WORKED (un-corrupted), but exposed a DEEPER failure → nuanced NEGATIVE. Trained a WikiArt latent→CSD
projector (val cos 0.83, geometry 98% retained) and retrained the specialist (v2) against it. Results:
(1) COHERENCE RESTORED — the projector diagnosis was right: v2 renders are coherent (not v1's garbage);
the base+v2-LoRA baseline is a coherent painterly scene. (2) STRONG STYLE — v2 fine-art painterly styleCSD
Δ **0.157** vs generic 0.058 vs corrupted-v1 0.028 (reference-specific, above its own always-painting
baseline). BUT (3) SUBJECT DESTROYED — promptAdh **0.084** (vs generic 0.327); the "robot in a desert" is
absent at α=0.4 AND at α=0.15/0.25 AND even at α=0 (no csd). The subject loss is in the LoRA WEIGHTS, not
the csd_mod scale → un-fixable by α. ROOT CAUSE: training ONLY on WikiArt paintings OVERFIT the LoRA to
"produce a painting," destroying prompt-following. **LESSON 2: narrow-data specialization overfits and
loses prompt-following; the generic's DIVERSE data (incl. photorealistic) is what preserves the subject.**
VERDICT: the router/specialist direction, tested rigorously TWICE, does NOT yield a usable specialist via
"train only on the type." The GENERIC adapter is the robust answer. A "done-right" specialist would need
MIXED data (WikiArt + diverse content, or a diverse-recon anchor / lower rank / regularization) — an
uncertain further experiment. Artifacts kept: checkpoints/sref_painterly_specialist_v2,
checkpoints/latent_csd_wikiart, the WikiArt caches.

STAGED PLAN (cheapest wins first):
1. MEASURE THE MAP — scorecard per style-type × per-method on a BROADENED eval set → the actual
   "CSD-region → best-method" table (router ground truth AND shows where a specialist even helps).
2. ROUTE EXISTING METHODS (near-zero cost, no new models) — CSD-confidence router over band-control (free)
   vs the style-CFG adapter, + a PER-TYPE α (graphic wants higher strength, painterly lower — measured).
   Add a guidance-embed MODE iff step 1 says painterly needs it (needs the deferred C guidance-embed
   single-forward path, not yet built — SREF-JOINT-C-PORT fork option 2).
3. BUILD THE HIGHEST-VALUE SPECIALIST (likely painterly) only where step 1 shows the biggest gap; A/B vs
   the generic adapter before adding it to the router.
v1 = a router over the methods we ALREADY have, NOT a fleet of new models. Specialists follow the map.
Possible upgrade: fit a small CSD→best-method meta-model directly on the step-1 scorecard results rather
than hand-coding style-type buckets.

Refs: SREF-JOINT-C-PORT (the generic floor / v5.3.0), retrieval-hybrid (the CSD→LoRA skeleton),
SREF-STYLE-CEILING (band-control's painterly ceiling), SREF-EVAL-COVERAGE-GAP (eval breadth),
[[data_selection_principle]] + WikiArt painterly lever (specialist data).

### 🟢 SREF-JOINT-C-PORT (2026-07-13) — adapter reimplemented in C AND WORKING via STYLE-CFG. Generic style adapter now runs in the shipped iris/Metal engine + discriminates references at the pixel level.
**RESOLVED via style-CFG (option 3).** csd_delta is injected ONLY in the conditional forward
(`iris_sample.c` raises `tf->csd_skip` on the uncond pass via `iris_transformer_set_csd_skip`), so
`v = v_u + g(v_c − v_u)` AMPLIFIES the style instead of cancelling it. Result @512px, prompt "a robot
standing in a desert", seed 7: strong content (crisp robot from prompt-CFG) + reference-specific style,
α-controlled. **Discrimination proven:** woodcut ref → dark monochrome ink on paper; impressionism ref →
light silvery painterly shimmer — SAME seed/prompt, visibly different styles. **α range shifts DOWN** (CFG
~g× amplifies): usable 0.2–0.4, over-amplifies to blank by 0.85; sweet spot **α≈0.35–0.4** (≈ Python's
0.85 / 3.5). Default locked at **α=0.4** (C scorecard: styleCSD Δ graphic 0.30 / painterly 0.12 /
overall 0.19 — style-CFG RESCUES graphic, the fragile Python type). **SHIPPED to web** (2026-07-13):
`train/lora/dump_csd.py` (image→768-f32 CSD, matches the training encoder cos 1.0) + `web/server.py`
`resolve_sref_adapter()` + a new **"adapter"** reference mode (routing → job → generate() sets
`lora=joint_lora + sref_csdmod + sref_csd + sref_scale`, forces CFG) + the "Learned Style" UI option
(index.html/app.js). Env `IRIS_SREF_ADAPTER_DIR` / `IRIS_SREF_ADAPTER_SCALE` (0.4); needs the resident
daemon on flux-klein-4b-base. Web tests green (no regression). Original port details:
The joint adapter (LoRA r64 + CSDModulation) now runs in the shipped C/Metal `iris`:
- **Export** `train/lora/export_joint_to_c.py`: joint ckpt → Diffusers-named `joint_lora.safetensors`
  (strip `flux.`, add `.weight`, double `to_out`→`to_out.0`) + `csd_mod.safetensors`. LoRA loads via the
  EXISTING `--lora` (iris_lora.c already does Diffusers double+single) — "loaded 80 adapters", zero new code.
- **New C**: `iris_csdmod.c/.h` (fc1→silu→fc2 temb delta) + temb injection in all 3 flux forwards
  (`iris_transformer_flux.c`, `tf->csd_delta`, NULL-guarded = bit-identical off) + `iris_set_sref_csd()`
  (iris.c) + CLI `--sref-csdmod/--sref-csd/--sref-scale` (main.c). **Parity guard** `debug/test_csdmod.c` +
  `gen_csdmod_fixture.py`: corr 1.000000, max_abs **3e-08** under production flags. `make mps` clean.
  Artifacts in `/Volumes/2TBSSD/sref_eval/joint_v1_c_export/`. CSD encode stays in Python (pass the 768-vec).
- **THE ISSUE (guidance mechanism):** C base model = **CFG** (2 forwards, `v=v_u+g(v_c−v_u)`). csd_delta is in
  temb of BOTH forwards → the style is common-mode and **cancels in the CFG difference**. Measured @512px
  impressionism ref: g=3.5 → crisp robot, NO style; g=1.0 (pure cond) → strong impressionist style, NO
  content. Training/Python used the **guidance EMBEDDING** (single forward, guidance in temb, no CFG) → got
  BOTH. The C has no guidance-embedding path (base=CFG, distilled=guidance pre-baked).
- **FORK (open):** (1) ship `-g 1.0` (works now, weak content, zero code); (2) add guidance-embedding
  single-forward for base weights (matches training exactly; re-architects base inference, needs base
  guidance_embed weights); (3) inject csd_delta ONLY in the conditional forward under CFG → CFG AMPLIFIES
  the style (the "style-CFG amplifier" the plan wanted; needs a forward-side cond/uncond flag; α re-tunes
  for the ~g× amplification). (3) is likely the best result/effort; (2) is the faithful match.

### 🔴 SREF-JOINT-V2-CONTENT (2026-07-12) — wider t-band is the WRONG content lever. NEGATIVE: it weakened style and did NOT improve content. v1 recipe stays champion.
Hypothesis (v2, `sref_joint_v2_content.yaml`): the v1 render over-weighted style (promptAdh 0.19), so widen
`t_range` [700,950]→[200,950] (low/mid-noise steps train content-recon) + raise `null_style_prob` 0.1→0.25.
Trained 10k steps on the EXPANDED data (see below): in-loop discrimination STILL stabilized (final diags:
acc pinned 1.000, pair to 0.006, cos(z_a,z_b) to −0.23) — so the recipe change did NOT break discrimination.
BUT the render gate (ckpt 7000, same harness) REGRESSED on what matters:
- styleCSD Δ overall **0.114 → 0.053** (painterly **0.216 → 0.076**, −65%); visually the styles are MUTED
  (impressionism = washed-out haze vs v1's committed pastel brushwork; woodcut = gray relief vs stark ink).
- promptAdh **0.190 → 0.178** — NO content gain (slightly worse). cross-ref output corr 0.57 → 0.28 (more
  varied but weaker style). Method caveat: the gate reused the v1 baseline PNG (existed, no --regen), so v2's
  null-baseline coherence wasn't assessed; the styleCSD Δ *comparison* is valid (same baseline).
- **Lesson:** widening the band just dilutes the high-noise style-learning steps without buying prompt
  adherence — the content weakness is the model prioritizing style OVER the prompt subject, which the band
  doesn't touch. Content-vs-style is NOT a t-band slider. **Next lever = INFERENCE-side style-scale**
  (`temb += α·csd_mod(csd)`, sweep α on the WORKING v1 ckpt-7000) — trades style↔content with no retrain and
  yields a per-generation user knob. Champion remains v1 `joint_probe_0007000.safetensors`.
- **✅ STYLE-SCALE RESULT (2026-07-13): the inference knob WORKS — solves weak-content with no retrain.**
  Added `--style-scale α` to `debug/sref_gate_joint.py` (`temb += α·csd_mod(csd)`); swept α on v1 ckpt-7000
  (coarse 0.3/0.5/0.7 → fine 0.75–0.95 → two-decimal 0.80–0.90). Clean monotonic style↔content slider
  (styleCSD Δ / promptAdh): α=1.0 0.114/0.190 · 0.90 0.104/0.204 · **0.85 0.092/0.208** · 0.80 0.082/0.211 ·
  0.70 0.052/0.208 · 0.50 −0.041/0.227 · 0.30 −0.138/0.239. Content saturates by ~0.85; style declines
  gently; **graphic** styleCSD crosses 0 at α≈0.77 (graphic is the fragile type — painterly stays strong
  throughout, 0.15–0.22). **DEFAULT α≈0.85** (81% of full style, graphic safely +, subject legible, visually
  confirmed on anime/impressionism/vector); expose [0.70–1.0] as the user slider. This is the on-device fix:
  ship v1 ckpt-7000 + a style-strength slider, no cloud, no retrain. (Method note: renders land in
  `/tmp/sref_scorecard/` — /tmp got day-rollover-cleaned mid-sweep, losing the baseline PNG for the last
  point; move gate output to a persistent dir when this is productionized.) NEXT = wire into web/server.py as
  a generic "upload a reference" mode (MLX inference path: base + LoRA r64 + csd_mod + α), like the Style
  Library but reference-generic.
- **DATA (kept, positive):** expanded CSD+VAE coverage 200→5000/shard across the 22 hot shards →
  usable training pairs **16,656 → 101,771 (6.1×)**, 0 neighbor slots dropped (was 455,406). Isolated caches
  `/Volumes/2TBSSD/universe_csd_full` + expanded `/Volumes/2TBSSD/precomputed/vae_sref256px` (READMEs). qwen3
  already covered 5000/shard. This unlock is reusable for any future on-device run.

### 🟢🟢 SREF-JOINT-STAGE0.5 (2026-07-11) — STAGE-0.5 GATE PASSED. First learned generic style-reference adapter to break collapse on this stack (4th attempt). GREENLIGHT cloud Stage-1.
The joint-backbone recipe (`plans/sref-joint-backbone-project.md`: TRAINABLE LoRA r64 backbone +
content-shared-PAIR contrastive with a FOREIGN-ref branch + high-noise `t∈[700,950]`) is the first that
does NOT collapse to a reference-inert constant. 8000-step probe, 256px, batch-1 ×2 split-backward,
~4 s/step, peak 16.6 GB. Checkpoint `/Volumes/2TBSSD/checkpoints/sref_joint_probe/joint_probe_0007000.safetensors`
(160 LoRA + 4 csd_mod tensors). All three gates:
- **Gate 1 (in-loop discrimination) — PASS, and STABILIZED.** Early (steps 100–1500) the foreign-row acc
  oscillated 0.5↔1.0 and `pair` bounced across the 0.6931 ref-blind floor; by steps 7000–7900 acc is
  **pinned at 1.000** and `pair` is **consistently 0.04–0.39 (all below floor)**, cross-ref x0 corr < 0.9 in
  7/10. cos(z_a,z_b) drops to 0.03–0.18 at the strong steps. Prior 3 adapters were permanently pinned at
  corr≈1.0 / acc 0.5.
- **Gate 2a (render cross-ref output corr < 0.90) — PASS.** 11 refs / 55 pairs, seed+prompt fixed:
  **mean 0.5715, max 0.8856, min 0.2332**. (Champion collapse was ≥0.984.) The renders are DISTINCT,
  COHERENT stylized images — monochrome-ink (woodcut) / soft-pastel (impressionism) / anime / cubist /
  bright-vector — not different noise.
- **Gate 2b (styleCSD Δ > 0.009) — PASS.** overall **0.1143**; painterly **0.216** (strong), graphic
  **0.051**; semi_real −0.077 but that's **n=1** (anime, which visually looked BEST — kept content, so its
  CSD moved less vs the degenerate null baseline; a metric artifact, not a style failure).
- **Tooling:** render gate = `debug/sref_gate_joint.py` (injects LoRA r64 + csd_mod from the joint ckpt into
  the SAME `flux_forward_film` forward the probe trained; reuses FILM harness generate/BN-unpack/CSD encoder —
  no train↔infer divergence). Scorecard `debug/sref_scorecard.py --score-only`. Renders in
  `/tmp/sref_scorecard/joint_7000/`.
- **Caveats / Stage-1 tuning axes (honest):** (1) CONTENT/prompt adherence is WEAK — promptAdh 0.190, the
  "robot in a desert" is only faintly legible except in anime; the model over-weights style. Levers: lower
  csd_mod/LoRA scale at inference, widen the `t` band to include low-noise (so it practices content), more
  recon weight, add the decoupled cross-attn token path (plan Stage 2). (2) The NULL-CSD baseline DEGENERATES
  to a texture (blue stripes) — the model trained almost always WITH a ref (null_style_prob 0.1), so the
  unconditional path is undertrained → styleCSD Δ (measured vs baseline) is noisy; raise null_style_prob if
  CFG is wanted, and/or gate on absolute CSD(out,ref). (3) composition leak moderate (0.055; graphic 0.095).
  (4) eval set small (11 refs, semi_real n=1) — broaden it (SREF-EVAL-COVERAGE-GAP). (5) 512px inference of a
  256px-trained LoRA renders coherently (good resolution generalization). Checkpoint used is 7000 (loop
  breaks before an 8000 save); the late diags show 7000 is the converged state.
- **Verdict:** the "learned style adapters are dead" conclusion is OVERTURNED. Proceed to cloud Stage-1
  (full recipe @512px, batch 32, ~50–100k steps) per plan §4 — the real product go/no-go — with content
  adherence as the #1 thing to tune. Retrieval-hybrid remains the shipped baseline until Stage-1 ships.

### 🟢 SREF-LATENT-CSD-PROJ (2026-07-10) — latent→CSD projector TRAINED and PASSES its gate. The contrastive's `t` band is [700, 950], NOT Fable's `t<0.7`.
Built the prerequisite for the content-shared-pair contrastive (SREF-PAIR-VS-BANK): `z = proj(x0_pred)` must
live in CSD space so it can be compared to the frozen `CSD(ref)`. `LatentCSDProjector`
(`train/ip_adapter/latent_csd.py`, 4.46 M params, conv + multi-scale mean/std pool → 768-d, L2-normed)
distilled from the ~104k cached `(vae latent, CSD)` pairs (`precomputed/vae/v_2232c1` × `universe_csd`,
520 populated shards). 20 000 steps, batch 64, ~9 it/s, ~35 min, no 4B DiT in the loop.
Trainer `train/lora/train_latent_csd_projector.py`; ckpt `/Volumes/2TBSSD/checkpoints/latent_csd/`.
  • Held-out `cos(proj, CSD)` = **0.8353** full-res (64×64 latent), **0.7936** at a 32×32 crop (the 256px
    training resolution) — resolution-agnostic as designed.
  • **GEOMETRY GATE PASSED (the real one):** `cos(proj(target), CSD(style-neighbour))` **0.7151** vs
    `cos(proj(target), CSD(foreign))` **0.1423** → separation **0.5728**, i.e. **90% of the true-CSD
    separation (0.6332)** retained. The pair loss has the gap it needs to push on.
  • **Val cosine is NOT a sufficient gate — it has the same collapse shape one level down.** At 20 smoke
    steps the projector scored a respectable val cosine **0.44** while retaining **0%** of the separation:
    it was emitting the mean CSD direction. Always read the geometry check.
**NOISE BAND RESOLVED (`debug/sref_noise_band.py`).** Fable's review said restrict the contrastive to
`t < 0.7` because "x0_pred is mush at high noise". Both halves of that are wrong here:
  • *Analytic.* `x0_pred = (α·noisy − σ·v_pred)/(α²+σ²)`, so the INPUT pins `leak = α²/(α²+σ²)` of it and
    the model's authority is `∂x0_pred/∂v_pred = σ/(α²+σ²)`. leak/gain: t=300 → 84.5%/0.517;
    t=400 → 69.2%/0.769; t=600 → 30.8%/1.154; **t=700 → 15.5%/1.207 (peak authority)**; t=900 → 1.2%/1.098.
    **Below t≈400 the model CANNOT move style even if it reads the reference** — the contrastive has no lever
    exactly where Fable would put it.
  • *Empirical.* Readability of `proj(x0_pred)` is **flat in t**: separation at a realistic `v_pred` error
    (e=0.5) is 0.6608 @t300 vs **0.6499 @t900** (−1.6%). `x0_pred` is LINEAR in `v_pred`, so a competent model
    can emit any `x0_pred`; "mush at high noise" is a warmup concern about an untrained `v_pred`, not a
    structural bound. The only real trade-off shows at e=0.75, where the minimum is **at t=700** (0.5897) —
    peak authority also amplifies `v_pred` error most — recovering to 0.6089 @t900.
  • **→ Sample `t ∈ [700, 950]`, centred ~850** (leak ≤5.9%, authority 1.05–1.21, best in-band readability).
    Consistent with the SREF-STYLE-CFG-PROBE root cause (style is set at high noise), against Fable's item 4/§
    "restrict to low/mid noise".

### 🟢 SREF-PAIR-VS-BANK (2026-07-10) — the anti-collapse pressure lives in the FOREIGN-ref row. A negative bank alone does NOT supply it; a content-shared PAIR does. Batch-1 memory still suffices (split backward).
Second opinion (Fable) independently reached the corrected positive (`z_i` vs `CSD(ref_j)`, not vs the sample's
own target — see SREF-INFONCE-VOID) and concluded: since the reference-side encoder (CSD) is FROZEN, negatives
need no gradients, so a precomputed **negative bank** gives full InfoNCE at **batch 1** → "the M1 wall is an
artifact of the objective, not the hardware." Fable's anti-collapse argument: "a constant output has fixed
similarities to every `s`; it cannot align with `s_i` better than the negatives." **That argument kills the
CONSTANT solution but not OUR failure mode**, which is reference-blind yet strongly INPUT-dependent: a good
denoiser recovers `x̂0 ≈ x0_i`, and `r_i` is a same-STYLE neighbour of `x0_i`, so `CSD(x̂0) ≈ CSD(r_i) = s_i`.
It wins the bank without ever reading the reference.
Measured on the REAL CSD index (`sref_eval/style_cache`, 24 shards) + the REAL look-pairing
(`neighbors.sqlite`), 512 queries, 1023 bank negatives, τ=0.1 (chance CE = ln 1024 = 6.9315):
| model | Fable's batch-1 bank (correct ref only) | content-shared PAIR, foreign-ref row |
|---|---|---|
| REF-AWARE (adopts the handed style) | 0.4665 / acc 99.6% | 0.4342 / acc 99.4% |
| **REF-BLIND (denoises, ignores ref)** | **1.7766 / acc 98.0%** | **8.0978 / acc 0.0%** |
| CONSTANT (degenerate) | 8.6923 / acc 0.0% | — |
| RANDOM | 6.9755 / acc 0.0% | — |
  • **`cos(CSD(target), CSD(its style-neighbour)) = 0.7526`** — the reference is REDUNDANT with the target the
    model is already denoising. That single number is why the correct-ref row is nearly free (98% top-1 while
    fully collapsed — a NEW false-positive trap: bank top-1 accuracy looks great on a collapsed model).
  • **`cos(CSD(target), CSD(foreign ref)) = 0.1205`** — the FOREIGN-ref row cannot be reached by denoising. It
    is the only term a reference-blind model cannot satisfy. Pair-only (2-way softmax, chance 0.6931):
    ref-aware **0.0006** vs ref-blind **3.1660**; the foreign row alone is 0.0006 vs **6.3266**, acc 0%.
  • Fable's item 6 ("swap loss = a strict special case of the bank; skip as a training objective") is therefore
    **inverted**: the swap/pair IS the load-bearing term; the bank is the cheap bonus. Fable's item 1 ("the bank
    dissolves the wall — start here") is necessary-but-not-sufficient.
  • Bank negatives DILUTE the pair: they are easy (random style), so the softmax is dominated by them and the
    a-vs-b margin contributes little. Keep the pair as a 2-way (hard-negative) term; add a modest bank
    separately if at all.
**THE WALL STILL FALLS, for a different reason.** The two rows DECOUPLE — row A's softmax touches only `z_A`
plus frozen descriptors, row B's only `z_B`. So the pair is two SEQUENTIAL batch-1 forwards/backwards with
summed grad trees (Fable's own hygiene item 5b, "split the backward"), i.e. **batch-1 memory, 2× compute** —
no co-resident batch, no GradCache, no 32 GB graph. The joint-backbone falsification IS M1-feasible.
Corollaries adopted from Fable (sound, independent of the above): frozen-encoder negatives need no momentum
encoder (no MoCo staleness); `mx.eval` the grad tree every micro-step or the lazy graph regrows the stall;
don't `mx.compile` the 25-block training step; keep norms/adaLN/modulation in bf16 if quantizing (4-bit base
only when 512px or micro-batch >1 is wanted); train the mechanism at 256px, gate at 512px, never below 256.
**PREREQUISITE (new, unbuilt):** `z = proj(x̂0)` must live in CSD space, but our `style_stats` is AdaIN stats on
VAE latents — a DIFFERENT space; the two are not comparable. Either decode+CSD inside the graph (~2–3 GB) or
distil a **latent→CSD projector** offline from the cached `(vae latent, CSD)` pairs we already have
(`precomputed/vae/v_2232c1` × `universe_csd`, ~200/shard × 1281 shards). The projector is the cheap option and
is independently useful as an in-loop style metric. **Open tension to resolve with it:** the contrastive needs
a mid-noise band — at low `t`, `x̂0` is dominated by the input leak (`α²/(α²+σ²)` = 30.8% at t=600) so the model
cannot move style even if it reads the ref; at high `t`, `x̂0` is mush. Fable says t<0.7; our root cause says
style is set at HIGH noise. Sweep and measure whether a ref-AWARE oracle can even win row B at each `t`.
Sims: scratchpad `bank_infonce_test.py`, `pair_vs_bank.py` (throwaway; numbers above are the artifact).
Nothing trained; no GPU touched.

### 🟠 SREF-INFONCE-VOID (2026-07-10) — the Stage-0.5 in-batch InfoNCE CANNOT punish reference-blindness. The probe would have returned a FALSE NO-GO.
Audited `train/lora/probe_joint_contrastive.py:155-159` before spending the (wedged) batched run. The term is
`logits = l2(style_stats(x0_pred)) @ l2(style_stats(latent)).T / τ`, `labels = arange(B)`. **The reference
(`csd`) does not appear in the loss at all** — the positive for prediction `i` is sample `i`'s OWN target.
The design doc's claim ("a constant output cannot classify → collapse punished") conflates **output-constancy**
with **reference-independence**. The collapse we actually have is reference-independent but strongly
INPUT-dependent — and such a model classifies perfectly.
Numeric proof on 8 REAL VAE latents (`v_2232c1`), τ=0.1, B=8, chance CE = ln 8 = 2.0794:
| v_pred regime | t=900 | t=600 | t=200 |
|---|---|---|---|
| (a) `v_pred = v_target` — PERFECT but reference-blind | **1.2413 / acc 100%** | 1.2413 / 100% | 1.2413 / 100% |
| (b) `v_pred = const` — total degenerate collapse | 2.0777 / acc 12% | 1.6048 / 100% | 1.2563 / 100% |
| (c) `v_pred = v_target + 0.5·noise` — sloppy, reference-blind | 1.2928 / 100% | 1.2966 / 100% | 1.2451 / 100% |
  • **(a) is the killer, and it is FLAT in t (1.2413 at every noise level):** the InfoNCE global minimum is
    attained by a good reference-blind denoiser. The term rewards DENOISING FIDELITY, not reference use →
    **zero anti-collapse pressure**, at any timestep. The high-noise bias cannot fix this.
  • (b) only fails at high noise because of a second, separate defect: `x0_pred = (α·noisy − σ·v_pred)/(α²+σ²)`
    leaks `α²/(α²+σ²)` of the target straight from the model's INPUT (30.8% at t=600, 1.2% at t=900). So at
    t≤600 even a constant `v_pred` classifies 100%. `highnoise_bias=1.0` (median t≈730) masks THIS leak only.
  • Third pathology: at the recon optimum the CE gradient is still large (off-diag style-stat cosines ~0.9 on
    natural latents), so `λ_c=1.0` (vs recon ~0.3) actively pushes `x0_pred` style-stats to be MORE mutually
    orthogonal than the true targets — a content-driven repulsion that degrades recon and still never consults
    the reference.
**CONSEQUENCE:** the 8000-step Stage-0.5 run would have collapsed and been read as "in-batch contrastive
doesn't break it ⇒ the loss-bound verdict is ironclad, learned adapters are dead" — a false NO-GO on the
flagship hypothesis, from a test that never applied the pressure it advertised. The M1-vs-cloud blocker was
therefore being debated for the WRONG objective.
**THE FIX (mirrors the collapse METRIC, which is correctly specified):** contrast over REFERENCES with CONTENT
HELD FIXED. Share one noisy latent `x_t` across the row set, forward it with B distinct refs, and set
`logits[i,j] = sim(style(x0_pred_i), desc(ref_j))/τ`, labels = diag. A reference-blind model then emits
identical rows → chance accuracy → loss = ln B = maximum penalty. Sharing `x_t` also cancels the α·noisy leak
(identical for every i, so it cannot discriminate). Two structural corollaries:
  1. **It does NOT need batch 8–16.** The minimum unit is a content-DUPLICATED pair (same `x_t`, refs a≠b) —
     `style_repulsion_loss` (`train/ip_adapter/loss.py:104`) is already this shape. Effective batch 2.
  2. **Extra negatives are FREE (MoCo, brief item 1) because negatives are reference DESCRIPTORS, not
     forwarded outputs** — only the positive needs a forward. A queue of CSD vectors gives many negatives at
     batch 2. But the queue is a BONUS; the load-bearing term is the content-shared pair (brief item 6).
Also: data is ~74% self-paired (ref = the target's own image), which makes the reference redundant with the
noisy input — the exact shortcut that produces collapse. The corrected run must be **100% cross-paired**
(ref = a different image of the same style).
Sim: scratchpad `infonce_shortcut.py` (throwaway; numbers above are the durable artifact). Nothing trained.

### 🔴 SREF-FILM-1 (2026-07-10) — CSD→modulation (experiment B) COLLAPSES too. The collapse is LOSS-bound, not channel-bound.
Built + trained the FiLM rail A's probe pointed at: `CSDModulation` (768→1024→3072, adaLN-zero) FiLMs the
content-invariant CSD vector into the DiT's timestep-modulation embedding (`temb += csd_mod(csd)`) — the
UNIGNORABLE adaLN channel that scales every block's norm at every noise level, not the optional token stream.
DiT frozen, 4000 steps on the 4B base, `cond_mode=csd` on hot `universe_csd` + look-pairing (~26% cross).
Loss fell 0.47→0.28 (DEEPER than the token projector's ~0.4 plateau — the channel IS used strongly). Gated the
step-4000 ckpt (`debug/sref_infer_film.py` → `sref_scorecard.py`, base 24-step, seed 7, "a robot in a desert"):
  • **HARD-KILL COLLAPSE: cross-ref output corr 0.9998 (max 1.0000, min 0.9990)** — every one of 11 wildly
    different refs → near-identical output. WORSE than the token projector (0.98).
  • Scorecard styleCSD Δ painterly **0.0603** (graphic 0.010, semi_real −0.14, overall 0.019) — this BEATS
    band-control 0.009 but is a **CONFOUND** (textbook SREF-CHAMPION-COLLAPSE): a constant mildly-painterly
    transform correlates with painterly centroids. NOT reference-conditioned transfer. The mandatory collapse
    gate (corr ≥0.90 = false positive) fires → the 0.0603 is void.
  • **Diagnostic (rules out harness bugs):** module DID train (`fc2.weight` off zero-init, std 0.015); CSD
    inputs ARE distinct (cross-ref cosine 0.08–0.47); YET the `temb` deltas are near-identical across refs
    (cross-ref cosine **0.9998**, norm ~44). The module learned to map WILDLY different CSD inputs → the SAME
    large constant temb shift. It uses the channel hard, but constantly.
**ROOT CAUSE (durable, now proven across TWO channels):** the collapse is NOT the injection site — it's the
flow-matching objective. A noised-target loss on a FROZEN DiT never rewards reference-DISCRIMINATION, so the
easy minimum for ANY learned reference→conditioning module is a reference-INDEPENDENT constant — whether
injected as attention tokens (SREF-STYLE-STAGE1 / -CFG-PROBE) or as adaLN modulation (here). The "unignorable
channel" hypothesis was wrong: applying the channel forces it to be USED, not to be reference-DEPENDENT; the
module just puts a constant in it. This reconciles the whole SREF arc: the two things that WORK sidestep this
loss — in-context VAE-ref tokens ARE the image (the loss cannot reconstruct without them at high noise), and
per-style LoRA is direct weight surgery (no reference-mapping to collapse). → The learned-conditioning-module
direction is DEAD on this stack (both channels). Only remaining instant-generic bet = **C (hypernetwork→LoRA)**,
and ONLY if trained by DISTILLING known-good per-style LoRA weights (supervised regression reference→weights),
NOT the collapsing flow loss — else it collapses the same way. Retrieval-hybrid stays the shipped answer.
Artifacts: `/tmp/sref_scorecard/film_4000/`, ckpts `/Volumes/2TBSSD/checkpoints/sref_film/`. Scaffolding
committed e56d4a6; gate result here.

### 🔴 SREF-STYLE-CFG-PROBE (2026-07-09) — style-CFG CANNOT rescue the learned projector on base. Experiment A CLOSED.
Re-opened the learned-encoder death sentence in the BASE context: the Stage-1 gate ran a SINGLE forward with a
guidance EMBEDDING, but both models are `guidance_embeds:false` (true CFG) — so the style tokens were never
CFG-amplified (`debug/sref_infer_style.py:31-32` flags this open Q). The one base-only lever the distilled model
structurally lacks (true two-pass CFG `v = v_null + w·(v_cond − v_null)`) was never applied.
Ran a cheap velocity-space PRE-CHECK before spending render heat (scratchpad `vel_probe.py`): shared 8-step NULL
trajectory for realistic `x_t` at 4 noise levels, then per-ref `v_cond` vs shared `v_null` on the 4000-step
projector, 11 eval refs. **Verdict: CLOSE — CFG can't help, for two measured reasons:**
  • **Push is tiny exactly where style is decided.** ‖Δ‖/‖v_null‖ = **0.0060 @t1000** (high noise, sets global
    style/comp), rising to only **0.0242 @t250** (low noise, texture-only). The DiT down-weights the OPTIONAL
    style-token channel precisely at the high-noise steps that matter → amplifying a 0.6% push needs guidance
    ~20–30, which wrecks quality. This is the noised-target root cause shown as a number.
  • **The ref-dependent part is per-image noise, not style.** Cross-ref Δ corr ~0.51–0.59 (NOT a pure constant
    ~1.0), but **within-type ≈ cross-type at every noise level** (t1000: within 0.599 vs cross 0.588; t250:
    0.523 vs 0.518, gap ~0.01). If Δ encoded style, within-type would be much higher. So CFG amplifies
    idiosyncratic jitter, not style. Velocity-space confirmation of the Stage-2 probe's within-cluster 0.9975
    vs cross 0.9976 (gap 0.000).
`v_cond` cross-ref corr @t1000 = 1.0000 (dominated by shared `v_null` — the DELTA corr is the real signal).
**Consequence:** the base's CFG lever does NOT reopen the token-channel encoder (A dead). It DOES positively
point at B/C: the failure is specifically (a) high-noise down-weighting of an optional channel and (b) no
style-organized signal — both bypassed by FiLM/AdaLN modulation (unignorable, all-noise, CSD is style-organized
by construction) and by hypernetwork→LoRA (conditioning IS the velocity field; proven +0.176, style-organized).
The token channel was the one that put style where the DiT is free to ignore it at high noise — now MEASURED,
not theorized. Probe is throwaway (scratchpad); GPU freed (web stopped) for the run.

### 🔴 SREF-LEARNED-STAGE1 (2026-07-08) — projector-only (frozen DiT) does NOT transfer style. Stage-2 DiT LoRA is the gate.
Ran the full Phase-1 Stage-1 of the learned-encoder project (`plans/sref-learned-encoder-project.md`):
`StyleProjector` (SigLIP→192 in-sequence style tokens, TEXT|STYLE|IMAGE, text-like RoPE), DiT FROZEN,
4000 steps on the 4B base (`train/train_style_projector.py` + `sref_projector_v1.yaml`, look-paired data,
loss 0.71→~0.4, clean). Gated the step-4000 checkpoint with the new Python harness
(`debug/sref_infer_style.py` → renders eval set → `sref_scorecard.py --score-only`), 24-step base renders,
seed 7, prompt "a robot standing in a desert":
  • styleCSD Δ: graphic **0.0069**, painterly **0.0033**, semi_real **−0.0057** (OVERALL 0.0041).
    styleGram Δ **0.00000** everywhere. Band-control (the bar to beat): graphic 0.096 / painterly 0.009.
    → Does NOT beat band-control on ANY type; painterly ~1/3 of the (already-weak) band-control number.
  • Renders are COHERENT (std 43 vs baseline 43 — NOT washed out, so the single-forward guidance-embed
    base regime is valid; the CFG concern is ruled out). But every ref moves the output only ~2.5%
    (L1 6.3/255) and by nearly the SAME amount for every reference → style tokens near-inert.
  • Cross-ref output pixel corr 0.98 (heuristic collapse metric fired ≥0.90 — but pixel corr conflates
    seed-fixed LAYOUT with style, so the scorecard styleCSD/Gram Δ is the real verdict, and it's ~0).
INTERPRETATION: NOT the same as SREF-CHAMPION-COLLAPSE (that was a reference-INDEPENDENT constant,
to_v_ip rank ~6). Here the tokens carry a FAINT reference-dependent signal (styleCSD Δ preserves the
graphic>painterly>semi_real difficulty ordering; cross-ref corr 0.98 not 1.00) — just far too weak to
matter. Most consistent with the **frozen DiT having no pathway to convert in-sequence style tokens into
style**: with the backbone frozen, the flow-matching loss is minimized by near-inert tokens (the trivial
solution). This is exactly what the plan's **Stage 2 (DiT LoRA r128, trained jointly)** exists to fix,
and matches USO's finding that the LoRA stage delivers the real style transfer (projector-alone is weak).
DECISION (open, needs user): Stage 2 is the real style-transfer mechanism but is **M5-gated** (base+LoRA
training, ~2-day M1 run / 128 GB). Options: (a) cheap Stage-2 PROBE — inject a small LoRA into the style
forward and overfit ~8 eval refs a few hundred steps to confirm the capacity hypothesis before committing;
(b) full Stage 2; (c) fall back to the **retrieval-hybrid instant-LoRA** path (CSD-nearest trained per-style
LoRA + interpolation — reuses the WORKING per-style LoRA + CSD index, no collapse risk). Artifacts: `/Volumes/2TBSSD/sref_eval/learned_encoder_stage1/` (scorecard json + 12 renders),
checkpoints `/Volumes/2TBSSD/checkpoints/sref_projector/`.
TRAJECTORY (step-2000 vs 4000, done): OVERALL styleCSD Δ 0.0028 vs 0.0041; painterly 0.0069 vs 0.0033;
graphic −0.0008 vs 0.0069 — all sub-0.01 NOISE around zero, no strong-then-collapsed arc. → **flat
plateau, NOT collapse-over-training**. Supports the capacity-limit reading (projector never learned style
through the frozen backbone; no collapse dynamics) → Stage-2 DiT LoRA is the right thing to test, and a
cheap overfit probe is well-motivated before a full run.

CHEAP STAGE-2 PROBE (2026-07-08, `train/lora/probe_style_lora.py`): injected a rank-16 LoRA (18.7M, 80
double+single attn sites) + trained the projector JOINTLY, overfit 8 look-paired samples 300 steps with a
FIXED prompt (so SigLIP is the only cross-sample distinguisher). Result **KILL-leaning**: overfit loss
1.13→0.25 (oscillating ~0.3, NOT converging to 0), but the causal swap test (fixed seed+prompt, vary only
SigLIP) gave cross-ref output corr **0.9996** (Stage-1 was 0.98 — i.e. MORE inert) and CSD diagonal-
dominance **1/8 = chance**. CAVEAT (my probe's confound): the fixed-prompt overfit lets "predict the blurry
AVERAGE of 8" reach ~0.25 loss WITHOUT using SigLIP, so the loss-drop isn't evidence of binding and a
no-binding result is partly expected from the setup. Still weighty because: both paths cut loss, so easy
SigLIP-binding would have shown SOME swap divergence (it went the other way, to 0.9996); and Stage-1 (4000
steps, diverse data where the average escape can't win) independently landed near-inert. TWO setups → same
read: the in-sequence style→output path is HIGH-RESISTANCE on Flux.2 Klein; the LoRA improved the base
denoiser instead of routing SigLIP. A cleaner (unconfounded) cheap test = 2 distinct style clusters + same
prompt (average becomes high-loss → SigLIP required). Artifacts: `…/learned_encoder_stage1/stage2_probe*`.
CLEAN 2-CLUSTER PROBE (2026-07-08, `--two-cluster`, the UNCONFOUNDED test): CSD-clustered a 16-pool into
2 style groups (cross-cluster CSD dist 0.87) so the average is a muddy high-loss blend → SigLIP is REQUIRED
to fit both. 8 samples, rank-16 LoRA + projector, 300 steps. Result **DECISIVE KILL**: overfit loss
0.445→0.127 (below the muddy-average floor) BUT swap routing gap = **within-cluster corr 0.9975 vs
cross-cluster 0.9976 = −0.0001** (zero style routing) and CSD dominance 1/8 = chance. Outputs are identical
regardless of which reference's SigLIP is fed, even with the escape closed.
ROOT CAUSE (the durable learning): in flow-matching the model input IS a noised version of the TARGET, so
at low-to-mid noise it reconstructs each target from the input alone and NEVER needs the style tokens; the
conditioning is only forced at high noise, and that weak signal is never learned on this stack (4B, our
compute, in-sequence tokens). The loss-drop to 0.127 came from denoising the input latent, NOT from routing
SigLIP — the swap test (starts from PURE noise) exposes it. Explains ALL THREE negatives: Stage-1
(projector-only 4000 steps, near-inert), Probe A (LoRA fixed-prompt, corr 0.9996), Probe B (LoRA 2-cluster,
gap 0.000). Also explains WHY the shipped IN-CONTEXT path works (the VAE reference tokens ARE the actual
image content in the sequence — high-information, impossible to ignore) while a LEARNED abstracted style
token is never weighted. Both learned-adapter families have now failed on this stack: K/V injection
(collapsed, SREF-CHAMPION-COLLAPSE) AND in-sequence tokens (never bind, here). USO succeeded on FULL dev
(50-step, far larger model + reward learning) — out of our compute budget.
**DECISION: KILL the learned-encoder Stage-1/2 direction as tested; fall back to the RETRIEVAL-HYBRID
instant-LoRA path** (CSD-nearest trained per-style LoRA + interpolation — reuses the two things that DO work
here: per-style LoRA weight-space training + the CSD style index). Full Stage-2 (M5, bigger LoRA, reward
learning) is not worth the ~2-day/M5 spend against three negatives. Artifacts:
`…/learned_encoder_stage1/stage2_probe_2cluster*`. Probe tool retained: `train/lora/probe_style_lora.py`.

### SESSION INSIGHTS — 2026-06-20 (direct-trainer dataset experiments; read first)
- **[PARTLY SUPERSEDED — see "NULL-FLOOR REFRAME" below.]** Earlier read: "no adapter
  transfers style — all NEGATIVE sref_score (style_sim ~0.02–0.10 < content_leak ~0.35–0.47)."
  The negative *absolute* sref_score turned out to be FLOOR-confounded; measured vs a
  no-adapter null, the adapters DO inject style. The failure is content LEAKAGE, not absent
  style. Keep reading.
- **[SUPERSEDED — style pairing was TESTED and did NOT help; see NULL-FLOOR REFRAME.]** Earlier
  hypothesis: the arms were weak only because they ran with `style_pair=0%` (the direct
  campaign never wired `data.style_neighbors_db` → arbitrary pairing → ignore the reference).
  That wiring GAP is real and worth knowing — SREF-1W wired style pairing into the
  FLYWHEEL/orchestrator path, but the direct trainer needs `style_neighbors_db` set in its
  config explicitly — easy to forget, and it silently produces unpaired (weak) adapters.
- **Style pairing is VIABLE at scale (gate re-confirmed).** Built the CSD style index on the
  real 22-shard hot pool (`baseline_pool_hot`, 109,253 records): `style_neighbors.py` top-5
  NN/random ratio = **0.569** (gate ≤ 0.7; the ruled-out 4-bit SigLIP descriptors were
  0.86–0.96; the val-set CSD number was 0.679). 7,536 near-dup exclusions. 93% of records got
  a usable neighbor. So the SREF-1 substrate genuinely works — the blocker was always the
  descriptor, now solved by CSD.
- **The data-QUANTITY ladder result is a CONFOUNDED RED HERRING.** 4-shard ≈ 22-shard (no
  benefit from more data) is valid ONLY for the broken-signal (unpaired) regime, on nested
  ARBITRARY subsets — not meaningful composition. Do not treat "quantity doesn't help" as
  general. **Re-run quantity AND composition (per-source, curated-vs-random, style-coverage)
  on the working style-paired platform** — that's the first recipe ablation once the platform
  lands. The ladder still earned its keep: backed the intuition and flushed out the infra bugs.
- **Sequencing (load-bearing): platform FIRST, ablate SECOND.** Get one working style-paired
  adapter (positive sref_score) → freeze that config as the reference platform → run recipe
  ablations branching from it. Ablating recipes on a broken-signal adapter measures noise.
  Memory: `sref-platform-strategy`. Active run config: `/Volumes/2TBSSD/sref_eval/style_arm/`.
- **Eval method (SREF-EVAL-PARAMS, below):** an adapter is a style_sim↔content_leak FRONTIER
  over `--ip-scale`, not a scalar; compare arms at a MATCHED content_leak budget. Harness:
  `sref_sweep_eval.py` + `sref_eval.py` (CSD) on the fixed WikiArt eval set in
  `/Volumes/2TBSSD/sref_eval/`.
- **Infra lessons (all fixed/committed this session — guard against recurrence):**
  (1) NEVER train/precompute from COLD storage — the loader enumerates every shard tar before
  the first sample at ~31 s/shard cold vs ~0.8 s hot, so >3 cold shards exceed the 120 s
  `sample_q` timeout and the trainer dies at step 0 (AGENT.md invariant #6). (2) `--resume`
  crashed AdamW: `make_lr_schedule`'s resume branch returned a Python float, MLX does
  `learning_rate.astype()` (commit fixed). (3) A campaign stall-monitor regex `\d+` choked on
  the trainer's thousands-separator (`step 1,025/…`), false-killing healthy runs at ~step 975
  — the phantom that masqueraded as an "MLX wedge at step 1000" (there was no wedge).

#### NULL-FLOOR REFRAME — the most important SREF finding so far (2026-06-20, later)

After the style-paired arm ALSO scored ≈ the same negative sref_score as the unpaired arms,
ran the missing control: a **no-adapter plain-Flux** generation for the same 10 prompts,
scored against the same WikiArt refs. This is the NULL — a generation that, by construction,
has nothing to do with the reference. Results (CSD, 512px, seed 42, 10 pairs):

| config | style_sim | content_leak | sref_score | Δstyle vs null | Δleak vs null |
|---|---|---|---|---|---|
| **null (no adapter)** | **+0.004** | **+0.323** | **−0.319** | — | — |
| v4 baseline @0.5 | +0.091 | +0.468 | −0.376 | **+0.087** | **+0.145** |
| arm_4 (4sh, unpaired) @0.5 | +0.104 | +0.466 | −0.362 | +0.100 | +0.143 |
| style arm (paired) @0.5 | +0.089 | +0.466 | −0.377 | +0.085 | +0.143 |

**Conclusions (these correct earlier bullets):**
1. **Absolute sref_score is FLOOR-confounded — stop using it raw.** A generation unrelated to
   the reference already scores content_leak **0.32** and sref **−0.32**. CSD's content head
   has a high baseline cosine (~0.32) between any two natural images; style_sim's baseline is
   ~0 (0.004). So "−0.33 = bad" was a metric artifact, not model failure. **Report DELTAS over
   the no-adapter null**, per scale: Δstyle_sim, Δcontent_leak, and the real objective
   **Δstyle − Δleak** (and the injection ratio Δstyle/Δleak).
2. **The adapters are NOT inert — they DO inject style.** style_sim 0.004 → ~0.09 is a ~20× lift
   (Δstyle ≈ +0.086). The model genuinely does style transfer. The earlier "no adapter
   transfers style" read was wrong; it was reading the floor.
3. **The real failure mode is CONTENT LEAKAGE.** The adapter pulls the reference's *content*
   in even harder than its style: Δleak ≈ +0.144 > Δstyle ≈ +0.086. Injection ratio
   Δstyle/Δleak ≈ **0.6** (need > 1.0 to beat null). The objective is to **raise that ratio /
   maximize Δstyle − Δleak**, i.e. keep the style, kill the content bleed.
4. **Style pairing did NOT change the tradeoff at 3000 steps.** v4 (unpaired) and the style arm
   (paired, gate 0.569, 75% engaged) have ~identical Δstyle and Δleak. Pairing was *supposed*
   to reduce leak (different-content ref → copying it should hurt) but didn't at this budget.
   Real negative for pairing-at-3000-from-scratch — but now we know the metric to move.
5. **Quantity, pairing — both ruled out as the lever.** Everything sits at injection-ratio
   ~0.6. The lever is content–style DISENTANGLEMENT, not data treatment.

**Eval tooling change (done + validated):** `sref_sweep_eval.py` gains `--null <scores.json>` to
report Δstyle/Δleak/Δsref + injection ratio per scale against the no-adapter baseline. Null
baseline lives at `/Volumes/2TBSSD/sref_eval/noadapter/` (regen: plain `iris` over
`eval_set.json` prompts → `sref_eval.py`). All future SREF arms MUST be read null-relative.
Validated on the style arm — per-scale injection ratio (Δstyle/Δleak): **0.3 → 0.05** (barely
injects), **0.5 → 0.59** (best), **0.7 → 0.45**. So the operating point is ~0.5 and the number
to beat is **injection ratio 0.59** (Δstyle 0.085 / Δleak 0.143); need > 1.0 to clear the null.

- **SREF-LEAK-1: reduce content leakage (the actual objective — maximize Δstyle − Δleak).**
  The adapter injects style but leaks the reference's content harder (ratio ~0.6). Levers,
  cheap→expensive, to test as small arms on the working harness, each scored by Δstyle − Δleak
  null-relative (NOT raw sref_score):
  - **Aggressive reference content-destruction** (cheapest, highest-leverage): `patch_shuffle_prob`
    is only 0.5 — raise to ~1.0 and/or shuffle harder. Patch-shuffle scrambles the reference's
    spatial layout (content) while preserving local texture (style) — directly targets leak.
  - **Token compression** in the PerceiverResampler: fewer `num_image_tokens` (128 → 64/32)
    forces the resampler to drop fine content detail, keep gist/style. Architecture knob.
  - **Content-leak penalty in the loss**: add a CSD-content-distance term (push content(gen)
    AWAY from content(ref)); CSD encoder already in-repo. The principled fix; more work.
  - **Longer training WITH pairing**: pairing's content-invariance may only emerge past 3000
    steps (e.g. 15–20k). Expensive; gate behind the cheap levers first.
  - **Inference-side**: the ip-scale frontier already shows leak rises with scale — the
    matched-budget operating point (fixed leak ceiling) is the deployment lever, separate from
    training. Relates to SREF-1 (pairing), QUALITY-3 (patch-shuffle), QUALITY-2 (freeze
    double-stream — already on; double ip_scale=0).
  - **First arm LAUNCHED 2026-06-20: `leak1_pshuf`** — style-paired config, ONLY change
    `patch_shuffle_prob` 0.5→1.0 (one variable), from scratch, 3000 steps, hot pool + the
    CSD `neighbors.sqlite`. Config `/Volumes/2TBSSD/sref_eval/leak1_pshuf/`. **Success = injection
    ratio (Δstyle/Δleak @ best scale) beats the style arm's 0.59**, ideally toward 1.0. Score
    with `sref_sweep_eval.py --null` (NOT raw sref_score). If patch-shuffle alone doesn't move
    it, next is token compression, then the CSD content-leak loss term, then longer training.
  - **RESULT (2026-06-21): patch-shuffle 1.0 helped only modestly — config-knob augmentation
    looks like diminishing returns.** leak1 vs style arm @ best scale 0.5: injection ratio
    0.593 → **0.647** (+0.05). But Δstyle (0.085→0.120) AND Δleak (0.143→0.185) BOTH rose ~40% —
    the adapter got LOUDER, only slightly more SELECTIVE; Δsref actually went −0.058→−0.065.
    Held-out cond_gap crossed positive for the first time (+0.0031 vs style arm −0.0016). Read:
    scrambling patch ORDER doesn't remove content because each SigLIP patch still carries its
    local content. **Hypothesis update:** the leak is structural — the conditioning signal
    (SigLIP, content-laden) is the source, so per-token augmentation can't fix it. Higher-leverage
    levers than more config knobs:
    - **`cross_ref_prob` 0.5 → 1.0** (cheap; LAUNCHED 2026-06-21 as `leak2_xref`, one-variable
      change from leak1, patch_shuffle stays 1.0): force ~every conditioned step to predict from
      a different-content same-style neighbor → content-copying penalized nearly always. Changes
      the TASK (not just input augmentation) to demand content-invariance, but WITHOUT the
      collapse risk of an explicit embedding loss (below). Safe, on-target. Success = injection
      ratio > leak1's 0.647 @0.5.
      **RESULT (2026-06-21): REGRESSED to 0.506 (best @0.7) — cross_ref 1.0 HURT.** Both Δstyle
      (0.119→0.046) and Δleak (0.185→0.091) dropped: pure cross-ref makes the adapter inject
      LESS overall (no self-ref grounding) without improving selectivity. cond_gap −0.0011 (vs
      leak1 +0.0031). The 50/50 self/cross mix is better than 1.0 — there's a sweet spot, not a
      monotone.
    - **PLATEAU CONCLUSION (2026-06-21): the data/augmentation dimension is EXHAUSTED at ratio
      ~0.65.** Four arms (style 0.59, leak1 0.65, leak2 0.51) — very different data treatments —
      all pin the injection ratio at **0.5–0.65** (need ~1.0; Δsref stuck at −0.05 to −0.07).
      That stability means the style/content entanglement is a property of the **conditioning
      SIGNAL (SigLIP, 729×1152 content-laden patch tokens) + the architecture**, NOT the
      training. No data/augmentation knob will clear it. **Next moves are STRUCTURAL** (pick
      one, not more knobs):
        (i) **Condition on CSD STYLE embeddings instead of/with SigLIP** — root-cause fix. CSD's
            768-d style embedding is content-invariant BY CONSTRUCTION (that's why the neighbor
            gate passed at 0.569); SigLIP is not. The content simply isn't in the signal. We
            already have CSD embeddings precomputed (`style_cache`). Cost: new conditioning
            pipeline + perceiver input dim change + retrain; biggest change, highest expected
            payoff. NOTE inference parity: iris would need a CSD encoder (or precomputed CSD
            features) for `--sref`, replacing/augmenting the SigLIP `--ip-features` path.
        (ii) **Contrastive style-invariance loss** on ip_embeds (same-style close, diff-style
            FAR — needs negatives + collapse guard; see caveat above). Operates on the same
            SigLIP input so may be limited by (i)'s root issue, but cheaper than (i).
        (iii) Token compression (128→32) — cheap structural knob, limits content capacity but
            also style capacity; a quick probe, lower expected payoff.
    - **Style-INVARIANCE loss on ip_embeds (principled, cheap/no-decode — BUT collapse-prone, do
      NOT implement naively):** the tempting form `1 − cos(ip_embeds(ref), ip_embeds(neighbor))`
      pulls same-style embeddings together — with NO negatives this COLLAPSES the embedding space
      (every image → one embedding), exactly the failure the perceiver input-norm fix cured
      (constant injection → grid). Needs a CONTRASTIVE formulation (same-style close AND
      different-style far) with negatives; at batch_size=1 negatives must come from the cross-ref
      buffer / a random pool. Only attempt with collapse guards (monitor cross-token std of
      ip_embeds, the same ratio used to detect the grid). The decoded CSD content-leak loss is
      the obvious alternative but is PROXY-1-class expensive (~75 s/step); avoid.
    - Token compression (128→64/32 tokens) and longer training remain, lower priority.
    - Root-cause option (big): condition on CSD STYLE embeddings (content-invariant by
      construction — that's why the neighbor gate passed at 0.569) instead of/with SigLIP.

  #### CSD-CONDITIONING BUILD LOG (SREF-LEAK-2, 2026-06-21 — running design notes)
  Chose the root-cause fix (condition on CSD, not SigLIP). Decisions + reasoning as built:
  - **Architecture: `CSDImageProj` (FiLM-modulated learned queries), NOT a perceiver.** CSD is
    ONE 768-d vector, not a token set, so cross-attention/resampling is meaningless (128 queries
    over 1 kv → all identical → collapse). Instead: 128 learned query tokens, FiLM-modulated by
    the CSD vector (`tokens = q*(1+scale)+shift`, scale/shift = Linear(768→2·3072)), then
    LayerNorm. The 128 distinct queries carry diversity, the CSD vector carries style → cannot
    pool/collapse. Chose FiLM over self-attention deliberately: (a) simplest to MIRROR in C
    inference (Linear+FiLM+LN, no new attention kernel), (b) lowest collapse risk. If
    under-expressive, add a self-attn block later (also portable). **Validated** (phase 1a,
    commit 2f74871): cross-token std/|mean| 0.843 (vs grid ~0.007 = healthy/no-collapse),
    responsive to different CSD inputs (Δ 0.657), correct K/V shapes. `IPAdapterKlein` gains
    `cond_mode` "siglip"|"csd" (default siglip — SigLIP path untouched).
  - **NO eval signal until the C port lands.** The eval generates with `iris` (C); the trainer
    has no standalone image-generation path. So this is a full Python-train + C-infer build
    before any sref_score. Accepted (user: keep going).
  - **Primary RISK being tracked: 768-d fidelity bottleneck.** CSD's 768 numbers vs SigLIP's
    729×1152 ≈ 840k — a huge compression. CSD may reduce leak but also cap STYLE fidelity (low
    leak + low style = wash). First eval decides; if style is too weak, fall back to the
    contrastive style-invariance loss on SigLIP (keeps the rich signal) — option (ii) above.
  - **Phases:** 1a module ✓ · 1b CSD dataset loader (per-SHARD npz bundles, not per-record;
    neighbor CSD; coverage filter) + trainer guards (warmup/miss/val zero-shapes → [B,768]; skip
    patch-shuffle in csd mode — no tokens to shuffle) + smoke · 2 C inference (mirror in
    `iris_ip_adapter.c`, `--ip-features` → 768-d, `csd_features.py` producer) · 3 retrain + eval.
  - Cache: CSD features already precomputed at `/Volumes/2TBSSD/sref_eval/style_cache/*.npz`
    ([768] f16 per record, keyed by rec_id, per-shard bundles).
  - **PEDANTIC BUG SWEEP (2026-06-21, phase-1 complete) — findings + fixes:**
    1. **CRITICAL — CSD loss DIVERGED in the first smoke (2.5→129).** Root cause: `CSDImageProj.film`
       used default `nn.Linear` init → random scale/shift at step 0 → unstable. Fix: FiLM-ZERO init
       (zero film weight AND bias) so tokens = query_tokens at init (adaLN-zero identity start);
       gradients still flow so it learns. Re-smoke: loss bounded ~1–4 across the LR ramp (no spike).
       The original 2.5→129 vs SigLIP smokes staying <1.2 is what flagged it as CSD-specific.
    2. `use_siglip_live` was `siglip_cache_dir is None` — a CSD config without a siglip cache would
       load SigLIP and could feed [729,1152] to a CSD adapter. Fix: `... and _cond_mode != "csd"`.
    3. `_load_csd_bundles` left `NpzFile` handles open → `with np.load(p) as d:`.
    4. `warmstart_path` + `cond_mode="csd"` would silently build a SigLIP perceiver → now raises.
    5. Stale `get_image_embeds` docstring (siglip-only) → generalized. Validated end-to-end:
       CSD map (109,253) loads, neighbors resolve via CSD map, warmup compiles with the [1,768]
       dummy, trains 20 steps bounded, best.safetensors writes. **Deferred to phase 2/3 (NOT bugs,
       expected):** `export_adapter.py` + iris C have no CSD key mapping yet (image_proj.film.* vs
       perceiver.*); the held-out cond_gap val is disabled in CSD mode (no CSD for val shards).
  - **PHASE 2 — C inference + parity PROVEN (2026-06-21).** Mirrored `CSDImageProj` in
    `iris_ip_adapter.c`: struct gains `cond_mode`/`csd_dim`/`film_weight`/`film_bias`; load
    branches on cond_mode; `perceive` runs the FiLM path for csd (film = film_weight @ csd +
    film_bias → scale/shift → token = query_tokens*(1+scale)+shift → LayerNorm; NO cross-attn).
    `iris.c` `--ip-features` reads a [csd_dim] vector (one row) in csd mode. `export_adapter.py`
    maps `image_proj.film.{weight,bias}` → `perceiver.film_*`, branches required keys by mode,
    infers csd_dim, writes cond_mode/csd_dim to meta, and defaults the (unused) perceiver_heads.
    New producer `train/scripts/csd_features.py` (image → 768-d CSD → raw f32, the csd analogue
    of siglip_features.py). **Train↔infer parity is GUARDED, not assumed:** extended the fixture
    harness (`debug/gen_ip_adapter_fixture.py` builds a synthetic csd bundle + Python goldens
    from the real `CSDImageProj`, FiLM RANDOMISED so the matmul is exercised) and
    `debug/test_ip_adapter.c` (new `run_csd_bundle`). Result: **15/15 PASS, csd perceive
    corr=1.000000 max_abs=0.00000** vs the Python golden — the FiLM math/shape mirror is exact,
    a committed regression guard against the IP-ADAPTER-INFER-1 mismatch class. `make test-unit`
    green; SigLIP path unaffected. Remaining = Phase 3 (train a real 3000-step CSD arm + null-
    relative eval vs the 0.65 SigLIP plateau; watch the 768-d fidelity risk).
  - **DEEP SWEEP (2026-06-21) — beyond the parity test:**
    1. **Encoder/preprocess parity VERIFIED** (the real train↔infer risk, not covered by the
       fixture which feeds the same bytes both sides): training precompute (`style_precompute.py`)
       and inference (`csd_features.py`) both call the IDENTICAL `csd_mlx.preprocess` (deterministic
       resize-short-side-224 BICUBIC + center-crop + CLIP mean/std) and `enc.encode` (→ L2-normed
       768-d). So the inference CSD vector matches what the adapter trained on. f16-cache (training)
       vs f32 (inference) is the same accepted convention as SigLIP (siglip_features is f32 too).
    2. **BUILD-TARGET CATCH (important):** bare `make` only prints help — it does NOT build. The
       parity test passed on a STALE iris binary because `make test-unit` compiles `iris_ip_adapter.c`
       directly. The generation binary needs **`make mps`** (rebuilt clean, links iris_ip_adapter.mps.o,
       no errors). Always `make mps` after touching the C inference, or the eval runs the old binary.
    3. **Eval-harness CSD integration:** `sref_sweep_eval.py` now reads the bundle's `cond_mode`
       and uses CSD reference features (`csd_feat_dir`, default `…/refs_feat_csd`) for csd bundles
       instead of SigLIP `--ip-features`. `eval_set.json` gains `csd_feat_dir`. (CSD ref features
       are produced by `csd_features.py` post-training, when the GPU is free.)
    4. iris attach-log now reports "CSD style vector [768]" vs "729 SigLIP rows" (was misleading).
    5. **Extra robustness checks (while Phase-3 trained):** (a) switched the CSD parity fixture to
       **bf16** — the real arm's export format — so `make test` now guards the bf16 dequant/load
       path (was only f16/int8); 15/15 still exact. (b) **AddressSanitizer**: the C perceive +
       adapter path is memory-clean (no overrun/use-after-free). (c) **UBSan** surfaced a real,
       PRE-EXISTING (not-CSD) UB in the `exp2` kernel (`iris_kernels.h:30` left-shift of a
       negative) — fixed with a numerically-identical unsigned shift; UBSan now clean, parity
       intact, `make test-unit` all-green, iris rebuilt.
  - **PHASE 3 (2026-06-22) — CSD arm TRAINED; verdict eval BLOCKED on disk access:**
    The real CSD arm finished (3000 steps, `…/csd_arm/ckpt/best.safetensors`, EMA). Config =
    `style_arm` with only `cond_mode=csd` (cross-ref style pairing on; 768-d CSD). Training health:
    loss avg 0.378, style_loss 0.0058→0.0048 (falling), ip_scale mean 0.69 (double 0.0 / single 0.86),
    grad_norm ~0.4. cond_gap was NOISY per-25-step window (n≈16/9): negative early (−11%), +16…+19%
    mid/late, −12.7% on the final window with a "loss_cond≈loss_null" warning — too small-n to read as
    a verdict; the null-relative sweep is the arbiter. Export OK → bf16 bundle (csd tensors, csd_dim=768,
    siglip_dim=0). 5 CSD ref features built (`refs_feat_csd/*.bin`, 768 f32 each).
    - **EXPORT-VALIDATOR BUG (found + fixed, `export_adapter.py`):** after FiLM keys were added to
      `_KEY_MAP`, `validate_bundle` built `expected = set(_KEY_MAP.values())` = the UNION of SigLIP
      (query/key/value/out_proj) AND CSD (film_*) tensors, so `--validate` could NEVER pass for EITHER
      mode (CSD bundle missing the siglip projs → false FAIL). The write path already branched on
      `is_csd`; the validator now branches on `meta["cond_mode"]` to the mode-specific expected-set, the
      int8 scale-key set is intersected with it, and the int8 dequant spot-check is guarded for the
      siglip-only `query_proj`. Re-export → "Validation PASSED (8 tensors, bfloat16)". (Self-contained
      export fix; does not touch the C inference or the parity-proven path.)
    - (env note) `/Volumes/2TBSSD` (and 16TBCold) went EPERM mid-session — macOS **TCC
      removable-volume** protection (stat works, data read fails, EXTERNAL volumes only; sandbox-off
      didn't help). A newly-granted Full Disk Access only applies to processes started AFTER the grant,
      so the fix was to fully restart the VS Code `claude` native binary (Cmd-Q + reopen). After
      restart, reads worked again and the verdict ran. Keep this for next time the SSD sleeps/remounts.
  - **PHASE 3 VERDICT (2026-06-22) — CSD-only does NOT beat the 0.65 SigLIP plateau. Decision:
    proceed to SREF-COMBINE-1 (hybrid).** Sanity gens first: scale 0.15 = clean coherent "cat on a
    windowsill" (content intact, ~no style); scale 0.5 = content-free purple wash. So the CSD adapter
    is a smooth dial but FAR more potent than SigLIP — usable band is low scales. Null-relative sweep
    (`csd_arm`, scales 0.2/0.3/0.4, 10 prompts × 5 styles, seed 42; null style_sim +0.0042 / leak
    +0.3231):
    | scale | Δstyle | Δleak | inj_ratio Δstyle/Δleak | prompt_adh |
    |------:|-------:|------:|----------------------:|-----------:|
    | 0.2  | −0.0009 | +0.0094 | −0.10 | +0.152 |
    | 0.3  | +0.0113 | +0.0520 |  0.22 | +0.144 |
    | 0.4  | +0.0695 | +0.1103 |  0.63 | +0.079 |
    Injection ratio rises with scale but TOPS OUT ~0.63 at 0.4 — at/below the SigLIP best (0.65, leak1/
    patch_shuffle). And it's a false 0.63: **eyeballing the images shows NO faithful style at any scale**
    — impressionism_landscape 0.3 = sharp PHOTO (no style) → 0.4 = blue-grey wash (content gone);
    expressionism violinist 0.3 = clean PHOTO → 0.4 = dark scratchy smear (subject destroyed). The
    +0.07 Δstyle at 0.4 is wash-texture raising the SigLIP style cosine, not real style. **No scale has
    both coherent content AND faithful style** → the 768-d-bottleneck fidelity risk CONFIRMED.
    - **ROOT CAUSE (architectural, log it):** `CSDImageProj` FiLM-modulates 128 SHARED query tokens
      with a single per-channel (scale, shift) → the whole injection is ONE global modulation direction
      (rank-limited). Structurally it can only apply a global color/texture shift, never spatial/textural
      style — hence the wash. A richer CSD head (768 → 128 DISTINCT tokens via small MLP/attn) might
      help, but the cleaner next step is the already-scoped hybrid.
    - **NEXT = SREF-COMBINE-1** (`cond_mode="hybrid"`): CSD supplies the content-free style DIRECTION,
      SigLIP's 729→128 cross-attention supplies the local detail the single vector can't carry. The two
      arms fail in opposite ways (SigLIP leaks-with-style; CSD washes), so combine. Do NOT start the
      hybrid training run without confirming the plan first.

- **SREF-COMBINE-1: hybrid SigLIP + CSD conditioning for stronger style transfer (High —
  next major architecture experiment after the CSD-only test).** Status: IMPLEMENTED + inference
  parity-proven; training RECIPE needs tuning (smoke showed early instability). Released v4.1.0.
  - **IMPLEMENTATION (2026-06-22, cond_mode="hybrid").** Design = dual-module concat with a
    PACKED single feature so the trainer's `mx.compile`d step stays untouched. The conditioning
    is one `[B, 730, 1152]` array (rows 0..728 = SigLIP, row 729 = the 768-d CSD vector zero-padded
    to 1152). `IPAdapterKlein.get_image_embeds` slices it → runs the SigLIP `PerceiverResampler`
    (→128 tokens) AND the CSD `CSDImageProj` FiLM (→128 tokens) → concatenates to **256** image
    tokens. Both sub-modules are the EXISTING parity-proven modules, reused verbatim; the only new
    math is the concat + slice. `to_k_ip/to_v_ip/ip_scale` shapes are unchanged (per-channel).
    Surface: model.py (hybrid branch + slice), dataset.py (`_load_cond` packs siglip+csd row;
    needs BOTH caches; neighbor probe requires both), trainer (`_cond_dummy_shape=(1,730,1152)`,
    csd_cache+warmstart guards, val gate), iris_ip_adapter.c/.h (refactored perceive into
    `perceive_siglip_mha` + `perceive_csd_film` helpers + a hybrid path that writes the two halves;
    new `csd_*` struct fields), iris.c (`--ip-features` reads packed [730,1152]; attach log),
    export_adapter.py (`csd_proj.*`→`csd.*` key map, mode detect, `_infer_dims` total tokens =
    siglip-half + csd-half, validate hybrid expected-set), sref_sweep_eval.py (hybrid feat dir),
    `train/scripts/hybrid_features.py` (NEW producer: composes siglip_features.py + csd_features.py
    → packed [730,1152], guaranteeing identical preprocess/encode to training).
  - **PARITY PROVEN: 20/20** (`debug/test_ip_adapter.c` `run_hybrid_bundle`, fixture FiLM randomised),
    hybrid perceive corr=1.000000 max_abs=1e-5 vs the Python golden, through the REAL bf16 export/load
    path; green under full production flags (`-O3 -march=native -ffast-math -flto -DUSE_BLAS`);
    `make test-unit` all-green; `iris` relinked (`make mps`). Both halves + the concat order are guarded.
  - **SMOKE (40 steps) — trainer runs end-to-end but RECIPE IS HOT.** Caches overlap well (109K CSD
    records, 101K with usable style neighbors, 0 dropped for missing SigLIP). BUT loss ~4.2 (not
    falling), grad_norm 100–1700, **100% grad-clipped**, loss_null 1.98 (other arms ~0.4),
    loss_cond−loss_null gap −2.9. Cause is structural, not a code bug (inference is parity-exact):
    doubling to 256 injected tokens at ip_scale≈1.0 perturbs the frozen Flux ~2×. **NEXT TUNING
    LEVERS for the full run:** lower ip_scale init (e.g. 0.5) and/or per-module gate, longer warmup,
    lower LR. Re-smoke until grad settles before the 3000-step arm; then null-relative eval vs the
    0.65 plateau (acceptance: inj ratio >0.75 with lower leak than SigLIP).
  - **VERDICT (2026-06-23) — hybrid is the BEST arm; marginal quantitative win, CLEAR qualitative
    win.** Trained 3000 steps at ip_scale_init=0.5 (loss settled 0.34, grad ~0.4, healthy; cond_gap
    ended ≈0 with the "may not be learning" warning — but that surrogate is MISLEADING here, the eval
    shows real style injection). Null-relative sweep (scales 0.3/0.5/0.7, 10 prompts × 5 styles; null
    style +0.0042 / leak +0.3231):
    | scale | Δstyle | Δleak | inj_ratio | prompt_adh |
    |------:|-------:|------:|----------:|-----------:|
    | 0.3  | +0.0069 | +0.0306 | 0.225 | +0.149 |
    | 0.5  | +0.1076 | +0.1567 | **0.687** | +0.113 |
    | 0.7  | +0.1191 | +0.1770 | 0.673 | −0.006 |
    Inj ratio peaks **0.687 at scale 0.5 — above the 0.65 SigLIP plateau AND the 0.63 CSD-only**, and
    Δstyle +0.108 is the **HIGHEST absolute style injection of any arm** (CSD +0.07, SigLIP arms ~+0.086).
    **The decisive result is QUALITATIVE:** at scale 0.5 the hybrid keeps CONTENT intact AND applies REAL
    style across all 5 styles — violinist = recognizable figure + scratchy expressionist brushwork;
    baroque = portrait + chiaroscuro; impressionism = landscape + painterly haze — where CSD-only at 0.5
    was a content-free WASH and SigLIP-low had no style. The two failure modes (no-style / wash) are GONE.
    This is the first usable style adapter: the SigLIP half holds structure, the CSD half adds the
    content-free style direction. **But it does NOT clear the >0.75 acceptance bar, and leak (+0.157) is
    NOT lower than SigLIP** — leak still tracks style (ratio <1.0). So: a working PLATFORM, an incremental
    metric win, not the decisive leak-reduction. Released v4.1.0 (code); validator query_tokens hybrid fix
    follow-up. **NEXT LEVERS to push past 0.75 (reduce leak at fixed style):** (a) widen style-only block
    zeroing / push CSD into early blocks + SigLIP into late (hierarchical, the per-block ip_scale already
    supports it); (b) down-weight or subsample the SigLIP half (it carries the leak); (c) per-module gate
    biased toward CSD; (d) longer training. Also: cond_gap is a POOR surrogate for style adapters (ended
    ≈0 while the eval shows +0.108 Δstyle) — trust the null-relative frontier, not cond_gap.
  - **LEAK-REDUCTION SWEEP (2026-06-23) — per-block per-group injection gate.** Built a
    [num_blocks,2] gate scaling each group's V per block (parity 20/20, commit 60ae70d). Arms vs
    the 0.687 hybrid baseline (inj ratio @ best scale; null style +0.0042 / leak +0.3231):
    - **(b) hybrid_siglipdown** (fixed SigLIP V×0.3): peak **0.575 @ 0.5** (Δstyle +0.084, Δleak
      +0.146) — WORSE than 0.687. Down-weighting SigLIP cut leak (0.146<0.157) but cut STYLE more
      (0.084<0.108). **Key negative: style and leak are ENTANGLED within the SigLIP signal** — a V-gate
      trades the group's TOTAL contribution, it can't disentangle, so scaling SigLIP down loses both.
      This undercuts the other V-gate arms (a/c are gate variations on the same entangled signal).
    - (a) hierarchical (SigLIP 0.3→1.0 / CSD 1.0→0.5 ramp across blocks): RUNNING — the one
      genuinely different hypothesis (per-block selectivity, not uniform down-weight). (c) learned
      and (d) longer DEFERRED given (b)'s negative (decided 2026-06-23): (c) is a gate variation on
      the same entangled signal with no leak penalty (weakest); (d)'s base already converged. Run
      only if (a) shows promise. If (a) also fails → the V-gate lever is exhausted; next lever is a
      LEAK PENALTY in the loss or fewer/curated SigLIP tokens (disentangle the signal, don't scale it).
  **Rationale:** the two signals fail in opposite ways, so combine them.
  - Pure SigLIP leaks content — each of the 729 patch tokens is heavily content-laden — which
    is the structural cause of the injection-ratio ceiling (~0.5–0.65) measured across the
    leak-reduction campaign (style/leak/cross-ref arms; see NULL-FLOOR REFRAME + SREF-LEAK-1).
  - Pure CSD is content-invariant by construction (768-d style head; that's why the neighbor
    gate passed at 0.569 and SigLIP descriptors failed at 0.86–0.96) → low leak, but its big
    compression likely caps STYLE FIDELITY / fine texture (the SREF-LEAK-2 768-d-bottleneck
    risk being tested now).
  - **Hybrid = CSD for the primary, content-free style DIRECTION + SigLIP for local detail/
    texture.** Likely pushes Δstyle higher while holding Δleak (and prompt adherence) down —
    a STRUCTURAL fix, consistent with the finding that data/augmentation knobs are exhausted
    (the lever is the conditioning signal + architecture, not training).
  **Implementation ideas (build on SREF-LEAK-2's `cond_mode`; add `cond_mode="hybrid"`):**
  - Concatenate into the PerceiverResampler: feed the 729 SigLIP patch tokens AND the CSD
    global vector (as an extra token, or FiLM-modulating the perceiver queries by CSD) so the
    resampler reads both. Reuses the existing perceiver (and its C inference path).
  - Learned gating / weighting between the two signals (a scalar or per-channel gate, init
    biased toward CSD so it starts content-safe).
  - Optional hierarchical injection: CSD into the early/double-stream blocks (global style),
    SigLIP into the later/single-stream blocks (detail) — the per-block `to_k_ip/to_v_ip` and
    `ip_scale` already support per-block control.
  - Start with a smoke on the current champion to validate the pipeline (collapse guard:
    cross-token std ratio, as in IP-ADAPTER-INFER-1), then a short 3000-step arm comparing
    pure-CSD vs hybrid, scored null-relative (`sref_sweep_eval.py --null`).
  **Acceptance criteria:**
  - Injection ratio (Δstyle/Δleak @ best scale) **> 0.75** with LOWER content_leak than pure
    SigLIP (vs the 0.65 SigLIP plateau).
  - Visibly stronger style transfer than the current best without prompt collapse
    (prompt_adherence not tanking).
  - A clear comparison table — pure SigLIP vs pure CSD vs hybrid — on the SREF eval triad
    (style_sim / content_leak / prompt_adherence, all null-relative).
  **Cross-refs:** NULL-FLOOR REFRAME (deltas-over-null is the metric), SREF-LEAK-1 (the ~0.65
  plateau + why), SREF-LEAK-2 / CSD-CONDITIONING BUILD LOG (the `cond_mode` machinery + the
  768-d fidelity risk this addresses), SREF-EVAL-PARAMS (ip-scale frontier), SREF-1 (style
  pairing substrate — reused unchanged). Gate the GO on the CSD-only eval: if pure CSD already
  beats 0.65, hybrid is the upgrade path; if pure CSD tanks style fidelity, hybrid is the fix.

- **SREF-CAMPAIGN-1: unify SREF recipe experiments into the standard orchestrator/campaign
  tooling (stop maintaining the ad-hoc direct-trainer path).** The 2026-06-20 dataset
  experiments ran through bespoke scripts (`sref_dataset_campaign.py`, `sref_sweep_eval.py`,
  manual `style_precompute`/`style_neighbors`, hand-built symlink pools) that DUPLICATE a thin
  slice of infrastructure that already exists — and lose everything that makes it science:
  - **Selection:** arms were arbitrary nested symlink subsets. The right substrate is
    `campaign_manager.py` — reproducible, score-ranked shard MANIFESTS with boolean
    composition (`source(wikiart)`, `top_pct(50)`, `balanced`, `… AND NOT source(laion)`),
    backed by `shard_scores.db`. This is exactly the per-source / curated / quantity
    composition lever the recipe ablations need. USE IT instead of ad-hoc pools.
  - **Tracking/championing:** the ad-hoc runner writes a flat `campaign_report.json`; the
    flywheel/ablation path has persistent champions + ablation DBs (cross-run comparison,
    `quality_gate.py`, attribution). Recipe arms should land there.
  - **Pipeline reuse:** the orchestrator already does select→stage→precompute→**style-index
    (SREF-1W, the style_neighbors step is wired)**→train→eval→score→champion, with hot-pool
    staging and an existing `ablation_sref_v1.yaml`. The ad-hoc path re-implements staging
    (badly at first — the cold-tar bug) and skips the style index entirely (→ the `style_pair=0%`
    unpaired arms).
  - **Why the detour happened (and the ONE enabling fix):** the flywheel WEDGES (BUGS MLX-2)
    and its prior results were invalid (pre-v4 adapters), so a working path was stood up fast.
    The missing primitive that forced the fork is a **non-flywheel, direct-single-process
    execution mode** for campaign/ablation arms — same selection + staging + DB + eval, but the
    arm trains via a plain `train_ip_adapter.py` subprocess (≈ what the ad-hoc runner does)
    instead of the wedging flywheel loop. Add that mode, then recipe experiments get
    `campaign_manager` selection + ablation/champion DBs + orchestrator staging/style-index
    WITHOUT the wedge.
  - **Also fold in** the genuinely-new good parts of the ad-hoc tooling so they aren't lost:
    the ip-scale **frontier** eval (`sref_sweep_eval.py`, SREF-EVAL-PARAMS) as a first-class
    campaign eval step that ranks arms by **sref_score** (the real metric) not just cond_gap;
    and hot-pool enforcement (AGENT.md invariant #6).
  - **DB schema gap (required for this):** `ablation_harness.py`'s `experiments` table records
    only `cond_gap`/`ref_gap`/`final_loss` and a cond_gap-weighted `score` — there is NO
    `sref_score`/`style_sim`/`content_leak` column, so it cannot rank SREF arms by the real
    metric. Add those columns (+ make `score` selectable to a sref_score-based objective).
    Corollary: do NOT back-import the 2026-06-20 ad-hoc arms — the quantity arms are invalid
    (unpaired) and would pollute champion selection, and all of them lack campaign provenance
    (no manifest, frontier-not-cond_gap) so DB rows would imply a false comparability with
    future proper-campaign arms. Keep them as documented reference (frontier JSONs + the
    SESSION INSIGHTS block), not DB rows. The working style platform adapter is better
    registered via a clean proper-tooling re-run than imported.
  - **Sequencing:** do this when promoting the post-platform recipe ablations (see SESSION
    INSIGHTS above + memory `sref-platform-strategy`) — i.e. AFTER one working style-paired
    platform adapter exists. Until then the ad-hoc path is an acceptable bring-up shim, but it
    should not become the permanent home for recipe science. Relates to ABL-FIDELITY,
    SREF-OPT-1, SREF-DATA-1.

- **SREF-1 (HIGHEST): style-paired training data via style clustering.** Root-cause
  finding: `cross_ref` swaps in the PREVIOUS LOADER IMAGE's SigLIP features
  (train_ip_adapter.py ~1780) — an arbitrary pairing. Asked to predict a target from a
  RANDOM other image's features, the optimal policy is to ignore the reference — which is
  exactly what ref_gap≈0 and weak cond_gap show. sref needs SAME-STYLE / DIFFERENT-CONTENT
  pairs so the reference's style genuinely helps and content-copying genuinely hurts.
  Plan: (a) per-image style descriptors from the EXISTING SigLIP cache (mean+std token
  pooling v1; content-invariant style statistics), (b) cluster the pool (k-means, K~256),
  (c) store rec_id→cluster, (d) cross-ref samples from the SAME cluster (+ small random
  fraction as negatives). Multiplies the value of the production run — land BEFORE it.
  **Status 2026-06-10: harness built; cheap descriptors RULED OUT by measurement.**
  `style_cluster.py` (sidecar sampling, descriptor build, k-means, persistence,
  pair-quality metric) works end-to-end. But on a 20K sample, same-cluster vs random
  pair-distance ratios: pooled mean+std 0.86, centered 0.86, PCA-whitened 0.90 (worse —
  amplifies quantization noise), low-rank Gram-of-patch-tokens 0.96 (worst). Conclusion:
  the 4-BIT-QUANTIZED SigLIP cache does not carry usable style signal for hand-crafted
  descriptors — 4-bit noise destroys the second-order statistics style lives in, and
  SigLIP's semantic objective concentrates style weakly. **Required: a dedicated
  style-descriptor precompute pass** (GPU window): preferred = a style-trained encoder
  (CSD-class, trained for style retrieval); fallback = unquantized SigLIP + a learned
  style head. Fits PRECOMP-3 cleanly as a new encoder identity ("style"). The clustering
  harness + pair-ratio gate consume it unchanged (descriptor field is versioned).
  Acceptance gate: pair-ratio <= 0.7 before wiring cross-ref to clusters.
  **Update (same night): CSD style encoder BUILT IN MLX and gate MET.**
  `train/style_encoder/csd_mlx.py` — official CSD ViT-L (tomg-group-umd) converted once
  from torch to safetensors (`/Volumes/2TBSSD/models/csd_vit_l_style.safetensors`, 297
  tensors; torch never used at runtime), reimplemented in pure MLX (~56 ms/img incl. JPEG
  decode). Sanity: self-vs-cropped cosine 0.972, cross-image 0.03–0.46. On the 994-record
  held-out val set: style space is a CONTINUUM (mean pairwise cos 0.184) so k-means caps at
  ~0.82–0.90 pair-ratio — but **nearest-neighbor pairing meets the gate: top-1 0.591,
  top-5 0.679** vs random. Design therefore: per-record top-k style-neighbor lists (k~5),
  and cross-ref loads a NEIGHBOR's SigLIP features (dataset.py change) instead of the
  previous loader image. Both builder scripts LANDED and val-validated 2026-06-11
  (style_precompute.py, style_neighbors.py — ratio 0.683, near-dupe exclusion working,
  visual confirmation: Frazetta-style ref's top neighbor = same style, different content).
  **SREF-1W: DONE (2026-06-17, commit a82514c + follow-ups).** The orchestrator wiring
  for the fourth (style) encoder step is fully landed across every touchpoint below.
  The ONE remaining item — cache_manager **v2** style-encoder identity registration —
  is explicitly DEFERRED/future (v1 ships with style's own per-shard manifest, which is
  sufficient for the run5 campaign); see the cache_manager line in the checklist.
  Implemented spec (**orchestrator wiring, decided 2026-06-11**):
  (a) Style runs as a FOURTH per-iteration precompute step in the ORCHESTRATOR (not
  inside precompute_all's model juggling): after precompute_all completes, run
  style_precompute.py over the staged shards (~30 min/42 shards), then
  style_neighbors.py over the result — same sentinel/heartbeat/log pattern as the other
  steps; publish bundles to cold (encoder-identity dir) for PRECOMP-3 reuse.
  (b) **Iteration-local neighbors**: build lists WITHIN the staged shards only (~210K
  records — the gate was met on 994, coverage is ample). The loader reads a neighbor's
  SigLIP features from HOT staging — zero cold I/O at train time, no global neighbor
  state to maintain. (c) Style keeps its per-SHARD bundle layout + own manifest — do NOT
  force it into precompute_all's per-record-npz conventions (the million-tiny-files
  trap); coverage checks read bundle key counts instead. (d) dataset.py cross-ref:
  sample a same-style reference from neighbors.sqlite, falling back to current behavior
  when the table is absent (inert until the orchestrator provides it).
  **SREF-1W touchpoint checklist (operational surface for the fourth encoder step):**
  orchestrator flywheel step (sentinels style.done/.error, heartbeat, retry, publish
  bundles to cold, gated by flywheel-config `style_pairing: true`, default OFF);
  data_stager/pipeline_setup (stage existing cold style BUNDLES hot for re-selected
  shards — the reuse path; only first-contact shards encode); pipeline_doctor
  (cold_precompute summary + staleness/cold checks add "style" with BUNDLE-aware
  coverage = npz key counts; new failure modes: style step failed, neighbors.sqlite
  missing while pairing expected; heartbeat staleness); pipeline_status (step in live
  view + log tails); trainer (`data.style_neighbors_db` + style_pair_pct telemetry in
  heartbeat/log); flywheel_lib (optional telemetry parse); ablation harness (inherits
  the dataset fallback — nothing breaks; later: style_pair_prob sweep variable);
  cache_manager (v2: register "style" encoder identity — DEFERRED/future; v1 = own
  manifest, shipped);
  DISPATCH.md telemetry reference (new heartbeat/sentinel/log names).
  Campaign decision: run4 may relaunch as-is (its purpose is data-selection warmth);
  the first style-paired campaign (run5) starts once SREF-1W lands.
- **SREF-2: style-specific evaluation.** CLIP-I conflates style and content, so the
  golden-set eval cannot see sref quality. Add per-generation: style similarity to ref
  (Gram-distance / CSD-like, content-invariant), content-leak from ref (subject copied?
  lower=better), prompt adherence (CLIP-T). This becomes the shippable-champion criterion
  (the held-out cond_gap remains the training-internal signal).
- **SREF-EVAL-PARAMS (read before any recipe/dataset comparison).** The sref eval is NOT a
  single number per adapter — its outcome is dominated by inference-time knobs that are
  ORTHOGONAL to the training recipe, so an unmatched comparison silently mis-ranks recipes.
  Measured 2026-06-19 on the v4 adapter (`bundle_inputnorm`, input-norm fix), 4-step
  distilled, seed 42, CSD refs from `/Volumes/2TBSSD/sref_eval`:
  - **`--ip-scale` is the dominant axis.** It multiplies the trained per-block `ip_scale`.
    The transition from "no effect" to "collapse" is SHARP and depends on BOTH the reference
    AND the prompt/content. Measured matrix:
    - cubism-stilllife ref + "cat on a windowsill": **0.4** clean cat, ~zero style → **0.5**
      cat + muddy texture (transitional) → **0.6/1.0** content-free collapse.
    - impressionism-landscape ref + "cat on a windowsill": **0.6** already collapsed to the
      SAME brown texture as cubism@0.6.
    - expressionism-portrait ref + "portrait of a woman": **0.6** still semi-coherent
      (stylized sepia portrait, content survives).
    Two consequences: (a) the coherent scale is **reference- AND prompt-dependent** — a
    portrait tolerates higher scale than a cat-on-windowsill; a single fixed scale is
    coherent for some pairs and collapsed for others. (b) The high-scale failure is a
    **generic, reference-INDEPENDENT brown texture** (cubism@0.6 ≈ impressionism@0.6), i.e.
    v4's style SPECIFICITY is weak — at the strength where it does anything, it collapses to
    a non-specific mush rather than transferring THIS reference's style. That collapse-
    specificity (does a stronger injection still look like the reference?) is itself a
    primary thing the dataset/recipe experiments must MOVE, and the eval must reward it.
  - **Consequence for recipe experimentation:** sweep `--ip-scale` per arm (e.g.
    {0.3,0.5,0.7,0.9}) and compare the **style_sim ↔ content_leak frontier** (the sref_score
    envelope), not a single point. Pick the operating point by a FIXED budget — a
    content_leak ceiling or a prompt_adherence floor — then compare style_sim at that matched
    point across arms. This decouples the recipe effect from the scale effect. Single-scale
    scoring is only valid if every arm is coherent at that one scale (verify, don't assume).
  - **Secondary axes to PIN (hold constant across all arms):** seed (high per-image variance
    at 4 distilled steps — use a fixed seed set, ≥2 seeds/pair, report mean), step count,
    prompt set, reference set, and `freeze_double_stream_scales` (v4 trains style via
    single-stream only, so double-stream scale changes are inert — don't sweep them).
  - Wire this into the harness (SREF-2): a per-arm scale sweep + frontier/matched-budget
    comparison, not a scalar. The current `sref_eval.py` scores a fixed pair set; the
    missing piece is the scale-sweep driver and the frontier aggregation.
- **SREF-3: app integration path (no Phase-3 blocker).** web/server.py (Python) computes
  SigLIP features for the user upload → f32 file → `iris --ip … --ip-features …`. Ships
  the feature with the C engine generating. Then: multi-ref (concat SigLIP rows from
  several images into one perceive — Perceiver accepts any n_siglip), strength UX =
  --ip-scale (validated 0/0.3/1.0), style codes = library of stored SigLIP embeddings
  (Midjourney "--sref random"/reusable-code UX). bf16/MPS-native inject + SigLIP-in-C
  (G-1 Phase 3) are latency work, not blockers.

- **SIGLIP-MLX-1: pure-MLX SigLIP vision encoder (drop the torch fallback) — mirrors
  csd_mlx.py** (Low priority — latency/cleanliness, NOT correctness; the torch fallback
  works). The app's style sidecar (`train/scripts/siglip_features.py`) and precompute
  both encode references with **torch/transformers** because `mlx_vlm` has no siglip
  module (`vlm_load('...siglip-so400m...')` → "Model type siglip not supported"). torch
  load is ~seconds + ~GB; an MLX-native encoder is faster/lighter for web serving.
  - **Right form:** NOT a patch to the pip-installed mlx_vlm (lost on upgrade; whole VLM
    class for just the vision tower). Write `train/style_encoder/siglip_mlx.py`, a sibling
    to `csd_mlx.py` (121-line pure-MLX ViT that loads converted weights, no torch). We
    output `last_hidden_state` patch tokens `[729, 1152]` — skip the attention-pool head.
  - **SigLIP-so400m specifics vs CLIP ViT-L (csd_mlx):** larger tower; NO CLS token;
    `gelu_pytorch_tanh` activation; learned position embeds for 729 patches; check the
    LN placement. One-time weight conversion (torch checkpoint → safetensors, like CSD).
  - **HARD PARITY GATE (the real cost, same lesson as VAE-teacher / CSD parity):** run5's
    adapter was TRAINED on torch-SigLIP features (precompute uses the torch path), so a
    pure-MLX SigLIP MUST validate bit-close vs torch-SigLIP (`debug/siglip_parity`,
    cos ≥ 0.999 on real images) BEFORE replacing it — else inference features diverge from
    training and conditioning silently degrades.
  - **Scope note:** this is the PYTHON app-sidecar target (MLX). The C-engine target
    (SigLIP in C for a Python-free engine) is the separate G-1 Phase 3 — different
    reimplementation, same parity discipline. Effort ~1 day incl. parity. Do when app
    latency matters or alongside G-1 Phase 3; not urgent (the 2026-06-14 torch fallback
    is correct and in use).
  - **PRECOMPUTE adoption (the bigger prize, but a gated training-pipeline change).**
    Today precompute AND the sidecar both use torch-SigLIP → train/infer are CONSISTENT.
    Adopting MLX for the sidecar ONLY would *introduce* a small train/infer mismatch
    (cos≈0.999, tolerable but deliberate); the clean end-state is MLX on BOTH sides.
    Precompute is *potentially* where MLX pays off most — the flywheel is precompute-
    dominated (millions of encodes), so IF MLX batch-encodes faster it speeds every
    iteration, far more than one-image app latency. BUT switching the precompute encoder is
    a high-risk training-data-pipeline change (VAE-Q1 / encoder-identity lesson): the
    existing pool SigLIP cache is torch-built, so either (a) treat MLX as the same identity
    and reuse it (only if drift is acceptably small) or (b) bump ENCODER_CODE_VERSION
    (PRECOMP-3) and RE-PRECOMPUTE the whole pool's SigLIP — clean, expensive. Never
    mid-campaign; only at a clean boundary (e.g. before the production foundation run).
  - **PREREQUISITE A/B BENCHMARK — do NOT assume MLX is faster.** MLX-vs-torch-MPS speed
    for batch SigLIP is UNKNOWN; torch on MPS is well-optimised and may match or beat MLX.
    Before any precompute switch, run a head-to-head A/B on the SAME batch (e.g. 1–2K real
    images): wall-clock img/s for MLX vs the current torch path, on the production batch
    size, warm. The precompute adoption is justified ONLY if MLX wins by a margin that
    outweighs the re-precompute cost; a tie or loss means keep torch for precompute and use
    MLX for the sidecar only (accepting the small train/infer cos≈0.999 drift). The parity
    gate (cos≥0.999) governs CORRECTNESS; this A/B governs whether the switch is WORTH it.
- **SREF-4: sequencing.** warmup-run4 warms attribution under the new held-out cond_gap →
  ablation when warm (first arm: freeze_double_stream_scales — double-stream injection may
  matter for style) → production 512 foundation (~12d) only AFTER SREF-1 lands → Stage 2/3
  (configs ready, TRAIN-7 gate passed). Data recipe: weight natural/style-rich sources
  (coyo ≫ journeydb for conditioning; grow wikiart-like sources — style diversity feeds sref).

- **SREF-METRIC-1 (FOUNDATIONAL — read before trusting any cond_gap-ranked result).**
  The whole optimisation stack ranks by **cond_gap, which is a SEARCH SURROGATE, not a
  style metric.** Stated plainly so it is not forgotten:
  - **What cond_gap is:** loss_null − loss_cond — "does the reference help the diffusion
    loss." Chosen as the primary ranking signal because it is the only per-iteration
    signal that is both STABLE (low-variance/monotone at 1000-step budgets) and CHEAP
    (straight from the loss, no image generation). It was NOT discounted — what was
    discounted is CLIP-I (conflates style+content, useless for sref) and ref_gap
    (theoretically the right "uses-the-reference" signal but too noisy to RANK by at
    this budget; kept only as a weak secondary).
  - **The catch:** cond_gap is a GENERIC conditioning metric. An adapter lowers it just
    as well by copying the reference's CONTENT/composition as by transferring its STYLE.
    It is structurally blind to the style-vs-content distinction — which is the entire
    objective of an sref model. So cond_gap is NECESSARY-BUT-INSUFFICIENT: failing it
    means the adapter ignores the reference; passing it does NOT mean it used the
    reference for style.
  - **Evidence the two diverge (not hypothetical):** the `sref_eval` ip-scale sweep on a
    champion showed content_leak rising 0.28→0.49 as ip-scale increases — i.e. what
    cond_gap rewards includes content-copying, the exact failure mode.
  - **The discipline:** cheap surrogate for the SEARCH (cond_gap — rank thousands of
    iterations / ablation arms), expensive truth for the VERDICT (`sref_eval` triad:
    style_sim↑, content_leak↓, prompt_adherence — run on demand on a champion). Champion
    cond_gaps reported in status updates are SEARCH numbers, not ship numbers.
  - **Therefore the sref sweep is a VALIDITY GATE on the search currency, not a nicety.**
    `debug/sref_sweep.sh` tests whether cond_gap is even pointed at style. Branch:
    (a) sweep shows clean style transfer → cond_gap is a valid surrogate, the stack is
    sound; (b) sweep shows content-copying → cond_gap is misaligned: either fold a
    style-aware term into the ranking objective, or demote cond_gap to a mere gate and
    rank champions by an occasional `sref_eval` pass. Run ONE sweep before leaning
    further on any cond_gap-ranked conclusion. A true per-iteration style metric is
    rejected on cost (needs decode + CSD-encode every step).

- **SREF-OPT-1: the optimisation framework (how the pieces compose into the production run).**
  Two complementary AXES, each explored by the same hold-one-out methodology, sharing
  cond_gap as the currency that makes their findings stackable:
  - **Data axis** — which shards/sources. CORRELATIONAL: flywheel champions
    (`debug/champions.py`). CAUSAL: the source-holdout campaign
    (`flywheel_source_probe.yaml` — a DATA ablation, structurally identical to the
    hyperparameter ablation, just on the data source instead of a hparam).
  - **Hyperparameter/architecture axis** — cross_ref_prob, style_loss_weight,
    freeze-double-stream, lr. CAUSAL: the ablation harness (DP-4).
  - **Composability** rests on the shared cond_gap yardstick — which is why the
    cross-campaign convention mixing mattered (`rescope_shard_scores.py`; `champions.py`
    tags eras and refuses to rank cond_gap across the held-out-EMA boundary).
  - **Separability caveat:** the axes are explored INDEPENDENTLY (ablation fixes data,
    champions/source-probe fix hparams) — neither explores the data×hparam INTERACTION.
    The stacked result is a strong, evidence-backed STARTING config for the production
    run (DP-5), NOT a proven joint optimum; the production run (or a final confirmation
    ablation on the chosen data) validates the combination.
  - **Compounding knowledge base (the meta-flywheel):** flywheel_history.db +
    shard_scores.db + ablation_history.db ACCUMULATE; every reader (`champions.py
    --seeds`, `source_attribution.py`, ablation warm-start/ABL-3) starts from prior
    learning instead of cold. The picture of the optimal run sharpens with every
    campaign and each new experiment inherits the priors.
  - **Today's state (keep grounded):** one data-axis campaign (run5), source-probe
    QUEUED (causal data evidence, runs during the 2026-06-15 absence), ablation axis
    LOCKED (DP-4, gated on attribution warmth), `sref_eval` BUILT-BUT-UNRUN — so the
    validity gate (SREF-METRIC-1) is the conspicuous gap. The two next steps that
    sharpen the picture without DP-4: the source-probe and the first visual sweep.

- **SREF-DATA-1: image-level curation + hero shards (strategic — value real, but gated
  and split-in-two).** Shards are arbitrary I/O packaging; a shard's average quality
  masks large within-shard variance, and we already compute rich PER-IMAGE signal that
  selection discards (CSD style embedding, strong-neighbor/pair-richness count, aesthetic
  / light scores, dedup). Going granular has real value — but only in one of its two
  forms, with real risks, and not yet:
  - **DO: image-level CURATION by cheap OFFLINE proxies** (style coherence, pair-richness,
    aesthetic, caption quality) — tractable, ingredients mostly already computed.
  - **DON'T: image-level cond_gap ATTRIBUTION** — cond_gap is a per-iteration whole-adapter
    metric; attributing it needs the noisy incl/excl contrastive machinery over many
    iterations. The SHARD is roughly the finest grain at which cond_gap attribution has
    any SNR; per-image is statistically hopeless (millions of images × multiple obs each).
  - **Hero-shard risk = proxy-optimal ≠ optimal** (the SREF-METRIC-1 surrogate trap one
    level down): harvesting "the best images" by a proxy risks (a) diversity collapse →
    fails the long tail of real user references; (b) overfitting the proxy. Mitigation:
    high-signal CORE blended with diversity, never a pool replacement; ADDITIVE new shards
    (re-sharding the pool breaks the carry-forward cache + scores — PRECOMP-4 lesson).
  - **SEQUENCING (load-bearing): gated behind SREF-METRIC-1.** Granularity amplifies
    whatever the search currency rewards — if cond_gap is misaligned with style (content-
    leak hint), finer selection just concentrates the misalignment (hero shards optimised
    for content-copying). Validate the signal (the sweep) BEFORE harvesting to it.
  - **Cheap first step (captures most value, no re-shard): a style-signal-weighted SAMPLER
    in the loader.** At batch 1 × 1000 steps the trainer already sees only ~1K of the
    staged ~40 shards' images — essentially at random. Sampling the highest-pair-richness /
    most-style-coherent staged images instead of random ones captures most of the
    "train on the best signal" benefit using data we already compute, with no re-shard, no
    cache breakage, reversible. Hero shards (curated re-sharding) is the heavier follow-on
    IF the sampler proves the gain. Groundwork already laid: source-probe + pool-wide
    pair-richness map + CSD embeddings are exactly the image-level signals this consumes.
  - **Secondary benefit if it works:** denser per-iteration signal also raises cond_gap
    SNR (less dilution from junk images), improving the SEARCH itself, not just the ceiling.

## Training & Model Quality

**TRAIN-7: IP-Adapter production quality roadmap** (High priority, next major release)

Proof-of-concept validated (2026-05-11, `train/reports/ip_adapter_v1/`): the adapter architecture and training signal are sound. The model responds to the style reference with coherent, stable output (CLIP-I 0.53, no NaN, correct image structure). The gap to production quality is entirely a matter of scale and refinement — no architectural rethink is required.

**PRODUCTION-READINESS PREREQUISITES (gating — from the warmup-run2/3 over-training finding, 2026-06).**
The IP-adapter over-trains: cond_gap (held-out conditioning quality) degrades while train_loss
falls. In the warmup flywheel this was severe (cond_gap +0.0273 → −0.0057 after ~2000 steps on
42 shards) because tiny per-iteration data → high epochs-per-sample. Production trains over
~200k-record chunks so the *severity* is far lower, but it shares the warm-start-and-keep-training
structure and is **currently unguarded**. Do NOT start a production run without these:

- **PROD-1: Create the held-out validation set + enable T-05.** The trainer's held-out
  validation (`_compute_val_loss`, T-05) is wired but DISABLED because no val set exists
  ("Validation held-out: not found — T-05 disabled"; the doctor's standing `val_set` warning).
  Run `pipeline_ctl create-val-set` + precompute it so T-05 has `_val_shards`. Prerequisite for
  ANY production run — without it, checkpoint selection has zero held-out signal.
- **PROD-2: Select checkpoints + early-stop on held-out COND_GAP, never train_loss.** The core
  lesson: train_loss (and even raw val_loss) fall while cond_gap degrades, so they are misleading
  stop signals. Extend T-05 to compute held-out cond_gap (loss_null − loss_cond) and select/early-
  stop at its peak. Bake the over-training signature (cond_gap down while train_loss down — already
  in the doctor's cond_gap-stall detector) into production monitoring. Also size the per-chunk step
  budget to chunk size (epochs-per-sample is the real overfit driver) — tune via the warmup/ablation.
  *Progress (2026-06-10):* the pure decision logic is built + tested — `train/ip_adapter/
  cond_gap_stop.py` (`best_point` selection, `should_stop` plateau early-stop, `is_overtraining`
  signature; mirrors the doctor detector), `train/tests/test_cond_gap_stop.py` (14 tests). Remaining:
  wire it into the trainer's T-05 eval loop (needs PROD-1's val set; touches the live trainer).
- **FLYWHEEL-CKPT-1: per-iteration checkpoint archival (start_step=0 collision). DONE 2026-06-10.**
  With `--warmstart-weights` (resume_from_champion) or from-scratch mode, every iteration saved
  `step_0001000.safetensors`, so `ckpt_path = ckpts[-1]` resolved to the same clobbered file each
  iter → get_best's recorded path pointed at the latest (not best) weights, reintroducing
  compounding at iter-3+. **Fixed:** in start_step=0 modes the orchestrator now `os.replace`s the
  iteration's file to a unique `iter{N}_step_*.safetensors`, records that, and prunes archived
  files to keep only the champion (get_best) + current. resume_from_champion is now safe.

- **PRECOMP-6 — FIXED 2026-06-10: precompute resume marked shards "fully done" that the
  coverage verifier rejects.** Observed: a run whose VAE output was 0/100 (tiled-VAE bug)
  exited 1 from the coverage check, but a rerun resumed it as "fully done" and exited 0
  with the gap intact. **Fix (both ends):** (a) shards with per-record skips
  (`skipped_q`/`skipped_v`) are no longer appended to `.precompute_done.json`; (b) the
  coverage-gap exit prunes gapped shards from the resume state, so a rerun re-processes
  exactly the holed shards.

- **TRAIN-PAD-1 (DEFERRED 2026-06-17): training text-pad convention diverges from reference inference.**
  GPU measurement required before picking fix strategy (a/b/c). Precompute slices to [:sl],
  training zero-pads, inference keeps non-zero pads — confirmed mismatch, unknown loss delta.
  The
  trainer/precompute zero-pads text embeddings to 512 (precompute stores only real-token
  rows), but the reference (mflux) and C inference pass **masked-encoder outputs** at pad
  positions (non-zero, prompt-derived — pad queries attend to real tokens). The IP-adapter
  is therefore trained under a slightly different text-conditioning distribution than it
  will see at C inference (relevant to G-1 Phase 2). Discovered while fixing QWEN-1
  (BUGS.md), where the WRONG resolution — changing inference to match training — regressed
  all generation. Options: (a) precompute/store full 512-row embeddings (storage cost),
  (b) mask text pads out of attention in the trainer's frozen base (closer to a true mask
  than either convention), (c) measure the adapter-quality impact first and accept if
  negligible. Do (c) first once a golden-set eval exists.
  - **Re-encode cost if (a):** Qwen3 only — h8 is fused with h17+h26 into one
    7680-dim quantized blob, so there is no partial re-encode; all 482,054 files
    (73 GB on `/Volumes/2TBSSD/precomputed/qwen3/`) regenerate. VAE (237 GB) and
    SigLIP (192 GB) live in separate dirs and are unaffected. Bumps the encoder
    version (PRECOMP-3 cache key); the old `v_*` dir stays until manually pruned.
    Wall-clock ≈ 5h (range ~4.5–6h), single worker / batch 16 — estimate from
    observed throughput: 604 probe shards averaged 8.5s per ~200-record subsample
    (~24 rec/s including per-shard warmup; steady-state ~29 rec/s as warmup
    amortizes over full shards). Caveat: probe captions averaged seq≈36; a
    longer-caption mix in the full set would push toward the upper end.
  - **Probe recipe for (c)** (sized 2026-06-11; one GPU hour, no trainer surgery):
    take a trained champion + ~32 val records. For each, build BOTH text variants —
    cached zero-pad rows (training convention) and a live Qwen3 encode of the padded
    sequence keeping pad-position outputs (inference convention) — then run the
    paired held-out loss (same noise/timestep per record, mirroring
    `_compute_val_loss`) under each. Report Δloss_cond and Δcond_gap. If |Δ| is
    within the val noise band → accept (c) and close; else implement (b) behind a
    config flag and ablate it. Implementation note: do this as a standalone script
    importing the trainer's loss pieces is NOT currently possible (`_compute_val_loss`
    is nested in main) — either hoist it module-level first, or add a
    `--pad-probe` early-exit mode to train_ip_adapter.py.

**Phased plan (full detail in memory file `train7_plan.md`):**

1. **Memory profiling run — GATE PASSED 2026-06-10.** 60 cached steps at each resolution
   (val-set shard, `memory_profile: true`):
   | | 768px | 1024px |
   |---|---|---|
   | fwd / bwd+param / ema peak (GB) | 16.86 / 19.33 / 16.99 | 17.13 / **21.32** / 17.26 |
   | system peak (GB) | ≤22.4 | ≤21.5 |
   | step time | ~9 s | ~18 s |
   System peak at 1024px ≈ 21.5 GB — matches the corrected ~21–22 GB estimate, ~10 GB
   headroom on 32 GB, **no gradient checkpointing needed → Stage 2 AND Stage 3 unblocked.**
   Caveats: probes ran style_loss off / batch 1 / cached encoders (production adds margin
   but stays well inside budget). Probe logs: /tmp/probe_{768,1024}px.log. Getting here
   flushed out MLX-1 (online-encode segfault, BUGS.md), the tiled-VAE bf16→numpy bug
   (fixed — PRECOMP-1 had never produced a real latent), and PRECOMP-6 (resume/coverage
   disagreement).

2. **Stage 2: 768px fine-tune** (20K steps, ~1 day) — `train/configs/stage2_768px.yaml`
   exists but is missing quality signals from Stage 1: `correct_forward_q`, `cross_ref_prob`,
   `patch_shuffle_prob`, `style_loss_weight`, `freeze_double_stream_scales`. Patch before
   launching. Target CLIP-I > 0.62. See `train7_plan.md` §3.

3. **Stage 3: 1024px fine-tune** (10K steps, conditional on profiling gate) — create
   `train/configs/stage3_1024px.yaml`. Warmstart from Stage 2. Target CLIP-I > 0.68.
   Requires adding `(1024, 1024)` to `BUCKETS` in `train/ip_adapter/dataset.py:62`.
   See `train7_plan.md` §4.

**Shard resolution — essential insight for 1024px:** Images are stored in shards at their
**original JPEG resolution** — `build_shards.py:321` only filters images smaller than 256px on
either dimension, no resizing. All resizing happens at training time in `dataset.py:444-445`
(Pillow LANCZOS to bucket+32px, then GPU random-crop). This means no shard rebuild is needed for
1024px training. However, any source image smaller than 1024px will be upscaled, which degrades
the VAE latent quality used for training (note: SigLIP is unaffected — it always resizes to
384×384 regardless of source size). **Before committing to a 1024px flywheel, run a shard sweep
to measure what fraction of images are < 512px on their shortest side.** See `train7_plan.md` §5.

**Training-time small-image filter (proposed, needs experimentation):** Rather than accepting all
upscaled images, add a `min_source_size` filter in `dataset.py` that skips images whose shortest
side is below a threshold relative to the training bucket. A 2× max-upscale heuristic (source ≥
50% of training resolution) is the natural starting point — e.g., 512px minimum for 1024px
training, 256px minimum for 512px training (current effective floor). The optimal threshold is an
open question: too aggressive a filter shrinks the effective dataset and risks domain bias; too
permissive lets in degraded signal. The right answer comes from an ablation across thresholds
(0, 384, 512, 768, native-only) measuring val CLIP-I alongside effective dataset size at each
threshold. See `train7_plan.md` §5.4 for the implementation sketch and ablation design.

**Dependency summary:** Profiling run (§1) and shard resolution sweep (§5) both unblocked and should run together; Stage 2 (§2) unblocked pending profiling; Stage 3 (§3) conditional on gate + sweep results. TRAIN-6, PIPELINE-27, QUALITY-10 — done, see COMPLETED_BACKLOG.md.

---

## Pipeline Improvements

**PERF-IO-1: Routine ops = metadata lookups, not disk scans** (Medium — keeps the cold
HDD quiet during the flywheel). Principle: any non-troubleshooting path (status, doctor
summary, shard selection) should answer from metadata (SQLite DBs + per-version
`manifest.json` `record_count`/`shard_count` + sentinels/heartbeats), and full disk scans
should sit behind an explicit `--deep` flag for when you suspect metadata↔disk drift.

Audit (2026-06): the backbone is good — 5 DBs (`shard_index`, `shard_scores`,
`flywheel_history`, `experiments`, `monitoring`), manifests already carry the counts, and the
doctor's summary `records` is already manifest-backed. Remaining offenders that still walk the
cold dirs on routine calls:
- **DONE (2026-06):** `pipeline_doctor` listed the same cold precompute version dir ~2-3× per
  encoder per run (coverage count + `.tmp.npz` + `.npz.tmp.npz` hunts) — added a
  per-invocation memoized `_listdir_cached` so each cold dir is walked once.
- `pipeline_doctor._count_shards_with_precomp` (per-chunk precompute coverage) still walks the
  cold version dir — needs **per-chunk/shard coverage metadata** (extend the manifest, or
  derive from `shard_scores.db`) to become an O(1) lookup. Gotcha: must stay chunk-scoped.
- `pipeline_doctor._count_shards_for_chunk` → `shard_index.db` (count shard_ids in range).
  Gotcha: DB = *indexed* shards vs listdir = shards *present on disk*; validate equivalence
  before swapping.
- **`.tmp.npz` / `.npz.tmp.npz` crash-artifact hunts** → move behind `--deep` (they're
  troubleshooting, not routine — currently run on every invocation).
- `shard_selector.scan_shard_pool` globs the cold pool → `shard_index.db`.
- `pipeline_status._count_precomputed/_count_tars_recursive` rglob the **hot** staging dir
  (TTL-cached, lower priority — no manifest on in-flight staging; add a per-iter staging
  manifest if it becomes hot).

**PIPE-ORCH-1: Orchestrator coverage gaps — paths not exercised by smoke run** (Low priority, code bugs fixed)

Smoke run 3 (2026-05-11) validated the happy path across all 14 steps × 2 chunks. Three code bugs found in audit were fixed (commit `cdd9fb0`, 2026-05-13):

- ~~`_check_hot_space()` dead code~~ — now called in `_stage_shards()` and `_stage_precomputed()` with pre-scanned transfer size before any copies begin. `staging_margin_gb: 50` is now enforced.
- ~~GPU_TOKEN race~~ — `_start_training()` returns early when window is gone but `EXIT_CODE` not yet written.
- ~~Duplicate dispatch on restart~~ — `_stager_dispatched_errors` pre-seeded from `dispatch_queue.jsonl` via `_load_open_dispatch_ids()`.

**Remaining: validation gaps only (no known code bugs)**
- LAION/COYO/WikiArt download paths and chunk 3+ sequencing — code generalises correctly; untested at scale.
- Real two-device stager (cold→hot copy path) — `_check_hot_space()` now wired; needs a real `/Volumes/16TBCold` → `/Volumes/2TBSSD` transfer to verify. Will be exercised on the next pipeline run (PIPELINE-25 done, cold pool active).
- `stage.done` gate blocking training, `_poll_stager` retry after error, training crash one-retry + escalate — all coded correctly; never exercised end-to-end.
- GPU_TOKEN contention at production timing — documented; code fix applied; no observed failure.
- Download throttle stall false-positive — documented in DISPATCH.md Gap 6 as a known operator issue.
- `dispatch-resolve` UI-only clarification — documented in DISPATCH.md.

**PIPELINE-25b: Stream-convert downloads — eliminate raw tgz disk writes** (Low priority, long-term)

Currently `download_convert.py` downloads each JDB tgz to disk, then reads it back for conversion. Since downloads are sequential (one tgz at a time) and tgzs are small enough to hold in memory (~2-3 GB each, well within the 32 GB system RAM), the raw bytes could be streamed directly through `_convert_tgz()` without touching disk at all.

**When applicable:** only when `raw_pool_root` is not configured (i.e. no persistent raw pool needed). If the raw pool is enabled, the tgz must land on disk anyway. This optimisation targets the no-pool path or cases where the caller explicitly opts out of raw storage.

**How it would work:**
- `hf_hub_download()` supports a streaming/file-object mode; alternatively, download to a `tempfile.SpooledTemporaryFile` in memory.
- Pass the in-memory buffer directly to `tarfile.open(fileobj=buf, mode="r:gz")` inside `_convert_tgz()`.
- WebDataset output tar is written to disk as today (it is the persistent artifact).
- Eliminates one full disk write + one full disk read per tgz; saves ~2-3 GB × N tgzs of I/O.

**Constraint:** HuggingFace's `hf_hub_download` API always writes to a local path; would need to switch to `huggingface_hub.file_download.http_get()` or `requests` + streaming response to avoid the intermediate file. Alternatively, download to a RAM-backed tmpfs (`/dev/shm` on Linux; macOS has no equivalent — would need to use `tempfile` with a memory-sized cap). Investigate feasibility before committing to this path.

**Interaction with converted pool:** if `converted_pool_root` is set, the Level 0 hit (skip download+convert entirely) already makes this optimisation irrelevant for warm runs. This item only matters for the first-time conversion of each tgz.

---

~~**DEDUP-1: Clean the converted pool at source + retroactive pool cleaning script** — Done. See COMPLETED_BACKLOG.md.~~

**DEDUP-3: clean_wds_pool self-dupe false positives on interrupted restart** — Done (2026-06-17). Structural fix landed.

When `clean_wds_pool.py` is killed mid-tar (after FAISS index is written to disk but before the `.deduped` sentinel is written), the FAISS index on disk contains vectors from the partially-processed tar. On restart, that tar has no sentinel so it is reprocessed — its images search against the index and find themselves, scoring similarity ≈ 1.0 above the 0.95 threshold, causing false-positive duplicate removal.

**Confirmed occurrence (2026-05-17):** 003.tar was partially indexed (23 vectors added, no sentinel written) before the process was killed due to swap pressure. The 4-49 batch was kicked off separately. 003.tar must be processed with a trimmed index to avoid false positives.

**Manual fix for 003.tar:**
```python
import faiss, numpy as np
index = faiss.read_index('/Volumes/16TBCold/metadata/dedup_index.faiss')
ids = open('/Volumes/16TBCold/metadata/dedup_index.ids').read().splitlines()
n_clean = len(ids) - 23  # remove 003.tar partial vectors (last 23 in insertion order)
vecs = np.zeros((n_clean, index.d), dtype=np.float32)
index.reconstruct_n(0, n_clean, vecs)
new_idx = faiss.IndexFlatIP(index.d)
new_idx.add(vecs)
faiss.write_index(new_idx, '/Volumes/16TBCold/metadata/dedup_index.faiss')
open('/Volumes/16TBCold/metadata/dedup_index.ids', 'w').write('\n'.join(ids[:n_clean]) + '\n')
```
Then run: `train/.venv/bin/python train/scripts/clean_wds_pool.py --tgz-range 3 3`

Note: trim must be done AFTER the 4-49 batch completes, as the 4-49 batch uses the current index as its duplicate reference. Trimming during that run would break its dedup consistency.

**Structural fix (DONE 2026-06-17):** `clean_wds_pool.py` now writes a `.processing`
marker (`{index.ntotal}\n{tar_name}`) before each tar's vectors are added. On startup,
if the marker exists, the index is truncated back to the saved `ntotal` (rebuilding the
`IndexFlatIP` over the first-N reconstructed vectors, in insertion order), the `.ids`
sidecar is truncated to match, the interrupted tar's `.deduped` sentinel (if any) is
removed so it is reprocessed cleanly, and the marker is deleted. The same rollback
(`_truncate_index` / `_truncate_ids`) also runs between retry attempts and after a
tar's final failure, so a partially-added tar never pollutes the next tar or a restart.
Interrupted runs are now automatically safe to restart. Guarded by
`TestCleanWdsPoolRollback` in `train/tests/test_scripts.py`.

---

~~**DEDUP-2: Re-dedupe 001.tar after redownload** — Done (2026-05-17)~~

---

**DEDUP-4: Run clean_wds_pool on WikiArt and other datasets for cross-dataset dedup** (Medium — data quality)

The FAISS index at `COLD_METADATA_DIR/dedup_index.faiss` is persistent and global — `clip_dedup.py build-index` extends it on each run rather than rebuilding from scratch. Running `clean_wds_pool.py` with a different `--pool-dir` pointing at another dataset's converted tars will deduplicate against the full accumulated index, catching cross-dataset duplicates (e.g. WikiArt paintings that appear in LAION as scraped web images).

**Run order matters** — whichever dataset goes through first keeps its copy; later datasets lose their duplicates. Recommended order:

1. WikiArt — small, curated; run first so curated entries are preserved
2. JourneyDB — already running (tgz 1–100 current run)
3. LAION / COYO — largest, most overlap with everything; run last so dupes are removed from scraped data

**To run** (after WikiArt tars are converted and available on cold):
```bash
train/.venv/bin/python train/scripts/clean_wds_pool.py \
  --pool-dir /Volumes/16TBCold/converted/wikiart \
  # --index and --blocklist default to COLD_METADATA_DIR, same shared index
```

Repeat for each additional dataset. The `.deduped` sentinel per-tar makes runs idempotent.

---

~~**PIPELINE-DATA-1: Dedicated LAION/COYO download script + pipeline_setup integration** — Done (2026-05-17)~~

Implemented: `train/scripts/download_laion_coyo.py` (new), `downloader.py` (cold-pool symlink staging in `check_laion`/`check_coyo`), `pipeline_setup.py` (`_check_laion_coyo_pools` detection), `v2_pipeline.yaml` (`data_sources` block). Uses HF-hosted repos (no URL staleness). Configure `laion_hf_repo` in `v2_pipeline.yaml` before first run.

---

**TELEMETRY-1: Pipeline telemetry gaps — audit findings** (Multiple priority levels — see report)

Full audit report: [`plans/telemetry-audit.md`](plans/telemetry-audit.md)

**Summary of dead telemetry (logged but never consumed):**
- ~~`validator_chunk{N}.jsonl` events~~ — covered by T5 fix; metrics.json surfaces same signal
- ~~`ema_drift` in trainer heartbeat~~ — now displayed in `pipeline_status.py` (QW-5 done)
- ~~`siglip_coverage_pct` in trainer heartbeat~~ — now displayed in `pipeline_status.py` (QW-5 done)
- ~~`bucket_stats` in heartbeat/log~~ — now displayed as `buckets:` line in trainer status (T4 done 2026-05-17)
- ~~`logs/val_chunk{N}/metrics.json` (CLIP-I, adapter_delta)~~ — now shown per-chunk in pipeline_status.py (T5/QW-10 done 2026-05-17)
- ~~`selection_log` table in `shard_scores.db`~~ — now queryable via `data_explorer selection-history <campaign>` (T6 done 2026-05-17)

**Quick wins (low effort, implement opportunistically):**
- ~~**QW-1**: Add `grad_norm_final`, `ip_scale_final` indexed columns to `ablation_history.db`~~ — Done (2026-05-17)
- ~~**QW-3**: Add `steps_per_second` to trainer heartbeat~~ — Already done (was in heartbeat)
- ~~**QW-4**: Persist `_restart_counts` to `pipeline_state.json`~~ — Done (2026-05-17)
- ~~**QW-5**: Surface `ema_drift` and `siglip_coverage_pct` in `pipeline_status.py`~~ — Done (2026-05-17)
- ~~**QW-6**: Add `stopped_early` + `stop_step` to ablation DB~~ — Done (2026-05-17)
- ~~**QW-7**: Log `threshold_loss` and `skipped` count to `mine_hard_examples.jsonl` done event~~ — Done (2026-05-17)
- ~~**QW-8**: Dispatch WARNING when trainer heartbeat `mem_available_gb < 3.0`~~ — Done (2026-05-17)
- ~~**QW-10**: Make `pipeline_status.py` read `logs/val_chunk{N}/metrics.json`~~ — Done (2026-05-17)

**High-value deeper changes:**
- ~~**DA-6**: Write per-shard hard-example density into `shard_scores.db`~~ — Done (2026-05-17): `hard_example_count` column added; updated after each mine run
- ~~**DA-3**: Campaign-level summary table in persistent DB~~ — Done (2026-05-17): `campaign_summary` table in FlywheelDB; refreshed after each iteration; viewable via `data_explorer campaign-summary`
- ~~**DA-2**: Per-shard loss percentile distribution in mining output~~ — Done (2026-05-17): writes `shard_loss_percentiles.json` (p50/p75/p95/p99 per shard) after each mine run
- **DA-7**: Link validation metrics back to `ablation_history.db` via `post_train_validation` table — deferred (requires validator subprocess from ablation harness, ~1 day)

**PIPELINE-32: Small dedicated Hot ↔ Ultrahot tier movement helper script** (Medium — experiment velocity, unblocked)

With the three-tier storage model now active (Cold = 16 TB HDD as immutable source of truth, Hot = 2 TB external SSD as primary training compute, Ultrahot = internal NVMe for small/fast runs), operators need an easy, safe way to move a small active working set between Hot and Ultrahot.

Current situation:
- `data_stager.py` handles Cold ↔ Hot staging/archiving very well (including background operation and the orchestrator integration).
- Moving between the two SSD tiers (e.g. "I want to run a quick 8-shard ablation burst at maximum speed on the internal NVMe" or "I did some interesting small runs on Ultrahot, promote the new checkpoints + metrics back to Hot") is still manual, error-prone, and lacks the safety nets (size checks, manifests, provenance, heartbeats, resume) that the main stager provides.

Deliver a small, focused helper (suggested name: `ultrahot_stage.py` or a `data_stager.py ultrahot-*` subcommand) that supports:
- Stage a curated subset (explicit shard list, top-N from `shard_scores.db`, latest ablation run, specific campaign window, etc.) from Hot → Ultrahot.
- One-way or bidirectional sync of results (new checkpoints, updated `ablation_history.db`, metrics, small precompute deltas) back from Ultrahot → Hot.
- Proper handling of versioned precompute `current` symlinks, checkpoint lineage, and small manifests so the main pipeline/doctor can see what's resident on Ultrahot.
- Space estimation + safety abort if Ultrahot would go below a configured margin.
- Heartbeat + sentinel files compatible with the existing observability stack.
- Optional "ephemeral experiment" mode that cleans up after itself on Ultrahot when done (or on explicit `cleanup` command).

The tool should be usable both standalone (for quick experiments) and callable from `pipeline_ctl.py` / the ablation harness.

**PIPELINE-33: Define and codify lightweight Hot ↔ Ultrahot movement policy** (Low–Medium — operational clarity)

Accompanying the script in PIPELINE-32, document (and where helpful, lightly enforce) the intended policy for when work should live on Hot vs. Ultrahot:

- Size / duration heuristics (e.g. < ~40 shards and < 10k steps → prefer Ultrahot; larger or multi-day runs → Hot).
- How the ablation harness and `test_quality_features.py` / smoke runners declare their working-set size so the right tier is chosen automatically.
- Rules for promoting "interesting" results from Ultrahot back to Hot (and then to Cold archive).
- Visibility in `pipeline_status.py`, `pipeline_doctor.py --ai`, and `data_explorer` so an operator always knows what is currently on Ultrahot.
- Interaction with the main orchestrator (should it ever auto-stage tiny windows to Ultrahot, or is Ultrahot strictly a manual "fast lane" for the operator?).

Start with clear documentation + sensible CLI defaults in the new helper; wire deeper automation later once real usage patterns emerge.

**Effort estimate:** PIPELINE-32 ~2–3 days (script + integration + tests). PIPELINE-33 ~1 day (policy doc + small CLI/policy hooks). Both are high-leverage for daily experiment velocity on the current hardware.

---

## Ablation Harness Improvements

**ABL-FIDELITY: ablation-proxy fidelity + cross-run quality gate** (the ablation arm is a short, cold-start proxy for a warm-started long run — two fixes, both landed as opt-in/instrument, validation deferred to a GPU-free-window):

- ~~**Arm warm-start (opt-in)**~~ — DONE (default OFF). Arms cold-start (harness gets caches+shards, no checkpoint); the flywheel's own per-iter training warm-starts — a fidelity gap. Added `train_ip_adapter --warmstart-weights` (loads adapter weights but keeps start_step=0 — fresh schedule, unlike `--resume` which continues the step count), harness passthrough, and `orchestrator._ablation_warmstart_ckpt(fw_cfg, ckpt_dir)` gated on `ablation_warmstart_arms` (default false). Kept off because warm-start trades fidelity for *discrimination* (arms diverge less in 1000 steps from a shared checkpoint) — which is better is empirical, settle it with the gate below. Tests in test_orchestrator_state.py.
- ~~**Cross-run quality gate (instrument)**~~ — DONE (built, GPU eval deferred). `train/scripts/quality_gate.py`: `compare_quality(current, previous)` (pure verdict core — clip_i/clip_t/aesthetic higher-better, lpips/fid lower-better; REGRESSION/IMPROVED/NEUTRAL/NO_BASELINE) + `run_quality_gate` glue (golden-set eval → weight_registry register → compare-to-prior; eval/registry injectable) + CLI (`--fail-on-regression` gates a pipeline). 16 tests in test_quality_gate.py. This is the apples-to-apples *output*-quality comparison that validates whether an ablation-chosen config actually improved the long run vs the previous one — and the instrument to A/B warm-start vs cold-start arms.
- ~~**Auto-run wiring**~~ — DONE (default off). Flywheel campaign-end hook runs `quality_gate.py` on the champion checkpoint (`fw_db.get_best`) when `quality_gate: true` is set in the flywheel config; pure `orchestrator._quality_gate_target` gates it (4 tests). Skipped entirely when the flag is off (today). **OPEN (needs GPU / golden set):** configure the golden set + run the gate on the M5 Max; then the warm-start-arms A/B to set that default. No automatic golden regression *gate* in the chunk pipeline yet (GROK-TEST-4).

~~**ABL-1: Trial-level wallclock timeout** (High — safety)~~ — DONE (class `TrialTimer`
in `ablation_harness.py`, wired into `_run_one`; `trial_timeout_secs` config, default
14400). Unit-tested 2026-06 in `train/tests/test_ablation_safety.py` (fires SIGTERM after
timeout; cancel() before timeout is a no-op). The backlog entry was stale — the feature
shipped but had no test coverage.

---

~~**ABL-2: Multi-signal early stopping** (High — quality + time savings)~~ — DONE.
`EarlyStopper` monitors all four signals (cond_gap floor, loss/NaN instant-kill,
grad-norm explosion over 3 snapshots, ref_gap style-dead) with `trigger_reason` for
DB logging. Unit-tested 2026-06 in `train/tests/test_ablation_safety.py` (incl. the
backlog's own success criterion: constant loss=9.0 → kill on snapshot 1). Backlog
entry was stale — shipped without test coverage. Original design retained below.

`EarlyStopper` currently monitors only `cond_gap < min_cond_gap`. Two failure modes it misses:

1. **Loss explosion / NaN**: `loss_smooth > 5.0` should cause an immediate kill (no patience needed) — the run is unrecoverable. Currently only detected at scoring time, wasting potentially hours of GPU.
2. **Style signal absent**: `ref_gap < min_ref_gap` for `ref_gap_patience` consecutive snapshots means the adapter is not learning from the style reference at all. This is a distinct failure from low cond_gap (adapter may learn from null prompt but style is dead).

Extend `EarlyStopper.__init__` with `nan_loss_threshold`, `grad_norm_threshold`, `ref_gap_min`, `ref_gap_patience`. Add `trigger_reason` property for DB logging. Config YAML:

```yaml
early_stopping:
  enabled: true
  min_cond_gap: -0.3
  patience: 4
  min_snapshots: 5
  nan_loss_threshold: 5.0      # instant kill
  grad_norm_threshold: 50.0    # explosion guard (3 consecutive snapshots)
  ref_gap_min: -0.5            # style-dead guard
  ref_gap_patience: 6
```

**Success criteria:** Deliberately broken training (return constant loss=9.0) triggers immediate kill within 2 snapshots. Normal runs unaffected. Trigger reason recorded in DB.

---

**ABL-3: Pareto-front warm-start for NSGA-II** (Medium — quality)

`_warm_start_candidate` loads the single highest-scored experiment from a prior campaign and injects it as the first candidate. For NSGA-II this is sub-optimal: the multi-objective sampler needs multiple diverse seed points to bootstrap the Pareto front estimate — a single point gives it nothing to work with for N_initial trials. Add `_warm_start_pareto(top_k=3)` that loads the top-K Pareto-optimal experiments from the prior DB (excluding already-tried) and injects them as forced first candidates. Single-objective campaigns use the existing single-best path; NSGA-II campaigns use the Pareto path.

**Success criteria:** NSGA-II campaign warm-started from a prior run with 3 Pareto seeds produces a first-generation Pareto front within 5 trials vs 15+ without seeding.

---

~~**ABL-4: Pareto scatter plot in HTML report** (Medium — interpretability)~~ — DONE.
The scatter (`drawPareto` in `ablation_harness.py`: ref_gap X / cond_gap Y, Pareto
points starred + outlined, a gold "best-compromise" point, zero-lines, axis labels)
already existed — the entry was stale. The remaining spec gap, a **hover tooltip
showing the combo + score + metrics for any trial**, was added 2026-06 (vanilla JS,
no deps): `paretoTip` div + canvas-coord hit-testing in `drawPareto`. Locked in by
`train/tests/test_ablation_report.py` (renders synthetic results, asserts the
template fully formats and the tooltip/hover elements are present — the report is a
brittle str.format() template where one unbalanced brace silently breaks rendering).

---

## Shard Scoring Improvements

**SHARD-1: Temporal momentum in shard score updates** (Medium — data quality)

`shard_selector.py score_update()` computes a cumulative running mean: older iterations count equally with recent ones. After 20+ flywheel iterations a shard's score is dominated by its earliest runs, which may have been under a very different hyperparam regime (before ablation tuning). Add exponential moving average (EMA) weighting in `score_update` with a configurable `temporal_decay` (default 0.85 — each new iteration counts as 85% new, 15% history). The existing `n_scored` counter is preserved for confidence estimation. Add `temporal_decay` as a top-level config option in `shard_selector.yaml` / flywheel YAML.

**Success criteria:** After 10 simulated score updates, the EMA-weighted mean responds to a regime change (scores flip from 0.3→0.7) in 3 iterations vs 10+ for the flat mean. Old campaigns unaffected (omitting `temporal_decay` keeps the existing equal-weight cumulative mean; temporal_decay=1.0 gives last-value, not flat mean).

---

**SHARD-2: Hyperparam-conditioned score table** (Low–Medium — long-term quality)

A shard that scores well with `style_loss_weight=0.12` may score poorly with `style_loss_weight=0.0`. The current IPS attribution averages across all hyperparam configurations, conflating effects. Add a `shard_scores_by_regime` table to `shard_scores.db` that stores a separate (score, n_obs) pair per shard per param-regime hash. The flywheel's `_select_shards` call can pass the current ablation best-params as the active regime, and `score_update` partitions attribution by regime. Useful once ≥3 ablation iterations have been run under a stable best-config.

**Dependencies:** Requires ABL-3 (stable best-params are needed as the regime key).

---

## Precompute / VAE Improvements

**PRECOMP-1: VAE tiling for high-resolution precomputation** (High — enables Stage 2/3, unblocked)

`precompute_all.py _preprocess_vae` encodes images at a single fixed resolution by resizing to `image_size` before encoding. Stage 2 (768px) and Stage 3 (1024px) training requires latents precomputed at that resolution. Encoding a 1024×1024 image through the Flux VAE in one pass requires ~6 GB of intermediate GPU memory — tight but feasible at 14 GB limit. However, for robustness and future-proofing (1280×1280+ or multi-crop), implement overlap-blend tiling: split into 512px tiles with 64px overlap, encode each tile independently, blend the overlap regions with a bilinear feather mask, reassemble the full latent.

**Implementation:** New function `_encode_vae_tiled(vae, img_tensor, tile_size=512, overlap=64)`. Used in the VAE encoding pass when `image_size > 512`. The existing single-pass path is preserved as the fast path for ≤512px.

**Success criteria:** A 1024×1024 image encoded via tiling produces a latent with ≤0.5% mean absolute difference vs single-pass encoding. Peak GPU memory stays below 8 GB during tiling (each tile is independent). No regression on 512px precompute.

---

~~**PRECOMP-2: Tiny proxy VAE encoder** (Low — long-term throughput)~~ — IMPLEMENTED
in v3.18.0 + v3.19.0 (`train/vae_distill/`). The design deviated from this sketch
(task-specific stride-8 encoder, not EfficientNet-B0; composite loss = channel-norm
MSE + decoded MSE + frequency-weighted MSE + distribution matching, not plain MSE)
for the reasons in `plans/precomp2-proxy-vae-design.md`. Ships small/default/medium
variants (3.4M/6.0M/9.3M), ProxyVAE with confidence gating + regression detection +
quality modes, evaluate_vae_proxy.py, benchmark_vae_proxy.py, compare_downstream_quality.py,
and precompute_all/orchestrator integration. **Still pending validation** (idle GPU +
a trained proxy): the 5–7× speedup number and the downstream A/B verdict — see the
flywheel-gated items. Migration: `plans/proxy-vae-v3.19-migration.md`.
  - **Validation runbook: `plans/proxy-vae-validation-runbook.md`** — train→Tier1→
    benchmark→Tier2 with pass gates, fallback-threshold tuning, subsampling alt arm,
    and sequencing against flywheel-idle / M5 bring-up.
  - **Tier-1 verdict (2026-06-11): SMALL variant FAILS on capacity.** 27K steps
    (latent-only after PROXY-1), loss flat from ~18K; on 994 held-out pairs:
    cosine 0.860 (>0.95), ch_std_ratio 0.868 (0.95–1.05, under-dispersed),
    fft_corr 0.918 (>0.98). Flat-loss + wide misses + under-dispersion = capacity
    ceiling, not under-training. Checkpoint kept at
    `/Volumes/2TBSSD/checkpoints/vae_proxy/proxy_step_0026000.safetensors`;
    report `/Volumes/2TBSSD/proxy_vae_eval.json|.html`.
  - **Next: MEDIUM variant overnight retrain** (same recipe, 9.3M params) at the
    next free GPU night. **Capacity bisection plan:** small (3.4M, fails) and
    medium now bracket the threshold; if medium passes Tier-1 with large margins
    (e.g. cosine ≥0.98), train an intermediate size — the proxy's value is speed,
    so the smallest PASSING model wins; each candidate costs one overnight run +
    an 85 s Tier-1 verdict on the same 994 pairs. Not on the SREF critical path
    (run5 uses DP-2c subsampling with the real VAE); needed before the
    full-coverage production precompute.
  - **Sizing by extrapolation, not bisection (if medium overshoots):** fit the
    scaling law `1−cosine ≈ a·N^(−b)` through small (N=3.4M, err=0.140) and
    medium's result, solve for the N where err crosses the 0.05 gate, add ~25%
    parameter margin (two-point fits don't validate the exponent). Fit on cosine
    only — ch_std_ratio is a capacity *symptom* that snaps to ~1.0 once adequate;
    verify it + fft on the candidate. Free helpers: (a) the DEFAULT 6.0M variant
    is a third trainable point (or may simply BE the answer); (b) speed(N) needs
    no training — benchmark_vae_proxy on random weights maps the throughput curve
    in minutes, so pick N maximizing speedup subject to predicted gate-pass;
    (c) Tier-1 is 85 s on any 2K-step checkpoint — eval the fitted candidate
    mid-training and stop early once all three gates clear.
  - **Two BLOCKING pre-trust items (must land before any proxy latent ships), from
    `grok_proxy_vae_analysis.md`:** (C-1) no golden test that C `iris_vae` encode/decode
    matches the teacher VAE the proxy is trained against — proxy can pass Tier-1/2 vs
    teacher yet degrade real C inference; add per-channel + decoded-LPIPS/PSNR golden +
    CPU/Metal decode parity. (C-2) brittle ad-hoc `vae/config.json` strstr/atoi parse can
    select the wrong z/scale/shift/BN branch; schema-validate or golden the resolved VAE
    config. These are doable NOW (no GPU) and are prerequisites regardless.

---

**PRECOMP-3: Precompute version key is over-sensitive — bound to whole-repo git SHA, not encoder identity** (High — directly causes redundant re-precompute; compounds with the v3.23.0 cache fix) — **CODE FIX LANDED; cold-data migration pending run-end.**

**Status (2026-06-03):** `cache_manager.version_hash` no longer mixes in the git SHA — the key is now `sha256(config_subset)` where `config_subset` carries a per-encoder `code_version` (`ENCODER_CODE_VERSION = {qwen3,vae,siglip → "1"}`, bump only on an encoder-output-semantics change). `encoder_config_subset` folds `code_version` in; `git_sha` is still accepted (call-site compat) and recorded in the manifest for provenance but does not affect the key. Added `cache_manager.py consolidate <encoder|all> [--apply]` (hardlink-based union of same-identity version dirs, dedup by filename, repoint `current`, drop redundant dirs) + `list`. `list_versions` now skips the `current` symlink (was a phantom version). Tests in `train/tests/test_cache_manager.py` (git-SHA-independence, code_version bump, consolidate dry-run/apply/skip-different-identity/ignore-empty-stub). **Remaining:** after the warmup-run1 flywheel finishes (iter 15) and the orchestrator is restarted to pick up the new key, run `cache_manager.py consolidate all --apply` on `/Volumes/16TBCold/precomputed` to fold the per-SHA dirs (`v_d9a32b`, `v_c56d1c`, + iters 12–15) into the single canonical `v_059443`-class version. Until that restart the running orchestrator keeps the old git-SHA key (it won't reload the edited module), so iters 12–15 still publish per-SHA dirs — expected, the consolidation captures them.

Original analysis (kept for context):

`version_hash(config_subset, git_sha)` in [cache_manager.py:60](train/scripts/cache_manager.py#L60) computes the precompute cache version as `sha256(json(config_subset) + git_sha[:8])`. The `config_subset` correctly captures encoder identity (e.g. qwen3: model + extracted layers + think_tags), but mixing in the **whole-repo git SHA** means *any* commit anywhere in the tree — orchestrator, doctor, docs, an unrelated C change — produces a brand-new version dir and forces a full re-precompute of the same shard pool, even though the encoder and its outputs are byte-identical.

**Observed impact (flywheel warmup-run1).** iters 10 and 11 published functionally identical Qwen3 config (`Qwen3-4B`, layers `[8,17,26]`, `think_tags:true`) under two different versions purely because we committed code between iterations:
- iter 10 → `v_d9a32b` (git_sha `31e647a4`, 194,127 records, 40 shards)
- iter 11 → `v_c56d1c` (git_sha `5d36078c`, 199,271 records, 40 shards)

The two iters share 12 anchor shards (`000000`–`000011`); those 12 were re-encoded from scratch in iter 11 instead of being reused from iter 10's cold version, despite zero change to the encoder. With the cache key fixed, iter 11 would have reused iter 10's overlapping shards (and the hot/cold `current` symlinks would not fragment across SHAs).

**Fix direction.** Key the precompute version on **encoder identity only**, not the global repo SHA:
- Replace `git_sha[:8]` in the version hash with a narrow fingerprint of the code that actually affects this encoder's output — e.g. a hash of the relevant encoder module(s) (`precompute_all.py` encoder fn + the model/config), or a hand-maintained `ENCODER_CODE_VERSION` bumped only when the encoding semantics change. The principle is already stated in this file's Warm-Start section ("6-month-old embeddings remain valid if the encoder is unchanged") — the implementation contradicts it.
- Keep `encoder_config_subset` as-is (it's correct).
- On change, add a one-shot migration/alias so existing `v_d9a32b`/`v_c56d1c` data is discoverable under the new key (or a `cache_manager.py --warm-start-precompute` pass to fold them forward) — don't strand the ~68 shards already on cold.

**Tests:** version_hash is stable across an unrelated git commit (same encoder config + different repo SHA → same version); version_hash *changes* when the encoder config subset changes; migration/alias resolves old version dirs. Pure-function, no GPU, no network — same pattern as the existing cache_manager tests.

---

**PRECOMP-4: Aspect-ratio bucketing, end-to-end** — full implementation plan: [plans/precomp4-aspect-bucketing.md](plans/precomp4-aspect-bucketing.md) (decision: per-image-bucket, NOT re-shard, to preserve the carry-forward cache + scores; shared `aspect_bucket` primitive landed in train/ip_adapter/bucketing.py). Original sketch: (re-shard by shape → multi-res precompute → aspect-aware loader → refine)** (High for final quality; **not** needed for the warmup bootstrap, which is correctly pinned to 512² square)

The data path squashes every image to a single **square** resolution and the training
loader does not honour aspect ratio, so the multi-resolution machinery that exists is
non-functional. Discovered when the quick training test hit 100% VAE cache-miss
(latent shape ≠ training bucket); root-caused to a chain of three issues:

1. **Precompute is single-resolution square.** `_preprocess_vae` does
   `PilImage.resize((image_size, image_size))` ([precompute_all.py:162](train/scripts/precompute_all.py#L162)) —
   a squash, not aspect-preserving. Every latent is `(32, S/8, S/8)` regardless of
   source aspect, so only one training bucket can ever match.
2. **The shard loader random-buckets.** `make_prefetch_loader(bucket=None)` picks
   `rng.choice(BUCKETS)` **per shard** ([dataset.py:445](train/ip_adapter/dataset.py#L445)),
   not the per-image aspect bucket. The aspect-aware `_select_bucket` exists but is
   unused by the shard loader. So even with per-bucket precompute, batches would
   mismatch. `_load_vae_latent` then rejects any shape mismatch as a cache miss
   ([dataset.py:235](train/ip_adapter/dataset.py#L235)); in cached mode (no live VAE)
   a miss is an unrecoverable skip → ~100% skip → trainer exits.
3. **Shards mix aspect ratios.** Tars hold arbitrary-shape images, so a per-shard
   single bucket squashes all of a shard's varied images into one shape.

**Current workaround (shipped):** `data.bucket: [512, 512]` pins training to the one
bucket that matches the square precompute (wired in train_ip_adapter.py; set in
stage1_512px.yaml). Training is then *consistent* (squashed image ↔ squashed latent)
and valid — the bootstrap IP-adapter learns aspect-tolerant style conditioning and is
a usable warm-start. Quality cost: geometric distortion + square output bias.

**Roadmap to distortion-free multi-aspect (do after the warmup proves the loop):**
- **Re-shard by aspect** — group images into aspect-homogeneous shards (recommended:
  makes the existing per-shard-single-bucket loader *correct*, gives clean same-shape
  caches and retrace-free homogeneous batches). Alternative: keep tars, do per-image
  bucket precompute (variable-shape latents per shard) + an aspect-grouping loader.
- **Multi-res precompute** — encode each shard at its bucket's native (non-square)
  resolution using the **teacher VAE** as ground truth, with the **distilled proxy
  (PRECOMP-2)** for speed (confidence-gated, teacher fallback; teacher authoritative
  for golden/validation). Large buckets use VAE tiling (PRECOMP-1). Retrain the proxy
  on the new pairs — the current proxy only saw square 512².
- **Fix the loader** — assign each image to `_select_bucket(w, h)` instead of
  `rng.choice`; build homogeneous batches; drop the 512² pin.
- **Keep SigLIP consistent** — SigLIP is architecturally fixed to 384² (so "full
  image" still resizes); today's 384² squash is *coherent* with the 512² VAE squash
  (conditioning aligns with target). Under the aspect regime, feed SigLIP the *same*
  aspect-bucketed image resized to 384² so conditioning and target stay aligned.
  Higher-detail style needs a larger/tiled vision encoder — separate, lower priority.
- **Refine the adapter** — warm-start from the 512² foundation weights; fine-tune on
  the multi-aspect cache to correct geometry/proportion handling.

Relates to PRECOMP-1 (tiling for large buckets) and PRECOMP-2 (proxy). The re-shard is
the big one-time data job; everything else composes on top.

**PRECOMP-4b: Make the startup warmup's bucket set configurable (small, do with PRECOMP-4).**
The trainer's multi-bucket *machinery* still exists end-to-end — the 6-shape `BUCKETS`
table (train/ip_adapter/dataset.py / bucketing.py), `_select_bucket`, the aspect-aware
loader path, and the per-bucket graph warmup loops in train_ip_adapter.py — so distortion-
free multi-aspect training is a config flip away once PRECOMP-4's per-bucket precompute
lands. As of commit 38c1d69 the three startup warmup loops compile **only the pinned
`data.bucket`** (`_warmup_buckets = [_fixed_bucket] if _fixed_bucket else BUCKETS`), because
training is single-bucket-pinned today and warming all 6 inflated the Metal PSO/allocator
surface ~6x at startup (MLX-2 lever-1). That fallback already does the right thing when
`data.bucket` is unset (warms all `BUCKETS`), but the *intended* future state is to drive
the warmup set from the **actual set of buckets training will use** — e.g. a
`training.warmup_buckets` config list or a `--warmup-buckets` CLI flag — so a multi-aspect
run warms exactly its active shapes (not necessarily all 6, not just one). Wire this when
PRECOMP-4 removes the 512² pin; until then the single-bucket restriction is correct.

**PRECOMP-5: Hot precompute cache (defer-rmtree LRU) — cut ~30% of cold↔hot transfers**
(Medium — pipeline throughput) — full plan: [plans/hot-precompute-cache.md](plans/hot-precompute-cache.md).
Each iter currently archives precompute to cold then `rmtree`s it from hot, so recurring
shards get **re-copied** cold→hot next time (hot/cold are different physical devices, so
staging is a real copy, not a symlink). **Measured** on warmup-run1 (13 iters,
flywheel_history.db `selected_shards`): **30.6% of each iter's shards were selected in the
immediately prior iter** (LRU-1 ≈ LRU-3 ≈ unbounded at 30.6/31.0/33.1% — almost all reuse is
consecutive; rising 30%→40% as a campaign matures). So a **1-iteration** cache captures ~92%
of the benefit. Fix: in the publish step, **decouple archive-to-cold (keep) from
rmtree-from-hot (defer)** — keep a content-addressed hot precompute cache, link recurring
shards into the per-iter dir, copy only cache-misses, evict LRU under a hard
`storage.hot_precompute_cache_gb` budget; add the hot cache as the first tier in
`effective_dir`. Footprint ≈ one iter's precompute (~170–210 GB — VAE 0.5 MB/rec × ~4.9k
rec/shard ≈ 2.5 GB/shard, ~4–5 GB/shard all encoders, × ~42 shards; from the cold VAE
manifest — the order already staged transiently today). Transfer-I/O win only (not encode/compute); complementary to PRECOMP-2
(proxy) and `--subsample-per-shard`. Touches live orchestrator publish/cleanup + DataStager
+ `effective_dir` — **land in a non-campaign window.**

---

## Flywheel Management

**FLYWHEEL-1: Long-term campaign management and cross-campaign analysis** (unblocked — PIPELINE-29 done)

Individual campaigns are managed by the orchestrator. This item is the layer above: tracking how quality evolves across campaigns over weeks and months, detecting when a campaign strategy is played out, and deciding when to launch a new campaign vs. continue the current one.

**Campaign lifecycle states:**
- **Active** — training flywheel running, metrics improving.
- **Plateau** — campaign-level cond_gap trend flat for N flywheel iterations (distinct from step-level plateau in the ablation harness, which is per-run). Triggers a recommendation to either change strategy (new ablation config) or warm-start a new campaign.
- **Completed** — operator-marked as done; weights archived; summary written to cold.
- **Superseded** — a later campaign has exceeded this one on all metrics; annotated in weight archive.

**Key capabilities:**

1. **Campaign-level plateau detection** — track rolling mean cond_gap and CLIP-I over the last N flywheel iterations (e.g. N=5). If neither metric has improved by more than `min_delta` for N iterations, emit a WARNING to the doctor and recommend: (a) launch a new ablation run to find a better config, or (b) warm-start a new campaign from a different checkpoint.

2. **Cross-campaign comparison** — powered by `data_explorer compare`. Answers: is campaign B better than campaign A? Are we regressing on CLIP-I while improving cond_gap? The flywheel logs in `metadata/flywheel_logs/` store per-iteration metrics to make this tractable.

3. **Warm-start decision support** — when a plateau is detected, `data_explorer suggest-warmstart` queries the weight archive and ablation history to recommend the highest-leverage starting point for the next campaign. Considers: best historical CLIP-I, which ablation configs are Pareto-optimal, and what training steps have already been covered to avoid redundant work.

4. **Campaign summary generation** — at the end of each campaign (or on demand), generate a structured summary: total steps, peak CLIP-I, cond_gap trajectory, ablation iterations run, shards consumed, wall-clock time. Written to `metadata/flywheel_logs/campaign-{date}.json` and to `weights/flywheel-{date}/summary.json`.

**Implementation:**
- `flywheel.py`: add `_campaign_plateau_check()`, `_write_campaign_summary()`.
- `pipeline_doctor.py`: surface campaign-level plateau as a WARNING with suggested next action.
- `data_explorer.py`: `compare` and `suggest-warmstart` subcommands (see PIPELINE-28).
- `metadata/flywheel_logs/`: structured JSON per iteration, written by `flywheel.py`.

---

## v3.21.0 Quality Loop — Remaining (GPU-gated + wiring)

v3.21.0 shipped the four quality-loop subsystems (experiment tracking, monitoring,
golden-set gate, preference store) with all decision/recording logic tested. These
are the deferred pieces — the GPU-bound execution and the orchestrator wiring that
needs a restart. See `plans/quality-loop-v3.21-migration.md` §7–8.

- **QL-1 (GPU): wire `run_golden_eval`'s 3-arm training + scoring** —
  `evaluate_golden_set.py` currently raises `NotImplementedError`. Reuse the
  `compare_downstream_quality.py` per-arm machinery (precompute golden shards per
  arm: real / proxy+fallback / proxy-forced → train short IP-Adapter → score
  CLIP-I/T + aesthetic + LPIPS + FID). Feed the arm metrics into the already-tested
  `regression_gate()` + `write_results()` + `maybe_disable_proxy()`. Run on an idle
  GPU. ~1–2 days. This is what actually exercises the trust gate end-to-end.
- **QL-2 (one-time, needs data): build the golden-set manifest** — a fixed,
  stratified ~3000-image set (≥30% natural LAION/COYO vs synthetic JourneyDB) via
  `campaign_manager.py`. Set `golden_set.manifest` in the config. Prereq for QL-1.
- **QL-3 (GPU): champion image-generation loop for the data flywheel** — sample
  candidate images from the Champion model, score them, and register the keepers via
  `PreferenceStore.record_synthetic()` so they re-enter scoring. The provenance store
  + `apply_preferences()` blend are done and tested; the generation/scoring loop is
  the GPU-bound piece. ~1–2 days.
- **QL-4 (wiring, needs orchestrator restart): auto-populate the new stores** —
  call `experiments.tracker.record_from_campaign()` from the orchestrator's
  post-campaign path, and the `monitoring.collector` recorders (proxy fallback rate,
  precompute speed, train loss, champion quality, system disk/mem) from the flywheel
  loop, so `--quality-report` / `--monitor` populate automatically. Pure code, but
  takes effect only after an orchestrator restart (don't interrupt a live campaign).
- ~~**QL-5: Slack alert sink**~~ — DONE (2026-06). `monitoring/sinks.py`:
  slack_payload (header + per-alert blocks + text fallback) + dispatch_slack
  (severity filter, env-var webhook resolution, mockable poster) + post_to_slack
  (stdlib urllib, no deps). Wired as `pipeline_doctor.py --monitor --notify`;
  webhook URL is a secret read from $SLACK_WEBHOOK_URL (config holds only the env
  var NAME). 16 tests (test_slack_sink.py); network POST mocked, never hit.

- **QL-6: Bidirectional Slack — trigger pipeline actions from Slack** (Medium —
  ops velocity; outbound QL-5 is the prereq, done). Let the operator drive the
  pipeline from Slack (e.g. `/iris status`, `/iris pause-flywheel`,
  `/iris resume`, `/iris quality-report`, `/iris promote-champion`, `/iris
  golden-eval`). This is a different shape from the outbound webhook — it needs an
  *inbound* path, so it carries real design + security weight:

  - **Transport.** Two options: (a) Slack Slash Commands / Events API → needs a
    public HTTPS endpoint Slack can POST to (ngrok/Cloudflare Tunnel/Tailscale
    Funnel from the Mac, or a tiny relay). (b) Socket Mode → a persistent outbound
    WebSocket (no inbound port; best for a home/NAT box). **Prefer Socket Mode** for
    a single Mac behind NAT — no public endpoint, no tunnel.
  - **Security (the hard part).** Verify Slack request signatures (`X-Slack-Signature`
    + timestamp, HMAC over the raw body with the signing secret) OR Socket Mode app
    token; allow-list the workspace + a specific channel + specific user IDs;
    map each command to an explicit, **whitelisted** action (never exec arbitrary
    strings); destructive actions (pause/promote/disable-proxy/launch) require a
    confirm step or a separate `--armed` token. Secrets via env (`SLACK_APP_TOKEN`,
    `SLACK_BOT_TOKEN`, `SLACK_SIGNING_SECRET`), never in repo/config.
  - **Action surface.** Map to the existing control plane, don't reinvent it:
    `pipeline_ctl.py` (pause/resume/stop flywheel, restart-orchestrator,
    clear-error, force-next-chunk), `pipeline_doctor.py --ai/--monitor/--quality-report`,
    `experiments.tracker promote`. Read-only commands (status/report) are safe to
    enable first; gate state-changing ones behind the confirm/allow-list.
  - **Process model.** A small long-running `monitoring/slack_bot.py` (Socket Mode
    listener) in its own tmux window, supervised like the other pipeline processes;
    structured command log + audit trail of who triggered what; rate-limit.
  - **Dependency note.** Socket Mode needs the `slack_sdk` package in train/.venv
    (acceptable as a train-only dev dep; pin it). The Slash-Command/HTTP route can
    stay stdlib but requires the public endpoint.
  - **Build order.** (1) read-only `/iris status|report` over Socket Mode with
    signature/allow-list; (2) safe controls (pause/resume) behind confirm;
    (3) heavier actions (promote-champion, golden-eval launch) with `--armed`.
  - **Effort:** ~2–3 days incl. auth, allow-list, audit log, tests (command
    parsing + auth + action mapping are unit-testable with the Slack transport
    mocked, same pattern as test_slack_sink.py).

- **QL-7: `iris_slackd` — minimal hardened command daemon** — CODE+TESTS DONE
  (2026-06-05), deploy gated on Slack tokens. `train/scripts/slackd_core.py` (pure
  policy: command table, auth, parse, resolve, rate limit, audit — 39 tests incl. a
  2000-case fuzz proving no message text can synthesise an out-of-table argv) +
  `train/scripts/iris_slackd.py` (thin Socket-Mode shell + `Daemon.handle`, 15
  tests, `--self-test` runs the full pipeline with no SDK/network). All three build
  phases (read-only, armed gate, confirm-gated `stop`) implemented. Remaining to go
  live: `train/.venv/bin/pip install slack_sdk`, create the Slack app (Socket Mode +
  scopes per plan), export the 4 env vars, launch in an `iris-slackd` tmux window.
  See `plans/slack-command-daemon.md`. Original design notes below.

  A deliberately tiny long-running daemon whose only job is: listen to one Slack
  channel, and on a recognised command, invoke a **fixed, whitelisted pipeline
  script** — nothing else. Security and smallness are the features.

  - **Threat model first.** Assume the channel could be seen by more people than
    intended and that Slack messages are attacker-influenceable. Therefore: the
    daemon must be incapable of running anything not in its compiled-in allow-list,
    regardless of message content.
  - **Hard rules (non-negotiable):**
    - **Socket Mode only** — outbound WebSocket, no inbound port, no public
      endpoint, no tunnel. App token + bot token from env (`SLACK_APP_TOKEN`,
      `SLACK_BOT_TOKEN`); never in repo/config.
    - **Single channel allow-list** (`SLACK_CMD_CHANNEL` id) + **user allow-list**
      (`SLACK_CMD_USERS`, comma-sep ids). Messages from anywhere/anyone else are
      ignored and audit-logged.
    - **Fixed command table** mapping a short keyword → an explicit `argv list`
      (e.g. `status → [venv_py, pipeline_status.py]`, `doctor → [venv_py,
      pipeline_doctor.py, --ai]`, `quality → […, --quality-report]`,
      `pause → […, pipeline_ctl.py, pause-flywheel]`, `resume → […, resume-flywheel]`).
      **No user-supplied arguments** in v1 (or a strict per-command validator —
      e.g. chunk must match `^[1-9][0-9]?$`). **Never** `shell=True`, never string
      interpolation of message text into a command.
    - **Default read-only.** State-changing commands (pause/resume/clear-error)
      require `IRIS_SLACKD_ARMED=1` in the daemon's env; otherwise they are
      acknowledged-but-refused. Destructive ones (stop, force-next-chunk) always
      require an explicit `confirm <token>` reply.
    - **No GPU, no direct pipeline mutation** — the daemon only spawns the existing
      scripts (which already own locking/sentinels); it never touches state itself.
    - **Audit everything** to `logs/slackd.jsonl`: ts, user, channel, raw text,
      matched command (or "rejected: reason"), exit code. Rate-limit per user.
    - **Output back to the channel** = the script's stdout tail (truncated), via the
      QL-5 sink. Long output → a file snippet, not a wall of text.
  - **Process model.** `monitoring/slack_bot.py` (or `train/scripts/iris_slackd.py`),
    run in its own `iris-slackd` tmux window under caffeinate, supervised like the
    other pipeline processes; clean shutdown on SIGTERM; reconnect on socket drop.
  - **Why separate from QL-6.** QL-6 is the general design space (could grow
    interactive buttons, broad actions). QL-7 is the *minimum viable, maximally
    hardened* realisation: a closed command set that can only launch known scripts.
    Ship QL-7; let QL-6's richer surface be optional later.
  - **Dependency.** `slack_sdk` (Socket Mode) pinned in train/.venv — the one new
    train-only dep, flagged as a conscious choice.
  - **Tests (transport mocked, no network):** allow-list enforcement (wrong
    channel/user rejected + logged), command-table mapping (keyword → exact argv),
    unknown/again-malformed message rejected, armed-gate on state-changing commands,
    confirm-token flow, audit-log shape. Same mock pattern as test_slack_sink.py.
  - **Build order:** (1) listener + auth/allow-list + `status`/`doctor`/`quality`
    (read-only) + audit log; (2) `pause`/`resume` behind `IRIS_SLACKD_ARMED`;
    (3) `confirm`-gated heavier actions. **Effort:** ~1.5–2 days for (1)+(2).

---

## C Binary / CLI

- ~~**B-001: --vary-from / --vary-strength CLI wiring**~~ — ✅ DONE (verified 2026-07-06, stale entry).
  `main.c` has `--vary-from` (opt 265), `--vary-subtle` (266 → strength 0.2), `--vary-strong`
  (267 → 0.6), `--vary-strength N` (268); `vary_from` is used as an img2img input at main.c:1510.
- **B-002: Z-Image CFG infrastructure** (~1 day) — `iris_sample.c`, `iris.c`, `iris.h` — unblocks Z-Image-Omni-Base
- ~~**B-003: Negative prompt for distilled Flux**~~ — ✅ DONE (verified 2026-07-06, stale entry).
  `iris.c` (~line 902): distilled models with a non-empty `negative_prompt` and no explicit guidance
  nudge `guidance = 1.5f` and enable CFG (`use_cfg = !is_distilled || negative_prompt`), encoding the
  negative as the uncond pass. CLI `-N/--negative`, daemon `negative_prompt` key, and web request key
  all plumbed. (Web UI Feature 1 = exposing this in the UI — soft-blocked by P0 UI-STYLE-UX.)

---

## Web UI Features

- 🔴 **P0 — UI-STYLE-UX: make style transfer the obvious, working default; cut the confusing options
  (2026-07-06, user report).** User feedback: "too difficult as a user to get style transfer — too
  many confusing options to choose from and tweak — and I did not get any style transfer from the
  default options." ROOT CAUSE (confirmed 2026-07-06): each reference slot has a per-slot
  **"Comp / Style" dropdown** (`web/static/index.html`, `<option value="composition">` listed FIRST →
  default). Band-control (the shipped training-free style rail; job.sref_shf/slf, engaged only when a
  slot's mode == "style") therefore NEVER fires on the default path — an uploaded reference is treated
  as **composition** (literal img2img copy), not style. So the default UX silently gives composition
  copying, not style transfer, and the fix (per-slot dropdown) is buried + jargon. WHAT TO DO (design
  is the user's call — DO NOT just add more controls; this P0 is about REMOVING confusion):
  (a) make the common case one obvious action — e.g. an upload prompts "Match this composition" vs
  "Transfer this style", or default an uploaded reference to STYLE (band-control) with a single clear
  toggle; (b) hide the advanced knobs (shf/slf/strength, per-slot Comp/Style, ip-schedule) behind an
  "advanced" disclosure; (c) verify end-to-end that the default path produces VISIBLE style transfer
  (the CLI band-control gate passes — the gap is UI wiring/defaults, not the mechanism). NOTE: this
  supersedes/soft-blocks Feature 1 (adding a negative-prompt field would add clutter) until the UI is
  simplified. Cross-ref: CHANGELOG v5.0.0/v5.1.0 (band-control + style codes), SREF-ROPE-PHASE1/2.
  ✅ CORE FIX SHIPPED (2026-07-06): uploaded references now default to STYLE (band-control) so the
  default path actually transfers style; the 4 per-slot "Comp/Style" dropdowns were replaced by ONE
  global toggle ("Use as: Style — adopt the look / Composition — keep the layout", Style default,
  shown only when a reference is present); the per-slot strength slider is now hidden in Style mode
  (full-strength in-context; only shown for Composition). Verified end-to-end against the running
  server: default (Style) routes through band-control and is distinct from Composition (corr 0.406,
  output not a literal reference copy); app.js syntax-clean; no leftover dropdowns.
  REMAINING (optional polish, not blocking): real in-browser user check that the default now "feels"
  like style transfer; consider an even simpler first-run affordance; the shf/slf/strength/ip-schedule
  knobs remain server-env only (not surfaced), so nothing to hide there.

- 🔴 **SREF-STYLE-CEILING (2026-07-06) — band-control style transfer is STYLE-DEPENDENT: it works for
  GRAPHIC / high-contrast references but FAILS for PAINTERLY / subtle ones, on BOTH distilled AND
  4B-base. A mechanism ceiling, not a tuning bug. Corrects the over-optimistic v5.0.0/v5.1.0 framing.**
  User report: "composition leaks in both cases / no style transfer from defaults." Reproduced visually
  this session (a baroque portrait ref, prompt "a dog running in a grassy field", seed 7, 512px):
    • WOODCUT (graphic) ref + band-control (shf0.0/slf1.5) → output dog rendered in clean woodcut
      linework, composition suppressed. STYLE TRANSFERS. ✓
    • BAROQUE (painterly) ref → NO setting transfers the painterly rendering: shf0.0 kills composition
      but yields a GENERIC PHOTO (no style); shf0.5/0.7 brings back a specific composition element (the
      reference's ruff collar leaks onto the dog) but STILL no painterly style; slf up to 2.0,
      --sref-strength up to 3.0, and the 4B-BASE model (24-step CFG) ALL still produce a plain photo dog.
      Old patch-shuffle ALSO fails on painterly.
  ROOT: in-context conditioning transfers high-contrast/graphic texture (survives the distilled
  first-step structure commit + the RoPE-band scaling) but not soft painterly rendering / subtle
  palette. Consistent with the MODEST CSD gate numbers (style_adh ~0.35 = weak style); the gate's
  amenable refs (woodcut/flat) masked the painterly failure. METRIC CAVEAT: pixel copy_corr AND CSD both
  under-measured this — reinforces SREF-METRIC-1 (need a real style-vs-composition-leak metric).
  IMPLICATION: v5.x "training-free style transfer" is real ONLY for bold/graphic reference styles.
  Painterly / photographic-subtlety style transfer remains UNSOLVED on this stack via in-context
  methods → needs the deferred learned-encoder / base-adapter architectural work (see
  SREF-CHAMPION-COLLAPSE). ACTIONS: (a) set UI expectations ("works best with bold/graphic references"),
  do not over-promise; (b) for painterly/semi-real, the learned-encoder project is the only path —
  PLANNED in `plans/sref-learned-encoder-project.md` (USO-style in-sequence style tokens, base-first
  training + transfer, gated on a real style metric); (c) consider
  gating/labelling references by "graphic-ness" so users know what will work.
  USER CASE REPRODUCED (2026-07-06, anime glossy food-illustration ref, 720×1280): confirms a THIRD
  regime between graphic-works and painterly-fails — SEMI-REALISTIC illustration transfers SUBTLY: the
  glossy/saturated/vibrant *rendering quality* comes through (band-control robot is visibly glossier +
  blue-accented vs the matte no-reference txt2img robot, corr 0.71) but NOT the flat cel-shaded/outlined
  aesthetic; raising slf 1.5→2.5 and shf 0→0.3 barely strengthens it. TWO extra findings: (1) the user's
  "composition leaks" was the OLD composition-mode default copying the plate — the P0 style-default fix
  addresses it; (2) ASPECT RATIO MATTERS — rendering at the reference's aspect (portrait) noticeably
  strengthened the transfer vs a square render (the web already auto-sets dims from the ref, so this is
  mostly handled). Net: for semi-realistic refs, expectation-setting is the fix, not a stronger default.
- [ ] **18. Batch prompt generation** — Submit a list of different prompts to generate in sequence.
- [ ] **20. Per-job timeout** — Prevent hung generations from blocking the queue forever.

**Advanced (prerequisite: extract `fetchImageAsBase64()` — duplicated across 4 files)**
- **Feature 3: Enhanced Vary-from-History** (~2–3h) — fastest win, no C backend changes needed
- **Feature 2: Per-Slot Reference Strength + Style Reference Mode** (~3h UI / ~8h full C with backend)
- **Feature 4: Outpaint UI** (~5–7h)
- **Feature 1: Negative Prompt** (~3–4h server+UI + 4h C backend) — blocked on B-003

---

---

## Metal / GPU Performance

- **BL-004: simdgroup_matrix for Custom GEMM Tiles** — M3+ only
- **BL-005: Native bfloat MSL Type** — M3+ only
- **BL-009: GPU-fused single-block LoRA — restore block fusion under an active adapter (P3, speed).**
  Context: the single-block-LoRA-dropped-on-Metal bug (review 2026-07-30, H1) was fixed *conservatively*
  by routing single blocks to the C-orchestrated `single_block_forward` whenever a single-block adapter
  is active (mirrors how double blocks already behaved under LoRA). That path still runs every GEMM +
  attention on the GPU (`iris_linear`/`iris_matmul` → `iris_metal_sgemm`; `mha_forward` → SDPA) — it is
  NOT a CPU fallback — but it loses the fully-fused `single_block_forward_gpu` kernel's on-GPU residency,
  so a LoRA/SREF generation pays extra CPU↔GPU dispatch+readback across the ~20–24 single blocks each
  step vs a no-LoRA run. Correctness is unaffected; only LoRA/SREF latency (the app's Style Library /
  Learned Style path). **Optimization:** apply the LoRA delta `A·Bᵀ·scale` directly inside the fused GPU
  single-block kernel (an extra GPU matmul or a fused epilogue) so LoRA generation keeps the fused path.
  Caveat: won't be bit-exact vs the CPU reference (f32 delta vs bf16 GEMM rounding) — needs its own
  tolerance-based parity check, not the exact `test_lora.c` guard. Only worth building if LoRA/SREF
  inference latency becomes a product concern. Note: the old hard-disabled `if (0 && …)` bf16 single-block
  inject block was REMOVED in the H1 cleanup, so **HW-M5-4's "re-enable the bf16 GPU inject" premise is
  now "implement it fresh"** — update that item's file:line when this lands. Cross-refs: HW-M5-4, BL-004,
  BL-005, review-2026-07-30 H1.

### Metal Kernel Audit (Grok, 2026-05) — Triaged / mostly SUPERSEDED (2026-07-20)

External metal-kernel audit package (3 docs + 2 HTML) archived to `reviews/`
(`metal_kernel_audit_{summary,comprehensive}.md`, `metal_optimization_backlog.md`).
Re-checked against the perf work landed *after* the May audit — most of it is stale:

- **B-METAL-01 (CPU softmax fallback) / B-METAL-02 (make `attention_fused` dominant)
  — SUPERSEDED.** The f32 attention path now routes through MPSGraph native SDPA (Flash
  Attention) as the PRIMARY path (BL-001, done), with the custom kernel only a fallback;
  BF16 already used Apple `scaledDotProductAttentionWithQueryTensor`. The audit's premise
  (hand kernel is the hot path, CPU softmax leaks) no longer holds.
- **B-METAL-03 (float4 vectorize the custom-kernel inner loops) — LOW VALUE now.** The
  custom kernel is a fallback; BL-002 already replaced its barriers with simd_sum/simd_max.
- **B-METAL-04 (simdgroup_matrix QK^T / scores@V) — == BL-004, already tracked, M3+ only.**
- **B-METAL-05 (block super-kernel fusion) — speculative; not scheduled.**
- The audit's "IP-Adapter C path missing" framing is STALE (fused inject shipped 2026-06-29
  B2; joint adapter ported to C 2026-07-13). The one concrete keeper — a metal
  parity/perf regression harness — is committed as `debug/metal_regression_suite.sh`.

---

## Test Gaps

- ~~**TB-001: Qwen3 Tokenizer Correctness** (P1)~~ — DONE. `debug/test_tokenizer.c`
  (58 tests) runs in `make test`.
- ~~**TB-010: Flash Attention vs Naive Attention Parity** (P2)~~ — DONE.
  `debug/test_kernels.c` flash-vs-naive parity runs in `make test`.
- **TB-002: Base Model Regression 4B-base** (P1) — requires model
- **TB-004: VAE Encode/Decode Roundtrip** (P2) — requires model
- **TB-005: img2img Strength Sweep** (P2) — requires model
- **TB-006: CFG Guidance Value Validation** (P2) — requires model
- **TB-003: Z-Image Regression** (P2) — requires model
- **TB-007: Step Preview (--show-steps) Output** (P3) — requires model
- **TB-008: Backend Parity MPS vs generic** (P3) — requires model
- **TB-011: LoRA Integration load+apply in transformer** (P3) — requires model
- **TB-009: 9B Model Regression** (P3) — requires model

---

## Known Bugs — Code Review 2026-05-15

32 of 33 bugs fixed — see COMPLETED_BACKLOG.md. One remains open:

**BUG-M-004: Batch mode atomicity broken in iris_gpu_attention_mps_bf16** — latent (harmless at current call sites)
`iris_metal.m` (~lines 3149–3151). Between Phase 2 (CPU softmax) and Phase 3 (scores @ V), the function unconditionally commits and resets `g_tensor_cmd` regardless of `g_tensor_batch_mode`. In batch mode this splits the attention's Phase 1 and Phase 3 across separate command buffers.

**Investigation findings (2026-05-15):** The only call site is `iris_gpu_attention_bf16` (line 3293), which calls `iris_gpu_sync()` at line 3270 *before* entering, flushing any prior batch work. Phase 1 and Phase 3 being in separate command buffers is an inherent constraint of the CPU softmax — it cannot be avoided without a GPU softmax kernel. Ordering is fully maintained. Current risk: zero. Proper fix: implement GPU softmax to eliminate the CPU readback; medium effort, not needed until a future model requires true bf16 batch-mode attention across this path.

---

## Known Bugs — Pipeline Code Review 2026-05-15 (second pass)

All 14 bugs fixed in commit `76564f8` (2026-05-15). See COMPLETED_BACKLOG.md.

---

## Known Bugs — Inference/Training Cross-Reference Review 2026-05-15

All 9 bugs fixed: INFER-C-001 in commit `76564f8`; INFER-H-001 and INFER-M-001 in commit `ffecfcc`; INFER-M-002 and INFER-L-001–005 in commit `76564f8`. See COMPLETED_BACKLOG.md.

---

## C-Engine Static Review (Grok 4.3, 2026-06) — Triaged

> NOTE (2026-07-20): the raw Grok audit reports referenced in this and the following
> three "— Triaged" sections are archived under `reviews/` (git-ignored). Findings were
> already extracted here; the reports are kept locally as reference only.

External static review of the C inference engine (`grok_bug_report.md`, archived in `reviews/`).
Verified accurate on spot-check (C-01, H-01, H-04, L-01, M-08 confirmed against
source). Triaged below — severities re-calibrated from the original report. Scope
is the C binary only; none of this affects the Python training/flywheel pipeline.

### Do soon (cheap, low-risk)

- ~~**GROK-1 (was H-01): Remove dead `rope_freqs`**~~ — DONE (2026-06). Removed the
  `rope_freqs` field, the 1D `compute_rope_freqs` helper, all 3 alloc/compute load
  sites + the free in `iris_transformer_flux.c` (verified never read in any forward).
  `iris_qwen3.c`'s own `compute_rope_freqs` is unrelated and untouched. Clean BLAS build,
  `make test-unit` green.
- ~~**GROK-2 (was M-08): Remove dead `iris_vae_load(FILE*)`**~~ — DONE (2026-06). Removed
  the legacy `.bin` VAE loader + its exclusive helpers (`read_uint32`, `read_floats`,
  `load_resblock`, `load_attnblock`) from `iris_vae.c` and the decl from `iris.c`
  (zero call sites). `free_resblock`/`free_attnblock` and the `_sf` safetensors loaders
  kept. Clean build.
- ~~**GROK-3 (was L-01): Dedup AGENT.md / CLAUDE.md**~~ — NOT A BUG (verified 2026-06).
  Already deduplicated: `AGENT.md` is the real file (git mode 100644) and `CLAUDE.md`
  is already a symlink to it (git mode 120000). Both the Grok report and a shallow
  `diff -q` were fooled — the files "match" precisely *because* one is a symlink to
  the other. No action; drift is already impossible.
- **GROK-4 (was H-04): Z-Image pad/pos-id cross-path golden test** (~2–3h) —
  `iris_transformer_zimage.c`. NOT a bug fix — a regression guard for the documented
  "CPU/GPU position-id mismatch under padded captions" pitfall. Add a unit test
  asserting byte-identical pos-ids for real tokens between the GPU-table and CPU-unified
  paths, exercised with `cap_len % 32 != 0`. Highest-value item here: cheap insurance
  on a pitfall that has already bitten the project.

### Roadmap decision (not a bug)

- **GROK-5 (was C-01/C-02): `--sref` / C IP-Adapter is declared but not implemented.**
  `train/export/iris_ip_adapter.h` provides the full public surface; no `.c` implements
  it; `main.c:1273-1275` cleanly rejects `--sref` ("not yet implemented, planned for
  v2.6"). This is a **deferred feature with a clean guard**, not a correctness defect
  (re-severed from the report's CRITICAL). Decision required: either (a) implement the
  C IP-Adapter (loader + Perceiver + get_kv + inject, wired into double/single block
  forwards, parity-checked vs `test_ip_adapter_inference.py`) and make it the P0, or
  (b) remove the public surface and stop advertising `--sref` until ready. Do not leave
  it half-advertised.

### Fix opportunistically (when already in that file — do not sweep)

- **GROK-6 (was H-02): Unify ad-hoc JSON extraction.** Multiple independent strstr/atoi
  config parsers (`iris.c`, `iris_transformer_flux.c`, `iris_safetensors.c`, `main.c`,
  tokenizer). Extract one minimal `iris_json.c` helper (no new deps) + adversarial tests.
  Real debt, but a dedicated refactor is risky for no functional gain.
- **GROK-7 (was H-03): Collapse duplicated block load/forward/free variants** (f32 / bf16
  / mmap / GPU / debug). Largest structural debt; touches every historical pitfall
  (timestep cache, RoPE indexing, sgemm B-cache). High-risk multi-week refactor — only
  undertake when a concrete feature (e.g. full IP-Adapter) forces touching these paths.
- **GROK-8 (was H-05): Per-ctx progress callbacks.** Global callback pointers in
  `iris_kernels.c` are not reentrant. Move to `iris_ctx` or document "not thread-safe,
  set from main thread only" at minimum.
- ~~**GROK-9 (M-02/M-03/M-06): Error-reporting + cleanup consistency.**~~ — CLOSED:
  WON'T DO (2026-06). Standardise on `set_error`, add `d[N-1]='\0'` after
  `strncpy(,,N-1)`, tighten OOM cleanup. Closed as a defensive sweep across the
  inference-path C files with no functional gain (the `strncpy` buffers are already
  `calloc`-zeroed, so the null-term is a no-op today). Same risk/reward as GROK-10.
  Reconsider only if a real defect surfaces or these files are being refactored anyway.
- ~~**GROK-10 (M-01/M-04/L-07): Architectural-invariant asserts + magic-number cleanup.**~~
  — CLOSED: WON'T DO (2026-06, user-skipped). Validate derived dims after config parse
  (`hidden == heads*head_dim`, `head_dim == 4*axis_dim`); add `static_assert`. Closed
  because it requires touching 4 config-parse sites across the critical 5.3k-LOC
  inference file for marginal benefit — the dims come from the model's own (correct)
  config; a speculative sweep the backlog itself flags "do not sweep". Reconsider only
  when already editing those load paths for a real feature.

### Build / nits (observation)

- **GROK-11 (M-07/M-10): Monolithic units + always-clean backend builds.** `iris_transformer_flux.c`
  (~5.3k LOC), `iris_metal.m` (~7k LOC); per-backend build dirs and only re-`xxd` shaders
  on change would speed dev. Cosmetic — defer.

Original report retained as `grok_bug_report.md` (untracked) for full detail.

---

## Proxy-VAE Design vs C Inference Review (Grok, 2026-06) — Triaged

External static review of the proxy-VAE design (`grok_proxy_vae_analysis.md`,
untracked) cross-referenced against the C `iris_vae`. File:line refs verified
accurate; severities re-calibrated. Net: the report inflated two "Criticals" that
block nothing today (the flywheel runs the real VAE; the proxy is pending
validation and not enabled), and most findings duplicate existing items. The one
genuine, non-duplicate insight is the **three-distribution framing**: teacher
(Python diffusers/mflux, makes precompute latents) → proxy (MLX student,
approximates teacher encoder) → **C `iris_vae` decoder** (inference). The proxy is
trained to match the teacher *encoder*, but generated latents are decoded by the C
decoder, so the binding invariant is proxy-distribution ↔ C-decoder. The design's
decoded-LPIPS loss already binds proxy outputs to decode correctly under the *real
decoder*; the residual gap is teacher-vs-C parity, which is **pre-existing and
independent of the proxy** (precompute already uses the Python teacher today).

Duplicates (no new action): config-parser brittleness = GROK-6; dead `.bin` VAE
loader = GROK-2; "pending validation" already stated under PRECOMP-2; GPU-resident
encode / scalar patch loops are perf-backlog.

**Full triage of all 7 grok reports: `plans/grok-review-triage.md`** (de-duplicated +
prioritized with disposition). Net: parser brittleness (G-5), VAE↔teacher parity (G-6),
generic-build correctness (G-9), and dead code (G-10) are RESOLVED; live priorities are
**G-1 (C-side IP-Adapter — the endgame: trained adapters can't yet run in the `iris`
binary)**, **G-2 (MLOps state-machine tests — in progress via pure-core extraction)**, and
GPU/idle-gated **G-3 (hardcoded paths) / G-4 (B-METAL-01 CPU softmax fallback)**.

- **GROK-VAE-1: C VAE inference-ground-truth guard** — DONE (2026-06). Added
  `debug/test_vae.c` (wired into `make test-unit`, CPU-only, flywheel-safe): builds
  small architecturally-real Flux/Z VAEs with seeded synthetic weights and exercises
  the real CPU encode/decode to assert shapes (16× compression), finiteness,
  bit-determinism, z_channels→latent_channels wiring (32→128, 16→64), and that the
  latent normalization **branch** is config-selected and uses the exact
  `(x-shift)*scaling` vs `(x-mean)/sqrt(var+eps)` form (the path the brittle
  `vae/config.json` parser feeds — guards C-2/GROK-6 at the integration level). Also
  added a "ground truth" contract note to the `iris_vae.c` header + AGENT.md/CLAUDE.md
  Flux-VAE section. This is the foundation a future teacher-golden comparison plugs
  into (a true cross-impl golden needs the GPU + Python teacher → flywheel-gated).
- **GROK-VAE-2 (Low): Z-Image proxy gap** — the proxy design is Flux-32ch only;
  Z-Image VAE (16ch + explicit scale/shift, no BN/quant) gets no precompute relief
  despite full C support. Note for if/when a Z-Image training campaign is run; not a
  defect. Would need a 16ch student preset + Z teacher path.

Original report retained as `grok_proxy_vae_analysis.md` (untracked) for full detail.

---

## train/ Static Review (Grok 4.3, 2026-06) — Triaged

External static review of the Python training pipeline (`grok_train_bug_report.md`,
untracked). Verified accurate on spot-check (shell=True at doctor:3176, the
`/Users/fredrikhult/ultrahot` literal, path-asserting tests, 3k+ LOC files all
confirmed). Character note: found **zero CRITICAL and zero concrete correctness
defects** — entirely architectural/portability debt, most of it a known and
accepted single-host constraint. It also **missed the live precompute→train
fresh-write race** that failed flywheel iter 10, illustrating that static review
surfaces debt but not the operational bugs that actually bite. Severities below
re-calibrated to *urgency* (the report's HIGHs are mostly "hurts at multi-machine
/ handoff time", not "wrong results now").

### Done (cheap wins — actioned 2026-06)

- **GROK-T-1 (H-T01): ultrahot username literal** — DONE. `ULTRAHOT_ROOT` was
  `/Users/fredrikhult/ultrahot`; now `~/ultrahot` via `Path.home()` + a
  `PIPELINE_ULTRAHOT_ROOT` env override matching the `DATA_ROOT` convention. Same
  value on this machine, no behaviour change.
- **GROK-T-2 (H-T01): path-asserting storage tests** — DONE.
  `test_pipeline_storage.py::test_importable` asserted exact `/Volumes/16TBCold`
  literals (machine-coupled, tested a constant against its own literal). Rewritten
  to assert the real invariant — importable absolute `Path`s — leaving the
  derivation check to `test_derived_from_cold_root`.

### Open (real but deferred)

- ~~**GROK-T-3 (H-T05): `shell=True` in doctor --fix**~~ — AUDITED + HARDENED (2026-06).
  `run_fix_mode` is human-gated: each fix is printed verbatim and run only after a
  per-issue `y` confirm (not auto-exec); `shell=True` is deliberate (fix strings use
  pipes/globs/`&&`). Audited every interpolated value in every `fix=` string: all come
  from the pipeline's controlled namespace — path constants (DATA_ROOT/SENTINEL_DIR/
  PRECOMP_DIR/CKPT_DIR/LOG_DIR/…), the fixed qwen3/vae/siglip encoder set, numeric
  chunk/step (checkpoint ones `:07d`-forced to int), and compiled-in tmux window-name
  constants. **No external/untrusted data** (tar member names, file contents,
  heartbeat/log JSON, env, argv) reaches any fix string. The one operator-typed value
  — the campaign `name` in `_check_campaigns` — is now `shlex.quote()`d (defense-in-depth:
  robust to names with spaces/metachars; it was already human-gated, never
  attacker-reachable on a single-user box). 70 doctor tests green.
- **GROK-T-4 (H-T01 remainder): centralise storage roots.** ~90 hardcoded
  `/Volumes`/`/Users` literals across leaf scripts + argparse defaults + docstrings.
  Proposal: one `pipeline_lib.get_storage(cfg)` returning a `StoragePaths` object;
  ban new literals. P0 *when* moving to a second machine / V3; not before.
- **GROK-T-5 (H-T03): mflux dependency + sys.path hacks.** Unvendored mflux +
  `sys.path.insert` in every entrypoint. Add `requirements-train.txt` pinning
  mflux/mlx/etc., make `train/` a real package (`python -m train.scripts.x`),
  drop the path shims. Reproducibility win; do at V3 packaging time.
- **GROK-T-6 (H-T04): LiveEncoderManager.** The 32 GB attach/detach + mflux
  private-weight manipulation is the most regression-prone code. Encapsulate in a
  `with live_encoders_for_batch():` context manager + a mem-profile assert + a
  hard error (not silent fallback) when precompute coverage is 100%. Worth doing
  before adding any new live-encoder feature.
- **GROK-T-7 (M-T03): config schema + load_and_validate.** DONE (2026-06).
  `train/scripts/config_schema.py`: `validate_config(cfg)` flags unknown top-level
  sections (typos), non-mapping sections (ERROR), unknown `flywheel_health` knobs
  (doctor-consumed, silent-default class), and missing flywheel required keys; a
  standalone CLI and a `pipeline_ctl.py validate-config <path>` subcommand (exits
  non-zero on ERROR). Conservative — all 16 real configs pass clean. 11 tests in
  `test_config_schema.py`. Deferred: full per-key dataclass typing + wiring into
  `load_config` (kept non-invasive — validate is opt-in, not silent on every load).
- **GROK-T-8 (M-T05): state-machine / resume scenario tests.** Synthetic
  sentinels+heartbeats asserting `derive_chunk_state` + next-action across phantom,
  jetsam, last-chunk, manual-rm, hard-ex-mixing. Catches resume regressions before
  days of compute are wasted. (Note: would NOT have caught iter 10's fs race.)

### Fix opportunistically (when already in that file)

- **GROK-T-9 (H-T02): split god modules** (orchestrator 3480, doctor 3340,
  train_ip_adapter 3071, ablation 3011 LOC). Legitimate but high-risk multi-week
  refactor of the live state machine — only when V3/containerization forces it.
- **GROK-T-10 (M-T01/M-T02): sentinel TOCTOU + resume special-cases.** Optional
  sqlite step-state alongside sentinels; central pure-function "mixing policy" +
  "step-range math" that's unit-tested. Pay down when next touching transitions.
- **GROK-T-11 (M-T04): tar hardening.** Per-member size caps, optional sha256 at
  build, per-shard `.error` sentinels. Pedantic — current data is trusted JDB/LAION.
- **GROK-T-12 (M-T06): config-only script entrypoints.** Collapse duplicate argparse
  (paths that already live in yaml) so standalone and orchestrated paths can't drift.
- **GROK-T-13 (L-T03): log rotation/retention** — or at least a doctor check for
  total `logs/` size (fixed-name jsonl/.log can accumulate across runs).

Original report retained as `grok_train_bug_report.md` (untracked) for full detail.

---

## Testing-Suite Review (Grok 4.3, 2026-06) — Triaged

External review of the test suite's completeness/accuracy (`grok_testing_bug_report.md`,
untracked). Most numerically precise of the three Grok reports — verified exactly 209
Python test functions, and confirmed **zero** dedicated test coverage for orchestrator,
pipeline_doctor, ablation_harness, flywheel_lib, precompute_all, mine_hard_examples,
cache_manager, campaign_manager (all 8 grepped to 0 test files). Central finding is
correct and important: 209 tests but all concentrated in the ML core (loss/ema/model/
dataset/export) + storage primitives + the parity guard; the autonomous MLOps "brain"
is unit-test-free.

**One verified inaccuracy in the report:** it credits the storage tests (lines 26, 240)
as "improved post-audit to avoid machine-specific literals" — but `test_importable` was
still asserting exact `/Volumes/16TBCold` literals until commit `dff58d8` *this session*
(GROK-T-2). The report conflated the one abstract test with the suite as a whole.

**Relevance:** this is the report most validated by live failures — the flywheel failed
iters 1–10, iter 10 on the precompute→train shard-cache handoff. The report's P0
(synthetic handoff/state tests) is exactly the missing class.

### Done (targeted test for the live failure — actioned 2026-06)

- **GROK-TEST-1: shard-cache-filter contract test** — DONE. Extracted the
  precompute→train filter (was closures inside `train()`) into module-level
  `shard_internal_prefix` / `shard_has_cache` / `filter_shards_with_cache` in
  `train_ip_adapter.py`, and added `train/tests/test_shard_cache_filter.py` (12 tests).
  Covers the exact iter-10 failure modes: empty cache → 0 shards, naming mismatch
  (tar stem ≠ npz key prefix) → excluded, partial precompute (missing `_0049`) →
  excluded, qwen3/vae-only → excluded, ordering preserved. Pure (fake tar paths +
  empty npz touch-files; no mflux/Metal/data). Note: guards the *contract*, not the
  fs-visibility race that triggered iter 10 — that needs a different guard (GROK-TEST-2).

### Open (real but substantial — net-new test suites)

- **GROK-TEST-2 (P0): synthetic orchestrator state-machine harness.** PARTIALLY DONE
  (2026-06). `train/tests/test_orchestrator_state.py` added (14 tests): `derive_chunk_state`
  across all steps + error-precedence + last-done-wins + chunk independence, the
  CHUNK_STEPS/_STEP_TO_STATE contract, and ResourceManager non-GPU token semantics, plus _resolve_proxy_vae_args
  (config→precompute flags + per-campaign overrides). Hermetic via
  `pipeline_lib.SENTINEL_DIR` monkeypatch (flywheel-safe — separate process).
  EXTENDED (2026-06): jetsam-vs-code-error retry/backoff now covered — extracted the
  pure `_retry_policy(reason, restarts)` decision out of the crash-handler method and
  tested it (jetsam → JETSAM_MAX_RETRIES with backoff, code error → 1 retry no delay,
  stop-at-limit), plus the already-pure crash functions `_parse_exit_code_from_msg`,
  `_diagnose_crash` (non-137 code_error / 137 jetsam-confirmed / 137 jetsam-assumed via
  monkeypatched system-log query), `_parse_last_mem_from_log`. 11 new tests.
  COMPLETED (2026-06): the remaining decision logic is now extracted + tested —
  `_ready_gate` (chunk-transition: wait_prev_train/wait_gpu/wait_stage/proceed_no_stage/
  proceed, incl. chunk-1-never-staged), `_should_attempt_stage` (predecessor-promoted
  staging gate), `_restart_plan` in pipeline_ctl (resume-from-N: reset exactly N..total,
  never a chunk before N), `_load_open_dispatch_ids` (dispatch seeding: open vs
  resolved-after-open), and the doctor phantom hard-ex last-chunk branch (INFO on last
  chunk vs CRITICAL mid-pipeline). All behaviour-preserving extractions; tests in
  test_orchestrator_state.py / test_pipeline_ctl.py / test_pipeline_doctor.py.
  Note: none of these would catch iter 10's fs race — that's the separate shard-cache
  guard (GROK-TEST-1 / GROK-VAE-1).
- **GROK-TEST-3 (P0): pipeline_doctor black-box tests.** PARTIALLY DONE (2026-06).
  `train/tests/test_pipeline_doctor.py` (12 tests): `_check_proxy_vae` (all 8 branches —
  silent/misconfigured/missing-ckpt/no-eval/failed-gates/healthy/unreadable) and
  `_check_error_sentinels` (none/critical/done-skip/multi-chunk), `_check_stale_logs`
  (log-older-than-sentinel WARNING / fresh-silent / no-sentinel-silent), and
  `_check_phantom_completions` (promoted.done-but-no-shards CRITICAL / shards-in-range-ok
  / not-promoted-silent — the headline phantom, same class as iter-10). Hermetic via
  DATA_ROOT + SENTINEL_DIR (both doctor and pipeline_lib bindings) + LOG_DIR/SHARDS_DIR/
  PRECOMP_DIR monkeypatch + _issues reset. EXTENDED (2026-06): training-integrity
  (NaN loss, non-zero exit, short-log-but-done, resume-past-end, clean-silent) and
  precompute-forensics (orphaned .tmp.npz, double-extension crash artifact, low
  coverage, clean-silent) detectors now covered. Also added tests for the newer
  flywheel observability checks — liveness reconciliation, failure-loop +
  fingerprinting, cache-coverage confirmation, trainer-anomaly (chunkless), campaign
  ETA, progress-stall, logs-disk. STILL OPEN: none of note for the doctor.
- **GROK-TEST-4 (P1): model-quality regression automation.** Make `test_quality_features`
  emit machine-readable goldens (final cond/null gap, ip_scale stats, cross/self gap) on a
  fixed small set; gate in a `make test-quality`. No golden quality regression exists today.
- **GROK-TEST-5 (P2): perf/memory assertions.** Assert on telemetry the mini-loop already
  computes (step time, `mx.get_peak_memory`, grad-norm, ema-drift); micro-bench for the
  online-encoder overhead. Critical for the 32 GB tightrope; currently absent.
- **GROK-TEST-6 (P3): precompute_all + cache_manager units.** cache_manager PART DONE
  (2026-06): `train/tests/test_cache_manager.py` (13 tests) covers version_hash
  (determinism, key-order stability, config/git-sha sensitivity, format) and
  encoder_config_subset (vae flux_model dir-stripping, image_size version separation,
  qwen3/siglip/unknown subsets, end-to-end vae version stability). precompute_all
  building blocks DONE too (2026-06): `train/tests/test_precompute_all.py` (14 tests)
  — _scan_existing (the "already done" set), iter_shard (jpg+txt pairing, png,
  corrupt-tar-warns, whitespace strip), _save_npz_atomic (no torn writes) + the
  done→skip round-trip. STILL OPEN: full _process_shard_inner orchestration (needs
  model stubs).
- **GROK-TEST-7 (P4): property-based + flywheel/ablation DB roundtrip.** DB PART DONE
  (2026-06): `train/tests/test_flywheel_db.py` (10 tests) — insert/get roundtrip,
  selected_shards JSON, ordering, update (status/metrics/failed exit codes), get_best
  cond_gap selection (null-excluded, highest-wins), checkpoint_log + mark_best, campaign
  isolation. Uses an explicit tempdir db_path (never touches live flywheel_history.db).
  ABLATION DB DONE (2026-06): `train/tests/test_ablation_db.py` (15 tests) — AblationDB
  insert/get/update roundtrip, params_hash determinism + is_duplicate (per-run scoped),
  get_best (score-desc, unscored-excluded, limit), scored_only filter, run isolation,
  3-objective Pareto front (dominated excluded / non-dominated both kept), and
  post_train_validation roundtrip (weight_errors JSON, verdict defaults). Tempdir db_path,
  never touches live ablation_history.db. STILL OPEN: property-based (hypothesis) for
  bucket selection / schedule / quant / hard-ex t-sampling — deferred (would add the
  `hypothesis` dev dependency); ablation warmstart roundtrip.

Also new (2026-06): `train/tests/test_dataset_bucketing.py` (17 tests) — the
precompute↔train resolution contract (`_select_bucket` aspect math, `_load_vae_latent`
shape-rejection, and the headline "square-512 latent matches ONLY the (512,512) bucket"
that mandates the `data.bucket` pin). Guards the 4th warmup-run2 blocker; see PRECOMP-4.

### Maintainability (nice-to-have)

- **GROK-TEST-8: pytest markers (DONE 2026-06-17).** Markers `slow` / `requires_shards` /
  `requires_mps` / `quality` registered in `train/tests/conftest.py` (via `pytest_configure`,
  each with a description). `make test-ci` runs the fast, dependency-free subset:
  `pytest tests/ -x -q -m "not slow and not requires_shards and not requires_mps and not quality"`
  (582 pass / 163 deselected). The deselected set = the MLX/Metal-GPU modules
  (`test_smoke`, `test_dataset`, `test_ema`, `test_export`, `test_loss`, `test_model`) marked
  `requires_mps`, plus `test_training_loop` (`requires_mps` + `slow`). No current test does real
  shard I/O, so `requires_shards` has no users yet; `quality` is reserved for golden-set GPU eval
  (the existing `test_quality_gate` / `test_golden_gate` are pure logic and stay in CI).
  Untested-modules gate: `train/scripts/check_untested_modules.py` (also `make check-untested`)
  flags source modules >50 lines in `train/scripts/`, `train/ip_adapter/`, `debug/` that have no
  `test_<stem>.py` and are not referenced from any test; warn-only by default (48 current gaps),
  `--strict` to fail CI, `--min-lines` to tune the threshold.

Original report retained as `grok_testing_bug_report.md` (untracked) for full detail.

**CKPT-PRUNE-1 (DONE 2026-06-17): per-iteration checkpoint pruning orphans .json lineage sidecars** (Low —
cosmetic but noisy). The FLYWHEEL-CKPT-1 iter-keyed prune deleted step_*.safetensors but
left the matching .json sidecars, so every flywheel iteration accrued "incomplete write?"
doctor WARNINGs (7 after run5's first 3 iters; cleaned manually 2026-06-12). Fixed: both
deletion sites now drop the companion sidecar with its checkpoint — `_purge_old_checkpoints`
(train_ip_adapter.py keep_last_n path) and the orchestrator's FLYWHEEL-CKPT-1 iter-tagged
prune (`_run_flywheel_loop`, both the `iter*_step_*` and `iter*_best` globs). Takes effect on
next orchestrator restart. Guard: `train/tests/test_checkpoint_prune.py`.

**COLD-CHAMPION-PRUNE-1 (DONE 2026-06-17): cold champion archive accumulated unbounded** (Low —
silent disk creep). `_archive_flywheel_champion` copies the champion into
`cold_root/weights/flywheel-{name}-{YYYYMMDD}/` (~2-4 GB). Same-day improvements overwrite the
date-keyed dir, but across days these dirs were never pruned — one per campaign-day a champion
improved, so a long campaign accrued unbounded cold dirs. Fixed: after a successful archive the
function globs `flywheel-{name}-*`, sorts by name (YYYYMMDD sorts chronologically), and
`shutil.rmtree`s all but the newest `keep_cold_champions` (new config key, default 5; set in
`flywheel_source_probe.yaml`). Other flywheel names are untouched. Guard:
`train/tests/test_cold_champion_prune.py`.

**SRC-ATTR-1: per-source data-selection attribution — read-out + autonomous surfacing (DONE 2026-06-13).**
`ShardScoreDB.source_attribution(flywheel_name)` + `source_iteration_mix` (shard_selector.py,
single source of truth, 5 tests), `debug/source_attribution.py` CLI, and a `source_attribution`
block in the doctor `--ai` summary. Campaign-scoped from `score_updates` to dodge two traps
(cond_gap is per-iteration → naive mean confounded; shard_scores.db blends cond_gap conventions
cross-campaign). Run5 finding (iter 13): NO source separable — all sources ubiquitous (every
iteration), so contrastive cond_gap can't A/B them; the clean signal is the selection-MIX drift
(journeydb 23%→49% over the campaign — reverting toward its 58% pool share). **Open follow-up:**
clean per-source quality requires HOLDING A SOURCE OUT — BUILT 2026-06-13: not an ablation-harness
arm (that varies hyperparams on fixed data) but a flywheel selection feature.
`select_shards(exclude_sources)` + `resolve_source_holdout(cfg,iter)` + orchestrator wiring
(`flywheel.source_holdout`, default off) drop one source/iter on a rotation; the existing
whole-pool excluded-scoring then gives that source the clean excluded baseline
`source_attribution()` needs. Ready campaign: `train/configs/flywheel_source_probe.yaml`
(24 iters, 4 sources × 6 rotations); read with `debug/source_attribution.py --campaign source-probe`.
This informs `per_source_min` / the production data recipe (DP-5) with evidence, not a prior.

**SRC-ATTR-2: source-holdout result needs a CORRECTED confidence model + dominant-source
fix (from the 2026-06-18 source-probe, 18 iters).** The holdout ran and produced PARTIAL,
heavily-caveated results — DO NOT act on them as-is:
- **Statistical artifact (the trust gate is FOOLED):** source-level holdout excludes ALL of
  a source's shards in the SAME iterations, so every shard of that source gets an IDENTICAL
  per-shard attributed_cond_gap (std reported as ±0.0000). So `n_attributed=12` is really ~12
  copies of ONE source-level measurement; effective n = number of ROTATIONS (~4-5 over 18
  iters), not 12. The `★ separable` flag (≥5 attributed shards) overstates confidence. FIX:
  `source_attribution()` should compute effective-n from distinct holdout iterations (or do
  the contrast at SOURCE level directly), and gate trust on that, not shard count.
- **Dominant source unanswerable per-shard:** `journeydb` (744 shards, 58% of pool — the key
  question) never separated because its shards are spread too thin to each reach ≥3 incl &
  ≥3 excl. Per-shard rollup structurally can't attribute big sources; need source-level
  contrast or far more rotations.
- **Validity-gated (SREF-METRIC-1):** results are cond_gap (conditioning), not style. The raw
  verdict (journeydb+wikiart +0.0054 helps, coyo+laion+wikiart −0.0138 hurts) is NOT a recipe
  signal — "hurts cond_gap" may mean "harder to condition on" = possibly better for style.
  Re-run / re-interpret only AFTER IP-ADAPTER-INFER-1 + a valid sweep settle the validity gate.

**SMOKE-ISOLATION (DONE 2026-06-13): smoke/test campaigns must not write the production shard_scores.db.**
Smoke runs score ALL 1280 shards per iteration (excluded-EMA), so the EMA-lag smoke's cond_gap -5.066
contaminated every shard's excluded mean in the cross-campaign KB, corrupting run5's selection
(non-uniformly → distorted rankings). Fix: orchestrator routes campaigns with `ephemeral_scores: true`
(auto for names starting smoke/test) to `shard_scores_scratch.db`. Cleanup tool
`train/scripts/rescope_shard_scores.py --keep-campaign <name> --apply` rebuilds the shards-table EMAs
from one campaign's score_updates (single convention; backs up first; requires a pause). Applied to run5
2026-06-13: dropped run2/run3/smoke-style2/smoke-style3 rows (8960), 0 contaminated shards remain.
Also fixed: `pipeline_ctl pause-flywheel` lacked `--free-gpu` (the documented flywheel-pause command
silently targeted the chunk-pipeline control file) — now exposed.

**FW-PAUSE-1 (DONE 2026-06-13): free-gpu pause left an orphaned trainer process tree.**
`_kill_flywheel_gpu` killed the tmux training window / wrapper Popen but not the descendant
chain (bash -c → caffeinate → python → MLX), which survived holding GPU memory — silently
defeating --free-gpu on an UNATTENDED pause (the day/night sharing use case). Found when the
shard-score rescope required a real free-gpu pause and the trainer had to be pkill'd by hand.
Fix: `_reap_proc_tree(pattern)` SIGTERM→(10s)→SIGKILL-reaps the chain by campaign-scoped cmdline
(trainer config path / precompute staging path — never touches an unrelated trainer) and verifies
the device is clear. Hermetic test confirms reap + scoping. Takes effect on next orchestrator restart.

**FLYWHEEL-CHAIN (DONE 2026-06-13): unattended campaign auto-chaining + launchd /Volumes gotcha.**
The orchestrator has no campaign queue, so `train/scripts/flywheel_chain.sh` watches the active
flywheel's tmux window and launches the next config when it closes (debounced 2×, settle 90s,
sentinel-idempotent, MAX_WAIT_H cap that refuses to stack onto a hung run). MUST run in TMUX, not
launchd: **launchd LaunchAgents cannot write the external /Volumes mount without Full Disk Access**
(fails EX_CONFIG/Operation-not-permitted) — discovered when the launchd chain exited 78 and its
/Volumes log writes were denied. tmux runs in the user session (full access) with the same durability
as the flywheel it chains. Same bug fixed in nightly_health.sh: results now go to
`$HOME/Library/iris-health/` (boot volume), doctor reads there first, /Volumes second; launchd plist
stdio also moved to $HOME. Armed for the 2026-06-15→18 absence: run5 → flywheel_source_probe.yaml (18 iters).

  - **CLEAN RE-RUN (2026-06-24) — single-shard confound fixed (seed + interleave).** The cond-attrib
    JSONL exposed that every 3k-step arm trained on ONE random shard (unseeded shuffle + drain-one-
    shard); fixed with data.seed + records_per_shard_visit (commit). Clean baseline (hybrid, no gate,
    seed=42, cap=135 → all 22 shards × 135 records): inj ratio **0.576 @ 0.5, 0.634 @ 0.7** — vs the
    old single-shard 0.687. So the HONEST baseline is ~0.58–0.63; the 0.687 was ~0.07–0.11 inflated by
    a lucky shard. The confounded arm-b (0.575) == clean baseline @ 0.5, i.e. down-weighting SigLIP is
    likely ~NEUTRAL not negative on honest footing. Clean gate arms (siglipdn, hier; same seed+data)
    running next for a trustworthy A/B. NOTE: earlier CSD/SigLIP/hybrid METRIC numbers share this
    single-shard confound (qualitative content+style win stands — it was visual).

  - **CLEAN GATE A/B (2026-06-24, identical seeded data — gate is the ONLY difference):**
    clean_base: 0.576 @0.5 / **0.634 @0.7**.  clean_siglipdn (SigLIP V×0.3): **0.629 @0.5** / 0.603 @0.7.
    Down-weighting SigLIP is NOT negative (reverses the confounded arm-b 0.575 — that was the shard):
    it improves @0.5 and ties at the peak (~0.63). Both cap at ~0.63 — the V-gate shifts the usable
    scale but can't break the ceiling (can't disentangle style/leak, only trade total contribution).
    hier arm pending. If hier also caps ~0.63 → V-gate lever exhausted; next = leak penalty in the
    loss or curated/fewer SigLIP tokens (disentangle the signal, not scale it).

  - **FINAL CLEAN GATE VERDICT (2026-06-25) — V-gate lever EXHAUSTED; ~0.65 is a hard ceiling.**
    All three clean arms (identical seeded multi-shard data, gate the only difference):
      clean_base 0.576/**0.634**,  clean_siglipdn **0.629**/0.603,  clean_hier 0.555/**0.648**.
    All cap ~0.63–0.65, within n=10 noise — NO gate config beats the baseline meaningfully, none
    nears 0.75. Scaling SigLIP's V (uniform or per-block) shifts the peak SCALE but can't break the
    ceiling: it trades the entangled signal's total contribution, it does not disentangle style from
    leak. ~0.65 is the SAME ceiling as the original SigLIP leak campaign → a fundamental property of
    conditioning on entangled SigLIP patches, robust across SigLIP / CSD / hybrid / all gate variants.
    (clean_hier @0.3 drives leak≈0 but style≈0 too — CSD-early routing suppresses leak but not
    usefully.) **NEXT LEVER — disentangle the SIGNAL, don't scale it:** (1) explicit LEAK PENALTY in
    the training loss (penalize gen↔ref content-head cosine, or an adversarial/contrastive content
    term) so the adapter is trained to inject style WITHOUT content; (2) CURATE/REDUCE SigLIP tokens
    (drop content-laden patches, keep style-salient ones / use a style-pooled descriptor). Gate code
    + cond-attrib + seeded loader all retained as infrastructure. (d) longer-training NOT run — base
    converged by 3k and the ceiling is signal-bound, not training-bound.

  - **LEAK PENALTY VERDICT (2026-06-25) — FIRST lever to break the ~0.65 ceiling; weight 0.5 too hot.**
    clean_leak (leak_loss_weight=0.5, hybrid, seed=42, all 22 shards — crashed thermally @2k, resumed
    to 3k via the skip_shards fix so coverage matches baseline). vs clean_base (identical data, no leak):
    | scale | base ratio | leak ratio | leak Δstyle | leak prompt_adh |
    |------:|----:|----:|----:|----:|
    | 0.3 | −0.40 | 0.543 | +0.016 | +0.153 |
    | 0.5 | 0.576 | **0.719** | +0.150 | +0.097 |
    | 0.7 | 0.634 | **0.911** | +0.243 | −0.013 |
    The 0.911@0.7 is a FALSE win (images WASH — content-free texture games the CSD style metric;
    prompt_adh −0.013 flags it). The REAL result is the MATCHED-CONTENT comparison: at the same
    prompt_adh (~+0.097, scale 0.5), leak gives ratio 0.719 / Δstyle +0.150 vs base 0.576 / +0.083 —
    ~DOUBLE the style at the same content level. Genuine disentanglement (the content-preservation-vs-null
    penalty trains a style-purer injection), the first lever to clear 0.65 at usable content. **BUT
    weight 0.5 is too hot:** the style-pure injection washes content at usable scales (0.5 landscape
    washed, violinist heavily smeared; 0.7 fully washed). **NEXT: weight sweep DOWN (0.1, 0.25) to land
    clear content + ratio>0.65 simultaneously.** Also: leak penalty raised peak mem to 27GB → thermal
    crash in the heatwave; trim peak mem (free cond graph before null forward) before long runs.

  - **LEAK WEIGHT SWEEP + SOBER RE-READ (2026-06-26) — earlier "ceiling break" was at SMEARED content.**
    Three doses, identical seeded data (ratio @0.5 / Δstyle@0.5 / prompt_adh@0.5):
      base(0): 0.576 / +0.083 / +0.096 · leak0.25: 0.427 / +0.056 / +0.128 · leak0.5: 0.719 / +0.150 / +0.097.
    Dose response is NON-MONOTONIC: 0.25 preserves content BEST (sharp violinist, prompt +0.128) but
    suppresses style → WORST ratio (0.427 < baseline). 0.5 = strong style but smears/washes. Eyeballing
    every arm: **the high ratios (0.69–0.91 @0.7, 0.719 @ leak0.5/0.5) all coincide with content
    WASHING/smearing** — the CSD style_sim metric rewards a ref-matching texture (high style, no content
    → low leak → high ratio), so washing GAMES the ratio; prompt_adh→0 is the tell. Sorted by CONTENT
    QUALITY: sharp content (prompt≥+0.13) tops out ~0.43–0.54 ratio; only smeared/washed content reaches
    >0.65. **So clean-content style transfer still plateaus ~0.5–0.58; the leak penalty shifts the
    sharp-content⟷strong-style frontier but doesn't break it with content intact.** The earlier
    "0.719 broke 0.65" was at SMEARED content (prompt +0.097 = smeared-but-present violinist), not clean.
    **Implications:** (1) the injection-ratio metric is GAMEABLE by washing — needs a content-quality
    gate (only credit ratio where prompt_adh stays high) before more ratio-chasing. (2) dose-tuning the
    leak penalty is not a clean dial; 0.1 unlikely to help. (3) untried lever = option 2 (curate/reduce
    SigLIP tokens) targets clean-content style directly. (4) the hybrid PLATFORM (content+style at
    moderate scale) remains the usable deliverable regardless of the ratio number.

  - **CONTENT GATE + HONEST RE-JUDGE (2026-06-26) — wash artifacts removed; honest plateau ~0.54.**
    Added a content gate to the eval (sref_sweep_eval.py + sref_regrade.py): inj_ratio counts as a
    real win ONLY where prompt_adherence ≥ 75% of the no-adapter null's (0.1516) — i.e. content
    retained, not washed. Re-judged all saved arms (no re-gen). EVERY ratio >0.65 was a WASH artifact
    (retain ≤ 0.64). Honest best CONTENT-PRESERVING inj_ratio, clean arms:
      clean_leak(0.5) 0.543 @0.3 (retain 1.01) · clean_leak025 0.427 @0.5 (0.84) · clean_base −0.40 @0.3.
    Findings: (1) the metric IS gameable by washing — confirmed; the gate fixes it. (2) at
    content-safe scales the absolute style is TINY (Δstyle +0.016–0.056) → honest ratio ~0.43–0.54;
    strong style requires content-washing. (3) the leak penalty still edges the field on the gated
    metric (0.543), but the gain is modest. (4) NO lever (gate/CSD/hybrid/leak) delivers strong style
    + preserved content — the ~0.54 honest plateau holds. Threshold note: 0.75 is strict (marks the
    eyeballed-good hybrid@0.5 retain-0.74 as borderline); the qualitative conclusion holds for any
    reasonable floor 0.70–0.80. The eval default is now --content-retain 0.75; sref_regrade.py re-grades
    saved runs offline.

  - **DATA-QUALITY / ULTRA-SIGNAL SHARDS (2026-06-26) — the pool is style-signal-DILUTED.**
    New CPU tool `style_strength_select.py` ranks records by style_strength = mean top-5 CSD-neighbor
    cosine (dense high-cos neighborhood = strong repeatable style + clean cross-ref pairs) from the
    existing neighbors.sqlite — no GPU. Analysis of the current 22-shard pool (109,253 records):
    strength min 0.316 / med 0.718 / p90 0.817 / max 0.950 — WIDE. Per-shard ultra-signal (top-25%,
    strength≥0.774) concentration varies ~50x: shards 000000/000004/000008/000012 ≈ 39–40% ultra-signal
    vs 000812/000779/000883/001179 ≈ 0.7–4%. The interleaved loader trains on all 22 equally → ~half the
    data teaches WEAK/generic style. Plausible cause of the weak clean-content style transfer (the
    data-bound H2 hypothesis). **Cheap first test (no new precompute — the shards are already cached):**
    train an ULTRA-SIGNAL arm on the top-25%-by-strength subset (or signal-rich shards only) → content-
    gated eval vs the diluted pool. If it beats ~0.54 → data-bound (then scale via whole-universe
    cheap-CSD-score → select → precompute-only-the-subset, using campaign_manager/shard_scores). If not →
    architecture-bound, ship hybrid. csd_mlx.encode_both() also exposes a content head for future
    style/content-ratio scoring. Pending: pool9 (option-2) verdict first, then the ultra-signal arm.

  - **DATA-QUALITY HARVEST — SAMPLING-BIAS CAVEAT (2026-06-26, capture before scaling).** The universe
    CSD (`/Volumes/16TBCold/precomputed/style/v1_csd`, per-image CSD, 768-d L2, keyed by rec_id) is a
    200-records/SHARD SCOUT (256K recs, 1280 journeydb shards). Universe style-strength matches the
    22-shard pool (med 0.717/p90 0.815/max 0.949) → NO higher quality tier outside, but ~25x more
    ultra-signal VOLUME, concentrated by shard (top shards 000200/000098/000481 ≈28-29% ultra-signal;
    many at 0%; the richest are mostly OUTSIDE the current pool). **CAVEAT (do NOT shard-select on the
    scout for the final set):** a 0/200 shard is not empty — rule-of-3 → true ultra-signal rate ≤~1.5%
    → up to ~75 great images per "dud" shard × hundreds of shards = real loss. Shard-level scout
    selection is PRIORITIZATION only; the FINAL extreme-signal set must be PER-IMAGE over FULL coverage
    (full CSD sweep of ALL records, then pick top images regardless of shard). Cost decision at harvest:
    full sweep all records (no gems lost, expensive 6.4M CSD) vs scout-prioritized partial sweep
    (cheaper, quantified ≤~1.5%-rate loss in skipped shards). The CHEAP concentration test is UNAFFECTED
    — it uses full per-record CSD of the 22-shard pool (pool_top25.json, 27,313 recs, record-level, no
    scout). Built: dataset.record_allowlist + style_strength_select.py; arm clean_concentrate queued.

  - **==== SREF CAMPAIGN — MASTER RESUME STATE (2026-06-26) ====**
    GOAL: break the injection-ratio plateau for style transfer with CONTENT PRESERVED. Metric =
    null-relative Δstyle/Δleak read THROUGH the CONTENT GATE (sref_sweep_eval.py / sref_regrade.py:
    a scale only counts if prompt_adherence ≥ 75% of the no-adapter null 0.1516 — washed gens that
    game the CSD style metric are excluded). HONEST PLATEAU ≈ 0.54; raw ratios >0.65 were all
    content-WASH artifacts.
    LEVERS TRIED (all clean, seed=42, all-22-shards, content-gated):
      • V-gate (per-block per-group SigLIP/CSD injection weight): EXHAUSTED — caps ~0.63–0.65, can't
        disentangle (scales the entangled signal). siglipdn/hier/learned all ~baseline.
      • Leak penalty (content_leak_loss = instance-norm content drift cond-vs-null, train style-without-
        content): the ONLY lever that edges the honest metric (best content-preserving ~0.543 @ w=0.5,
        low scale). w=0.5 over-stylizes/washes at usable scales; w=0.25 too weak (non-monotonic).
        Cheap (~13% via shared qs/h_final). Memory hot (27GB → thermal crashes in the heatwave).
      • Option-2 SigLIP pooling (data.siglip_pool_grid=9 → 729→81 tokens): clean_pool9 TRAINING NOW
        (verdict pending). Token-count agnostic → no model/C/export change.
    DATA-QUALITY LEVER (the live thread):
      • Pool is style-signal DILUTED: per-record CSD-neighbor-density (style_strength_select.py) spans
        0.32–0.95; ultra-signal concentration varies ~50x across the 22 shards (art-ish 39-40% vs
        photo-ish <4%); loader trains all equally → ~half teaches weak style.
      • Universe CSD ALREADY precomputed: /Volumes/16TBCold/precomputed/style/v1_csd = per-image CSD
        (768-d L2, keyed rec_id), 200-recs/SHARD SCOUT × 1280 journeydb shards = 256K recs. Staged to
        /Volumes/2TBSSD/universe_csd; universe_neighbors.sqlite + universe_ultra_signal.json (top-10%,
        25,598 recs) built (CPU). Universe style dist == pool dist (max 0.95) → NO higher quality tier
        outside, but ~25x more VOLUME, concentrated in shards the pool MISSED (richest 000200/000098/
        000481 ≈29%, NOT in pool). Hot pool tests PURITY; universe adds VOLUME.
      • SAMPLING CAVEAT: 0/200 scout shard ≠ empty (rule-of-3 ≤1.5% rate). Shard-level scout select is
        PRIORITIZATION only; FINAL extreme-signal set = PER-IMAGE over FULL coverage.
    COST-LADDERED PLAN (each rung de-risks the next; only pay big precompute once proven twice):
      1. HOT POOL, FREE (22 shards already fully precomputed): clean_concentrate arm BUILT + queued —
         data.record_allowlist=/Volumes/2TBSSD/sref_eval/pool_top25.json (top-25% by style strength,
         27,313 recs, RECORD-LEVEL over full-CSD 22 shards — NOT the scout, no sampling bias). Tests:
         does concentrating signal beat 0.576? If promising, sweep top-10/50%, optionally repack
         extreme-signal shards as a pipeline rehearsal — all free.
      2. TARGETED EXPANSION, ~days (if rung1 wins): precompute VAE/Qwen3/SigLIP for the scout-identified
         RICH shards (~50-100, not all 1280) → more VOLUME at fraction of cost; validates data lever at
         scale. Known loss: ≤1.5%-rate gems in skipped shards (deliberate).
      3. FULL UNIVERSE SWEEP, ~weeks (only if rung2 still winning): full per-image CSD over ALL records
         → universe-wide top-IMAGE selection (per-image, not per-shard) → precompute keepers → repack
         extreme-signal shards → ultimate run.
    NEW CODE (committed): dataset.record_allowlist + siglip_pool_grid + skip_shards + seed/cap;
    sref_sweep_eval content gate + sref_regrade.py; style_strength_select.py; content_leak_loss +
    leak_loss_weight; hybrid_features --siglip-pool-grid. All Python; no C/export/parity change since
    the hybrid (v4.1.0). make mps NOT needed for these (training/eval only).
    OPERATIONAL: pool9 training (GPU, ~step 1225/3000, slow in 35°C heatwave — thermal crashes happen,
    runs are crash-RESUMABLE via warmstart_path=step_ckpt + auto skip_shards). clean_concentrate queued
    behind it. Heatwave → don't stack GPU jobs; CPU analysis (ranking/regrade/select) is safe in parallel.

  - **==== SREF POST-PLATEAU DECISION TREE (2026-06-26) — if the data axis also fails ====**
    DIAGNOSTIC MEANING of "clean_concentrate (rung-1 data lever) flat": the bottleneck is NOT
    "the model could disentangle style from content given cleaner signal." Architecture levers
    exhausted (V-gate, leak penalty, pooling) + data lever flat ⇒ the plateau is a property of
    the REPRESENTATION + INJECTION MECHANISM, not the training data. We inject SigLIP-DOMINANT
    features (content-rich by construction) via KV cross-attention (leaks content at any scale).
    No amount of cleaner pairs fixes a representation that entangles style+content at the source.
    SIGNATURE to watch: ip_scale double=0.0000 across EVERY run — the double-blocks never engage;
    injection lives only in single-blocks. Possible capacity/representation fingerprint (→ Tier 3).

    NEXT STEPS, cheapest-and-most-diagnostic first (run in this order; each gates the next):
      TIER 0 — ship the deliverable regardless, ~free, do in parallel:
        Expose the hybrid as a tunable `--sref-strength` knob; call ~0.54 the RECOMMENDED operating
        point, not a ceiling. MJ `--sref`/`--sw` is exactly this content↔style tradeoff knob. A
        plateau reframed as a user-tunable Pareto point is a PRODUCT, not a failure. Already built
        (web/server.py --ip path + --ip-scale); just document/surface it. This is the floor.
      TIER 1 — cheap, attacks the root cause, little/no retrain:
        1.1 INJECTION SCHEDULE OVER TIMESTEPS (untried; highest ROI). Today ip_scale is a CONSTANT
            scalar across all denoising steps. Content/layout is decided in EARLY steps; texture/
            style in LATE steps. Inject style only after step ~k (or ramp) so content forms first,
            unconditioned → style-without-content for free, NO training. Prototype in the Python MLX
            forward as a diagnostic; if it works, wire per-step scale into the C injection path
            (iris_sample.c / adapter inject). DO THIS FIRST.
        1.2 CSD-DOMINANT CONDITIONING. We inject SigLIP-dominant; CSD is the STYLE-SPECIALIZED
            encoder, currently 1 padded row in hybrid. Rebalance the hybrid toward CSD, or run
            cond_mode=csd. Attacks entanglement at the SOURCE not at injection. One config-only run.
      TIER 2 — one training run each, different mechanism/objective:
        2.1 AdaIN / FEATURE-STATISTICS injection (channel mean/std transfer) instead of/alongside KV
            injection — content-agnostic by construction; a categorically different mechanism than
            every KV-injection variant tried so far.
        2.2 DISENTANGLEMENT OBJECTIVE beyond the soft leak MSE: swap-consistency / contrastive style
            loss (same-style-diff-content refs must yield the SAME conditioning effect) or an
            adversarial content-classifier the conditioning must fool. Harder structural constraint.
      TIER 3 — expensive, strategic, last:
        3.1 9B BASE. The 4B distilled base may lack capacity to honor prompt AND style at once (the
            double=0.0000 signature hints at this). Same adapter recipe on 9B tests base-bound vs
            mechanism-bound. Most expensive → gates on all the cheap ones failing first.
    RECOMMENDATION if rung-1 is flat: Tier 0 (ship knob) + Tier 1.1 (schedule sweep) IMMEDIATELY —
    the schedule is the one mechanism axis genuinely untouched and is eval-time cheap. Only if
    schedule is ALSO flat → 1.2 CSD-dominant, then 2.1 AdaIN, then 3.1 9B.
    HONEST CAVEAT: content-preserving style transfer is a real Pareto tradeoff; ~0.54 may be near
    where the frontier sits for THIS base + feature family. The schedule sweep (1.1) is the
    strongest remaining reason to think the plateau is beatable rather than fundamental.

  - **SREF pool9 (option-2 SigLIP pooling 729→81) VERDICT (2026-06-26): NULL — does not break the
    plateau.** Trained clean (seed42, all-22-shards, cap135, NO leak penalty, siglip_pool_grid=9),
    thermal-crashed @2500/3000 (Metal "Impacting Interactivity", 2nd of the heatwave); step_2500 EMA
    bundle graded — 2500 vs 3000 won't flip a content-gated verdict. Content-gated frontier (pooled
    [82,1152] refs, null prompt-adherence 0.1516): @0.3 Δstyle +0.0055 retain 0.996 ✓ (≈zero style);
    @0.5 ratio 0.787 retain 0.188 ✗WASH; @0.7 ratio 0.971 retain −0.108 ✗WASH. Best content-preserving
    ratio 0.160 @0.3 — but Δstyle≈0 there, so the ratio is meaningless. SAME shape as clean_base
    (clean_base @0.3 Δstyle −0.0099 retain ~1.0; @0.5 prompt-adher 0.0957 = retain 0.63 ✗WASH — its
    0.576/0.634 are WASH-scale ratios, NOT content-preserving). So pooling neither helps nor hurts at
    the content-preserving operating point; it's another plateau null. The noisy training-loss gap
    swings (+12.6% … −51.9% window-to-window) were NOISE, not signal — don't read cond/null gap as a
    lever verdict; only the content-gated eval decides. CONFIRMS architecture levers exhausted. Pooled
    refs at /Volumes/2TBSSD/sref_eval/refs_feat_hybrid_pool9 (eval_set_pool9.json). Minor: hybrid_
    features.py JSON stdout mis-reports "rows":730 when pooling (file is correctly [82,1152] by byte
    size 377856=82*1152*4) — cosmetic log bug, fix later. NEXT: data lever (clean_concentrate), then
    DP-7 if flat.

  - **SREF clean_concentrate (rung-1 DATA lever, top-25% style-signal) VERDICT (2026-06-27):
    DIRECTIONAL WIN — the data lever measurably shifts the content/style frontier, but does not yet
    clear a strictly content-preserving operating point.** First crash-free 3000-step run (heatwave
    passed). Trained on record_allowlist=pool_top25.json (27,313 recs, top-25% by CSD-neighbor style
    strength over full-CSD 22 shards), else IDENTICAL to clean_base (hybrid, seed42, cap135, 729
    SigLIP, NO leak penalty). Content-gated frontier (null prompt-adher 0.1516), vs the two matched
    baselines:
      arm                @0.3 Δstyle/ratio/retain      @0.5 Δstyle/retain        @0.7 retain
      clean_base         -0.0099 / -0.40 / 1.0 ✓        +0.079 / 0.63 ✗WASH        ✗WASH
      pool9 (pooled)     +0.0055 / 0.16 / 1.0 ✓         +0.168 / 0.19 ✗WASH        ✗WASH
      clean_concentrate  +0.0068 / 0.245 / 1.0 ✓        +0.112 / 0.742 ✗(edge)     -0.08 ✗WASH
    The ROBUST signal is content RETENTION at scale 0.5 (where real style transfers, Δstyle≈0.11):
    clean_base 0.63 → concentrate 0.742, i.e. concentration pushed the content-preserving boundary
    OUTWARD, nearly clearing the 0.75 gate. At the strict content-preserving point (0.3) concentrate
    is the ONLY arm with a positive ratio (0.245 vs clean_base -0.40). So data quality IS a real lever.
    CAVEAT: 0.245 (strict-gated) is still BELOW the leak arm's ~0.543; one data arm alone does NOT
    break the plateau — frontier shifted, gate not yet cleared. NEXT (cheap, GPU now cool): STACK the
    two positive levers — concentrate data + leak_loss_weight=0.5 (clean_concentrate_leak arm) — if
    they compose, the 0.5-retain 0.742 should clear 0.75 with Δstyle~0.11. Then optionally top-10%
    concentration sweep (free). Only escalate to rung-2 targeted precompute if the stacked arm clears
    the gate. ip_scale double=0.0000 persists (every arm).

  - **SREF clean_concentrate_leak (data+leak STACKED) VERDICT + COARSE-GRID CONFOUND (2026-06-27):
    stacking did NOT cleanly break the plateau, BUT exposed that the ~0.543 plateau is likely a
    SCALE-SAMPLING ARTIFACT.** Full 3000-step run (no crash), content-gated frontier (null prompt
    0.1516, gate retain≥0.75). Cross-arm table at the fixed 0.3/0.5/0.7 grid (gate-OK rows):
      arm                     sc   Δstyle   Δleak   ratio  retain  gate
      clean_leak              0.3  0.0159  0.0293  0.543   1.010  OK   ← incumbent "plateau"
      clean_concentrate_leak  0.3  0.0468  0.1057  0.443   1.001  OK   ← stacked: 3x Δstyle, lower ratio
      clean_leak025           0.5  0.0557  0.1303  0.427   0.841  OK   ← MOST gate-passing Δstyle
      clean_concentrate       0.3  0.0068  0.0277  0.245   1.009  OK
      clean_pool9             0.3  0.0055  0.0343  0.160   0.996  OK
      clean_base              0.3 -0.0099  0.0245 -0.404   1.008  OK
    STACKING: best content-preserving ratio 0.443 @0.3 (< leak-alone 0.543) — levers TRADED OFF
    (more Δstyle 0.047 AND more Δleak 0.106 → net lower ratio), did NOT compose additively. At 0.5 it
    washes catastrophically (retain 0.742→0.040): the leak penalty makes injection more potent per
    unit scale, moving the wash CLIFF down below 0.5.
    THE CONFOUND (matters more than any single arm): every potent arm sits at retain≈1.0 @0.3 (huge
    headroom) then washes by 0.5 → its TRUE content-preserving optimum is at an UNSAMPLED scale
    ~0.33–0.45, and the coarse fixed grid catches each arm at a DIFFERENT frontier position. So
    (a) the ~0.543 "plateau" is probably a sampling artifact — interpolating clean_concentrate_leak
    between 0.3(retain1.0,ratio0.443) and 0.5(retain0.04) puts its gate-crossing ~0.36–0.40 with
    higher Δstyle+ratio than 0.443, plausibly >0.543; (b) clean_leak025 was WRONGLY dismissed as "too
    weak" — at 0.5 it PASSES the gate (retain0.841) with the highest gate-Δstyle (0.0557), its cliff
    is higher. The eval docstring's own rule: compare at MATCHED content budget, NEVER a single fixed
    scale (SREF-EVAL-PARAMS). We've been violating it.
    NEXT (cheap, EVAL-ONLY, no retrain, GPU cool): fine scale sweep 0.35/0.40/0.45 on the top arms
    (clean_concentrate_leak, clean_leak; add clean_leak025 @0.55/0.60) to locate each true peak
    content-preserving ratio, then compare at matched budget. This likely RAISES the honest plateau
    number and gives the real stacking verdict. Only after that decide rung-2 vs DP-7.
    STAGED-SWEEP METHODOLOGY (hierarchical bisection, not brute force): coarse 0.3/0.5/0.7 (locate
    arm) → medium 0.35/0.40/0.45 (localize each arm's gate-crossing cliff, pick champion) → MICRO
    0.01 increments across a ~0.06 window around the CHAMPION's crossing only (~70 gens, cheap). The
    content gate is a hard threshold, so the true champion = the exact scale just below retain<0.75
    where Δstyle/ratio peaks; the 0.01 sweep pins it. That champion scale IS the shippable DP-7
    Tier-0 `--sref-strength` default. CAVEAT: at 0.01 granularity per-pair noise (n=10,1 seed) may
    exceed the inter-scale signal — bump to multiple seeds / more pairs at the micro stage so a 0.01
    difference is statistically real, not jitter.

  - **SREF DP-7 INJECTION-SCHEDULE VERDICT (2026-06-28): timing does NOT beat the ~0.70 ceiling →
    ceiling is MECHANISM-BOUND (3 independent levers now converge there).** Implemented --ip-schedule
    (none|late:F|early:F) in C (commit bdfd7ff, golden parity 20/20, inert verified) + harness
    passthrough (1c19802). The schedule gates ip_scale by denoising-step fraction so style injects only
    in late (low-noise) steps, letting content form first. Content-gated sweep of the champion
    (clean_concentrate_leak, distilled 4-step) across late:0.5 full grid:
      late:0.5: 0.36 r0.452/ret0.962 · 0.39 r0.540/ret0.931 · 0.42 r0.615/ret0.883 · 0.44 r0.653/
                ret0.831 · 0.46 r0.678/ret0.796 · 0.50 ret0.702 WASH · 0.7/0.9 WASH
      BEST content-preserving: ratio 0.678 @0.46 (Δstyle 0.131, retain 0.796).
    CONTROL @0.39 (schedule on vs off): late:0.5 Δstyle0.094/r0.540/ret0.931 vs no-sched
    Δstyle0.127/r0.696/ret0.768 — at fixed scale the schedule TRADES style for content (less style,
    more retain, lower ratio). The schedule DOES work mechanically (at scale 0.5 it lifts retain
    0.040→0.702 vs no-schedule — content protection is real), but its content-preserving PEAK (0.678)
    is TIED-marginally-below the no-schedule champion (0.696). So timing shifts the operating point and
    adds content headroom, but does NOT raise the achievable frontier. late:0.25 protects less (more
    inject steps); more-aggressive gating just trades along the same ~0.70 frontier.
    THREE INDEPENDENT LEVERS now converge on ~0.70: data concentration, leak objective, injection
    timing → the ceiling is MECHANISM-bound (entangled SigLIP-dominant repr + KV cross-attn injection),
    NOT data/objective/timing-addressable. DP-8 GATE FAILS → the mega data run is NOT justified (more
    data can't break a mechanism ceiling). CAVEAT: verdict is on the 4-step DISTILLED model (≤4 schedule
    grid points, no clean content/texture temporal split); a 50-step BASE model has more room and is the
    one untested angle for timing — separate, more expensive (CFG) experiment. Remaining >0.70 shots are
    all speculative RETRAINING: CSD-dominant conditioning, AdaIN/stats injection, or 9B base.
    SHIP: Tier-0 --sref-strength at champion 0.38–0.39, NO schedule (schedule adds a knob with no
    ratio benefit; could expose late:0.5 as an optional "max-content-safety" mode — secondary).
    SHIPPED 2026-06-28 (commit 97e8c66): web/server.py dispatches hybrid feature production
    (hybrid_features.py, CSD row) when the bundle is cond_mode=hybrid; --sref-strength default 0.38
    (server fallback + frontend style-mode slider; override IRIS_SREF_STRENGTH). Deploy:
    IRIS_IP_BUNDLE=/Volumes/2TBSSD/sref_eval/clean_concentrate_leak/bundle. 125 web tests green.

  - **SREF-BASE-1 (BACKLOG, 2026-06-28): re-run the injection-SCHEDULE + lever sweeps on the 50-step
    BASE model (flux-klein-4b-base --base) — the ONE untested angle that could beat the ~0.70 ceiling.**
    WHY: the DP-7 timing verdict (ceiling mechanism-bound) was measured on the DISTILLED 4-step model,
    which gives a schedule only ≤4 grid points and NO clean content(early)/texture(late) temporal
    separation. A 50-step base model gives the schedule 50 grid points (late:0.5 = inject last 25
    steps) and a real early-structure/late-texture split — the regime where "let content form first,
    inject style late" can actually work. This is the last cheap-ish shot at >0.70 before the expensive
    speculative retrains (CSD-dominant / AdaIN / 9B).
    MEMORY (measured 2026-06-28, M1 Max, 256x256, /usr/bin/time phys_footprint): base ≡ distilled —
    noIP 12.55G vs 12.73G; +IP(champion) 19.38G vs 19.37G. IDENTICAL footprint (same 4B arch, both 15G
    on disk). Step count & CFG do NOT change peak memory (per-step buffer reuse; CFG forwards are
    sequential). IP-adapter adds ~6.7G (weights + CPU-block f32 buffers — GPU fast paths are disabled
    when tf->ip set). So MEMORY IS NOT A BLOCKER for the base experiment at ≤512px; the cost is TIME
    (~25x forwards: 50 steps x2 CFG vs 4 steps) AND the IP path forces CPU blocks (~4x slower) → base+IP
    generation is ~slow; budget eval time accordingly. Caveat: at >512px the +IP footprint (~19.4G@256)
    grows toward the ~21.5G system ceiling — check headroom before high-res.
    RUNNABLE NOW (no new code — --ip-schedule is wired into all CFG euler variants, commit bdfd7ff):
      iris -d flux-klein-4b-base --base --ip <champion> --ip-features <hybrid> --ip-scale S \
           --ip-schedule late:F --steps 50 -p PROMPT
    Then content-gated sweep (sref_sweep_eval --ip-schedule, commit 1c19802) over (S, F) — find if any
    base-model schedule clears the gate with ratio >0.696.
    CAVEATS to resolve first: (1) the champion adapter was TRAINED against the DISTILLED transformer's
    attention; using it on the BASE transformer is an unproven weight-transfer — it attaches (same
    25-block/3072-dim structure) but quality transfer is unknown. May need a base-trained adapter, or
    first measure how the distilled adapter behaves on base. (2) CFG×IP interaction: inject currently
    applies to BOTH uncond and cond passes — verify that's intended (image conditioning is arguably
    CFG-independent, but confirm vs training). (3) re-derive the base model's own no-adapter null +
    champion scale (the 0.39 champion scale is distilled-specific; base will have a different sweet
    spot). Gates DP-8: only if base-model timing/levers beat ~0.70 does the data mega-run reopen.
    TRANSFER-CHECK VERDICT (2026-06-28): the distilled-trained champion does NOT transfer to the base
    model → the CHEAP path of SREF-BASE-1 is DEAD. Visual check (base, --base, 24-step, 256x256,
    artnouveau ref, prompt "a cat sitting on a chair"): base NO-adapter = clean coherent ginger cat
    (base model is fine); base + champion @0.38/0.5/0.8 = ALL muddy brown/speckled OOD texture — content
    destroyed AND the reference palette NOT reproduced, at every scale. Cause: the per-block to_k_ip/
    to_v_ip were trained against the DISTILLED transformer's attention distribution; the undistilled
    base transformer's attention is different → injecting distilled K/V is out-of-distribution (CFG×IP
    dual-pass injection likely compounds). CONSEQUENCE: testing timing/levers on a 50-step model now
    requires TRAINING A BASE-MODEL ADAPTER first → SREF-BASE-1 moves from "last cheap shot" to the
    SPECULATIVE-RETRAIN tier alongside CSD-dominant / AdaIN / 9B. The cheap levers are now FULLY
    exhausted; the ~0.70 distilled champion (shipped) is the deliverable, and every remaining >0.70 path
    is an expensive speculative retrain. The 5-min transfer check (run BEFORE any base sweep) saved
    hours of garbage base schedule sweeps — keep this gate for any "use adapter X on model Y" idea.

  - **SREF FEATURE COMPLETION (2026-06-28): productionized the shipped single-reference adapter
    across engine/web/UI + added the missing test coverage.** DONE:
    • Web: `--ip-schedule` now forwarded end-to-end — `run_generation_sref(..., ip_schedule)` adds the
      flag; `/generate` reads `ip_schedule`, validates via `normalize_ip_schedule()` (mirrors the C
      parser; 400 on malformed), returns it; UI "Style timing" selector (Constant/Late/Early) under
      Advanced options. (Schedule doesn't raise the ceiling — DP-7 — but is a real content-headroom knob.)
    • Web: single-reference is explicit, not silent — `/generate` with >1 style slot uses the first and
      returns a `warning`; the UI surfaces it. (The trained adapter is single-ref; fusion is deferred.)
    • Engine: removed the DEAD `--sref`/`--sref-scale` flags + the "v2.6 not implemented" runtime block +
      the `iris_params.sref_*` fields + `IRIS_PARAMS_DEFAULT` trailing values (positional-macro hazard —
      caught the excess-initializer warnings). CLI style-ref path is `--ip`/`--ip-features`; help points
      there. The literal `--sref` flag was an abandoned design distinct from the shipped IP-adapter.
    • Tests: web `TestSrefCompletion` (8) — schedule validator accept/reject, schedule passthrough,
      bad-schedule 400, none-default, 0.38 default strength, multi-style warning, hybrid-vs-siglip
      feature dispatch by cond_mode. C `test_schedule()` in debug/test_ip_adapter.c (17 checks) — set_
      schedule parse + set_step per-step multiplier (late/early/none, edges, bad-spec inert, 1-step
      div-guard). make test green (28 unit + 125 web + 7 smokes); test_ip_adapter 37/0.
    DEFERRED (out of "use the existing adapter" scope → new feature / big engine work):
    • SREF-MULTIREF: true multi-reference style FUSION (perceiver/feature concat for N refs). Today N>1
      uses the first + warns. Needs perceiver redesign + a retrained multi-ref adapter.
    • SREF-CLI-IMG (= G-1 Phase 3): `--sref <image>` in the C binary needs a C-native SigLIP+CSD encoder
      (today features are produced by Python: siglip_features.py / hybrid_features.py). Until then, CLI
      style-ref requires precomputed `--ip-features`; the web does image→features via the Python sidecar.
    • SREF-3 latency (TWO independent costs — confirmed 2026-06-28 by source read):
      (A) DAEMON BYPASS / model reload per gen (~30-60s, the TRACTABLE one): sref gens do NOT use the
          resident daemon (iris --server, which keeps the base model loaded). server mode parses no
          ip_* fields (run_server_mode main.c:797; fields main.c:868-869), and the daemon is launched
          without --ip — so the IP-adapter only exists on the one-shot CLI path (iris_load_ip_adapter
          main.c:1356). run_generation_sref therefore spawns a fresh `iris --ip` per style gen that
          reloads Qwen3+transformer+VAE every time. FIX: add ip_features/ip_scale/ip_schedule to the
          server-mode JSON request + attach the bundle to the resident model PER-REQUEST (load bundle
          weights once/lazily; attach tf->ip for sref requests, detach after) and route sref through
          IrisServer.generate. CAUTION: attaching tf->ip disables GPU fast-paths (the !tf->ip guards),
          so it MUST be per-request attach/detach, not load-at-startup (else normal gens go CPU too).
      (B) GPU SKIP / CPU denoising (~1.7x at 256px, more at higher res): attaching ANY adapter sets
          tf->ip, which disables the fully-fused bf16 GPU pipeline (the `!tf->ip` guards at
          iris_transformer_flux.c:3728/4230/4475) → falls to the per-block path (still GPU matmuls +
          attention) WITH the IP injection on CPU (iris_ip_adapter_inject → mha_sdpa). Two ways to fix,
          BOTH saved here per owner request (2026-06-28):
            • B2 (#1, the real fix, big): implement IP injection INSIDE the fused bf16 pipeline —
              per-block GPU K/V projection (ip_embeds @ to_k_ip/to_v_ip) + SDPA(img_q,k_ip,v_ip) +
              scaled add, inside double_block_bf16/single_block_bf16, with the adapter weights uploaded
              as GPU bf16 tensors at attach (mind the dynamic-weight GPU-cache hazard — cf. MPS B-cache
              bug). Gate so the NON-adapter path stays byte-identical (normal gens untouched). Verify vs
              golden parity (debug/test_ip_adapter.c, corr>0.999) + a real-image CPU-vs-GPU A/B.
              ~half-day, core-path risk (contained to sref path). Biggest win (~1.7x denoising +).
            • B1 (#2, safer interim, partial): GPU-accelerate just the inject in the EXISTING per-block
              fallback (route iris_ip_adapter_inject's K/V matmul + SDPA through GPU primitives instead
              of CPU mha_sdpa). Keeps the per-block (non-fused) path, so it removes the literal CPU SDPA
              but NOT the fusion-overhead gap. Low risk, quick, smaller win.
      Doing (A) alone removes the per-gen model reload (the dominant fixed cost) even though (B) keeps
      denoising on the slower path. Also: per-request feature compute inside the resident server (today
      a Python subprocess per new image; cached by sha after).
    DONE (2026-06-28): UI no longer LOOKS stalled during sref gens — run_generation_sref streams iris's
      per-step (`Step N/M`) + phase stderr as progress/status job events (was buffered subprocess.run
      with no progress); initial "Loading model" phase covers the pre-step reload. (Cosmetic.)
    (A) STATUS (2026-06-28): C server-mode IP support BUILT (commit 3872f90) + timings verified (no
      model reload), but the per-request attach produces CORRUPT (noise) output on a warm daemon —
      BUGS.md SREF-DAEMON-1. Web reverted to the working one-shot path; (A) is BLOCKED on that bug.
    ✅ (A) DONE (2026-06-29, commit 5796d92): unblocked by B2 (which fixed the warm-daemon corruption).
      The web sref /generate branch now sets ip_bundle/ip_features/ip_scale/ip_schedule on the job and
      routes through queue_generation → IrisServer.generate (resident daemon, per-request attach/detach,
      NO base-model reload). Removed the dead one-shot run_generation_sref. Live web e2e @576/4steps:
      first gen 43s (incl. one-time model load), warm daemon **25s** (vs the old one-shot ~224s EVERY
      time — ~9×). SREF-3 is now fully shipped: fast engine (B2) + model-reuse daemon (A).
    ✅ B2 DONE (2026-06-29, commit beae1b9): IP inject implemented INSIDE the fused bf16 single-block
      pipeline, fully on-GPU (ip_fused_prepare precomputes per-block k_ip + ip_scale*v_ip as bf16 GPU
      tensors once per gen — they're step-independent; Phase-11 in single_block_forward_bf16 slices the
      post-QK-norm image-Q, runs iris_gpu_attention_fused_bf16, zero-pads + adds). Gated to style-only
      adapters (double-block |ip_scale|≤1e-6). Result @576/4steps: one-shot 224s→50s, warm daemon 38s
      (4.5–6×), parity vs per-block corr 0.9998 (max|Δ| 9/255), no extra memory, MPS/BLAS/generic build
      clean. **This ALSO resolves SREF-DAEMON-1** (the fused path uses MPSGraph linears, not the buggy
      MPSMatrixMultiplication), so the daemon attach now renders correctly — i.e. (A) is unblocked.
      The web one-shot (`iris --ip`) gets the speedup automatically (txt2img gate). Remaining for the
      FULL (A) daemon model-reuse: re-enable IrisServer.generate's per-request IP forwarding (the
      web/main.c wiring that was reverted) — saves the extra ~12s reload (50s→38s) + keeps the model
      resident. B1 is now moot (B2 supersedes it). img2img sref still uses the per-block path (gates
      4230/4475 left as !tf->ip — txt2img is the web path). [UPDATE 2026-06-29: img2img fused-inject
      DONE too, gates 4384/4629 wired via ip_fused_prepare, parity corr 0.9997 — commit 6c4000e.]

  - **SREF QUALITATIVE LEVER TEST — CONTENT-LEAK BY REFERENCE TYPE (2026-06-29, owner-driven UI
    testing of the shipped champion).** First real-world tests of the shipped feature on owner-supplied
    references (the curated eval set is 5 fine-art paintings only — see SREF-EVAL-COVERAGE-GAP below).
    FINDINGS:
    • REFERENCE TYPE is the dominant factor in content leak, separate from ip-scale. SIMPLE styles —
      line-art (Churchill drawing), flat sticker (cyberfika) — transfer CLEANLY at the 0.38 champion
      strength: style adopted, zero subject bleed. COMPLEX PHOTO references with a strong foreground
      subject (flamenco dancer) LEAK that subject's content (fan/arm/tie appeared on the cat). The
      ~0.70 sref_score ceiling is a STYLE-vs-LEAK average; for low-content refs the usable strength is
      effectively higher because there's little content to leak. Practical guidance for the UI: simple/
      graphic refs → champion 0.38 constant is fine; busy photo refs → use the late schedule (below)
      and/or crop to the texture.
    • LEVER RANKING (4-way grid, baroque_portrait ruff-collar ref — a gently-leaking proxy for the
      flamenco; seed 7, 576px/4steps; /tmp/bq_{A,B,C,D}). Strength {0.38,0.25} × schedule {constant,late:0.5}:
        A 0.38 const → strong painterly style, SUBTLE collar leak
        B 0.25 const → weak style, no leak (style mostly gone — photographic cat)
        C 0.38 late  → strong painterly style, CLEAN (no leak)   ← winner
        D 0.25 late  → ~no style (photographic)
      CONCLUSION: the LATE injection schedule is a strictly better content-leak lever than lowering
      strength. Dropping ip-scale (A→B, C→D) sheds leak but also kills the style (style and leak fall
      together — they're the same KV-injection signal, consistent with the ~0.70 mechanism-bound
      ceiling). The late schedule (A→C) keeps FULL style intensity while letting image structure lock
      in during early steps before style injects, so leaked content recedes WITHOUT losing style. This
      is a QUALITATIVE/content-headroom win, NOT a ceiling break — matches DP-7 (schedule doesn't raise
      sref_score, but is a real content-headroom knob). The UI "Style timing → Late" selector exposes it.
    • CAVEAT: tested on baroque_portrait, not the exact flamenco — the flamenco source/features were
      not identifiable on disk (web only persists unlabeled feature .bin hashes in web/output/sref/, no
      source image). baroque leaks more gently than the flamenco fan, so the cleanest demo of these
      levers on the owner's exact image is to load it in the UI at 0.38 + late.

  - **SREF-EVAL-COVERAGE-GAP (BACKLOG, opened 2026-06-29): the eval reference set is 100% complex
    fine-art paintings — no simple-style refs, and no source-saved real-world refs.** The curated
    A/B set (`/Volumes/2TBSSD/sref_eval/refs` + eval_set.json, the basis of every sweep/champion verdict)
    is exactly 5 images: impressionism_landscape, cubism_stilllife, baroque_portrait, artnouveau,
    expressionism_portrait — ALL busy painterly fine-art. The reference types that transfer BEST in
    practice (line-art, flat vector, sticker, monochrome ink, logo/graphic) are ENTIRELY ABSENT. So
    every sref_score/content_leak number to date is measured on the HARD (leak-prone) end of the
    distribution and the eval never scores the easy wins or quantifies leak-severity-by-reference-type
    (the 2026-06-29 finding above is qualitative only). Consequence for future tuning/retrained models:
    an A/B that "wins" on the painting set could regress on simple styles undetected, and vice-versa.
    ACTION (cheap, do before the next champion comparison): curate a SECOND eval set of ~6–10 simple/
    graphic references (line-art, flat-color sticker, monochrome ink, bold vector, halftone/comic,
    woodcut) spanning the easy end; precompute hybrid + CSD features (hybrid_features.py / csd_mlx);
    add an `eval_set_simple.json` with matched neutral prompts (cat/landscape/portrait); report BOTH
    sets side-by-side in sref_sweep_eval so champion selection sees the full style-complexity spectrum.
    Also: persist the SOURCE image (not just the feature .bin) for web-uploaded refs so real-world
    leak cases (the flamenco) are reproducible. Harness already supports a second --pairs file; this is
    asset curation + one json, no code. Tie-in: this set also becomes the regression guard for any
    future base-model adapter (SREF-BASE-1) or multi-ref work.
    DONE (2026-06-29): simple set BUILT — `/Volumes/2TBSSD/sref_eval/eval_set_simple.json`, 8 refs in
    `refs_simple/` (6 generated anchors: lineart_ink, flat_sticker, mono_inkwash, bold_vector_poster,
    halftone_comic, woodcut + owner cyberfika sticker + Churchill coloring-page line-art), hybrid+CSD
    features in `refs_feat_hybrid_simple/` + `refs_feat_csd_simple/`. Building it immediately exposed
    SREF-CHAMPION-COLLAPSE below.

  - **🟣 SREF ARCHITECTURAL RETRAIN — CHARTER & NEXT STEPS (opened 2026-06-30, the active forward item).**
    After the diagnostic campaign PROVED the collapse is not loss-fixable (root cause + 6 failed loss
    experiments — see "SREF ADAPTER RETRAIN — DIAGNOSTIC-FIRST PLAN" and STEP-1A FINAL below; full design in
    `plans/sref-architecture-retrain.md`), the path to true --sref is ARCHITECTURAL. This is the charter.
    WHAT WE KNOW (the constraints any new architecture must respect):
      • The conditioning ENCODERS are not the bottleneck on their own: the SigLIP perceiver's output
        ip_embeds still DISCRIMINATE refs (cross-ref cos 0.407). The CSD FiLM path is rank-1 global
        (`out = q*(1+scale)+shift`, same scale/shift for all 128 tokens — model.py CSDImageProj) and DOES
        collapse (0.978), but siglip-only arms collapse too, so CSD is not the sole cause.
      • The INJECTION is the collapse site + the structural suspect: per-block ADDITIVE cross-attention
        K/V (to_k_ip/to_v_ip) at a learned low scale (~0.38) into the FROZEN DISTILLED base. to_v_ip
        learns rank ~6 → V near-constant; forcing it otherwise is gamed/overpowered (6 experiments).
      • THE CENTRAL CLUE: plain IN-CONTEXT conditioning (reference tokens placed IN the transformer
        sequence — the img2img path) DISCRIMINATES references fine (it's the shipped web path). The
        frozen base already knows how to use in-sequence tokens; the adapter's separate additive K/V
        side-channel is a mechanism the frozen base was NOT trained for, and its loss-minimum is a
        generic style push (collapse). → The most promising redesign makes the adapter produce
        IN-SEQUENCE conditioning, not a side-channel injection.
    CANDIDATE ARCHITECTURES (to be detailed + sequenced in the review; ranked by impact×tractability):
      1. **Learned in-context conditioning (LEADING).** Encode the reference (SigLIP/CSD) → a small set of
         learned TOKENS that are CONCATENATED into the sequence (like img2img in-context, which provably
         discriminates), trained to carry STYLE while a content term keeps the prompt's subject. Inherits
         the proven in-context discrimination; the training learns compression + style-isolation. Cheapest
         to validate against the proven mechanism.
      2. **Higher-capacity CSD conditioning.** Replace the rank-1 global FiLM with a real cross-attention /
         higher-rank modulation so the content-invariant CSD signal can carry style structure. Pairs with (1)
         or the existing perceiver.
      3. **Base-model adapter (highest impact, highest cost).** Train against the UNDISTILLED base (CFG,
         50 steps, more capacity) — the distilled base may be too rigid to steer without collapse. SREF-BASE-1
         showed the distilled adapter does NOT transfer → this is a fresh train + new CFG/inject code. Untested
         whether base also collapses; gate cheaply.
      4. **Different injection (AdaIN/stats or higher scale w/ content preservation).** Speculative; only if 1–3 stall.
    NEXT STEPS (sequenced) — UPDATED 2026-06-30 after the first experiment VALIDATED the direction AND
    surfaced a NO-RETRAIN shortcut:
      ✅ (a) ARCHITECTURE REVIEW done — mechanism confirmed (adapter=side-channel additive SDPA; in-context=
          in-sequence concat; no base support). FIRST EXPERIMENT (patch-shuffled refs through the EXISTING
          in-context path) → cross-ref corr 0.158 (vs adapter 0.93–0.99) with STYLE TRANSFER and NO
          COMPOSITION LEAK (churchill→line-art, woodcut→engraving, flat→sticker all excellent; cyberfika
          partial). ⇒ "in-sequence style-only" works TODAY with ZERO training by content-destroying the
          reference. plans/sref-architecture-retrain.md.
      (b) NO-RETRAIN SHORTCUT (do first): tune content-destruction (grid/blur/frequency/multi-shuffle) to fix
          graphic styles + maximize style fidelity; wire into the web "style" path (preprocess → in-context),
          upgrading the shipped default from style+composition to true style-only --sref with NO model change.
      (c) Validate on both eval sets + the discrimination gate (<0.90) + quality eyeball; if it beats plain
          in-context, SHIP it as --sref.
      (d) LATER quality lever — a LEARNED content-destroyer/style-token encoder (the original "learned
          in-context"), only if crude preprocessing plateaus; now de-risked. Base-model adapter only if that stalls.
    GUARDRAILS (carried): every train↔infer reimpl needs the parity fixture + prod-flag compile + `make mps`
    (AGENT protocol); cached-mode only (live-encode segfaults MLX, BUGS MLX-1); never train from cold storage
    (AGENT #6); promote ONLY on the discrimination gate, never sref_score; web stays on in-context meanwhile.
    Reusable from the diagnostic campaign: sref_kv_rank_audit.py(--ckpt), sref_ref_discrimination.py, the
    loss primitives (gated off), the simple-style eval set + features.

  - **🟣 PLUGGABLE CONDITIONING FRAMEWORK — train + serve LoRAs, in-sequence encoders, etc. as plugs
    (opened 2026-06-30). Full roadmap: `plans/pluggable-conditioning-framework.md`.** Turn iris.c into a
    FRAMEWORK for building/training/validating/serving pluggable conditioning against the frozen Flux base,
    composably. Born from the SREF journey: condition WITH the base's native mechanisms; validate every plug
    with a discrimination/eval gate. THE RAILS: (1) content-destruction style path — SHIPPED; (2) learned
    IN-SEQUENCE encoders (reference → compact style/subject tokens in the sequence; trained once, any
    reference; style→subject→face) — the SREF Phase-2 milestone; (3) LoRAs — WEIGHT-space deltas, today
    iris_lora.c only LOADS external ones (BFL/Kohya/Diffusers/XLabs), **training is NEW**; (4) frontier:
    reference→LoRA hypernetwork (instant custom LoRA). Rails compose (weight-space ⟂ activation-space; stack
    a character LoRA + an instant style reference). NOT "a better LoRA" — the in-sequence rail is the
    standard COMPLEMENT (instant, no per-concept training).
    PIECES — TRAINING ("build a plug"): generalize train_ip_adapter.py (already a frozen-Flux loop w/
    precompute caches + flow-matching loss + EMA + ckpt) into a PLUGGABLE trainable-module interface; a LoRA
    TRAINING pipeline (Phase 1, foundational — train low-rank deltas w/ the diffusion objective, export to
    the BFL/Kohya format the engine ALREADY loads → train→serve closes immediately); in-sequence encoder
    training (Phase 2); the discrimination gate as the shared validation. SERVING ("use a plug"): a unified
    conditioning-plug interface in the C engine (LoRA = iris_lora.c half exists; in-sequence = produce tokens
    → concat into the sequence); compose multiple plugs w/ per-plug strength; web/CLI surface.
    SEQUENCING: Phase 1 LoRA TRAINING pipeline (lowest risk, reuses infra, lets the owner build custom
    LoRAs in-house instead of only consuming external) → Phase 2 generalized trainer + learned in-sequence
    STYLE encoder (de-risked; beats the shuffle on graphic styles + saveable codes) → Phase 3 subject/face
    encoders + serving composability → Phase 4 hypernetwork. GUARDRAILS (carried): train↔infer parity +
    prod-flag compile + make mps for any C reimpl; cached-mode only; never train from cold; promote only on
    the gate. This subsumes the SREF "learned encoder" next-step into a broader framework.
    CROSS-MODEL PORTABILITY (owner plan: 4B distilled → 4B base 50-step → future 9B base): Rail 1
    (content-destruction) is WEIGHT-LESS → free on every variant. Rail 2 (in-sequence encoders): recipe
    ports, WEIGHTS don't (4B-distilled→base = same dim 3072 but diff attention → fine-tune; 4B→9B = dim
    3072→4096 → new encoder). Rail 3 (LoRAs): pipeline ports, weights per-model (iris_lora.c already loads
    4B+9B). UPSIDE: base (CFG, more capacity) may make trained rails work BETTER — distilled rigidity was
    part of the collapse. DESIGN PRINCIPLE (bake in now): build MODEL-AWARE — dims-from-config (no
    hardcoding) + a CFG-CAPABLE training path (base needs null+cond dual-pass; trainer currently hardcodes
    guidance=None, base training = net-new code). Phase 1 LoRA trainer supports distilled (no CFG) AND base
    (CFG) from day one so the port is cheap.

  - **✅ LORA-TRAIN-1 (2026-06-30) — Phase 1 LoRA TRAINING pipeline WORKING end-to-end (train→export→serve
    closed). First targeted-style LoRA learned a sharp, intentional style from a curated pool subset.**
    Built `train/lora/` (lora.py LoRALinear + double-block injection; train_step.py checkpointed
    flux_forward_lora w/ parity corr 1.0 vs flux.transformer; export.py → Diffusers safetensors iris_lora.c
    already loads; train_lora.py reusing the IP-adapter's VALIDATED data path — make_prefetch_loader +
    VAE-Q1 _bn_pack_latents — so LoRA trains in C's exact latent space). REGULAR-PIPELINE LESSONS applied:
    module-aware gradient checkpointing (`mlx.nn.utils.checkpoint`, NOT bare `mx.checkpoint` which zeroes
    captured-param grads — verified grad 0.0→113.0); MLX memory limit (BUGS MLX-2 wedge); cached-mode only;
    hot-SSD shards only; bucket [512,512] pinned to square precompute; EMA over trainable_parameters() only
    (NOT update_ema = whole frozen 4B) w/ decay-warmup. GUARD: `debug/test_lora.c` train↔infer parity
    fixture (gen_lora_fixture.py golden = MLX LoRALinear delta; C lora_apply reproduces corr 1.000000
    max_abs 0.00000) in `make test-unit` — green.
    STYLE DEMO: curated densest CSD cluster (N=250, curate_style_subset.py) → 300-step rank-16 train,
    loss 0.58→0.09; on-vs-off corr 0.575 (strong, content-preserving). Eyeballed: "a cat sitting on a
    chair" goes from bright clean stock-photo → coherent desaturated warm-earth, heavy-grain, weathered/
    aged aesthetic; SUBJECT preserved (still a tabby on a chair). A real, deliberate style — NOT the
    reference-inert collapse the IP-adapter suffered. Validates the framework thesis: weight-space LoRA
    deltas trained with the diffusion objective give a clean, controllable conditioning rail. Distilled 4B
    only so far; base (CFG dual-pass) + single-block coverage + pluggable-module generalization are the
    open Phase-1/2 follow-ups (see PLUGGABLE CONDITIONING FRAMEWORK above).

  - **🧭 DATA-SELECTION PRINCIPLE (2026-07-01) — DECORRELATE the conditioned attribute from content;
    COVER its range, don't cluster it. A genuinely new lever (data axis) for training ANY conditioning
    module on the frozen base — LoRA, IP-adapter, in-sequence encoder — distinct from every prior
    loss/architecture-axis fix. Emerged from the LoRA arc (LORA-TRAIN-2/3/4).**
    THE TRAP: selecting training data by style-cluster tightness (the obvious "coherent style" choice,
    e.g. CSD kNN) silently CONFOUNDS the target attribute (style) with CONTENT — CSD-tight clusters are
    also content-tight (LORA-TRAIN-3: the densest far cluster was all fantasy portraits). A module trained
    on it entangles style with subject → won't transfer / collapses off-distribution.
    THE RULE: a conditioning module only learns to USE its conditioning if, in the training data, the
    conditioned attribute VARIES INDEPENDENTLY of content AND the target can't be predicted without it.
    Cluster-by-similarity violates both. Instead:
      • single-attribute module (a LoRA = one baked style): DECORRELATE — hold the LOOK coherent (palette/
        tone/contrast/grain — a cheap content-invariant style coordinate) while forcing content DIVERSITY
        (penalise csd_coh). curate_lookstyle_subset.py; validated LORA-TRAIN-4 (transfers to any subject,
        scales to bold w/o collapse).
      • multi-attribute conditioning module (IP-adapter = reads any reference): DECORRELATE **and** COVER —
        stratify the reference set across the FULL look-space so no constant output can win, and build
        reference↔target pairs that share look but differ in content.
    WOULD THIS HAVE HELPED THE IP-ADAPTER COLLAPSE (SREF-CHAMPION-COLLAPSE — reference-INERT, near-constant
    injection, to_v_ip rank ~6)? PLAUSIBLY, and it is the UNTRIED axis: all 6 prior collapse-fixes were
    LOSS-side; data selection was never varied (SREF pairs were CSD-neighbour-based; the eval was the
    homogeneous all-painterly set that MASKED the collapse). Reference-inertness = the model produced a
    constant "average" because (a) the references clustered in one narrow region (warm/painterly) so a
    constant scored well → no gradient to discriminate, and (b) content-confounded pairs let it shortcut.
    The look descriptor directly attacks both — content-invariant coordinate to STRATIFY references
    (coverage) + a way to build content-DECORRELATED pairs — and it also builds the right EVAL (vary look,
    hold content = debug/sref_ref_discrimination.py), which the painterly eval was not.
    HONEST CAVEAT: the collapse was also characterised as STRUCTURAL (rank ~6). If that rank limit is truly
    architectural, data won't lift it — BUT rank collapse is usually DOWNSTREAM of a training signal that
    never demanded discrimination, so look-stratified + content-decorrelated data + a discrimination reward
    is a real shot at the ROOT, not a retune. PROPOSED TEST (dormant behind the SREF adapter re-enable):
    rebuild the SREF pair dataset with look-stratified reference coverage + content-decorrelated pairs, add
    a reference-discrimination term, re-measure cross-ref corr on the discrimination gate. See
    [[sref_platform_strategy]] / SREF-CHAMPION-COLLAPSE.

  - **❌ SREF-ROPE-PHASE3 (2026-07-06) — reference-KV reuse is a NO-GO on this stack. Viability probe
    (asymmetric mask, the prerequisite) FAILS both gates: it degrades style quality AND regresses perf.
    Phase 3 abandoned; probe code reverted (throwaway measurement), negative result logged here.**
    Plan Phase 3 (plans/sref-rope-band-control.md) proposed masking reference-token queries off the noisy
    target image (so ref K/V becomes step-invariant and reusable across the 4 steps → ~20-30% wall-clock).
    The plan mandated validating the MASK ALONE first ("changes output; run the gate + golden diff"); I
    implemented it behind an opt-in `--sref-kv-reuse` flag (bf16 fused kernel gained a query-dependent mask:
    ref queries [ref_start,seq) skip target keys [txt_seq,ref_start); per-call query ranges for single vs
    double blocks; default off = bit-identical) and measured on the 8-ref gate atop shf0.0/slf1.5.
    RESULT (8-ref gate, "a cat on a chair", seed 42, 512px; data: debug/sref_p3_mask_probe_2026-07-06.jsonl):
      • mask OFF (= P2 γ=1 baseline): max_cross 0.394  style_adh 0.354  copy_corr 0.162   ~4 s/gen (MPSGraph SDPA)
      • mask ON  (--sref-kv-reuse):   max_cross 0.615  style_adh 0.255  copy_corr 0.141   ~148 s/gen (custom kernel)
    TWO INDEPENDENT FAILURES:
      1. QUALITY: the mask DROPS CSD style adherence 28% (0.354→0.255) and WORSENS reference discrimination
         (max_cross 0.394→0.615). Decoupling ref queries from the target makes the reference tokens carry a
         weaker, less target-adapted style signal → less adherence and more similar (less discriminating)
         outputs. This is exactly the quality risk the plan flagged as "unvalidated on 4-step distilled",
         now CONFIRMED negative. (Output is finite/coherent — the -INF masking doesn't NaN — it's just worse.)
      2. PERF: the mask has NO MPSGraph SDPA support, so it forces the custom fused kernel — ~40× SLOWER
         (148 s vs 4 s/gen). Phase 3's whole point is a SPEEDUP; the prerequisite makes attention dramatically
         slower. K/V caching cannot recover a 40× attention penalty; the only path to a net win would be to
         port the mask INTO MPSGraph SDPA (an additive mask tensor) — a large, risky effort — and even then
         the quality loss (failure 1) stands. Not worth it.
    VERDICT: abandon reference-KV reuse. The quality win is Phases 1-2 (band-control + strength); the
    distilled path is already fast (4 steps), so the perf-only Phase 3 was low-value and both gates killed it.
    (Decided with the user: "viability probe first" — the probe did its job. Committed nothing but this log +
    the probe data; all probe code reverted so the tree stays at Phase-2.)

  - **✅ SREF-ROPE-PHASE2 (2026-07-06) — `--sref-strength` γ strength-bias shipped; PASSES acceptance
    (γ monotonically trades style strength, discrimination intact, default γ=1 a verified no-op).**
    Implements Phase 2 of plans/sref-rope-band-control.md. Additive log(γ) on the reference-token KEY
    columns of attention (OminiControl B(γ)): γ<1 weakens, γ>1 amplifies the reference's pull. Layered on
    band-control; dials style strength independently of the RoPE bands. IMPLEMENTATION (GPU/Metal path):
    a file-scope bias in iris_metal.m (`iris_metal_set_attention_bias`, set per-generation by the flux
    ref-aware forward via `sref_set_strength_bias`, cleared before VAE decode) is read by the fused
    custom kernels (attention_fused / attention_fused_bf16 gained buffer 9=ref_start, 10=log(γ)); when the
    bias is non-zero the fused wrappers BYPASS the MPSGraph SDPA fast path (which has no additive mask) and
    use the custom kernel. So γ=1 → bias 0 → untouched MPSGraph path (bit-identical); γ≠1 → custom kernel
    with bias (still GPU). References are the trailing keys, so ref_start = total_seq - ref_seq is one
    constant for all flux block attentions; VAE/zImage/ip-adapter run while the bias is 0 and are
    unaffected. Chose a global over threading γ through ~13 attention call sites + joint_attention_bf16
    (which lacks a tf pointer). Plumbing: CLI `--sref-strength` → daemon `sref_strength` JSON key → web
    `IRIS_SREF_GAMMA` (default 1.0=off, opt-in) → iris_params.sref_strength → iris_transformer_set_sref_strength
    (clamps γ to [1e-3, 8]). CPU-only builds (BLAS/generic): γ is a documented no-op (bias applied only in
    the Metal kernels); default-off keeps them bit-identical (both compile clean).
    RESULT (8-ref gate, "a cat on a chair", seed 42, 512px, atop band-control shf0.0/slf1.5):
      • γ=0.5:  max_cross 0.383  style_adh 0.322  copy_corr 0.132
      • γ=1.0:  max_cross 0.394  style_adh 0.354  copy_corr 0.162  ← EXACT match to Phase-1 shf0.0/slf1.5 (no-op ✓)
      • γ=2.0:  max_cross 0.522  style_adh 0.380  copy_corr 0.204
      • γ=4.0:  max_cross 0.395  style_adh 0.433  copy_corr 0.236
    ACCEPTANCE MET: style_adh rises monotonically with γ (0.322→0.433, +34% from γ=0.5→4) while discrimination
    holds (max_cross < 0.53 throughout, well under 0.90). copy_corr rises with γ too (the expected strength
    trade-off — stronger γ pulls everything toward the reference). γ is a COMPLEMENTARY strength knob to slf:
    γ=4 atop slf1.5 reaches style_adh 0.433 (vs 0.354 at γ=1), approaching the Phase-1 shf0.6/slf1.5 cell
    (0.476) by a different lever. Sweep data: debug/sref_p2_strength_sweep_2026-07-06.jsonl.
    VERIFIED: make mps clean (no warnings); make test-unit 28/28; BLAS+generic compile clean; CLI end-to-end
    (γ=2 vs γ=1 same seed/ref → outputs differ corr 0.945, no corruption → bias engages on the shipped bf16
    path + VAE-clear works). GOTCHA (logged): the gate's CSD encoder is MLX → must run via train/.venv/bin/python
    (web/venv lacks mlx); and `make blas`/`make generic` overwrite the `iris` binary (all targets emit `iris`) —
    do not build other backends while an MPS sweep using ./iris is running. Web default γ=1 (off, opt-in via
    IRIS_SREF_GAMMA) until a value is chosen. NEXT: Phase 3 (reference-KV reuse; perf) per the plan.
    KNOWN LIMITATION: at very large seq the custom kernel exceeds threadgroup memory (~7680 keys) → the bias is
    dropped and the gen falls back (unbiased) at that resolution; fine at ≤~768px.

  - **✅ SREF-ROPE-PHASE1 (2026-07-05) — RoPE band-control shipped for the in-context style rail;
    PASSES the gate and DOMINATES patch-shuffle. Confirms "Untwisting RoPE" on our 4-step DISTILLED model
    (first test of that mechanism off 50-step Flux).** Implements shortlist #1 from SREF-ARCH-RESEARCH
    (plan: plans/sref-rope-band-control.md). Scales the K-side reference-token RoPE H/W bands in
    SINGLE-STREAM blocks only: s(d) = shf + (slf-shf)*(d/15)^2 baked into a K-only combined table (per-pair
    scalar commutes with the 2x2 rotation → exact, zero per-step cost). CLI `--sref-shf`/`--sref-slf`
    (shf∈[0,1] high-freq attenuation, slf>=1 low-freq boost; 1.0/1.0 = OFF, default). Default-off is
    BIT-IDENTICAL (corr 1.000000, maxabs 0); make test-unit green; BLAS+generic compile-clean; f32+bf16
    paths both wired. Gate: debug/sref_rope_gate.py (8 refs the adapter collapsed on; discrimination +
    CSD style adherence + copy_corr). Sweep data: debug/sref_rope_sweep_2026-07-05.jsonl.
    RESULT (8-ref gate, "a cat on a chair", seed 42, 512px):
      • baseline B raw in-context:  max_cross 0.365  style_adh 0.496  copy_corr 0.257
      • baseline A patch-shuffle6 (SHIPPED): max_cross 0.791  style_adh 0.219  copy_corr 0.076
      • shf0.0 slf1.0 (full atten):  max_cross 0.646  style_adh 0.248  copy_corr 0.117
      • shf0.0 slf1.5 (RECOMMENDED):  max_cross 0.394  style_adh 0.354  copy_corr 0.162
      • shf0.6 slf1.5 (strong style): max_cross 0.290  style_adh 0.476  copy_corr 0.346
    ACCEPTANCE MET: multiple cells satisfy max_cross<0.90 AND copy_corr<B AND style_adh>=A; e.g.
    shf0.0/slf1.5 (style 0.354=1.6x patch-shuffle, copy 0.162<0.257, discrim 0.394). Band-control STRICTLY
    beats patch-shuffle: +60-120% style adherence AND ~2x better reference discrimination (max_cross
    0.29-0.39 vs 0.79). MECHANISM CONFIRMED on distilled: full high-freq attenuation (shf=0.0) suppresses
    positional COPYING — the woodcut-owl reference correctly yields a CAT at shf=0.0, but low-freq boost
    WITHOUT high-freq attenuation (shf=1/slf=1.5) COPIES the owl. So shf is the copy-killer, slf the
    style-strength knob. CAVEAT: copy_corr conflates composition-copy (bad) with palette-adoption (good) —
    strong style raises it; rely on style_adh + discrimination + visual (no content copy) for tuning.
    BUG FIXED mid-sweep: setter guard mapped a legitimate shf=0.0 to off; now shf=0.0 (full attenuation)
    is usable (daemon params + IRIS_PARAMS_DEFAULT default to 1.0).
    ✅ WEB-WIRING DONE (2026-07-05): band-control is now the DEFAULT web style rail. The web server talks to
    the resident `iris --server` daemon over JSON (not CLI flags), and the daemon previously hardcoded
    band-control OFF ("daemon does not expose it"), so this needed BOTH sides: (1) main.c daemon now parses
    `sref_shf`/`sref_slf` JSON keys (default 1.0 = off) → server_job_t → iris_params (was hardcoded 1.0f);
    (2) web/server.py adds IRIS_SREF_SHF (default 0.0) / IRIS_SREF_SLF (default 1.5), sets them on a job only
    when a STYLE-mode reference is present (composition refs untouched), and forwards them in the daemon
    request; the patch-shuffle (IRIS_SREF_SHUFFLE_GRID) default flipped 6→0 (now OPT-IN, composes with bands).
    VERIFIED: make mps relinked; make test-unit 28/28; DAEMON END-TO-END — two gens in one warm session, same
    seed/prompt/ref, one with sref_shf=0.0/slf=1.5 vs one with no keys → outputs DIFFER (band-control engages
    via the daemon path) while the no-keys path is byte-identical to pre-change (params 1.0/1.0 by
    construction); web server launches (port bound, GET / → 200), config defaults confirmed via import.
    NEXT: Phase 2 (strength bias gamma) + Phase 3 (reference-KV reuse) per the plan.

  - **📚 SREF-ARCH-RESEARCH (2026-07-05) — deep research on SREF architecture options COMPLETE; ranked
    shortlist in `plans/sref-architecture-options.md`.** 24 sources, 118 claims, adversarial verification +
    6 targeted source-verification agents. HEADLINES: (1) the SEQUENCE PATH is the only conditioning
    mechanism validated on Flux-family DiTs (OminiControl/DreamO/USO/Kontext all concat reference tokens +
    small LoRA; NO validated K/V-injection adapter on any DiT) — matches our empirical record exactly.
    (2) Published triangulation of our collapse: encoder features entangle style+semantics and
    reconstruction training over-relies on the reference (InstantStyle/DEADiff, verified); distilled
    few-step models COMMIT STRUCTURE IN THE FIRST STEP (2503.10637, verified) → a weak side-channel on a
    4-step model has nothing to steer, so "ignore the reference" is the easy optimum. (3) Control LoRAs
    TRANSFER base↔distilled WITHOUT retraining (verified, SDXL) → train conditioning on the 4B base when
    ported, ship on distilled; hybrid first-step-from-base sampling restores diversity. RANKED SHORTLIST:
    #1 RoPE frequency-band control on reference keys (Untwisting RoPE, 2602.05013 — copying is POSITIONAL/
    high-freq-RoPE; attenuate high + amplify low bands, reference keys only, single-stream blocks only;
    training-free, C-implementable) + OminiControl2 reference-KV reuse across steps + attention-bias
    strength — a zero-training upgrade of the SHIPPED in-context rail (replaces crude patch-shuffle).
    #2 USO-style learned in-sequence style tokens (SigLIP→projector→192 tokens with text RoPE ids;
    Stage-1 trains projector ONLY with DiT frozen → fits 32GB MLX, consumes our cached SigLIP; verified in
    paper+code). #3 base-training+transfer as enabler. #4 i2L hypernetwork→instant LoRA (verified on
    FLUX.2-klein-base-4B AND Z-Image — our exact backbones; output = plain LoRA our loader already serves;
    8×A100×7d at their scale → Phase-4; cheap interim = CSD-retrieval over our own style-LoRA library).
    AVOID (verified): K/V side-channel adapters; naive shared-attention on DiT (collapses); RB-Modulation
    (it is per-step test-time optimization through the CSD ViT, NOT learnable modulation — un-C-implementable;
    and "768-d CSD alone is sufficient" was REFUTED); CSGO-scale triplet pipelines (8×H800, 210k triplets).
    Also: "style-consistent, content-diverse" training data (i2L MegaStyle) independently validates our
    DATA-SELECTION PRINCIPLE. Full citations in the plan doc. IMPLEMENTATION PLAN for shortlist #1 (RoPE band-control +
    strength bias + reference-KV reuse, phased, with code anchors, gate script, sweep protocol and
    acceptance criteria): `plans/sref-rope-band-control.md`.

  - **🔬 SREF-DATA-TEST (2026-07-01, planned) — concrete recipe + feasibility to test the DATA-SELECTION
    PRINCIPLE on the IP-adapter reference-inert collapse. Pipeline mapped; the intervention is a
    SWAPPABLE pairing DB + reused caches (no trainer changes, no re-precompute).**
    PIPELINE MAP (code investigation, file:line):
      • PAIRING is a swappable SQLite DB. `neighbors.sqlite` schema: `neighbors(rec_id PK, neighbor_ids
        TEXT[json], neighbor_cos TEXT[json])` + `meta(k=10, dedup_cos=0.95)`, 109 253 rows = whole pool.
        Built by `train/scripts/style_neighbors.py` (CSD kNN, cos≥0.6 gate `_STYLE_NBR_MIN_COS`, excl
        cos>0.95). Loader `train/ip_adapter/dataset.py:539-670` reads it when `data.style_neighbors_db`
        is set: per record picks one same-style neighbor and loads ITS cond features as `style_ref`.
      • BASELINE trains WITHOUT it: `stage1_512px.yaml` sets no style_neighbors_db → arbitrary cross-ref
        (`cross_ref_prob 0.5`, swap SigLIP across batch) + `patch_shuffle_prob 0.5` + Gram `style_loss
        0.05`. So the shipped adapter learned from RANDOM references — a plausible root of the constant
        "average" injection (nothing forces per-reference discrimination).
      • Loader yields per step: target image/vae/qwen3/siglip + `style_ref` = neighbor's SigLIP[729,1152]
        / CSD[768] / hybrid[730,1152]; `data.cond_mode` ∈ siglip|csd|hybrid (champion = hybrid).
      • CACHES 100% REUSED from baseline_pool_hot — vae v_2232c1, qwen3 v_059443, siglip v_336c6e (all
        verified present) → a short standalone train is ~1 h (stage1 ~3.8 s/step), NOT the ~day flywheel.
        Entry: `python train/train_ip_adapter.py --config <cfg>` (override cache dirs + shard_path).
      • DISCRIMINATION-REWARD terms already EXIST but default OFF (`train/ip_adapter/loss.py:104-155`):
        `style_repulsion_loss` (repulsion_weight; output-space style-stats hinge), `vproj_decorr_loss`
        (vproj_decorr_weight; per-block V-cosine hinge), `vproj_rank_penalty` (rank_weight; anti low-rank
        to_v_ip). Leaving them OFF isolates the DATA lever.
      • GATE ready: `debug/sref_ref_discrimination.py` — vary ref, hold prompt/seed/scale, pixel-corr;
        PASS = max_cross < 0.90 AND mean_base < 0.95. Needs `--bundle` + ≥3 ref `.bin`
        (csd_features.py / siglip_features.py). Known collapse: max_cross ≥ 0.984.
    EXPERIMENT (data-only A/B):
      1. Build `neighbors_look.sqlite` (same schema) — for each record, neighbours = LOOK-similar
         (curate_lookstyle 17-d look vec) but CONTENT-different (penalise cached-CSD cosine): look-shared,
         subject-varied pairs. Optionally stratify the training set across look-space (coverage).
      2. A/B short trains from the SAME init: control = arbitrary cross-ref (as-is); treatment =
         `style_neighbors_db: neighbors_look.sqlite`. Discrimination losses OFF. Gate both on 5 diverse
         reference styles; compare max_cross to 0.984.
      3. If data-only is insufficient, add `repulsion_weight`/`vproj_decorr_weight` (data + loss).
    FEASIBILITY — RESOLVED (2026-07-01): champion FOUND at
    `/Volumes/2TBSSD/sref_eval/clean_concentrate_leak/` — `bundle/adapter_weights.safetensors` (exported,
    what the gate/web load) + `ckpt/{best,step_0003000}.safetensors` (training ckpts) + `config.yaml`.
    Champion recipe (config.yaml): cond_mode HYBRID, `style_neighbors_db = neighbors.sqlite` (CSD-based!),
    `leak_loss_weight 0.5`, `cross_ref_prob 0.5`, `style_loss_weight 0.05`, `ip_scale_init 0.5`,
    num_image_tokens 256, 3000 steps from scratch, allowlist `pool_top25.json` (27 313 rec_ids),
    records_per_shard_visit 135. KEY: the champion ALREADY used CSD-neighbour pairing + content-leak loss
    and STILL collapsed (gate max_cross ≥ 0.984) → it IS the matched CONTROL. So the A/B needs only the
    TREATMENT run: identical recipe, swap `neighbors.sqlite → neighbors_look.sqlite` (look-similar /
    content-different pairs), 3000 steps from scratch, then gate and compare to 0.984. No warm-start
    needed. ~3.2 h for the single treatment run (3.8 s/step × 3000).
    MEMORY (2026-07-03, hard-won) — the treatment run repeatedly thrashed on the 32 GB dev machine; the
    champion trained fine because its machine had more headroom. THREE distinct causes, each fixed
    non-confoundingly (infra only — no effect on trained weights or the A/B):
      1. This hybrid/256-token/correct_forward_q config peaks at 24.56 GB — ~4 GB ABOVE the standard
         training's documented 20.44 GB (TRAIN-7) that `mlx_memory_pct 0.6` (≈19 GB) was tuned for. 0.6
         sits BELOW this config's working set → MLX reclaim-churn wedge (BUGS MLX-2). FIX: raise to 0.80.
      2. A one-time +4 GB persistent jump at every step%500 — NOT the checkpoint (which streams safely via
         `_save_safetensors_streaming`), but the EMA-DRIFT TELEMETRY (`dict(_flatten(adapter.parameters()))`
         + float32 casts, line ~2242) pushing peak 24.56→28.43. FIX: `training.log_ema_drift: false`
         (new flag, gates the block; default true keeps champion behaviour).
      3. Python-side anonymous creep ~15 MB/step (loader numpy arrays in MLX/numpy reference cycles),
         invisible to MLX active but growing swap 13→20 GB over ~1700 steps → thrash. FIX: `gc.collect()`
         every 20 steps in the loop (slowed to ~4 MB/step; not fully eliminated — a fresh reboot clears the
         baseline swap so the residual creep has room).
    Runs 3/4/5 pushed the thrash point 300→400→1700 as fixes landed; run5 (gc + telemetry-still-on)
    reached step 1700 before tipping. Deferring checkpointing was a MISTAKE (no salvageable ckpt) — keep
    checkpointing ON (streaming save is memory-safe). Final config = `train/configs/sref_look_treatment.yaml`
    (now: checkpoint_every 500, mlx_memory_pct 0.80, prefetch_batches 6, log_ema_drift false). Run on a
    FRESH reboot (clears accumulated swap). This memory profile is the "what's different from the past
    MLX-2 fix": 0.6 was calibrated to the LIGHTER standard workload, not this heavier SREF config.
    RESULT 2026-07-04 — ❌ HYPOTHESIS FALSIFIED. Treatment (look-stratified/content-decorrelated
    neighbors_look.sqlite, full 3000-step matched recipe, EMA) vs champion control, identical 8-ref gate
    (5 hybrid WikiArt paintings + churchill_lineart/flat_sticker/woodcut, scale 0.38, seed 42):
      • TREATMENT: cross-ref corr mean 0.987, MAX 0.997; vs-baseline 0.446 → FAIL: MODE COLLAPSE.
      • CONTROL (champion): cross-ref corr mean 0.984, MAX 0.996; vs-baseline 0.380 → FAIL: MODE COLLAPSE.
    Treatment collapsed exactly as hard as the champion (Δmax +0.001 = noise, marginally WORSE). Both apply
    a strong (vs-baseline ~0.4, NOT inert) but reference-INDEPENDENT transform. Look-based content-
    decorrelated PAIRING does NOT move the collapse → the DATA-selection axis is RULED OUT for the
    IP-adapter reference-inertness — the OPPOSITE of its effect on the LoRA (LORA-TRAIN-4, where content-
    agnostic data FIXED content-entanglement). CONCLUSION: confirms the STRUCTURAL cause (SREF-CHAMPION-
    COLLAPSE, to_v_ip rank ~6) — the perceiver+ip_k/ip_v mechanism collapses diverse references to a
    near-constant K/V regardless of how training pairs are chosen; not a data or loss artifact. The
    DATA-SELECTION PRINCIPLE's honest caveat ("if the rank-6 limit is truly architectural, data won't lift
    it") is CONFIRMED. Remaining levers if the adapter is revisited are ARCHITECTURAL (raise to_v_ip rank /
    redesign injection / the in-sequence-encoder rail), NOT data or the already-tried loss terms.
    Artifacts: treatment bundle `/Volumes/2TBSSD/sref_eval/sref_look_test/bundle`, ckpts step_2500/3000+best.
    CAFFEINATE (2026-07-04, root-cause of the 'thrash'): the standalone trainer run under plain
    nohup was IDLE-THROTTLED by macOS when the operator stepped away — step rate swung 0.20→0.02
    with user attendance, NOT purely memory. The slow throttled steps gave swap time to accumulate,
    which masqueraded as the MLX wedge. FIX: run under `caffeinate -dimsu -w <pid>` (or launch via
    start_pipeline.sh, which already does this). The memory fixes (telemetry-off peak 24.56, gc,
    mlx 0.80) are still needed, but caffeinate is what makes the run finish at ~0.2 steps/s.
    STATUS 2026-07-01 — READY TO RUN, blocked on a fresh machine. Look-pairing DB built
    (`/Volumes/2TBSSD/sref_eval/neighbors_look.sqlite`, 27,313 recs, 100% coverage, look-cos 0.71-0.81
    from different shards). Treatment config committed: `train/configs/sref_look_treatment.yaml` (champion
    recipe, only style_neighbors_db + checkpoint_dir changed). Builder tool: `train/lora/build_look_neighbors.py`.
    OPERATIONAL FINDING (memory): the 3000-step run THRASHED twice on the dev machine — MLX working set
    ~23 GB peak (leaner than the champion's 27 GB) but a long prior session had left ~18 GB of macOS swap,
    leaving < 1 GB headroom on 32 GB → forward-pass paging → rate collapsed 0.19→0.02 steps/s around step
    ~400. Root: session-accumulated swap, not the recipe (champion trained fine on a FRESH machine).
    Gradient checkpointing is a NO-OP for this path (code: only helps the TRAIN-6 block-injection forward,
    not correct_forward_q). FIX = reboot to clear swap, then run on the clean machine (23 GB fits ~28 GB
    fresh headroom). RELAUNCH after reboot:
      `train/.venv/bin/python train/train_ip_adapter.py --config train/configs/sref_look_treatment.yaml`
    then export `train/export/export_adapter.py --checkpoint .../sref_look_test/ckpt/best.safetensors
    --output .../sref_look_test/bundle --use-ema --perceiver-heads 16 --validate`, and gate BOTH the
    treatment bundle and the champion (`clean_concentrate_leak/bundle`) with
    `debug/sref_ref_discrimination.py --feat` (5 hybrid paintings + churchill_lineart/flat_sticker/woodcut
    from refs_feat_hybrid[_simple]) at `--scale 0.38 --seed 42`; compare max cross-ref corr to 0.984.

  - **✅ LORA-TRAIN-4 (2026-07-01) — CONTENT-AGNOSTIC curation FIXES the transfer/collapse failure of
    LORA-TRAIN-3. A "same look, varied subject" cluster trains a style that transfers to ANY subject and
    scales to bold WITHOUT collapsing off-distribution content.**
    Root cause of LORA-TRAIN-3: the CSD-far cluster was coherent BY being subject-specific (fantasy
    portraits) → the LoRA entangled style with content → worked on portraits, collapsed on cats.
    Fix — `train/lora/curate_lookstyle_subset.py`: decode a sample of pool images → a ~17-d content-
    AGNOSTIC LOOK vector (RGB/HSV/luminance stats, warmth, colorfulness, grain). Cluster tight in
    look-space but score = look_coh − μ·csd_coh, penalising cached-CSD coherence so the cluster shares a
    LOOK while spanning diverse SUBJECTS. Selected cluster: look_coh 0.961, csd_coh −0.010 (vs the far
    cluster's ~0.75), ALL 22 shards — a high-key/bright/light-palette look across a cathedral, a glass cup,
    a cyber-portrait, a bride, a pencil sketch, a white tiger (6 unrelated subjects, one look). Trained
    full-coverage (80 modules, 300 steps, loss→0.08 — cleanest fit of the three).
    RESULT: on-vs-off @scale1.0 — cat 0.847, portrait 0.900 (BALANCED; far-LoRA was portrait-biased
    0.819 vs cat-inert 0.899). @scale2.5 — cat 0.653 and portrait 0.727 with BOTH subjects fully intact
    and coherently restyled (warmer/richer/polished "premium-photo" look), whereas the far-LoRA at 2.5
    already smeared the cat and fully collapsed it by 4.0. So content-agnostic curation delivers the
    property CSD-distance did not: a TRANSFERABLE style, controllable to bold without content collapse.
    CAVEAT: the specific look selected by the densest+diverse objective is high-key/warm-polish — coherent
    and controllable but tasteful rather than dramatic; the distilled 4-step prior still caps how far a
    default-scale restyle can go (bold = use scale ~2-2.5, or move to the base/CFG model). The METHOD is
    the deliverable: to target a different look, bias curate_lookstyle_subset's look descriptor (e.g.
    require high grain / dark / saturated). Artifacts (lora_look_v1, lora_look_subset, log) archived to
    cold; tool + config committed.

  - **⚠️ LORA-TRAIN-3 (2026-07-01) — "curate a cluster FAR from the base prior → bolder style" FALSIFIED.
    Far-in-CSD-space does NOT yield a bolder-at-default LoRA; it yielded a WEAKER, content-entangled one.**
    Followed up LORA-TRAIN-2's hypothesis (near-prior cluster = subtle → so pick a FAR one). Built
    `train/lora/curate_far_subset.py`: CSD-encode 12 base-model generations → prior centroid (the model's
    own style signature; CSD is content-invariant), then pick the densest pool cluster maximising
    distance-from-prior subject to a coherence floor. It cleanly selected a genuinely far, coherent cluster:
    prior_cos −0.113 vs the photographic v2 cluster's +0.037 (pool range −0.314..+0.556), intra-cos 0.746,
    18/22 shards — visually a fantasy digital-illustration / pop-surrealism PORTRAIT style (mermaids,
    sea-elves, jewel-toned painterly faces). Trained full-coverage (80 modules, 300 steps, loss→0.17, EMA).
    RESULT (honest, counterintuitive): on-vs-off corr @scale1.0 = 0.899 — WEAKER than the near-prior v2's
    0.847. On "a cat on a chair" @1.0 the output is ~photographic; on a matching "portrait of a woman" @1.0
    corr 0.819 (still mostly photographic). Diagnostic (push scale): portrait @3.0 DOES become a painterly
    digital-illustration portrait (style IS learned) but cat @4.0 collapses to warm painterly smears (no cat).
    LESSONS: (a) CSD-distance of the training IMAGES predicts neither transfer strength nor "boldness" — it
    measures how different the data's style is, not how easily a LoRA distills a TRANSFERABLE style or
    overrides the sticky 4-step DISTILLED prior. (b) A far cluster is far partly by being CONTENT-specific
    (fantasy female portraits), so the LoRA entangles style with subject → transfers to portraits, distorts/
    collapses on off-distribution content (cats). (c) The near-prior photographic cluster transferred MORE
    cleanly precisely because its "style" (warm-snapshot rendering) is nearly content-agnostic. TAKEAWAYS
    for a bold general style: prefer a coherent cluster whose style is content-AGNOSTIC (color/grain/light,
    not subject); and/or bake higher alpha (bold at scale 1.0 = current scale ~2, but a knob not new
    capability); and/or move to the BASE (non-distilled, CFG) model whose prior is less rigid. The distilled
    4-step prior is the real ceiling, not the data's CSD-distance. Artifacts (lora_far_v1, far_subset, prior
    imgs, log) archived to cold; tool + config committed.

  - **✅ LORA-TRAIN-2 (2026-06-30) — FULL single-block coverage + two bugs caught. The LoRA now adapts
    all 25 blocks (was 5 double only), and the targeted-style curation is no longer inert.**
    (1) SINGLE-BLOCK COVERAGE: mflux Flux2's single block fuses its projections into TWO Linears that
    map 1:1 onto the C engine's fused single-block adapters — `attn.to_qkv_mlp_proj` (norm→[Q,K,V,gate,
    up]) → `single_linear1`, `attn.to_out` ([attn_out,mlp_out]→hidden) → `single_linear2`. The C
    transformer ALREADY applies both (iris_transformer_flux.c:3708/3796); only the Diffusers LOADER was
    missing them (it loaded `proj_out`→single_linear2 only and warned "use Kohya"). Extended
    `load_diffusers` to read the fused keys (kept the HF `proj_out` fallback). Python: `SINGLE_ATTN_TARGETS`
    + `inject_lora_single_blocks` (wraps both fused Linears; recursive unfreeze covers double+single
    regardless of call order); export iterates `single_transformer_blocks`. train_step.flux_forward_lora
    already loops the (checkpointed) single blocks → single-block LoRA gets gradients with NO train-step
    change. Per-adapter count 40→80; trainable params 3.9M→18.7M. GUARD: extended `debug/test_lora.c` —
    fixture now carries the fused single-block keys with DISTINCT out dims (12 vs 8) so a mis-route can't
    pass; C loads `to_qkv_mlp_proj`→single_linear1 & `to_out`→single_linear2 at corr 1.000000 (prod-flag
    compile: single2 max_abs 1e-5 = ffast-math noise). make mps relinked; make test-unit green. Real-weights
    round-trip confirmed: a live train exports 160 tensors = 80 adapters (double 0-4 + single 0-19, all 20
    to_qkv_mlp_proj present), keys match the loader verbatim. v2 train (single+double): loss 1.00→0.11 by
    step 150, peak 23.3 GB stable (32 GB machine, mlx_memory_pct 0.6) — no wedge.
    (2) BUG — record_allowlist was INERT: `make_prefetch_loader` accepts `record_allowlist` but
    train_lora.py never passed it, so the LORA-TRAIN-1 "targeted" run actually trained on the FULL 22-shard
    pool, not the 250-id curated cluster. The corr-0.575 style there came from the whole pool, not the
    curation. Fixed: train_lora.py now loads `data.record_allowlist` (json {"rec_ids":[...]}) → set →
    passes to the loader (log confirms "record_allowlist: 250 rec_ids"). v2 is the first genuinely
    cluster-targeted run.
    VERIFICATION (v2, EMA, 300 steps, loss→0.28): live round-trip in the SHIPPED binary loads "80
    diffusers adapters across 5 double + 20 single blocks" (max_rank 16, no errors) — single-block
    coverage confirmed on real weights end-to-end. On-vs-off pixel corr scales monotonically with
    --lora-scale: 0.847 @1.0, 0.790 @1.5, 0.677 @2.5 → genuinely controllable. Style = a coherent
    warm soft-focus photographic portrait (creamy bokeh, warm grading), subject preserved, no quality
    breakdown even at 2.5. HONEST CAVEAT: at default scale 1.0 the effect is MODEST (corr 0.847) —
    the genuinely-curated tight cluster (intra-cos 0.77) is a photographic style close to the base
    model's default, so the delta is small; a clear look needs scale ~1.5–2.5. The DRAMATIC v1 look
    (corr 0.575, warm-earth/weathered) was the inert-allowlist FULL-POOL aggregate (a generic
    painterly average), NOT curation — so "tighter curation = stronger style" does NOT hold; tighter
    curation = more COHERENT but potentially SUBTLER (closer to the prior). For a bold targeted style,
    curate a cluster FAR from the base prior (or raise scale). All artifacts archived to cold
    (/Volumes/16TBCold/weights/lora: v1, v2, realdata, subset, full-history git bundle, README;
    SHA-256 verified).

  - **🔴 SREF-CHAMPION-COLLAPSE (2026-06-29) — THE SHIPPED CHAMPION IS REFERENCE-INERT: it applies a
    near-CONSTANT warm-painterly transform almost INDEPENDENT of the reference image. The ~0.70
    sref_score was an artifact of an all-painterly eval set. Recontextualizes the ENTIRE SREF campaign.**
    Discovered via the new simple-style eval set (every simple ref came out as the same muddy-brown cat).
    CONTROLLED TEST (hold prompt="a cat sitting on a chair", seed, scale 0.38, 512px; vary ONLY the
    reference; pixel-correlate the outputs):
      • 7 wildly different refs — 5 in-distribution WikiArt paintings (impressionism/cubism/baroque/
        expressionism/artnouveau) + Churchill line-art + kawaii sticker — produce NEAR-IDENTICAL outputs:
        pairwise corr ≥ 0.984, painting-vs-painting mean 0.991 (min 0.986), painting-vs-sticker 0.989;
        differ by < 3/255 per channel.
      • vs no-adapter baseline: corr 0.375 — so the adapter transforms the image STRONGLY; the injection
        is strong but CONSTANT, not weak. (Rules out "0.38 is just too low to see differences".)
      • Replicated at seed 123: cross-ref corr mean 0.988 (line-art vs graphic vs painting). Not a
        seed/prompt pathology — the controlled design isolates the reference effect and it is ~0.
      • Distinct features → identical outputs: different reference FEATURES correlate only ~0.30–0.42 in
        feature space (e.g. churchill-vs-artnouveau feat corr 0.33), yet OUTPUTS corr 0.99. The perceiver
        + ip_k/ip_v projections collapse diverse style inputs to a near-constant K/V injection.
      • New simple-style eval @0.38 (16 pairs): ALL 8 styles NEGATIVE Δstyle (style_sim −0.077..+0.037 vs
        null +0.004), per-style sref_score −0.23..−0.40, aggregate Δstyle −0.026 WASH. Contrast the
        painting set's Δstyle +0.116 — the SAME constant output scores + on warm/painterly paintings and
        − on high-key graphic styles.
    WHY THE PAINTING EVAL NEVER CAUGHT IT: all 5 eval paintings are warm/dark/painterly, so a constant
    painterly output correlates with each one → inflated style_sim → "0.70 sref_score". The eval measured
    "applies a generic painterly look," NOT "transfers THIS reference's style." It is a SEVERELY
    confounded metric on a homogeneous eval set.
    NOT A BUG IN TODAY'S WORK: the per-block inject path (IRIS_NO_FUSED_IP=1, used for ALL historical
    champion selection, golden-parity-guarded corr>0.999) collapses identically to the B2 fused path;
    iris loads distinct per-ref features and the output ≠ no-adapter, so the adapter IS using the
    features — it just maps them all to ~the same injection. This is TRAINING-side adapter mode collapse
    (candidates: only 3000 steps; 128-query perceiver bottleneck; style-pairing signal too weak / largely
    ignored; conditioning underused), NOT an inference knob.
    IMPLICATIONS / REREADS:
      (a) The "~0.70 ceiling is MECHANISM-bound" conclusion (DP-7 / fine-sweep / clean_concentrate_leak
          verdicts) is EXPLAINED by this: the mechanism injects a near-constant, so data concentration,
          leak objective, and injection timing ALL converge to ~0.70 because none of them touch
          per-reference fidelity. The convergence wasn't three levers hitting a representational wall —
          it was three levers all measuring the same painterly-prior artifact.
      (b) EVERY SREF VERDICT scored on eval_set.json (the 5 paintings) is confounded and should be re-read
          as "painterly-prior strength," not "style transfer." Champion selection among arms may be noise.
      (c) The champion is SHIPPED in the web (commit 97e8c66). It still works as a "make-it-painterly"
          filter (pleasing, ref-agnostic) — a product call for the owner, but it is NOT Midjourney --sref
          (match the uploaded image's specific style). Flag honestly in any user-facing copy.
      (d) FIX IS A RETRAIN, and the FIRST diagnostic is cheap: before any new training, measure
          reference-discrimination on the existing checkpoints (vary ref at fixed seed/prompt; cross-ref
          output corr must drop well below ~0.9 to claim real transfer). If even early/other checkpoints
          collapse, the conditioning path itself (perceiver/FiLM/ip-KV) is suspect, not the data recipe.
    NEW GATE (mandatory going forward): every champion A/B must report (1) BOTH eval sets, and (2) a
    reference-discrimination corr (cross-ref output corr at fixed seed/prompt). A high sref_score with
    cross-ref corr ≥ ~0.9 is a mode-collapse FALSE POSITIVE, not a win. Repro scripts:
    `scratchpad/painting_discrim.sh` + the corr snippets in this session; promote to `debug/` if reused.
    ⚠️ THE "MISATTRIBUTION / NO REGRESSION" CLAIM IN THE NEXT ADDENDUM IS WRONG — SUPERSEDED BY THE
    "REAL REGRESSION" CORRECTION further below (2026-06-29, later same day). Kept for the trail.
    ADDENDUM (2026-06-29) — COLLAPSE IS RESOLUTION/ASPECT-INDEPENDENT, AND THE USER'S "REGRESSION" IS A
    MISATTRIBUTION (not a code bug). User reported the web sref "used to work" (a crisp ginger-cat result)
    but now produces warm mush, and hypothesized the 512px collapse was a LOW-RES artifact (adapter
    over-powering at low res). Tested at the user's exact successful settings (prompt "a fluffy cat with a
    hat", seed 7, 1200×1024, scale 0.38, default sigmoid schedule):
      • Discrimination still collapses at 1200×1024 AND at 1024²: swap ONLY the reference, hold seed →
        output corr 0.994 (user's own two refs) / 0.97–0.99 (6 synthetic refs). Resolution and aspect do
        NOT rescue discrimination — the threshold is SCALE, not resolution. Low-res-artifact hypothesis REFUTED.
      • Good-vs-bad reconciliation (pixel corr, 600×512): the user's loved "good" result (a6db) correlates
        +0.627 with a NO-ADAPTER seed-7 base gen but only +0.372 with the adapter @0.38 — i.e. the good
        result lived in the BASE-MODEL family (effectively un-adapted / barely-adapted), NOT the adapter
        family. The "bad" result (6838) correlates +0.735/+0.746 with the adapter @0.38 family — it IS the
        adapter's true 0.38 behavior. So the adapter at the shipped 0.38 ALWAYS produced the warm mush; the
        crisp result the user loved was base output. No regression: commit dates confirm both gens ran
        identical post-B2 code, and pre-B2 vs post-B2 inject corr 0.9998.
      • No hidden scale override: web/server.py:1303 sets ip_scale = UI slider "strength" (default 0.38);
        there is NO resolution/aspect/step conditional that changes it.
      • Scale just cross-fades base↔collapse, never restores transfer: at scale 0.08 the output is 0.958 to
        the no-adapter base gen (adapter ~inert); raising scale walks toward the single collapsed warm-mush
        transform. There is no scale at which the adapter both stays crisp AND discriminates references —
        because the injection is a near-constant, lowering scale removes the adapter rather than sharpening it.
    PRODUCT IMPLICATION: the shipped default is a "make-it-painterly" filter, not --sref. Honest options
    until the retrain: (1) lower the default strength toward ~0.10–0.15 so it nudges rather than overrides
    (still ref-agnostic, but less destructive), or (2) gate the feature behind a "stylize" label. The real
    fix remains the retrain (discrimination gate now in debug/sref_ref_discrimination.py).
    CORRECTION (2026-06-29, later same day) — THE REGRESSION IS REAL AND IS A ROUTING/ENV FLIP, NOT A
    MISATTRIBUTION. The collapse finding above STANDS (the trained adapter IS mode-collapsed, corr 0.983
    across 6 distinct refs). What was wrong: the prior addendum concluded "no regression — the loved good
    result was base output." It analyzed the WRONG exemplar (a6db, a 1200×1024 crisp-cat that is in the
    base-model family — likely plain txt2img, no/ineffective reference) and over-generalized. The user then
    posted TWO unambiguous clean STYLE-TRANSFER pairs at "0.38": Churchill line-art ref → clean line-art
    cat (= web gen aab2c30b, favorited), and CyberFika graphic ref → bold flat-color cartoon cat. Base
    output CANNOT produce reference-specific style transfer, so these refute "good = base."
    PROVEN MECHANISM (CLI repro, seed 7532383631326939674, 1024×768, prompt "a fluffy cat with a hat"):
      • IN-CONTEXT img2img conditioning (`-i REF --img2img-strength 1.0`; reference enters as extra tokens,
        prompt drives content) reproduces BOTH user images EXACTLY: churchill → clean line-art fluffy cat
        w/ hat+bowtie in churchill's hat position; cyberfika → bold cartoon cat in the vibrant flat-sticker
        style on the same yellow-green-cyan gradient. THIS is the path that produced the loved results.
      • Strength semantics are INVERTED from intuition: `--img2img-strength 1.0` = in-context (style+prompt
        freedom → a CAT); LOWER strength = more literal copy of the ref (0.19 → reproduces churchill / the
        cyberfika logo itself, not a cat). So clean transfer needs HIGH strength, i.e. in-context.
      • The trained-adapter sref path (style_slots AND IRIS_IP_BUNDLE set, server.py:1273) → the collapsed
        soft cream cat. The in-context path is the LEGACY fail-open branch (server.py:1330+) used when no
        bundle is configured OR the ref is composition-mode (full strength, no ×0.5 half-weight).
    ROOT CAUSE OF GOOD→BAD: a web-server restart (the user's "installer") in the 13:41→15:13 window
    (history.json created_at 1782736860 → 1782742372) brought the server up with
    IRIS_IP_BUNDLE=/Volumes/2TBSSD/sref_eval/clean_concentrate_leak/bundle SET (the currently-running
    server, pid 41014 @21:48, has it set). With a bundle set, style-mode refs route to the collapsed
    trained adapter instead of in-context conditioning → the warm mush. NOT the venv (train/.venv encoders
    unchanged since Jun 19; web/venv only pip self-upgraded), NOT a binary/C-code change (v4.4.0..HEAD
    touch only backlog/install.sh; inject corr 0.9998 pre/post-B2), NOT the features (fresh vs cached ref
    bins corr 0.9995). Pure routing/env flip.
    FIX TO RESTORE THE LOVED BEHAVIOR (until adapter retrain): route web style references through
    in-context conditioning, not the trained adapter — i.e. either (a) launch the web server WITHOUT
    IRIS_IP_BUNDLE and ensure style-mode refs reach in-context at strength ~1.0 (the current legacy style
    branch applies ×0.5 → 0.19 → literal copy, so that half-weight must be removed/raised for style mode),
    or (b) in the UI add the reference as a composition/regular reference (routes to in-context @1.0 today).
    The trained-adapter sref champion stays gated behind the retrain + discrimination gate.
    Repro images: scratchpad/img2img_repro/{churchill,cyberfika}_incontext.png (clean) vs
    scratchpad/isolate/*.png (6 collapsed adapter outputs).

  - **🟢 SREF WEB GOOD→BAD REGRESSION — FULL ROOT-CAUSE INVESTIGATION + ROUTING FIX SHIPPED
    (2026-06-29).** The web "style reference" feature produced clean, reference-specific style
    transfer for the user on 2026-06-29 ~13:00–13:42, then produced "warm mush" (a constant
    soft-cream cat) from ~15:12 onward. This entry documents the end-to-end diagnosis, the proven
    mechanism, the exact root cause, the shipped fix, and the remaining work. Supersedes the earlier
    "no regression / misattribution" addendum under SREF-CHAMPION-COLLAPSE (which analyzed the wrong
    exemplar). The mode-collapse finding itself (the adapter is reference-inert) STANDS.

    SYMPTOM / USER GROUND TRUTH. User posted two unambiguous reference→output pairs at the UI default
    "0.38", prompt "a fluffy cat with a hat": (1) a CyberFika vibrant graphic → bold flat-color cartoon
    cat on the same yellow-green-cyan gradient; (2) a Churchill line-art coloring page → clean line-art
    fluffy cat with hat+bowtie in the ref's hat position. These are reference-SPECIFIC transfers — two
    different references gave two different, on-style outputs. The current behavior instead gives a
    near-constant warm cream cat regardless of reference. User asked: "explain these — proper clean
    transfer both at 0.38" and "did the installer change/replace the venv?"

    INVESTIGATION (what was tested, in order, with the ruling):
      1. Feature extraction integrity — RULED OUT. Fresh web-path SigLIP+CSD bin vs the cached/older
         ref bins: corr 0.9995 (SigLIP rows 0.9995, CSD row 0.9999). Extraction is stable & correct.
      2. The venv (user's hypothesis) — RULED OUT. train/.venv (encoders) unchanged since Jun 19;
         web/venv only had pip self-upgrade (Jun 29 13:47); Pillow/Flask/numpy unchanged since Mar 4.
      3. Binary / C-code regression — RULED OUT. Commits v4.4.0..HEAD touch only BACKLOG + install.sh;
         NONE touch IP-adapter / inject C. Per-block (IRIS_NO_FUSED_IP=1) vs B2 fused inject corr
         0.9998. The installer rebuilds via `make mps` from identical inject source.
      4. Inject path (fused vs per-block) — RULED OUT. Both produce IDENTICAL mush.
      5. Is the collapse real on the current binary? — CONFIRMED. Ran the current iris at the user's
         exact good-gen config (seed 7532383631326939674, 1024×768, scale 0.38, clean_concentrate_leak
         bundle) across 6 DISTINCT feature bins (fresh churchill + 5 cached web uploads). All 6 →
         the SAME soft-cream-cat-with-red-tophat. Mean pairwise output pixel corr 0.983 (min 0.962).
         The trained adapter is reference-inert at the shipped operating point.
      6. What produced the CLEAN transfers, then? — PROVEN: IN-CONTEXT img2img conditioning, NOT the
         adapter. CLI repro `./iris -d flux-klein-model -p "a fluffy cat with a hat" -i REF
         --img2img-strength 1.0 -S 7532383631326939674 -W 1024 -H 768` reproduced BOTH user images
         EXACTLY (churchill → clean line-art cat; cyberfika → bold cartoon cat). Strength semantics
         are INVERTED from intuition: --img2img-strength 1.0 = in-context (ref as tokens, prompt drives
         content → a CAT); LOWER strength = a more literal copy of the reference (0.19 reproduced
         Churchill / the CyberFika logo itself, not a cat). So clean transfer requires HIGH strength.

    CONCLUSION — TWO PATHS, ONE FLIP. There are two reference paths in web/server.py /generate:
      • IN-CONTEXT conditioning (legacy fail-open, server.py ~1341+): single reference → input_image →
        img2img at strength 1.0; reference is extra tokens, prompt drives content. → CLEAN, reference-
        specific style transfer. THIS is what the user loved.
      • TRAINED IP-ADAPTER (SREF-3, server.py ~1281): fires when a style slot is present AND a bundle
        is configured. → the collapsed constant cat.
    ROOT CAUSE of good→bad: a web-server RESTART (the user's "installer") between the last good gen
    (history.json created_at 1782736860 ≈ 13:41) and the first mush (1782742372 ≈ 15:13) brought the
    server up with IRIS_IP_BUNDLE=/Volumes/2TBSSD/sref_eval/clean_concentrate_leak/bundle SET (the
    server running at investigation time, pid 41014 @21:48, had it set). With a bundle set, style-mode
    references stopped falling through to in-context conditioning and started routing to the collapsed
    trained adapter. Pure routing/ENV flip — not venv, not binary, not features. (Note: history.json
    records img2img_strength 1.0 for BOTH good and bad gens — for the bad gen that's just the untouched
    Job default; the sref path never sets it. So metadata alone can't distinguish the paths; the visual
    + CLI repro does.)
    WHY THE EARLIER ADDENDUM SAID "NO REGRESSION": it analyzed a6db3f1e (a 1200×1024 crisp cat that
    correlates +0.627 with a no-adapter base gen — i.e. base-family, likely plain txt2img / ineffective
    ref) and generalized "the loved result was base output." The real style-transfer exemplar is
    aab2c30b (favorited, line-art cat) which is unmistakably in-context conditioning. Wrong exemplar →
    wrong conclusion.

    FIX SHIPPED (web/server.py, 2026-06-29): default the web "style" path to IN-CONTEXT conditioning;
    make the trained adapter OPT-IN behind a retrain.
      • New env flag SREF_USE_ADAPTER (IRIS_SREF_ADAPTER, default OFF). The adapter route now requires
        `SREF_USE_ADAPTER and IP_BUNDLE and bundle.exists()`. With the flag off (default), style refs
        route to in-context EVEN IF IRIS_IP_BUNDLE is still set — so the fix needs only a server
        restart, no env surgery.
      • Style-mode slots in the legacy path now use FULL-STRENGTH in-context (img2img_strength forced
        to 1.0), replacing the old ×0.5 half-weight that yielded 0.19 → a near-literal ref copy.
      • Saved style_codes (no image upload) can only be realized by the adapter; with the adapter
        disabled they now return a clear HTTP 400 ("upload the reference image … or set
        IRIS_SREF_ADAPTER=1") instead of silently misbehaving.
      • Verified by an in-process Flask test-client routing test (queue_generation patched):
        A (adapter off, bundle set): style image → in-context, img2img_strength 1.0, sref=None; style_code → 400.
        B (adapter on, bundle set): style image → trained adapter (sref=True). C (adapter on, no bundle): fail-open to in-context.
        Python-only change; no C/binary touched, so make mps/make test not implicated. Repro script:
        scratchpad/test_routing.py.
      • ACTION FOR USER: restart the web server to load the new code (`web/venv/bin/python web/server.py`).
        After restart the style upload path gives clean in-context transfer by default.

    NEXT STEPS (priority order):
      1. RETRAIN the IP-adapter to fix the mode collapse — the only path to TRUE --sref (match THIS
         reference's specific style, not a generic painterly/in-context-composition look). Before any
         training spend, run the cheap diagnostic in debug/sref_ref_discrimination.py on existing
         checkpoints (vary ref at fixed seed/prompt; cross-ref output corr must drop well below ~0.9).
         If even early/other checkpoints collapse, suspect the conditioning path (perceiver bottleneck /
         FiLM / ip-KV projections), not the data recipe. Collapse candidates on record: only 3000 steps;
         128-query perceiver bottleneck; style-pairing signal too weak / largely ignored.
      2. MANDATORY GATE going forward: every champion A/B must report BOTH eval sets AND a reference-
         discrimination corr. A high sref_score with cross-ref corr ≥ ~0.9 is a mode-collapse FALSE
         POSITIVE. (Codified under SREF-CHAMPION-COLLAPSE.)
      3. PRODUCT/UX: in-context conditioning is a real, shippable "style transfer" (it IS what the user
         loved) but it leans on the reference's COMPOSITION more than Midjourney --sref does. Decide UI
         copy: label the current default honestly (e.g. "style transfer (in-context)") and keep the
         trained-adapter --sref behind the retrain. The "strength" slider is inert for style mode while
         in-context (forced to 1.0) — either hide it for style mode or repurpose it.
      4. CLEANUP / CONSISTENCY: revisit whether IRIS_IP_BUNDLE should still be set in run.sh / the
         launch env at all while the adapter is disabled (currently harmless given the flag gate, but
         confusing). Once the retrain lands and the adapter is re-enabled, flip IRIS_SREF_ADAPTER=1.
      5. KEEP: the B2 fused-inject + SREF-3 resident-daemon speed work (4.5–9×) is unaffected and stays
         — it just won't be exercised until the adapter is re-enabled.
    Artifacts: scratchpad/img2img_repro/{churchill,cyberfika}_incontext.png (clean, = user's images),
    scratchpad/isolate/*.png (6 collapsed adapter outputs, corr 0.983), scratchpad/test_routing.py.

  - **SREF IN-CONTEXT vs TRAINED ADAPTER — NO DOUBLE-WHAMMY (mutual exclusivity, verified 2026-06-29).**
    Owner hypothesis: when the adapter is active, is the reference ALSO fed through in-context img2img
    conditioning — stacking two style injections into an over-cooked "too hot" state that explains the
    collapse — and could we fix it by dialing the in-context part down while the adapter runs? ANSWER: NO,
    there is no stacking.
      • The two paths are MUTUALLY EXCLUSIVE in web/server.py /generate. The adapter branch calls
        `queue_generation(..., input_image_path=None, reference_image_paths=None, ...)` (server.py:1333-1335)
        then returns immediately — the reference image is NEVER passed as an img2img/in-context input. It is
        encoded to SigLIP+CSD features (`job.ip_features`, server.py:1328) and injected ONLY via
        cross-attention K/V. The legacy in-context path passes the image AS input_image and attaches NO
        adapter. One request → exactly one path.
      • PREMISE CORRECTION: "img2img is how we supply the reference for the adapter" is false. The adapter is
        fed by FEATURE ENCODING (the SigLIP+CSD .bin), not img2img. img2img/in-context is a separate
        mechanism that only looks similar from the UI ("upload a reference").
      • COLLAPSE IS INTRINSIC, already proven IN ISOLATION: the isolation test ran the adapter with
        features-only and NO -i image (pure adapter, zero in-context) and 6 distinct refs STILL gave one
        constant cat (output corr 0.983). Removing in-context cannot help — there is none in that path. So
        "too hot from stacking" is ruled out; the adapter maps every reference to ~the same injection on its
        own. There is nothing to dial down.
      • DELIBERATE DESIGN (keep): input_image=None in the adapter path is correct — --sref is txt2img + style
        injection (the Midjourney model), not img2img. Feeding the image in-context TOO would create the real
        double whammy; the architecture already avoids it.
      • FUTURE OPTION (only AFTER the adapter works): a deliberate HYBRID — in-context for composition-aware
        base + adapter style push, each at partial strength (balance the two knobs). With a COLLAPSED adapter
        this is strictly worse (smears the constant cat onto an otherwise-good in-context result); revisit
        post-retrain.

  - **SREF ADAPTER RETRAIN — DIAGNOSTIC-FIRST PLAN (the one path to true --sref; opened 2026-06-29).** True
    --sref (match THIS reference's specific style, content-INDEPENDENT — not the in-context composition look)
    requires a non-collapsed adapter. Do NOT spend training time before the cheap diagnostic localizes the
    fault — it branches the entire retrain strategy.

    STEP 0 — DISCRIMINATION SWEEP ON EXISTING CHECKPOINTS (no training; GPU-bound on gens only). Tool:
    `debug/sref_ref_discrimination.py --bundle B --feat r1.bin r2.bin ... [--scale 0.38 --seed S --size 512]`.
    PASS = max cross-ref output corr < ~0.90 AND clearly above the no-adapter floor. Material confirmed on
    disk (2026-06-29):
      (a) 15 EXPORTED ARMS in /Volumes/2TBSSD/sref_eval/*/bundle: clean_base, clean_concentrate,
          clean_concentrate_leak (champion), clean_leak, clean_leak025, clean_pool9, clean_hier,
          clean_siglipdn, style_arm, hybrid_arm, hybrid_hier, hybrid_siglipdown, leak1_pshuf, leak2_xref,
          csd_arm. 8 simple-set ref bins in /Volumes/2TBSSD/sref_eval/refs_feat_hybrid_simple/.
      (b) PRE/POST INPUT-NORM bundles: /Volumes/2TBSSD/sref_sweep/{bundle_inputnorm, bundle_confirm_fix} —
          the decisive test of whether the IP-ADAPTER-INFER-1 grid fix ever bought reference-specificity.
      (c) Champion intermediate ckpts: clean_concentrate_leak/ckpt/step_{0002500,0003000}.safetensors + best
          — but BOTH are LATE; NO genuinely-early snapshot exists (gap — a within-run early/late curve needs a
          fresh run, STEP 2). Testing step_2500 needs an export first (export_adapter.py).

    THE BRANCH (what STEP 0 decides):
      • If ALL arms + bundle_inputnorm collapse (cross-ref corr ≥ ~0.9 everywhere) → the CONDITIONING PATH is
        the suspect, NOT data/recipe. LIKELY OUTCOME: the historical IP-ADAPTER-INFER-1 collapse (a few
        massive-activation SigLIP dims dominate the perceiver Q·K → every learned query attends ONE token →
        pooled/constant injection) was fixed for its GRID symptom by input-norm but, per memory, may have
        survived as a CONSTANT-STYLE output. If bundle_inputnorm ALSO collapses on discrimination, that
        CONFIRMS input-norm only removed the grid, never bought reference-specificity → fix is ARCHITECTURAL
        (STEP 1A).
      • If SOME arms discriminate (corr < ~0.9) → recipe/data/duration matters → STEP 1B. NOTE: the
        all-painterly eval confounded every prior sref_score, so DO NOT trust the old arm ranking — re-rank
        arms by DISCRIMINATION, not score.

    STEP 1A — IF CONDITIONING-PATH BOUND (architectural retrain). Candidate levers, cheapest first, each
    GATED by the discrimination test before a full run:
      1. Per-query DIVERSITY regularization in PerceiverResampler — attention-entropy / query-orthogonality
         penalty so queries can't all collapse onto one key. Directly targets the collapse mechanism.
      2. K/V injection RANK audit (cheap OFFLINE probe, no training): measure variance of to_k_ip/to_v_ip
         outputs across references. If the perceiver output IS diverse but the projections collapse it to a
         near-constant injection, the fault is the projection, not the perceiver.
      3. CSD/FiLM path check: FiLM-zero init starts inert — confirm the CSD style vector actually contributes
         (ablate SigLIP rows → does style survive on CSD alone? ablate CSD → does discrimination change?). If
         style rides entirely on the collapsed SigLIP perceiver, rebalance toward CSD (CSD-dominant
         conditioning is a logged speculative lever).
      4. Stronger/different input normalization than per-dim z-score+affine (the learned affine can relearn
         the outlier imbalance) — clip massive-activation dims or use fixed standardization.
      5. Wider/deeper perceiver ONLY if 1–4 implicate capacity (less likely than attention collapse).
      PLUS a DISCRIMINATION-AWARE TRAINING SIGNAL (not just eval): a contrastive/repulsion term that punishes
      reference-agnostic output, so the optimizer can't minimize loss by injecting a generic style. This is
      the single most likely MISSING ingredient — the current loss lets a constant injection win.

    STEP 1B — IF RECIPE-BOUND. Train the discriminating recipe longer than 3000 steps (undertraining is a
    logged candidate) with that arm's data-concentration + leak objective; re-gate on discrimination at each
    checkpoint.

    STEP 2 — INSTRUMENT THE RETRAIN (either branch). Checkpoint frequently (e.g. every 250–500 steps) and run
    the discrimination test PER checkpoint → an early/late discrimination CURVE answering "did it ever
    discriminate then collapse, or never discriminate?" (the within-run question the step-2500/3000-only ckpts
    can't). Promote a checkpoint ONLY if it PASSES discrimination — the new mandatory gate, NOT sref_score.

    STEP 3 — RE-ENABLE + VALIDATE. When a checkpoint passes discrimination: export it, set IRIS_SREF_ADAPTER=1,
    and A/B the trained adapter vs in-context on the DIVERGENCE test (style/content-mismatched pair, e.g. a
    painterly landscape ref + "a portrait of a woman") to confirm it delivers what in-context structurally
    cannot (style WITHOUT the reference's composition). Until then the web default stays on in-context.

    PREREQS / PITFALLS (carried): probes run CACHED only (live-encode segfaults MLX — BUGS MLX-1); NEVER train
    from cold storage (AGENT #6 — copy shards to hot SSD first); `make mps` after any C inference/inject
    change; the B2 fused inject + SREF-3 resident daemon are unaffected and ready the moment the adapter is
    re-enabled.

    STEP 0 RESULT (2026-06-30) — DECISIVE: 17/17 CHECKPOINTS COLLAPSE → CONDITIONING-PATH / TRAINING-SIGNAL
    BOUND, NOT RECIPE-BOUND. Ran debug/sref_ref_discrimination.py (4 maximally-distinct refs: churchill
    line-art, cyberfika graphic, woodcut, flat sticker; scale 0.38, seed 42, 512px, prompt "a cat sitting on
    a chair") across every existing checkpoint. EVERY ONE collapses (max cross-ref output corr ≥ 0.90):
      • 11 HYBRID arms: clean_base 0.997, clean_concentrate 0.994, clean_concentrate_leak (champion) 0.993,
        clean_hier 0.987, clean_leak 0.998, clean_leak025 0.990, clean_pool9 0.915 (lowest, still collapse),
        clean_siglipdn 0.995, hybrid_arm 0.995, hybrid_hier 0.994, hybrid_siglipdown 0.996.
      • 1 CSD arm: csd_arm 0.980 — a STRUCTURALLY DIFFERENT conditioning path (FiLM-modulated queries, no
        SigLIP perceiver) and it STILL collapses → the failure is NOT perceiver-specific.
      • 5 SIGLIP arms: bundle_inputnorm 0.999, bundle_confirm_fix 0.997, style_arm 0.991, leak1_pshuf 0.983,
        leak2_xref 0.992.
      styled-vs-no-adapter corr 0.22–0.68 everywhere → the adapters transform STRONGLY but reference-
      INDEPENDENTLY (not inert; rules out "scale too low").
    THE SMOKING GUN — bundle_inputnorm (the 2026-06-18 IP-ADAPTER-INFER-1 grid fix) collapses at 0.999.
    This CONFIRMS the long-standing suspicion (memory [[project-sref-state]]): input-norm fixed cross-TOKEN
    collapse (the grid — all queries attend one token WITHIN a reference, raising ip_embeds cross-token ratio
    0.0033→0.43) but NEVER touched cross-REFERENCE collapse (DIFFERENT references → ~the same injection).
    Those are TWO ORTHOGONAL axes; only the first was ever measured/fixed. The "echo of IP-ADAPTER-INFER-1"
    note was right: same collapse class, surviving as a constant-style output instead of a grid.
    CONSEQUENCES:
      (a) MORE/BETTER DATA IS DEFINITIVELY RULED OUT. Every data/objective lever the whole campaign explored
          (concentration, leak penalty ×weights, pooling, hierarchical inject, SigLIP downscale, CSD-only)
          collapses identically. Prior sref_score arm rankings are noise (confirmed; re-rank by discrimination).
      (b) The fault is in the SHARED downstream (cond-encoder output → to_k_ip/to_v_ip → cross-attn injection)
          AND/OR the TRAINING SIGNAL — both are cond_mode-agnostic, which is the only thing that explains
          siglip+csd+hybrid all collapsing identically. The loss never rewards reference DISCRIMINATION, so
          the optimizer's easy minimum is a generic "make-it-stylish" injection.
    STEP 1A — REVISED PRIORITY (architectural retrain; each gated by re-running discrimination):
      1. K/V INJECTION RANK AUDIT (offline, no training, CHEAPEST + most diagnostic): for ≥3 distinct refs,
         measure cross-ref variance of (i) cond-encoder output ip_embeds and (ii) to_k_ip/to_v_ip outputs.
         Memory says input FEATURES already differ (ref-vs-ref feat corr 0.30–0.42) yet OUTPUTS corr 0.99 →
         locate WHERE that diversity dies (perceiver/FiLM pooling vs the K/V projection vs the attention).
         The universal (all-cond_mode) collapse points at the projection/injection, not the encoder.
      2. DISCRIMINATION-AWARE TRAINING SIGNAL (most likely ROOT, cond_mode-agnostic): add a contrastive /
         repulsion term that punishes reference-agnostic output (e.g. different-ref injections must differ;
         or maximize cross-ref output distance at fixed prompt/seed). Without it a constant injection wins.
      3. Per-query diversity regularization in the perceiver (hybrid/siglip path) — secondary now that CSD
         (no perceiver) also collapses.
      4. CSD/FiLM contribution check + rebalance.
    Artifacts: scratchpad/{sref_discrim_sweep.py, sref_discrim_siglip.py}, discrim_step0[_siglip]/results.json,
    plans/sref-retrain-diagnostic.md. Tool bug fixed en route (commit ca8577c: NameError in the summary print).
    New siglip simple-set features at /Volumes/2TBSSD/sref_eval/refs_feat_siglip_simple/.

    STEP 1A.1 RESULT (2026-06-30) — ROOT CAUSE PINNED: `to_v_ip` IS CATASTROPHICALLY LOW-RANK → V is
    near-constant across references → output collapse. UNIVERSAL across cond_modes. Tool:
    debug/sref_kv_rank_audit.py (offline; runs cond-encoder → ip_embeds → to_k_ip/to_v_ip on N distinct
    refs; reports cross-ref cosine per stage + the SVD stable_rank of the K/V weight matrices).
    STAGE CROSS-REF COSINE (cos→1 = references stop mattering):
      • Champion (hybrid, 6 refs): raw SigLIP 0.348 → ip_embeds SigLIP-half 0.407 (STILL DISCRIMINATES —
        the perceiver is NOT the collapse site; input-norm worked) | ip_embeds CSD-half 0.978 (FiLM rank-1
        collapse, matches the 2026-06-22 CSD verdict) | K 0.864 | **V 0.953** (var_ratio 0.205).
      • bundle_inputnorm (siglip-only, 4 refs): raw 0.332 → ip_embeds 0.916 (this perceiver IS collapsed —
        undertrained 600-step proof) | K 0.917 | **V 0.998** (var_ratio 0.035 — only 3.5% of V is
        reference-specific). The output collapses regardless of where ip_embeds lands.
    WEIGHT-MATRIX STABLE_RANK (Σσ²/σ1²; 3072 = full rank, low = collapses inputs to a few directions):
      • Champion to_k_ip blocks 5/12/24 = 104/255/311 (near-full) vs **to_v_ip = 5.9/6.7/18.5** (RANK ~6).
      • bundle_inputnorm to_k_ip ≈ 770 all blocks vs **to_v_ip = ~25** all single-blocks.
      • Block 0 (a double-block) ≈ 770 for BOTH K and V in both bundles — consistent with double-blocks
        never engaging (ip_scale_double ≈ 0; the persistent campaign fingerprint).
    MECHANISM: cross-attn injection = softmax(Q·Kᵀ)·V. `to_v_ip` projects every reference's ip_embeds onto
    a ~6–25-dim subspace → V is dominated by a few SHARED directions → the injected value is
    reference-INDEPENDENT → output collapses, no matter how diverse the perceiver/K are. K stays full-rank
    (the adapter LOOKS at refs differently) but V collapsed (it INJECTS the same thing). This is the easy
    minimum of a loss that rewards a generic "make-it-stylish" push and never rewards reference-specific V.
    Reconciles everything: the perceiver/grid fix (input side) never touched this OUTPUT-side collapse;
    every cond_mode shares to_k_ip/to_v_ip so every cond_mode collapses; data levers can't move a low-rank
    projection. RECONCILES the prior "perceiver + ip_k/ip_v collapse" memory note — it's specifically ip_V,
    not the perceiver and not ip_K.
    STEP 1A — REVISED FIX (sharpened by the rank finding; each gated by re-running discrimination):
      1. DISCRIMINATION-AWARE TRAINING SIGNAL (the cause): contrastive/repulsion term forcing different refs
         → different V/output. Removes the incentive that drives to_v_ip low-rank. PRIMARY fix.
      2. RANK/VARIANCE REGULARIZER on to_v_ip (direct symptom fix): penalize low stable_rank / preserve input
         variance through to_v_ip (nuclear-norm ratio, or a V-output decorrelation term). Re-audit rank after.
      3. CSD FiLM redesign (secondary; hybrid only): rank-1 global (scale,shift) can't carry style — but
         siglip-only collapses too, so this is NOT the bottleneck; deprioritize vs (1)/(2).
      4. K is healthy — no work.
      Probe the retrain with debug/sref_kv_rank_audit.py per checkpoint (cheap, offline) alongside the
      discrimination gen-gate — to_v_ip stable_rank rising AND cross-ref V cosine dropping is the leading
      indicator that the fix is working, before spending gens.
    Artifacts: debug/sref_kv_rank_audit.py (new, committed). Numbers reproduced 2026-06-30.

    STEP 1A IMPLEMENTATION + SMOKE (2026-06-30) — rank penalty WIRED + VALIDATED; symptom fix helps but
    is INSUFFICIENT alone → repulsion (cause fix) is next. Implemented two pure, unit-tested loss
    primitives (commit 30293b1): `style_repulsion_loss` (cause: different refs at same prompt/noise must
    produce different AdaIN style stats; hinge) + `vproj_rank_penalty` (symptom: spectral penalty
    σ1²/‖W‖_F² on to_v_ip, σ1 via warm-started power iteration). Wired the RANK penalty into both trainer
    loss paths behind `training.vproj_rank_weight` (commit bac44e5; threads persistent power-iter state
    _rank_u; no signature change). Added `sref_kv_rank_audit.py --ckpt` (per-checkpoint weight-rank, the
    leading indicator).
    SMOKE: warmstart from the collapsed champion (step_0003000), vproj_rank_weight=2.0, 300 steps, 512px,
    cached hot data (config /Volumes/2TBSSD/sref_eval/smoke_rank/). Results:
      • WIRING VALIDATED: trains clean, NO MLX wedge, loss finite/stable (avg 1.24→0.75), mlx_mem peak
        24.8 GB. (Direct trainer, not flywheel.)
      • to_v_ip stable_rank ROSE monotonically (blocks 5/12/24): baseline 5.9/6.8/18.7 → step150
        13.4/18.4/32.9 → step300 15.1/24.1/43.7 (2.3–3.5×, still climbing). top1 energy fell
        (block5 0.170→0.066). Symptom fix works as designed.
      • DISCRIMINATION (export step300 → bundle → sref_ref_discrimination.py, 4 refs @0.38/512): cross-ref
        output corr mean 0.977→**0.886**, max 0.993→**0.926**; styled-vs-noadapter 0.373→0.262. So
        capacity↑ DID move discrimination the right way (~0.09 mean) — references start to matter — but it
        STILL FAILS the 0.90 gate. EXACTLY as predicted: rank-penalty grants V the CAPACITY to carry
        ref-specific info but does not FORCE its USE; a low-rank generic injection is still the loss's easy
        minimum. (Caveat: 300 warmstart steps, rank not plateaued — a longer run would push further, but
        the gap to <0.90 is the repulsion's job.)
    NEXT — wire the CAUSE fix (`style_repulsion_loss`), the more invasive change (batch=1 → needs a
    different-style 2nd reference): ring-buffer of recent cond_features; each cond step compute
    x0_other = _pred_from_embeds(get_image_embeds(buffered)) on the SAME precomputed Flux state (cheap),
    add repel_w·style_repulsion_loss(x0_pred, x0_other). Signature change through loss_fn/compiled_step.
    Then re-run the smoke (rank + repulsion together) and re-gate on discrimination — target max cross-ref
    corr < 0.90 (and ideally ≪). Artifacts: /Volumes/2TBSSD/sref_eval/smoke_rank/{config.yaml,ckpt,bundle},
    scratchpad task logs.
    REPULSION WIRED (commit 717a1f9) + FIRST RESULT (2026-06-30) — NEGATIVE as tuned, needs rethink.
    Ring buffer of recent refs → cheap 2nd prediction via _pred_from_embeds → repel_w·style_repulsion_loss.
    Combined smoke (rank_w 2.0 + repel_w 0.5 / margin 1.0): repel ACTIVE (repel_loss 0.98→~0.5–0.9) but
    DESTABILIZED — loss CLIMBED 0.6→2.0 (no content anchor; fights reconstruction). step150 discrimination
    0.904/0.939 — WORSE than rank-only@300 (0.886/0.926); styled-vs-base 0.218. So it disrupts WITHOUT
    disentangling. Likely a TRAIN/INFER MISMATCH: x0_other = ref-B's V in ref-A's shared Q/h_final context,
    which may not transfer to per-reference inference. Now testing gentle (repel_w 0.1 / margin 0.3,
    /Volumes/2TBSSD/sref_eval/smoke_gentle/). FALLBACKS if it still fails <0.90: (a) decorrelate to_v_ip
    OUTPUTS across buffered refs directly (penalize cross-ref V cosine — the exact 0.95–0.998 quantity from
    Step 1A.1, no x0 round-trip); (b) give the 2nd ref its OWN Q context (extra correct-forward-Q pass);
    (c) longer RANK-ONLY (best so far at 0.886; may cross 0.90 with more steps — cheap, stable). Full detail
    + hypotheses in plans/sref-retrain-diagnostic.md. NET STATUS: root cause solid (to_v_ip rank), rank
    penalty is a real partial win (0.977→0.886), cause-fix repulsion is unproven and under active iteration.
    UPDATE (2026-06-30, C + A done): (C) longer rank-only @600 = V cosine still 0.965 → rank raises V
    *rank* but leaves V *vectors* collinear → explains the plateau; stopped. (A) vproj_decorr_loss
    (commit 95c2c53) penalized V cosine directly: V cosine DROPPED 0.965→0.578 but gen discrimination got
    WORSE (0.983/0.995) — the optimizer GAMED the V-space proxy (ref-specific V variation that doesn't
    propagate to the image at scale 0.38). KEY LESSON: intermediate-tensor proxies (x0-style-stats, V-cosine)
    get gamed/destabilize; only the OUTPUT is non-gameable. to_v_ip rank is necessary-not-sufficient. Remaining
    principled lever = OUTPUT-space repulsion done right (Option B: x0 repulsion with the 2nd ref's OWN Q
    context + a content anchor). 5 training experiments done; cause fix UNSOLVED; best still rank-only 0.886.
    FINAL (2026-06-30, Option B done) — CAUSE FIX IS UNREACHABLE BY LOSS DESIGN; COLLAPSE IS STRUCTURAL.
    (B) x0 repulsion with the 2nd ref's OWN IP-influenced Q (commit, fixes the train/infer mismatch) +
    content anchor (leak 0.5): repel_loss dipped early (0.48→0.29, styles separating) then RE-COLLAPSED
    (back to 0.49 by step 300) as the content anchor (0.5) overpowered the repulsion (0.2) — restored
    content but lost the separation. Discrimination 0.950/0.961 (FAIL, worse than rank-only). FULL SCOREBOARD
    (max cross-ref output corr, PASS <0.90): champion 0.993 · rank-only 0.926 (BEST) · aggressive x0-repel
    0.939 · gentle x0-repel 0.945 · A V-decorr 0.995 · B own-Q repel 0.961. SIX experiments, ALL FAIL.
    The model has an overwhelming preference for the collapsed solution at scale-0.38 injection into the
    FROZEN DISTILLED base; every loss lever either (a) is GAMED in an intermediate space (V-decorr), (b)
    DESTABILIZES (aggressive x0), or (c) is OVERPOWERED by the flow/content terms (gentle, B re-collapse).
    The rank penalty (symptom) is the only partial mover (0.977→0.886) and even it plateaus. CONCLUSION:
    the SREF mode collapse is STRUCTURAL / mechanism-bound — NOT a training-loss-design problem. This
    INDEPENDENTLY CONFIRMS the earlier "~0.70 mechanism-bound ceiling" (SREF CHAMPION PINNED / FINE-SWEEP)
    from a fresh angle (6 loss-design experiments). REAL fixes are architectural, all speculative-retrain
    tier: (1) CSD-dominant conditioning (the SigLIP perceiver path is the collapsing one); (2) a BASE-model
    adapter (undistilled → CFG, more capacity, higher effective scale — but SREF-BASE-1 showed the distilled
    adapter doesn't transfer, so this is a fresh train); (3) a different injection mechanism / higher scale
    with content preservation. RECOMMENDATION: BANK the diagnosis (root cause + the proof that loss-design
    can't fix it is durable, valuable); KEEP the web on the working IN-CONTEXT path (real per-reference
    style transfer for users TODAY, IRIS_SREF_ADAPTER off); treat true --sref as a larger future project,
    NOT a loss tweak. Reusable assets committed: loss primitives (style_repulsion_loss, vproj_rank_penalty,
    vproj_decorr_loss) + trainer wiring (all gated off by default) + sref_kv_rank_audit.py(--ckpt) +
    sref_ref_discrimination.py. Run dirs: /Volumes/2TBSSD/sref_eval/{smoke_rank,smoke_rank_repel,smoke_gentle,
    run_C_rankonly,run_A_decorr,run_B_ownq}/.

  - **SREF FINE-SWEEP RESULT (2026-06-27): the ~0.543 plateau was a MEASUREMENT ARTIFACT; the true
    content-preserving frontier is ~0.62, and the data & objective levers TIE there → leans
    MECHANISM-bound.** Ran combined fine grids (0.35/0.40/0.45 added to clean_concentrate_leak +
    clean_leak; 0.55/0.60 to clean_leak025), reusing coarse gens. Full content-gated frontiers
    (null prompt 0.1516, gate retain≥0.75):
      clean_concentrate_leak: 0.30 r0.443/ret1.00 · 0.35 Δs0.080 r0.516/ret0.926 OK · 0.40 r0.659/ret0.688 WASH
      clean_leak:             0.30 Δs0.016 r0.543/ret1.01 · 0.45 Δs0.080 r0.512/ret0.888 OK · 0.50 ret0.637 WASH
      clean_leak025:          0.50 Δs0.056 r0.427/ret0.841 OK · 0.55 ret0.679 WASH
    KEY 1 — the famous 0.543 (clean_leak @0.3) is a DEGENERATE point: Δstyle only 0.016 (≈no style
    transfer); ratio looks high only because Δleak≈0 too (the "safe-but-useless corner"). NEVER a
    usable operating point. The honest metric must be read at a MATCHED USEFUL Δstyle budget.
    KEY 2 — at matched Δstyle≈0.08: clean_concentrate_leak @0.35 r0.516/ret0.926 vs clean_leak @0.45
    r0.512/ret0.888 — near-tied ratio, stacked has more retain headroom. Interpolating each arm to its
    gate-crossing (retain→0.75): BOTH peak at ~ratio 0.62 @ Δstyle~0.115 (stacked at scale~0.39, leak
    at scale~0.48). So the TRUE content-preserving frontier is ~0.62, not 0.54 — the coarse grid
    under-measured it by missing the crossing scales. This is a MEASUREMENT CORRECTION, not a new
    capability (no lever NEWLY broke anything; the model was always ~0.62-capable at the right scale).
    KEY 3 — data lever (concentration) and objective lever (leak) CONVERGE to the same ~0.62 ceiling;
    two independent levers hitting the same wall ⇒ likely MECHANISM-bound (representation/KV-injection),
    which ELEVATES DP-7 (injection schedule / CSD-dominant / AdaIN) as the next real lever over more
    data. clean_leak025 weaker (peak ~0.5, less headroom) — drop.
    NEXT: micro-sweep CHAMPION clean_concentrate_leak (ties on ratio, more headroom, data benefit
    compounds at rung-2) at 0.01 increments around the crossing (~0.36–0.40) with MULTIPLE SEEDS to
    MEASURE the peak (vs interpolated 0.62) + pin shippable --sref-strength. Then DP-7 mechanism work
    is the likely next major direction (per KEY 3). All frontier.json now carry the full fine grids.

  - **SREF CHAMPION PINNED (2026-06-28): clean_concentrate_leak, true content-preserving frontier
    MEASURES ~0.70 @ scale 0.39 — even better than the interpolated 0.62, and the 0.543 "plateau" is
    now definitively dead.** Micro-sweep (0.01 steps 0.36–0.40, 3 seeds 42/123/7, n=30/scale — clean
    MONOTONIC curve, so it's signal not noise):
      scale  Δstyle  ratio  retain  gate
      0.36   0.096   0.604  0.904   OK
      0.37   0.106   0.633  0.873   OK
      0.38   0.116   0.666  0.827   OK
      0.39   0.127   0.696  0.768   OK (gate edge; crossing ~0.393)
      0.40   0.137   0.721  0.690   WASH
    CHAMPION operating point: ratio 0.696 @ 0.39 (Δstyle 0.127, retain 0.768) = best content-preserving
    point of the entire campaign. SHIP default --sref-strength ≈ 0.38 (ratio 0.666, retain 0.827 —
    margin for per-image variance); 0.39 = aggressive edge; users can push past with the knob.
    The measured peak (0.70) > interpolated (0.62) → interpolation under-estimated; clean_leak's true
    measured peak is likely also >0.62 (not micro-swept — would only matter to claim stacked>leak, but
    we ship the stacked arm regardless since the data benefit compounds at rung-2). This is still a
    MEASUREMENT CORRECTION (right scale), NOT a new mechanism — ~0.70 is the recipe's ceiling.
    frontier.json for clean_concentrate_leak now holds the micro grid (0.36–0.40); the fine grid
    (0.3–0.7) is preserved in the FINE-SWEEP entry above. Gens at clean_concentrate_leak/gens (3 seeds).
    NEXT: (a) ship Tier-0 knob at 0.38; (b) DP-7 mechanism test — injection-SCHEDULE sweep (inject
    style only in late denoising steps) to see if ~0.70 can be pushed further; if it can't, ~0.70 is
    the mechanism ceiling and the hybrid platform at 0.38 is the deliverable.
