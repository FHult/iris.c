# FLUX.2 / iris.c — Improvement Backlog

Completed items are archived in [COMPLETED_BACKLOG.md](COMPLETED_BACKLOG.md).

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

The end goal: a user uploads a reference image; generations adopt its STYLE (not
content) via the IP-adapter on Flux.2 Klein, served by the iris engine. Gap analysis
2026-06-10 (post Phase-2 / TRAIN-7 / held-out-cond_gap session):

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

- **B-001: --vary-from / --vary-strength CLI wiring** (~1 hour) — `main.c`, `iris.h`
- **B-002: Z-Image CFG infrastructure** (~1 day) — `iris_sample.c`, `iris.c`, `iris.h` — unblocks Z-Image-Omni-Base; do this before B-003
- **B-003: Negative prompt for distilled Flux** (~2 hours) — `iris.c`, `main.c` — prerequisite for Web UI Feature 1

---

## Web UI Features

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

External static review of the C inference engine (`grok_bug_report.md`, untracked).
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
