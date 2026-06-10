# Deep Code Review — C engine + training pipeline (2026-06-10)

Pedantic source review covering bugs, inefficiencies, conventions, MLOps and model-quality
issues. Method: full-tree pattern sweeps (memory/string/overflow classes in C; exception/
concurrency/SQL classes in Python), then deep reads of the highest-risk files — including the
two flagged as never-audited in the standing backlog (`iris_lora.c`, `embcache.c`) and the
code added in the last week (pause `--free-gpu`, checkpoint archival, detectors). Every
claim below was verified against source; refuted suspicions are listed at the end so they
aren't re-investigated.

Severity: **High** = wrong results or data loss on a realistic path · **Medium** = real bug,
constrained trigger · **Low** = latent/edge · **Info** = convention/hygiene.

---

## 1. C engine — bugs

### C1 (Medium) `iris_lora.c`: OOM path leaves dangling pointers → use-after-free + double-free
`load_adapter_split_qkv` (iris_lora.c:228-235): on partial malloc failure it frees all six
buffers but does **not NULL the adapter fields**, and rank/dims stay set. The transformer
gates LoRA application on `lora->double_img_q[i].lora_A != NULL`
(iris_transformer_flux.c:2270-2320) → it would **apply freed weights** (UAF), and
`lora_free` → `free_adapter` then **double-frees**. Trigger: OOM during LoRA load only.
Fix: NULL all six pointers after the frees (mirror `load_adapter`'s error path, which is
correct at :169-173).

### C2 (Medium-Low) Silent-OOM void APIs leave outputs uninitialized
Pattern in three places: on malloc failure the function silently returns and the caller
consumes garbage with no error signal.
- `iris_kernels.c:690` (`scores`) and `:943` (`tile_scores`): attention silently returns →
  uninitialized attention output propagates NaNs/garbage into the image.
- `iris_ip_adapter.c` `perceive()` (:187): returns leaving `ip_embeds` **uninitialized**
  (worse than skipping); `inject()` (:230) silently skips injection (quietly produces an
  unconditioned image — a silent quality failure).
Fix direction: return an int status (or at minimum `memset` outputs and log once).

### C3 (Low) Cross-file magic-constant coupling: LoRA scratch vs resolution clamp
`iris_lora.c:559` hardcodes `scratch_len = 14000` justified by "max 1792×1792 → 13056
tokens"; the actual clamp lives in `main.c:900-903`. If the 1792 limit is ever raised, the
LoRA scratch silently overflows (heap corruption in `lora_apply`). Derive the scratch size
from the same limit (shared #define) or from runtime dims at load.

### C4 (Low, latent) `int`-typed malloc size math
`iris_qwen3.c:1541-1558`: products like `num_heads * seq_len * seq_len` are computed in
`int` before widening (`attn_scores` overflows int32 at seq≈8192 with 32 heads). Today
Qwen3 seq is capped at 512 so it's unreachable; it becomes live if longer contexts are ever
enabled (BL-008 already lifted the GPU path past 512). Same pattern across the alloc block.
Cheap hardening: `(size_t)` the first operand (as `iris_ip_adapter.c` and `iris_lora.c:210`
already do).

### C5 (Low) `iris_ip_adapter.c` int8 dequant trusts the companion scale tensor
`load_tensor_f32` (:93-100) computes `rows = numel/cols` from the weight tensor and indexes
`scale[r]` without validating the scale tensor's numel == rows. A malformed/truncated bundle
reads out of bounds. Bundles are self-produced today; validate anyway (one compare).

### C6 (Info) `iris_ip_adapter_load` meta buffer truncation
`meta[4096]` (:117): an adapter_meta.json over 4095 bytes is silently truncated; keys past
the cut parse as missing. Current metas are ~300 bytes. Either size-check `fread` against
file size or bump + log.

### C7 (Info) Metal deferred-release overflow falls through to immediate release
`pool_release_buffer` (iris_metal.m:440-454): if the deferred queue is full **during an
active batch**, the buffer is released immediately — exactly the race the deferred queue
exists to prevent (slot can be re-acquired while the in-flight command buffer still reads
it). `DEFERRED_POOL_RELEASE_MAX` is 16384 so this is practically unreachable; still, the
fall-through is silent. Add an `NSLog` + leak-instead-of-race choice if it ever fires.

### C8 (Info) `calloc(n,…)` with `n==0` portability
`alloc_adapters(0)` (iris_lora.c:121-124) for models with zero single blocks: `calloc(0)`
may legally return NULL, which `lora_load` (:531-538) treats as allocation failure. macOS
returns a unique pointer so it works today; portability nit.

### C9 (Info) Possible dead struct field
`iris_ip_adapter_t` carries a `_heap` member that the implementation never touches (load
uses only `_sf_handle`). If confirmed unused, remove (project rule: no dead code).

## 2. C engine — inefficiencies

- **`iris_ip_adapter.c` `mha_sdpa` allocates 4 buffers per call** (:61-64). Phase 2 wiring
  calls `inject` per block per denoise step (~25×28 ≈ 700 calls/image, plus head
  pack/unpack copies in `to_heads`/`from_heads`). Correct but wasteful — preallocate a
  workspace in the adapter struct when wiring Phase 2. (CPU-path only today.)
- **BF16 cache-full fallback** (iris_metal.m:1085-1094) creates an uncached buffer per call
  — a silent per-step perf cliff if the 1024-slot budget is ever exceeded by a new model
  combo. The sizing math in memory notes covers current models; a one-time NSLog on first
  fall-through would make regressions visible.

## 3. Python pipeline — bugs

### P1 (Medium) `pause --free-gpu` restart inserts a duplicate `iterations` row — **mine, last week**
`row_id = fw_db.insert_iteration(...)` (orchestrator.py:3042) runs **before** the GPU-work
`try`; the `_RestartIteration` → `continue` path re-enters the loop with the same iteration
number and inserts a **second row**. Duplicates then pollute `get_iterations`/`get_best`/
refgap/doctor (a stale `running` row for the same iteration). Fix: reuse the existing row
on re-entry (look up by `(name, iteration)` and update status) or delete the orphaned row
in the restart path.

### P2 (Medium) `pause --free-gpu` mid-staging: stage thread not joined — **mine, last week**
The restart path `rmtree(staging_dir)` + `continue` (orchestrator.py:~3415) without joining
`_stage_thread` (daemon, :3060s). If paused while cross-device staging is in flight: the
old thread keeps copying into a deleted tree (wasted IO/errors), and the re-entered
iteration starts a **second** stager against the same campaign. Fix: signal/join the stage
thread in the restart path before `rmtree` (or skip rmtree and let re-entry reuse it).

### P3 (Convention, risk-tiered) `except: pass` density
Multiline `except → pass` counts: doctor 30, data_explorer 28, **orchestrator 19**,
mobile_mode 15, ablation_harness 14, **train_ip_adapter 13**, pipeline_lib 8. In
status/doctor tools these are mostly legitimate fail-open reads. In the **orchestrator and
trainer** they can mask real failures (this review itself found `_RestartIteration` being
swallowed by one such broad except during development — caught only by a unit test). Sweep
the orchestrator/trainer subset and narrow each to specific exception types or add a
debug-level log.

### P4 (Low) `time.time()` used for durations
26 uses in train_ip_adapter.py, 11 in orchestrator.py — elapsed/ETA math breaks under NTP
steps (multi-day runs on a laptop that sleeps are exactly the at-risk case).
`time.monotonic()` for durations; keep `time.time()` for timestamps.

### P5 (Info) f-string SQL with interpolated identifiers
`data_explorer.py:1039,1410` interpolates `{table}` into SQL. Sources are internal lists
(not user input) but the pattern invites copy-paste injection later; whitelist-assert the
identifier. (`debug/data_selection_report.py` interpolates a constant — fine.)

### P6 (Accepted, documented) `eval()` in ablation conditions; `shell=True` in doctor fix-runner
`ablation_harness.py:942` evals YAML-authored condition strings with empty `__builtins__`
(repo-controlled input, fails open); `pipeline_doctor.py:4076` runs displayed fix commands
with deliberate `shell=True` + noqa. Both are conscious choices; listing so they stay known.

## 4. MLOps

### M1 (High — the central finding) `cond_gap` is a train-batch log-parsed stat, not a held-out eval
`flywheel_lib.collect_metrics_from_log` (:586) regex-parses the **last-seen** in-training
snapshot from the trainer log. That number is the campaign's champion criterion
(`get_best`), the shard-attribution input (`update_scores`/`update_excluded_scores`), and
the plateau/stall signal. Three compounding problems: (a) it's computed on **training
batches** — it measures fit to the shards being trained, not generalization; (b)
"last-seen" makes it end-of-run-weighted and high-variance; (c) regex parsing means a log
format drift silently yields `cond_gap=None` (iteration recorded with no signal, no error).
The data-selection flywheel ranks shard mixes by this number, so ranking noise flows
straight into curation. Mitigations are already staged — PROD-1 (held-out val set) +
PROD-2 (`cond_gap_stop.py` selection logic, built and tested) — but they must ALSO be wired
into the **flywheel's** metric collection, not just production training. Until then, treat
attribution rankings as provisional.

### M2 (Medium-High) The champion's EMA weights are never preserved
The trainer maintains EMA (`best.safetensors`) — typically the better-generalizing weights
— but the flywheel only globs/archives `step_*.safetensors` (orchestrator.py:3304;
FLYWHEEL-CKPT-1 archival also only covers `step_*`). `best.safetensors` is overwritten by
every subsequent iteration, so **the EMA corresponding to the recorded champion is lost**
(warmup-run2's champion EMA is already gone; only the raw-weights backup survives). Fix:
archive `best.safetensors` alongside the step file per iteration (`iterNNNN_best.safetensors`)
and record both in `checkpoint_log`.

### M3 (Medium) Attribution smearing + selection concentration (live)
By design, one iteration-level cond_gap is written to all ~42 included shards
(orchestrator.py:3316-3328) — cohorts carry identical scores until repeated draws
disentangle them (`data_selection_report.py` shows the cohort banding live). Meanwhile the
UCB head is already over-concentrated: top shards have `n_selected=17` vs the runbook's
warm-up criterion of ≤8 — the recency penalty (0.20, window 2) is too weak against the
performance term for a 1280-shard pool. Watch it; if the head keeps absorbing selections,
raise `recency_penalty`/window or cap per-shard selections per K iterations.

### M4 (Tracked, open) Production-readiness gaps
PROD-1 (no held-out val set — T-05 disabled, doctor warns), PROD-2 wiring, and per-chunk
step-budget sizing are tracked in BACKLOG with the over-training analysis. Listed here for
completeness: **no production run before PROD-1/PROD-2 land.**

### M5 (Known/accepted) Operational gaps already documented elsewhere
Fixed-name logs pollute status until cleaned; orchestrator restart orphans in-flight prep
(`_active_prep` in-memory); per-iter hot-staging rmtree forces ~130-170 GB re-copy
(PRECOMP-5 deferred); `iteration += 1` after a failed iteration (no in-place retry) —
acceptable for data-selection mode where iterations are independent.

## 5. Model quality

### Q1 (Resolved → monitor) Adapter saturation / over-training
Established this week: warm-start + continued training degrades cond_gap regardless of
mechanism; from-scratch per-iteration (warmup-run4) is the working mode. The doctor's
over-training detector (cond_gap↓ while train_loss↓) guards regression.

### Q2 (Deliberate, unvalidated) Conditioning is single-stream-only
`freeze_double_stream_scales: true` (stage1_512px.yaml:24) pins double-stream ip_scale to
0 — all reference conditioning flows through single blocks (live logs confirm
`double=0.0000, single≈0.98`). This is a documented design choice, but it has never been
A/B-validated on this data; it's in the ablation sweep (`ablation_sref_v1.yaml`) — make it
one of the first arms when ablation unlocks.

### Q3 (Signal) Conditioning strength is weak overall, and source-dependent
Best-ever cond_gap is +0.027 on loss ≈1.0 (~3%). Per-source curation (live report): coyo
+0.0139 > journeydb +0.0026 > journeydb+wikiart −0.0028 — synthetic journeydb pairs appear
to condition weakly. This is exactly what the data-selection flywheel is for; the early
read says the production recipe should over-weight natural-image sources pending more
iterations.

### Q4 (Signal) ref_gap ≈ 0 — style/content separation not yet learned
`loss_ref self≈cross` (gap +0.0002–0.0007) across runs: the adapter doesn't yet distinguish
self-reference from cross-reference. Either `cross_ref_prob`/`style_loss` settings or
SigLIP features at 512px aren't style-discriminative enough. Ablation variables already
cover the former.

### Q5 (Bounded, documented) BN-pack transfer residual
Training↔C-inference latent-space correlation is 0.9995, not 1.0 (VAE-Q1 fix). Bounded,
guarded by `test_bn_pack.py` + `debug/vae_parity.c`; revisit only if end-to-end quality
gates fail with no other explanation.

### Q6 (Note) 1000-step from-scratch regime compresses ranking signal
Each data-selection iteration trains a heavily under-trained model; cond_gap differences
between shard mixes are measured in a low-signal regime. Comparability holds because the
budget is constant across iterations, but small deltas are noise — require repeated draws
(which the attribution design already does) before trusting per-shard conclusions.

## 6. Verified non-issues (don't re-investigate)

- `realloc` lost-pointer: all 10+ sites use the tmp-local pattern correctly.
- `strcpy` sites (qwen3:1496, zimage:2239, flux:466): all length-guarded upstream
  (`len >= 128` skip / 256-byte buffers vs d_name).
- Qwen3 alloc block: full NULL-check after the batch (:1561-1572).
- Heartbeats: atomic (`.pid.tmp` + rename) — readers can't see partial JSON.
- GPU lock: O_EXCL acquisition, pid-liveness stale-steal, EPERM workaround — correct
  (recycled-pid false-positive is the standard residual risk, acceptable single-host).
- Flywheel warmup clamp: `_build_flywheel_train_config` clamps warmup to `steps//10`
  (=100 at 1000-step iters) — the "whole iteration is lr-warmup" suspicion is false.
- `yaml.safe_load` everywhere; no mutable default args; no `tempfile.mktemp`; sqlite
  `check_same_thread=False` is paired with a `threading.Lock` in both DB wrappers.
- Stager copies via tmp + size check + rename.
- embcache.c: clean throughout (single-threaded by design; consistent slot lifecycle).

## 7. Prioritized actions

| # | Item | Sev | Effort |
|---|------|-----|--------|
| 1 | ~~P1: dedupe `iterations` row on `--free-gpu` restart~~ **FIXED same day** (`delete_iteration` + restart-path cleanup) | Medium | done |
| 2 | ~~P2: join/cancel stage thread in restart path~~ **FIXED same day** (join before rmtree) | Medium | done |
| 3 | ~~M2: archive champion EMA per iteration~~ **FIXED same day** (`iterNNNN_best.safetensors` archive + iter-keyed prune) | Med-High | done |
| 4 | ~~M1: held-out cond_gap for the flywheel metric~~ **WIRED same day**: trainer prints a paired held-out `VAL … cond_gap=` line at training end (same-noise cond/null per record); `collect_metrics` parses it and it SUPERSEDES the train-batch gap for champion/attribution (train value kept as `cond_gap_train`). Val-eval failure degrades to the train gap, never fails the iteration. Remaining: one GPU smoke of the end-of-run eval at next window. | High | wired |
| 5 | ~~C1: NULL adapter fields on `load_adapter_split_qkv` failure~~ **FIXED same day** | Medium | done |
| 6 | C2: status returns (or memset+log) for silent-OOM attention/ip_adapter paths | Med-Low | small |
| 7 | M3: watch selection concentration; tune recency penalty if head keeps growing | Medium | config |
| 8 | P3: narrow broad excepts in orchestrator + trainer only | Hygiene | medium |
| 9 | C4/C5/C9 **FIXED same day** (qwen3 size_t casts ×20; int8 scale-numel validation; dead `_heap` removed); C3 (shared resolution constant) remains | Low | partial |
| 10 | Q2: prioritize `freeze_double_stream_scales` arm when ablation unlocks | Quality | config |

Items 1-2 should land before the next `pause --free-gpu` use; item 3 before any iteration
produces a champion worth shipping; item 4 before trusting attribution rankings for the
production data recipe.
