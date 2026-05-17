# Telemetry Audit: iris.c ML Training Pipeline

_Generated 2026-05-16. Covers all scripts in `train/scripts/`._

---

## 1. What Exists Today

**Heartbeat files** (`/Volumes/2TBSSD/.heartbeat/*.json`, written every 30–60 s):

| Process | File | Key fields |
|---|---|---|
| trainer | `trainer_chunk{N}.json` | `step`, `total_steps`, `loss`, `loss_smooth`, `grad_norm`, `grad_norm_smooth`, `eta_sec`, `ema_drift`, `siglip_coverage_pct`, `loss_cond`, `loss_null`, `loss_self_ref`, `loss_cross_ref`, `ip_scale_mean`, `ip_scale_double`, `ip_scale_single`, `loader_wait_pct`, `loader_wait_ms_avg`, `buckets`, `mlx_active_gb`, `mlx_peak_gb`, `mem_available_gb` |
| orchestrator | `orchestrator.json` | `step=poll` |
| precompute | `precompute_chunk{N}.json` | `done`, `total`, `pct`, `eta_sec`, `current_shard`, `current_phase` |
| mine_hard_examples | `mine_hard_examples_chunk{N}.json` | `done`, `total`, `pct`, `threshold_loss`, `eta_sec` |
| ablation | `ablation.json` | `run_name`, `status`, `current_step`, `current_loss`, `current_ref_gap` |
| stager | `stager_chunk{N}.json` | `phase`, `status`, `shards_staged`, `shards_total`, `npz_staged`, `npz_archived`, `ckpt_archived`, `bytes_transferred` |
| download_convert | `download_convert_chunk{N}.json` | `done`, `total`, `pct`, `phase`, `dl_speed_mbps`, `current_tgz`, `in_flight_gb` |
| dedupe_filter, clip_dedup, build_shards | per-step files | basic `done/total/pct` |

**JSONL structured logs** (`/Volumes/2TBSSD/logs/`):

- `orchestrator.jsonl` — every `log_orch()` call (chunk lifecycle events, errors, decisions)
- `mine_hard_examples_chunk{N}.jsonl` — `start`, `eval_done`, `done` events with counts
- `validator_chunk{N}.jsonl` — v01/v03_04 events with CLIP-I score, adapter delta
- `train_chunk{N}.log` — full trainer stdout (parsed by ablation harness regex)

**SQLite databases**:

- `ablation_history.db` — per-run experiments table (params, score, ref_gap, cond_gap, final_loss, snapshots JSON, is_pareto flag)
- `shard_scores.db` — per-shard scores (ref_gap_mean, cond_gap_mean, loss_mean, attributed scores, effective_score, siglip_mean_emb, n_scored, n_excluded, selection history)
- `flywheel_history.db` — per-iteration records (ref_gap, cond_gap, train_loss, hyperparams, shard selection, checkpoint_log)

**Sentinel files** (`/Volumes/2TBSSD/pipeline/chunk{N}/*.done|*.error`):
Authoritative step completion state. No metric content.

**JSON outputs**:
- `run_metadata.json` — run provenance (config, git SHA, scale, started_at)
- `logs/val_chunk{N}/metrics.json` — post-training CLIP-I, adapter_delta, cross-chunk delta
- `cold_root/metadata/tgz_scores.json` — per-JDB-tgz quality score
- `dispatch_queue.jsonl` — escalated issues with suggested actions

---

## 2. Gaps by Use Case

### Ablation Decisions

**Missing from `ablation_history.db` schema:**

- **Per-run gradient norm trajectory** — `grad_norm` and `grad_norm_smooth` are in snapshot blobs but no indexed column. Cannot SQL-filter "runs with stable grad_norm" without deserializing every blob.
- **ip_scale trajectory** — In snapshot blobs, but no indexed scalar columns for `mean_ip_scale_final` or `ip_scale_double/single`. Ablations on `freeze_double_stream_scales` cannot be analyzed by actual scale magnitudes.
- **loss_cond / loss_null / cond_gap trajectory** — Only `cond_gap` tail mean is stored as a top-level column. Early training progression curve buried in blob, inaccessible to SQL.
- **Early stopping reason** — `EarlyStopper` SIGTERM becomes -2 exit code, metadata lost. No `stopped_early` boolean or `early_stop_step` column. Cannot distinguish early-stopped runs from crashes.
- **Warmstart source** — No column recording `--warm-start-from` and which prior experiment was used. Cross-campaign learning cannot track config lineage.
- **Data shard fingerprint** — No record of which shard set was used. A config that performed well on one subset may differ on another.
- **Variance / run count per config** — No support for multiple runs of same config. Noise estimation requires samples; schema prevents re-runs.

**Scoring function rigidity:**

`_score_weighted()` uses hard-coded normalization ranges (ref_norm clipped via ×2; cond_norm via /2.5; stab_norm via /4). No `score_version` column. Score values non-comparable across campaigns with different normalizations.

### Flywheel / Shard Selection

**Missing from `shard_scores.db`:**

- **Per-sample loss distribution** — Miner computes per-sample loss and selects top-K, but only mean of hard examples is emitted. No persistent histogram or percentile distribution per shard. Cannot distinguish "uniform hard shard" vs "bimodal shard with one hard cluster".
- **Hard example overlap across chunks** — `mine_hard_examples.py` does not record which `rec_id`s landed in hard examples. No persistent record of "shard X contributed Y records to hard examples in chunk N". Cannot track whether hard examples correlate with shard quality scores.
- **Shard loss variance / std** — EMA means stored, no variance. High-variance shards may spike grad_norm despite equal mean.
- **Gradient spike attribution** — Orchestrator tracks `_grad_spike_polls` (line 353) but has no shard context. Bucket stats logged per step but shard not recorded. No current tool connects spikes to specific shards.
- **Cross-chunk shard persistence** — EMA merges chunk-N and chunk-N+1 observations without tagging curriculum phase.

**`tgz_scores.json` gaps:**

- Only computed post-mine. If shard_scores.db is empty (first chunk), file is empty — no signal for download ordering of future data.
- No timestamp per tgz; scores may be stale after many flywheel iterations.

### Pipeline Diagnosis

**Precompute:**

- `precompute_all.py` writes `skipped_q` and `skipped_v` counts to per-shard result dict and prints them, but never calls `log_event`. Encoding failures are silent to doctor and status tools.
- No `cache_hit_rate` metric — the count of cache-hit shards is only in console stdout.
- No latency breakdown in heartbeat — `current_phase` is a string, wall-clock time per phase (Qwen3 vs VAE vs SigLIP) not queryable.

**Trainer:**

- `ema_drift` computed and written to heartbeat but `pipeline_status.py` does not display it. Signals EMA divergence before `best.safetensors` is updated.
- `bucket_stats` computed per log interval, in the heartbeat `buckets` field, but `pipeline_status.py` does not surface them and `ablation_harness.py` `MetricCollector` has no bucket parser.
- `loader_wait_ms_avg` in heartbeat but not displayed by `pipeline_status.py` (only `loader_wait_pct` warning print exists).
- **No VRAM headroom metric.** `mlx_active_gb` and `mlx_peak_gb` in heartbeat but no `vram_headroom_gb` computed or alerted on.
- **No step timing (steps/s).** ETA derived internally, but throughput not stored. Cross-campaign speed comparison requires manual log parsing.
- **No pre-OOM alert.** `memory_pressure.log` only read reactively on crash. No proactive `mem_available_gb < 3GB` dispatch alert.

**Orchestrator:**

- `_restart_counts` in-memory only. After orchestrator restart, retry count resets to 0 — potentially allowing more retries than intended.
- `_retry_after` backoff timing also in-memory. Restart during backoff causes immediate retry.
- GPU lock acquire/release via `ResourceManager` not logged. Lock contention invisible in `orchestrator.jsonl`.

### Cross-Campaign Learning / Warmstart

- **No campaign-level summary record.** `run_metadata.json` has config and git SHA but not final quality metrics. New campaign cannot query prior final cond_gap without log parsing.
- **Ablation warmstart is one-way.** Current campaign cannot write results back to a shared "campaign history" DB.
- **No actual vs planned steps.** `steps` column is planned steps; early-stopped runs have different actual steps. No `images_seen` column.
- **Flywheel checkpoint lineage not linked to ablation results.** `flywheel_history.db` has `ablation_run` text but `ablation_history.db` has no `flywheel_iteration` back-reference.

---

## 3. Dead Telemetry — Logged But Never Consumed

| # | What | Where written | Why dark |
|---|---|---|---|
| Dead-1 | `validator.py` `log_event` calls (v01_done, v03_04_done) | `validator_chunk{N}.jsonl` | No reader; doctor and status ignore the JSONL |
| Dead-2 | `bucket_stats` in trainer log structured output | Training log + heartbeat `buckets` field | No regex in MetricCollector; not surfaced in status |
| Dead-3 | `ema_drift` in trainer heartbeat | Heartbeat line 1868 | Not displayed by pipeline_status.py, not parsed by doctor or ablation harness |
| Dead-4 | `siglip_coverage_pct` in trainer heartbeat | Heartbeat line 1870 | Not displayed in status view; warning only in training log |
| Dead-5 | `clip_dups_report_chunk{N}.json` | `logs/` via `--report-out` flag | No downstream tool reads it |
| Dead-6 | `run_summary.txt` | `logs/run_summary.txt` via `_write_run_summary()` | Not machine-readable; not linked from run_metadata.json |
| Dead-7 | `logs/val_chunk{N}/metrics.json` (CLIP-I, adapter_delta) | Written by validator.py line 186 | Not read by doctor, data_explorer, or pipeline_status |
| Dead-8 | `selection_log` table in `shard_scores.db` | Written by ShardScoreDB | Never queried; flywheel_lib uses `iterations.selected_shards` JSON instead |

---

## 4. Quick Wins (Low Effort, High Value)

**QW-1: Add `grad_norm_final` and `ip_scale_final` to `ablation_history.db`.**
In `AblationDB.update_experiment()`, add two columns. In `_run_one()`, compute from `collector.snapshots[-1]`. Data already in snapshots blob; ALTER TABLE migration pattern already exists (line 299). Enables `ORDER BY grad_norm_final` for stability analysis.

**QW-2: Add `log_event` in `precompute_all.py` at completion.**
After final summary print (~line 1128), log `qwen3_written`, `vae_written`, `qwen3_skipped`, `vae_skipped`, `shards_processed`. Makes encoding failures visible to doctor without log parsing.

**QW-3: Add `steps_per_second` to trainer heartbeat.**
ETA already computed as `(total_steps - step) / steps_per_sec` internally. One line to store it. Enables cross-campaign throughput comparison and performance regression detection.

**QW-4: Persist `_restart_counts` to `pipeline_state.json`.**
In `_handle_error()` (line 1774), write restart counts to state after each update. On `Orchestrator.__init__()`, seed from `read_state()`. Prevents retry limit resetting on orchestrator restart.

**QW-5: Surface `ema_drift` in `pipeline_status.py`.**
Heartbeat already carries this field (line 1868). Add 3 lines to `print_human()` training block to display it. Zero infrastructure cost.

**QW-6: Add `stopped_early` and `stop_step` to ablation DB.**
Pass `early_stopped=(exit_code == -2)` and `stop_step` into `update_experiment()`. Two new columns. Distinguishes early-stopped runs from crashes in Pareto analysis.

**QW-7: Log mining `threshold_loss` and per-chunk hard example counts to `mine_hard_examples.jsonl`.**
Final `log_event("mine_hard_examples", "done", ...)` (line 663). Add `threshold_loss` and `skipped`. Creates a trend-able quality signal across chunks.

**QW-8: Add `mem_available_gb` low-headroom dispatch alert in orchestrator.**
In `_check_heartbeats()` (~line 2018), where trainer heartbeat is already read for loss/grad checks: if `hb.get("mem_available_gb", 99) < 3.0`, dispatch a WARNING. Catches pre-OOM conditions.

**QW-9: Store precompute cache-hit fraction in precompute heartbeat.**
`shards_before - len(shards)` count of skipped shards is computed (line 996). Store as `shards_skipped_cached` in first heartbeat write. Visible to status tool and doctor.

**QW-10: Make `pipeline_status.py` read `logs/val_chunk{N}/metrics.json` and display CLIP-I trend.**
Data exists (Dead-7 above). One read per chunk, add to status output. Zero new computation.

**QW-11: Surface `siglip_coverage_pct` in `pipeline_status.py`.**
Already in heartbeat (Dead-4 above). One display line in `print_human()`.

---

## 5. Deeper Additions (Higher Effort, High Value)

**DA-1: Per-step loss-to-shard mapping for gradient spike attribution.**
When a grad_norm spike exceeds 10× smooth (trainer line 1569), emit `log_event("trainer", "grad_spike", chunk, step, shard_id, grad_norm)`. Orchestrator's `_check_heartbeats()` already tracks `_grad_spike_polls` but has no shard context. Connecting spikes to shards lets `shard_selector.py` penalize high-spike shards in effective_score.

**DA-2: Per-shard loss distribution in mining output.**
`mine_hard_examples.py` evaluates up to `eval_records` samples and discards all but top-K. Compute P50/P75/P90/P99 loss per shard before discarding. Log in `done` event. Store in new `shard_loss_profiles` table in `shard_scores.db`. Reveals uniform-hard vs bimodal shards.

**DA-3: Campaign-level summary table.**
Add a `campaigns` table to `shard_scores.db` (or `campaign_history.db`) with `final_cond_gap`, `final_ref_gap`, `final_loss`, `total_steps`, `chunk_count`, timestamps. Write at pipeline completion. Enables "which campaign had best final cond_gap" queries across months of history.

**DA-4: Ablation DB `score_version` column and re-scoring support.**
Add `score_version TEXT` and `score_params TEXT` (JSON of weights and normalizers) to `experiments`. Add `rescore_all(run_name, new_objective)` to `AblationDB` that recomputes from snapshots without re-running experiments. Enables retroactive Pareto analysis when objective definition changes.

**DA-5: Precompute throughput breakdown logging.**
Track wall-clock time per phase per shard (Qwen3, VAE, SigLIP). Emit `log_event("precompute", "shard_done", qwen3_secs, vae_secs, siglip_secs, n_records)`. Feeds `pipeline_profile.py` and enables bottleneck identification.

**DA-6: Mining feedback loop into `shard_scores.db`.**
After `mine_hard_examples.py` completes, scan output tar IDs, group by shard stem, write hard-example density per shard into a `score_update(role='hard')` path in `ShardScoreDB`. A shard that consistently contributes to top-K hard examples should receive higher training weight in subsequent chunks.

**DA-7: Validation metrics linked back to ablation DB.**
Add `post_train_validation` table to `ablation_history.db`:
```sql
(experiment_id INTEGER, chunk INTEGER, clip_i REAL, adapter_delta REAL,
 clip_i_delta_vs_prev REAL, ts TEXT)
```
After pipeline validation completes, look up the experiment by git_sha + param hash and insert. Closes the loop between ablation selection and downstream image quality metrics.

---

## 6. Priority Order

| Rank | Item | Effort | Impact |
|---|---|---|---|
| 1 | DA-6 + QW-7: Wire mining output into shard_scores.db | Medium | Closes most important missing feedback loop |
| 2 | QW-1 + QW-6: grad_norm_final, ip_scale_final, stopped_early to ablation DB | Low | Unlocks Pareto filtering by stability |
| 3 | QW-4: Persist _restart_counts across orchestrator restarts | Low | Correctness fix, not just observability |
| 4 | QW-10 (Dead-7 fix): Surface val metrics.json in data_explorer + status | Low | CLIP-I trend across chunks currently invisible |
| 5 | DA-3: Campaign-level summary table | Medium | Essential for cross-campaign warmstart decisions |
| 6 | QW-8: mem_available_gb pre-OOM alert | Low | Proactive OOM prevention |
| 7 | QW-3: steps_per_second in trainer heartbeat | Trivial | Cross-campaign throughput comparison |
| 8 | QW-5: ema_drift in pipeline_status.py | Trivial | EMA divergence visibility |
| 9 | QW-2: log_event in precompute_all.py at completion | Low | Encoding failures visible to doctor |
| 10 | DA-2: Per-shard loss distribution in mining output | High | Reveals shard heterogeneity |
