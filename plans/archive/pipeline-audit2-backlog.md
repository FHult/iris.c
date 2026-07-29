# Pipeline Audit 2 — Backlog

Issues identified in audit (2026-04-02). Issues 1-5 were fixed immediately.
Items below are ordered by severity within each category.

---

## CORRECTNESS

### CA-06 — MEDIUM: Background filter loop killed with raw SIGTERM mid-write
**File:** `run_training_pipeline.sh:468`
`kill "$FILTER_LOOP_PID"` sends SIGTERM to the bash subshell; the Python process inside
has no signal handler and may exit mid-write, losing buffered log output.
**Fix:** Send SIGTERM, wait briefly, then SIGKILL; or move filter loop to a named script
with a trap that flushes before exit.

### CA-07 — MEDIUM: precompute_all.py single-record fallback failure is silent
**File:** `precompute_all.py:309-321`
In the batch failure path, per-record retries swallow all exceptions with no counter
and no log of which rec_ids failed. No way to know what fraction of the shard was lost.
**Fix:** Count per-record failures per shard; include in the returned dict and print summary.

### CA-08 — MEDIUM: pipeline_watchdog.sh stale alert re-fires every second
**File:** `pipeline_watchdog.sh:118`
`_last_alert="stale:$AGE_S"` — AGE_S increments by 1 each loop iteration, so the
debounce condition always evaluates false and a new notification fires every INTERVAL seconds.
**Fix:** Debounce on stale epoch (e.g., `_last_alert="stale"`) and reset only when heartbeat
becomes fresh again.

### CA-09 — MEDIUM: Inline HuggingFace download Python block may exit 0 on exception
**File:** `run_training_pipeline.sh:751-763`
The inline PYEOF block has no try/except. If `snapshot_download` raises, Python prints
the traceback and exits 1 (good), but if a partial download fills disk and the library
silently returns, no non-zero exit is raised.
**Fix:** Wrap in try/except, print error, and `sys.exit(1)` explicitly.

### CA-10 — MEDIUM: Anchor set training proceeds silently if ANCHOR_PID never set
**File:** `run_training_pipeline.sh:524`
`wait "$ANCHOR_PID" || die` is only reached when ANCHOR_PID is set. If
`create_anchor_set.py` never launched (already-done path returns early), training
starts without validating that the anchor dir is non-empty.
**Fix:** After the anchor step, assert `count_tars "$ANCHOR_DIR" -gt 0` regardless of
whether it was newly created or pre-existing.

### CA-11 — MEDIUM: JDB WDS "already exists" skip doesn't validate shard count
**File:** `run_training_pipeline.sh:769`
`if [[ $(count_tars "$JDB_WDS") -gt 0 ]]` — a single corrupt/partial shard causes the
entire convert step to be skipped.
**Fix:** Check count ≥ expected (JDB_DOWNLOAD_N / 5000 * 5000 ≈ JDB_DOWNLOAD_N shards).

### CA-12 — LOW: filter_shards.py fast path accepts 0-record shard silently
**File:** `filter_shards.py:159`
When `len(kept_records) == original_count == 0` (truncated tar), the fast path writes
the sentinel without validating that the shard actually has content.
**Fix:** Add `if original_count == 0: return {... "error": True}` before the fast path.

### CA-13 — LOW: mine_hard_examples.py heap ordering relies on implicit tuple comparison
**File:** `mine_hard_examples.py:~364`
Tuple `(-loss, rec_id, shard_path)` — correct but fragile; any reordering of fields
silently breaks the max-heap semantics. No comment explaining the negation trick.
**Fix:** Add a comment; consider a named dataclass with explicit `__lt__`.

### CA-14 — LOW: pipeline_status.sh parent-process heuristic brittle
**File:** `pipeline_status.sh:354-361`
Root pipeline PID detection uses `ps -o ppid` and checks if parent command is bash with
`run_training_pipeline` in args. Fails in containers or under debuggers.
**Fix:** Write the root PID to the lock file at pipeline startup and read it here instead.

---

## RESILIENCE

### RE-15 — HIGH: No disk-full recovery during precompute or shard build
**File:** `run_training_pipeline.sh`
The initial disk check runs once at startup. If disk fills during the 350+ GB precompute
write, Python crashes mid-`.npz`, leaving zero-byte files (now caught by fix 2) but no
graceful recovery path. The pipeline will retry 3 times (each ~20h) before dying.
**Fix:** Add a `df`-based disk check at the start of each major write step (build_shards,
precompute, precompute chunks 2-4). Fail fast with a clear message rather than spending
20h discovering full disk at the end.

### RE-16 — MEDIUM: No rollback when chunk 2-4 convert fails after partial write
**File:** `run_training_pipeline.sh:773-784`
Fix 4 prevents tgz deletion on zero shards. But if convert produced some shards (say
10/32) before crashing, the next run skips convert (`count_tars > 0`) and uses the
partial WDS dir. Training then uses an incomplete chunk silently.
**Fix:** Write a `.convert_done` sentinel only after convert succeeds; use that as the
skip condition instead of raw shard count.

### RE-17 — MEDIUM: clip_dedup.py incremental dedup appends to a shared file
**File:** `run_training_pipeline.sh:801` / `clip_dedup.py`
`duplicate_ids.txt` is appended to during cross-chunk dedup. A crash mid-run leaves a
partial append. Subsequent runs will have a partially deduped blocklist, silently
allowing some duplicates through.
**Fix:** Write to a temp file and atomically replace, or use a chunk-specific file and
merge only on success.

### RE-18 — MEDIUM: Prefetch chunk 2 failure is non-fatal
**File:** `run_training_pipeline.sh:671-679`
Prefetch failure logs a warning and continues. Chunk 2 will then fail to find its
download (24h later, after chunk 1 training completes).
**Fix:** Either make prefetch failure fatal (die) or add a `PREFETCH2_FAILED` flag that
causes chunk 2 to re-download before converting, rather than assuming the files are there.

### RE-19 — MEDIUM: precompute_all.py worker crash drops entire shard silently
**File:** `precompute_all.py:724-745`
`pool.imap_unordered` returns results one per shard. If a worker process crashes (OOM,
Metal crash), the result dict has `error=True` but no indication of which records were
lost. No retry at the pool level.
**Fix:** On `result["error"]`, log the shard path prominently and count total failed shards
in the final summary. Consider re-running failed shards once with workers=1 as a fallback.

---

## MLOPS / OBSERVABILITY

### MO-20 — HIGH: No disk-space check between precompute phases
See RE-15 above — operational impact is HIGH for MLOps (silent corruption → wasted training).

### MO-21 — HIGH: No checksum/hash validation of any written file
**Files:** `filter_shards.py`, `precompute_all.py`, `build_shards.py`
Tar writes, npz writes, and safetensors writes have no integrity verification after close.
A silent disk error (bad block, NFS hiccup) produces a file that passes `os.path.exists`
and even size > 0 check (fix 2), but is corrupt.
**Fix:** Write and verify a SHA-256 sidecar for each shard on creation; re-verify before
use in precompute and training. Or at minimum, verify numpy can load each npz during the
precompute resume check.

### MO-22 — MEDIUM: No unified log index
**Files:** all scripts
`build_shards.log`, `filter_background.log`, `precompute*.log`, `mine_hard_chunk*.log`,
`watchdog.log` accumulate with no manifest. On failure, the operator doesn't know which
log to check.
**Fix:** Write a `$DATA_ROOT/logs/pipeline_manifest.json` at each step transition listing
the current step name, its log file path, start time, and status (running/done/failed).

### MO-23 — MEDIUM: filter_shards.py doesn't warn on unusually high drop rate
**File:** `filter_shards.py:186-280`
A shard that drops 30%+ of records (vs expected 3-5%) is filtered silently. Could
indicate a data quality regression in the source.
**Fix:** Emit a warning (to stderr) when `dropped / original_count > 0.15` for any shard.

### MO-24 — MEDIUM: No shard/embedding provenance metadata
**Files:** `build_shards.py`, `precompute_all.py`
No JSON written alongside shard or embedding outputs recording which version of the
script, config, and source files produced them.
**Fix:** Write a `{output_dir}/.manifest.json` on completion with script version (git SHA),
args, shard count, record count, and timestamp.

### MO-25 — MEDIUM: Training doesn't log which shards it actually used
**File:** `train_ip_adapter.py`
If a shard causes a crash or is silently skipped, the training log doesn't say which
shard was being read.
**Fix:** Log each shard path when the loader first opens it (one line per shard boundary).

### MO-26 — LOW: heartbeat.json schema not documented or validated
**Files:** `train_ip_adapter.py`, `pipeline_watchdog.sh`, `pipeline_status.sh`
Three separate scripts assume the same key names with no shared schema definition.
**Fix:** Add a `HEARTBEAT_SCHEMA_VERSION` field to the heartbeat; fail fast in watchdog/status
if version is missing or mismatched.

---

## PERFORMANCE

### PE-27 — MEDIUM: `retry 3 60` on 20h precompute is catastrophic on transient failure
**File:** `run_training_pipeline.sh:569`
A single transient error (network blip, Metal timeout) triggers up to 3 full reruns
totalling up to 60h. The per-record resume logic means each retry redoes only failed
records, but the overhead of restarting the Python process and reloading models is ~10min.
**Fix:** Replace with `retry 3 300` (5min backoff) and add a max-elapsed-time guard:
if the total retry time exceeds 30min, abort and require manual intervention.

### PE-28 — MEDIUM: filter_shards.py `pool.imap_unordered` chunksize=1
**File:** `filter_shards.py:265`
Each worker processes one shard and blocks waiting for the next dispatch. With 6 workers
and I/O delays, 10-20% idle time per worker.
**Fix:** Use `chunksize=2` or `chunksize=4`.

### PE-29 — MEDIUM: mine_hard_examples.py scans all output tars at startup
**File:** `mine_hard_examples.py:~237-249`
Opens every output tar to build `existing_ids`. Grows to 5+ min with 100+ shards.
**Fix:** Write a `{output_dir}/.existing_ids.txt` manifest on each mining run; read
that instead of re-scanning tars on resume.

### PE-30 — LOW: pipeline_status.sh runs `du -sh` on every call
**File:** `pipeline_status.sh:435-443`
`du -sh` on 350+ GB dirs traverses every inode, taking 10+ seconds per call.
**Fix:** Cache `du` output in a `.du_cache` file updated at most every 60 seconds.

### PE-31 — LOW: pipeline_status.sh `pgrep` calls are broad
**File:** `pipeline_status.sh:167-179`
`pgrep -f "build_shards"` matches subshells and test processes. Already filtered
for `grep` itself but other false positives are possible.
**Fix:** Use more specific patterns, e.g. `pgrep -f "build_shards\.py"`.

---

## CODE QUALITY

### CQ-32 — MEDIUM: run_training_pipeline.sh is 935+ lines with duplicated chunk logic
**File:** `run_training_pipeline.sh`
Wait loops, JDB_PID handling, and training invocation appear twice (chunk 1 and chunks 2-4)
with slightly different logic. A bug fix in one path is easy to miss in the other.
**Fix:** Extract shared patterns into helper functions at the top of the script.

### CQ-33 — MEDIUM: precompute_all.py global `_W` dict stores model state without synchronization
**File:** `precompute_all.py:~104`
`_W` is module-level and mutated by `_worker_init` and encoding functions. Any future
addition of threading (not multiprocessing) will introduce races.
**Fix:** Pass model references explicitly through function args rather than via globals.

### CQ-34 — LOW: Shard size constant (5000 records) hardcoded in 3+ places
**Files:** `build_shards.py`, `mine_hard_examples.py`, `run_training_pipeline.sh`
If shard format changes, these must all be updated manually.
**Fix:** Define once as a constant or read from a config file.

### CQ-35 — LOW: pipeline_status.sh comment "4 phases" should say "3 chunks (2-4)"
**File:** `pipeline_status.sh:68`
Minor documentation inaccuracy.

### CQ-36 — LOW: pipeline_watchdog.sh `_last_alert` described as debounce but doesn't work
See CA-08 above. The variable is also in scope across loop iterations but isn't
documented as intentionally persistent.

---

# Audit 3 — Deep Review (2026-04-05)

Issues identified in a comprehensive audit covering performance, MLOps, mobility,
directory structure, code quality, and subtle bugs.

---

## HIGH

### A3-01 — HIGH: dataset.py queue timeout has no recovery path
**File:** `train/ip_adapter/dataset.py:474`
`sample_q.get(timeout=120)` raises `RuntimeError` with no retry or shard-skip logic.
Any transient crash in a loader thread (bad tar, Metal stall, memory pressure) kills
training permanently. No way to skip the bad shard and continue.
**Fix:** Add retry with exponential backoff in shard_loader/sample_decoder; emit shard
path and exception chain in error message.

### A3-02 — HIGH: build_shards.py resume assumes exact record order
**File:** `train/scripts/build_shards.py:185`
Records are shuffled with `rng.shuffle()` before processing. On resume, a different
Python hash seed or changed blocklist shifts shard boundaries: shard 001 on run 1 ≠
shard 001 on resume. Output is silently corrupted.
**Fix:** Freeze the shuffle seed explicitly in config, or checkpoint the record→shard
boundary mapping and restore it on resume.

---

## MEDIUM

### A3-03 — LOW (blocks medium/large scale): Shard cache filter checks `{stem}_0000.npz` but files are `{rec_id}.npz`
**File:** `train/train_ip_adapter.py:429`
```python
return (os.path.exists(os.path.join(qwen3_dir, f"{stem}_0000.npz")) and ...)
```
`precompute_all.py` saves as `{rec_id}.npz` (no `_0000` suffix). This check always
fails — all 432 shards appear uncached, training falls back to the 34 shards that
pass a different check. Training dataset is ~12x smaller than it should be.
**Note:** Does not affect `--scale small` (21 shards, already within cached set).
Must be fixed before any `--scale medium` or larger run.
**Fix:** Remove `_0000` suffix: `f"{stem}.npz"`.

### A3-04 — MEDIUM: checkpoint_dir hardcoded to `/Volumes/2TBSSD` in config
**File:** `train/configs/stage1_512px.yaml:48`
```yaml
checkpoint_dir: "/Volumes/2TBSSD/checkpoints/stage1"
```
Absolute path to external SSD. Won't work on any other machine or if SSD mounts
under a different name. Stage 2 correctly uses a relative path `"checkpoints/stage2"`.
**Fix:** Use a relative path and resolve via `DATA_ROOT` in the pipeline script, as is
done for all other data paths.

### A3-05 — MEDIUM: Watchdog auto-detect still searches wrong paths
**File:** `train/scripts/pipeline_watchdog.sh:60`
Auto-detect fallback searches `$TRAIN_DIR/checkpoints/...` but checkpoints now live
at `/Volumes/2TBSSD/checkpoints/`. If watchdog is started manually or the `--heartbeat`
flag is omitted, it never finds the heartbeat.
**Fix:** Pass `CKPT_DIR` into lock file at pipeline startup; read it in watchdog auto-detect.

### A3-06 — MEDIUM: build_shards.py error log uses wrong dict key
**File:** `train/scripts/build_shards.py:309`
```python
f"rec={rec['id']} src={rec.get('src', '?')}: {_exc}"
```
Key is `'shard'` not `'src'`; always prints `src=?`. Corrupt image source is undiagnosable.
**Fix:** Use `rec.get('shard', '?')`.

### A3-07 — MEDIUM: mine_hard_examples.py manifest append race
**File:** `train/scripts/mine_hard_examples.py:450`
Manifest is appended *after* tar extraction. Crash between write and append causes
duplicate extraction on next run.
**Fix:** Write manifest atomically before extraction or use per-tar sentinel flags.

### A3-08 — MEDIUM: Heartbeat JSON read non-atomically in status/watchdog
**File:** `train/scripts/pipeline_status.sh:260`
Status script reads `heartbeat.json` while training may be mid-write. Partial JSON
causes silent parse failure; status shows no progress. Training already writes via
`.tmp` → rename (atomic), so this is safe — but watchdog reads directly without
the same guard.
**Fix:** Confirm watchdog also handles `json.JSONDecodeError` gracefully; add retry.

### A3-09 — MEDIUM: Anchor mix ratio math over-represents anchors
**File:** `train/ip_adapter/dataset.py:350`
With `anchor_mix_ratio=0.20` and `hard_mix_ratio=0.05`:
```python
n_anchor = int(len(epoch_paths) * anchor_mix_ratio / max(remaining_ratio, 0.01))
         = len(paths) * 0.20 / 0.75 = len(paths) * 0.267
```
Intended 20% becomes ~27%. Anchors are over-represented by ~33%.
**Fix:** Compute n_anchor relative to `len(paths)` (main set only):
```python
n_anchor = int(len(paths) * anchor_mix_ratio / (1 - anchor_mix_ratio - hard_mix_ratio))
```
Or document that the ratio is relative to the main set only.

### A3-10 — MEDIUM: Prefetch threads never signal EOF on normal training completion
**File:** `train/ip_adapter/dataset.py:343`
Loader threads loop `while True` indefinitely. After training finishes, threads remain
alive consuming CPU. The `None` sentinel only fires after 10 consecutive errors.
**Fix:** Add a threading.Event stop signal; set it after the training loop exits.

### A3-11 — MEDIUM: filter_shards.py shard index parsing will crash on stray files
**File:** `train/scripts/filter_shards.py:243`
```python
int(os.path.splitext(os.path.basename(s))[0])
```
A `.tar.filtered` file left in the shard dir causes `int()` to raise. No try/except.
**Fix:** Use `s.split('.')[0]` or wrap in try/except with a warning.

### A3-12 — MEDIUM: Relative path expansion not applied to model paths
**File:** `train/train_ip_adapter.py:1404`
Data paths (qwen3_cache_dir, vae_cache_dir, etc.) are expanded to absolute using
`DATA_ROOT`. Model paths (`flux_model_dir`, `qwen3_model_dir`) are not expanded and
rely on CWD being the repo root. Fragile if launched from a different directory.
**Fix:** Apply the same relative-path expansion to all paths in all config sections.

---

## LOW

### A3-13 — LOW: Dead legacy pgrep for non-existent scripts
**File:** `train/scripts/pipeline_status.sh:203`
```bash
pgrep -f "precompute_qwen3\.py"  # these scripts no longer exist
pgrep -f "precompute_vae\.py"
```
These individual scripts were consolidated into `precompute_all.py`. Always returns
false. Dead code.
**Fix:** Remove.

### A3-14 — LOW: EMA update frequency not tuned for Stage 2
**File:** `train/configs/stage2_768px.yaml:28`
`ema_update_every: 10` was tuned for 105K steps. Stage 2 is 20K steps at
~8s/step — every 20 steps would save wall-clock time with negligible quality loss.
**Fix:** Consider `ema_update_every: 20` for Stage 2.

### A3-15 — LOW: mine_hard_examples.py manifest scan counts .txt and .jpg separately
**File:** `train/scripts/mine_hard_examples.py:248`
Both `.jpg` and `.txt` members are added to `existing_ids` set. Since both share
the same stem, the set deduplicates, but the loop does 2× the work unnecessarily.
**Fix:** Break after first match per stem, or filter to `.jpg` only.

### A3-16 — LOW: SigLIP dependency check incomplete
**File:** `train/train_ip_adapter.py:935`
Checks `_HAS_MFLUX` but SigLIP also requires `mlx_vlm` or `transformers`. If
mflux is present but not mlx_vlm, the error is cryptic.
**Fix:** Add explicit try-import for `mlx_vlm` / `transformers` and print a clear message.

### A3-17 — LOW: Inconsistent array expansion guard in pipeline_start.sh
**File:** `train/scripts/pipeline_start.sh:98`
Fixed for the `PIPELINE_ARGS` empty-array case but the same pattern appears in
`run_training_pipeline.sh` with `TRAIN_ARGS` and `CHUNK_TRAIN_ARGS`. If those are
ever empty, bash 3.2 nounset will crash the same way.
**Fix:** Audit all `${ARRAY[*]}` expansions in pipeline scripts; guard with `${#ARRAY[@]} -eq 0`.

### A3-18 — LOW: No stage2 VAE cache at 768px
**File:** `train/configs/stage2_768px.yaml:19`
`vae_cache_dir: null` with no comment explaining that 768px VAE cache must be
pre-computed separately before Stage 2 starts. A user launching Stage 2 without
pre-computing will hit confusing errors.
**Fix:** Add a preflight check in the pipeline that pre-computes 768px VAE cache
before Stage 2 training begins, or document this requirement prominently.

### A3-19 — LOW: Shell array `${ARRAY[*]:-}` used inconsistently
**File:** `train/scripts/pipeline_start.sh:103`
```bash
CMD="caffeinate -i -d bash $PIPELINE_SCRIPT ${PIPELINE_ARGS[*]:-}"
```
`[*]` joins on IFS (space), breaking args that contain spaces. Should use `"${PIPELINE_ARGS[@]}"`.
**Fix:** Use `"${PIPELINE_ARGS[@]}"` throughout; quote array expansions.

### A3-20 — LOW: pipeline_stop.sh checkpoint path lookup still wrong
**File:** `train/scripts/pipeline_stop.sh`
Stop script reports "No checkpoint found in `train/checkpoints`" (wrong path).
Checkpoints are now at `/Volumes/2TBSSD/checkpoints/stage1`. The message is
cosmetically wrong and may confuse operators.
**Fix:** Read checkpoint_dir from config or lock file, same as pipeline_status.sh does.
