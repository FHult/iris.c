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
