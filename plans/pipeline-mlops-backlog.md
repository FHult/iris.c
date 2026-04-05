# Pipeline MLOps Backlog

Issues identified during pipeline audit (2026-04-01) that were not fixed in the initial pass.
Items are ordered by operational risk.

---

## MLX-10 — No model lineage in checkpoints ✅ DONE

**Problem:** Checkpoints contain only weights. There is no record of which config file,
data shards, git commit, or training args produced a given `.safetensors` file.
If you want to reproduce a checkpoint or understand why two checkpoints differ, you have no
reliable record beyond filename and wall-clock timestamp.

**Risk:** Medium. Silent config drift between chunks is invisible. Any ablation or
post-mortem is guesswork.

**Fix:** At checkpoint save time, write a sidecar `step_NNNNNN.json` with:
- git commit SHA (`git rev-parse HEAD`)
- config file path + full resolved YAML dict
- training args passed to the script
- timestamp, step, loss

Cost: negligible (one JSON write per checkpoint).

---

## MLX-11 — No pre-flight doctor check ✅ DONE

**Problem:** `run_training_pipeline.sh` starts the multi-day pipeline with no upfront
validation of environment, dependencies, or data layout.
Failures (missing Python packages, wrong MLX version, missing model weights, wrong shard
path) surface hours into the run, after non-resumable steps have already consumed time.

**Risk:** Medium. A single missing pip package can kill a 12-hour step.

**Fix:** Add a `doctor` subcommand (or run inline at startup) that checks:
- Python + MLX version meets minimum (`mlx >= 0.31.1`)
- Required Python packages importable (`turbojpeg`, `safetensors`, `yaml`, `wandb` if enabled)
- Model weight directory exists and contains expected files
- `DATA_ROOT` writable with ≥ 50 GB free
- Shard path non-empty if resuming past step 4
- Precompute cache non-empty if training is about to start
- `tmux` available if sessions are used

Exit with a clear per-check ✅/❌ table before doing anything destructive.

---

## MLX-12 — Hard example mining failure is silent in pipeline log ✅ DONE

**Problem:** If `mine_hard_examples.py` crashes, the pipeline logs a one-line warning and
continues. The Python traceback is swallowed. The operator has no way to diagnose the root
cause from the pipeline log alone.

**Status:** Partially fixed (fix 9 above now prints the re-run command and marks stderr).
Remaining gap: the Python traceback itself is still not surfaced because the process output
goes to the pipeline log only if it was captured with `tee`; bare `python ...` output
goes to stdout/stderr which is whatever the tmux session has.

**Fix:** Wrap the mining call in a subshell that captures both stdout+stderr and tees to a
dedicated log file `$DATA_ROOT/logs/mine_hard_chunk${CHUNK}.log`. On failure, print the
last 30 lines of that log to the pipeline log.

---

## MLX-13 — No alerting on training stall ✅ DONE

**Problem:** If training hangs (deadlock in prefetch thread, OOM kill, Metal timeout),
the heartbeat file goes stale. `pipeline_status.sh` already detects this (⚠️ stale
heartbeat) but only if you actively poll the status. There is no push notification.

**Risk:** Low–Medium. A stall costs wall-clock time proportional to how long until the
operator notices.

**Fix options (in order of effort):**
1. A cron job (every 5 min) that runs `pipeline_status.sh`, checks heartbeat age, and
   sends a macOS notification (`osascript -e 'display notification ...'`) if stale.
2. `wandb` alert (already supported in wandb; requires the wandb project to be configured).
3. A background watcher process launched by the pipeline that `kill -0` the training PID
   and sends a notification if it exits unexpectedly.

---

## MLX-14 — build_shards concurrent filter can miss final window ✅ DONE

**Problem:** The background `filter_shards` loop runs every 60 seconds. If
`build_shards` writes a shard in the last 59 seconds before it exits, that shard may not
be filtered until the explicit final pass. The final pass runs synchronously after
`build_shards` exits, so it will catch it — this is correct. But the log says
"background filter running" which implies continuous coverage, which is slightly misleading.

**Risk:** Low. The final pass is a hard guarantee; this is cosmetic.

**Fix:** Log the final-pass invocation as "Final filter pass (catching remaining shards)"
and suppress the "running" implication in background loop comments.

---

## MLX-15 — Siglip precompute not in precompute_all done gate ✅ DONE

**Problem:** `precompute_all.py` writes `$PRECOMP_DIR/.done` when qwen3+vae are complete.
Siglip is optional and runs separately. The training script reads `siglip_cache_dir` from
config; if the cache exists but is incomplete (e.g., only 11/34 shards done when training
starts), some batches fall back to zero-feature mode silently.

**Risk:** Medium. Zero-feature batches produce incorrect gradients; the effect is diluted
by batch diversity but the training signal is degraded for those steps.

**Fix:** In `train_ip_adapter.py`, at startup when `siglip_cache_dir` is set, count the
cache files and compare against the shard count. Warn loudly (or assert) if coverage < 95%.
Optionally: add `--siglip-min-coverage 0.95` flag to fail fast.
