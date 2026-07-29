# Flywheel Precompute Architecture (Claude/AI Summary)

## History / Problem
Early flywheel "warmup" runs (e.g. flywheel_warmup_run1.yaml) selected shards from cold pool, staged them, and launched training using `stage1_512px_online.yaml` (qwen3/vae/siglip_cache_dir: null).

This forced live encoding inside `train_ip_adapter.py` training loop for every batch:
- `if text_np is None and text_encoder: _encode_text(...)` (Qwen3)
- `elif vae is not None: _vae_encode(...)`
- `elif siglip is not None: siglip(...)`

Even after adding per-step `_ensure_live_encoders` / `_release_live_encoders` (unload after encode window before Flux forward) + `[online-encode]` debug prints + 0.55 memory cap, it was wasteful:
- Encoders (re)loaded/unloaded per step (SigLIP full load often).
- Encode work repeated on data replay within 1k-5k step iters.
- Latency + memory pressure during training steps.
- Crashes (jetsam) until the unload hack.

The normal chunk pipeline did it right: dedicated precompute phase (models once, 1 tar read, writes npz to versioned cold), *then* training with caches only (Flux+adapter).

## The Fix (Implemented)
Flywheel now mirrors the pipeline for its selected shards (see `_run_flywheel_loop` in `orchestrator.py` and `publish_precomp_from_flywheel_iter` in `data_stager.py`).

### Per-Iteration Flow (after select + stage)
1. **Precompute phase** (under GPU lock acquired for whole iter):
   - Run `precompute_all.py --shards <staging_dir>` (always --siglip) → per-iter flat tree `staging_dir/precomputed/{qwen3,vae,siglip}/`.
   - Uses precomp's own 14GB cap, inter-shard releases, prefetch, skip-already-done.
   - Heartbeat "precomputing".
   - (pre_cmd built using draft config for flux_model etc.)

2. **Force cached training**:
   - Override temp train config (even if base was online.yaml) to set the 3 cache_dirs to the per-iter tree.
   - Capture `_final_train_cfg_for_publish`.
   - Launch training in TMUX_TRAIN_WIN (now takes fast cached path: only Flux+adapter, npz in loader, no live encoders).
   - Monitor until window gone.

3. **Post-training publish** (still inside lock try, before unlink):
   - Call `_stager.publish_precomp_from_flywheel_iter(src=precomp_base, selected=shard_ids, training_cfg=_final...)`.
   - Inside stager (if training_cfg):
     - For each encoder: `subset=encoder_config_subset(encoder, cfg)`, `ver=version_hash(subset, git_sha)`.
     - `pcache=PrecomputeCache(cold_precomp_root, encoder, subset, git_sha)`.
     - Copy matching .npz into `pcache.cache_dir()` (the v_XXXXXX/ on cold).
     - `pcache.write_manifest_incomplete()` + `mark_complete(...)` (writes manifest.json with complete=True, counts, timestamps, config; **atomically updates "current" symlink**).
     - This makes data fully participate: versioned invalidation, PrecomputeCache.is_complete/all_records, effective_dir, future stagings (stager follows "current"), list_versions etc.
   - Fallback (no cfg): copy to existing current target dir (old behavior).
   - **Always** `shutil.rmtree(src_precomp_base, ignore_errors=True)` at end (ephemeral staging copy cleaned; canonical data on cold).

4. Continue with metrics, score updates, etc. (as before).

### Key Benefits / Why
- **Efficiency**: Encode once (precomp's optimized path) vs. per-step live. Training steps are pure cached (faster, lower mem, no churn).
- **Reusability (no waste)**: Data lands in cold under proper versioned layout + manifest + current. Future flywheel iters (overlapping shards via stager) and main pipeline get it for "free".
- **Pipeline parity**: Same mechanisms as chunk precomp/promote (see `_promote_chunk`, precompute_all's PrecomputeCache usage at start/end, stager's version logic).
- **Incremental**: Re-select = cheap skips in precomp.
- **Compatibility**: Original online path + unload/debug prints in `train_ip_adapter.py` (the `_ensure`/`_release` + banner) remain for manual runs or partial precomp fallback. (See lines ~707, 1235, 1542 etc. for [online-encode] and ensure/release.)
- **Config updates**: warmup_run1 now points to cached yaml (comments explain auto-precomp). online.yaml comments note it's mainly for manual now.

### Files Touched
- `train/scripts/orchestrator.py`: big explanatory comment block (Claude-readable), precomp setup, capture _final cfg, publish call inside try, removal of old post-finally publish.
- `train/scripts/data_stager.py`: enhanced `publish_precomp_from_flywheel_iter` (versioned path + fallback + rmtree at end; detailed docstring).
- Configs: comments + run1 training_config switch (for completeness).
- Earlier (still relevant): unload/debug in `train/train_ip_adapter.py` (for non-flywheel or fallback online).

### Related Code (for full understanding)
- `cache_manager.py`: PrecomputeCache (write_manifest_*, mark_complete, cache_dir, version(), current_dir, effective_dir), version_hash, encoder_config_subset, get_git_sha, _atomic_symlink.
- `precompute_all.py`: how it uses PrecomputeCache for manifests when output matches version; skip logic.
- `data_stager.py` (other methods): stage_iteration_shards (how it reads from cold "current" for future iters), archive logic.
- `orchestrator.py` (normal path): _start_precompute, _promote_chunk (the model we copied).
- `train_ip_adapter.py` (loader + training): how cache_dirs control np vs live; the ensure/release for online compatibility.
- `plans/warmup-campaign-runbook.md`: original flywheel goals/motivation.

This change was driven by the observation that "actions same but efficiency vastly different" — precompute the selected set once (like pipeline) instead of live inside training.

See git history (commits ea7d211, 04d2836, a904519, bd0eab0, 5d3a157) for the evolution (unload first for stability, then precomp, then full publish versioning).

If re-running flywheel on cold, it will now precomp selected shards each iter, train cached, publish versioned+manifest+current to cold, rmtree temp, and logs will show the precomp/publish steps.
