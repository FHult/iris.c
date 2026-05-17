# Pipeline Efficiency Audit — iris.c Training Pipeline

**Audit date:** 2026-05-16  
**Scope:** train/scripts/ — all V2 pipeline scripts  
**Method:** static code analysis; no runs executed

---

## Executive Summary

Five highest-impact waste items:

1. **CLIP neural embedding runs after build_shards on already-filtered data, but before a second dedup round that removes a fraction of those same images.** The `clip_embed` → `clip_index` → `clip_dups` sequence (three sequential GPU steps, ~1–2h each) happens on the production shards. The blocklist output is only used by the *next* chunk's `build_shards`, not the current one — meaning the current chunk trains on duplicates the CLIP pass already flagged.

2. **`mine_hard_examples.py` loads Flux Klein 4B (7 GB) and the IP-Adapter on every chunk just to score 5,000 records (~0.1% of a 5M-image chunk).** The eval budget default is 5,000 out of potentially 500,000+ precomputed records. The model loading cost is amortised over that tiny sample, and the mining output (2,000 records) is only 0.4% of a shard by count.

3. **`precompute_all.py` coverage verification re-opens every shard tar a second time after encoding.** At lines 1139–1153, after completing Qwen3 + VAE encoding, it calls `iter_shard(shard_path)` for every shard in `work_items` to count expected records. This is a full tar re-read (header parse for every member) on potentially hundreds of shards.

4. **`clip_dedup.py find-dups` reconstructs all stored vectors from the FAISS flat index** (line 437: `index.reconstruct_n(0, n, all_vecs)`) to do the KNN pass. For a cumulative index across 4 chunks, this means reading back and searching the entire corpus each time, including vectors from prior chunks that have already been checked against each other.

5. **`dedupe_filter.py` (now the primary dedup path) and `clip_dedup.py` (post-build dedup) both embed images with CLIP, but on different stages of the same data.** `dedupe_filter` runs on raw converted tars (pre-build), then `clip_embed`/`clip_index`/`clip_dups` runs again on the assembled production shards. The same images get CLIP-embedded twice — once before and once after `build_shards`.

---

## 1. Filter Ordering: Dedup Happens Before and After Shard Build — Twice

### Description

The V2 pipeline runs CLIP deduplication at two points on the same underlying images:

1. **`dedupe_filter` step** (between `convert` and `build_shards`): calls `dedup_wds_tar()` in `clip_dedup.py` (`dedupe_filter.py` line 36: `from clip_dedup import dedup_wds_tar`). This embeds all images in the raw converted tars, searches the cumulative FAISS index, rewrites tars in-place removing duplicates.

2. **`clip_embed` → `clip_index` → `clip_dups` steps** (between `build_shards` and `precompute`): embeds all images in the built production shards (`clip_dedup.py cmd_embed`, lines 247–341), extends the index, and writes a new blocklist.

The blocklist output from step 2 goes to `--blocklist` in `build_shards.py` (orchestrator.py line 1210), but this blocklist is consumed by the *next* chunk's `build_shards` call — not by the current chunk. So:
- Current chunk images get CLIP-embedded in `dedupe_filter` (GPU work).
- The same images get CLIP-embedded again in `clip_embed` (more GPU work).
- The `clip_dups` blocklist only protects the *next* chunk from cross-chunk duplicates.
- The current chunk already has within-chunk duplicates removed by `dedupe_filter`, but cross-chunk duplicates remain in the shards being trained on *right now*.

### Magnitude

Each CLIP embed pass on a full chunk (e.g. 80 shards × 5,000 images = 400,000 images) takes roughly 400,000 / 39 img/s ≈ 2.8 GPU-hours. Running it twice doubles that cost per chunk, wasting approximately 2–3 GPU-hours per chunk, or 8–12 GPU-hours total for a 4-chunk run.

### Affected Scripts and Lines

- `dedupe_filter.py` line 36: imports `dedup_wds_tar` from `clip_dedup`
- `orchestrator.py` lines 1255–1301: `_start_clip_embed`, `_start_clip_index`, `_start_clip_dups`
- `orchestrator.py` line 1210: blocklist passed to `build_shards` (next chunk, not current)
- `clip_dedup.py` lines 247–341: `cmd_embed` runs full CLIP pass on built shards

### Fix Direction

If `dedupe_filter` is already running CLIP and building a chunk-local index, the subsequent `clip_embed`/`clip_index`/`clip_dups` steps are redundant for within-chunk dedup. The post-build CLIP pass is only useful for cross-chunk dedup, but the blocklist it produces arrives too late to clean the current chunk.

**Option A:** Eliminate the `clip_embed`/`clip_index`/`clip_dups` steps entirely, and modify `dedupe_filter` to also search the cumulative cross-chunk index (not just the chunk-local one). `dedupe_filter` already calls `dedup_wds_tar` which accepts `index_path` — wire it to the cumulative index.

**Option B:** Keep `dedupe_filter` as the primary dedup (it's pre-build, so it prevents low-quality images from entering `build_shards` at all). Demote `clip_embed`/`clip_index`/`clip_dups` to an optional background step that updates the cross-chunk index without blocking training.

---

## 2. Hard Example Mining: Full Model Load for 0.1% Sampling

### Description

`mine_hard_examples.py` loads Flux Klein 4B (frozen, ~7 GB) plus the IP-Adapter to score a random sample of records. The default eval budget is `--eval-records 5000` (line 299). The precomputed pool for a chunk can contain 500,000+ records (100 shards × 5,000 records). So mining samples and evaluates 1% or less of the pool.

The model loading sequence (lines 421–467) is:
```python
flux = Flux2Klein(model_path=args.flux_model, quantize=None)   # ~7 GB
adapter = IPAdapterKlein(...)                                   # ~0.5 GB
```

This adds significant startup overhead (model loading + Metal warmup) to what is ultimately a 5,000-record scoring job. The Flux model is frozen — its weights never change during mining. Yet it is loaded fresh on every mining invocation (after every chunk).

Additionally, mining evaluates only `5,000` records but extracts only `top_k=2,000` (lines 298–302). The result is 2,000 records packaged into hard example `.tar` files at 500 records/file. These 2,000 records represent only 0.04% of a 5M-image chunk's data.

When siglip cache is not available (which happens on chunks where siglip was not precomputed), the fallback at lines 554–564 opens each shard `.tar` individually per record to re-extract the JPEG for on-the-fly SigLIP encoding. This is extremely expensive: for 5,000 records in potentially hundreds of different shards, this requires up to 5,000 shard `.tar` open operations.

### Magnitude

- Model load time: ~2–5 minutes per chunk × 4 chunks = 8–20 wasted minutes for model load alone.
- Shard re-open fallback: if SigLIP cache is absent and 5,000 records span many shards, each `tarfile.open()` on a large shard (.tar files are ~100–500 MB) just to extract one JPEG is catastrophically slow. At 500 ms/open × 5,000 opens = 2,500 seconds = 41 minutes of pure I/O overhead.

### Affected Scripts and Lines

- `mine_hard_examples.py` lines 421–467: model loading
- `mine_hard_examples.py` lines 295–302: `--eval-records 5000` default
- `mine_hard_examples.py` lines 543–564: per-record shard open fallback when SigLIP cache absent

### Fix Direction

1. **Increase eval-records** to a larger fraction (e.g., 50,000 records) to get a more representative ranking at modest extra cost — the model is already loaded, so evaluating 10× more records costs 10× the forward pass time but the same model load time.

2. **Group shard opens by shard** when SigLIP cache is absent. The fallback (lines 554–564) opens `shard_path` inside a per-record loop. Instead, pre-group candidates by shard (as is done in Pass 2 at lines 604–629) and open each shard once, extracting all needed records in one pass. This is already done efficiently in Pass 2 but not in the SigLIP fallback in Pass 1.

3. **Make SigLIP precompute mandatory** when mining is enabled (it's optional in precompute but the mining fallback is expensive when absent). The orchestrator should enforce this if `--siglip` is set.

---

## 3. Coverage Verification Re-Opens Every Shard After Encoding

### Description

At the end of `precompute_all.py`, after all encoding is complete (Qwen3 + VAE passes), the coverage verification loop at lines 1134–1168 iterates over every shard in `work_items` and calls `iter_shard(shard_path)` to count expected records:

```python
for shard_path, *_ in work_items:
    stem = os.path.splitext(os.path.basename(shard_path))[0]
    try:
        expected = sum(1 for _ in iter_shard(shard_path))   # re-opens the tar
    ...
```

`iter_shard()` opens the tar file and yields `(id, jpg_bytes, caption)` for every record — reading all JPEG bytes and text. This is a full tar-reading pass, not a header-only scan.

For 100 shards × 5,000 records each, this is 100 tar reads, each requiring sequential I/O through ~100–500 MB of data. This adds an unnecessary I/O pass after the (already expensive) encoding is complete.

### Magnitude

If each shard is 200 MB and read at 500 MB/s (NVMe), that's 0.4 seconds per shard × 100 shards = 40 seconds minimum for pure I/O. In practice, the tar member parsing overhead and context switching make it closer to 2–5 minutes for 100 shards. This happens on every precompute run (not just the first one).

### Affected Scripts and Lines

- `precompute_all.py` lines 1134–1168: coverage verification loop
- `precompute_all.py` line 1143: `expected = sum(1 for _ in iter_shard(shard_path))` — reads all JPEG bytes

### Fix Direction

The record count per shard is already known during encoding: `precompute_all.py` iterates `iter_shard()` in Phase 1 (`_process_shard_inner` lines 628–648). The count could be recorded as part of the result dict returned by `process_shard()` and accumulated in the main loop. Then the coverage check uses the accumulated count instead of re-opening each tar.

Alternatively, replace `iter_shard()` in the coverage check with a header-only scan: open the tar and count members matching `*.jpg`/`*.jpeg`/`*.png` without calling `extractfile()`. This would reduce I/O by ~99% (member headers are small; only the data portions are large).

---

## 4. FAISS find-dups Reconstructs All Vectors From Scratch Each Chunk

### Description

`clip_dedup.py cmd_find_dups` (lines 411–498) loads the cumulative FAISS index and then reconstructs all stored vectors to search for duplicates:

```python
index   = faiss.read_index(str(index_path))
all_ids = ids_path.read_text().splitlines()
n       = index.ntotal

d   = index.d
all_vecs = np.zeros((n, d), dtype=np.float32)
index.reconstruct_n(0, n, all_vecs)   # lines 436–438
```

It then does a KNN search of every vector against the full index in batches (lines 451–470). This is O(n²) in the total corpus size. As the cumulative index grows across chunks (chunk 1: ~400K vectors, chunk 2: ~800K, chunk 3: ~1.2M, chunk 4: ~1.6M), the search time and memory cost grow quadratically.

By chunk 4, `all_vecs` is 1.6M × 768 float32 = ~4.7 GB just for the embedding matrix. The KNN search with k=5 over 1.6M vectors takes significant time.

Additionally, `find-dups` is re-checking vectors from prior chunks that have already been compared against each other. Only the newly added vectors (current chunk's embeddings) are truly new; the cross-chunk pairs involving prior-chunk vectors were already checked when those chunks ran `find-dups`.

### Magnitude

At 1.6M vectors (chunk 4), with FAISS `IndexFlatIP.search` at ~10,000 vectors/second for k=5 on CPU: 1.6M vectors / 10,000 vec/s = 160 seconds = ~2.7 minutes. Memory: ~4.7 GB for `all_vecs`. This doubles or triples vs. chunk 1. Also, the `reconstruct_n` call (reading all vectors from the flat index) is itself O(n) in disk I/O.

### Affected Scripts and Lines

- `clip_dedup.py` lines 428–498: `cmd_find_dups`
- `clip_dedup.py` lines 436–438: `index.reconstruct_n(0, n, all_vecs)`
- `clip_dedup.py` lines 450–470: KNN search over all n vectors

### Fix Direction

Track which index positions correspond to the current chunk (new vectors). Only new-to-new and new-to-old comparisons are needed; old-to-old pairs were already checked in prior chunk runs. The cumulative `.ids` sidecar file already records all IDs in add order — the slice `[prior_n:]` would be the current chunk's vectors. Search only those vectors against the full index:

```python
new_vecs = all_vecs[prior_n:]  # only current chunk's vectors
D, I = index.search(new_vecs, k)
```

This reduces the search from O(n_total²) to O(n_new × n_total) per chunk, roughly a 4× reduction by chunk 4.

---

## 5. Precompute Runs on All Shards Even When --max-shards Caps Training

### Description

`precompute_all.py` uses `--max-shards` to randomly select a subset of shards to encode (lines 864–897). However, this selection is random — it does not preferentially select shards that are not yet in the training rotation, nor does it guarantee alignment with which shards `build_shards` actually produced for this chunk.

More critically, the `--new-shards-first` heuristic (lines 869–890) requires checking every candidate shard for whether it has existing precomputed output:

```python
def _has_output(shard_path: str) -> bool:
    stem = os.path.splitext(os.path.basename(shard_path))[0]
    return any(
        any(
            f.startswith(stem + "_") and f.endswith(".npz")
            for f in os.listdir(d)            # O(N_files) per shard per dir
        )
        for d in out_dirs
        if os.path.isdir(d)
    )
```

This is called for every shard in the pool (potentially hundreds) and for each shard does a linear scan of the precomputed output directory (which can contain millions of `.npz` files). For a pool of 200 shards and 500,000 `.npz` files, this is 200 × 500,000 = 100M string comparisons just for the shard selection step.

### Magnitude

At 200 shards × `os.listdir` over a 500,000-file directory, if `os.listdir` takes 1 second (reasonable for a large directory), this is 200+ seconds = 3+ minutes of overhead before any encoding begins. The linear scan per shard makes it O(n_shards × n_npz_files).

### Affected Scripts and Lines

- `precompute_all.py` lines 869–895: `--new-shards-first` logic
- `precompute_all.py` lines 872–879: `_has_output()` function with O(N) scan
- `precompute_all.py` lines 883–895: `new_shards`/`old_shards` partitioning

### Fix Direction

Build a set of shard stems that already have output once (one `os.listdir` call, extract the stem prefix before `_`), then check each shard against that set in O(1):

```python
existing_stems = {f.split("_")[0] for f in os.listdir(out_dir) if f.endswith(".npz")}
new_shards = [s for s in shards if Path(s).stem not in existing_stems]
```

This turns O(n_shards × n_files) into O(n_files + n_shards).

---

## 6. Ablation Harness Runs Full Training From Scratch — No Checkpoint Reuse

### Description

`ablation_harness.py` runs each combination by launching `train_ip_adapter.py` as a subprocess with `--max-steps` set to `steps_per_run` (typically 300–12,000 steps). The run configuration (line 952) sets:

```python
cfg["output"]["checkpoint_every"]     = steps * 100   # effectively never
cfg["output"]["skip_checkpoint_save"] = True
```

Each ablation run starts from scratch: the base config is loaded, the trainer initialises from random weights (or from whatever the `data.shard_path` and base adapter state is), and runs for `steps_per_run` steps. There is no mechanism to warm-start ablation runs from a shared checkpoint that has already seen a common data prefix.

For a 300-run ablation campaign at 12,000 steps each, this is 3.6 million trainer steps total, each starting from the same cold-start position. If 200 of those runs could have started from a shared 5,000-step warm checkpoint (common warmup), that would save 200 × 5,000 = 1M redundant steps.

Additionally, the `_build_run_config` function (lines 910–960) sets `anchor_shard_dir=None` and `hard_example_dir=None` (lines 929–930), disabling those inputs. The ablation dataset is smaller (fewer shards) but all embeddings are re-read from disk per run. The precomputed caches are shared (wired to `_DEFAULT_QWEN3`, `_DEFAULT_VAE`, `_DEFAULT_SIGLIP` at lines 124–126), which is correct. But the Metal PSO warmup (graph compilation) runs fresh for each ablation run, adding ~10 minutes of compilation overhead per run that is not tracked in the step metrics.

### Magnitude

For a medium ablation (12 combos × 12,000 steps at ~0.19 s/step): 12 × 12,000 × 0.19 ≈ 27,360 seconds ≈ 7.6 hours of GPU time. If warmup overhead is 10 minutes per run: 12 × 10 = 120 minutes = 2 hours of untracked GPU time.

For a long-term ablation (300 combos × 12,000 steps): 300 × 12,000 × 0.19 ≈ 684,000 seconds ≈ 190 hours. If each run saves 5,000 warmup steps via a shared checkpoint: 300 × 5,000 × 0.19 ≈ 285,000 seconds ≈ 79 hours saved.

### Affected Scripts and Lines

- `ablation_harness.py` lines 910–960: `_build_run_config`
- `ablation_harness.py` lines 952–954: `checkpoint_every = steps * 100`, `skip_checkpoint_save = True`
- `ablation_harness.py` line 929–930: `anchor_shard_dir=None`, `hard_example_dir=None`
- `ablation_harness.py` lines 1229–1280: `_run_one` subprocess launch — no warm-start

### Fix Direction

Generate a shared warmup checkpoint (e.g. 2,000 steps) once at the start of an ablation campaign, then warm-start all ablation runs from that checkpoint. This eliminates the cold-start learning curve from every run's scoring. The `_run_one` function already knows `ckpt_dir`; the harness could pre-compute the shared checkpoint and pass `--resume <shared_ckpt>` to all runs.

Metal PSO warmup can be shared via the machine-wide Metal cache (`training_warmup` sentinel logic in orchestrator.py lines 1496–1504) — one warmup call suffices for all subsequent runs with the same bucket shapes.

---

## 7. JDB Downloads: All 202 Tgzs Pre-Screened Only By Chunk Assignment

### Description

`download_convert.py` downloads JDB tgzs sequentially. The orchestrator assigns tgzs 000–049 to chunk 1, 050–099 to chunk 2, etc. (orchestrator.py lines 1168–1183). There is no quality pre-screening of tgzs before download — all 50 tgzs per chunk are downloaded and converted regardless of their quality. The `dedupe_filter` step removes low-quality records post-conversion, but the download and conversion cost has already been paid.

The quality of JDB tgzs is not uniform. Some tgzs contain higher proportions of low-resolution, corrupt, or short-caption images. Running `dedupe_filter` on already-downloaded tgzs shows the per-tar filtering statistics, but this information is not fed back to prevent re-downloading high-waste tgzs in future campaigns.

Additionally, BACKLOG.md notes (PIPELINE-25b) that downloads write each tgz to disk before conversion, then read it back. For tgzs of ~2-3 GB each, 50 tgzs per chunk means 100–150 GB of disk reads/writes that serve as intermediate staging only.

### Magnitude

If 20% of JDB records are filtered by `dedupe_filter` (quality + dedup), then ~20% of download and conversion compute is wasted. At 50 tgzs per chunk × ~30 min/tgz = 25 hours of download+convert per chunk, 20% waste = 5 hours per chunk, 20 hours total.

The write-then-read pattern for tgz files (PIPELINE-25b) adds one full NVMe write + read cycle per tgz: 50 tgzs × 2.5 GB = 125 GB of avoidable disk I/O per chunk.

### Affected Scripts and Lines

- `download_convert.py` (the downloader itself; not read in detail)
- `dedupe_filter.py`: runs quality filter post-download
- `orchestrator.py` lines 1168–1183: `_start_download_convert`
- `BACKLOG.md` (PIPELINE-25b): stream-convert optimization not yet implemented

### Fix Direction

1. Build a per-tgz quality score (fraction passing quality filter, CLIP dedup rate) from `dedupe_filter` results and persist it in `shard_scores.db`. Use this to deprioritize or skip very low-quality tgzs in future campaigns.

2. Implement PIPELINE-25b (stream-convert): pipe the HF download directly into the converter without writing the raw tgz to disk. Saves 125 GB of disk I/O per chunk.

---

## 8. Post-Precompute Coverage Check: Directory Listing O(N) per Shard per Encoder

### Description

At precompute_all.py lines 1148–1154, the coverage check does:

```python
for out_dir in _check_dirs:
    actual = sum(1 for f in os.listdir(out_dir)
                 if f.startswith(stem + "_") and f.endswith(".npz"))
```

This calls `os.listdir(out_dir)` once per shard per encoder directory. If there are 100 shards and 2 encoder directories (qwen3, vae), that is 200 calls to `os.listdir` on directories containing up to 500,000 files each. Each call reads the full directory listing and then filters it with Python string matching.

### Magnitude

`os.listdir` on a 500,000-file directory may take 0.5–2 seconds (filesystem-dependent). 200 calls × 1 second = 200 seconds = ~3 minutes of pure directory listing overhead, separate from the coverage check computation itself.

### Affected Scripts and Lines

- `precompute_all.py` lines 1148–1154: coverage check inner loop

### Fix Direction

Call `os.listdir` once per encoder directory at the start of the coverage check, build a dict mapping shard stem to count, then look up each shard in O(1):

```python
# Outside the shard loop:
stem_counts = {}
for f in os.listdir(out_dir):
    if f.endswith(".npz"):
        s = f.split("_")[0]
        stem_counts[s] = stem_counts.get(s, 0) + 1

# Inside the shard loop:
actual = stem_counts.get(stem, 0)
```

This is a simple one-liner fix that reduces 200 `os.listdir` calls to 2.

---

## 9. Data Stager Stages All Historical Precompute Versions Not Just Current

### Description

`data_stager.py _stage_precomputed` (lines 306–396) stages files from the "current" version directory for each encoder. This is correct. However, `_archive_precomputed` (lines 402–464) iterates every version directory found on hot storage:

```python
for ver_dir in hot_enc.iterdir():
    if not ver_dir.is_dir() or ver_dir.name == "current":
        continue
    ver = ver_dir.name
    cold_ver_dir = cold_enc / ver
    files_to_copy = [
        f for f in ver_dir.iterdir()
        if not (cold_ver_dir / f.name).exists()
    ]
```

This means every run archives all version directories that exist on hot, even stale non-current ones. If hot storage accumulates multiple old precompute versions (e.g., from prior cache invalidations), every archive run re-inspects those directories. The `if dst.exists(): return 0` guard prevents re-copying, but the directory enumeration cost grows with the number of stale versions.

More significantly, when hot storage is cleared between campaigns and rebuilt by staging, only the "current" version is staged. But after several campaigns, cold storage accumulates many version directories, and the stager archives all of them (even empty or tiny manifest-only dirs) on every archive cycle.

### Magnitude

Minor per-run overhead (extra directory traversals). At 10 stale versions × 500,000 files each = 5M file existence checks across encoders. On a spinning cold disk, this could add tens of minutes to archive runs.

### Affected Scripts and Lines

- `data_stager.py` lines 402–464: `_archive_precomputed`
- `data_stager.py` lines 427–440: iterates all version dirs, not just `current`

### Fix Direction

Archive only the `current` version on each cycle, or track a "dirty" flag per version dir and only archive dirs that have new content since the last archive run.

---

## 10. Checkpoint Archiving: Copies Both Step Checkpoints and Best Checkpoint Separately

### Description

`_archive_chunk_checkpoint` in orchestrator.py (lines 576–605) copies the latest `step_*.safetensors` checkpoint pair to `archive/chunk{N}_final.*`. Then `_archive_checkpoints` in `data_stager.py` (lines 478–536) also copies `best.safetensors` as `campaign_dir/final.safetensors`. This means there are two archive paths for essentially the same weights:

1. `archive/chunk{chunk}_final.safetensors` — the last step checkpoint
2. `cold_root/weights/flywheel-YYYYMMDD/final.safetensors` — a copy of best.safetensors

If the last step checkpoint IS the best checkpoint (common when training converges monotonically), this is two copies of the same file. On a cold HDD, each copy could be several GB (the full safetensors file).

Additionally, EMA weights (`.ema.safetensors`) are copied alongside the main weights at every archive point. The EMA file is the same size as the main weights file, doubling the checkpoint storage footprint.

### Magnitude

Safetensors file size for an IP-Adapter: ~1–2 GB. EMA file: another ~1–2 GB. Two archive paths: 4 GB per chunk × 4 chunks = 16 GB of potentially redundant checkpoint storage, plus the time to copy 4–8 GB to a cold spinning disk.

### Affected Scripts and Lines

- `orchestrator.py` lines 576–605: `_archive_chunk_checkpoint`
- `data_stager.py` lines 478–536: `_archive_checkpoints`
- `orchestrator.py` lines 588–599: copies step checkpoint
- `data_stager.py` lines 518–523: copies best.safetensors as `final.safetensors`

### Fix Direction

Check if the latest step checkpoint and `best.safetensors` are the same file (compare size and step number from `best.json`). If so, write a symlink instead of a copy for one of the archive paths.

---

## 11. Shard Validation Runs After Precompute But Opens All Tars a Third Time

### Description

The `validate_shards` step (orchestrator.py lines 1473–1483) runs a tarfile header scan before training. This is the third time the shards are read:
- First: `build_shards.py` writes them (reads source tars, writes output tars)
- Second: `precompute_all.py` reads them (Phase 1 + coverage check)
- Third: `validate_shards.py` reads them (integrity scan)

The validation step is lightweight (header-only scan), but it still adds another I/O pass over potentially hundreds of GB of shards.

### Magnitude

Relatively minor if it's genuinely header-only. The main concern is sequencing: validation happens after precompute, meaning precompute already found and processed every shard. Any corruption that validation would detect was already handled (or caused an error) during precompute's Phase 1 tar read.

### Affected Scripts and Lines

- `orchestrator.py` lines 1473–1483: `_start_shard_validation`
- `precompute_all.py` lines 131–152: `iter_shard` — reads every tar member
- `build_shards.py` lines 383–396: closes and renames output tars

### Fix Direction

Consider merging shard validation into the precompute pass. Since precompute already opens every tar and reads every record, any corruption detectable by a header scan would be caught in the precompute Phase 1 loop (which already has `try/except` error handling). The separate validate step adds an I/O pass without adding detection capability beyond what precompute already provides.

---

## Quick Wins

These are low-effort, high-impact fixes that can be implemented without restructuring the pipeline:

### QW-1: Fix os.listdir in coverage check (15 min)

**File:** `precompute_all.py` lines 1148–1154  
**Change:** Replace per-shard `os.listdir(out_dir)` with one scan per encoder directory at the start of the coverage check, building a stem→count dict. Saves ~3 minutes per precompute run.

### QW-2: Fix _has_output in --new-shards-first (15 min)

**File:** `precompute_all.py` lines 872–879  
**Change:** Replace the inner `os.listdir` loop with a pre-built set of shard stems. Saves 3+ minutes of shard selection overhead.

### QW-3: Record shard sizes during encoding, skip re-open in coverage check (30 min)

**File:** `precompute_all.py` lines 583–716 (process_shard), 1139–1168 (coverage check)  
**Change:** Return `n_records` from `process_shard` / `_process_shard_inner`. Accumulate in the main loop. Use accumulated counts in the coverage check instead of re-opening tars. Saves 2–5 minutes per precompute run plus avoids full JPEG re-reads.

### QW-4: Group SigLIP fallback reads by shard in mine_hard_examples (45 min)

**File:** `mine_hard_examples.py` lines 543–564  
**Change:** Pre-group the `sample` list by `shard_path` before the eval loop. Open each shard once and extract all needed records. This is the same grouping that Pass 2 already does (lines 604–629). Saves potentially 41+ minutes when SigLIP cache is absent.

### QW-5: Use incremental FAISS search in find-dups (1 hour)

**File:** `clip_dedup.py` lines 428–498  
**Change:** Track `prior_n = n_previous_chunks_total` in the `.ids` file or a sidecar. In `find-dups`, reconstruct only the current chunk's new vectors (`all_vecs[prior_n:]`) and search them against the full index. Skip re-searching old-to-old pairs.

---

## Structural Changes

These require more architectural work but offer larger returns:

### SC-1: Consolidate dedupe_filter and clip_embed/clip_index/clip_dups into one GPU step

Currently CLIP embedding runs twice per chunk on essentially the same images. Eliminating the post-build CLIP steps and wiring `dedupe_filter` to use the cumulative index saves 2–3 GPU-hours per chunk.

**Affected:** `orchestrator.py` `CHUNK_STEPS` list, `_start_clip_embed`, `_start_clip_index`, `_start_clip_dups`; `dedupe_filter.py`; `clip_dedup.py`

### SC-2: Add shared warm-start checkpoint for ablation runs

Generate a shared warm checkpoint (2,000–5,000 steps) at the start of each ablation campaign. All ablation runs warm-start from it. This eliminates the cold-start learning curve from scoring and makes `n_snapshots` more informative. Expected savings: 20–40% of GPU time for large campaigns.

**Affected:** `ablation_harness.py` `_run_one`, `_build_run_config`

### SC-3: Increase hard-example mining eval budget proportionally

The 5,000-record default is very small relative to the corpus (0.1%). Increase to 50,000–100,000 records with batch eval. The Flux model is already loaded; the marginal cost of evaluating more records is small relative to the fixed model load overhead. Better coverage gives a more representative ranking and higher-quality hard examples.

**Affected:** `mine_hard_examples.py` `--eval-records` default (line 299)

### SC-4: Build per-tgz quality score and skip known-low-quality tgzs

After the first pass through the JDB data, persist per-tgz quality statistics (% passing quality filter, dedup rate) in `shard_scores.db`. In subsequent campaigns, skip downloading tgzs with historically high waste rates. This reduces the download + convert + filter work for known-bad data.

**Affected:** `dedupe_filter.py`, `download_convert.py`, `shard_scorer.py`, orchestrator download step

### SC-5: Stream-convert JDB tgzs without intermediate disk write (PIPELINE-25b)

Already documented in BACKLOG.md. Implementation would save one full disk write+read cycle per tgz (125 GB I/O per chunk). Applicable only when the raw pool is not needed for persistent storage.

**Affected:** `download_convert.py`

### SC-6: Merge shard validation into precompute

Since precompute already reads every shard record during Phase 1, shard validation adds no detection coverage. Remove the separate `validate_shards` step and emit the integrity report as a sidecar from precompute. Saves one I/O pass over the full shard set.

**Affected:** `orchestrator.py` `CHUNK_STEPS`, `validate_shards.py`, `precompute_all.py`

---

## Issues Not Found (Already Addressed)

The following potential issues were investigated and found to be already well-handled:

- **Precompute re-runs**: `precompute_all.py` has a `.precompute_done.json` resume file and per-record existence checks, so restarts are efficient.
- **Model loading in precompute**: The "sample the last shard" optimization (lines 956–972) correctly avoids loading 8+ GB of model weights when all records are cached.
- **Build shard max-shards alignment**: The orchestrator wires `--max-shards` consistently between `build_shards` and `precompute` (orchestrator.py lines 1224–1227).
- **Shard ID space**: The 200,000 ID block per chunk prevents .npz collisions across chunks (pipeline_lib.py `SHARD_BLOCK`).
- **FAISS lock safety**: `clip_dedup.py _faiss_lock` prevents concurrent index corruption.
- **Checkpoint archiving atomicity**: `_archive_chunk_checkpoint` uses tmp+rename correctly.
- **Stager idempotency**: `_atomic_copy` checks `dst.exists()` before copying, so restarts are safe.
