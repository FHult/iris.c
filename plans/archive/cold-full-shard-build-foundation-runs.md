# Cold-Only Full Shard Build + At-Scale Precompute + Progressive Medium Foundation Runs

**Date:** 2026-05 (prepared for current hardware state)  
**Context (your hardware):**  
- **Cold**: 16 TB HDD — source of truth + long-term archive. All raw/converted data + all shards + all versioned precompute + final weights live here permanently. Slow random I/O.  
- **Hot**: 2 TB external SSD — primary compute tier for real training runs and at-scale precompute. Fast enough random I/O for shard reading during training.  
- **Ultrahot**: Internal NVMe SSD — for small/quick experiments, ablation bursts, smoke tests, and low-latency serving. Logistics for moving active working sets between Hot ↔ Ultrahot are still WIP.  

All 4 datasets (~4–5 TB total, ~6 M images) are already downloaded and sitting on the cold HDD.  
**Goal:** Full cold-only shard build on the HDD first, then use the stager to pull only the active subsets needed for progressive medium runs + ablation onto Hot (or Ultrahot for small work). This becomes the living foundation for curation and later 1024 px work.  
**Scale target:** ~6 M images, 4–5 TB on cold. Target adapter weights ~4 GB range.

---

## High-Level Phases (Three-Tier Foundation)

1. **Attach Cold HDD** (source of truth) — do the full cold-only shard build directly on the slow HDD using `--cold-only`.
2. **Attach Hot external SSD** (primary compute) — let the stager pull only the active shard subset + precomputed caches from cold HDD → hot SSD.
3. **Foundation Precompute** on the hot external SSD (much faster random reads than HDD during VAE encoding of the active set).
4. **Progressive Medium Training Runs + Parallel Ablation** on hot SSD (or ultrahot internal NVMe for small experiments). Use the orchestrator + `background_staging: true` so the next chunk's data is already staged while the current one trains.
5. **Archive back** new precomputed caches + checkpoints from hot → cold HDD after each chunk (automatic via stager when `archive_after_chunk: true`).
6. **Hard-Example Mining + Shard Scoring** — results accumulate on cold `metadata/` (shard_scores.db, ablation_history.db, etc.).
7. **Progressive Curation** — later flywheel / manual campaigns use the growing scores on cold to select better shards.
8. **1024 px work** — once you have good curated subsets on cold, stage them to hot/ultrahot and run higher-res passes (VAE tiling already exists).
9. **Hot ↔ Ultrahot logistics** (WIP) — design policy for when to run small ablations / quick iterations on the internal NVMe vs full medium runs on the external 2 TB SSD.

All persistent knowledge lives on **cold** (HDD): `cold_root/metadata/`, `cold_root/precomputed/v_*/`, `cold_root/checkpoints/.../archive/`, `cold_root/shards/`, etc. Hot and ultrahot are transient working areas.

---

## Phase 0: Attach Drives & Environment Prep

When the cold (and optionally hot) volumes are physically attached:

```bash
# 1. Identify the mount points (example)
diskutil list
# Assume cold appears as /Volumes/16TBCold (or whatever label you gave the 16 TB+ volume)
# Hot as /Volumes/2TBSSD if also attached

# 2. Verify all 4 converted pools exist on cold (critical — you said they are already downloaded)
ls -d /Volumes/16TBCold/converted/{laion,journeydb,coyo,wikiart} 2>/dev/null || echo "Missing one or more"

# 3. Create the required persistent directories on cold (idempotent)
mkdir -p /Volumes/16TBCold/{shards,precomputed,hard_examples,anchor_shards,dedup_ids,checkpoints/stage1/archive,logs,pipeline,.heartbeat,metadata}

# 4. (Optional but recommended) Create a stable symlink or env var for the session
export PIPELINE_DATA_ROOT=/Volumes/16TBCold
```

**Single-volume mode (recommended while only cold is attached)**  
In this mode `cold_root == hot_root` (or storage block is omitted). All staging/archiving become no-ops. This is exactly "cold-only".

---

## Phase 1: Full Cold-Only Shard Building

The repo already has first-class support via `build_shards.py --cold-only`.

**Direct invocation (most reliable for a true cold-only run):**

```bash
# From the repo root, with cold volume mounted and converted pools present
COLD=/Volumes/16TBCold   # adjust to your actual mount

python train/scripts/build_shards.py \
  --sources \
    "$COLD/converted/laion" \
    "$COLD/converted/journeydb" \
    "$COLD/converted/coyo" \
    "$COLD/converted/wikiart" \
  --output "$COLD/shards" \
  --shard_size 5000 \
  --workers 6 \
  --compression zstd \
  --compression_level 1 \
  --blocklist "$COLD/metadata/duplicate_ids.txt" \
  --cold-only \
  2>&1 | tee "$COLD/logs/build_shards_cold_full_$(date +%Y%m%d-%H%M).log"
```

**What `--cold-only` does:**
- Validates every source and the output live under `/Volumes/16TBCold` (hard-coded COLD_PREFIX in the script; update the script constant only if your cold volume has a different top-level name).
- Enables full resume: any shard that already has both `NNNNNN.tar` + `NNNNNN.provenance.json` is skipped.
- Writes heartbeats and provenance sidecars.

**Resume safety:** The script is designed to be killed and restarted. It cleans stale `.tar.tmp` files at startup.

**Expected output scale (your numbers):**
- ~6 M images → ~1 200 shards at 5 k images/shard.
- On a fast external drive with 6 workers + turbojpeg + zstd-1 this is I/O bound but feasible in 1–3 days depending on drive speed and how much dedup work remains.

**Alternative (orchestrator-driven path):**  
Once you have a config with `storage:` pointing both roots at the cold volume, you can let the orchestrator drive `dedupe_filter` → `build_shards` for each logical chunk. For a true one-shot "all data now" build, the direct `--cold-only` command above is simpler and matches your stated intent.

**Post-build verification:**
```bash
python train/scripts/validate_shards.py --shards "$COLD/shards" --quick
ls "$COLD/shards" | wc -l
# Should be ~1200 .tar + provenance pairs
```

---

## Phase 2: At-Scale Precompute on Cold

Use `precompute_all.py` pointed directly at the cold shards.

Recommended first run (foundation pass):

```bash
COLD=/Volumes/16TBCold

python train/scripts/precompute_all.py \
  --shards "$COLD/shards" \
  --qwen3-output "$COLD/precomputed/qwen3" \
  --vae-output   "$COLD/precomputed/vae" \
  --siglip-output "$COLD/precomputed/siglip" \
  --siglip \
  --flux-model flux-klein-model \
  --qwen3-model flux-klein-model \
  --vae-batch 4 \
  --max-shards 300     # or omit for "everything" (your 1200+ shards); start with a large but bounded set for first foundation
  --workers 2 \
  2>&1 | tee "$COLD/logs/precompute_foundation_$(date +%Y%m%d).log"
```

**Important knobs for your scale:**
- `--max-shards` caps the first pass while still giving broad coverage. You can run multiple passes with different seeds or `--new-shards-first` later.
- The script supports resume via `.precompute_done.json` state files per shard.
- Cache versioning (`cache_manager.py`) will automatically create `v_<hash>` directories under each encoder. The `current` symlink points to the active one. Training always follows `current`.
- SigLIP adds significant time and ~2 KB/record; decide whether your first foundation run needs it or can add it in a second pass.

**After precompute, force a cache manifest check:**
```bash
python train/scripts/cache_manager.py --data-root "$COLD" --verify
```

---

## Phase 3: Progressive Medium Runs + Ablation Foundation (The Real Goal)

Do **not** jump straight to the `all-in` 540 k + 200 k step monster defined in `v2_pipeline.yaml`. Instead run a sequence of overlapping **medium** runs (105 k / 40 k step pattern) that progressively cover more of the 6 M image pool while using the ablation harness to discover good QUALITY hyper-parameters on your actual data distribution.

### Recommended First Config (create this)

Create `train/configs/cold_foundation_v1.yaml` (copy of v2_pipeline + overrides):

```yaml
# Cold-only foundation run — single large volume, progressive medium coverage
recipe: ip_adapter_flux4b
chunks: 4
scale: medium                 # start here; later campaigns can go larger
poll_interval: 60

storage:
  cold_root: "/Volumes/16TBCold"
  hot_root:  "/Volumes/16TBCold"   # single-volume mode — staging is a no-op
  data_prep_tier: hot
  archive_after_chunk: false       # nothing to archive when cold==hot

training_config: "train/configs/stage1_512px.yaml"

training:
  siglip: true
  mine: true
  mine_use_ema: true
  hard_mix_ratio: 0.05
  # Use the "medium" step counts from v2_pipeline.yaml as baseline
  # Override per-campaign as needed

precompute:
  max_shards:
    medium: 120                # generous for foundation; adjust after first run

# Optional: point orchestrator at a custom active config
# (pipeline_setup.py can generate v2_pipeline_active.yaml for you)
```

Launch the first medium foundation run (after attaching cold):

```bash
# Preferred: use the wrapper (it handles tmux + caffeinate)
./train/start_pipeline.sh \
  --data-root /Volumes/16TBCold \
  --config train/configs/cold_foundation_v1.yaml
```

While that (or a direct training run) is going, run the long-term ablation harness in its own tmux window for QUALITY feature discovery on the real distribution:

```bash
# In a separate terminal / tmux window (after cold is mounted)
python train/scripts/ablation_harness.py \
  --config train/configs/ablation_sref_v1.yaml \
  --output-dir /Volumes/16TBCold/ablation_foundation_sref_v1 \
  --warm-start-from /Volumes/16TBCold/ablation_sref_v1   # if you have prior ablation results
```

The ablation DB + HTML report on cold will become part of the permanent foundation.

---

## Phase 4–5: Curation, Hard Examples, Shard Scoring, 1024 px Prep

- After each medium chunk finishes: `mine_hard_examples` runs automatically (or manually) and writes to `hard_examples/`.
- `shard_scorer.py` + `shard_selector.py` populate `metadata/shard_scores.db` and `metadata/tgz_scores.json`.
- These DBs on cold become the source of truth for all future flywheel / progressive campaigns.
- For 1024 px:
  - You have already filtered very small images — excellent.
  - Add `(1024, 1024)` (and perhaps 896/1024 variants) to `BUCKETS` in `train/ip_adapter/dataset.py`.
  - Create `stage3_1024px.yaml` (warm-start from best 768 px or 512 px checkpoint).
  - Run a shard sweep (see BACKLOG.md) to decide minimum native resolution policy before committing full 1024 px precompute.
  - VAE tiling (`_encode_vae_tiled`) is already implemented in `precompute_all.py` — just pass the higher `image_size`.

---

## Exact One-Liner Commands You Will Run (when drives are attached)

```bash
# === BOOTSTRAP (run once after mount) ===
export COLD=/Volumes/16TBCold
mkdir -p "$COLD"/{shards,precomputed/{qwen3,vae,siglip},hard_examples,metadata,logs,checkpoints/stage1/archive}

# === PHASE 1: FULL COLD-ONLY SHARDS (if not already complete) ===
python train/scripts/build_shards.py \
  --sources "$COLD/converted/laion" "$COLD/converted/journeydb" "$COLD/converted/coyo" "$COLD/converted/wikiart" \
  --output "$COLD/shards" \
  --shard_size 5000 --workers 6 --compression zstd --compression_level 1 \
  --blocklist "$COLD/metadata/duplicate_ids.txt" \
  --cold-only 2>&1 | tee "$COLD/logs/build_cold_full.log"

# === PHASE 2: FOUNDATION PRECOMPUTE (start with 200–300 shards for speed) ===
python train/scripts/precompute_all.py \
  --shards "$COLD/shards" \
  --qwen3-output "$COLD/precomputed/qwen3" --vae-output "$COLD/precomputed/vae" --siglip-output "$COLD/precomputed/siglip" \
  --siglip --flux-model flux-klein-model --qwen3-model flux-klein-model --vae-batch 4 \
  --max-shards 300 2>&1 | tee "$COLD/logs/precompute_foundation.log"

# === PHASE 3: FIRST MEDIUM FOUNDATION RUN + PARALLEL ABLATION ===
./train/start_pipeline.sh --data-root "$COLD" --config train/configs/cold_foundation_v1.yaml

# In another window (ablation, fire-and-forget)
python train/scripts/ablation_harness.py \
  --config train/configs/ablation_sref_v1.yaml \
  --output-dir "$COLD/ablation_foundation_sref_v1"

# Later: more medium runs, warm-started from best checkpoint of previous, expanding shard coverage
```

---

## Storage Layout on Cold (Expected After Foundation)

```
/Volumes/16TBCold/
  converted/{laion,journeydb,coyo,wikiart}/...          (your existing downloads)
  shards/                                               (~1200 .tar + .provenance.json)
  precomputed/
    qwen3/v_<hash>/ ... ; current -> v_...
    vae/v_<hash>/ ...
    siglip/v_<hash>/ ...
  hard_examples/                                        (persistent, never delete)
  metadata/
    shard_scores.db
    ablation_history.db
    tgz_scores.json
    duplicate_ids.txt
    checkouts/ (if you ever use mobile)
  checkpoints/stage1/                                   (foundation weights + archive/)
  logs/
  pipeline/                                             (sentinels for any orchestrator runs)
```

---

## Next Actions / Open Items (you can do these now or after mounting)

1. Create `train/configs/cold_foundation_v1.yaml` (example above) and a 1024 px bucket expansion patch.
2. (Optional) Add a small helper `train/scripts/cold_bootstrap.sh` that creates the dir skeleton and prints the exact build/precompute commands.
3. Update `COLD_PREFIX` in `build_shards.py` only if your cold volume root is not literally `/Volumes/16TBCold`.
4. Run a tiny smoke on a 5-shard subset once cold is mounted to validate the full cold-only path before the big run.
5. When the fast hot volume is later attached, simply change the `hot_root` in the config and let the stager move the active working set.

This plan re-uses every existing mechanism (`--cold-only`, versioned caches, ablation harness, shard selector, VAE tiling, single-volume mode, etc.) and matches the philosophy already documented in `train/DISPATCH.md`, `TRAINING.md`, `BACKLOG.md`, and `plans/warmup-campaign-runbook.md`.

When the drives are attached, paste the commands in Phase 0–3 and you will be running the exact foundation campaign you described.

---

**References**
- `build_shards.py:547` (`--cold-only`)
- `data_stager.py:144` (single-volume no-op when cold==hot)
- `plans/warmup-campaign-runbook.md` (3–5 medium runs strategy + 1024 px targets)
- `BACKLOG.md` (PRECOMP-1 VAE tiling, Stage 2/3 768/1024, shard resolution sweep)
- `train/configs/v2_pipeline.yaml` (all-in / medium scale definitions)
- `ablation_harness.py` + `ablation_sref_v1.yaml`

Ready when you are — just say the word and I will generate the exact `cold_foundation_v1.yaml`, any small script patches, or the 1024 px bucket/config additions.
---

## Generated Artifacts (created in this session)

These files were written to make the plan immediately actionable the moment your cold volume is mounted:

| File | Purpose |
|------|---------|
| `plans/cold-full-shard-build-foundation-runs.md` | This runbook |
| `train/configs/cold_foundation_v1.yaml` | Ready-to-use pipeline config for the first progressive medium foundation campaign using your real three-tier setup (cold HDD / hot external SSD / ultrahot internal NVMe) |
| `train/configs/stage3_1024px.yaml` | Skeleton for eventual 1024 px high-res fine-tune (warm-starts from Stage 2/1) |
| `train/scripts/cold_foundation_bootstrap.sh` | Executable helper that validates the cold mount, creates the dir skeleton, and prints every copy-paste command for Phases 1–3 |
| `train/ip_adapter/dataset.py` | Added commented 1024 px bucket placeholders + explanatory comment (no behaviour change until you uncomment) |
| `BACKLOG.md` | New entries **PIPELINE-32** (small Hot ↔ Ultrahot movement helper script) and **PIPELINE-33** (policy + integration) added under Pipeline Improvements |

Run the bootstrap script the instant the drives appear:

```bash
bash train/scripts/cold_foundation_bootstrap.sh /Volumes/16TBCold /Volumes/2TBSSD /Users/fredrikhult/ultrahot
```

(The script accepts cold, hot, and ultrahot paths and will emit the correct commands for each phase.)

It will emit the exact build_shards --cold-only, precompute, start_pipeline, and ablation commands tailored to your mount point.

All other mechanisms (ablation harness, shard selector, VAE tiling, single-volume no-op stager, cache versioning, etc.) were already present and are used unchanged.

---

## Updated Hardware Reality (your latest note)

- **Cold (16 TB HDD)**: Source of truth. Everything important lives here permanently. All shards are archived back here. Slow for random I/O during training or VAE precompute of large sets.
- **Hot (2 TB external SSD)**: Primary compute tier. This is where you want active shards + precomputed caches during real medium runs and at-scale precompute. Good balance of capacity and speed.
- **Ultrahot (internal NVMe SSD)**: Fastest tier, smallest capacity. Ideal for quick ablation experiments, smoke tests, small foundation subsets, and low-latency web serving. The policy/logistics for automatically or semi-automatically moving working sets between Hot and Ultrahot is still work in progress (as you said).

### Recommended Initial Workflow With This Hardware

1. **Attach only the 16 TB HDD (cold).**  
   Run the full shard build with `--cold-only` directly on the HDD.  
   This is the only time you want the entire 4–5 TB dataset touching the slow drive in bulk.

2. **Attach the 2 TB external SSD (hot).**  
   Use the proper three-tier `cold_foundation_v1.yaml` (cold = HDD, hot = external SSD).  
   The orchestrator + stager will copy only the shards + precomp needed for the current medium run from cold → hot in the background.

3. **Do precompute and training on Hot.**  
   VAE encoding and especially the training data loader (random shard reads every step) will be dramatically faster on the external SSD than on the HDD.

4. **Archive on completion.**  
   New precomputed version dirs and checkpoint snapshots get copied hot → cold automatically (`archive_after_chunk: true`).

5. **Small work on Ultrahot.**  
   For now, manually stage tiny subsets (e.g. 10–20 shards for an ablation burst) to the internal NVMe when you want maximum speed for a short experiment. We can build a small helper script later once the movement policy is clearer.

This is exactly what the existing `data_stager.py`, `storage:` block, `background_staging`, and `archive_after_chunk` machinery was designed for.

The generated `cold_foundation_v1.yaml` and bootstrap script have been updated to match this three-tier model.
