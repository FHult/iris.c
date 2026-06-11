# SREF Execution Plan — steps and decision points

The sequenced path from today's state to a shippable Midjourney-style `--sref` feature,
with the explicit decision points. Companion to BACKLOG "SREF Objective" (the gap
analysis + wiring specs) and plans/warmup-campaign-runbook.md §3 (campaign mechanics).
Written 2026-06-11, immediately after SREF-1W landed (style step + neighbor pairing,
config-gated, fail-open).

## Status update 2026-06-11 (evening) — checklist executed through DP-2b

- DP-1 CLOSED: distillation stopped at 27K (loss flat from ~18K); Tier-1 on the
  26K ckpt FAILED decisively (cos 0.860 / std 0.868 / fft 0.918) = small-variant
  capacity ceiling. Medium retrain + scaling-law sizing in BACKLOG PRECOMP-2;
  NOT run5-blocking.
- Combined smoke PASSED all criteria (style_pair=47%, VAL parsed, cold reuse) and
  exposed + fixed the EMA-lag bug (decay warmup ramp — prior champions were ~90%
  random init at 1000 steps). Val set staged to hot; final-window telemetry added.
- DP-2b DONE: whole-pool style map — 1280/1280 shards, 255,987 CSD embeddings,
  pool report `/Volumes/2TBSSD/style_clusters/pool_report.json`:
  **pair_rich 86.4%** pool-wide (vs 44% val sample) → shard_manifest NOT needed;
  free selection + per_source_min suffices. Every run5 style step is now a copy.
- SREF-2 triad complete: style_sim + content_leak (CSD heads) + prompt_adherence
  (SigLIP text tower, --prompt-adherence).
- DP-6 started: web/server.py style-mode slots route through the IP-Adapter when
  IRIS_IP_BUNDLE is set (siglip_features.py sidecar, one-shot iris --ip path).
- run5: cleared pending smoke3 (EMA-ramp GPU validation, in flight); launches tonight.

## Current state (what is DONE)

- C inference: G-1 Phase 2 — `iris --ip BUNDLE --ip-features F [--ip-scale N]` runs
  image-conditioned generation end-to-end (CPU block path; bit-exact adapter math;
  inertness gates bit-identical).
- Held-out signal: PROD-1 val set + paired held-out cond_gap supersedes the train-batch
  gap for champion/attribution (review M1).
- Style encoder: CSD ViT-L in pure MLX (`train/style_encoder/csd_mlx.py`), validated —
  NN pairing meets the ≤0.7 gate (top-5 0.679; visual confirmation).
- SREF-1W wiring: style step in the orchestrator (gated by `style_pairing: true`,
  default OFF), per-shard bundles + cold reuse + publish, iteration-local
  `neighbors.sqlite`, dataset/trainer style-paired cross-ref with full fail-open
  fallback, `style_pair_pct` telemetry, doctor/DISPATCH visibility.
- Memory gate: TRAIN-7 passed — Stage 2 (768) and Stage 3 (1024) configs ready.
- Proxy-VAE distillation: training overnight (decision point 1 below).

## Step sequence with decision points

### DP-1 (breakfast): proxy distillation — stop early or run to 50K?
- By ~09:00 the run is at ~step 19-20K (checkpoints every 2K).
- **Stop-early path (default):** kill the run, run Tier-1 manually on the newest
  checkpoint against the held-out val set:
  `evaluate_vae_proxy.py --proxy <ckpt> --tier 1 --shards .../validation/held_out
   --vae-cache .../validation/precomputed/vae --out /Volumes/2TBSSD/proxy_vae_eval.json`
  → frees the GPU for the day. Distillation curves usually saturate well before 50K
  (latent_mse was 1.06→0.31 in the first 4K steps).
- **Run-to-50K path:** only if Tier-1 on the early checkpoint *fails marginally*
  (more steps might close the gap). Costs the daytime GPU window + collides with DP-3.
- Tier-1 verdict consumes: per-channel std ratio 0.95-1.05 (critical), LPIPS < 0.04,
  PSNR > 35. Pass → proxy usable for flywheel precompute (PRECOMP-2 promotion
  decision later, after a Tier-2 A/B). Fail on std-ratio → `medium` variant retry
  (overnight job).

### DP-2 (daytime): GPU window allocation
Owner's coding work has priority. Idle slices, in value order:
1. **make test** golden re-run (validates the C hygiene batch fully) + one GPU smoke of
   the end-of-run VAL cond_gap eval (the only unsmoked new trainer path).
2. **SREF-1W live smoke**: tiny flywheel config with `style_pairing: true` on 2-3
   shards — confirms the orchestrator step + telemetry end-to-end before run5 bets on it.
3. Style backfill for already-precomputed shards (~10-16h full pool — overnight-class;
   partial backfill of high-value shards is fine, reuse makes it incremental).

### DP-2b (next free GPU window): style-map the WHOLE pool (~4h), then targeted backfill
Style encoding needs only the IMAGES (tars), not the qwen3/vae precompute — so the map
is not bound to the trainable subset. **Sample-encode ~200 records/shard across ALL
1280 shards (~4h at 56 ms/img)**, then `style_shard_report.py` ranks the entire pool:
per-shard **diversity**, **pair_rich** (records with ≥3 strong style neighbors —
isolated styles can't form training pairs), **cross-shard connectivity** (which shards
to CO-STAGE for rich iteration-local neighbor lists). Baseline (val sample): ≈44%.
The pool is already style-rich — 413 coyo + 95 coyo+laion+wikiart + 28
journeydb+wikiart ≈ 42% non-pure-journeydb — so NO data acquisition is needed; the
lever is SELECTION: run5 raises per_source_min for wikiart/coyo sources and co-stages
style kin per the connectivity map. Full style encode happens on demand for staged
shards only (per-iteration SREF-1W step + bundle reuse). With subsampled precompute
(DP-2c below), first-contact on never-precomputed wikiart/coyo shards is cheap (~1-2h),
so the style-rich 42% becomes genuinely trainable immediately.

### DP-2c: subsampled flywheel precompute (~5-10x iteration throughput)
At batch 1 × 1000 steps an iteration TRAINS on ~1K records but precomputes ~210K.
Pass `--subsample-per-shard ~200` (flag exists; deterministic, shared across encoders;
add the same flag to style_precompute) through a flywheel-config key → precompute
phase drops ~20h → ~1-2h, i.e. several iterations per GPU-night. Training signal is
unchanged (it never consumed more); full cache coverage is only needed before the
PRODUCTION run. This compresses run5's style-pairing verdict from weeks to days and
lets run5 warm attribution itself — making run4 largely redundant (see DP-3).

### DP-3 (night): which campaign?
- **Default: relaunch warmup-run4 as-is** (`pipeline_ctl start-flywheel
  train/configs/flywheel_warmup_run4.yaml`): data-selection warmth, now ranked by the
  held-out cond_gap. Style pairing stays off — run4's job is attribution, and changing
  two variables at once muddies both.
- **run5 (first style-paired campaign)** starts after: SREF-1W live smoke passes (DP-2
  #2) AND run4 has warmed attribution enough to be useful (or is superseded by the
  decision below).
- **Judgment call:** if the SREF-1W smoke passes early, it is defensible to skip
  straight to run5 (style pairing changes what cond_gap measures anyway, so run4's
  trajectory won't be comparable to run5's regardless). Cost of skipping: less
  attribution warmth. Default remains run4-first unless the owner prefers speed to
  style results.

### DP-4: ablation unlock
When the cond_gap-stall/over-training detectors fire on the active campaign AND
attribution is warm (refgap's "ablation-ready" line): enable
`ablation_sref_v1.yaml` (add `learning_rate` to its variables first). First arm:
`freeze_double_stream_scales` — double-stream injection may matter for style.

### DP-5: production foundation run (~12 days, 512px)
Gate: ALL of —
- style-paired training validated (run5 shows ref_gap/cond_gap improving vs run4),
- SREF-2 style eval exists (style similarity + content leak + CLIP-T) — without it the
  shippable-champion call is a guess,
- per-chunk step budget sized from warmup data (PROD-2 note),
- data recipe folds per-source curation (coyo ≫ journeydb; grow style-rich sources).
Then Stage 2 (768, ~2.1d) → Stage 3 (1024, ~2.1d) fine-tunes per TRAIN-7.

### DP-6: app integration (parallel track, not gated on training)
- web/server.py SigLIP sidecar → `--ip-features` (ships with ANY checkpoint; quality
  improves as checkpoints do).
- Multi-ref (concat SigLIP rows), strength = `--ip-scale`, style codes = stored
  embedding library.
- Latency track when it matters: bf16/MPS-native inject (adapter currently forces CPU
  blocks, ~4x slower), G-1 Phase 3 (SigLIP/CSD in C) for a Python-free engine.

## Morning checklist (push-button DP-1 → run5)

```bash
# 1. DP-1: distillation verdict (doctor shows the run + step count)
train/.venv/bin/python train/scripts/pipeline_doctor.py --ai | head -40
#    Stop-early (default once >=20K steps). The EXIT_CODE marker tells the
#    doctor this was deliberate (else it reports a crash until the eval runs):
tmux kill-session -t iris; pkill -f train_vae_proxy
sleep 2; echo "EXIT_CODE=130" >> /Volumes/2TBSSD/logs/proxy_vae_train.log
CKPT=$(ls -t /Volumes/2TBSSD/checkpoints/vae_proxy/*.safetensors | head -1)
train/.venv/bin/python train/scripts/evaluate_vae_proxy.py --proxy "$CKPT" --tier 1 \
  --shards /Volumes/16TBCold/validation/held_out \
  --vae-cache /Volumes/16TBCold/validation/precomputed/vae \
  --flux-model flux-klein-model --out /Volumes/2TBSSD/proxy_vae_eval.json \
  --report /Volumes/2TBSSD/proxy_vae_eval.html

# 2. SREF-1W + VAL-eval smoke (~30 min, one combined run)
train/.venv/bin/python train/scripts/pipeline_ctl.py start-flywheel \
  train/configs/flywheel_smoke_style.yaml
#    pass: style step log shows neighbors; trainer log shows style_pair>0% + VAL line

# 3. DP-2b: whole-pool style map (~4h, can run unattended)
train/.venv/bin/python train/scripts/style_precompute.py \
  --shards /Volumes/16TBCold/shards --out /Volumes/16TBCold/precomputed/style/v1_csd \
  --subsample-per-shard 200
train/.venv/bin/python train/scripts/style_shard_report.py \
  --style-cache /Volumes/16TBCold/precomputed/style/v1_csd \
  --out /Volumes/2TBSSD/style_clusters/pool_report.json

# 4. Launch run5 (optionally set shard_manifest from the report first)
train/.venv/bin/python train/scripts/pipeline_ctl.py start-flywheel \
  train/configs/flywheel_warmup_run5.yaml
```

## Standing constraints
- Never start a production run without PROD-1/PROD-2 active (they are now).
- Style precompute/backfill and proxy training are GPU-window tasks — schedule against
  the owner's day/night cycle (`pause --free-gpu` handles the flywheel side).
- Every SREF mechanism is fail-open; a missing artifact must degrade, never crash.
