# SREF Execution Plan — steps and decision points

The sequenced path from today's state to a shippable Midjourney-style `--sref` feature,
with the explicit decision points. Companion to BACKLOG "SREF Objective" (the gap
analysis + wiring specs) and plans/warmup-campaign-runbook.md §3 (campaign mechanics).
Written 2026-06-11, immediately after SREF-1W landed (style step + neighbor pairing,
config-gated, fail-open).

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

## Standing constraints
- Never start a production run without PROD-1/PROD-2 active (they are now).
- Style precompute/backfill and proxy training are GPU-window tasks — schedule against
  the owner's day/night cycle (`pause --free-gpu` handles the flywheel side).
- Every SREF mechanism is fail-open; a missing artifact must degrade, never crash.
