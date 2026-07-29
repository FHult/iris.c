# Foundation Quality Roadmap

**Created 2026-07-22.** The next major thrust after the SREF workstream closed. This is a living
roadmap; supersedes the May scaffolding in `plans/archive/cold-full-shard-build-foundation-runs.md` (which
predates the actual corpus build) and folds in the TRAIN-7 higher-res work.

## What this is — and isn't

The Flux.2 Klein 4B **base is frozen** (BFL weights, Apache 2.0) — we do NOT retrain a foundation
model. "Foundation quality" here means maximizing the quality of everything we DO train on top of the
frozen base — the conditioning adapters (IP-Adapter for subject/image conditioning; the SREF Learned
Style adapter for style) — by lifting the three levers that actually move quality: **resolution**,
**data scale + curation**, and the **meta-flywheel** that picks what to train on. The "foundation" is
the full curated data corpus + versioned precompute + accumulated scores that every campaign draws from.

**Honest ceiling note.** The SREF campaign established that per-reference *style discrimination* on the
frozen distilled base is mechanism-bounded, and per-type specialists don't beat the generic (3
negatives). So the headroom is NOT "make the style adapter discriminate more"; it's **sharper, higher-
resolution, better-curated conditioning** — detail, fidelity, prompt-adherence, subject conditioning —
where resolution and data quality genuinely help. Scope campaigns to that.

## Current state (verified 2026-07-22)

The data foundation is ~80% built already — this roadmap is mostly *completion + exploitation*, not a
cold build:

- **Corpus: 1,280 shards on cold** (`/Volumes/16TBCold/shards`, ~6.4M images; sources coyo / journeydb
  / laion / wikiart). Build (cold-foundation Phase 1) is DONE.
- **Precompute on cold** (`/Volumes/16TBCold/precomputed`): qwen3 `v_059443` (complete, ~1.15M records),
  siglip `v_336c6e` (~1.15M), style CSD `v1_csd` (~256k). **VAE `v_2232c1` is INCOMPLETE** (doctor:
  `complete=false`) — the one real data-foundation gap.
- **Meta-flywheel knowledge base exists** on cold `metadata/`: `shard_scores.db`, `ablation_history.db`,
  `flywheel_history.db` (+ `_prior` backups). Curation history is accumulating.
- **Higher-res gate PASSED (TRAIN-7, 2026-06-10):** measured 1024px backward peak **21.32 GB** / system
  ≤21.5 GB (~18 s/step), 768px 19.33 GB (~9 s/step) — **no grad-checkpointing needed**, fits 32 GB.
  Stage 2 (768px) and Stage 3 (1024px) are unblocked. (Probes MUST run cached — live-encode segfaults
  MLX, BUGS.md MLX-1.)
- **Throughput reality:** flywheel iterations are **precompute-dominated (~a day/iter)** — VAE/Qwen3/
  SigLIP encode dominates, not the training step (see `memory/flywheel_throughput_strategy`). Any plan
  must treat precompute as the long pole, not GPU-steps.

## The quality levers (ranked)

1. **Resolution (highest-confidence lever).** 512px → 768px → 1024px. Gate already passed; the frozen
   base renders far more detail at 1024px, and sharper conditioning latents help IP-adapter fidelity.
   This is the clearest quality win and the least speculative.
2. **Data curation.** The meta-flywheel (shard scoring + hard-example mining + ablation) picks better
   shards. Scores already accumulate; the lever is running more curation and *acting* on it (top-pct
   / hard-focus campaigns).
3. **Data scale.** The full 1,280-shard pool vs the ~250-shard subsets used so far. More coverage helps
   generalization — but is throughput-bounded by precompute, so scale is gated on Phase A.
4. **Proxy-VAE (throughput multiplier, not a quality lever directly).** The dominant precompute cost is
   the VAE encoder; a distilled proxy (PRECOMP-2, design reviewed) would ~9× precompute throughput,
   unlocking more scale/resolution per unit time. Enables the above rather than improving quality itself.

## Phased roadmap

### Phase A — Complete & verify the data foundation (prerequisite; ~days, precompute-bound)
Goal: a single, complete, verified precompute set at the training resolution so no campaign silently
skips samples (the cache-miss "dies at step 0" class — CLAUDE.md invariant #1/#6).
- **A1. Finish the VAE precompute** (`v_2232c1` is incomplete). Run `precompute_all.py --vae-output`
  over the full shard set; verify with `cache_manager.py --verify`. This is the gating data gap.
- **A2. Write correct `manifest.json`s** for the hot subset versions (the 3 cosmetic doctor warnings)
  so completeness is machine-checkable. Small.
- **A3. Decide the training resolution's precompute strategy.** 512px latents exist; 768/1024px need
  their OWN precompute (VAE latent shape is resolution-specific — the bucketing contract, CLAUDE.md
  invariant #1). Either precompute a 768px set now (long pole) or gate it behind the proxy-VAE.
- **Gate A:** `pipeline_doctor --ai` clean on precompute completeness; a 1-shard cached smoke trains
  at step 0 with 0% cache-miss.

### Phase B — Higher-resolution SREF + IP-adapter training (the core quality lift)
Goal: re-train the shipped adapters at 768px (then 1024px) on curated data — sharper, more detailed
conditioning. Gate already passed; this is execution.
- **B1. 768px SREF adapter** — re-run the joint-backbone recipe (the v5.3.0 winner:
  `sref_joint_probe_v1.yaml`) at `bucket: [768,768]` on a 768px-precomputed curated subset. Gate with
  `debug/sref_scorecard.py` vs the shipped 512px adapter (styleCSD Δ, promptAdh, **and the collapse
  gate cross-ref corr < 0.90** — mandatory). Ship as `sref-adapter-v2` only if it beats v1.
- **B2. 768px IP-adapter** — Stage 2 config `stage2_768px.yaml` (warm-start from the best 512px
  checkpoint); the flywheel's own eval (cond_gap / CLIP-I) is the gate.
- **B3. Smoke before every long run** (BACKLOG lesson #1): 100 steps, confirm the measured step time +
  memory peak match the TRAIN-7 profile before committing ~a day.
- **Gate B:** a 768px adapter that beats its 512px counterpart on the scorecard/flywheel eval, at a
  memory peak matching TRAIN-7 (~19 GB @768px).

### Phase C — Curation-driven quality flywheel (compounding, runs alongside B)
Goal: let the meta-flywheel improve *what* we train on while B improves *how*.
- **C1. Run the ablation harness** on the real full-corpus distribution (`ablation_harness.py`) to
  discover quality hyperparameters, writing to cold `ablation_history.db`.
- **C2. Shard-score-driven campaigns** — use `shard_scores.db` top-pct / hard-focus selections (the
  campaigns already defined: `high_signal_pruned`, `hard_example_heavy`) as the training pool for B,
  instead of a flat subset.
- **C3. Close the loop** — each B run's eval metrics feed back to the scores (the dual-flywheel in the
  BACKLOG platform vision). Every campaign starts from a richer foundation.
- **Gate C:** a curated-pool 768px run beats a flat-pool 768px run on the same eval (proves curation
  moves quality, not just resolution).

### Phase D — 1024px + release-grade adapters (the top of the ladder)
Goal: the highest-quality adapters the hardware supports, packaged for release.
- **D1. 1024px precompute** (proxy-VAE strongly advised here — 4× the tokens, precompute is the wall)
  on the best-curated subset from C.
- **D2. 1024px SREF + IP-adapter** (Stage 3, `stage3_1024px.yaml`, warm-start from the 768px winner);
  add `(1024,1024)` to `BUCKETS` (revert if not adopted).
- **D3. Release** the winners as `sref-adapter-v2` / an IP-adapter bundle, reusing the now-established
  release runbook (`plans/archive/sref-adapter-release-runbook.md`) + model-card pattern.
- **Gate D:** 1024px adapter beats 768px on the scorecard AND passes the collapse gate; memory peak
  ≤21.5 GB (TRAIN-7 measured).

## Constraints & risks

- **M1 Max 32 GB.** 1024px fits (21.32 GB measured) but with little margin; batch-1, no headroom for
  larger batches. Probes/smokes MUST be cached (live-encode segfaults MLX — BUGS.md MLX-1).
- **Precompute is the long pole** (~a day/iter). Resolution multiplies it (768px ~2.25×, 1024px ~4×
  the tokens). The proxy-VAE (Phase A3/D1) is the throughput unlock; without it, higher-res is
  wall-clock-expensive. Never precompute/train from cold storage directly — stage to hot SSD first
  (CLAUDE.md invariant #6, "dies at step 0" hazard).
- **Adapter quality ceiling (scope guard).** Don't chase style *discrimination* gains — that's
  mechanism-bounded on the frozen base (SREF 3 negatives). Chase resolution/detail/curation, which
  genuinely help. Every adapter A/B keeps the collapse gate.
- **Verify the train↔infer boundary** for any new resolution/precompute path (CLAUDE.md correctness
  protocol; VAE-Q1 BN-pack, bucketing contract) — a silent latent-space mismatch is the recurring cost.

## Concrete first actions (in order)

1. `pipeline_doctor --ai` → confirm the exact precompute completeness state (VAE gap).
2. **A1:** finish the VAE precompute over the full shard set; `cache_manager.py --verify`.
3. Write `stage2_768px.yaml` (from `stage1_512px.yaml` + `bucket:[768,768]`) and precompute a 768px
   VAE set for a curated ~120-shard subset (the long pole — consider the proxy-VAE first).
4. **B3 smoke** the 768px SREF recipe (100 steps) → confirm ~19 GB / step time vs TRAIN-7.
5. Launch the first 768px SREF run; gate on the scorecard vs the shipped 512px adapter.

## Open decisions (owner)

- **Proxy-VAE now or later?** It ~9× precompute throughput (the wall for higher-res) but is itself a
  build+validate effort (PRECOMP-2). Recommended before 1024px (D); optional for 768px (B).
- **Target adapter for the lift:** SREF (style) first, IP-adapter (subject) first, or both in parallel?
  Recommend **SREF first** — it's the shipped, released, eval-harnessed path with the cleanest gate.
- **Scale vs resolution ordering:** push 768px on a small curated pool first (fast, high-confidence),
  or widen the pool first? Recommend **resolution-first on a curated subset** (Phase B before scale).

## References
- `plans/archive/cold-full-shard-build-foundation-runs.md` (the original data-build scaffolding, now mostly done)
- `memory/train7_plan` + BACKLOG TRAIN-7 (higher-res memory profile + gate-passed numbers)
- `memory/flywheel_throughput_strategy` (precompute-dominated throughput)
- `plans/pipeline-v2-architecture.md`, `plans/warmup-campaign-runbook.md` (pipeline + campaign mechanics)
- `plans/precomp2-proxy-vae-design.md` (the throughput unlock)
- `docs/sref.md`, `plans/archive/sref-joint-backbone-project.md` (the SREF adapter to lift)
