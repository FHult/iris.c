# Warm-Up Campaign Runbook & Gap Analysis
# iris.c / IP-Adapter flywheel — 300–400 shard dataset

Generated: 2026-05-22. Based on deep code read of `flywheel_lib.py`, `shard_selector.py`,
`orchestrator.py`, `ablation_harness.py`, and live DB inspection.

Current state at time of writing: 80 shards (42 laion_coyo, 38 journeydb), 68 scored,
avg 4.4 observations/shard, cond_gap range [0.164, 0.389], campaign sref-v1 at 21 iterations.

---

## Section 1: Gap Analysis of Current Campaign Tool

### What Works Well (the Foundation)

**Shard scoring infrastructure** — `ShardScoreDB` maintains included EMA and excluded EMA,
enabling contrastive IPS attribution. Attribution confidence (`attr_confidence = hmean(n_inc,
n_exc) / MIN_ATTR_OBS`) correctly guards against premature trust. Hard-example density boost
and stability penalty (p95 loss) are wired in.

**Selection policy** — four tiers: top performers by effective_score, SigLIP diversity floor
(cosine max-min-distance), exploration budget, weighted random fill. Adaptive exploration
(`EXPLORE_THRESH=0.30`) boosts exploration when >30% of pool is unscored.

**Flywheel infrastructure** — checkpoint lineage, GPU lock, tmux-based training, poll-based
monitoring, plateau detection, ablation bursts on schedule and on plateau, hyperparameter
persistence across restarts, HTML report generation.

**Ablation harness (as of 2026-05-22 bug fixes)** — TrialTimer (ABL-1), multi-signal
EarlyStopper (ABL-2), Pareto warm-start for NSGA-II (ABL-3), temporal decay on scores
(SHARD-1), VAE tiling for high-res precompute (PRECOMP-1).

---

### Gap 1: No Explicit Coverage Goal or Progress Tracking

**What exists:** `shard_stats["scored"]` / `shard_stats["total"]` shown as a single stat chip
in the HTML report. No per-source breakdown, no coverage trajectory over iterations, no alert
when a threshold is crossed.

**What's missing:**
- `coverage_target: 0.70` config key that adjusts exploration and alerts when reached
- Coverage-by-source breakdown (laion_coyo vs journeydb hit rates may diverge)
- Coverage trajectory chart (new shards gaining first-contact per iteration)
- ETA to coverage target: with n_shards=20 from 300-shard pool, 70% takes ~17-20 iterations;
  the system cannot tell you this

**Impact:** You cannot declare "warm-up complete" automatically. Must manually query
`shard_selector.py status`, do the arithmetic, and decide when to move to phase 2.

---

### Gap 2: No Exploration Rate Schedule

**What exists:** `exploration_rate` is a static float in the flywheel YAML, plus hardcoded
adaptive boost (`EXPLORE_THRESH/EXPLORE_BOOSTED`). The adaptive boost only floors the rate
when unscored fraction is high — it never schedules a decay.

**What's missing:**
```yaml
exploration_schedule:
  - {through_iteration: 10, rate: 0.40}
  - {through_iteration: 20, rate: 0.25}
  - {after_iteration: 20, rate: 0.15}
```

**Impact:** Transitioning from exploration-heavy to performance-biased requires stopping the
flywheel and manually editing the YAML. Breaks autonomous operation; easy to forget.

---

### Gap 3: No Multi-Run Campaign Orchestration Layer

**What exists:** Individual campaigns are self-contained. `mark_superseded_by()` annotates
old campaigns. No automated "when campaign 1 plateaus, start campaign 2 with warm-start."

**What's missing:**
- Campaign sequencing with handoff conditions
- Auto warm-start: campaign 2's `base_checkpoint` auto-populated from campaign 1's best
- Accumulated shard coverage accounting across campaigns
- A `campaign_goals` YAML section: `{phase: "warm_up", coverage_target: 0.70, then: "performance"}`

**Impact:** The 3-5 medium runs strategy requires 3-5 manual config edits and restarts with
hand-copied checkpoint paths.

---

### Gap 4: No Per-Iteration Discovery Rate Metric

**What exists:** `n_unscored` is logged in `selection_log` and the HTML shows the static
total. No metric tracks how many new shards gained first-contact this iteration.

**What's missing:** `n_first_contact` per iteration (shards going from n_scored=0 to
n_scored=1). Directly measures warm-up progress; plateaus when coverage saturates.

**Impact:** Cannot see in the HTML report whether the system is still exploring or has stalled
on a subset of high-scoring shards.

---

### Gap 5: n_shards Not Tuned for 300-Shard Pool

**What exists:** `n_shards: 20` set for 80-shard pool (25% of pool per iteration — good
density for that pool).

**Coverage math for 300 shards, n_shards=20:**
- Effective new-shard rate: ~18 in iter 1, ~14 in iter 5, ~8 in iter 10, ~5 in iter 15
- Cumulative first-contact: iter 10 ≈ 120 shards (40%), iter 20 ≈ 210 shards (70%)
- For 70% in 15 iterations: need `n_shards ≈ 35-40`

**Impact:** Current config makes the warm-up phase ~2× longer than necessary for 300 shards.

---

### Gap 6: No Source-Level Diversity Quota

**What exists:** SigLIP max-min-distance diversity and source-tag fallback. No source parity
enforcement.

**What's missing:** `per_source_min: {laion_coyo: 5, journeydb: 5}` in `shard_selection`.
Important because if journeydb shards consistently score higher, laion_coyo shards will be
under-explored and have noisier attribution scores.

---

### Gap 7: FLYWHEEL-1 (Cross-Campaign Analysis) Not Implemented

BACKLOG item FLYWHEEL-1 describes campaign-level plateau detection, cross-campaign quality
comparison, warm-start decision support, and structured campaign summary generation.
`write_campaign_summary_json()` exists but `data_explorer compare` and `suggest-warmstart`
cross-campaign analysis are not implemented.

---

### Gap Summary Table

| Gap | Severity | Affects |
|-----|----------|---------|
| No coverage goal/progress tracking | High | Warm-up exit decision |
| No exploration rate schedule | High | Manual config edits between phases |
| No multi-run orchestration | High | 3-5 run strategy requires manual restarts |
| No first-contact metric per iteration | Medium | Can't see discovery rate in HTML report |
| n_shards not tuned for 300-shard pool | High | Warm-up takes 2× too long |
| No source-level diversity quota | Medium | Attribution noise asymmetry between sources |
| FLYWHEEL-1 cross-campaign analysis | Medium | No automated quality comparison across runs |

---

## Section 2: Warm-Up Campaign Runbook

**Context:** 300-400 total shards (laion_coyo + journeydb mix), M1 Max 32 GB,
~3.8 s/step at 512px. Goal: ≥70% first-contact shard coverage across 3 medium runs
while building reliable `shard_scores.db` for attribution-driven selection in later campaigns.

**Compute budget:** 1000-step iteration ≈ 63 min. 15-iteration run ≈ 15.75 h (overnight).
Three runs ≈ 47 h total wall-clock, spread across 3 nights.

---

### Run 1 — Bootstrap (Exploration-Heavy, 15 iterations)

**Goal:** First-contact coverage ≥55% of shards (≈165 of 300). Populate shard_scores.db
with enough signal that attribution starts working for a subset.

**Expected outcome:** ~165-180 shards with n_scored≥1, ~20-30 with n_scored≥2, cond_gap
baseline established, no hyperparameter ablation yet (too early).

**Config — `train/configs/flywheel_warmup_run1.yaml`:**

```yaml
flywheel:
  name: "warmup-run1"
  max_iterations: 15
  steps_per_iteration: 1000
  n_shards: 40               # increased from default 20 — covers pool faster
  poll_interval: 60
  min_free_gb: 40

  base_checkpoint: null      # start fresh, or set to existing sref-v1 best ckpt path

  training_config: "train/configs/stage1_512px.yaml"

  hyperparams:
    cross_ref_prob: 0.35
    style_loss_weight: 0.07

  shard_selection:
    performance_weight: 0.30   # low — most scored shards have only 1-2 observations
    min_diversity_pct:  0.30   # high — maximize visual diversity
    exploration_rate:   0.40   # high — aggressively explore unscored shards
    recency_penalty:    0.20   # light — avoid returning to same 40 every iteration
    recency_window_iters: 2

  min_attribution_obs: 3       # attribution won't fire much at this stage
  siglip_sample_n: 50
  shard_manifest: null

  plateau_patience:   0        # disabled — too early
  ablation_every_n:   0        # disabled — not enough attribution data yet
```

**Commands:**

```bash
# Launch
pipeline_ctl start-flywheel train/configs/flywheel_warmup_run1.yaml

# Monitor every 30 min (iterations are ~63 min)
pipeline_ctl flywheel-status

# Coverage check (run from separate terminal)
train/.venv/bin/python train/scripts/shard_selector.py status

# Full doctor check if anomaly observed
train/.venv/bin/python train/scripts/pipeline_doctor.py --ai
```

**Coverage check one-liner:**

```bash
train/.venv/bin/python - <<'EOF'
import sqlite3
db = sqlite3.connect('/Volumes/2TBSSD/shard_scores.db')
total, scored = db.execute(
    "SELECT COUNT(*), SUM(CASE WHEN n_scored>0 THEN 1 ELSE 0 END) FROM shards"
).fetchone()
two_plus = db.execute(
    "SELECT COUNT(*) FROM shards WHERE n_scored>=2"
).fetchone()[0]
print(f"Coverage: {scored}/{total} ({100*scored/total:.0f}%)")
print(f"2+ observations: {two_plus}/{total} ({100*two_plus/total:.0f}%)")
for src, n, sc in db.execute(
    "SELECT source, COUNT(*), SUM(CASE WHEN n_scored>0 THEN 1 ELSE 0 END) "
    "FROM shards GROUP BY source"
).fetchall():
    print(f"  {src}: {sc}/{n} ({100*(sc/n):.0f}%)")
EOF
```

**Success criteria to exit Run 1:**
- ≥55% of shards have n_scored≥1 (≈165 of 300)
- Both sources: ≥45% coverage each (guard against source skew)
- cond_gap mean across done iterations: ≥0.20
- No more than 2 consecutive failed iterations

**Timing:** 15 iterations × 63 min ≈ 15.75 h — leave running overnight.

---

### Run 2 — Validation (Balanced, 15 iterations)

**Goal:** Coverage ≥72%, attribution confidence building on 15-20% of shards, first ablation.

**Pre-run checklist:**
1. Run coverage check — confirm ≥55%
2. Find latest checkpoint: `ls -t /Volumes/2TBSSD/checkpoints/stage1/step_*.safetensors | head -1`
3. Paste path into `base_checkpoint` below

**Config — `train/configs/flywheel_warmup_run2.yaml`:**

```yaml
flywheel:
  name: "warmup-run2"
  max_iterations: 15
  steps_per_iteration: 1000
  n_shards: 40
  poll_interval: 60
  min_free_gb: 40

  # Warm-start from Run 1's latest checkpoint
  base_checkpoint: "/Volumes/2TBSSD/checkpoints/stage1/step_XXXXXX.safetensors"

  training_config: "train/configs/stage1_512px.yaml"

  hyperparams:
    cross_ref_prob: 0.35       # updated by ablation if it fires
    style_loss_weight: 0.07

  shard_selection:
    performance_weight: 0.45   # increase — more shards now have scores
    min_diversity_pct:  0.25
    exploration_rate:   0.30   # moderate — still covering new shards
    recency_penalty:    0.25
    recency_window_iters: 3

  temporal_decay: 0.3          # weight recent iterations 30% — activated now

  min_attribution_obs: 2       # lower — shards getting 2nd observations now
  siglip_sample_n: 50
  shard_manifest: null

  plateau_patience:   7        # soft check — won't fire in 15 iters normally
  plateau_threshold:  0.03

  # First ablation burst at iteration 8 (midpoint)
  ablation_every_n: 8
  ablation_max_runs: 4
  ablation_config: "train/configs/ablation_sref_v1.yaml"
```

**Note:** Ablation fires at iteration 8 (≈8h into Run 2), taking ~4h.
Total Run 2 wall-clock: 15×63 min + 4h ablation ≈ 20 h.

**During Run 2:**

```bash
# When ablation fires at iter 8
pipeline_ctl ablation-status

# Attribution report — useful once 15+ shards have ≥2 observations
train/.venv/bin/python train/scripts/shard_selector.py attribution | head -30
```

**Success criteria to exit Run 2:**
- ≥72% coverage (≈215 of 300 shards with n_scored≥1)
- ≥15% of shards have n_scored≥2 (attribution data for ~45 shards)
- attr_confidence≥1.0 on at least 10 shards
- Ablation identified improved hyperparams (visible in `best_hyperparams` in flywheel_history.db)
- cond_gap best ≥0.35 in at least one iteration

**Read ablation best hyperparams:**

```bash
train/.venv/bin/python - <<'EOF'
import sqlite3, json
db = sqlite3.connect('/Volumes/2TBSSD/flywheel_history.db')
db.row_factory = sqlite3.Row
r = db.execute(
    "SELECT best_hyperparams FROM campaign_summary WHERE flywheel_name=?",
    ("warmup-run2",)
).fetchone()
print(json.loads(r["best_hyperparams"]) if r and r["best_hyperparams"] else "none yet")
EOF
```

---

### Run 3 — Performance-Biased (15 iterations)

**Goal:** Coverage ≥80%, establish attribution Pareto in shard scores, cond_gap ≥0.40 sustained.

**Pre-run checklist:**
1. Coverage ≥72% confirmed
2. Pull Run 2 ablation best hyperparams (command above) — fill into `hyperparams` below
3. Find latest Run 2 checkpoint: `ls -t /Volumes/2TBSSD/checkpoints/stage1/step_*.safetensors | head -1`

**Config — `train/configs/flywheel_warmup_run3.yaml`:**

```yaml
flywheel:
  name: "warmup-run3"
  max_iterations: 15
  steps_per_iteration: 1000
  n_shards: 35               # slight reduction — focus on top performers
  poll_interval: 60
  min_free_gb: 40

  # Warm-start from Run 2's latest checkpoint
  base_checkpoint: "/Volumes/2TBSSD/checkpoints/stage1/step_XXXXXX.safetensors"

  training_config: "train/configs/stage1_512px.yaml"

  hyperparams:                # fill from Run 2 ablation output
    cross_ref_prob: ???
    style_loss_weight: ???

  shard_selection:
    performance_weight: 0.60   # shift to exploit — attribution reliable for top shards
    min_diversity_pct:  0.20
    exploration_rate:   0.20   # lower — most shards scored, focus on coverage gaps
    recency_penalty:    0.30
    recency_window_iters: 4

  temporal_decay: 0.3

  min_attribution_obs: 2
  siglip_sample_n: 50
  shard_manifest: null

  plateau_patience:   5
  plateau_threshold:  0.025
  plateau_ablation_runs: 4

  ablation_every_n: 5
  ablation_max_runs: 4
  ablation_config: "train/configs/ablation_sref_v1.yaml"
```

**Success criteria — warm-up complete:**
- ≥80% of shards have n_scored≥1 (240 of 300)
- ≥30% have n_scored≥2 (attribution data for ~90 shards)
- attr_confidence≥1.0 on at least 20 shards
- Stable cond_gap: mean of last 5 done iterations ≥0.35
- No single shard dominates: top shard n_selected ≤8 across all three runs
- Attribution Pareto identified: ≥5 shards with positive attributed_composite confirmed

---

### Exploration Rate Schedule (Manual)

The system has no built-in schedule; follow this table when starting each run:

| Run | Iterations | exploration_rate | performance_weight | ablation_every_n | Goal |
|-----|------------|-----------------|---------------------|------------------|------|
| 1   | 1–15       | 0.40            | 0.30                | 0 (disabled)     | First contact ≥55% |
| 2   | 16–30      | 0.30            | 0.45                | 8                | Coverage ≥72%, first ablation |
| 3   | 31–45      | 0.20            | 0.60                | 5                | Coverage ≥80%, exploit best shards |
| 4+  | 46+        | 0.15            | 0.65                | 5                | Phase 2 performance exploitation |

---

### Monitoring Commands Reference

```bash
# Full system check — run this first
train/.venv/bin/python train/scripts/pipeline_doctor.py --ai

# Campaign quality trend
train/.venv/bin/python train/scripts/pipeline_status.py

# Shard pool coverage and scores
train/.venv/bin/python train/scripts/shard_selector.py status

# Attribution ranking (which shards causally contribute)
train/.venv/bin/python train/scripts/shard_selector.py attribution

# View HTML report (auto-regenerated each iteration)
open /Volumes/2TBSSD/reports/flywheel/warmup-run1/index.html

# Flywheel control
pipeline_ctl pause-flywheel
pipeline_ctl resume-flywheel
pipeline_ctl stop-flywheel
pipeline_ctl flywheel-status
```

---

## Section 3: Proposed Next Steps Roadmap

### Phase 2 — Performance-Biased Exploitation (post warm-up, ~1 week after Phase 1)

**Trigger conditions:**
- Warm-up complete: ≥80% coverage, ≥20 shards with attr_confidence≥1.0
- cond_gap plateau in Run 3 (plateau_detected fires or last 5 iters improvement < 0.02)
- Ablation has identified stable best hyperparam set

**Strategy:** Shift decisively to performance-driven selection. Let attribution prune the bottom
quartile. Launch a full NSGA-II ablation campaign to find the Pareto-optimal hyperparam frontier.

**Phase 2 flywheel config changes vs Run 3:**

```yaml
n_shards: 30                    # tighter focus on top performers
shard_selection:
  performance_weight: 0.70
  exploration_rate:   0.12      # keep 12% to catch late-bloomer shards
  min_diversity_pct:  0.18
steps_per_iteration: 1500       # longer runs — better signal at this stage
ablation_every_n: 5
```

**Phase 2 ablation — switch to NSGA-II** (edit `ablation_sref_v1.yaml`):

```yaml
ablation:
  strategy: "nsga2"              # multi-objective Pareto search
  max_total_runs: 64
  n_initial: 6
  # ABL-3 Pareto warm-start seeds 3 Pareto-optimal configs from prior campaign automatically
  variables:
    cross_ref_prob:              [0.0, 0.15, 0.25, 0.35, 0.50]
    patch_shuffle_prob:          [0.0, 0.20, 0.35, 0.50]
    freeze_double_stream_scales: [true, false]
    style_loss_weight:           [0.0, 0.03, 0.07, 0.10, 0.15]
    ip_scale_init:               [0.8, 1.0, 1.2]   # new variable, open once baseline stable
```

**Phase 2 success criteria:**

## Flywheel Precompute + Versioned Publish (Added Later)

See dedicated `plans/flywheel-precompute-architecture.md` for the full Claude/AI-readable architecture doc (problem statement, the per-iter precompute + cached train + publish flow, versioning with PrecomputeCache + current update + rmtree, integration points, why it matches the "actions same but efficiency different" observation).

Inline: big explanatory comment block right before the per-iter precomp setup in `orchestrator.py` (search for "FLYWHEEL PER-ITER PRECOMPUTE").

The change lives in:
- orchestrator.py (setup, cfg capture/override, publish call inside lock try)
- data_stager.py (enhanced publish_... method with if training_cfg: versioned path using cache_manager symbols + always rmtree at end; fallback preserved)
- Config comments (run1 now uses cached base; online.yaml notes manual use)

This + the earlier per-step encoder unload/debug (in train_ip_adapter.py) gives both efficiency for flywheel and compatibility for online fallbacks.
- best_cond_gap ≥0.45 (sustained, not a spike)
- ref_gap positive (≥0.01) in at least 3 of last 5 iterations
- NSGA-II Pareto front has ≥5 non-dominated configs
- Top 10 shards by effective_score unchanged across 5 consecutive iterations

---

### Phase 3 — Long-Term Autonomous Flywheel (~1 month in)

**Strategy:** Flywheel runs continuously with minimal intervention. Ablation every 5 iters.
Checkpoints archived. Hard-example mining feeds back into shard scores. Campaigns self-supersede.

**Key enablers:**

**3a. Enable VAE tiling for 768px training (PRECOMP-1 — already implemented)**

```yaml
training_config: "train/configs/stage2_768px.yaml"
steps_per_iteration: 2000       # more steps needed at higher resolution
n_shards: 25                    # fewer shards, longer per-shard gradient signal
```

**3b. Activate temporal decay for long campaigns (SHARD-1 — already implemented)**

```yaml
temporal_decay: 0.25   # 25% weight for new obs — good for 40+ iteration campaigns
```

Early-campaign shard scores carry outdated signal (different hyperparams, lower quality
checkpoint). Temporal decay down-weights them automatically.

**3c. Hard-example mining integration**

Run after each ablation burst or every 5 iterations:

```bash
train/.venv/bin/python train/scripts/mine_hard_examples.py \
  --checkpoint /Volumes/2TBSSD/checkpoints/stage1/step_LATEST.safetensors \
  --shards-dir /Volumes/2TBSSD/shards \
  --output-dir /Volumes/2TBSSD/hard_examples
```

The hard-example density boost is already wired into `_recompute_attributed()` in
`shard_selector.py` — mining results flow into shard selection automatically.

**3d. FLYWHEEL-1 auto-succession (not yet implemented — backlog item)**

When plateau fires and ablation doesn't improve after 2 bursts, the system should
automatically archive the current campaign and warm-start a new one. Current state:
requires manual operator action. Implement after Phase 2 validates the pattern works.

Pseudocode for `_run_flywheel_loop` addition:
```python
if plateau_reason and not ablation_improved_after_n_tries(name, n=2):
    best_ckpt = fw_db.get_best(name)["checkpoint"]
    new_name = f"{name}-v{next_version}"
    fw_db.set_campaign_status(name, "superseded", superseded_by=new_name)
    launch_successor_campaign(new_name, base_checkpoint=best_ckpt, fw_cfg=fw_cfg)
```

**3e. SHARD-2 hyperparam-conditioned scores (backlog item)**

Once Phase 2 has ≥3 stable ablation regimes: partition shard scores by regime hash.
A shard scoring well under `style_loss_weight=0.12` may score poorly under `0.0`.
Implement after Phase 2 ablation has produced ≥2 stable Pareto-optimal configs.

---

### Roadmap Timeline Summary

```
Week 1:   Run 1 Bootstrap    — 15 iters, explore-heavy, ≥55% first-contact coverage
          Run 2 Validation   — 15 iters, balanced, first ablation, ≥72% coverage

Week 2:   Run 3 Performance  — 15 iters, exploit-biased, ≥80% coverage
          → Warm-up complete declared

Week 3:   Phase 2 Campaign 1 — NSGA-II ablation, 512px, push cond_gap ≥0.45
Week 4:   Phase 2 Campaign 2 — Performance plateau → archive → warm-start Campaign 2

Month 2+: Phase 3 — 768px training (PRECOMP-1), continuous autonomous flywheel,
          hard-example mining loop, temporal decay at 0.25
          → Archive best checkpoint, implement FLYWHEEL-1 auto-succession

Release:  Phase 3 target — 1024px-capable checkpoint, ref_gap ≥0.05 sustained
```

---

## Recommended Code Changes (Priority Order)

These close the most critical gaps without major refactors:

**1. Add `n_first_contact` to per-iteration HTML report** (~15 lines in `flywheel_lib.py`)

In `render_flywheel_index()`, add a "Discovery Rate" trend chart. Source: before each score
update, count shards going from `n_scored=0` to `n_scored=1`. Log in `FlywheelDB.insert_iteration`
as a new column `n_first_contact`. Without this, you can't see discovery rate in the report.

**2. Add coverage check to pipeline_doctor** (~10 lines in `pipeline_doctor.py`)

After reading shard stats, emit `coverage_pct` in the doctor JSON. Flag as WARNING if <50%,
INFO if 50-80%, OK if ≥80%. This makes the "warm-up complete?" question answerable from
the doctor without a separate query.

**3. Add exploration schedule support to `select_shards()`** (~20 lines in `shard_selector.py`)

Accept `exploration_schedule: list[{through_iteration, rate}]` in the cfg dict. Look up current
rate from schedule before computing n_explore. No orchestrator changes needed.

**4. Log `base_checkpoint` to campaign_summary** (~5 lines in `flywheel_lib.py`)

Add `base_checkpoint` column to `campaign_summary`. When starting Run 2, query cold DB to find
what the next run should warm-start from instead of grepping log files.

Items 1 and 2 are highest priority — without them, warm-up phase progress is opaque.

---

## Section 3: Attribution Warmth and the Run-vs-Ablate Decision

*(added 2026-06, from warmup-run2 iter 1–2 — capture so the reasoning isn't lost)*

### How a shard earns attribution (and why it's slow)

`ShardScoreDB` keeps two EMAs per shard: an **included** EMA (updated when the shard is
trained on) and an **excluded** EMA (updated when an iteration runs *without* it). The
causal/contrastive score is `included_mean − excluded_mean`, trusted only once
`attr_confidence = hmean(n_inc, n_exc) / MIN_ATTR_OBS` reaches 1.0 (`MIN_ATTR_OBS = 3`).

Because it's a **harmonic mean**, attr_confidence is dominated by the *smaller* of the two
counts. Excluded obs pile up fast (a shard sits out most iterations), so **`n_inc` — the
number of iterations the shard was actually *trained on* — is the binding constraint.** In
practice attr_confidence hits 1.0 once a shard has been *included* ~2–3 times (excluded
plentiful). So: **a shard needs to have been selected into ≈3 training iterations to be
attributed**, and the excluded side is automatic.

### Why full-pool attribution is structurally unreachable here

With ~42 shards selected per iteration from a 1280-shard pool, a *uniformly* chosen shard
expects only ≈ 15 × 42 / 1280 ≈ **0.49 inclusions over the entire 15-iteration budget.**
Under uniform selection essentially **no shard would ever reach 3 inclusions** — attribution
could never warm.

It works at all only because selection is **UCB, not uniform**: it re-selects high-value
shards (exploitation), so inclusions concentrate on the "good" head of the pool. Those warm
up; the long tail never does. **This is correct, not a bug** — you only need causal
attribution on the shards you'd actually keep. Attribution is a property the bandit *earns
for its favourites*; full-pool coverage is neither expected nor needed within a campaign.
Live confirmation (warmup-run2 after iter 2): `0/1280 fully attributed, 88 touched` — cold
this early, exactly as the math predicts.

### The decision: run the warmup out, or stop and ablate a recipe?

Two optimisation loops, often confused:
- **Shard bandit** (UCB over `shard_scores.db`) optimises *which data*.
- **Ablation harness** (`ablation_sref_v1.yaml`) optimises *the recipe* (hyperparams).

Ablation is **deliberately disabled** in the warmup run (`ablation_every_n: 0 — no
attribution signal yet`) because tuning the recipe against cold, noisy shard scores tunes
against a moving target.

**So the branch trigger is NOT "iter 15" and NOT "first plateau" — it is "stall detector
fires AND attribution is warm."**
- Discontinuing at the first plateau would ablate against cold attribution — the exact
  failure the config guards against.
- Running the warmup forward warms **two** assets in parallel: the amortised precompute pool
  **and** the shard-attribution scores ablation needs. "We may as well run it" is right — the
  warmup is *earning* the signal that makes a later ablation meaningful, not just burning
  compute. (Precompute is encoder-identity-keyed, so it's a shared capital investment reused
  by every future run — see the throughput note.)
- The **champion is always preserved** (`get_best` keeps the max-cond_gap iter), so extra
  iterations never damage the result; a bad draw is discarded.

When the trigger fires, branch to **run3 (recipe campaign)**: warm-start from the champion,
inherit the now-warm shard scores, enable ablation (`ablation_config:
train/configs/ablation_sref_v1.yaml`, `ablation_every_n: >0`, `ablation_max_runs: ~12`).
Precompute amortisation makes run3's first-contact cost small.

### The empirical signal so far (don't over-read one iteration)

| iter | train_loss | cond_gap | note |
|------|-----------|----------|------|
| 1 | 1.0043 | **+0.0273** ★ champion | |
| 2 | 0.5328 | **−0.0054** | flow loss halved but cond_gap went negative |

iter-2's pattern — **flow loss falls while cond_gap (the signal that matters) drops** — is the
classic "fitting the objective, not the conditioning." But at iter 2 it's indistinguishable
between (a) stinker shard draws (data) and (b) a recipe ceiling, *because attribution is cold
and you can't yet ablate to tell.* Discriminator: if iter-3/4 draw different mixes and recover
toward +0.0273 it was data; if they keep regressing across mixes it's the recipe.

### The recipe-ablation config (`ablation_sref_v1.yaml`)

Already sweeps the four knobs that matter for this regression, via `strategy: bayesian`
(Optuna) over `cross_ref_prob`, `patch_shuffle_prob`, `style_loss_weight` (the augmentation/
loss terms shaping ref_gap/cond_gap; `style=0` isolates pure conditioning), and
`freeze_double_stream_scales` — which **directly tests the observed `double=0.0` single-only
injection.**

**Recommended addition for the iter-2 regression:** add `learning_rate: [5e-5, 1e-4, 2e-4]`
to `variables`. iter-2's loss-falls / cond_gap-drops is the signature of an lr too high for
stable conditioning, and lr is currently NOT in the sweep. (Expands the grid; bayesian
sampling absorbs it.)

### Observability in place (read these to make the call)
- `debug/flywheel_refgap.py <campaign>` — per-iter cond_gap/ref_gap + champion (★) + an
  **attribution-warmth line** ("attribution: N/1280 fully attributed; M touched —
  ablation-ready: yes/no", floor self-calibrated to 2× the median per-iter shard count).
- `pipeline_doctor.py` — the `cond_gap`-stall detector fires after 3 done iters set no new
  high, and its WARNING now states attribution-readiness inline (branch-to-ablation vs
  keep-warming). `_attribution_warmth()` reads `shard_scores.db`, fail-open.

### Open item — recalibrate the ablation-ready floor
The floor is currently `attr ≥ 2 × per_iter` (≈84). Since only the **exploited head** warms
(not the pool), that's probably too high — a usable warm set may be 30–40 shards. Recalibrate
once a few iterations show how fast the head warms; tie it to "enough warm shards to fill one
ablation training set," not a pool fraction.

---

## Appendix: Key File Locations

| Resource | Path |
|----------|------|
| Hot shard scores DB | `/Volumes/2TBSSD/shard_scores.db` |
| Hot flywheel history DB | `/Volumes/2TBSSD/flywheel_history.db` |
| Cold metadata | `/Volumes/16TBCold/metadata/` |
| Cold shard scores DB | `/Volumes/16TBCold/metadata/shard_scores.db` |
| Cold ablation history DB | `/Volumes/16TBCold/metadata/ablation_history.db` |
| Checkpoints | `/Volumes/2TBSSD/checkpoints/stage1/` |
| Flywheel HTML reports | `/Volumes/2TBSSD/reports/flywheel/<name>/index.html` |
| Shard pool | `/Volumes/2TBSSD/shards/` |
| Hard examples | `/Volumes/2TBSSD/hard_examples/` |
| Flywheel configs | `train/configs/flywheel_*.yaml` |
| Ablation config | `train/configs/ablation_sref_v1.yaml` |
| Key scripts | `train/scripts/shard_selector.py`, `flywheel_lib.py`, `orchestrator.py` |
