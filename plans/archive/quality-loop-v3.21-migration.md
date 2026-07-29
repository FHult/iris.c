# v3.20.0 → v3.21.0 Migration Guide — Closing the Quality & Experiment Loop

v3.21.0 adds four subsystems that make large flywheel campaigns trustworthy with
the proxy VAE enabled: **experiment tracking**, **monitoring + alerts**,
**golden-set quality validation**, and **data-flywheel closure (preferences)**.

**Nothing changes for existing runs unless you opt in** — all four are additive,
config-gated, and store state in their own SQLite DBs on the hot volume.

The pure decision/recording logic is implemented and unit-tested now (227 → ~291
train/ tests). The GPU-bound execution (golden-set 3-arm training, image
generation) is scaffolded and runs when the pipeline is idle — see "GPU-gated"
below.

---

## 1. Config changes (`v2_pipeline.yaml`)

Three new top-level sections (all optional; defaults are safe):

```yaml
experiments:
  rank_metric: "clip_i"          # Champion/Challenger ranking metric
  promote_min_margin: 0.01       # hysteresis vs the incumbent champion

monitoring:
  enabled: true
  alerts: []                     # [] = built-in defaults (fallback rate, quality drop, disk/mem)

golden_set:
  manifest: null                 # path to the golden-set shard manifest (built once)
  size: 3000                     # 2000–5000 images, stratified by source
  steps: 500
  seed: 1234
  regression_tolerance: 0.03     # auto-disable proxy if it degrades a key metric > 3%
  eval_frequency: "per_campaign"
```

No existing keys are renamed or removed.

---

## 2. New state (SQLite, hot volume)

| DB | Module | Holds |
|----|--------|-------|
| `experiments.db` | `experiments/registry.py` | per-model records + golden results + Champion/Challenger |
| `monitoring.db`  | `monitoring/trends.py`    | metric time-series (fallback rate, quality, disk/mem) |
| `preferences.db` | `experiments/preferences.py` | human/self preference signals + synthetic provenance |

All take an explicit `db_path` (override via constructor or `PIPELINE_DATA_ROOT`),
so they never collide with test or other-machine state.

---

## 3. New `pipeline_doctor.py` commands

```bash
# Monitoring dashboard — trends + active alerts over a window
python train/scripts/pipeline_doctor.py --monitor --history 30

# Experiment Champion/Challenger ranking from golden-set evals
python train/scripts/pipeline_doctor.py --quality-report
```

Both degrade gracefully when no data has been recorded yet.

---

## 4. Experiment tracking (`experiments/`)

Every trained model becomes an experiment record. CLI:

```bash
python -m experiments.tracker list
python -m experiments.tracker report   --metric clip_i
python -m experiments.tracker champion  --metric clip_i
python -m experiments.tracker promote   --metric clip_i --min-margin 0.01
python -m experiments.tracker compare exp_0001 exp_0002
```

Programmatic (e.g. from the orchestrator after a campaign finishes):

```python
from experiments.registry import ExperimentRegistry
from experiments.tracker import record_from_campaign
reg = ExperimentRegistry()
eid = record_from_campaign(reg, campaign="warmup-run1",
                           weights_path="…/step_1000.safetensors",
                           proxy_enabled=True, proxy_mode="balanced",
                           proxy_fallback_rate=0.03, flywheel_db=fw_db)
```

Golden results attach via `reg.attach_golden(eid, arm_metrics)`; `promote_champion`
applies the Champion/Challenger statuses (with a hysteresis margin so the champion
doesn't churn on metric noise).

---

## 5. Golden-set validation (`evaluate_golden_set.py`)

The trust test: a 3-arm downstream A/B (real / proxy+fallback / proxy-forced) on a
fixed stratified golden set, gated by relative degradation vs the real-VAE arm.

```bash
python train/scripts/evaluate_golden_set.py \
    --proxy …/proxy_final.safetensors \
    --golden-manifest …/campaigns/golden/manifest.json \
    --config train/configs/v2_pipeline.yaml \
    --steps 500 --seed 1234 --tolerance 0.03 \
    --campaign golden-eval-2026-06
```

The **regression gate** (`regression_gate()`) decides pass/fail per metric; on
failure `maybe_disable_proxy()` sets `proxy_vae.enabled=false` in the config with a
reason, and `write_results()` records the experiment + trend points. These are pure
and tested; the per-arm training/scoring is **GPU-gated** (see below).

**Recommended golden set:** 3,000 images, stratified ≥30% natural (LAION/COYO) vs
synthetic (JourneyDB). **Eval frequency:** once per campaign (`per_campaign`), plus
any time the proxy checkpoint changes.

---

## 6. Data-flywheel closure (`experiments/preferences.py`)

Lets human / self / automated preference signals nudge future unified scores, and
tracks synthetic-generation provenance so generated images stay distinguishable
from real data.

```python
from experiments.preferences import PreferenceStore, apply_preferences
store = PreferenceStore()
store.record_preference("000042", source="human", value=0.8)   # value ∈ [-1, 1]
new_scores = apply_preferences(unified_scores, store, pref_weight=0.10)
```

`blend_preference(base, pref, weight)` is conservative by default (max ±0.10 shift)
so preference nudges ranking without overriding static quality. Per-source trust:
human 1.0 > self 0.5 > auto 0.3.

---

## 7. What is GPU-gated (runs when the pipeline is idle)

These need the GPU, which a live flywheel holds — they are built and refuse to run
while the GPU lock is held (pass `--force` to override on a dedicated box):

- **Golden-set 3-arm training + scoring** (`run_golden_eval`) — reuses the
  `compare_downstream_quality.py` per-arm machinery; raises a clear
  `NotImplementedError` pointing at the idle-GPU requirement until wired.
- **Image generation** for the data flywheel — sampling from the champion model to
  produce candidate synthetic data. The provenance store + scoring blend are ready;
  the generation loop is the GPU-bound piece to add.

Everything else (records, ranking, gate decision, trends, alerts, preference blend,
the doctor dashboards) works now without a GPU.

---

## 8. Recommended rollout

1. Build the golden-set manifest once (3,000 stratified images) →
   `golden_set.manifest`.
2. On an idle GPU, run `evaluate_golden_set.py` for the current proxy → writes an
   experiment + the gate verdict.
3. Check `pipeline_doctor.py --quality-report` (ranking) and `--monitor` (trends).
4. Enable the proxy (`proxy_vae.enabled: true`) only after the gate passes; the
   gate will auto-disable it again if a later eval regresses.
5. Wire `record_from_campaign` into the orchestrator's post-campaign path and the
   monitoring collectors into the flywheel loop so the trends populate automatically.
