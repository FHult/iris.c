# FLYWHEEL-1: Long-term Campaign Management and Cross-Campaign Analysis

## Status

**Design complete. Not yet implemented.**

Prerequisite: TRAIN-7 Stage 2 running (need real multi-campaign data to exercise this code).
Can be implemented in parallel with TRAIN-7 — no shared files changed.

---

## What already exists (do not re-implement)

Reading the code before designing revealed that significant infrastructure is already in place:

| Component | Location | Status |
|---|---|---|
| `FlywheelDB` with `iterations`, `checkpoint_log`, `campaign_summary` tables | `flywheel_lib.py:129` | Done |
| Within-campaign plateau detection (`check_plateau`) | `flywheel_lib.py:373` | Done |
| Plateau → auto-pause via `FLYWHEEL_CONTROL_FILE` | `orchestrator.py:2929` | Done |
| Per-iteration HTML report with cond_gap trend + Pareto scatter + shard heatmap | `flywheel_lib.py:411` | Done |
| `campaign_summary` upsert after each iteration | `flywheel_lib.py:266` | Done |
| `data_explorer campaign-summary` subcommand | `data_explorer.py:2330` | Done |
| `data_explorer compare` (two-campaign side-by-side) | `data_explorer.py:1711` | Done, but limited (see §3.2) |
| `data_explorer suggest-warmstart` | `data_explorer.py:1543` | Done, but limited (see §3.3) |
| `data_explorer selection-history` | `data_explorer.py:2296` | Done |

---

## 1. Problem Statement

The orchestrator and flywheel loop manage a single campaign running iteration-by-iteration. What does not exist is anything above that level: tracking whether a campaign is still productive vs. played out, knowing when to start a fresh campaign vs. continue the current one, comparing quality across multiple campaigns that ran weeks apart, and generating a durable summary when a campaign ends. Currently:

- `campaign_summary.status` is always `'active'` — nothing ever writes `plateau`, `completed`, or `superseded`
- The doctor has no campaign-level signals: it shows step-level loss/NaN anomalies but is silent about whether the current campaign is progressing
- Cross-campaign comparison reads cold weight dir JSONs (not FlywheelDB), so campaigns not yet archived to cold are invisible; and there's no regression detection
- No campaign summary JSON is ever written to cold storage — accumulated knowledge is only in the SQLite DB
- `suggest-warmstart` finds the best cond_gap checkpoint but ignores ablation history and step coverage

---

## 2. Outcomes (from backlog)

Quoting the backlog verbatim as the specification anchor:

1. **Campaign lifecycle states**: `active`, `plateau`, `completed`, `superseded`
2. **Campaign-level plateau detection** — rolling mean cond_gap flat for N flywheel iterations; emit doctor WARNING with recommendation (new ablation config or warm-start new campaign)
3. **Cross-campaign comparison** — is campaign B better than A? Are we regressing on CLIP-I while improving cond_gap?
4. **Warm-start decision support** — when plateau detected, recommend highest-leverage starting point; consider best historical CLIP-I, Pareto-optimal ablation configs, step coverage
5. **Campaign summary generation** — at campaign end: total steps, peak CLIP-I, cond_gap trajectory, ablation iterations, shards consumed, wall-clock time; written to `metadata/flywheel_logs/` and `weights/flywheel-{date}/summary.json`

---

## 3. Gap Analysis

### 3.1 Campaign lifecycle states

The `campaign_summary` table has a `status` column (TEXT, default `'active'`), but it is never written. There are four target states:

- **active** — flywheel loop running or paused temporarily; normal progression
- **plateau** — within-campaign cond_gap spread below threshold for `plateau_patience` iterations; operator must decide whether to continue or move on
- **completed** — operator explicitly marked this campaign as done; weights archived; summary written
- **superseded** — a later campaign exceeds this one on all primary metrics; annotated but weights preserved

Transitions:
```
active ──plateau detected──► plateau ──operator force-continue──► active
                                    ──operator mark-completed───► completed
active ──operator mark-completed──► completed
completed / plateau ──new campaign exceeds on all metrics──► superseded
```

The plateau → active re-entry is already handled via `pipeline_ctl force-continue-flywheel` which deletes the control file. The missing pieces are:
- Writing `status='plateau'` to `campaign_summary` when plateau auto-pause fires
- `pipeline_ctl mark-completed <name>` — writes `status='completed'`, triggers summary export
- Automatic `superseded` annotation when a new campaign's `best_cond_gap` exceeds an older one on all metrics

### 3.2 `data_explorer compare` limitations

Current implementation (`data_explorer.py:1711`):
- Reads checkpoint JSONs from `cold_weights/flywheel-{date}/` — requires campaigns to be archived to cold; active or recently-completed campaigns are not visible
- Compares only `cond_gap`, `ref_gap`, `clip_i` as min/max/last across checkpoint files
- No regression detection (e.g., CLIP-I declining while cond_gap rises)
- No Pareto frontier view across campaigns
- Requires knowing the exact date key of both campaigns

Target: `compare` reads from `FlywheelDB.campaign_summary` and `FlywheelDB.iterations` directly, which are always current. The cold weight dir read is a fallback for campaigns predating the DB.

### 3.3 `suggest-warmstart` limitations

Current implementation (`data_explorer.py:1543`):
- Iterates cold weight dirs to find the checkpoint with maximum `cond_gap` in metadata JSON
- No reference to `ablation_history.db` (which configs are Pareto-optimal)
- No step coverage awareness (avoid warm-starting from a checkpoint that already ran 50K steps on a poor config)
- No explanation beyond the campaign name and cond_gap value
- Cannot identify if the warm-start candidate is from a superseded campaign

Target: query FlywheelDB + ablation_history.db together to emit a structured recommendation with reasoning.

### 3.4 Doctor integration

`pipeline_doctor.py` currently shows:
- Active training step, loss, ETA
- Prep worker status
- Dispatch queue issues

It does not show:
- Whether the current flywheel campaign is in `active` or `plateau` state
- How many iterations have run and what the cond_gap trajectory looks like
- Whether a cross-campaign regression has been detected

Campaign-level signals belong in the doctor because they are actionable at the same operator decision horizon as dispatch issues.

### 3.5 Campaign summary export

`refresh_campaign_summary()` (`flywheel_lib.py:266`) maintains the in-DB row. What does not exist:

- A durable JSON file written to `metadata/flywheel_logs/campaign-{name}.json` when a campaign ends (or is checkpointed mid-run). This is the cold-storage accumulator that persists beyond DB corruption or migration.
- A `summary.json` written alongside weights in `weights/flywheel-{date}/` so the weight archive is self-describing without needing the DB.

---

## 4. Design

### 4.1 Schema additions (flywheel_lib.py)

Add two columns to `campaign_summary` via migration:

```sql
ALTER TABLE campaign_summary ADD COLUMN clip_i_best REAL;
ALTER TABLE campaign_summary ADD COLUMN clip_i_final REAL;
```

`clip_i` is already in the cold weight checkpoint JSONs but not in the in-DB campaign summary. Needed for cross-campaign regression detection (CLIP-I declining while cond_gap rises).

No other schema changes. The `status` column already exists; we just start writing it.

### 4.2 Campaign lifecycle state machine

**Where it runs:** three call sites in `orchestrator.py._run_flywheel_loop` + one in `pipeline_ctl`.

**Write `status='plateau'`** — in `_run_flywheel_loop`, immediately after the existing plateau pause block (line 2929):

```python
if plateau_reason:
    fw_db.set_campaign_status(name, "plateau")
```

**Write `status='active'`** — when the operator resumes (control file deleted), reset status to `'active'` at the top of the next iteration before training starts. Add a call in the existing "check control file" path in `_check_flywheel_control`.

**`pipeline_ctl mark-completed <campaign_name>`** — new subcommand:
1. Set `campaign_summary.status = 'completed'`
2. Call `_write_campaign_summary_json(name, db, cold_root)` — see §4.5
3. Check all other campaigns: if any has `best_cond_gap < new_completed.best_cond_gap` AND `best_ref_gap < new_completed.best_ref_gap`, mark those as `superseded`
4. Print confirmation

**Automatic `superseded` check** — also run at the end of `_run_flywheel_loop` after `max_iters` is reached (the loop naturally completes). Same logic: compare new campaign's bests against all completed campaigns.

New methods on `FlywheelDB`:
- `set_campaign_status(name, status)` — writes `status` and `ts_last`
- `get_completed_campaigns()` — returns `campaign_summary` rows where `status IN ('completed', 'superseded')`

### 4.3 Cross-campaign comparison (data_explorer compare)

Rewrite `_cmd_compare` to read from FlywheelDB first, falling back to cold weight dir JSONs:

```
data_explorer compare [--a <name>] [--b <name>] [--all] [--json]
```

- `--all`: print a table of all campaigns sorted by `best_cond_gap DESC`; no argument required
- `--a / --b`: side-by-side as today, but reading from FlywheelDB

**Regression detection** — when comparing A vs B, flag if:
- `cond_gap_b > cond_gap_a` (cond_gap improved) BUT `clip_i_b < clip_i_a - 0.02` (CLIP-I regressed by >0.02)

This is the canonical "we're overfitting or trading off" pattern. Report it as a WARNING in the comparison output.

**Pareto frontier** — across all completed campaigns, compute the Pareto frontier on (best_cond_gap, best_clip_i). A campaign is Pareto-optimal if no other campaign exceeds it on both metrics simultaneously. Print which campaigns are on the Pareto front.

**Output format (text mode)**:
```
  All campaigns (4 total, 2 completed, 1 plateau, 1 active)

  name              status      iters  steps   best_cond_gap  best_ref_gap  clip_i  pareto
  ────────────────  ──────────  ─────  ──────  ─────────────  ────────────  ──────  ──────
  sref-v1           completed      12   60000        +0.1823       +0.0921   0.571   ★
  sref-v2           completed       8   40000        +0.2141       +0.1043   0.549      ← regression: CLIP-I -0.022
  sref-v3-768px     plateau         6   30000        +0.2389       +0.1102   0.562   ★
  sref-v4-768px     active          3   15000        +0.2511          —         —
```

### 4.4 Warm-start decision support (suggest-warmstart enhanced)

Rewrite `_cmd_suggest_warmstart` to:

1. Query FlywheelDB `campaign_summary` for the campaign with highest `best_cond_gap` that is not `superseded`
2. Cross-reference `ablation_history.db` to find if any Pareto-optimal config has not yet been used as the training config for a full campaign (i.e., ablation proved it's better but it was never promoted to a full run). If yes, include that in the recommendation.
3. Report total step coverage for the recommended checkpoint (total steps across all iterations in that campaign), so the operator knows how much training has already been invested
4. If the recommended campaign is in `plateau` state, warn explicitly — warm-starting from a plateau checkpoint is likely correct, but should be intentional

Output format:
```
  Warm-start recommendation:

    Campaign      : sref-v3-768px  (status: plateau)
    Checkpoint    : /Volumes/16TBCold/weights/flywheel-20260521/step_30000.safetensors
    cond_gap      : +0.2389
    clip_i        : 0.562
    Steps covered : 30,000 (6 iterations × 5,000 steps each)

    Reasoning:
      - Best cond_gap across non-superseded campaigns
      - sref-v3-768px is in plateau (last 5 iters spread < 0.02)
        Suggest: warm-start a new campaign with a different config rather than continuing

    Ablation insight:
      - Ablation run 'sref_abl_004' found config A Pareto-optimal (cond_gap +0.031, clip_i +0.008)
        but it was never promoted to a full flywheel campaign
      - Consider: new campaign using config A as hyperparams

  CLI:  --warmstart "/Volumes/16TBCold/weights/flywheel-20260521/step_30000.safetensors"
```

### 4.5 Campaign summary export

New function `_write_campaign_summary_json(name, db, cold_root)` called from:
- `pipeline_ctl mark-completed`
- `_run_flywheel_loop` at natural loop completion (`iteration > max_iters`)
- `_run_flywheel_loop` on plateau pause (writes a mid-run snapshot, not final)

JSON written to two locations:
1. `cold_root/metadata/flywheel_logs/campaign-{name}.json`
2. The matching cold weight dir `cold_root/weights/flywheel-{date}/summary.json` (if dir exists)

Schema:
```json
{
  "name": "sref-v3-768px",
  "status": "completed",
  "n_iterations": 12,
  "total_steps": 60000,
  "total_elapsed_secs": 172800,
  "best_cond_gap": 0.2389,
  "best_ref_gap": 0.1102,
  "best_clip_i": 0.562,
  "best_checkpoint": "/Volumes/.../step_30000.safetensors",
  "final_cond_gap": 0.2301,
  "final_ref_gap": 0.1089,
  "cond_gap_trajectory": [0.18, 0.21, 0.23, 0.238, 0.239, ...],
  "ablation_runs": ["sref_abl_002", "sref_abl_004"],
  "shards_consumed": ["000042", "000107", ...],
  "ts_start": "2026-05-21T10:00:00Z",
  "ts_end": "2026-05-23T18:00:00Z",
  "git_commit": "abc1234"
}
```

The `cond_gap_trajectory` and `shards_consumed` arrays make this file useful for long-term trend analysis without querying the DB.

### 4.6 Doctor integration

Add a `_check_campaign_state(db)` function to `pipeline_doctor.py`:

```python
def _check_campaign_state(db: FlywheelDB) -> list[dict]:
    """Return doctor issues for campaign-level anomalies."""
    issues = []
    summaries = db.get_campaign_summaries()
    active = [s for s in summaries if s["status"] == "active"]
    plateau = [s for s in summaries if s["status"] == "plateau"]

    for s in plateau:
        issues.append({
            "level": "warning",
            "msg": f"Campaign '{s['flywheel_name']}' is in plateau state "
                   f"(best_cond_gap={s['best_cond_gap']:.4f})",
            "action": "run: data_explorer suggest-warmstart  OR  pipeline_ctl force-continue-flywheel",
            "context": {"campaign": s["flywheel_name"], "best_cond_gap": s["best_cond_gap"]}
        })

    for s in active:
        # Warn if active campaign has no iteration recorded in the last 4 hours
        # (flywheel may be stuck without having triggered a normal anomaly)
        age_h = _hours_since(s.get("ts_last"))
        if age_h is not None and age_h > 4:
            issues.append({
                "level": "warning",
                "msg": f"Active campaign '{s['flywheel_name']}' has no update for {age_h:.1f}h",
                "action": "check flywheel tmux window and orchestrator log",
                "context": {"campaign": s["flywheel_name"], "stale_hours": age_h}
            })

    return issues
```

Surface these in `pipeline_doctor.py --ai` output under `issues`, alongside the existing step-level anomalies.

---

## 5. Implementation Phases

### Phase 1 — Campaign lifecycle states + doctor (≈ 4h)

**Files changed:** `flywheel_lib.py`, `orchestrator.py`, `pipeline_ctl.py`, `pipeline_doctor.py`

1. **`flywheel_lib.py`**:
   - Add `set_campaign_status(name, status)` method to `FlywheelDB`
   - Add `get_completed_campaigns()` method
   - Add v4 schema migration for `clip_i_best` and `clip_i_final` columns on `campaign_summary`

2. **`orchestrator.py`**:
   - After plateau auto-pause fires (line 2932), call `fw_db.set_campaign_status(name, "plateau")`
   - At loop completion (`iteration > max_iters`), call `fw_db.set_campaign_status(name, "completed")` then `_write_campaign_summary_json`
   - In `_check_flywheel_control`, when resume is detected, call `fw_db.set_campaign_status(name, "active")`

3. **`pipeline_ctl.py`**:
   - Add `mark-completed <campaign_name>` subcommand
   - Add `mark-superseded <campaign_name> --by <newer_name>` subcommand (manual annotation)

4. **`pipeline_doctor.py`**:
   - Add `_check_campaign_state()` function
   - Call it from `run_doctor()` and include in `--ai` JSON `issues` list

**Validation:** run `data_explorer campaign-summary` after a plateau pause fires; confirm status shows `plateau`. Run `pipeline_ctl mark-completed sref-v1`; confirm JSON written to cold.

---

### Phase 2 — Rich cross-campaign comparison + regression detection (≈ 3h)

**Files changed:** `data_explorer.py`

1. Rewrite `_cmd_compare` to:
   - Read from `FlywheelDB.campaign_summary` as primary source
   - Support `--all` flag for multi-campaign table
   - Add regression detection (CLIP-I decline while cond_gap rises) with warning annotation
   - Add Pareto frontier computation and marking

2. Update `data_explorer` help text and `_SUBCMDS` accordingly

**Validation:** run `data_explorer compare --all` with at least two campaigns in the DB. Manually inject a regression scenario (campaign B has higher cond_gap but lower clip_i_best) and confirm the warning appears.

---

### Phase 3 — Campaign summary export + suggest-warmstart enhancement (≈ 3h)

**Files changed:** `flywheel_lib.py`, `data_explorer.py`, `orchestrator.py`

1. **`flywheel_lib.py`**:
   - Add `_write_campaign_summary_json(name, db, cold_root)` function

2. **`orchestrator.py`**:
   - Call `_write_campaign_summary_json` at loop completion and on `mark-completed`
   - Call a mid-run checkpoint version on plateau pause

3. **`data_explorer.py`**:
   - Rewrite `_cmd_suggest_warmstart` to query FlywheelDB + ablation_history cross-reference
   - Add step coverage and ablation insight to output

**Validation:** run `pipeline_ctl mark-completed sref-v1`; confirm both `metadata/flywheel_logs/campaign-sref-v1.json` and `weights/flywheel-{date}/summary.json` are written with correct fields. Run `data_explorer suggest-warmstart` with multiple campaigns and confirm it picks the non-superseded best.

---

## 6. What Phase 1 unblocks

Phase 1 alone delivers the most operator value: the doctor starts reporting campaign plateau state, and `pipeline_ctl mark-completed` provides a defined workflow for ending campaigns cleanly. Phases 2 and 3 are additive — they improve analysis quality but do not block anything.

**Recommended order:** implement Phase 1 as soon as TRAIN-7 Stage 2 is running and a first campaign exists with real data. Implement Phases 2 and 3 after the second campaign completes (need two campaigns to test comparison).

---

## 7. What this does NOT do

- No automated campaign termination — the operator always decides when a campaign is complete. The system recommends; the operator acts.
- No automated launch of a new campaign on plateau — too risky as an auto-action. The doctor issues a recommendation; `pipeline_ctl` executes it.
- No model of why a campaign plateaued (data exhaustion vs. config ceiling vs. architecture limit) — that requires ablation, which is already wired.
- No changes to the within-campaign flywheel loop logic, shard scoring, or ablation harness — FLYWHEEL-1 is purely the meta-layer above those.
