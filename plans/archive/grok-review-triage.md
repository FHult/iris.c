# Grok Review Triage — consolidated & de-duplicated

Triage of the 7 `grok_*.md` static-review reports (bug_report, train_bug_report,
testing_bug_report, safe_flywheel_metal, safe_flywheel_vaeparser_roadmap,
safe_flywheel_reviews, audit_plan, plus proxy_vae_analysis). The same findings
recur across reports; this is the merged, prioritized list with disposition.
Disposition key: **DONE** · **FIX-NOW** (safe, GPU-free, not live-pipeline) ·
**GATED** (needs GPU / hardware / live-pipeline-idle) · **TRACKED** (in BACKLOG) ·
**WON'T** (intentional / not a defect).

## CRITICAL

- **G-1: C-side IP-Adapter not implemented.** `train/export/iris_ip_adapter.h` declares
  the full loader/Perceiver/inject surface; there is no `.c`, and `main.c` hard-errors on
  `--sref`. Trained adapters (the entire flywheel output) cannot yet run in the `iris`
  binary. Recurs in bug_report C-01, audit_plan CRITICAL, vaeparser A4, reviews.
  **Disposition: GATED / headline open feature — now has a phased plan:
  `plans/c-ip-adapter.md`.** Port Perceiver + inject hooks from
  `train/ip_adapter/model.py`, wire into double/single block forward (CPU + MPS), parity
  vs `test_ip_adapter_inference.py`. The plan phases it: (0/1) `iris_ip_adapter.c` +
  parity fixtures, (2) transformer hooks, (3) SigLIP-in-C for interactive `--ip`, (4) the
  separate training-free `--sref`. Key unlock: a **precomputed-feature path**
  (`--ip-features`) ships adapter evaluation in C before the SigLIP port. Start once the
  campaign yields a champion checkpoint.

## HIGH

- **G-2: MLOps state machine ~0 unit tests.** orchestrator (~3.5k LOC), doctor (~3.3k),
  cache_manager, flywheel, data_stager have near-zero dedicated tests (testing_report).
  **Disposition: IN PROGRESS.** The GROK-TEST pure-core extraction pattern
  (`_retry_policy`, `_ready_gate`, `_resolve_exploration_rate`, `_quality_gate_target`,
  `test_orchestrator_state.py`, `test_shard_selector.py`, …) is the ongoing answer;
  continue extracting decision cores + hermetic tests.

- **G-3: Pervasive hardcoded paths.** ~90 `/Volumes/…` + `/Users/fredrikhult/…` literals
  (incl. a personal-username `ultrahot` default and test assertions on exact mounts:
  `test_pipeline_storage.py`). Runs only on the author's layout (train_bug_report H-T01).
  **Disposition: FIX-NOW but live-pipeline-sensitive.** Env/`load_config` overrides exist
  for the orchestrated paths (the flywheel sets `PIPELINE_DATA_ROOT`), so it's not breaking
  today. Centralize the remaining module-level constants + argparse defaults behind config/
  env, and make test assertions layout-relative. Do it in a flywheel-idle window (touches
  `pipeline_lib.py` constants the next spawned worker reads).

- **G-4: B-METAL-01 — CPU softmax fallback in the hot attention path.** If shaders fail to
  init, every attention call syncs to CPU + 2× memcpy (metal reports; matches
  `plans/metal_optimization_backlog.md`). **Disposition: GATED (GPU).** Fix = fatal-on-init-
  fail or `MPSMatrixSoftMax` guaranteed-GPU fallback + a regression test asserting no
  `waitUntilCompleted` inside attention. Needs GPU validation; do at M-series idle.

## MEDIUM

- **G-5: Brittle ad-hoc JSON config parsers (5+).** **Disposition: DONE (this session).**
  VAE config → `iris_vae_config.h` + golden tests (C-2); transformer/Z-Image/model_index →
  `iris_config_parse.h` (`cfg_int/float/bool/int_array/contains`) + `test_config_parse.c`.

- **G-6: VAE↔teacher parity assumed, not tested.** **Disposition: DONE (this session).**
  `debug/vae_parity.c` + `gen_vae_parity_fixture.py` (real-weights, on demand),
  `test_bn_pack.py` + `test_encode_golden` (hermetic). Surfaced + fixed VAE-Q1 (train/infer
  latent convention) and VAE-1 (conv buffer aliasing).

- **G-7: Monolithic modules** (flux transformer ~5k, metal ~7k, orchestrator/doctor ~3k+).
  **Disposition: TRACKED.** Maintainability; addressed incrementally by pure-core extraction
  (pipeline) — C-side split is larger and lower priority than G-1.

- **G-8: Doctor fix commands use `shell=True` + broad `except Exception`.** Operator-
  triggered remediation (train_bug_report). **Disposition: TRACKED.** Lower risk (operator-
  initiated, not attacker-facing — contrast the QL-7 slackd which is hardened against
  exactly this). Worth narrowing the excepts + dropping shell where argv suffices.

- **G-9: Generic (non-BLAS) build correctness.** **Disposition: DONE (this session).**
  Fixed the pthread build break and the naive-conv aliasing corruption; generic now
  bit-matches BLAS end-to-end.

## LOW / WON'T

- **G-10: Legacy `.bin` VAE loader + dead `rope_freqs`.** **DONE** (GROK-1/2, commit
  c49546c — "remove dead rope_freqs + legacy .bin VAE loader").
- **G-11: `fprintf` vs `set_error` inconsistency in iris.c.** **TRACKED** (LOW cleanup).
- **G-12: `AGENT.md` == `CLAUDE.md` "duplication".** **WON'T** — it's an intentional
  symlink (one file, two tool conventions).
- **G-13: Empirical test tolerances (mean_diff=20, 1e-5) could mask drift.** **TRACKED** —
  acknowledged GPU-nondeterminism tradeoff; revisit if a real drift slips through.
- **G-14: Proxy-VAE pending validation.** **GATED** — `plans/proxy-vae-validation-runbook.md`
  + the model-free `--subsample-per-shard` alternative (this session); execution needs idle GPU.

## Summary
Of the recurring findings, the parser-brittleness (G-5), VAE parity (G-6), generic-build
correctness (G-9), and dead code (G-10) classes are **resolved**. The live priorities are
**G-1 (C IP-Adapter — the endgame)**, **G-2 (MLOps tests — in progress)**, and the
GPU/idle-gated **G-3/G-4**. Nothing here blocks the current campaign.
