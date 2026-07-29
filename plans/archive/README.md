# plans/archive — historical plan trail

These plans are **done, shipped, superseded, or closed-negative**. They are kept as the durable
research/engineering trail (per the "log learnings" rule), not as open work. Moved here 2026-07-29
to declutter `plans/` so the live roadmap is unambiguous.

**Live plans stay in `plans/`:**
- `foundation-quality-roadmap.md` — the current major thrust (higher-res + curated adapter training
  on the frozen 4B base).
- `pipeline-v2-architecture.md` (implemented), `pipeline-v3-architecture.md` (future research),
  `pipeline-mlops-backlog.md`, `warmup-campaign-runbook.md` — pipeline design + operations.
- `precomp2-proxy-vae-design.md`, `precomp4-aspect-bucketing.md` — open/gated precompute work.
- `shard-data-composition.md` — current corpus reference.

**Canonical guides (not plans) that supersede much of this trail:**
- `docs/sref.md` — the single source of truth for style reference; the `sref-*` docs here are its trail.
- `CLAUDE.md`, `BACKLOG.md`, `BUGS.md`, `train/DISPATCH.md`.

## What's here, by theme

- **SREF (all closed — the workstream shipped v5.3.0 + Style Library and is done):**
  `sref-*.md` — band-control, retrieval-hybrid (Style Library), joint-backbone (learned adapter),
  the learned-encoder direction (killed), specialist/router (3 negatives), architecture/retrain
  exploration, execution planning, retrospective, and the release runbook.
- **IP-adapter / LoRA:** `ip-adapter-training.md`, `c-ip-adapter.md`, `lora-support.md`,
  `lora-training-pipeline.md`, `pluggable-conditioning-framework.md`.
- **Pipeline (V1 / early designs, audits, superseded):** `flywheel-1-design.md`,
  `flywheel-precompute-architecture.md`, `hot-precompute-cache.md`, `pipeline-audit2-backlog.md`,
  `pipeline-efficiency-audit.md`, `telemetry-audit.md`, `roadmap.md`.
- **Proxy-VAE / quality-loop migrations (done):** `proxy-vae-v3.19-migration.md`,
  `proxy-vae-validation-runbook.md`, `quality-loop-v3.21-migration.md`.
- **Training perf / forward-split (done/superseded):** `training-perf-backlog.md`,
  `training-split-forward-plan.md`.
- **Data-build scaffolding (mostly done; superseded by the foundation roadmap):**
  `cold-full-shard-build-foundation-runs.md`.
- **Reviews / audits / misc (point-in-time):** `code-audit-2026-05.md`, `code-review-2026-06-10.md`,
  `grok-review-triage.md`, `bug-check-2026-06-11.md`, `17-prompt-templates.md`,
  `slack-command-daemon.md`.
