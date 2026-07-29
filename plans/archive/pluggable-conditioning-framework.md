# Pluggable Conditioning Framework — Roadmap

Turn iris.c from "an inference engine + a single SREF trainer" into a **framework** for building,
training, validating, and serving pluggable conditioning **plugs** against the frozen Flux base —
LoRAs, in-sequence encoders, and the simple style path — composably. Born from the SREF journey
(`plans/sref-journey-retrospective.md`): the core lesson is to condition WITH the frozen base's
native mechanisms and to VALIDATE every plug with a discrimination/eval gate.

## The conditioning rails (plug types)
1. **Content-destruction style path** — SHIPPED. Instant per-reference style, no training, no weights.
   Patch-shuffle the reference → in-context. (web/server.py `content_destroy_png`.)
2. **Learned in-sequence encoders** — reference → compact style/subject tokens placed IN the sequence
   (the frozen base's native channel, proven to discriminate). Trained ONCE, works on ANY reference.
   Activation-space. Roadmap: style first (sharper than the shuffle, handles graphic styles, saveable
   codes), then subject / face.
3. **LoRAs** — trained per concept, WEIGHT-space low-rank deltas on attention/MLP projections; strong,
   persistent. Today iris_lora.c only LOADS external LoRAs (BFL/Kohya/Diffusers/XLabs) — **training is
   new** and the foundational framework piece.
4. **(frontier) Reference→LoRA hypernetwork** — generate a LoRA delta from a reference for an instant,
   training-free custom "LoRA". Research-tier; the in-sequence encoder is the stepping stone.

## Why these compose (not compete)
LoRA is weight-space; in-sequence encoders are activation-space — **orthogonal axes**. They stack
freely in one generation (e.g. a character LoRA + an instant style reference), each with its own
strength. None of these is "a better LoRA"; the in-sequence rail is the standard *complement* to
LoRAs (instant, no per-concept training) — see the SREF retrospective for the full reasoning.

## The framework pieces
### Training side — "build a plug"
- **Generalize the trainer.** `train/train_ip_adapter.py` already is a frozen-Flux training loop with
  the precompute caches (VAE/Qwen3/SigLIP/CSD), flow-matching loss, cached-mode discipline, EMA, and
  checkpointing. Refactor its IP-adapter specifics into a **pluggable trainable module** interface so
  the same loop trains LoRAs, in-sequence encoders, etc.
- **LoRA training pipeline (Phase 1, foundational).** Train low-rank deltas on the (distilled and/or
  base) Flux transformer with the diffusion objective; export to the **BFL/Kohya format iris_lora.c
  already loads** → train→serve round-trip closes immediately. Reuses the precompute + training infra.
- **In-sequence encoder training (Phase 2).** The learned style-token encoder (the SREF next
  milestone), then subject/face encoders. Builds on the validated in-sequence mechanism.
- **Shared validation.** The discrimination gate (`debug/sref_ref_discrimination.py`) + eval sets as
  the standard "does this plug actually do what it claims" check for ANY plug — the hard-won lesson.

### Serving side — "use a plug"
- **Unified conditioning-plug interface in the C engine.** Load + apply, for both kinds: LoRA
  (weight delta — iris_lora.c is the half that exists) and in-sequence encoder (produce tokens →
  concat into the sequence). One registry, one apply path.
- **Composability.** Stack multiple plugs in one generation; per-plug strength; deterministic order.
- **Surface.** Web/CLI selection + combination of plugs.

## Guardrails (carried from the protocol)
Train↔infer PARITY fixture + prod-flag compile + `make mps` for any C reimpl (AGENT protocol);
cached-mode only (live-encode segfaults MLX — BUGS MLX-1); NEVER train from cold storage (AGENT #6);
promote ONLY on the discrimination/eval gate; `make test` after changes.

## Sequencing
- **Phase 1 — LoRA training pipeline (lowest risk, immediate value).** Well-trodden technique; reuses
  the precompute + frozen-base training infra; exports to a format the engine already loads. Lets the
  owner build custom LoRAs (style/subject/concept) in-house instead of only consuming external ones.
- **Phase 2 — generalize the trainer + learned in-sequence STYLE encoder.** De-risked mechanism;
  beats content-destruction on graphic styles, output cleanliness, and saveable style codes.
- **Phase 3 — subject/face in-sequence encoders + serving composability** (stack LoRA + in-sequence).
- **Phase 4 (frontier) — reference→LoRA hypernetwork.** Instant custom LoRA; speculative.

## Cross-model portability (4B distilled → 4B base 50-step → future 9B base)
Iterate on **4B distilled first** (fastest: 4 steps, no CFG, cheapest to train + eval), then port.
What ports vs what is per-model:
- **Rail 1 (content-destruction): FREE on every variant.** Weight-less (preprocessing + in-context) →
  works on any model with in-context img2img, no retrain. Verify CFG (base) applies the in-context
  reference in BOTH the cond and uncond passes.
- **Rail 2 (in-sequence encoders): the RECIPE ports, the WEIGHTS don't.** Trained against a specific
  transformer's attention. 4B-distilled→4B-base: same hidden dim (3072), base attention differs →
  likely a fine-tune (should transfer BETTER than the side-channel adapter did — native channel;
  empirical). 4B→9B: different hidden dim (3072→4096) + depth → new encoder + retrain.
- **Rail 3 (LoRAs): the PIPELINE ports, the WEIGHTS are per-model.** A LoRA is bound to its model's
  weights; iris_lora.c already loads 4B and 9B by format. Retrain against whichever model you target.
- **UPSIDE (not just compatibility):** base (CFG, 50 steps, more capacity) may make the trained rails
  work BETTER — the distilled base's RIGIDITY was part of why the side-channel adapter collapsed
  (SREF Act 5). The base port is a potential ceiling-RAISER, not just a checkbox.

**DESIGN PRINCIPLE — build the framework MODEL-AWARE from day one so the port is cheap:**
(1) dims from config, never hardcoded (already the project rule) → 3072 vs 4096 is automatic;
(2) a CFG-CAPABLE training path — base needs a null+cond dual-pass + guidance; the current trainer
hardcodes `guidance=None` with no dual-pass (base-model training is NET-NEW code — SREF-BASE-1).
Phase 1's LoRA trainer should support BOTH distilled (no CFG) and base (CFG) from the start.

## Status
- 2026-06-30: roadmap opened. Rail 1 (content-destruction style) SHIPPED. Rails 2–4 = this framework.
  Owner sequencing: build/validate on 4B distilled → check 4B base (50-step) compatibility → future 9B base.
