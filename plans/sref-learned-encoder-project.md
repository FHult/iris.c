# SREF Learned-Encoder / Base-Adapter — Project Plan (the "strong style transfer" project)

Status (2026-07-07): **Phase 0 DONE, Phase 1 STARTED.**
- Phase 0 ✅ — scorecard (`debug/sref_scorecard.py`) + frozen eval set (`debug/sref_eval_set.json`).
  Full band-control baseline (the target to beat): **styleCSD Δ graphic 0.096 / painterly 0.009 /
  semi_real 0.121** — the ceiling, now measured.
- Phase 1 🔨 — `StyleProjector` implemented + smoke-tested (`train/ip_adapter/model.py`; discriminates
  refs at init, no collapse). Trainer path + exact injection point settled in
  `plans/sref-phase1-projector.md`. Remaining: trainer script + smoke overfit + training run + gate.
- Phases 2–4 (DiT LoRA / base-train+transfer / reward) not started; M5-gated.

The larger future project to achieve *strong* reference-style
transfer — including the painterly and semi-realistic references that the shipped training-free
band-control rail cannot handle (BACKLOG **SREF-STYLE-CEILING**). Self-contained; written to be
executed by a fresh session and to survive the hardware gate.

Grounded in the verified deep-research report `plans/sref-architecture-options.md` (this is its
shortlist **#2 — learned in-sequence style tokens**, executed via **#3 — base-model-first
training + transfer**), and in the empirical record: `plans/sref-rope-band-control.md`,
BACKLOG `SREF-CHAMPION-COLLAPSE`, `SREF-STYLE-CEILING`, `SREF-DATA-TEST`, `DATA-SELECTION PRINCIPLE`.

---

## 1. Why this project exists (what's already ruled out)

Two shipped/attempted rails have known ceilings, so a *learned* rail is the remaining path:

- **Band-control (shipped, v5.x)** is training-free and works — but only for **bold/graphic**
  references (woodcut, line-art, flat illustration). It's **subtle** for semi-realistic
  illustration and **fails** on painterly/photographic references (SREF-STYLE-CEILING, reproduced
  visually 2026-07-06 on both distilled and 4B-base, at every shf/slf/strength). It suppresses
  positional copying but can't *impart* a soft/complex look. Ceiling, not a tuning bug.
- **Learned K/V-injection IP-adapter (FAILED, exhaustively)** mode-collapses to a
  reference-inert constant (`to_v_ip` rank ~6; 17 checkpoints, 6 loss experiments, 1 full
  data-side A/B — all failed). **Structural to the injection design** (SREF-CHAMPION-COLLAPSE).
  Literature agrees: no style adapter is validated on any Flux/DiT backbone; the whole ecosystem
  conditions via the **sequence path**.

**Thesis:** put the learned style signal **in the attention sequence** (the only mechanism proven
to discriminate references on Flux), produced by a **small learned encoder→projector**, trained on
the **steerable 4B base** and transferred to distilled. This is USO's validated architecture
(arXiv 2508.18966) adapted to our stack.

---

## 2. The non-negotiable prerequisite — a real style metric (Phase 0)

Every prior effort was flown blind: pixel `copy_corr` and CSD cosine BOTH under-measured the
painterly/semi-realistic failure (SREF-STYLE-CEILING), and `cond_gap` is a search surrogate, not a
style metric (SREF-METRIC-1). **We cannot train what we cannot measure — build the gate first.**

Phase-0 deliverable — a **style-transfer scorecard** that separates the two axes that matter, on a
fixed, diverse eval set (graphic + semi-realistic + painterly + photographic references × unrelated
prompts):

1. **Style adherence** — output vs reference in a *content-invariant* style space. Use CSD **plus**
   at least one independent signal (Gram-matrix distance on a VGG/backbone, or a learned style
   classifier), because CSD alone proved insufficient. Report per-reference-*type*, not just a mean
   (the mean hid the painterly failure).
2. **Composition-leak** — output vs reference *structural* similarity with content held constant:
   e.g. self-similarity / layout descriptors (DINO patch-structure, or an edge/segmentation-map
   IoU) at a fixed prompt — NOT pixel corr (which conflates palette with layout).
3. **Prompt adherence** — CLIP/SigLIP text-image alignment (did we keep the prompt's subject?).

Gate rule for the whole project: a method **wins** only if it raises style-adherence **on painterly
+ semi-realistic references** while composition-leak stays low and prompt-adherence holds. Ship the
scorecard as `debug/sref_scorecard.py` and freeze the eval set (`plans/warmup-campaign-runbook.md`
style). **No training run starts until Phase 0 is green.**

---

## 3. Architecture — learned in-sequence style tokens (USO-style)

```
style ref ─► SigLIP semantic encoder (frozen; we already precompute these)
                 │  (optionally + CSD style vector as an extra input row)
                 ▼
        Hierarchical Projector  (small: linear/attention over SigLIP layers −2/−11/−20)
                 │
                 ▼
        ~192 STYLE TOKENS  ([n_tok, hidden]) ── concatenated IN-SEQUENCE with the image/text
                                                 tokens, RoPE ids = text-like (zeros / L-axis),
                                                 so native joint attention consumes them.
```

Non-negotiable design choices (each is an anti-failure measure with evidence):

- **In-sequence, NOT K/V-injection, NOT modulation/AdaLN.** The sequence path is the only validated
  Flux mechanism; K/V injection collapsed (ours); a 768-d CSD→FiLM rail was refuted as insufficient
  capacity (research "Avoid" list). Style tokens sit in the *proven-discriminative* attention path.
- **Separate encoders for style vs content** (SigLIP/CSD for style; VAE only for the img2img
  content path) — USO's and DEADiff's architectural anti-leakage measure. Do NOT feed VAE latents
  as the style signal.
- **Style tokens carry NO spatial RoPE** (text-like ids) — they contribute *appearance*, not
  layout, so they can't impose the reference's composition. This is the learned analogue of what
  band-control does positionally, and it directly targets the leak the user hit.
- **Compose with band-control + KV-reuse** (already shipped): the two rails stack — learned style
  tokens for *what* look, band-control for *not copying layout*, KV-reuse for speed.

Inference cost: +~192 tokens ≈ +19% sequence at a 1024-token image (vs +100% for a full 512px VAE
reference), and reference-KV reuse across steps amortizes it.

---

## 4. Training strategy — base-first, transfer to distilled (#3)

Distilled 4-step models **commit image structure in the first denoising step** (2503.10637), so a
new conditioning signal has almost nothing to steer — this is *why* both the adapter and band-control
underperform on distilled. Therefore:

- **Train against the 4B BASE** (50-step, CFG, gradual structure formation = a steerable target,
  higher style ceiling — we confirmed base has headroom even if band-control didn't use it).
- **Transfer to distilled** (control modules transfer base↔distilled without retraining, verified
  SDXL result) and additionally implement **hybrid sampling** (step 1 = base, steps 2–4 = distilled)
  in `iris.c` as a diversity/steerability option.

Two training stages (USO's recipe; fits our tooling):

- **Stage 1 — projector only, DiT fully frozen.** Tiny trainable footprint (a projector: a few
  matmuls/attention), consumes exactly the **SigLIP features we already precompute**, comfortably
  fits **32 GB MLX on the M1 Max**. This is the cheap go/no-go: does an in-sequence learned signal
  *discriminate references and add style* without collapsing?
- **Stage 2 — add a DiT LoRA (r128)** trained jointly (we have the LoRA trainer + C loader,
  `iris_lora.c`). Only if Stage 1 passes the scorecard.
- **Stage 3 (optional, expensive) — Style Reward Learning** (a CSD/scorecard-similarity reward,
  USO says it's essential for their *top* scores). Approximate cheaply first: CSD-scored best-of-N
  data or reward-weighted sampling; full RL only if the metric says it's worth it.

**Data** (the DATA-SELECTION PRINCIPLE, already validated as the right *shape*): style-consistent,
content-diverse pairs — our 100k hot pool + `neighbors_look.sqlite` (look-similar / content-different
pairs). Decorrelate the conditioned attribute (style) from content; cover the style range. Precompute
is SigLIP-features-dominated (already cached). **NEVER train from cold storage** (hot SSD only) —
CLAUDE.md invariant.

---

## 5. Phases and go/no-go gates

Each phase ships or dies on the scorecard; kill early — do not tune a collapsed run for weeks
(the SREF-CHAMPION-COLLAPSE lesson).

| Phase | Deliverable | Gate to proceed | HW |
|---|---|---|---|
| **0. Metric** | `debug/sref_scorecard.py` + frozen eval set | Scorecard separates style-adherence, composition-leak, prompt-adherence; reproduces the known band-control ceiling (graphic high, painterly low) | M1 (inference only) |
| **1. Projector (Stage 1)** | SigLIP→projector→192 tokens, frozen DiT, trained on 4B base | On the eval set: **references DISCRIMINATE** (no collapse — cross-ref output corr well below the 0.90 collapse line) AND style-adherence rises on **painterly+semi-real** vs no-reference | M1 32 GB (cached) |
| **2. + DiT LoRA (Stage 2)** | Projector + r128 LoRA on base | Style-adherence beats band-control on painterly/semi-real with leak held; prompt-adherence holds | base training — likely **M5-gated** (HW-M5-2/3) |
| **3. Transfer + hybrid** | Base→distilled transfer; step-1-from-base hybrid in iris.c | Distilled inference retains ≥ most of the base gain; hybrid restores diversity | M5 for training; M1 for inference |
| **4. Reward + hardening** | Optional Style Reward Learning; C inference + parity guard | Top-tier scorecard; C matches Python to the train↔infer tolerance | M5 |

Kill criteria (any → stop, log a negative result like SREF-CHAMPION-COLLAPSE): Stage-1 collapses
(reference-inert) OR style-adherence on painterly/semi-real does not exceed band-control after
Stage 2.

---

## 6. C inference plan (the train↔infer boundary)

The learned rail adds ONE new inference module: the **encoder→projector** producing style tokens,
then reusing the *existing* in-context sequence path (no new attention mechanism).

- **Encoder**: SigLIP must run at inference to produce the same features training used. Today
  SigLIP features are produced out-of-process by the training venv (mlx_vlm); the C engine needs a
  path to the identical features (either a C SigLIP — larger effort, see G-1 — or the existing
  out-of-process producer wired into the web/daemon, already used for the adapter path).
- **Projector**: a few matmuls / a small attention — straightforward C (`iris_transformer_flux.c`
  neighbor), emitting `[192, hidden]` tokens concatenated into the sequence with text-like RoPE ids.
- **Mandatory train↔infer correctness protocol** (CLAUDE.md, the IP-ADAPTER-INFER-1 lesson): golden
  parity fixture (Python golden, C reproduces to corr > 0.999 / max_abs ≤ 1e-3, in `make test`);
  compile under production flags; rebuild the shipped binary (`make mps`); **encoder/preprocess
  parity** (inference must call the identical SigLIP resize/crop/normalize as precompute); source
  shape/dtype audit. This is where the last adapter attempt lost days — do it up front.
- Compose with the shipped band-control + KV-reuse (`iris_metal_set_attention_bias`, the K-side
  RoPE table) — the style tokens are additive to those.

---

## 7. Risks and mitigations

- **Collapse recurrence** (the #1 risk). Mitigation: in-sequence tokens are much harder to ignore
  than a side-channel (they sit in the discriminative path), separate style/content encoders, and
  the discrimination gate runs at **every checkpoint** — pull the plug the moment cross-ref corr
  climbs.
- **We can't measure style** (the #2 risk, already burned us). Mitigation: Phase 0 is a hard gate;
  multi-signal style metric; per-reference-type reporting.
- **Distilled untrainable** (USO trained on 50-step dev). Mitigation: base-first + transfer is the
  plan, not a fallback.
- **Hardware**. Stage 1 fits the M1 Max (cached, projector-only). Base training + Stage-2 LoRA
  realistically want the **M5 Max 128 GB** (HW-M5-2/3) — batch>1 at 512px, ~2-day base run vs 12
  on M1. So: **do Phase 0 + Phase 1 now on M1; Phases 2–4 are M5-gated.**
- **Effort**. Phase 0: ~2–4 days. Phase 1: ~1–2 weeks (trainer wiring exists; projector is new).
  Phases 2–4: multi-week, M5-gated. This is a project, not a sprint.

---

## 8. Explicitly avoid (evidence in the research doc)

Per-block K/V side-channel adapters (structural collapse); naive StyleAligned shared-attention on
DiT (leaks without the RoPE fix — which we already ship); RB-Modulation as a rail (it's test-time
optimization, impossible in C); a bare 768-d CSD→AdaLN/FiLM rail (refuted as insufficient capacity);
CSGO/i2L-scale training (8×A100-class, two orders beyond budget — i2L stays a Phase-4 "later,"
possibly as a **retrieval hybrid**: CSD-nearest trained style-LoRA + interpolation, reusing our CSD
index + LoRA trainer/loader for an "instant LoRA" UX cheaply).

---

## 9. Success criteria

The project succeeds when, on the frozen scorecard eval set, the learned rail delivers **visible
style transfer on painterly and semi-realistic references** (not just graphic) — style-adherence
clearly above band-control and above no-reference, with composition-leak low and prompt-adherence
held — reproducible in the shipped **C** binary (parity-guarded), and stacking with band-control +
KV-reuse. In UX terms: a user uploads *any* reference (a painting, a photo, an anime illustration)
and their prompt comes out convincingly in that look, subject preserved.

## 10. Immediate next actions (do now, M1-only, no M5 needed)

1. **Phase 0**: build `debug/sref_scorecard.py` + freeze the diverse eval set; validate it
   reproduces the band-control ceiling (graphic high / painterly low). ~2–4 days.
2. **Phase 1 Stage-1 prototype**: wire SigLIP-features → projector → in-sequence tokens into the
   MLX trainer against the 4B base (frozen DiT, cached features); run the discrimination gate at
   every checkpoint. Go/no-go on collapse + painterly style-adherence.
3. If Stage 1 is green → schedule Phases 2–4 against the M5 Max (HW-M5-2/3). If it collapses → log
   the negative result and fall back to the retrieval-hybrid instant-LoRA path.
