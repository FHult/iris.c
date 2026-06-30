# SREF — The Journey (Retrospective, 2026-06-17 → 2026-06-30)

How we set out to build a trained IP-Adapter for Midjourney-style `--sref` (apply a reference
image's *style*), spent two weeks discovering the sophisticated thing we built couldn't work and
*why*, and shipped a no-retrain solution that was hiding in plain sight the whole time.

This is the narrative + lessons. The technical records live in:
- `plans/sref-retrain-diagnostic.md` — Step 0 / Step 1A diagnosis + the six fix experiments.
- `plans/sref-architecture-retrain.md` — the architectural review + the shipped content-destruction.
- `BACKLOG.md` — SREF-CHAMPION-COLLAPSE, the diagnostic-first plan, the architectural charter.
- memory `project_sref_state.md` — dated anchors.

---

## Act 1 — The grid, and a fix that felt like the answer (≈ Jun 17–19)
We built an IP-Adapter so the app could do `--sref`. v1 produced a periodic **grid artifact**
(IP-ADAPTER-INFER-1). Root cause: the SigLIP perceiver had a few massive-activation feature
dimensions that dominated the cross-attention dot products, so every learned query attended the
SAME token → a pooled, constant injection → the grid. **Input normalization** (per-dim standardize
+ learned affine before cross-attn) fixed it (`v4.0.0`). The grid vanished; outputs became coherent
cats with reference style. It *looked* solved — and that belief shaped everything after.

## Act 2 — The plateau chase (≈ Jun 22–28)
With coherent output the goal became *quality* — push the style-transfer score (`sref_score` /
injection ratio) higher. The long stretch: V-gates, content-leak penalties, data concentration,
CSD conditioning, the SigLIP+CSD hybrid, a data-quality ladder, injection-timing schedules (DP-7).
Each lever climbed then stalled; the "ceiling" kept getting renamed as confounds surfaced (a
single-shard-training bug inflated metrics): 0.65 → 0.54 → 0.62 → ~0.70. Three *independent* levers
(data, objective, timing) all converged on ~0.70 ⇒ we called it "mechanism-bound," pinned a champion
(clean_concentrate_leak), and shipped it to the web at strength 0.38 (commit 97e8c66).

## Act 3 — The reckoning (Jun 29)
We built a **simple-style eval set** (the curated eval had been 100% warm fine-art paintings). One
test exposed the truth: the champion was **reference-inert** — a near-constant warm-painterly
transform *regardless of the reference* (7 wildly different refs → outputs ≥0.98 correlated). The
famous ~0.70 was a **confound**: every eval painting was warm/painterly, so a constant painterly
output correlated with all of them. The whole quality campaign had optimized "applies a generic
painterly look," not "transfers THIS reference's style."

Simultaneously the web app's style transfer had regressed (good → bad). Root cause: a server restart
set `IRIS_IP_BUNDLE`, silently rerouting references from the clean **in-context** path to the
collapsed adapter. The path that had been giving good results was in-context all along — dismissed in
the code as a "legacy heuristic." The adapter had never really worked. We restored in-context as the
default (`SREF_USE_ADAPTER` opt-in).

## Act 4 — The honest diagnosis (Jun 30)
Rather than guess, we measured everything. **Step 0:** the reference-discrimination gate on all 17
existing checkpoints — every one collapsed, across every cond_mode and recipe, *including the
original grid-fix bundle (0.999)*. The smoking gun: input-norm cured cross-**token** collapse (the
grid) but never cross-**reference** collapse — two orthogonal axes; we'd only ever fixed the first.
**Step 1A.1:** an offline rank audit (`debug/sref_kv_rank_audit.py`) pinned the mechanism — `to_v_ip`
is rank ~6, so the injected V is near-constant across references. Elegantly, K stayed full-rank: the
adapter *looks at* references differently but *injects* the same thing.

## Act 5 — Six honest failures (Jun 30)
We tried to fix it with loss design, documenting each failure (max cross-ref output corr; PASS <0.90):

| experiment | result |
|---|---|
| rank penalty (symptom fix) | 0.926 — best, partial; plateaus |
| output repulsion, aggressive | 0.939 — destabilizes (loss diverges) |
| output repulsion, gentle | 0.945 — inert / overpowered |
| longer rank-only | V-cosine stays 0.965 — rank ≠ decorrelation |
| V-decorrelation | 0.995 — **gamed the proxy** (V-cosine fell to 0.578, output got worse) |
| output repulsion + own-Q context | 0.961 — re-collapsed (content anchor overpowered it) |

All six failed. The lesson: intermediate-tensor proxies get gamed; only the OUTPUT is non-gameable,
and output-repulsion gets overpowered. The model overwhelmingly *prefers* the collapsed solution when
injecting at low scale into a FROZEN DISTILLED base. The collapse is **structural** — which finally
vindicated the "mechanism-bound ceiling" from Act 2, correctly labeled this time.

## Act 6 — The breakthrough hiding in plain sight (Jun 30)
The architectural review reframed it. The clue had been there since Act 3: **in-context discriminates;
the adapter collapses.** The difference is mechanical — in-context concatenates the reference INTO the
transformer's sequence (`[TEXT | TARGET | REFERENCE]`, the frozen base's native channel), while the
adapter injects via a per-block additive K/V **side-channel** the base was never trained for, whose
loss-minimum is a generic style push. First cheap probe: **patch-shuffle** the reference (destroy
composition, keep texture/style), feed it in-context. Result — strong discrimination (cross-ref output
corr **0.158** vs the adapter's 0.93–0.99) and clean **style-only** transfer (line-art, woodcut,
sticker all excellent; decorative patterns like cyberfika partial), **zero training**. Shipped through
the web style path the same day (`content_destroy_png`, commit 4b898b5), end-to-end validated.

---

## The shape of it (the lessons worth keeping)
- **Measurement was the whole game.** A confounded eval set (all-painterly) hid the collapse for two
  weeks; a one-line discrimination test (hold prompt/seed/scale, vary ONLY the reference, correlate
  outputs) exposed it in minutes. The breakthrough wasn't a smarter model — it was finally measuring
  the right thing. *Codified as the mandatory discrimination gate (`debug/sref_ref_discrimination.py`).*
- **The negative results were the work, not the waste.** Each of the six failures ruled out a class of
  fix and tightened the conclusion until "structural" was unavoidable — which is exactly what licensed
  the pivot from loss design to architecture. Without the exhaustive negative space, the cheap
  architectural probe would have read as a lucky guess instead of the obvious next move.
- **The simplest mechanism won.** A trained adapter lost to patch-shuffle + the frozen model's own
  attention. The answer was to work WITH the base's grain (its native in-sequence conditioning), not
  against it (a learned side-channel).
- **The good path was always there.** In-context conditioning — the thing the code called "legacy" and
  a routing flip accidentally bypassed — was the seed of the solution the entire time.
- **Beware the metric that rewards the easy minimum.** The adapter collapsed because nothing in the
  loss punished reference-agnostic output; the eval then *rewarded* the collapsed painterly output.
  Both the training objective and the evaluation were optimizing the wrong thing in the same direction.

## Where it stands
- SHIPPED: style-only `--sref` via content-destruction in the web style path; web default is true
  per-reference style transfer with no model change.
- BANKED: the full diagnosis, the rank-audit + discrimination tooling, the proof that loss-design
  cannot fix the adapter, and a charter for the optional future learned-encoder / base-model levers.
- The trained adapter, the loss saga, and the base-model rewrite are now *optional* quality levers,
  not the critical path.
