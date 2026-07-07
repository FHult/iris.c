# SREF Phase 1 — Learned style projector (build note)

Executable build note for Phase 1 of `plans/sref-learned-encoder-project.md`: train a
`StyleProjector` (SigLIP → in-sequence style tokens) against the frozen 4B base, DiT frozen, and
gate it with `debug/sref_scorecard.py`. Written so a fresh session can execute it.

## Status
- ✅ **StyleProjector implemented + smoke-tested** — `train/ip_adapter/model.py` (class `StyleProjector`).
  102 M params; at init it DISCRIMINATES references (cross-ref token corr 0.41) and produces distinct
  tokens (off-diag 0.03) — i.e. no collapse at init. Maps SigLIP `[B,729,1152]` → `[B,192,3072]`.
- ✅ **Phase-0 gate ready** — scorecard + frozen eval set; band-control baseline to beat:
  **styleCSD Δ graphic 0.096 / painterly 0.009 / semi_real 0.121** (`debug/sref_eval_set.json`).
- ✅ **Mechanics smoke PASSED** — `train/lora/smoke_style.py`: overfit a batch with the projector as
  the only trainable module, DiT frozen. Loss 0.75→0.30, projector query_tokens updated (max |Δ| 0.126)
  → gradients reach the projector THROUGH the frozen DiT (the whole premise works), no NaN. The
  in-sequence forward (`flux_forward_style` = `flux_forward_lora` + style concat + extended text RoPE) is
  correct. Note: gnorm spiked at step 0 (~400) then went tiny — tune projector init / LR / warmup for
  the real run. ⬜ Real data loader + training run on the 4B base + scorecard gate — the work below.

## Architecture decision (settled)
Train with **full backprop through the frozen DiT** — NOT the IP-adapter trainer's Q-caching trick
(`_flux_forward_no_ip`/`_pred_from_embeds`), which only works for a K/V side-channel and CANNOT train
an in-sequence signal (style tokens sit inside the frozen attention, so gradients must flow through
the DiT to the projector). Base it on the **LoRA trainer's** gradient-checkpointed forward
(`train/lora/train_step.py:flux_forward_lora`, TRAIN-7: ~19 GB at 512px, fits M1 32 GB). The
`StyleProjector` is the ONLY trainable module (flux `.freeze()`d).

## The exact injection point (train/lora/train_step.py `flux_forward_lora`)
Current forward (paraphrased):
```
hidden_states = tr.x_embedder(hidden_states)              # image tokens [B, seq_img, 3072]
encoder_hidden_states = tr.context_embedder(text_embeds)  # text tokens  [B, seq_txt, 3072]   (line 77)
rope = (concat([txt_rope[0], img_rope[0]]), concat([txt_rope[1], img_rope[1]]))                # (line 81)
# double blocks over (encoder_hidden_states, hidden_states) ; then
hidden_states = concat([encoder_hidden_states, hidden_states], axis=1)                          # (line 89)
# single blocks ; strip text: hidden_states[:, encoder_hidden_states.shape[1]:, ...]            # (line 95)
```
**Change (TEXT | STYLE | IMAGE):**
1. `style_tokens = style_projector(siglip_feats)`  → `[B, 192, 3072]` (already hidden_dim; does NOT go
   through context_embedder).
2. `encoder_hidden_states = mx.concatenate([encoder_hidden_states, style_tokens], axis=1)` (after line 77).
3. Build `style_ids` text-like (non-spatial): `[0,0,0, arange(seq_txt, seq_txt+192)]`, compute its rope,
   and extend the text rope: `txt_rope = concat([txt_rope, style_rope])` before line 81.
4. Nothing else changes — the line-95 strip uses `encoder_hidden_states.shape[1]` (now seq_txt+192), so
   it correctly drops text+style and keeps the image tokens for `norm_out`/`proj_out`. Double blocks see
   style in the context stream; single blocks see it in the full sequence — the proven in-sequence path.

Style tokens carry text-like RoPE (no H/W) → they contribute appearance, not layout (the learned
analogue of band-control's positional suppression; directly targets composition leak).

## Trainer (adapt `train/lora/train_lora.py`)
- Load `Flux2Klein(model_path=<4B BASE dir>, quantize=None)`; `flux.freeze()`. **Train on BASE**
  (steerable; distilled commits structure in step 1). Model dir by path (no config flag).
- Construct `StyleProjector(...)`; it is the only module with grads. `opt = AdamW(lr 1e-4, wd 0.01)`;
  `nn.value_and_grad` over the projector (flux frozen). Grad-clip 1.0; warmup+cosine.
- Loss: flow-matching MSE (reuse `ip_adapter.loss.fused_flow_noise` + `get_schedule_values`; same as
  `lora_loss`). Target = velocity from `fused_flow_noise(latents, noise, alpha, sigma)`.
- Data: reuse `ip_adapter/dataset.py` — it already yields SigLIP features `[B,729,1152]`, VAE latents,
  Qwen3 text embeds per batch; pin `data.bucket:[512,512]` (VAE-Q1 / precompute contract). Use the
  look-paired, content-decorrelated set (`neighbors_look.sqlite`, DATA-SELECTION PRINCIPLE).
- **Null-style dropout** (~10%): zero `siglip_feats` on some samples so the projector learns a real
  conditional (mirrors the adapter's `use_null_image`) and CFG works at inference.
- Memory: `mlx_memory_pct ~0.6`; gradient checkpointing already in `flux_forward_lora`. NEVER train from
  cold storage (hot SSD only).

## Gate (run at every checkpoint — kill early)
Export the projector, run inference with the style tokens spliced in (C or a Python inference harness),
then `debug/sref_scorecard.py`. GO/NO-GO:
- **Collapse check** (hard kill): cross-reference output corr must stay well below 0.90 (the adapter
  collapsed at ≥0.90). If it climbs, stop — same failure, log it.
- **Win check**: styleCSD Δ on **painterly + semi_real** must exceed the band-control baseline
  (painterly 0.009) with leak Δ held near band-control's and prompt adherence intact. Painterly is the
  whole point — a graphic-only win is not a win.

## Smoke test first (before any long run)
Overfit ~8 fixed samples for a few hundred steps (like `train/lora/smoke_overfit.py`): loss must fall,
and a checkpoint must (a) not NaN, (b) DISCRIMINATE the 8 references (distinct outputs). Only then start
a real run. If Stage 1 collapses or can't beat band-control on painterly after Stage 2 (add DiT LoRA
r128), log the negative result and fall back to the retrieval-hybrid instant-LoRA path.

## Inference (later, after a passing checkpoint)
C side: `StyleProjector` is a cross-attention + FFN (small) emitting `[192, hidden]` tokens concatenated
into the existing in-context sequence with text-like RoPE ids — mirror the exact train forward. SigLIP
features via the existing out-of-process producer (as the adapter path does) until a C SigLIP lands.
Mandatory train↔infer parity fixture (corr > 0.999) before shipping.
