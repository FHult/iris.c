# LoRA Training Pipeline — Design & Execution (Framework Phase 1)

Goal: train LoRA low-rank deltas on the frozen Flux Klein transformer with the flow-matching
objective, export to the BFL/Kohya safetensors format the C engine (`iris_lora.c`) ALREADY loads, so
the owner can build custom LoRAs (style/subject/concept) in-house. Foundational piece of the
Pluggable Conditioning Framework (`plans/pluggable-conditioning-framework.md`). Build on the existing
frozen-Flux trainer + precompute infra. Model-aware from day one (4B distilled first; 4B base / 9B
later).

## Why this is SIMPLER than the IP-adapter trainer
LoRA modifies the model's ACTUAL linear layers (weight-space), so the training is a STANDARD full
forward with LoRA active + backprop to the LoRA params only. None of the IP-adapter machinery: no
perceiver, no SigLIP/CSD conditioning, no side-channel injection, no `correct_forward_q` Q-collection.
Data is just (VAE latent, text embed) + the flow-matching loss that already exists.

## Architecture
- **LoRALinear wrapper:** wrap a frozen `nn.Linear` (weight W, frozen) with trainable `A [rank, in]`,
  `B [out, rank]`; forward = `W x + (alpha/rank) * B (A x)`. A ~ small normal init, B = zeros
  (delta = 0 at start, like adaLN-zero — stable identity start, gradients still flow).
- **Injection:** replace the target Linears in the frozen Flux transformer with LoRALinear (or attach
  a parallel delta), freeze ALL base params, mark only A/B trainable.
- **Targets (from iris_lora.c's supported set):** double-stream blocks img+txt Q/K/V + output proj;
  single-stream blocks linear1/linear2. Configurable subset (start: attention Q/K/V/out only).
- **Training loop:** reuse train_ip_adapter.py's frozen-Flux loop, precompute caches (VAE + Qwen3 —
  NO SigLIP/CSD), `fused_flow_noise` / `get_schedule_values`, logit-normal timestep sampling, EMA,
  checkpointing, the cached-mode + never-cold discipline. Loss = MSE(v_pred, v_target) (flow matching);
  the LoRA params are the only `value_and_grad` target.
- **Export:** convert MLX LoRA A/B → the BFL/Kohya keys + shapes iris_lora.c reads (lora_A [rank,in],
  lora_B [out,rank], the alpha/scale convention; fused-qkv vs separate handling). Round-trips to the
  engine immediately.
- **Model-aware:** dims from config (3072 vs 4096 auto); a CFG-capable path stub for base (null+cond
  dual-pass; current trainer hardcodes guidance=None) — distilled (no CFG) wired first.

## Train↔infer parity guard (mandatory — AGENT protocol)
LoRA is reimplemented in C (`lora_apply`: `out += scale * B @ (A @ x)`). A golden-fixture parity test:
the MLX LoRALinear forward vs the C `lora_apply` on the same random A/B/x to tight tolerance (corr
>0.999, max_abs ≤ 1e-3); randomise A/B (B not left zero) so the matmul is actually exercised; compile
under production flags; `make mps`. Plus an end-to-end: train a tiny LoRA, export, load in `iris`,
confirm it changes generation as expected.

## Pieces / sequencing (each testable)
1. **LoRALinear + injection + freeze** (MLX): wrap targets, only A/B trainable. Unit test: forward runs,
   gradients flow ONLY to A/B, base unchanged; delta=0 at init reproduces the base forward.
2. **Training step**: full Flux forward with LoRA + flow-matching loss; one smoke step finite/stable.
3. **Trainer**: wire into the existing loop (data = VAE+Qwen3 cache); short smoke run on hot cached data.
4. **Export**: MLX A/B → BFL/Kohya safetensors; parity fixture (MLX vs C lora_apply); load in iris.
5. **End-to-end**: tiny LoRA on a few hundred steps → export → generate with/without → visible effect.

## Guardrails (carried)
parity fixture + prod-flag compile + `make mps`; cached-mode only (live-encode segfaults MLX); never
train from cold storage; `make test`; promote on an eval, not vibes.

## RESOLVED interface (2026-06-30)
- **MLX targets (double blocks, the MVP):** `flux.transformer.transformer_blocks[i].attn.{to_q,to_k,to_v,
  to_out, add_q_proj, add_k_proj, add_v_proj, to_add_out}` — all nn.Linear [3072,3072], 5 blocks × 8 = 40.
  Single blocks (`single_transformer_blocks[i].attn.{to_qkv_mlp_proj,to_out}`, fused) DEFERRED (Diffusers
  export doesn't carry single q/k/v; needs Kohya/BFL + the fused/shared-A handling).
- **Export = DIFFUSERS format (clean 1:1 with the MLX names, separate q/k/v — no fused/shared-A problem):**
  `transformer.transformer_blocks.{i}.attn.{to_q,to_k,to_v}.lora_{A,B}.weight`, `...to_out.0.lora_{A,B}` (note
  the `.0`), `...{add_q_proj,add_k_proj,add_v_proj,to_add_out}.lora_{A,B}`. A=[rank,3072], B=[3072,rank], f32.
  iris_lora.c load_diffusers (iris_lora.c:359-398) reads exactly these for double blocks.
- **No alpha tensor in the loader** (iris_lora.c reads only lora_A/lora_B; scale is the inference --lora-scale).
  → at export, BAKE `alpha/rank` into lora_B so trained strength = `--lora-scale 1.0`; the user scales from there.
- **C apply math (to match in the parity fixture):** `out += scale * (x @ A^T) @ B^T`, A[rank,in], B[out,rank].
- **Freezing:** wrap each target Linear in LoRALinear (frozen base `.linear` + trainable `lora_A/lora_B`);
  `flux.freeze()` then `flux.unfreeze(recurse=True, keys=["lora_A","lora_B"])`; `nn.value_and_grad(flux, ...)`
  then grads ONLY the LoRA. (MLX freeze/unfreeze accept `keys`.)
- **Forward for LoRA = the model's STANDARD forward** (NOT `_flux_forward_no_ip`, which is the IP-adapter's
  Q-collection trick). Run all blocks normally with LoRA active; flow-matching MSE on the velocity.

## Piece-2 forward interface (resolved 2026-06-30)
`flux.transformer.__call__(hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids,
guidance=None)` → velocity prediction (mflux Flux2Transformer; signature at
.../flux2_transformer/transformer.py:67-74). `guidance` is the CFG hook (base model;
distilled passes None). Reference call/setup: mflux `.../variants/txt2img/flux2_klein.py` `predict()`
(patchify → img_ids/txt_ids → transformer(...)). The LoRA training step:
1. patchify the cached VAE latent → hidden_states; build img_ids/txt_ids (REUSE the IP-adapter trainer's
   input prep in `_flux_forward_no_ip` — it already patchifies/embeds/positions; just call the STANDARD
   transformer forward instead of the Q-collection variant — this is the train↔infer correctness boundary,
   verify the prep matches inference).
2. sample timestep (logit-normal), `fused_flow_noise` → (noisy, v_target); set hidden_states = noisy.
3. `pred = flux.transformer(noisy_patched, text_embeds, t, img_ids, txt_ids, guidance)`; loss = MSE(pred, v_target).
4. `nn.value_and_grad(flux, step)` → LoRA-only grads (piece 1 froze the base); AdamW; EMA; checkpoint.
Smoke: warmstart nothing, a few hundred steps on hot cached data (VAE+Qwen3), loss finite/decreasing.

## Status (2026-06-30)
- ✅ Piece 1 DONE + tested (commit bbdf2aa): LoRALinear + double-block inject/freeze; 7 tests pass.
- NEXT: piece 2 (training step + flow-matching loss + smoke) — careful pass on the input-prep boundary;
  then piece 4 export (Diffusers) + the parity fixture (MLX LoRALinear vs C lora_apply), then end-to-end.

## Log
- 2026-06-30: design opened; Explore mapped the MLX Flux structure + export format; interface RESOLVED
  (Diffusers double-block export, bake-scale-into-B, freeze/unfreeze keys); piece 1 built + tested;
  piece-2 forward interface mapped.
