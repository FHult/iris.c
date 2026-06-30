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

## Open questions (Explore agent answering 2026-06-30)
- [ ] Exact MLX Flux Linear attribute paths + dims (the modules to wrap).
- [ ] How the existing forward runs + how the base is frozen (value_and_grad scope vs explicit freeze).
- [ ] MLX freeze/trainable mechanics for a param subset.
- [ ] Exact BFL/Kohya export keys + shapes + alpha convention from iris_lora.c.

## Log
- 2026-06-30: design opened; Explore mapping the MLX Flux structure + export format.
