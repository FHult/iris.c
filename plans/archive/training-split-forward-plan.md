# Plan: Split Flux Forward (no-grad) from Adapter Loss (differentiable)

**Status: Superseded** — TRAIN-6 implemented Option C (`correct_forward_q` / `_flux_forward_with_ip_collect_q`) instead. This plan is retained for historical reference.

**Goal:** Reduce MLX graph tracing cost so step 1 goes from ~25 min (single-core
Python) to seconds, and `@mx.compile` cache makes steps 2+ GPU-bound.

**Status:** DRAFT — requires validation against timing results from TP-005 before
implementing. If step 2+ are already fast with `@mx.compile`, this may not be
worth the complexity.

---

## 1. The Problem (stated precisely)

`nn.value_and_grad(adapter, loss_fn)` asks MLX to differentiate `loss_fn` with
respect to `adapter` parameters. MLX builds this autodiff graph by tracing through
the Python `loss_fn` body in full — including all the Flux transformer ops — to
find which ops have a path to an `adapter` parameter. This trace is single-threaded
Python and proportional to the number of ops in the function.

For Klein 4B: 5 double blocks + 20 single blocks + patchify/embed/unpack = ~100+
named ops per block × 25 blocks = thousands of MLX graph nodes. Most nodes have
zero gradient with respect to adapter parameters (all the frozen Flux weights),
but MLX still has to visit them during tracing to determine this.

`@mx.compile` (currently applied) caches the traced graph after step 1, so steps
2+ are fast. But step 1 still pays the full cost.

The question is: **how long is step 1 in steady-state, and does it matter?**
At 50K steps, even a 25-min step 1 is 0.03% of training time — ignorable.
The concern is only if it happens repeatedly (e.g., after crashes/restarts).

---

## 2. The Proposed Fix

Split `_flux_forward_with_ip` into two phases:

**Phase 1 — no-grad Flux forward (runs outside `loss_fn`):**
- Runs the full Flux forward pass
- Captures, for each block, the Q vectors needed for IP attention
- Captures the `hidden_states` tensor at each injection point
- Captures the final `pred` (velocity prediction)
- All outputs are detached via `mx.stop_gradient()` before being passed to phase 2

**Phase 2 — adapter loss (runs inside `loss_fn`, differentiable):**
- `loss_fn` receives the captured Qs and hidden_states as stopped-gradient inputs
- Computes k_ip, v_ip from adapter (differentiable)
- Computes IP attention: `ip_out = SDPA(Q_stopped, k_ip, v_ip)`
- Adds IP contributions to hidden_states (stopped) at each block
- Computes final pred from the modified hidden_states
- Returns MSE loss

The MLX graph for autodiff is now: adapter → k_ip/v_ip → ip_out → modified_pred → loss.
No Flux ops are in this graph.

---

## 3. Critical Analysis of This Approach

### 3.1 Is the gradient correct?

**Yes — with one important caveat.**

In the original forward:
```
hidden_i = block_i(hidden_{i-1} + ip_scale_{i-1} * ip_out_{i-1})
```
The hidden state entering block `i` contains the IP contribution from block `i-1`.
This means the Flux forward actually *depends* on the IP output at each step.
The gradient of the loss with respect to k_ip[0] includes paths through blocks 1..24.

In the split approach, phase 1 runs Flux **without** any IP contributions:
```
hidden_i_noip = block_i(hidden_{i-1}_noip)   # pure Flux, no IP
```
Then phase 2 adds IP contributions to the **no-IP** hidden states.

**This is a different forward pass.** The gradients we compute are correct for
the split model, but they are an approximation to the gradients of the interleaved
model. Specifically:
- In the interleaved model: IP contribution at block `i` influences hidden states
  at blocks `i+1..N`, creating gradient paths through those blocks back to k_ip[i]
- In the split model: IP contributions don't influence subsequent Flux blocks at all

**Is this approximation acceptable for training?**
In practice, yes — this is exactly the training approach used by the original
InstantX IP-Adapter for Diffusers. Their reference implementation uses
`torch.no_grad()` for the base model forward and only differentiates through
the IP attention. The approximation has been empirically validated to produce
good IP-Adapter quality. The gradient signal for k_ip[i] comes from the loss
sensitivity to ip_out[i], which is well-captured even without the cross-block paths.

**Self-critique:** I am asserting this based on the InstantX implementation
pattern, not from a formal analysis. I have not verified that their code does
exactly this split. Should verify before committing.

### 3.2 Does stop_gradient on hidden_states break single-block gradients?

No. Within phase 2, the hidden_states at each injection point are stopped-gradient
inputs, but the IP contributions are differentiable:
```python
# Phase 2, block i:
h_ip = h_stopped[i] + ip_scale[i] * ip_out[i]  # ip_out[i] is differentiable
h_ip_stopped = mx.stop_gradient(h_ip)           # cut for next block
# ...
h_ip = h_stopped[j] + ip_scale[j] * ip_out[j]  # independent, still differentiable
```
Each block's k_ip/v_ip gradient is computed independently from `loss → ip_out[i] → k_ip[i]`.
The gradient for k_ip[i] does NOT depend on what happens in later blocks, because
we stop_gradient after each addition. This is the approximation described in 3.1.

### 3.3 What about the final layers after all IP injections?

In the original forward, after all IP injections:
```
hidden_states = norm_out(hidden_states)
pred = proj_out(hidden_states)
loss = mse(pred, target)
```
`norm_out` and `proj_out` are frozen Flux layers. In the split approach, `pred`
comes from phase 1 (the pure Flux forward with no IP). Phase 2 never feeds back
into `norm_out`/`proj_out`.

**This means the loss function changes.** In the interleaved model:
```
loss = mse(unpack(proj_out(norm_out(h_N + ip_out_N))), target)
```
In the split model:
```
loss = mse(unpack(proj_out(norm_out(h_N_noip))) + delta, target)
```
where `delta` is approximately the IP contribution to the final prediction.

Actually this is a **significant problem.** If we use `pred` from phase 1 (no-IP),
then the MSE loss is computed against the pure Flux prediction, not the IP-modulated
one. The adapter would never actually see its own output in the loss — it would
always be optimising against a moving target computed without its contributions.
The gradients would be structurally wrong, not just approximate.

**Correction to my earlier thinking:** This is NOT simply "run Flux with no_grad
and differentiate only through the adapter." The final prediction must include
the IP contributions. Phase 2 needs to compute a full modified pred that flows
through `norm_out` and `proj_out`.

But `norm_out` and `proj_out` are frozen Flux layers — if we include them in
phase 2, we're back to tracing through Flux ops in the autodiff graph.

**Two options:**

**Option A — stop_gradient at the block outputs only, keep final layers:**
Phase 2 receives stopped hidden_states at each injection point but does run
`norm_out` + `proj_out` in the differentiable graph. These are 2 linear ops,
not 25 blocks — tiny trace cost.

**Option B — absorb final layers into phase 1, approximate delta in phase 2:**
Treat the IP contribution as an additive delta to the final `pred`:
```
pred_with_ip ≈ pred_noip + sum_i(proj_out(norm_out(ip_contribution_i)))
```
This requires approximating the linearity of norm+proj, which is only valid
if the IP contributions are small relative to the hidden states. This is a
strong assumption and likely incorrect early in training.

**Option A is clearly correct; Option B is an approximation with unknown validity.**

### 3.4 Revised approach: stop_gradient at block outputs, keep final 2 layers

Phase 1: run all 25 blocks, capturing stopped-gradient hidden_states at each
injection point.
Phase 2 (inside loss_fn):
- Receives stopped h_0..h_24 (pre-IP block outputs)
- For each block: `h_with_ip[i] = h_stopped[i] + ip_scale[i] * SDPA(Q[i], k_ip[i], v_ip[i])`
- After all blocks: pass h_with_ip[-1] through `norm_out` + `proj_out` (frozen, but in graph)
- Compute MSE loss

Autodiff graph: adapter → k_ip → ip_out → h_with_ip → norm_out(frozen) → proj_out(frozen) → loss
- 25 SDPA ops + 2 linear ops = small
- No 5×double block + 20×single block ops in the graph

**Is this correct?**
The gradient of loss with respect to k_ip[i] is:
```
dL/dk[i] = dL/dpred × dpred/dh[-1] × dh[-1]/d(ip_out[i]) × d(ip_out[i])/dk[i]
```
`dpred/dh[-1]` passes through `proj_out` and `norm_out` — these are frozen but
their Jacobians are still needed for backprop through them. MLX will compute
these Jacobians but NOT update the parameters (they're frozen). This is correct
and unavoidable — we need the Jacobian to backprop through these layers.

**Self-critique:** We're still tracing `norm_out` and `proj_out` in the autodiff
graph. Are these expensive to trace? For Klein 4B:
- `norm_out`: RMSNorm(inner_dim=3072) — ~1 op
- `proj_out`: Linear(inner_dim=3072, out_dim=128) — 1 matmul

Negligible. This is fine.

### 3.5 What does phase 2 actually need from phase 1?

Per-block (5 double + 20 single = 25 total):
- `q_ip[i]`: [B, heads, seq_img, head_dim] — already stop_gradient'd in current code
- `h_noip[i]`: [B, seq_img|seq_total, inner_dim] — the hidden state *before* IP addition

Plus:
- `h_final_noip`: [B, seq_img, inner_dim] — final image hidden state from phase 1
  (post all blocks, pre norm_out) — needed to feed into norm_out in phase 2

Wait — actually h_final_noip IS h_noip[24] (after last block). But we also need
the text portion stripped (seq_img only, not seq_total). Need to track seq_txt.

**For double blocks (i = 0..4):**
- `h_noip[i]` = `hidden_states` after block i, before IP addition: [B, seq_img, inner_dim]
- `q_ip[i]` = Q computed from pre-block h_{i-1}: [B, heads, seq_img, head_dim]

**For single blocks (j = 0..19, block_ip_idx = 5+j):**
- At merge: `hidden_states = concat([encoder_hidden_states, hidden_states], axis=1)` → [B, seq_txt+seq_img, inner_dim]
- `h_noip[5+j]` = full hidden_states after single block j: [B, seq_txt+seq_img, inner_dim]
- `q_ip[5+j]` = Q from image portion only: [B, heads, seq_img, head_dim]

Phase 2 also needs `seq_txt` to extract image portion from single-block hidden_states.

### 3.6 Memory analysis

Currently (interleaved): MLX retains all intermediate hidden_states for backward
pass. At 512px batch_size=2:
- seq_img = 32×32 = 1024 tokens
- inner_dim = 3072
- Each hidden_states tensor: 2 × 1024 × 3072 × 2 bytes (bf16) = ~12MB
- 25 blocks × 12MB = ~300MB of activation tensors retained for backward

In split approach: these tensors are stopped-gradient numpy arrays (or mx.arrays
with no grad). Memory is similar but they're not in the gradient graph — MLX
won't retain the full computation graph for them. Slight memory saving but not
the primary benefit.

**The primary benefit is trace time, not memory.**

### 3.7 Interaction with `@mx.compile`

With `@mx.compile` already applied, the trace happens once. If step 1 is 25 min
and steps 2-50000 are e.g. 5s, the split-forward plan saves 25 min total but
adds ~100 lines of code complexity.

**This is the key question to answer before implementing.** If `@mx.compile`
caches the graph and steps 2+ are fast, the split plan is not worth doing.
The TP-005 timing run (currently executing) will answer this.

### 3.8 Risk of regression

The approximation in 3.1 (IP contributions don't feed into subsequent Flux blocks
in phase 1) is the standard IP-Adapter training approach and empirically validated.
Risk: LOW.

The revised approach (option A, keep norm_out+proj_out in phase 2): mathematically
identical gradient for those 2 layers, approximated for the 25 block paths. Risk: LOW.

The main implementation risk: getting the tensor shapes right for 25 injection
points, especially the transition from double-stream to single-stream (seq changes).
Risk: MEDIUM — requires careful index bookkeeping and testing.

---

## 4. Concrete Implementation Plan

### 4.1 New function: `_flux_forward_collect_qs_and_states`

```python
def _flux_forward_collect_qs_and_states(
    flux,
    noisy_latents: mx.array,   # [B, 32, H/8, W/8]
    text_embeds: mx.array,      # [B, seq_txt, 7680]
    t_int: mx.array,
) -> dict:
    """
    Run Flux forward with no IP injection; collect Q vectors and pre-IP
    hidden states at each injection point. All outputs are stop_gradient'd.

    Returns dict with:
        qs:      list[25] of [B, heads, seq_img, head_dim]
        hs:      list[25] of [B, seq_img or seq_txt+seq_img, inner_dim]
        h_final: [B, seq_img, inner_dim]  (post-blocks, pre norm_out)
        seq_txt: int
        seq_img: int
        temb:    [B, inner_dim]  (stop_gradient'd)
    """
```

This function runs steps 1-11 of the current `_flux_forward_with_ip` with no IP
addition, capturing h_before at each block and the Q computation.

All returned tensors are wrapped in `mx.stop_gradient()`.

**Must call `mx.eval()` on all captured tensors** before they are passed to
`loss_fn` — otherwise MLX may defer computation of Flux ops into the gradient
graph. This is a subtle but critical point.

### 4.2 Modified `loss_fn`

```python
def loss_fn(text_embeds, latents, siglip_feats,
            use_null_image, use_null_text,
            flux_state: dict):      # <-- new: pre-computed, stopped
    ip_embeds = adapter.get_image_embeds(siglip_feats)
    ...
    k_ip_all, v_ip_all = adapter.get_kv_all(ip_embeds)

    # For each block: compute ip_out = SDPA(q[i], k_ip[i], v_ip[i])
    # Add to stopped h[i]
    # After all blocks: run through norm_out + proj_out
    # Compute MSE loss
```

### 4.3 Modified training loop

```python
# OUTSIDE loss_fn (no grad):
flux_state = _flux_forward_collect_qs_and_states(flux, noisy, text_embeds, t_int)
mx.eval(flux_state)  # CRITICAL: force eval before entering grad scope

# INSIDE loss_fn (via nn.value_and_grad):
loss_val, grads = loss_and_grad(..., flux_state=flux_state)
```

**Problem with `@mx.compile` interaction:** `@mx.compile` traces based on
input signatures. `flux_state` is a dict of arrays whose values change every
step. If passed as a positional argument, `@mx.compile` will see different
arrays each step and may re-trace. Need to verify MLX compile handles dict
inputs correctly — MLX compile traces on shapes and dtypes, not values, so
this should be fine as long as shapes are consistent (they are: fixed 512px).

### 4.4 Removal of current code

- `_flux_forward_with_ip`: remove entirely, replace with the two new functions
- `ckpt_double`, `ckpt_single`: remove (already None, but clean up)
- The `h_before` / Q recomputation inside the block loops: remove (now done in phase 1)

---

## 5. What we don't know yet

1. **Are steps 2+ fast with current `@mx.compile`?** The TP-005 timing run will
   answer this. If yes, this plan may not be worth implementing.

2. **Does MLX `@mx.compile` correctly handle dict arguments without re-tracing?**
   Need to verify or restructure flux_state as positional array arguments.

3. **Does `mx.eval(flux_state)` fully detach tensors from the gradient tape?**
   MLX's autodiff model: `stop_gradient` creates a sentinel op in the graph that
   returns zero gradient. But if `flux_state` tensors are computed lazily and
   their evaluation is deferred into the `loss_fn` scope, MLX might trace through
   them anyway. The `mx.eval()` forces materialization, which should prevent this.
   But this needs to be verified empirically (run a grad trace and check no Flux
   ops appear in the gradient graph).

4. **Is the gradient approximation (3.1) empirically equivalent for training
   quality?** Known to work in PyTorch IP-Adapter training, but not yet verified
   for this MLX + Flux Klein 4B implementation specifically.

---

## 6. Decision criteria

**Implement if:** step 2+ time with current code is >10s/step (meaning the
`@mx.compile` cache still incurs significant Python overhead per step).

**Defer if:** step 2+ time is <5s/step. The one-time 25-min step-1 cost is
then <0.1% of a 50K-step run — not worth the complexity.

**Abandon if:** the timing data from TP-005 reveals that the bottleneck is
somewhere else entirely (data loading, eval/sync, etc.).

---

## 7. Open questions for review

1. Does `mx.eval()` on phase 1 outputs fully prevent MLX from including phase 1
   ops in the autodiff graph, or do we additionally need `mx.stop_gradient()`?
   (Answer: both are needed — `eval` forces materialization, `stop_gradient`
   creates the sentinel op. Using only `eval` without `stop_gradient` could still
   include phase 1 ops in the trace if MLX re-encounters them.)

2. The single-stream blocks operate on concatenated [TEXT | IMAGE] sequence.
   After IP injection we strip text tokens: `hidden_states[:, seq_txt:, :]`.
   In phase 2, when we add IP contributions to the final single-block output
   and then pass to norm_out, are we passing [seq_img] or [seq_txt + seq_img]?
   `norm_out` and `proj_out` in the current code receive `hidden_states[:, seq_txt:, :]`
   (image-only). Phase 2 must do the same strip before calling norm_out.

3. Phase 1 captures `h_noip[i]` **after** each block but **before** IP addition.
   In the original code for double blocks, the block outputs are used as the Q
   source for the *next* block too (via h_before = hidden_states). In phase 2,
   when we reconstruct the modified hidden_states, we add IP to h_noip[i] and
   then... what? We don't feed it into subsequent blocks (they're in phase 1).
   The h_noip[i] for the next block was already computed by phase 1 WITHOUT the
   IP contribution from block i-1. This is the 3.1 approximation — confirmed.
