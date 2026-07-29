# Code Audit — May 2026

Five parallel agents audited the end-to-end codebase: training core, signal alignment, Python↔C
crosswalk, pipeline scripts, and C/Metal inference. This document consolidates all findings with
severity ratings and implementation plans.

Severity scale:
- **CRITICAL** — loss of training-correct behaviour, unrecoverable data loss, or feature
  completely broken at runtime.
- **HIGH** — significant quality regression, or would cause confusing silent failure.
- **MEDIUM** — correctness risk or efficiency issue affecting a real code path.
- **LOW** — dead code, naming confusion, minor edge cases.

---

## CRITICAL

### INFER-1: C inference not implemented — `--sref` rejected at runtime — **CONFIRMED REAL (2026-05-22)**

| Field | Value |
|-------|-------|
| Files | `iris_ip_adapter.h`, `iris.c`, `main.c` |
| Agent | 3 (Python↔C crosswalk) |

`iris_ip_adapter.h` is a stub. The C binary has no IP-adapter forward pass. Any attempt to
use `--sref` hits an argument-parsing rejection or silent no-op. All ablation-trained adapter
weights are unreachable from the C binary.

**Implementation plan:**
1. Port `IPAdapterKlein.inject()` logic from `train/ip_adapter/model.py:180-207` into
   `iris_ip_adapter.c` (new file).
2. At each double-block and single-block in `iris_transformer_flux.c`, call
   `iris_ip_adapter_inject_double()` / `iris_ip_adapter_inject_single()`.
3. Use per-block injection (not end-sum accumulation used in
   `test_ip_adapter_inference.py`) to match training path.
4. Wire `--sref` → `iris_generate()` via `iris_ip_adapter_load()`.
5. Match `num_heads=perceiver_heads` (16 by default) not Flux transformer heads (24).

---

### ~~SIGNAL-1: Qwen3 layer extraction off-by-one~~ — **FALSE POSITIVE (verified 2026-05-22)**

| Field | Value |
|-------|-------|
| Files | `train/scripts/precompute_all.py:320`, `train/train_ip_adapter.py:2146` |
| Agent | 1 (signal alignment) |

**The two indexing schemes are equivalent — do not change either.**

Precompute (`_qwen3_hidden_states`) uses a manual loop `for i, layer in enumerate(layers)` and
collects the output **after** layer `i`. `target=(8, 17, 26)` therefore captures transformer
layers 8, 17, 26 (0-indexed).

Training uses mflux's `hidden_states_list` which is initialised with `embed_tokens` output at
index 0, then each layer's output appended in order. `hidden_states_list[9]` = output of
transformer layer 8, `[18]` = layer 17, `[27]` = layer 26. So `hidden_state_layers=(9, 18, 27)`
extracts exactly the same three tensors.

The mflux reference implementation also defaults to `(9, 18, 27)`. Changing to `(8, 17, 26)`
would be a real regression — it would extract layers 7, 16, 25 instead.

---

### SIGNAL-2: SigLIP receives bucket-resolution image, not 384×384 — **FIXED (2026-05-22)**

| Field | Value |
|-------|-------|
| Files | `train/scripts/precompute_all.py:194`, `train/train_ip_adapter.py:1409` |
| Agent | 1 (signal alignment) |

**Confirmed real bug, but never fires in current setup.**

Precompute (`_preprocess_siglip`) explicitly resizes to 384×384 before encoding. Training live
path (`train_ip_adapter.py:1409`) calls `siglip(images)` where `images` is `[B, 3, bH, bW]`
at bucket resolution after `augment_mlx` — no resize. The model docstring at line 2086
explicitly states "Input: [B, 3, 384, 384]".

Mitigating factor: the live SigLIP path (`elif siglip is not None`) only runs when
`siglip_cache_dir` is `None`. Current training always runs with 100% precomputed SigLIP
coverage, so this path is never reached. Fix remains important for any training run without
a precomputed cache.

**Fix (applied):** Added `_resize_images_for_siglip()` helper in `train_ip_adapter.py`
(near `_load_siglip`). Performs CHW→HWC, denorm→uint8, PIL LANCZOS resize to 384,
renorm→bf16. Call site changed to `siglip(_resize_images_for_siglip(images))`.

---

### EXPORT-1: `perceiver_heads` mismatch — trained as 16, exported as 24 — **FIXED (2026-05-22)**

| Field | Value |
|-------|-------|
| Files | `train/export/export_adapter.py:267`, `train/ip_adapter/model.py:86` |
| Agent | 1 (signal alignment), 3 (Python↔C crosswalk) |

**Confirmed real bug.**

Training config `stage1_512px.yaml` has `perceiver_heads: 16`. This controls the
`PerceiverResampler`'s internal cross-attention (how 128 query tokens attend to 729 SigLIP
tokens).

`_infer_dims()` at line 267 cannot recover `perceiver_heads` from weight shapes — all
projection matrices are `[3072, 3072]` regardless of head count. It falls back to
`num_heads = 24 if hidden_dim == 3072 else ...` (the Flux transformer head count).
`export()` then writes `"perceiver_heads": dims["num_heads"]` = 24 when `--perceiver-heads`
is not passed.

The test script (`test_ip_adapter_inference.py:142`) uses this metadata field to reconstruct
`IPAdapterKlein(perceiver_heads=24)`. Since `nn.MultiHeadAttention` weights are `[D, D]`
regardless of num_heads, the load succeeds silently, but the Perceiver now computes
cross-attention with head_dim=128 (24-head split) instead of the trained head_dim=192
(16-head split) — silent quality regression in any Python inference test.

**Fix (applied):** `save_checkpoint_async` already writes the full `config` dict (including
`config.adapter.perceiver_heads`) into the sidecar JSON. `export()` in
`train/export/export_adapter.py` now reads the sidecar (`step_NNNNNNN.json`) to extract
`perceiver_heads`; raises an explicit error if neither the sidecar nor `--perceiver-heads`
provides a value. The Flux head count fallback is removed.

---

### CROSS-1: `inject()` hardcodes `head_dim = hidden_dim // 24` — **MISLEADING CHARACTERISATION (2026-05-22)**

| Field | Value |
|-------|-------|
| Files | `train/ip_adapter/model.py:199`, `train/ip_adapter/model.py:86, 93` |
| Agent | 3 (Python↔C crosswalk) |

**The specific bug described is wrong; a related latent bug exists for 9B.**

`inject()` performs cross-attention between Flux Q (from the Flux transformer block,
`[B, flux_heads, img_seq, flux_head_dim]`) and IP K/V. The K/V tensors are `[B, 128, 3072]`
reshaped to `[B, flux_heads, 128, flux_head_dim]`. For Flux 4B: flux_heads=24,
flux_head_dim=128. The hardcoded `self.hidden_dim // 24 = 3072 // 24 = 128` is correct.

`perceiver_heads=16` controls a **separate** attention inside `PerceiverResampler.__call__`
(how 128 query tokens attend to 729 SigLIP tokens). These two head counts are independent.
The inject cross-attention always uses Flux model head dimensions, not Perceiver heads.

The real training forward pass (`_flux_forward_with_ip`) uses `H_s` and `Hd_s` from the
actual Flux block rather than a hardcoded constant — consistent with the analysis above.

**Actual latent bug:** The hardcoded 24 would be wrong for Flux 9B (32 heads,
head_dim = 4096 // 32 = 128, but `4096 // 24 = 170`). Currently 9B adapter is unsupported
so there is no impact.

**Fix (applied 2026-05-22):** Replace the hardcoded constant with the Flux head count read from
the Q tensor shape, correct for any Flux variant:
```python
_, _, _, head_dim = img_q.shape   # derive from actual Q, not hardcoded
```
Do NOT change to `self.hidden_dim // self.perceiver_heads` — that would be wrong.

Note: `inject()` is only called from `tests/test_model.py`, not from the main training loop.
`_flux_forward_with_ip` derives H and Hd from `block.attn.heads` / `block.attn.dim_head`
directly — the training path was never affected by this bug.

---

## HIGH

### A-1: `correct_forward_q` + `n_grad_steps > 1` = stale Q vectors — **FIXED (2026-05-22)**

| Field | Value |
|-------|-------|
| Files | `train/train_ip_adapter.py:1477-1494` |
| Agent | 2 (training core) |

When `correct_forward_q=True` and `n_grad_steps_per_fwd > 1`, Q vectors are computed once
before the inner gradient loop using **pre-update** adapter parameters. Steps 2..N reuse
stale Q — the corrected-Q guarantee is violated for every step after the first.

**Fix (applied in prior session):** Guard added that forces `n_grad_steps=1` when
`correct_forward_q=True`.

---

### SIGNAL-3: Null image conditioning applied after Perceiver, should be before — **FIXED (2026-05-22)**

| Field | Value |
|-------|-------|
| Files | `train/train_ip_adapter.py:957-958` |
| Agent | 1 (signal alignment) |

When `use_null_image=True`, the code was zeroing the Perceiver *output* (`ip_embeds`).
Correct null conditioning zeros the SigLIP features *before* Perceiver processing, so the
Perceiver learns to produce null embeddings from null input rather than having its output
overwritten. The previous approach meant the Perceiver never saw null inputs during training.

**Fix (applied):** Both `loss_fn` and `loss_fn_with_ip` now apply
`siglip_feats = mx.where(use_null_image, mx.zeros_like(siglip_feats), siglip_feats)` before
calling `adapter.get_image_embeds(siglip_feats)`. Post-hoc `zero_embeds mx.where` removed.

---

### SIGNAL-4: Text null conditioning inconsistency — cached path zeros embeddings, live path encodes empty caption — **FIXED (2026-05-22)**

| Field | Value |
|-------|-------|
| Files | `train/train_ip_adapter.py:1384-1390` |
| Agent | 1 (signal alignment) |

When text dropout fired: cached path set `text_embeds = mx.zeros_like(text_embeds)`; live
encoding path passed `[""]` to Qwen3, which produces the chat-template embedding of an empty
prompt — not zeros. These are different representations.

**Fix (applied):** Live path now always encodes the real caption, then zeros afterward:
```python
text_embeds = _encode_text(text_encoder, captions)
if null_text:
    text_embeds = mx.zeros_like(text_embeds)
```

---

### CROSS-2: Export script infers wrong `perceiver_heads` default if flag omitted

| Field | Value |
|-------|-------|
| Files | `train/export/export_adapter.py:267, 603` |
| Agent | 3 (Python↔C crosswalk) |

If `--perceiver-heads` is not passed, line 267 infers `24 if hidden_dim == 3072 else ...`.
Any export without the explicit flag embeds wrong metadata. `test_ip_adapter_inference.py:142`
silently falls back to `num_heads=24` masking the bug further.

**Fix:** See EXPORT-1 — store `perceiver_heads` in checkpoint and read from there. Fail
loudly if neither source provides it.

---

### CROSS-3: Perceiver `nn.MultiHeadAttention` bias not set explicitly — **FIXED (2026-05-22)**

| Field | Value |
|-------|-------|
| Files | `train/ip_adapter/model.py:45-49` |
| Agent | 3 (Python↔C crosswalk) |

MLX default for `nn.MultiHeadAttention` is `bias=False`. The C header `iris_ip_adapter.h`
lines 71-74 lists only weight pointers, confirming no bias expected. If someone accidentally
passes `bias=True`, trained weights include biases that the C loader ignores, causing silent
corruption.

**Fix (applied):** `bias=False` is now explicit on the `cross_attn` constructor in
`PerceiverResampler.__init__`.

---

### CROSS-4: Injection semantics mismatch — per-block (training) vs end-sum (test script) — **CONFIRMED REAL, DEFERRED**

| Field | Value |
|-------|-------|
| Files | `train/train_ip_adapter.py:2635, 2690`, `train/scripts/test_ip_adapter_inference.py:358-368` |
| Agent | 3 (Python↔C crosswalk) |

Training (`_flux_forward_with_ip`) injects IP contributions per-block inline, influencing Q
for subsequent blocks. `test_ip_adapter_inference.py:_ip_forward` collects all IP outputs
then adds them once after all blocks complete. These produce different numerical results — the
test script does not replicate training semantics.

**Status:** Confirmed real discrepancy. Deferred — the training path is canonical. The test
script needs updating to use per-block injection before any C inference validation against it.

---

### CROSS-6: `perceiver_heads` missing from `iris_ip_adapter_t` struct — **DEFERRED (stub)**

| Field | Value |
|-------|-------|
| Files | `iris_ip_adapter.h:50-122` |
| Agent | 3 (Python↔C crosswalk) |

The C struct has `hidden_dim` and `num_image_tokens` but no `perceiver_heads` or
`perceiver_head_dim`. C inference cannot validate configuration or compute correct head
dimensions.

**Status:** Deferred — `iris_ip_adapter.h` is currently a stub (INFER-1 not yet
implemented). Will be addressed as part of the full C inference implementation.

---

### C-1: `PerceiverResampler` missing residual connection — **DEFERRED (needs migration plan)**

| Field | Value |
|-------|-------|
| Files | `train/ip_adapter/model.py:52-59` |
| Agent | 2 (training core) |

`__call__` applies `cross_attn(q, kv, kv)` then `LayerNorm`, with no `q + out` residual.
Standard Perceiver and IP-Adapter reference implementations include the residual. Without it,
learned query tokens receive no gradient path that preserves their prior.

**Status:** Deferred. Adding the residual changes the forward semantics for any trained
checkpoint. The fix is architecturally correct but requires:
1. Verifying whether the warmstart source (InstantX) used residual or not.
2. Deciding whether to retrain from scratch or continue from current step 200 checkpoint.

If adopting the residual, restart training from scratch (step 200 represents negligible
sunk cost). Do NOT apply this mid-run without a plan.

---

### EXPORT-2: Perceiver biases trained but not exported (if bias=True was ever set)

| Field | Value |
|-------|-------|
| Files | `train/scripts/export_ip_adapter.py`, `train/ip_adapter/model.py` |
| Agent | 1 (signal alignment), 3 (Python↔C crosswalk) |

If `bias=True` is ever set on the Perceiver cross-attention (see CROSS-3), those bias terms
would be trained but not exported, shifting activations at inference relative to training.
The CROSS-3 fix (explicit `bias=False`) prevents this, but add a post-export verification
step regardless: count exported parameters and compare against
`sum(p.size for p in adapter.parameters())`.

---

### F-1: Preflight check rejects directory warmstart path

| Field | Value |
|-------|-------|
| Files | `train/train_ip_adapter.py:543-547` |
| Agent | 2 (training core) |

`_ws_ok = os.path.isfile(_warmstart)` fails for InstantX directory warmstart, causing a
spurious pre-flight failure and `sys.exit` before any model loading.

**Fix:**
```python
_ws_ok = os.path.isfile(_warmstart) or os.path.isdir(_warmstart)
```

---

### G-1: EMA holds live reference to `adapter.parameters()` — **FIXED (2026-05-22)**

| Field | Value |
|-------|-------|
| Files | `train/train_ip_adapter.py:882` |
| Agent | 2 (training core) |

`ema_params = adapter.parameters()` returned the live parameter dict reference. Fragile under
any MLX version that mutates parameter arrays in-place.

**Fix (applied):** Added `import mlx.utils as mx_utils` to MLX imports block. Init path now:
```python
ema_params = mx_utils.tree_map(lambda x: mx.array(x), adapter.parameters())
```

---

### ADALN-1: `adaln_norm_bf16` Metal shader uses numerically unstable variance formula — **FIXED (2026-05-22)**

| Field | Value |
|-------|-------|
| Files | `iris_shaders.metal:1138-1140` |
| Agent | 5 (C/Metal inference) |

The kernel computes variance as `E[x²] - E[x]²`. All arithmetic is done in float32 (bf16
inputs are upcasted), so catastrophic cancellation is unlikely for typical activations but
possible for highly biased hidden states. If variance goes negative, `rsqrt(var)` → NaN.

**Fix (applied):** Added `max(var, 0.0f)` clamp before `rsqrt` in all three norm kernels:
`adaln_norm`, `adaln_norm_bf16`, and `group_norm_f32`. Prevents NaN without two-pass cost.
Full two-pass Welford (the original recommendation) would be more robust but doubles memory
bandwidth — deferred unless NaN is observed in practice.

---

### PIPE-2 / CM-7/8: Log file staleness causes phantom sentinel reads; threadgroup bounds silently exit

| Field | Value |
|-------|-------|
| Files | `pipeline_lib.py:382-397` (log staleness); `iris_shaders.metal:1913, 629` (threadgroup) |
| Agent | 4 (pipeline scripts), 5 (C/Metal inference) |

**Log staleness:** When a script reuses the same log file path across runs, `last_exit_code()`
reads the EXIT_CODE from a *previous* run, causing false error/success states. Build_shards
and dedupe_filter both write to fixed-name log files. **Real — deferred.**

**CM-7/8 threadgroup bounds — FALSE POSITIVE (verified 2026-05-22):** The `if (seq > 512) return;`
guard inside `causal_attention_fused` / `causal_attention_fused_bf16` is redundant but harmless.
The C-level wrappers (`iris_gpu_causal_attention_f32` line ~6539 and
`iris_gpu_causal_attention_bf16` line ~5501) both check `if (seq > 512)` and fall through to
the MPSGraph path before dispatching the kernel. The kernel never receives seq > 512, so
output is never left uninitialised.

**Fix (log staleness):** Implement log rotation or include a run-unique token in the log
header that `last_exit_code()` validates matches.

---

### PIPE-3: Bare except blocks silently swallow critical errors — **PARTIALLY FIXED (2026-05-22)**

| Field | Value |
|-------|-------|
| Files | `pipeline_lib.py:326-327`, `pipeline_status.py:97,126,156`, `download_convert.py:279-283`, `build_shards.py:120,162` |
| Agent | 4 (pipeline scripts) |

Multiple bare `except:` or `except Exception:` blocks that log nothing and continue, hiding
I/O errors, data corruption, and subprocess failures.

**Fix (applied):**
- `pipeline_lib.py:dispatch_issue` rotation: `except Exception: pass` → prints traceback to stderr.
- `pipeline_status.py:_read_val_metrics`: `except Exception` → `except (FileNotFoundError, json.JSONDecodeError)`.

**Remaining:** `pipeline_status.py:97` (`_log_tail` `except OSError`) is fine — already specific.
`build_shards.py:120,162` already logs via `print(..., file=sys.stderr)` — defensible. `download_convert.py` inner excepts already log and set `error_event`.

---

### PIPE-5: FAISS index written without `fsync` — power-loss corruption — **PARTIALLY FIXED (2026-05-22)**

| Field | Value |
|-------|-------|
| Files | `dedupe_filter.py:202-207`, `clip_dedup.py:427`, `clean_wds_pool.py:165-171` |
| Agent | 4 (pipeline scripts) |

`faiss.write_index()` followed by `os.replace()` with no `fsync()`. Power loss between
file close and kernel flush leaves a truncated index. On restart, FAISS fails silently.

**Fix (applied to `dedupe_filter.py`):** `_flush_faiss_index()` now opens the tmp file
after write and calls `os.fsync(f.fileno())` before `os.replace()`.

**Remaining:** `clip_dedup.py:427` and `clean_wds_pool.py:165-171` write directly to the
final path (no atomic rename pattern at all) — these need the full tmp+fsync+replace
treatment in a follow-up.

---

### PIPE-7: Dedup filter writes done sentinel before verifying output integrity — **FALSE POSITIVE (verified 2026-05-22)**

| Field | Value |
|-------|-------|
| Files | `dedupe_filter.py:247-293` |
| Agent | 4 (pipeline scripts) |

**Verified not a bug.** The sentinel is written at line 293, which is OUTSIDE the
`try...except` block at lines 246-290. Any crash or exception during `_quality_filter_tar`,
`dedup_wds_tar`, or `_atomic_copy` propagates via `raise` at line 290, which skips line 293.
The sentinel is never written on failure. The audit was correct about the general pattern
but wrong about this specific code.

Legitimate enhancement (not a bug): there is no positive integrity check (file size > 0,
tar opens cleanly) before the sentinel. Adding one would improve robustness but is low
priority since FAISS dedup always produces a valid tar or raises.

---

### CM-1: `adaln_norm_bf16` unstable variance (same formula in `adaln_norm` and `group_norm_f32`)

| Field | Value |
|-------|-------|
| Files | `iris_shaders.metal:1138-1140`, `iris_shaders.metal:164-166`, `iris_shaders.metal:2164-2166` |
| Agent | 5 (C/Metal inference) |

The same `E[x²] - E[x]²` formula appears in three norm kernels. The f32 version is less
catastrophic but still unstable for large hidden dimensions.

**Fix:** Apply two-pass Welford algorithm to all three kernels (see ADALN-1).

---

### SIGLIP-1: SigLIP receives bucket-resolution image, not 384×384

See SIGNAL-2 above. Same issue, both agents identified it independently.

---

## MEDIUM

### SIGNAL-6: VAE latent dtype inconsistency — precomputed int8→f16, live path f32→bf16

| Field | Value |
|-------|-------|
| Files | `train/scripts/precompute_all.py:515`, `train/ip_adapter/dataset.py:222`, `train/train_ip_adapter.py:1365` |
| Agent | 1 (signal alignment) |

Precomputed VAE latents are stored as int8+f16 scale, loaded as f16, then cast to bf16. Live
encoding produces f32 latents cast to bf16. The int8 quantisation step is lossy relative to
the live path.

**Fix:** Store precomputed VAE latents as f32 (no quantisation), matching the live-encode path.

---

### SIGNAL-9: SigLIP cache miss forces `null_image=True` even when dropout was not selected

| Field | Value |
|-------|-------|
| Files | `train/train_ip_adapter.py:1410-1422` |
| Agent | 1 (signal alignment) |

If SigLIP cache misses and no live SigLIP model is loaded, the code forces `null_image=True`.
If that step was not an intended dropout step, this corrupts the training signal for that batch
and inflates the apparent dropout rate.

**Fix:** Track intent separately: `null_selected = random.random() < dropout_prob`. Only force
null on cache miss if not already null; log a warning when a non-dropout step is forced null.

---

### F-2: `from_pretrained_warmstart` silently ignores `num_double_blocks`

| Field | Value |
|-------|-------|
| Files | `train/ip_adapter/model.py:231-255`, `train/train_ip_adapter.py:782-789` |
| Agent | 2 (training core) |

`from_pretrained_warmstart()` does not accept or pass `num_double_blocks` to the constructor,
silently using the default (5) regardless of config. Causes `freeze_double_stream_scales` to
freeze the wrong block count.

**Fix:** Add `num_double_blocks=acfg.get("num_double_blocks", 5)` to the call site and
add it to `from_pretrained_warmstart`'s `**kwargs`.

---

### F-3: Text encoder layer extraction uses 1-indexed indices in training

See SIGNAL-1 — the `hidden_state_layers=(9, 18, 27)` issue. The fix is documented there.

---

### F-4: LR cosine-decay resume creates steeper decay than original

| Field | Value |
|-------|-------|
| Files | `train/train_ip_adapter.py:113-120` |
| Agent | 2 (training core) |

On resume mid-cosine-decay phase, `optim.cosine_decay(current_lr, decay_steps=remaining,
end=eta_min)` treats `current_lr` as a new cosine peak. At 50% through decay, resumed schedule
decays at ~2× the correct rate.

**Fix:** Replace with closed-form LR as function of global step:
```python
def lr_at_step(step):
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    t = step - warmup_steps
    return eta_min + 0.5 * (lr_max - eta_min) * (1 + math.cos(math.pi * t / decay_steps))
```

---

### H-1: `n_grad_steps_per_fwd > 1` is N optimizer steps on same data, not accumulation

| Field | Value |
|-------|-------|
| Files | `train/train_ip_adapter.py:1460-1494` |
| Agent | 2 (training core) |

Same noise, target, and `flux_state` reused for all N inner optimizer steps. Name implies
gradient accumulation semantics; actual effect is effective LR multiplied by N per image.

**Fix:** Document clearly in config schema. If true accumulation is desired, a separate code
path accumulating gradients across N different samples is needed.

---

### B-3: `gram_style_loss` magnitude varies ~1300× across bucket resolutions

| Field | Value |
|-------|-------|
| Files | `train/ip_adapter/loss.py:23-46` |
| Agent | 2 (training core) |

`gram_matrix` normalises by `C*H*W`, so MSE between Gram matrices scales as `1/(C*H*W)²`.
Across BUCKETS from 64×64 to 768×768, style loss varies by factor ~1296. Fixed
`style_loss_weight` produces wildly different effective gradient contributions per bucket.

**Fix:** Normalise Gram matrix by `H*W` only:
```python
return mx.matmul(f, f.transpose(0, 2, 1)) / (H * W)
```

---

### D-2: `correct_forward_q` single-stream Q extracted before injection (approximation undocumented)

| Field | Value |
|-------|-------|
| Files | `train/train_ip_adapter.py:2464, 2480-2488` |
| Agent | 2 (training core) |

In `_flux_forward_with_ip_collect_q`, single-stream block Q is extracted **before** IP is
injected into `hidden_states`. Only double-stream blocks correctly inject IP before Q
extraction.

**Fix (minimal):** Add comment documenting the approximation. Full fix: restructure
single-stream path to inject from the previous block before computing Q for the current block.

---

### J-1: Inner grad loop can overshoot `_end_step` by up to `n_grad_steps - 1`

| Field | Value |
|-------|-------|
| Files | `train/train_ip_adapter.py:1494-1496, 1555` |
| Agent | 2 (training core) |

Break check is at the **top** of the inner loop; `step += 1` is at the bottom. The last
iteration runs a full optimizer step and increments `step` to `_end_step + 1`.

**Fix:** Move `step += 1` before the break check:
```python
step += 1
if step >= _end_step:
    break
```

---

### H-2: `_do_style` evaluated once before inner loop, stale for steps 2..N

| Field | Value |
|-------|-------|
| Files | `train/train_ip_adapter.py:1500` |
| Agent | 2 (training core) |

`_do_style = ... step % _style_every == 0` computed before the inner loop. For
`n_grad_steps > 1`, inner steps 2..N always use the initial value, which may be wrong.

**Fix:** Move `_do_style` computation inside the inner loop, before `compiled_step`.

---

### CROSS-5: Q normalization assumption undocumented for C inference

| Field | Value |
|-------|-------|
| Files | `train/train_ip_adapter.py:2621`, `iris_ip_adapter.h` |
| Agent | 3 (Python↔C crosswalk) |

Training Q vectors are RMS-normalised by `block.attn.norm_q` before being stored in
`flux_state["qs"]`. `inject()` must NOT re-normalise. This assumption is correct in the
current Python code but is not documented in the C header.

**Fix:** Add comment to `iris_ip_adapter.h`:
```c
/* Q is passed in pre-normalised (norm_q applied by Flux block).
 * Do NOT apply additional RMS normalisation in ip_inject(). */
```

---

### CROSS-8: Export key namespace mismatch requires fragile key map

| Field | Value |
|-------|-------|
| Files | `train/export/export_adapter.py:74-85`, `train/scripts/test_ip_adapter_inference.py:67-78` |
| Agent | 3 (Python↔C crosswalk) |

Export uses `perceiver.*` key names; training model uses `image_proj.*`. A key map translates
between them. Any typo in the map silently loads wrong weights.

**Fix:** Use consistent naming throughout (`image_proj.*`) to eliminate the map, or add a
round-trip test that exports and reloads and compares outputs numerically.

---

### PIPE-1: Inconsistent `dispatch_queue.jsonl` parsing across 4 scripts

| Field | Value |
|-------|-------|
| Files | `pipeline_status.py:216-240`, `pipeline_doctor.py:280-300`, `pipeline_ctl.py:162-177`, `orchestrator.py` |
| Agent | 4 (pipeline scripts) |

Three different parsing implementations: `pipeline_status.py` uses last-write-wins,
`pipeline_ctl.py` does simpler parsing without deduplication, `pipeline_doctor.py` has
different integration. A closed issue in one script's view may appear open in another's.

**Fix:** Extract `read_dispatch_issues(path, include_resolved=False)` into `pipeline_lib.py`
and import in all four scripts.

---

### PIPE-4: Race in `download_convert.py` — exit code may be 0 despite download error — **FIXED (2026-05-22)**

| Field | Value |
|-------|-------|
| Files | `download_convert.py:344-348`, `download_convert.py:405-409` |
| Agent | 4 (pipeline scripts) |

If producer or consumer thread crashed with an unexpected exception (e.g., `ready.touch()`
failing, `_pool_link_or_copy` failing) outside the per-download inner `try/except`,
`error_event` would not be set and exit code 0 was returned despite the failure.

**Fix:** Added outer `except Exception: error_event.set(); raise` in both `producer()` and
`consumer()`, between the inner body and the `finally` clause. Any uncaught exception in
either thread now sets `error_event` before propagating, ensuring the main thread's
`if error_event.is_set(): raise RuntimeError(...)` fires correctly.

---

### PIPE-8: Heartbeat path collisions — `download_convert` and `dedupe_filter` race — **FIXED (2026-05-22)**

| Field | Value |
|-------|-------|
| Files | `pipeline_lib.py:240` |
| Agent | 4 (pipeline scripts) |

`write_heartbeat` used a fixed `.json.tmp` suffix for the atomic rename temp file. When
`heartbeat_loop` thread and the post-join main thread both call `write_heartbeat` for the
same process+chunk within the same process, they wrote to the same `.json.tmp` path — last
write wins on rename, potentially corrupting or losing a heartbeat write.

**Fix:** Temp file now uses PID: `p.with_name(p.stem + f".{os.getpid()}.tmp")`. Each
process gets its own tmp path, eliminating the collision. Cross-process races on the final
`.json` path are safe: `os.rename` is atomic on POSIX.

---

### PIPE-10: No retry on FAISS index read failures during concurrent access

| Field | Value |
|-------|-------|
| Files | `clip_dedup.py:396-402`, `clean_wds_pool.py:158-163` |
| Agent | 4 (pipeline scripts) |

If another process is writing the index, `faiss.read_index()` fails with no retry.
Dedup exits returning `(0, 0, 0)`.

**Fix:** Implement exponential backoff retry with timeout on FAISS read.

---

### PIPE-15: `last_exit_code()` returning `None` treated as success — **FALSE POSITIVE (verified 2026-05-22)**

| Field | Value |
|-------|-------|
| Files | `orchestrator.py:476-489` |
| Agent | 4 (pipeline scripts) |

`code = last_exit_code(log); if code == 0:` — in Python, `None == 0` is `False`, so `None`
exits to the `else` branch which logs the failure and calls `mark_error`. The orchestrator
already treats `None` as failure, not success. No fix required.

---

### CM-6: Flash attention OOM returns silently, leaving output uninitialised — **FIXED (2026-05-22)**

| Field | Value |
|-------|-------|
| Files | `iris_kernels.c:800-806, 939-979` |
| Agent | 5 (C/Metal inference) |

On malloc failure inside `flash_attention_head_tiled`, the early return at line 805 left
the output buffer with uninitialised content. The `memset(out, 0, ...)` was placed after
the check.

**Fix (applied):** Added `memset(out, 0, seq_q * head_dim * sizeof(float))` in the OOM
path before the early return, ensuring callers always receive zeroed output on failure.
The `break` paths in the outer head loop (lines ~941, ~978) return CPU-computed attention
for the heads processed before OOM; the zeroed-output approach is consistent with those.

---

### CM-12: Integer overflow in convolution tiling index cast — **FALSE POSITIVE (verified 2026-05-22)**

| Field | Value |
|-------|-------|
| Files | `iris_kernels.c:381` |
| Agent | 5 (C/Metal inference) |

`tile_rows = (int)(max_col_size / row_size)`. `max_col_size` is hardcoded to
`256 * 1024 * 1024 = 268,435,456` (~256 MB). The division result is
`max_col_size / row_size ≤ max_col_size = 268,435,456 < INT_MAX (2,147,483,648)`.
Overflow is impossible regardless of input size. No fix required.

---

### CM-5: `emb_cache_store` `strdup` failure not propagated — **CONFIRMED, DEFERRED**

| Field | Value |
|-------|-------|
| Files | `embcache.c:197-201` |
| Agent | 5 (C/Metal inference) |

On `strdup` failure, lines 201-202 already detect the failure (`!g_cache[target].prompt`)
and call `clear_slot(target)` — no slot corruption. The function returns void so callers
cannot detect failure. The consequence is a cache miss on the next lookup (prompt re-encoded).

**Status:** Real gap but low impact (graceful degradation, no corruption). Deferred —
changing return type requires updating all call sites; not worth the churn given the benign
failure mode.

---

### PIPE-11: Heartbeat loop in `download_convert.py` reads `cur_tgz[0]` without lock — **FIXED (2026-05-22)**

| Field | Value |
|-------|-------|
| Files | `download_convert.py:271`, all `cur_tgz[0]` sites |
| Agent | 4 (pipeline scripts) |

Heartbeat loop reads `cur_tgz[0]` (written by producer thread) without a lock. CPython GIL
makes individual `cur_tgz[0] = x` assignments atomic in practice, but this relies on
undefined behaviour from the memory model perspective.

**Fix:** Added `cur_tgz_lock = threading.Lock()` alongside `cur_tgz = [None]`. All five
producer write sites and the single heartbeat_loop read site now use `with cur_tgz_lock:`.

---

### PIPE-12: CLIP dedup threshold not validated against config

| Field | Value |
|-------|-------|
| Files | `clip_dedup.py:53`, `dedupe_filter.py:36` |
| Agent | 4 (pipeline scripts) |

`DUP_THRESHOLD = 0.95` in clip_dedup.py; different thresholds may be passed via CLI.
No validation that thresholds match across a chunk's dedup workflow.

**Fix:** Assert that CLI threshold matches the config value; log the threshold used in
every dedup run.

---

### PIPE-13: `build_shards.py` partial `.tar.tmp` not cleaned up on crash

| Field | Value |
|-------|-------|
| Files | `build_shards.py:244-256, 495-502` |
| Agent | 4 (pipeline scripts) |

If a worker crashes mid-write, the `.tar.tmp` stays on disk. Startup cleanup (lines 495-502)
deletes `.tar.tmp` files older than 5 min but misses recent ones. Shard plan becomes
corrupted.

**Fix:** Track tmp file ownership via process ID in filename; delete only files from dead PIDs.

---

## LOW

### E-1: Three Flux forward functions share ~120 lines of identical setup code

| Field | Value |
|-------|-------|
| Files | `train/train_ip_adapter.py:2155-2705` |
| Agent | 2 (training core) |

`_flux_forward_no_ip`, `_flux_forward_with_ip_collect_q`, `_flux_forward_with_ip` each
contain identical patchify, position-ID construction, timestep embedding, input projection,
RoPE, and modulation unpacking (~40 lines × 3 copies). A divergence already exists at
line ~2269 vs ~2642 (mod_gate_s unpacking).

**Fix:** Extract `_flux_prepare_inputs(flux, noisy_latents, text_embeds, t_int)` returning
the shared setup tensors. All three functions call it and branch only on IP-injection.

---

### E-2/E-3: `compiled_step` post-grad logic and `_nested_update` duplicated

| Field | Value |
|-------|-------|
| Files | `train/train_ip_adapter.py:1028-1034, 1079-1085, 450-470` |
| Agent | 2 (training core) |

Freeze-scale zeroing, `grad_clip_norm`, and `optimizer.update` are copy-pasted between
the two step functions. `_nested_update` and `_flat_to_nested` duplicate dict-building code.

**Fix:** Extract `_apply_grads(grads, adapter, optimizer)`. Make `_nested_update` call
`_flat_to_nested` directly.

---

### G-2: `save_ema()` in `ema.py` is dead code using bulk (non-streaming) save

| Field | Value |
|-------|-------|
| Files | `train/ip_adapter/ema.py:33-36` |
| Agent | 2 (training core) |

Never called from `train_ip_adapter.py`. Uses `mx.save_safetensors` (stages all tensors
~2 GB at once). Training script uses `_save_safetensors_streaming` instead.

**Fix:** Remove `save_ema()`.

---

### B-2: `flow_matching_loss()` in `loss.py` is dead code

| Field | Value |
|-------|-------|
| Files | `train/ip_adapter/loss.py:188-200` |
| Agent | 2 (training core) |

Recomputes target internally via `fused_flow_noise` but the training loop precomputes target
separately. Any caller would waste one `fused_flow_noise` dispatch.

**Fix:** Remove `flow_matching_loss()`.

---

### J-2 through J-5: Minor training loop issues

| Field | Value |
|-------|-------|
| Files | `train/train_ip_adapter.py:1995, 393, 278, 1988` |
| Agent | 2 (training core) |

- **J-2:** Final checkpoint lineage records stale `loss_smooth` (not updated if final step
  is not a log interval multiple).
- **J-3:** `sidecar = f.replace(".safetensors", ".json")` — use `os.path.splitext` instead.
- **J-4:** `save_checkpoint_async` is synchronous — rename to `save_checkpoint`.
- **J-5:** `del target` raises `NameError` if loop ran zero iterations — initialise
  `target = None` before loop.

---

### C-3: `inject()` hardcodes `num_heads=24` (same root as CROSS-1)

See CROSS-1. Resolved by the same fix.

---

### QUALITY-1: `patch_shuffle` and `cross_ref` compose to near-random conditioning

| Field | Value |
|-------|-------|
| Files | `train/train_ip_adapter.py:1436-1451` |
| Agent | 2 (training core) |

`patch_shuffle` runs first and stores shuffled features in `_cross_ref_buffer`. A cross-ref
step then uses shuffled features from a different image — both spatial order and image
identity randomised simultaneously. May be too destructive to provide a useful style signal.

**Fix:** Apply `patch_shuffle` only to the current step's features, not to the
`_cross_ref_buffer`.

---

### SIGNAL-7: SigLIP normalisation constants hardcoded in precompute, not shared

| Field | Value |
|-------|-------|
| Files | `train/scripts/precompute_all.py:159-160` |
| Agent | 1 (signal alignment) |

`_SIGLIP_MEAN = [0.5, 0.5, 0.5]` and `_SIGLIP_STD = [0.5, 0.5, 0.5]` hardcoded in
precompute. Not imported from a shared constants module, so any drift between precompute
and live encoding normalisation will be silently invisible.

**Fix:** Create `train/ip_adapter/constants.py` with `SIGLIP_MEAN`, `SIGLIP_STD`,
`SIGLIP_IMAGE_SIZE = 384`. Import in both precompute and training.

---

### SIGNAL-10: Qwen3 tokenizer `enable_thinking=False` flag not uniform across paths

| Field | Value |
|-------|-------|
| Files | `train/scripts/precompute_all.py:366`, `train/train_ip_adapter.py:2137-2148` |
| Agent | 1 (signal alignment) |

Precompute explicitly passes `enable_thinking=False` to `apply_chat_template`. Training
code does not show this flag — if the tokenizer default changes, the two paths diverge.

**Fix:** Pass `enable_thinking=False` explicitly in both paths.

---

### PIPE-16 through PIPE-30: Additional pipeline script issues

| ID | File | Issue | Severity |
|----|------|-------|---------|
| PIPE-16 | `pipeline_lib.py:255-270` | Heartbeat age calc fails on timestamp parse drift | LOW |
| PIPE-17 | `orchestrator.py:427-429` | Dispatch issues missing process/step context | LOW |
| PIPE-18 | `validator.py:175-183` | No warning if `prev_val_dir` exists but `metrics.json` missing | LOW |
| PIPE-19 | `precompute_all.py:289-305` | SigLIP embedding dim not validated against manifest | LOW |
| PIPE-20 | `download_convert.py:361-374` | Pool copy doesn't pre-check disk space | LOW |
| PIPE-22 | `filter_shards.py:76-87` | Empty sentinel treated as legacy default | LOW |
| PIPE-23 | `mine_hard_examples.py:356-375` | Manifest written without atomic tmp→final pattern | LOW |
| PIPE-24 | `shard_selector.py:247,317` | SQLite `check_same_thread=False` with cross-thread use | LOW |
| PIPE-27 | `orchestrator.py:397` | Log files unbounded — no rotation | LOW |
| PIPE-29 | `orchestrator.py:485` | Error messages not escaped for JSONL serialisation | LOW |

---

### CM-2/3: Same unstable variance formula in f32 norm kernels

| Field | Value |
|-------|-------|
| Files | `iris_shaders.metal:164-166` (`adaln_norm`), `iris_shaders.metal:2164-2166` (`group_norm_f32`) |
| Agent | 5 (C/Metal inference) |

Less catastrophic than bf16 but still the `E[x²] - E[x]²` formula. Apply two-pass fix to
all three norm kernels simultaneously.

---

### CM-10: `embcache` thread safety undocumented

| Field | Value |
|-------|-------|
| Files | `embcache.c:148-150` |
| Agent | 5 (C/Metal inference) |

Global mutable state (`g_cache`, `g_cache_clock`) with no locking and no documentation that
the cache is single-threaded only. Safe today; fragile if a background prefetch worker is
ever added.

**Fix:** Add comment to `embcache.h` that the cache is single-threaded. Or add a
`pthread_mutex_t` guard.

---

## Priority Order for Implementation

**Fix before next training run:**
1. ~~SIGNAL-1~~ FALSE POSITIVE — `(9,18,27)` is correct; do not change
2. ~~CROSS-1~~ MISLEADING — `inject()` hardcodes Flux head count (correct for 4B); real fix is derive from Q shape for 9B safety
3. F-1 (preflight isfile → isfile or isdir)
4. A-1 (stale Q guard)

**Fix before Stage 2 launch:**
5. ~~SIGNAL-2~~ FIXED — `_resize_images_for_siglip()` added; live path now resizes to 384px
6. SIGNAL-3 (null image conditioning gate)
7. ~~EXPORT-1~~ FIXED — export reads `perceiver_heads` from checkpoint sidecar JSON
8. EXPORT-2 (post-export param count check)
9. C-1 (Perceiver residual — verify against InstantX reference first)

**Fix in a dedicated cleanup PR:**
9. ADALN-1 + CM-1/2/3 (Welford variance across all norm kernels)
10. CROSS-4 (injection semantics — align test script with training)
11. PIPE-1 through PIPE-15 (pipeline reliability)
12. Remaining LOW items
