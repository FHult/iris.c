# IP-Adapter Training — Performance Backlog

Current baseline: **~7s/step** (0.15 steps/s) at batch_size=2, 512px, 32GB M1 Max.
- `fwd=5.5s` — Flux forward through 25 blocks (no grad), `mx.eval(flux_state)`
- `step=0.0s` — adapter backward (tiny graph)
- `eval=1.5s` — `mx.eval(loss_val, adapter.parameters(), optimizer.state)`

---

## TP-001: Split-forward ✓ DONE (commit 70f029e)

**Win:** ~190s/step → ~7s/step (27×).
Flux forward runs outside `nn.value_and_grad`; only adapter graph is traced.
`_flux_forward_no_ip` collects Q vectors (stop_gradient'd), returns `flux_state`.
`mx.eval(flux_state)` materialises before entering autodiff.

---

## TP-002: mx.compile on adapter step — INFEASIBLE

`flux_state` contains dynamically-created arrays passed via dict; MLX raises
"uncaptured inputs." Not worth working around: adapter step is already ~0.0s —
Python graph-tracing overhead is negligible at 7s/step.

---

## TP-003: Async eval with periodic sync (LOW)

**Expected win:** hides ~5–10ms Python overhead (negligible at 7s/step).
Not worth pursuing at current step time.

---

## TP-004: Resolution warmup (LOW)

Not worth implementing — JIT compilation is now amortised by step 3; first step
is ~10s (shader compilation), steady state 5-6s.

---

## TP-005: Per-phase timing ✓ DONE

Phase breakdown implemented. `fwd`/`step`/`eval` printed at each log interval.

---

## TP-006: Flux forward optimisation (MEDIUM)

Flux forward is 5.5s/step. Potential wins:
- The Flux forward in `_flux_forward_no_ip` could be replaced by mflux's own
  forward pass if we can hook into it to extract Q vectors, avoiding re-running
  the same computation twice. Currently we run our own copy of Flux forward.
- Check if mflux transformer blocks can be called without copy.

---

## DQ-001: Shard cache coverage ✓ FIXED (commit 90422a9)

**Root cause:** Only 34/432 shards had qwen3+vae cache. Loader read all 432 tars
(1.9 GB each) but 92% of batches were silently skipped by training loop. This
caused GPU data stalls when sample_q drained during runs of uncached shards.

**Fix:** Filter `shard_paths` to only shards with qwen3+vae before loader creation.
Active shards: 34 (from 432). Prints coverage at startup.

**Remaining data gap:**
- 34 shards × 5000 samples = 170k usable samples (qwen3+vae)
- 11/34 shards have siglip; other 23 use zero features (null-image conditioning)
- 105K steps × bs=2 = 210K samples needed → ~1.24 epochs through 170k samples
- To expand: run `train/scripts/precompute_all.py` on more shards

**SigLIP gap:** 23/34 usable shards lack SigLIP → those batches always train with
zero image features regardless of `image_dropout_prob`. Run precompute_siglip.py
on shards 000000–000033 to fill this gap.

---

## DQ-002: Truncated JPEG in dataset ✅ DONE

- `_decode_jpeg` in `dataset.py` already logs `rec={rec_id}` alongside any turbojpeg warning.
- `build_shards.py` now does a full `tj.decode()` (was `decode_header` only) for JPEG
  validation at shard-build time. Truncated files raise an exception, are logged with
  `rec={id} src={src}`, and are skipped. Pillow path uses `.verify()` equivalently.

---

## Notes

- At 7s/step: 105K steps ≈ 8.5 days. On target (was 58 days at 100s/step).
- Next action: run `precompute_siglip.py` on the 34 cached shards to fix the
  23-shard gap, then launch full training run.

---

## Experiment: batch_size=2 re-trial (2026-04-05)

### Background
The original pipeline used batch_size=2 but all training runs on 2026-04-02 were
killed by macOS jetsam (SIGKILL 9) after 200–300 steps. Six separate attempts all
failed at approximately the same point.

### Root cause (confirmed)
The OOM was caused by **EMA lazy-graph accumulation**, not raw model memory:

- `update_ema()` returns a new lazy MLX tree: `ema = 0.9999 * ema_old + 0.0001 * model`
- Without `mx.eval(ema_params)` after the update, MLX chains each update lazily
- By step 200 (20 EMA updates × ~950MB adapter): ~19GB of live intermediate arrays
- This is on top of the Flux transformer (~8GB), adapter (~1GB), optimizer state (~2GB),
  and activations — total exceeded the 14GB MLX limit and triggered jetsam

### Fix applied (commit 5639d32)
Added `mx.eval(ema_params)` immediately after each `update_ema()` call in the
training loop. This breaks the lazy chain — each update materialises in-place,
keeping peak EMA memory at ~950MB (single copy of adapter weights).

### Hypothesis for batch_size=2 success
With the EMA fix in place, peak memory profile per step is:
- Flux transformer weights: ~8GB (frozen, mmap'd, pre-evaluated)
- Flux activations (fwd pass): ~2–3GB peak, freed before backward
- Adapter + EMA (materialised): ~2GB total
- Optimizer state: ~1GB
- VAE/text precomputed (CPU numpy): negligible
- Total steady-state: ~13–14GB — within the 14GB MLX limit

batch_size=2 doubles activation memory during the Flux forward (~4–6GB peak),
which may push individual steps close to the limit. However jetsam should not
trigger because:
1. The sustained memory floor is the same (weights + adapter)
2. Activation peak is transient and freed before mx.eval
3. mx.clear_cache() after each step returns freed buffers to OS immediately
4. The EMA chain can no longer accumulate across steps

### Expected outcome
If batch_size=2 holds: ~0.35–0.38 steps/s (vs 0.19 at bs=1), halving total wall-clock.
- small run: ~3 days (was ~6)
- medium run: ~7 days (was ~14)

### Risk
If jetsam fires again at bs=2, the failure will likely occur later (step 500+)
as EMA is no longer the cause. New culprit would be activation peak during a
large-bucket forward (896×512 has ~3500 image tokens vs ~1600 for 512×512).
Mitigation: reduce to bs=1 for large buckets only (not yet implemented).

### Outcome (2026-04-05) — REVERTED to batch_size=1

Experiment ran to step 600 with no OOM — EMA fix confirmed effective.

However batch_size=2 provided **zero throughput improvement**:
- bs=1: fwd=416s/100steps → 0.19 steps/s
- bs=2: fwd=720s/100steps → 0.12 steps/s (worse per-sample)

Forward time scaled nearly linearly with batch size. The hypothesis that batching
would amortise fixed overhead was wrong. On M1 Max at these sequence lengths, the
Flux attention is **memory-bandwidth bound**, not compute bound. Doubling batch
doubles memory traffic and time — no amortisation benefit.

Additionally, bs=2 uses ~2× activation memory for no gain, reducing headroom for
future optimisations and increasing jetsam risk unnecessarily.

**Conclusion:** batch_size=1 is optimal for M1 Max at 512px. Reverted.
Batch_size=2 is not worth pursuing unless the attention implementation changes
(e.g. Flash Attention with fused BF16 kernels that could saturate compute rather
than memory bandwidth).
