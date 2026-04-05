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
