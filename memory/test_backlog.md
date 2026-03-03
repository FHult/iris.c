# Test Gap Backlog
*Generated 2026-03-03 from comprehensive test suite review*

## Already covered (no model needed)

| Test file | What it covers |
|-----------|---------------|
| `debug/test_lora.c` | LoRA math: scale=0 identity, rank-1 correctness, multi-token, NULL free |
| `debug/test_kernels.c` | softmax (stability, uniformity, multi-row), RMSNorm, matmul, matmul_t, SiLU/SiLU-mul, axpy/add, RoPE (identity at pos=0, norm preservation, freq range), patchify/unpatchify roundtrip, RNG reproducibility |
| `debug/test_embcache.c` | 4-bit quant roundtrip (uniform, linear ramp, signed), metadata, NULL safety, cache hit/miss/clear/overwrite/stats, multi-prompt slots |
| `jpg_test/test.c` | JPEG baseline/progressive/grayscale/subsampling, reference comparison, memory management, API, error cases |
| `png_compare.c` | PNG load + byte equality roundtrip |
| `web/tests/test_server.py` | Flask API endpoints (generate validation, history, favorites, queue reorder, LoRAs, server status, model info, active jobs, style presets, enhance-prompt, save-crop, job status, available-models, switch-model, download-model, cancel — including stuck-server regression) |

---

## Backlog: tests requiring a downloaded model

These are not currently automated. Implement when models are available in CI or when
a specific bug in that area needs a regression test.

---

### TB-001 — Qwen3 Tokenizer Correctness
**Priority:** P1 (pure C, no model weights — only needs tokenizer JSON files)
**Requires:** `flux-klein-4b/tokenizer/tokenizer.json`

The tokenizer can be tested with just the JSON vocab/merges file — no model weights needed.

**Tests to add (`debug/test_tokenizer.c`):**
- Known token IDs for single ASCII words (e.g. `"cat"`)
- Chat template wrapping produces expected prefix tokens
- Whitespace and punctuation tokenization
- Empty string tokenization does not crash
- BPE merge ordering (verify rank ordering is respected)
- `qwen3_pad_tokens` length and mask correctness
- `qwen3_get_id` for known tokens like `<|im_start|>`

---

### TB-002 — Base Model Regression (4B-base CFG path)
**Priority:** P1
**Requires:** `flux-klein-4b-base` model directory

Add a test case to `run_test.py` using `--model-dir flux-klein-4b-base`:
- 64×64 image, 2 steps, seed=42, CFG path
- Compare against reference PNG (generate once, commit to `test_vectors/`)
- Validates: CFG velocity combination, float32 attention, 50-step scheduler

---

### TB-003 — Z-Image Regression (promote optional to required)
**Priority:** P2
**Requires:** `zimage-turbo` model directory

The current Z-Image test in `run_test.py` is skipped if model not present.
- Make the Z-Image test a proper required test case with reference PNG
- Add a second test: 256×256, 4 steps, seed=99

---

### TB-004 — VAE Encode/Decode Roundtrip
**Priority:** P2
**Requires:** Any model's `vae/` directory

Test that VAE encode→decode recovers a near-identical image:
- Load a known PNG, encode to latents, decode back
- Measure PSNR (expect > 30 dB for a lossless-ish codec)
- Validates: patchify/unpatchify, VAE conv layers, scaling factors

---

### TB-005 — img2img Strength Sweep
**Priority:** P2
**Requires:** `flux-klein-4b` model

Test img2img with strength ∈ {0.1, 0.5, 0.9, 1.0}:
- strength=1.0 → result should differ maximally from reference image
- strength=0.1 → result should be close to reference image (low SSIM distance)
- Validates: strength parameter scaling in denoising loop

---

### TB-006 — CFG Guidance Value Validation
**Priority:** P2
**Requires:** `flux-klein-4b-base` model

- guidance=0.0 should produce same output as unconditioned path
- guidance=7.5 (default) vs guidance=1.0 should produce measurably different images
- Validates: CFG velocity combination: `v = v_uncond + g * (v_cond - v_uncond)`

---

### TB-007 — Step Preview (--show-steps) Output
**Priority:** P3
**Requires:** `flux-klein-4b` model

- Run generation with `--show-steps`
- Verify each step image is a valid PNG with correct dimensions
- Verify final step image matches the final output
- Validates: step image decode does not corrupt latents for subsequent steps

---

### TB-008 — Backend Parity (MPS vs generic)
**Priority:** P3
**Requires:** `flux-klein-4b` model, `generic` build target

- Generate the same 64×64 image with MPS and generic builds, same seed
- Compare pixel-by-pixel with a wider tolerance (mean_diff < 5)
- Validates: GPU and CPU paths produce numerically equivalent results

---

### TB-009 — 9B Model Regression
**Priority:** P3
**Requires:** `flux-klein-9b` model (~30 GB)

- 64×64, 2 steps, seed=42, standard prompt
- Reference PNG committed to `test_vectors/`
- Validates: 9B architecture constants (hidden=4096, 8 double blocks, 24 single blocks)

---

### TB-010 — Flash Attention vs Naive Attention Parity
**Priority:** P2 (no model — pure kernel test)
**Requires:** Nothing (pure computation)

`flux_attention` and `flux_flash_attention` must produce identical outputs for the
same inputs. This can be tested in `debug/test_kernels.c` without any model:
- Random Q/K/V tensors with known seed
- Both implementations on same input
- mean_diff < 1e-3 (float32 precision)

**Note:** This test does not require a model — it should be added to test-unit soon.

---

### TB-011 — LoRA Integration (load + apply in transformer)
**Priority:** P3
**Requires:** A LoRA `.safetensors` file + base model

The existing `test_lora.c` only tests the math. Add:
- Load a real LoRA file via `lora_load()`
- Apply it to a generation and compare output with/without LoRA
- Validates: safetensors parsing, weight shapes, scale application

---

## Notes

- **TB-010** (flash attention parity) is the highest-value addition that requires no
  model. It should be promoted from backlog to `debug/test_kernels.c` soon.
- **TB-001** (tokenizer) only requires the tokenizer JSON, not model weights — it
  should be next after TB-010.
- CI recommendation: gate on `make test-unit` + `make web-tests` (no model needed),
  and run `make test` (full inference) only on machines with downloaded models.
