# Proxy-VAE Validation Runbook — "Is a distilled VAE good enough?"

**Purpose.** The proxy-VAE distillation stack is fully implemented (v3.18.0 + v3.19)
but **never validated**: no proxy checkpoint has been trained, and the definitive
downstream A/B has never run (BACKLOG.md PRECOMP-2, "Still pending validation").
This runbook is the checklist that turns "good enough?" into a measured verdict.

**The one-line question.** Not "are proxy latents identical to the teacher's"
(unachievable and unnecessary), but: *does an IP-Adapter trained on proxy latents
match one trained on real latents on held-out quality, AND does the trained model
still decode correctly through the C `iris_vae` used at inference?* Both halves must
pass. The first is **Tier-2** below; the second is the **pre-trust gate** (the C↔teacher
parity items) that no Tier-1/2 number can catch.

**Why "good enough" is a dial, not a binary.** Precompute already supports a per-image
**confidence fallback** (`fallback_threshold`, default 0.75): the proxy runs where it is
confident and defers to the real VAE elsewhere. So the deliverable is a
speed/fidelity operating point, not pass/fail.

---

## Prerequisites

| Item | Status (2026-06-05) |
|------|---------------------|
| Distillation corpus (≥200K teacher latents) | **Ready** — 471,721 pairs, cold `vae/v_2232c1` |
| Proxy training configs | **Ready** — `vae_proxy_{small,512px,medium}.yaml` (3.4M / 6M / 9.3M) |
| Train / eval / benchmark / A/B tooling | **Ready** — `train/scripts/{train_vae_proxy,evaluate_vae_proxy,benchmark_vae_proxy,compare_downstream_quality}.py` |
| Idle GPU | **Blocked** — flywheel warmup-run2 holds the GPU lock |
| Pre-trust gate (C-1, C-2 below) | **Not started** — see triage |

**Sequencing.** Training + Tier-2 are GPU-heavy and `compare_downstream_quality.py`
*refuses under the GPU lock unless `--force`* (do not `--force` during a live campaign).
Natural execution window: **flywheel-idle or M5-Max bring-up** (the M5 is where you
re-precompute anyway — validating the proxy there kills two birds). Tier-1 latent eval
and the speed benchmark (`benchmark_vae_proxy.py --proxy ... ` without `--with-teacher`)
are lighter and proxy-only-safe, but still prefer idle GPU for clean numbers.

---

## Part A — Blocking pre-trust triage (from `grok_proxy_vae_analysis.md`)

These gate **production use of proxy latents**, independent of how good Tier-1/2 look.
A proxy can pass every Tier-1/2 metric *against the teacher* and still degrade real
inference if the C VAE (the actual inference engine) diverges from that teacher. Resolve
**C-1 and C-2 before any proxy latent enters a training set you intend to ship from.**

### 🔴 C-1 — C `iris_vae` ↔ teacher-VAE parity is assumed, never tested (BLOCKING)
- The proxy is validated against the **teacher** (diffusers/mflux) used for precompute.
  But inference uses the **separate C `iris_vae`** for ref encoding (img2img/sref/IP)
  and final decode. If C encode/decode differs in distribution or spatial structure
  (GroupNorm impl, eps 1e-4 vs 1e-6, pad, attn matmul order, unpatch index, Metal vs CPU
  path, BN-stats load), the model trains in one latent space and runs in another.
- Today there are **no golden latent vectors** and no C-side encode test against known
  teacher outputs (`debug/test_kernels.c` covers only patch roundtrip; `run_test` refs
  are end-to-end images with non-deterministic tolerances).
- Historical proof this bites: the MPS SGEMM B-cache VAE-decode corruption (CLAUDE.md
  "Known Pitfalls" #1).
- **Fix (do first):** add a golden test — encode a fixed image set with both the teacher
  and C `iris_vae`, assert per-channel mean/std and decoded LPIPS/PSNR within tight
  tolerance. This pins the invariant the proxy must align to and guards future C/teacher
  drift. Cross-backend (CPU vs Metal) decode parity belongs here too.
- **Concrete harness spec (verified signatures, 2026-06-05):**
  - *No teacher re-run / no GPU needed* — the teacher references already exist in the
    cold cache (`vae/v_2232c1`, 471K latents). Generation = extract one (image, teacher
    latent) pair from cold to raw files: source image from the shard `.tar` → `.ppm`,
    teacher latent from the `.npz` → float32 `.bin`. A `debug/gen_vae_parity_fixture.py`.
  - *C side* (`debug/vae_parity.c`, compiles against `iris_vae.c` + `iris_kernels.c`,
    CPU-only like `test_vae.c`, no GPU): load real weights via
    `iris_vae_load_safetensors_ex(sf, z, scale, shift)`; feed the image as **CHW float in
    [-1,1]** (`iris_vae_encode(vae, img, 1, H, W, &oh, &ow)`); assert per-channel
    mean/std-ratio (0.95–1.05) + max-abs error vs the teacher latent.
  - **Representation trap:** `iris_vae_encode` returns the **patchified** `[128, H/16,
    W/16]` form, but the teacher precompute stores the **un-patchified** `[32, H/8, W/8]`
    form (what `_load_vae_latent` consumes). The check MUST align them — unpatchify the C
    output (or patchify the teacher latent) with the deterministic `iris_kernels`
    patch/unpatch — and match the **normalization branch** (Flux BN vs Z scale/shift)
    that `iris_parse_vae_config` resolved. Build + verify this at idle against real
    weights/cold data; correctness hinges on these two alignments, so it is not a blind
    write.

### 🔴 C-2 — brittle ad-hoc `vae/config.json` parser can misconfigure z/scale/shift/BN (BLOCKING)
- `iris.c:~404` uses `strstr`/`atoi` on a fixed 4096 buffer for `latent_channels`,
  `scaling_factor`, `shift_factor`; Flux-vs-Z and BN-vs-scale decided by
  `safetensors_find` on optional tensors. A pretty-printed JSON, renamed key, or a
  variant missing an expected optional tensor silently selects the wrong normalization
  branch or latent channel count → wrong latent semantics for both transformer and proxy.
- **Fix:** schema-validate the VAE config parse (or a golden config-load test asserting
  the resolved `{z_channels, scale, shift, norm_branch}` for each known model dir).

### 🟠 H-1 — proxy is Flux-32ch only; Z-Image (16ch) has no proxy path (SCOPE, non-blocking now)
- The current flywheel trains a **Flux** IP-Adapter, so Flux-only is fine *today*. But if
  a campaign uses/switches to Z-Image-Turbo, precompute is VAE-gated again with no relief.
- **Action:** explicitly scope this runbook to Flux; file Z-Image proxy as a separate
  follow-up so the limitation is conscious, not accidental.

### 🟠 H-4 — Tier-2 is the only gate; no automated C-side enforcement (process)
- Resolved structurally by C-1's golden test + running this runbook. Track that the
  proxy version is keyed separately in the precompute cache (`encoder_config_subset`
  already carries a vae-proxy identity) so proxy latents never pollute real-VAE caches.

(H-2 C-encode-not-GPU-resident and H-3 scalar patch/unpatch are perf items, not
correctness gates for proxy trust — defer to the Metal backlog.)

---

## Part B — Execution: train → Tier-1 → benchmark → Tier-2 → tune

Run from repo root with `train/.venv/bin/python`. Start with the **`default` (6M)**
variant; it is the design's recommended first attempt. Fall back to `medium` if Tier-1
channel stats miss; try `small` only if Tier-2 already passes and you want more speed.

### Step 0 — pre-trust gate
Land C-1 (golden teacher↔C parity test) and C-2 (config-parse golden) and confirm
green. **Do not proceed to production promotion without these**, though you may train +
Tier-1/2 a candidate in parallel to get early signal.

### Step 1 — train the proxy (GPU, hours; needs idle GPU)
```bash
train/.venv/bin/python train/scripts/train_vae_proxy.py \
  --config train/configs/vae_proxy_512px.yaml
```
Corpus is the cold `vae/v_2232c1` latents (config points at it). Ensure the training
split holds out ≥2 full shards and ≥30% natural (LAION/COYO) shards per the design, so
the proxy isn't JourneyDB-overfit.

### Step 2 — Tier-1 latent quality (fast; the distributional gate)
```bash
train/.venv/bin/python train/scripts/evaluate_vae_proxy.py \
  --proxy <proxy_ckpt> --tier 1 --n-images 2000 \
  --report /Volumes/2TBSSD/proxy_vae_eval.html \
  --out    /Volumes/2TBSSD/proxy_vae_eval.json
```
**Pass gates (design Tier-1):**

| Metric | Target | If it fails |
|--------|--------|-------------|
| Per-channel **std ratio** | 0.95–1.05× real | **most important** — wrong → shifts diffusion SNR; go `medium`/Option-D |
| Decoded **LPIPS** | < 0.04 | spatial blur → add frequency-weighted loss |
| Decoded **PSNR** | > 35 dB | gross recon error |
| Per-channel norm. MSE | < 0.01 | channel structure mismatch |
| Per-channel mean error | < 0.001 | systematic channel bias |

`pipeline_doctor.py` `_check_proxy_vae` reads `proxy_vae_eval.json` and surfaces this.

### Step 3 — speed benchmark (confirm the 5–9× is real)
```bash
train/.venv/bin/python train/scripts/benchmark_vae_proxy.py \
  --proxy <proxy_ckpt> --variants default --with-teacher \
  --out /Volumes/2TBSSD/proxy_vae_bench.json
```
`--with-teacher` needs idle GPU. Target ≥5× encode speedup (design goal ~9× at ~20ms/img).

### Step 4 — Tier-2 downstream A/B (THE verdict; GPU-heavy, idle only)
```bash
train/.venv/bin/python train/scripts/compare_downstream_quality.py \
  --proxy <proxy_ckpt> --base-config train/configs/stage1_512px.yaml \
  --n-shards 4 --steps 500 --tolerance 0.05 \
  --out /Volumes/2TBSSD/proxy_downstream_ab.json
```
Trains a 500-step IP-Adapter on real vs proxy latents and compares held-out CLIP-I,
loss-curve convergence, and same-seed visual samples. **This is the "good enough"
answer.** Default `--tolerance 0.05` = proxy is good enough if downstream quality is
within 5% of real. Refuses under the GPU lock unless `--force` (don't, mid-campaign).

### Step 5 — pick the operating point (if Tier-2 is marginal)
If Tier-2 is close but not clean, don't discard the proxy — raise `fallback_threshold`
in the `proxy_vae` block of `v2_pipeline.yaml` (e.g. 0.75 → 0.90) so more images defer to
the real VAE. Re-run Tier-2 at the new threshold. This trades some speedup for fidelity
and is the intended way to reach "good enough." Per-campaign overrides live under
`proxy_vae.campaigns.<name>`.

---

## Part C — Cheaper alternative to validate in parallel: subsampled precompute

Before committing to the proxy, the design flags a config-only lever (no model):
precompute **1K images/shard instead of 5K** → ~4h/iter (vs ~20h), if downstream quality
holds within ~5%. Validate with the *same* Tier-2 harness (a real-vs-subsampled A/B). If
it passes, the proxy may be unnecessary for the near-term precompute speedup. Implement/
confirm the `--subsample-per-shard N` flag in `precompute_all.py` (design §339) and run
one `compare_downstream_quality`-style A/B. Cheapest experiment with the highest chance
of mooting the whole proxy effort — **do this first if the only goal is precompute time.**

---

## Promotion decision

Promote the proxy to production precompute **only when all hold:**
1. **C-1 + C-2 green** (golden teacher↔C parity + config-parse) — the pre-trust gate.
2. **Tier-1 pass**, std-ratio in 0.95–1.05 above all.
3. **Benchmark ≥5×** encode speedup.
4. **Tier-2 within tolerance** at a chosen `fallback_threshold`.
5. Proxy version keyed separately in the precompute cache (no real-VAE cache pollution).

If subsampled precompute (Part C) reaches the same precompute-time goal and passes Tier-2,
prefer it — it adds no model, no parity surface, and no new failure mode.

---

## Status / next action
- **Now (no GPU):** land C-1 and C-2 (golden tests). These are the only things that can
  be done without disturbing the flywheel, and they're prerequisites anyway.
- **At flywheel-idle / M5 bring-up:** run Part C (subsampling A/B) first, then Part B
  Steps 1–5 if the proxy is still wanted.
- Owner-gated on hardware; tracked under BACKLOG PRECOMP-2.
