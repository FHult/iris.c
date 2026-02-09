This is a C implementation of the Flux.2 Klein model, an image synthesis model
created by Black Forest Labs, supporting both 4B and 9B parameter variants.

Four model variants are supported:
- **4B Distilled** (flux-klein-4b): 4 steps, no CFG, fast.
- **4B Base** (flux-klein-4b-base): 50 steps default, Classifier-Free Guidance (CFG), higher quality but ~25x slower.
- **9B Distilled** (flux-klein-9b): 4 steps, larger model, higher quality. Non-commercial license.
- **9B Base** (flux-klein-9b-base): 50 steps default, CFG, highest quality. Non-commercial license.

All share the same architecture (rectified flow transformer) with different dimensions. Architecture parameters are read at runtime from the model's config JSON files (`transformer/config.json` and `text_encoder/config.json`). The model type is autodetected from `model_index.json` and the transformer config. The `--base` CLI flag can force base mode.

- The model works both in txt2img mode than in img2img mode with text conditioning.
- The text embedding is created via Qwen3 (4B for the 4B model, 8B for the 9B model).
- The conditioning images, that can be more than one, are VAE-encoded and concatenated with the text embeddings.

For deeper technical details, see the "Implementation Details" section at the bottom.

# File Structure

```
flux.c                  - Main library (generation, img2img)
flux_transformer.c      - Diffusion transformer
flux_sample.c           - Sampling/denoising loop (Euler ODE)
flux_qwen3.c            - Qwen3 text encoder
flux_qwen3_tokenizer.c  - BPE tokenizer
flux_vae.c              - VAE encoder/decoder
flux_kernels.c          - CPU kernels (softmax, RMSNorm, etc.)
flux_metal.m            - Metal GPU acceleration
flux_shaders.metal      - Metal compute kernels
flux_safetensors.c      - Weight loading
flux_image.c            - Image I/O (PNG/PPM/JPEG)
png.c                   - PNG encoder/decoder
jpeg.c                  - JPEG decoder
flux_cli.c              - Interactive CLI mode (REPL)
embcache.c              - Embedding cache (4-bit quantized)
linenoise.c             - Line editing library
terminals.c             - Terminal handling
main.c                  - CLI entry point
```

# Pipeline Overview

```
1. Text Encoding:    prompt → Qwen3 → [512, text_dim] embeddings (text_dim=7680 for 4B, 12288 for 9B)
2. Latent Init:      random noise [H/16, W/16, 128]
3. Denoising Loop (4 steps distilled, 50 steps base):
   └─ per step: double blocks → single blocks → final layer → velocity
4. VAE Decode:       latents → VAE decoder → RGB image

img2img: Reference images are VAE-encoded and passed as extra tokens (not noise).
The model attends to them via joint attention (in-context conditioning).

Base model CFG: each denoising step runs the transformer twice (once with empty
prompt, once with real prompt). Combined as: v = v_uncond + guidance * (v_cond - v_uncond).
The empty prompt is literal "" through the Qwen3 chat template. The two passes
are run sequentially (not batched).
```

# Key Architecture Constants

These are now read at runtime from the model's config JSON files. Reference values:

4B Transformer: hidden=3072, heads=24, head_dim=128, mlp=9216, double_blocks=5, single_blocks=20, text_dim=7680
9B Transformer: hidden=4096, heads=32, head_dim=128, mlp=12288, double_blocks=8, single_blocks=24, text_dim=12288

4B Qwen3: hidden=2560, q_heads=32, kv_heads=8 (GQA 4:1), layers=36, output_dim=7680 (3×2560)
9B Qwen3: hidden=4096, q_heads=32, kv_heads=8 (GQA 4:1), layers=36, output_dim=12288 (3×4096)

Shared: latent_ch=128, head_dim=128, VAE architecture identical across all models.

# Critical Implementation Details

These details have caused bugs and are easy to get wrong:

- **Concatenation order** is `[TEXT, IMAGE]` not `[IMAGE, TEXT]` for Q, K, V
- **AdaLN formula**: `out = (1 + scale) * norm(x) + shift` (apply shift after scale)
- **Final layer** projection output splits as `(scale, shift)` NOT `(shift, scale)`
- **RoPE rotation**: `out[0] = cos*x0 - sin*x1`, `out[1] = cos*x1 + sin*x0` (see RoPE section for axis details)

# Build Targets

This project implements three different targets:

- MPS: for Apple Silicon.
- BLAS: for optimized CPU inference via SIMD.
- generic: for CPUs, pure C, very slow.

# Development rules

- We don't add any dependency to this project. Even the PNG and JPG support is implemented internally. The only acceptable dependencies are the blas / openblas library and the Metal primitives that are part of MacOS.
- Don't accept speed improvements that are just marginal, like 1%: they may be just random fluctations among runs. Refuse small speed improvements especially if they make the code more complicated, however more complex code for important speed improvements is ok.
- Always test a code modification after implementing it, using `make test`
- Once you reach a positive result, commit it.
- Never add or commit unstaged files, unless you created them for a specific purpose.
- Code must be simple and understandable.
- No dead code must be left around.
- When you work on a single target (for instance MPS) you need to make sure OpenBLAS / CPU still work in case you did modifications that potentially created issues.
- Stick to standard C, no compiler-specific tricks, pragmas, ...

# How to run the project

    ./flux2 -d flux-klein-4b -p "a cat and a dog playing" -o /tmp/test.png
    ./flux2 -d flux-klein-4b-base -p "a cat and a dog playing" -o /tmp/test.png
    ./flux2 -d flux-klein-9b -p "a cat and a dog playing" -o /tmp/test.png
    ./flux2 -d flux-klein-9b-base -p "a cat and a dog playing" -o /tmp/test.png

You have your weights ready in `flux-klein-4b` (4B distilled), `flux-klein-4b-base` (4B base), `flux-klein-9b` (9B distilled), and `flux-klein-9b-base` (9B base). If you can't find them, there is a download script, but before using it ask the user.

# Where to find the reference implementation in Python

If you need to compare your outputs with the reference implementation, or if you need to inspect the reference implementation for clues about the model, you fill find it here:

- Python venv to run the Python inference pipeline is in ./flux_env/
- Python implementation of flux2 (official) is in ./flux2

Rules:

- Never add / commit such directories into the git repository.
- If you can't find such directories, ask the user before creating a venv and downloading the Flux2 official inference code.

# Debugging

Into ./debug you can find Python scripts useful for debugging. If you need more debugging scripts, please add them there, with a name that let humans and LLMs able to understand what they do easily. Put there only reusable testing components, otherwise just create them in /tmp and discard them later.

The JPEG code can be tested with the tools in ./jpg_test

# RoPE (Rotary Position Embedding)

4-axis RoPE with 32 dims per axis (128 total = head_dim):
- Axis 0 (dims 0-31): T position (temporal, used for img2img reference offset)
- Axis 1 (dims 32-63): H position (height/y coordinate)
- Axis 2 (dims 64-95): W position (width/x coordinate)
- Axis 3 (dims 96-127): L position (sequence index)

Token types:
- Image tokens: use axes 1,2 (spatial position), axes 0,3 = identity (cos=1, sin=0)
- Text tokens: use axis 3 only (sequence position), axes 0,1,2 = identity
- Reference image tokens (img2img): same as image but axis 0 has T offset (10, 20, 30...)

# Timestep Embedding

1. Input timestep scaled by 1000 (t=1.0 → 1000.0)
2. Sinusoidal embedding: 128 frequencies → 256 dims
3. MLP: linear(256→hidden) + SiLU + linear(hidden→hidden)

# Text Encoder (Qwen3)

Chat template format:
```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

Layer extraction: outputs from layers 8, 17, 26 (0-indexed) are concatenated → [seq, 3*hidden] (7680 for 4B, 12288 for 9B)

# VAE

Latent space: 32 channels, 16x spatial compression

Patchification (encode): [B, 32, H/8, W/8] → [B, 128, H/16, W/16]
Unpatchification (decode): [B, 128, H/16, W/16] → [B, 32, H/8, W/8]

Channel multipliers: [1, 2, 4, 4] → [128, 256, 512, 512]

# Double Block Flow

Input: img_hidden [img_seq, hidden], txt_hidden [txt_seq, hidden]

1. AdaLN normalize both streams (shift1, scale1)
2. Separate Q, K, V projections for each stream
3. QK normalization (per-head RMSNorm)
4. Apply RoPE (image: axes 1,2; text: axis 3)
5. Joint attention: concatenate [txt_k, img_k] and [txt_v, img_v], each Q attends to full KV
6. Output projection + gating (gate1)
7. Residual add
8. AdaLN normalize (shift2, scale2)
9. FFN with SiLU-gated MLP
10. Gating (gate2) + residual add

Modulation params per stream: shift1, scale1, gate1, shift2, scale2, gate2 (6 × hidden)

# Single Block Flow

Input: concatenated [txt_hidden, img_hidden] as single sequence

1. AdaLN normalize (shift, scale from t_emb)
2. Fused QKV + MLP projection → [Q, K, V, gate, up] per position
3. QK normalization
4. Apply RoPE: text portion uses axis 3, image portion uses axes 1,2
5. Self-attention over full sequence
6. SwiGLU: gate = silu(gate) * up
7. Concatenate attention output and MLP output
8. Output projection
9. Gating + residual add

Modulation params: shift, scale, gate (3 × hidden)

# Test Commands

Basic verification (2-step catches multi-step bugs):
```bash
make test
```

Manual verification:
```bash
./flux2 -d flux-klein-4b -p "A fluffy orange cat sitting on a windowsill" \
  --seed 42 --steps 2 -o /tmp/test.png -W 64 -H 64

python3 -c "
import numpy as np; from PIL import Image
ref = np.array(Image.open('test_vectors/reference_2step_64x64_seed42.png'))
test = np.array(Image.open('/tmp/test.png'))
diff = np.abs(ref.astype(float) - test.astype(float))
print('PASS' if diff.max() < 2 else 'FAIL')
"
```

# Known Pitfalls (Historical Bugs)

1. **Unified RoPE kernel indexing**: GPU must use consecutive pairs (d, d+1), not axis-based (i0, i0+half_axis)
2. **GPU caching of timestep params**: shift/scale/gate change each step, don't use weight cache for them
3. **CLI mode CFG routing**: In interactive CLI mode, the base model must go through `flux_generate()` (which handles CFG internally), not `flux_generate_with_embeddings()` which only supports single-embedding distilled path
