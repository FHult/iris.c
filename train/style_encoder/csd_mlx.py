"""
csd_mlx.py — CSD style encoder (ViT-L/14 + style head) in pure MLX (SREF-1).

CSD ("Measuring Style Similarity in Diffusion Models", Somepalli et al. 2024,
tomg-group-umd/CSD-ViT-L) is a CLIP ViT-L/14 visual tower fine-tuned for STYLE
retrieval, with the CLIP projection replaced by a style head (and a content head we
don't use). Output: a 768-d L2-normalised style embedding per image — the descriptor
the style-clustering harness needs (the quantized SigLIP cache measured ~no style
signal; BACKLOG SREF-1).

Weights: /Volumes/2TBSSD/models/csd_vit_l_style.safetensors — converted ONCE from the
official torch checkpoint (original OpenAI-CLIP naming, f32). No torch at runtime.

Architecture notes (parity-critical):
  - patch 14, image 224 -> 16x16=256 patches + class token = 257; learned pos embed.
  - pre-LN transformer, 24 blocks, width 1024, 16 heads; QuickGELU (x*sigmoid(1.702x)).
  - style embedding = ln_post(cls) @ last_layer_style, then L2 norm.
  - preprocessing: resize shorter side to 224 (bicubic), center-crop 224, CLIP
    mean/std normalisation.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from PIL import Image

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

WIDTH, LAYERS, HEADS, PATCH, IMG = 1024, 24, 16, 14, 224


def preprocess(img: Image.Image) -> np.ndarray:
    """PIL -> [3, 224, 224] f32, CLIP-normalised (resize short side + center crop)."""
    img = img.convert("RGB")
    w, h = img.size
    s = IMG / min(w, h)
    img = img.resize((max(IMG, round(w * s)), max(IMG, round(h * s))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - IMG) // 2, (h - IMG) // 2
    img = img.crop((left, top, left + IMG, top + IMG))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - CLIP_MEAN) / CLIP_STD
    return x.transpose(2, 0, 1)


class CSDStyleEncoder:
    def __init__(self, weights_path: str):
        self.w = mx.load(weights_path)
        # Pre-transpose the packed qkv/out/mlp weights once: stored [out,in] (torch
        # Linear); we compute x @ W.T as x @ pre-transposed.
        self._wt = {k: v.T for k, v in self.w.items() if k.endswith(".weight") and v.ndim == 2}

    def _linear(self, x: mx.array, name: str) -> mx.array:
        return x @ self._wt[f"{name}.weight"] + self.w[f"{name}.bias"]

    def _ln(self, x: mx.array, name: str) -> mx.array:
        return mx.fast.layer_norm(x, self.w[f"{name}.weight"], self.w[f"{name}.bias"], 1e-5)

    def _block(self, x: mx.array, i: int) -> mx.array:
        p = f"backbone.transformer.resblocks.{i}"
        # attention (packed in_proj)
        h = self._ln(x, f"{p}.ln_1")
        qkv = h @ self.w[f"{p}.attn.in_proj_weight"].T + self.w[f"{p}.attn.in_proj_bias"]
        q, k, v = mx.split(qkv, 3, axis=-1)
        B, T, _ = q.shape
        hd = WIDTH // HEADS
        q = q.reshape(B, T, HEADS, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, HEADS, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, HEADS, hd).transpose(0, 2, 1, 3)
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=hd ** -0.5)
        a = a.transpose(0, 2, 1, 3).reshape(B, T, WIDTH)
        x = x + self._linear(a, f"{p}.attn.out_proj")
        # mlp (QuickGELU)
        h = self._ln(x, f"{p}.ln_2")
        h = self._linear(h, f"{p}.mlp.c_fc")
        h = h * mx.sigmoid(1.702 * h)
        x = x + self._linear(h, f"{p}.mlp.c_proj")
        return x

    def encode_both(self, batch_chw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """[B,3,224,224] -> (style [B,768], content [B,768]), both L2-normalised.

        CSD trains TWO heads on the shared tower: last_layer_style (style
        retrieval) and last_layer_content (content). The content head is what
        makes a CONTENT-LEAK metric possible for sref evaluation: a generation
        should be style-close but content-FAR from its reference (SREF-2)."""
        cls_out = self._tower(batch_chw)
        outs = []
        for head in ("last_layer_style", "last_layer_content"):
            e = cls_out @ self.w[head]
            e = e / mx.maximum(mx.linalg.norm(e, axis=-1, keepdims=True), 1e-8)
            mx.eval(e)
            outs.append(np.array(e))
        return outs[0], outs[1]

    def _tower(self, batch_chw: np.ndarray) -> mx.array:
        """Shared ViT tower: [B, 3, 224, 224] -> ln_post(cls) [B, 1024]."""
        x = mx.array(batch_chw)
        B = x.shape[0]
        # patch embed: conv stride 14 == unfold + matmul
        wconv = self.w["backbone.conv1.weight"]            # [1024, 3, 14, 14]
        x = x.reshape(B, 3, IMG // PATCH, PATCH, IMG // PATCH, PATCH)
        x = x.transpose(0, 2, 4, 1, 3, 5).reshape(B, 256, 3 * PATCH * PATCH)
        x = x @ wconv.reshape(WIDTH, -1).T                 # [B, 256, 1024]
        cls = mx.broadcast_to(self.w["backbone.class_embedding"].reshape(1, 1, WIDTH),
                              (B, 1, WIDTH))
        x = mx.concatenate([cls, x], axis=1) + self.w["backbone.positional_embedding"]
        x = self._ln(x, "backbone.ln_pre")
        for i in range(LAYERS):
            x = self._block(x, i)
        return self._ln(x[:, 0], "backbone.ln_post")       # [B, 1024]

    def encode(self, batch_chw: np.ndarray) -> np.ndarray:
        """[B, 3, 224, 224] f32 (preprocessed) -> [B, 768] L2-normalised style embeds."""
        cls_out = self._tower(batch_chw)
        style = cls_out @ self.w["last_layer_style"]       # [B, 768]
        style = style / mx.maximum(mx.linalg.norm(style, axis=-1, keepdims=True), 1e-8)
        mx.eval(style)
        return np.array(style)
