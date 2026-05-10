"""
train/scripts/mlx_clip_embed.py — Native MLX CLIP ViT-L-14 image encoder.

Loads weights directly from the open_clip / timm safetensors file that
open_clip already caches on first use.  No weight conversion step needed.

Architecture: ViT-L-14 (openai pretrained)
  patch_size=14, image_size=224, width=1024, heads=16, layers=24, embed_dim=768
  QuickGELU activation, layer-norm before attention and MLP.

Usage:
    from mlx_clip_embed import MLXCLIPEmbedder
    embedder = MLXCLIPEmbedder()          # loads weights on first call
    embs = embedder.embed_batch(pil_list) # float32 np.ndarray [N, 768], L2-normed
"""

import math
import os
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# ---------------------------------------------------------------------------
# Architecture constants for ViT-L-14 (openai)
# ---------------------------------------------------------------------------

_PATCH   = 14
_SIZE    = 224
_N_PATCH = (_SIZE // _PATCH) ** 2    # 256
_N_SEQ   = _N_PATCH + 1             # 257 (CLS + patches)
_WIDTH   = 1024
_HEADS   = 16
_HEAD_D  = _WIDTH // _HEADS         # 64
_LAYERS  = 24
_MLP_W   = 4096
_EMB_DIM = 768

# CLIP normalisation (openai pretrained)
_MEAN = np.array([0.48145466, 0.4578275,  0.40821073], dtype=np.float32)
_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# Default weights: open_clip caches via timm HuggingFace hub
_SNAP_ID = "18d0535469bb561bf468d76c1d73aa35156c922b"
_DEFAULT_WEIGHTS = (
    Path.home() / ".cache" / "huggingface" / "hub"
    / "models--timm--vit_large_patch14_clip_224.openai"
    / "snapshots" / _SNAP_ID / "open_clip_model.safetensors"
)


# ---------------------------------------------------------------------------
# MLX ViT-L-14 modules
# ---------------------------------------------------------------------------

def _quick_gelu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(1.702 * x)


class _ResBlock(nn.Module):
    """One ViT transformer block (pre-LN, QuickGELU, fused QKV)."""

    def __init__(self) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(_WIDTH)
        self.ln_2 = nn.LayerNorm(_WIDTH)
        # Fused QKV stored as a single weight (3*width, width)
        self.in_proj_weight: mx.array = mx.zeros((_WIDTH * 3, _WIDTH))
        self.in_proj_bias:   mx.array = mx.zeros((_WIDTH * 3,))
        self.out_proj = nn.Linear(_WIDTH, _WIDTH)
        self.c_fc     = nn.Linear(_WIDTH, _MLP_W)
        self.c_proj   = nn.Linear(_MLP_W, _WIDTH)

    def _attn(self, x: mx.array) -> mx.array:
        B, N, _ = x.shape
        # Fused QKV projection
        qkv = x @ self.in_proj_weight.T + self.in_proj_bias   # (B, N, 3W)
        q, k, v = mx.split(qkv, 3, axis=-1)                   # each (B, N, W)
        # Reshape to (B, H, N, D)
        q = q.reshape(B, N, _HEADS, _HEAD_D).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, _HEADS, _HEAD_D).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, _HEADS, _HEAD_D).transpose(0, 2, 1, 3)
        # Flash attention (MLX fast path)
        scale = 1.0 / math.sqrt(_HEAD_D)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, _WIDTH)  # (B, N, W)
        return self.out_proj(out)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self._attn(self.ln_1(x))
        h = self.ln_2(x)
        h = _quick_gelu(self.c_fc(h))
        x = x + self.c_proj(h)
        return x


class _VisionTransformer(nn.Module):
    """ViT-L-14 visual encoder (image side only)."""

    def __init__(self) -> None:
        super().__init__()
        # Patch embedding: Conv2d with stride=patch_size (MLX: channels-last)
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=_WIDTH,
            kernel_size=_PATCH, stride=_PATCH, bias=False,
        )
        self.class_embedding    = mx.zeros((_WIDTH,))
        self.positional_embedding = mx.zeros((_N_SEQ, _WIDTH))
        self.ln_pre  = nn.LayerNorm(_WIDTH)
        self.blocks  = [_ResBlock() for _ in range(_LAYERS)]
        self.ln_post = nn.LayerNorm(_WIDTH)
        self.proj    = mx.zeros((_WIDTH, _EMB_DIM))

    def __call__(self, x: mx.array) -> mx.array:
        """x: float16/32 (B, H, W, C) normalised images."""
        # Patch embed: (B, H, W, C) → (B, H', W', width) → (B, N, width)
        x = self.conv1(x)
        B, pH, pW, W = x.shape
        x = x.reshape(B, pH * pW, W)                          # (B, 256, 1024)

        # Prepend CLS token
        cls = mx.broadcast_to(self.class_embedding, (B, 1, _WIDTH))
        x   = mx.concatenate([cls, x], axis=1)                # (B, 257, 1024)
        x   = x + self.positional_embedding                    # broadcast over B

        x = self.ln_pre(x)
        for block in self.blocks:
            x = block(x)

        # Extract CLS token, project to embedding space
        x = self.ln_post(x[:, 0, :])                          # (B, 1024)
        x = x @ self.proj                                      # (B, 768)
        return x


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _load_weights(model: _VisionTransformer, path: Path) -> None:
    """Load visual encoder weights from open_clip safetensors file."""
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as f:
        keys = [k for k in f.keys() if k.startswith("visual.")]

        def _get(key: str) -> mx.array:
            return mx.array(f.get_tensor(key))

        # Patch embedding weight: (out, in, kH, kW) → (out, kH, kW, in) for MLX
        w = f.get_tensor("visual.conv1.weight")        # (1024, 3, 14, 14)
        model.conv1.weight = mx.array(w.transpose(0, 2, 3, 1))  # (1024, 14, 14, 3)

        model.class_embedding     = _get("visual.class_embedding")
        model.positional_embedding = _get("visual.positional_embedding")
        model.ln_pre.weight       = _get("visual.ln_pre.weight")
        model.ln_pre.bias         = _get("visual.ln_pre.bias")
        model.ln_post.weight      = _get("visual.ln_post.weight")
        model.ln_post.bias        = _get("visual.ln_post.bias")
        model.proj                = _get("visual.proj")          # (1024, 768)

        for i, block in enumerate(model.blocks):
            pfx = f"visual.transformer.resblocks.{i}"
            block.in_proj_weight      = _get(f"{pfx}.attn.in_proj_weight")
            block.in_proj_bias        = _get(f"{pfx}.attn.in_proj_bias")
            block.out_proj.weight     = _get(f"{pfx}.attn.out_proj.weight")
            block.out_proj.bias       = _get(f"{pfx}.attn.out_proj.bias")
            block.ln_1.weight         = _get(f"{pfx}.ln_1.weight")
            block.ln_1.bias           = _get(f"{pfx}.ln_1.bias")
            block.ln_2.weight         = _get(f"{pfx}.ln_2.weight")
            block.ln_2.bias           = _get(f"{pfx}.ln_2.bias")
            block.c_fc.weight         = _get(f"{pfx}.mlp.c_fc.weight")
            block.c_fc.bias           = _get(f"{pfx}.mlp.c_fc.bias")
            block.c_proj.weight       = _get(f"{pfx}.mlp.c_proj.weight")
            block.c_proj.bias         = _get(f"{pfx}.mlp.c_proj.bias")

    mx.eval(model.parameters())


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _preprocess(pil_images: list) -> mx.array:
    """PIL images → float16 MLX (B, H, W, C) normalised for CLIP."""
    from PIL import Image as _PIL

    out = []
    for img in pil_images:
        # Resize: shortest side → 224, bicubic
        w, h = img.size
        scale = _SIZE / min(w, h)
        new_w, new_h = round(w * scale), round(h * scale)
        img = img.resize((new_w, new_h), _PIL.BICUBIC)

        # Centre crop
        left = (new_w - _SIZE) // 2
        top  = (new_h - _SIZE) // 2
        img  = img.crop((left, top, left + _SIZE, top + _SIZE))

        arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0  # (224, 224, 3)
        arr = (arr - _MEAN) / _STD
        out.append(arr)

    imgs = np.stack(out, axis=0)                                       # (B, 224, 224, 3)
    return mx.array(imgs, dtype=mx.float16)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class MLXCLIPEmbedder:
    """
    MLX ViT-L-14 image embedder.  Drop-in replacement for the PyTorch path.

    Loads the same weights already downloaded by open_clip (no separate
    conversion or download step required).

    Example::
        embedder = MLXCLIPEmbedder()
        embs = embedder.embed_batch(pil_images)  # np.float32 [N, 768], L2-normed
    """

    def __init__(self, weights_path: Optional[Path] = None) -> None:
        self._weights_path = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS
        self._model: Optional[_VisionTransformer] = None

    def load(self) -> None:
        """Load model weights (called automatically on first embed_batch)."""
        if self._model is not None:
            return
        if not self._weights_path.exists():
            raise FileNotFoundError(
                f"MLX CLIP weights not found: {self._weights_path}\n"
                "Run the embed command once with --clip-backend open_clip to "
                "download weights, then switch to --clip-backend mlx."
            )
        model = _VisionTransformer()
        _load_weights(model, self._weights_path)
        # Warm up compilation with a dummy batch
        _dummy = mx.zeros((1, _SIZE, _SIZE, 3), dtype=mx.float16)
        mx.eval(model(_dummy))
        self._model = model

    def embed_batch(self, pil_images: list) -> np.ndarray:
        """
        Embed a list of PIL images.

        Returns L2-normalised float32 numpy array of shape [N, 768].
        Compatible with the FAISS IndexFlatIP used downstream.
        """
        if self._model is None:
            self.load()
        imgs  = _preprocess(pil_images)
        feats = self._model(imgs)          # (B, 768) float16 MLX
        mx.eval(feats)
        arr   = np.array(feats, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        return (arr / norms).astype(np.float32)

    @property
    def weights_path(self) -> Path:
        return self._weights_path


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_parity(mlx_embedder: MLXCLIPEmbedder,
                    pil_images: list,
                    rtol: float = 1e-2) -> dict:
    """
    Compare MLX vs open_clip embeddings on the same images.
    Returns dict with mean/min cosine similarity and pass/fail verdict.
    """
    import torch, open_clip

    # MLX embeddings
    mlx_embs = mlx_embedder.embed_batch(pil_images)

    # open_clip reference
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14-quickgelu", pretrained="openai"
    )
    model = model.to(device).half().eval()
    batch = torch.stack([preprocess(img) for img in pil_images]).to(device)
    with torch.no_grad():
        ref_embs = model.encode_image(batch.half()).float()
        ref_embs = ref_embs / ref_embs.norm(dim=-1, keepdim=True)
    ref_np = ref_embs.cpu().numpy()

    # Cosine similarity (both L2-normed → dot product = cosine)
    sims = (mlx_embs * ref_np).sum(axis=1)
    return {
        "mean_cosine": float(sims.mean()),
        "min_cosine":  float(sims.min()),
        "pass":        bool(sims.min() > 1.0 - rtol),
        "n":           len(pil_images),
    }
