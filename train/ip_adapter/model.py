"""
train/ip_adapter/model.py — IP-Adapter architecture for Flux Klein 4B.

Components:
  PerceiverResampler  — compresses SigLIP [B, 729, 1152] → [B, 128, 3072]
                        Uses nn.MultiHeadAttention over learned query tokens.
  IPAdapterKlein      — wraps Perceiver + 25 per-block K/V projections

Key design decisions (from plans/ip-adapter-training.md §3.1 / §3.2):
  - Perceiver uses nn.MultiHeadAttention with key_input_dims=siglip_dim (1152)
  - All 25 to_k_ip / to_v_ip matrices stacked as [25, hidden, hidden] so
    get_kv_all() computes all blocks in 2 einsum dispatches (not 50 GEMMs).
  - einsum: 'btd,nde->bnte' — B batch, T tokens(128), D hidden(3072), N blocks(25)
  - ip_scale per block init to 1.0; trained to calibrate block contribution.
  - Style-only inference: caller sets ip_scale[0:5] = 0.0 (double-stream blocks).
"""

import mlx.core as mx
import mlx.nn as nn


class PerceiverResampler(nn.Module):
    """
    Compresses 729 SigLIP patch tokens → 128 image conditioning tokens at
    hidden_size=3072 using learned query vectors and cross-attention.

    Architecture (plans/ip-adapter-training.md §3.1):
      query_tokens: learnable [128, 3072] initialised to small normal
      cross_attn:   nn.MultiHeadAttention(3072, heads=24, key_input_dims=1152)
      norm:         LayerNorm(3072)
    """

    def __init__(
        self,
        hidden_dim: int = 3072,
        num_heads: int = 24,
        num_queries: int = 128,
        siglip_dim: int = 1152,
    ):
        super().__init__()
        self.query_tokens = mx.random.normal((num_queries, hidden_dim)) * 0.02
        self.cross_attn = nn.MultiHeadAttention(
            dims=hidden_dim,
            num_heads=num_heads,
            key_input_dims=siglip_dim,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def __call__(self, siglip_features: mx.array) -> mx.array:
        # siglip_features: [B, 729, 1152]
        B = siglip_features.shape[0]
        q = mx.broadcast_to(
            self.query_tokens[None], (B,) + self.query_tokens.shape
        )  # [B, 128, 3072]
        out = self.cross_attn(q, siglip_features, siglip_features)
        return self.norm(out)  # [B, 128, 3072]


class IPAdapterKlein(nn.Module):
    """
    IP-Adapter companion model for Flux Klein 4B.

    Trainable components (~522M parameters, ~1 GB BF16):
      - PerceiverResampler (~50M params)
      - 25 × to_k_ip [3072, 3072] (~236M params)
      - 25 × to_v_ip [3072, 3072] (~236M params)
      - 25 × ip_scale scalar (~25 params)

    All Flux Klein 4B weights are FROZEN. Only the adapter trains.

    Inference modes (controlled by zeroing ip_scale slots):
      - Style+content: all 25 blocks active
      - Style-only:    ip_scale[0:5] = 0.0 (skip 5 double-stream blocks)
    """

    def __init__(
        self,
        num_blocks: int = 25,
        hidden_dim: int = 3072,
        num_image_tokens: int = 128,
        siglip_dim: int = 1152,
        perceiver_heads: int = 24,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.num_image_tokens = num_image_tokens

        # Image projection: SigLIP patch tokens → 128 conditioning tokens
        self.image_proj = PerceiverResampler(
            hidden_dim=hidden_dim,
            num_heads=perceiver_heads,
            num_queries=num_image_tokens,
            siglip_dim=siglip_dim,
        )

        # Per-block K/V weights stacked for 2-dispatch batched einsum.
        # Shape: [num_blocks, hidden_dim, hidden_dim]
        # Initialised with small normal for symmetry breaking.
        scale = hidden_dim ** -0.5
        self.to_k_ip_stacked = mx.random.normal(
            (num_blocks, hidden_dim, hidden_dim)
        ) * scale
        self.to_v_ip_stacked = mx.random.normal(
            (num_blocks, hidden_dim, hidden_dim)
        ) * scale

        # Per-block learnable scale: start at 1.0
        self.scale = mx.ones((num_blocks,))

    def get_image_embeds(self, siglip_features: mx.array) -> mx.array:
        """
        siglip_features: [B, 729, 1152]
        Returns image_tokens: [B, 128, 3072]
        """
        return self.image_proj(siglip_features)

    def get_kv_all(self, ip_embeds: mx.array):
        """
        Compute K and V for all blocks in 2 Metal dispatches.

        ip_embeds: [B, 128, 3072]  (output of get_image_embeds)
        Returns:
          k: [B, num_blocks, 128, 3072]
          v: [B, num_blocks, 128, 3072]

        Einsum 'btd,nde->bnte' — one large batched GEMM per K and V,
        replacing 25 separate GEMMs each.
        From plans/ip-adapter-training.md §3.1.
        """
        k = mx.einsum("btd,nde->bnte", ip_embeds, self.to_k_ip_stacked)
        v = mx.einsum("btd,nde->bnte", ip_embeds, self.to_v_ip_stacked)
        return k, v

    def load_resampler_weights(self, weights: dict) -> None:
        """
        Load Perceiver Resampler weights transferred from InstantX
        Flux.1-dev IP-Adapter. Both models share hidden_size=3072.

        weights: dict of {key: mx.array} with keys stripped of "image_proj." prefix.
        Based on plans/ip-adapter-training.md §3.3.
        """
        self.image_proj.update(weights)
        transferred = len(weights)
        print(f"Warmstart: loaded {transferred} Perceiver Resampler weights")

    @classmethod
    def from_pretrained_warmstart(cls, instantx_path: str, **kwargs) -> "IPAdapterKlein":
        """
        Create model and warmstart the Perceiver Resampler from InstantX
        Flux.1-dev IP-Adapter weights (5.3 GB file).

        instantx_path: local dir containing ip_adapter.safetensors
        """
        import os
        from safetensors import safe_open

        model = cls(**kwargs)

        ckpt_file = os.path.join(instantx_path, "ip_adapter.safetensors")
        if not os.path.exists(ckpt_file):
            raise FileNotFoundError(f"Warmstart file not found: {ckpt_file}")

        with safe_open(ckpt_file, framework="numpy") as f:
            keys = [k for k in f.keys() if k.startswith("image_proj.")]
            weights = {
                k.removeprefix("image_proj."): mx.array(f.get_tensor(k))
                for k in keys
            }

        model.load_resampler_weights(weights)
        return model
