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
  - Style-only inference: use effective_scale(style_only=True) which returns
    self.scale with indices 0..num_double_blocks-1 zeroed out (non-mutating).
  - sref_strength: global scalar multiplier applied on top of ip_scale at
    inference; does not affect training (always 1.0 during training).
"""

import mlx.core as mx
import mlx.nn as nn


class PerceiverResampler(nn.Module):
    """
    Compresses 729 SigLIP patch tokens → 128 image conditioning tokens at
    hidden_size=3072 using learned query vectors and cross-attention.

    Architecture (plans/ip-adapter-training.md §3.1):
      query_tokens: learnable [128, 3072] initialised to small normal
      cross_attn:   nn.MultiHeadAttention(3072, heads=perceiver_heads, key_input_dims=1152)
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
            bias=False,  # explicit: C inference loader expects no bias weights
        )
        self.norm = nn.LayerNorm(hidden_dim)
        # Input normalization on the SigLIP features (IP-ADAPTER-INFER-1 root cause).
        # SigLIP has a few massive-activation DIMENSIONS that otherwise dominate the
        # cross-attention dot products, so every query attends to the same token →
        # the perceiver pools → constant injection → the grid. We standardize each
        # feature DIMENSION across the token sequence (per-image), then apply a learned
        # affine. NOTE: this is token-axis standardization, NOT a per-token LayerNorm —
        # the diagnostic showed per-token LayerNorm does not help; per-dim across tokens
        # lifts ip_embeds cross-token ratio 0.0068→1.6. Matched in iris_ip_adapter_perceive.
        self.in_gamma = mx.ones((siglip_dim,))
        self.in_beta = mx.zeros((siglip_dim,))

    def _norm_input(self, f: mx.array) -> mx.array:
        # f: [B, n_tokens, siglip_dim] — standardize each dim across the token axis.
        mean = f.mean(axis=1, keepdims=True)
        var = f.var(axis=1, keepdims=True)
        f = (f - mean) * mx.rsqrt(var + 1e-5)
        return f * self.in_gamma + self.in_beta

    def __call__(self, siglip_features: mx.array) -> mx.array:
        # siglip_features: [B, 729, 1152]
        B = siglip_features.shape[0]
        f = self._norm_input(siglip_features)
        q = mx.broadcast_to(
            self.query_tokens[None], (B,) + self.query_tokens.shape
        )  # [B, 128, 3072]
        out = self.cross_attn(q, f, f)
        return self.norm(out)  # [B, 128, 3072]


class CSDImageProj(nn.Module):
    """
    Content-invariant conditioning (SREF-LEAK-1 structural fix): a single CSD style
    embedding [B, csd_dim] → [B, num_queries, hidden_dim] image tokens.

    CSD's style vector is content-invariant BY CONSTRUCTION (the style head), unlike SigLIP's
    729 content-laden patch tokens — so the content can't leak (it isn't in the signal). CSD is
    ONE vector, not a set, so there's nothing to cross-attend/resample; instead FiLM-modulate
    128 learned query tokens by the (L2-normalised) CSD vector. The 128 distinct queries carry
    the diversity, the CSD vector the style → cannot pool/collapse the way the SigLIP perceiver
    did (IP-ADAPTER-INFER-1). Deliberately simple (Linear + FiLM + LayerNorm) to mirror in C
    inference (iris_ip_adapter.c) without a new attention kernel.
    """

    def __init__(self, hidden_dim: int = 3072, num_queries: int = 128, csd_dim: int = 768):
        super().__init__()
        self.query_tokens = mx.random.normal((num_queries, hidden_dim)) * 0.02
        self.film = nn.Linear(csd_dim, 2 * hidden_dim, bias=True)  # → (scale, shift)
        # FiLM-ZERO init (adaLN-zero style): zero the film weight AND bias so scale=shift=0 at
        # init → tokens = query_tokens (a clean, stable identity start), then the module LEARNS
        # modulation. With the default nn.Linear init the random scale/shift made the loss
        # diverge (smoke: 2.5→129). Gradients still flow through W from step 0, so it trains.
        self.film.weight = mx.zeros_like(self.film.weight)
        self.film.bias = mx.zeros_like(self.film.bias)
        self.norm = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries

    def __call__(self, csd: mx.array) -> mx.array:
        # csd: [B, csd_dim] (L2-normalised CSD style embedding)
        if csd.ndim == 3:  # tolerate [B, 1, csd_dim]
            csd = csd.reshape(csd.shape[0], -1)
        scale, shift = mx.split(self.film(csd), 2, axis=-1)  # each [B, hidden]
        q = mx.broadcast_to(self.query_tokens[None],
                            (csd.shape[0], self.num_queries, self.hidden_dim))
        out = q * (1.0 + scale[:, None, :]) + shift[:, None, :]
        return self.norm(out)  # [B, num_queries, hidden_dim]


class IPAdapterKlein(nn.Module):
    """
    IP-Adapter companion model for Flux Klein 4B.

    Trainable components (~522M parameters, ~1 GB BF16):
      - PerceiverResampler (~50M params)
      - 25 × to_k_ip [3072, 3072] (~236M params)
      - 25 × to_v_ip [3072, 3072] (~236M params)
      - 25 × ip_scale scalar (~25 params)

    All Flux Klein 4B weights are FROZEN. Only the adapter trains.

    Inference modes (use effective_scale() to get the per-block scale array):
      - Style+content: effective_scale()                  — all 25 blocks active
      - Style-only:    effective_scale(style_only=True)   — double-stream zeroed
      - Strength knob: effective_scale(sref_strength=0.7) — global multiplier
    """

    def __init__(
        self,
        num_blocks: int = 25,
        hidden_dim: int = 3072,
        num_image_tokens: int = 128,
        siglip_dim: int = 1152,
        perceiver_heads: int = 24,
        num_double_blocks: int = 5,
        cond_mode: str = "siglip",   # "siglip" | "csd" | "hybrid"
        csd_dim: int = 768,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.num_image_tokens = num_image_tokens
        self.num_double_blocks = num_double_blocks
        self.cond_mode = cond_mode
        self.csd_dim = csd_dim

        # Image projection: conditioning features → num_image_tokens conditioning tokens.
        # SigLIP path = PerceiverResampler (cross-attn over 729 patch tokens).
        # CSD path = CSDImageProj (FiLM-modulated queries; content-invariant, SREF-LEAK-2).
        # hybrid (SREF-COMBINE-1) = BOTH: PerceiverResampler over SigLIP (local detail/texture) +
        #   CSDImageProj over the content-invariant CSD vector (global style direction), each
        #   producing num_image_tokens//2 tokens, concatenated. The two parity-proven modules are
        #   reused verbatim; get_image_embeds slices a single packed [B,730,1152] feature so the
        #   trainer threads ONE conditioning array (rows 0..728 = SigLIP, row 729 = CSD padded).
        if cond_mode == "csd":
            self.image_proj = CSDImageProj(
                hidden_dim=hidden_dim, num_queries=num_image_tokens, csd_dim=csd_dim)
        elif cond_mode == "hybrid":
            half = num_image_tokens // 2
            self.image_proj = PerceiverResampler(
                hidden_dim=hidden_dim, num_heads=perceiver_heads,
                num_queries=half, siglip_dim=siglip_dim)
            self.csd_proj = CSDImageProj(
                hidden_dim=hidden_dim, num_queries=half, csd_dim=csd_dim)
        else:
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

    def get_image_embeds(self, cond_features: mx.array) -> mx.array:
        """
        cond_features:
          - "siglip": SigLIP [B, 729, 1152]
          - "csd":    CSD [B, csd_dim]
          - "hybrid": packed [B, 730, 1152] — rows 0..728 = SigLIP, row 729 = CSD vector
                      zero-padded to 1152 (the carrier; sliced back to csd_dim here).
        Returns image_tokens: [B, num_image_tokens, hidden_dim]
          (siglip/csd → [B,128,3072]; hybrid → [B,256,3072]).
        """
        if self.cond_mode == "hybrid":
            n = cond_features.shape[1] - 1                 # SigLIP token count (729)
            sig = cond_features[:, :n, :]                  # [B, 729, 1152]
            csd = cond_features[:, n, : self.csd_dim]      # [B, csd_dim]
            sig_tok = self.image_proj(sig)                 # [B, 128, 3072]
            csd_tok = self.csd_proj(csd)                   # [B, 128, 3072]
            return mx.concatenate([sig_tok, csd_tok], axis=1)  # [B, 256, 3072]
        return self.image_proj(cond_features)

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

    def effective_scale(
        self,
        style_only: bool = False,
        sref_strength: float = 1.0,
    ) -> mx.array:
        """
        Return the per-block scale to pass to _flux_forward_with_ip.

        Non-mutating — does not modify self.scale. Always safe to call from
        training code; with default args returns self.scale unchanged.

        style_only=True:
            Zeros double-stream block scales (indices 0..num_double_blocks-1).
            Single-stream blocks (indices num_double_blocks..num_blocks-1) are
            unaffected. This removes layout/content injection and keeps only
            the style signal carried by single-stream blocks.

        sref_strength:
            Global scalar multiplier applied after style_only masking.
            1.0 = no change. 0.0 = adapter fully disabled. Values > 1.0
            strengthen conditioning beyond what was learned during training.

        Returns: [num_blocks] float array, same dtype as self.scale.
        """
        scale = self.scale
        if style_only:
            nd = self.num_double_blocks
            mask = mx.concatenate([
                mx.zeros((nd,), dtype=scale.dtype),
                mx.ones((self.num_blocks - nd,), dtype=scale.dtype),
            ])
            scale = scale * mask
        if sref_strength != 1.0:
            scale = scale * sref_strength
        return scale

    def inject(
        self,
        img_q: mx.array,
        k_ip: mx.array,
        v_ip: mx.array,
        block_idx: int,
    ) -> mx.array:
        """
        Compute IP-Adapter attention output for one block using batched SDPA.

        Uses mx.fast.scaled_dot_product_attention (Flash Attention, Metal-backed)
        instead of manual softmax attention — 25→1 Metal SDPA call per block,
        saving ~0.2s/step vs manual softmax. (§3.5, optimisations table)

        img_q:     [B, num_heads, img_seq, head_dim] — image Q from the block
        k_ip:      [B, 128, 3072]                    — slice of get_kv_all output
        v_ip:      [B, 128, 3072]                    — slice of get_kv_all output
        block_idx: which ip_scale scalar to apply

        Returns:
          ip_contribution: [B, img_seq, 3072] scaled by self.scale[block_idx]
        """
        _, _, _, head_dim = img_q.shape  # derive from Q: correct for any Flux variant
        scale = float(head_dim ** -0.5)

        # Reshape K/V to [B, num_heads, 128, head_dim] for batched SDPA
        B, T, D = k_ip.shape
        num_heads = D // head_dim
        k = k_ip.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v_ip.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)

        # mx.fast.scaled_dot_product_attention: Flash Attention, 1 Metal dispatch
        # img_q: [B, num_heads, img_seq, head_dim]
        # k, v:  [B, num_heads, 128, head_dim]
        attn = mx.fast.scaled_dot_product_attention(img_q, k, v, scale=scale)
        # attn: [B, num_heads, img_seq, head_dim] → [B, img_seq, D]
        B2, H, S, hd = attn.shape
        attn_out = attn.transpose(0, 2, 1, 3).reshape(B2, S, H * hd)

        return self.scale[block_idx] * attn_out

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
