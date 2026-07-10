"""Latent→CSD projector: read a style descriptor straight off a VAE latent.

WHY THIS EXISTS (BACKLOG SREF-PAIR-VS-BANK, 2026-07-10). The content-shared-pair contrastive needs
`z = proj(x0_pred)` to live in the SAME space as the reference descriptor `CSD(ref)` so the two can be
compared by cosine. `style_stats` (AdaIN mean/std over VAE channels) is a DIFFERENT space and is not
comparable to CSD. The alternatives were (a) VAE-decode `x0_pred` and run the CSD ViT inside the training
graph (~2–3 GB of extra activations, and a second backward through the decoder), or (b) distil a small
latent→CSD projector offline from the `(vae latent, CSD)` pairs we already have cached. This is (b).

CONVENTIONS (get these wrong and it silently trains in the wrong space):
  • Input is the **BN-packed** latent — the space `reconstruct_x0` returns during training (VAE-Q1), NOT
    the raw std≈1.7 latent that `precomputed/vae/` stores. Call `bn_pack` on cached latents first.
  • Input layout is **NCHW** [B, 32, H, W] (what the trainer has); MLX convs are NHWC, transposed inside.
  • **No normalisation layers.** GroupNorm/InstanceNorm divide out per-sample mean/std over (H,W) — which
    is precisely the AdaIN style signal we are trying to read. Normalising here erases the target.
  • Output is L2-normalised [B, 768], matching the L2-normalised CSD vectors in `universe_csd`.
  • Crops must use **EVEN** offsets: BN packing is positional (feature = c*4 + (h%2)*2 + (w%2)), so an odd
    crop offset shifts the packing phase and changes the meaning of every channel.
"""
import os

import mlx.core as mx
import mlx.nn as nn


def load_vae_bn_stats(model_dir: str):
    """VAE BatchNorm running stats, reshaped to the [32, 2, 2] patchify layout. -> (mean, std)."""
    st = mx.load(os.path.join(model_dir, "vae", "diffusion_pytorch_model.safetensors"))
    m = st["bn.running_mean"].astype(mx.float32).reshape(32, 2, 2)
    s = mx.sqrt(st["bn.running_var"].astype(mx.float32).reshape(32, 2, 2) + 1e-4)
    mx.eval(m, s)
    return m, s


def bn_pack(lat: mx.array, m: mx.array, s: mx.array) -> mx.array:
    """Raw VAE latent [B,32,H,W] -> C-inference ("packed") space. Tiles the 2x2 BN stats over the grid."""
    _, _, Lh, Lw = lat.shape
    return ((lat.astype(mx.float32) - mx.tile(m, (1, Lh // 2, Lw // 2)))
            / mx.tile(s, (1, Lh // 2, Lw // 2))).astype(lat.dtype)


def _mean_std_pool(x: mx.array) -> mx.array:
    """Global (mean, std) over spatial dims of an NHWC feature map -> [B, 2C]. The style component."""
    mu = x.mean(axis=(1, 2))
    var = ((x - mu[:, None, None, :]) ** 2).mean(axis=(1, 2))
    return mx.concatenate([mu, mx.sqrt(var + 1e-5)], axis=1)


class LatentCSDProjector(nn.Module):
    """[B,32,H,W] BN-packed latent -> L2-normalised 768-d CSD-space style vector.

    Fully convolutional + global mean/std pooling, so it is resolution-agnostic: a 32x32 latent crop
    (a 256px image region) and a full 64x64 latent (512px) both map into the same descriptor space.
    Style is spatially stationary, so a crop carries the full image's style -- which is what makes
    random-crop augmentation a *correct* invariance to train in, not just a regulariser.
    """

    def __init__(self, in_ch: int = 32, width: int = 128, csd_dim: int = 768, mlp_dim: int = 1024):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, width, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(width, width * 2, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(width * 2, width * 2, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1)
        # pooled = mean/std of conv3 (2*2w) and conv4 (2*4w) -> multi-scale style stats
        self.fc1 = nn.Linear(2 * (width * 2) + 2 * (width * 4), mlp_dim, bias=True)
        self.fc2 = nn.Linear(mlp_dim, csd_dim, bias=True)

    def __call__(self, lat_nchw: mx.array) -> mx.array:
        x = lat_nchw.transpose(0, 2, 3, 1).astype(mx.float32)   # NCHW -> NHWC
        x = nn.silu(self.conv1(x))
        x = nn.silu(self.conv2(x))
        x3 = nn.silu(self.conv3(x))
        x4 = nn.silu(self.conv4(x3))
        h = mx.concatenate([_mean_std_pool(x3), _mean_std_pool(x4)], axis=1)
        z = self.fc2(nn.silu(self.fc1(h)))
        return z / mx.maximum(mx.linalg.norm(z, axis=-1, keepdims=True), 1e-6)


def load_projector(path: str, **kw) -> LatentCSDProjector:
    p = LatentCSDProjector(**kw)
    p.load_weights(path)
    p.eval()
    return p
