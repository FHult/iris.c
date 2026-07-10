"""Guards for the latent→CSD projector (BACKLOG SREF-LATENT-CSD-PROJ).

Pins the three design decisions that are easy to undo by accident:
  1. resolution-agnostic (a 32x32 latent crop = 256px must project into the same space as 64x64),
  2. NO normalisation layers — GroupNorm/InstanceNorm divide out the per-sample spatial mean/std
     that IS the style signal, so the projector must NOT be invariant to a style-changing rescale,
  3. bn_pack reproduces the C-inference ("packed") latent space (VAE-Q1), tiling 2x2 stats.
"""
import os
import sys

import mlx.core as mx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # train/

from ip_adapter.latent_csd import LatentCSDProjector, bn_pack


@pytest.fixture(scope="module")
def proj():
    mx.random.seed(0)
    return LatentCSDProjector(width=16)          # narrow: these are shape/invariance tests


def test_output_is_768d_and_l2_normalised(proj):
    z = proj(mx.random.normal((3, 32, 64, 64)))
    assert z.shape == (3, 768)
    norms = mx.linalg.norm(z, axis=-1)
    assert float(mx.max(mx.abs(norms - 1.0))) < 1e-5


@pytest.mark.parametrize("hw", [16, 32, 64, 96])
def test_resolution_agnostic(proj, hw):
    """Fully convolutional + global pooling: any spatial size maps to the same 768-d space.
    The 32x32 case is the 256px training resolution; 64x64 is the 512px precompute."""
    assert proj(mx.random.normal((2, 32, hw, hw))).shape == (2, 768)


def test_projector_is_not_invariant_to_a_style_rescale(proj):
    """The style signal lives in the per-sample spatial mean/std. A GroupNorm/InstanceNorm layer
    would divide it out and make this test fail — which is exactly why there are none."""
    mx.random.seed(1)
    x = mx.random.normal((4, 32, 32, 32))
    z1, z2 = proj(x), proj(3.0 * x)
    cos = float(mx.mean(mx.sum(z1 * z2, axis=1)))
    assert cos < 0.999, f"projector is scale-invariant (cos {cos:.5f}) -- style stats were normalised away"


def test_projector_responds_monotonically_to_a_mean_shift(proj):
    """Per-channel spatial MEAN is half the AdaIN style descriptor, so a growing shift must move the
    output further and further. Monotonicity is the invariant; the magnitude is not, because at
    RANDOM init the net is near-linear and barely responds (cos 0.9995 at +0.5). The trained
    checkpoint is dramatically more sensitive — cos −0.04 at the same +0.5 — so a fixed threshold
    here would be testing initialisation, not design."""
    mx.random.seed(2)
    x = mx.random.normal((4, 32, 32, 32))
    z0 = proj(x)
    cos = [float(mx.mean(mx.sum(z0 * proj(x + sh), axis=1))) for sh in (0.5, 1.0, 2.0, 4.0)]
    assert all(b < a for a, b in zip(cos, cos[1:])), f"response not monotone in the shift: {cos}"
    assert cos[-1] < 0.95, f"projector nearly ignores a +4.0 mean shift (cos {cos[-1]:.5f})"


def test_bn_pack_is_identity_for_unit_stats():
    x = mx.random.normal((2, 32, 8, 8))
    out = bn_pack(x, mx.zeros((32, 2, 2)), mx.ones((32, 2, 2)))
    assert float(mx.max(mx.abs(out - x))) < 1e-5


def test_bn_pack_applies_the_2x2_positional_stats():
    """BN packing is POSITIONAL: the stat used at (h, w) is stats[c, h%2, w%2]. Getting this wrong
    (or cropping at an odd offset) silently shifts the phase of every channel."""
    m = mx.zeros((32, 2, 2))
    s = mx.ones((32, 2, 2))
    s = mx.concatenate([mx.full((32, 1, 2), 2.0), mx.ones((32, 1, 2))], axis=1)   # std=2 on even rows
    x = mx.ones((1, 32, 4, 4))
    out = bn_pack(x, m, s)
    assert float(out[0, 0, 0, 0]) == pytest.approx(0.5)   # h%2 == 0 -> divided by 2
    assert float(out[0, 0, 1, 0]) == pytest.approx(1.0)   # h%2 == 1 -> divided by 1
    assert float(out[0, 0, 2, 0]) == pytest.approx(0.5)   # tiles with period 2
    assert out.shape == x.shape
