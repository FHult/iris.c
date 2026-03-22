"""
train/tests/test_smoke.py — Environment and import smoke tests.

Verifies all required dependencies are importable and the compute environment
looks sane before any training. Run first to catch missing deps early.
"""

import importlib
import sys
import pytest


# ---------------------------------------------------------------------------
# Core ML dependencies
# ---------------------------------------------------------------------------

def test_import_mlx():
    import mlx.core as mx
    import mlx.nn as nn
    assert mx.__version__
    assert nn.Linear  # nn submodule available


def test_import_numpy():
    import numpy as np
    assert np.__version__


def test_import_safetensors():
    from safetensors.numpy import save_file
    from safetensors import safe_open
    assert save_file and safe_open


def test_import_open_clip():
    import open_clip
    assert open_clip.__version__


def test_import_faiss():
    import faiss
    assert faiss.IndexFlatIP


def test_import_pillow():
    from PIL import Image
    assert Image.open


def test_import_torch_mps():
    """PyTorch with MPS is required for CLIP dedup on Apple Silicon."""
    import torch
    assert torch.backends.mps.is_available(), (
        "MPS not available — CLIP dedup will fall back to CPU (slow)"
    )


def test_turbojpeg_available():
    """TurboJPEG is strongly recommended — 2-4x faster JPEG decode."""
    try:
        from turbojpeg import TurboJPEG
        tj = TurboJPEG()
        assert tj
    except ImportError:
        pytest.skip("TurboJPEG not installed — Pillow fallback will be used (slower)")


# ---------------------------------------------------------------------------
# Our own modules
# ---------------------------------------------------------------------------

def test_import_ip_adapter_modules():
    """All ip_adapter submodules import without error."""
    for mod in ("loss", "model", "dataset", "ema", "utils"):
        m = importlib.import_module(f"ip_adapter.{mod}")
        assert m is not None


def test_import_convert_journeydb():
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        "convert_journeydb",
        os.path.join(os.path.dirname(__file__), "..", "scripts", "convert_journeydb.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(mod.load_annotations)
    assert callable(mod.convert)


def test_import_clip_dedup():
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        "clip_dedup",
        os.path.join(os.path.dirname(__file__), "..", "scripts", "clip_dedup.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(mod.run_embed)
    assert callable(mod.run_dedup)


# ---------------------------------------------------------------------------
# MLX Metal availability
# ---------------------------------------------------------------------------

def test_mlx_metal_available():
    import mlx.core as mx
    # mx.default_device() returns 'gpu' on Apple Silicon with Metal
    dev = mx.default_device()
    assert str(dev) == "Device(gpu, 0)", (
        f"MLX is not using GPU Metal (got {dev}) — training will be very slow"
    )


def test_mlx_basic_op():
    """Sanity: MLX can execute a small GPU operation."""
    import mlx.core as mx
    a = mx.array([1.0, 2.0, 3.0])
    b = mx.array([4.0, 5.0, 6.0])
    c = a + b
    mx.eval(c)
    assert list(c.tolist()) == [5.0, 7.0, 9.0]
