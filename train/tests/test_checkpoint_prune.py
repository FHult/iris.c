"""
train/tests/test_checkpoint_prune.py — checkpoint pruning drops .json sidecars.

CKPT-PRUNE-1: when keep-N pruning deletes an old step_*.safetensors checkpoint,
the companion step_*.json lineage sidecar must be deleted too, otherwise every
pruned checkpoint orphans a sidecar that pipeline_doctor flags as an
"incomplete write?" WARNING.

Marked requires_mps because importing train_ip_adapter compiles the loss Metal
kernel at import time (pulls in MLX/Metal); the pruning logic itself is pure I/O.
"""

from __future__ import annotations

import os
import sys

import pytest

pytestmark = pytest.mark.requires_mps

# conftest.py puts train/ on sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import train_ip_adapter as tia


def _touch(path: str) -> None:
    with open(path, "w") as f:
        f.write("x")


def _make_checkpoint(directory: str, step: int, sidecar: bool = True) -> tuple[str, str]:
    ckpt = os.path.join(directory, f"step_{step:07d}.safetensors")
    side = os.path.join(directory, f"step_{step:07d}.json")
    _touch(ckpt)
    if sidecar:
        _touch(side)
    return ckpt, side


def test_purge_deletes_companion_sidecars(tmp_path):
    d = str(tmp_path)
    pairs = {step: _make_checkpoint(d, step) for step in (100, 200, 300, 400, 500)}

    tia._purge_old_checkpoints(d, keep_last_n=2)

    # Kept: the two most recent checkpoints and their sidecars.
    for step in (400, 500):
        ckpt, side = pairs[step]
        assert os.path.exists(ckpt), f"step {step} checkpoint should be kept"
        assert os.path.exists(side), f"step {step} sidecar should be kept"

    # Pruned: older checkpoints AND their sidecars (the CKPT-PRUNE-1 fix).
    for step in (100, 200, 300):
        ckpt, side = pairs[step]
        assert not os.path.exists(ckpt), f"step {step} checkpoint should be pruned"
        assert not os.path.exists(side), f"step {step} sidecar should be pruned too"


def test_purge_tolerates_missing_sidecar(tmp_path):
    """A checkpoint without a sidecar must prune cleanly (no FileNotFoundError)."""
    d = str(tmp_path)
    _make_checkpoint(d, 100, sidecar=False)
    _make_checkpoint(d, 200, sidecar=False)
    _make_checkpoint(d, 300, sidecar=False)

    tia._purge_old_checkpoints(d, keep_last_n=1)  # must not raise

    assert not os.path.exists(os.path.join(d, "step_0000100.safetensors"))
    assert os.path.exists(os.path.join(d, "step_0000300.safetensors"))


def test_purge_keeps_all_when_under_limit(tmp_path):
    d = str(tmp_path)
    pairs = {step: _make_checkpoint(d, step) for step in (100, 200)}

    tia._purge_old_checkpoints(d, keep_last_n=5)

    for step, (ckpt, side) in pairs.items():
        assert os.path.exists(ckpt)
        assert os.path.exists(side)
