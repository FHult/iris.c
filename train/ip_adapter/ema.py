"""
train/ip_adapter/ema.py — Exponential Moving Average of adapter weights.

Uses mx.tree_map for parameter averaging (matches plans/ip-adapter-training.md §3.11).
Updated every N steps (default 10) to save ~23 minutes over 105K steps.
At decay=0.9999 the EMA moves <0.1% per step; every-10-step update is
indistinguishable in output quality.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as mx_utils


def update_ema(
    ema_params: dict,
    model: nn.Module,
    decay: float = 0.9999,
) -> dict:
    """
    One EMA update: ema = decay * ema + (1 - decay) * model.parameters()

    Returns updated ema_params dict (does not mutate in place).
    Based on plans/ip-adapter-training.md §3.11.
    """
    return mx_utils.tree_map(
        lambda e, m: decay * e + (1.0 - decay) * m,
        ema_params,
        model.parameters(),
    )


def save_ema(ema_params: dict, path: str) -> None:
    """Save EMA parameter dict as safetensors."""
    import numpy as np
    from safetensors.numpy import save_file

    mx.eval(ema_params)
    flat = _flatten(ema_params)
    weights = {k: np.array(v) for k, v in flat}
    save_file(weights, path)


def load_ema(path: str) -> dict:
    """Load EMA parameters from safetensors into a flat dict."""
    from safetensors import safe_open

    params = {}
    with safe_open(path, framework="numpy") as f:
        for k in f.keys():
            params[k] = mx.array(f.get_tensor(k))
    return params


def _flatten(params, prefix=""):
    """Flatten nested dict to [(key, mx.array)] pairs."""
    items = []
    if isinstance(params, dict):
        for k, v in params.items():
            items.extend(_flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(params, list):
        for i, v in enumerate(params):
            items.extend(_flatten(v, f"{prefix}.{i}" if prefix else str(i)))
    elif isinstance(params, mx.array):
        items.append((prefix, params))
    return items
