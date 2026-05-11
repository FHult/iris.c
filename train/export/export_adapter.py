#!/usr/bin/env python3
"""
train/export/export_adapter.py — Export IP-Adapter checkpoint to iris.c bundle.

Converts a trained checkpoint (step_*.safetensors from train_ip_adapter.py) into
a compact, C-loadable pair:

    adapter_weights.safetensors   — quantized weight tensors (mmap-ready)
    adapter_meta.json             — dimensions + quant info for the C loader

The bundle is consumed by iris_ip_adapter.h / iris_ip_adapter.c (iris.c v2.7).

Usage
-----
    # BF16 export (default, recommended):
    python train/export/export_adapter.py \\
        --checkpoint /Volumes/2TBSSD/checkpoints/stage1/step_050000.safetensors \\
        --output     /tmp/iris_ip_adapter/

    # Use EMA weights (recommended for inference quality):
    python train/export/export_adapter.py --checkpoint ... --use-ema

    # INT8 symmetric quantisation (~2× smaller, requires C-side INT8 kernel):
    python train/export/export_adapter.py --checkpoint ... --quant int8

    # Style-only mode (zero ip_scale for double-stream blocks 0–4):
    python train/export/export_adapter.py --checkpoint ... --style-only

    # Validate output bundle after writing:
    python train/export/export_adapter.py --checkpoint ... --validate

Checkpoint Key Map (training format → export format)
------------------------------------------------------
    image_proj.query_tokens                → perceiver.query_tokens   [Q, D]
    image_proj.cross_attn.query_proj.weight → perceiver.query_proj    [D, D]
    image_proj.cross_attn.key_proj.weight   → perceiver.key_proj      [D, S]
    image_proj.cross_attn.value_proj.weight → perceiver.value_proj    [D, S]
    image_proj.cross_attn.out_proj.weight   → perceiver.out_proj      [D, D]
    image_proj.norm.weight                 → perceiver.norm_weight    [D]
    image_proj.norm.bias                   → perceiver.norm_bias      [D]
    to_k_ip_stacked                        → ip_k_stacked             [N, D, D]
    to_v_ip_stacked                        → ip_v_stacked             [N, D, D]
    scale                                  → ip_scale                 [N]

Quantisation Modes
------------------
    bfloat16  BF16 storage (default). Natively supported by C backend.
    float16   F16 storage. Equivalent size to BF16; wider compatibility.
    int8      Symmetric per-row INT8 + F32 per-row scale tensors.
              Scale stored as <name>.scale alongside each quantised tensor.
              Dequant: x_f32[i,:] = q_i8[i,:] * scale[i]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPORT_VERSION = 1
ADAPTER_TYPE   = "ip_adapter_klein"

# Mapping: flat checkpoint key → export tensor name
_KEY_MAP = {
    "image_proj.query_tokens":                 "perceiver.query_tokens",
    "image_proj.cross_attn.query_proj.weight": "perceiver.query_proj",
    "image_proj.cross_attn.key_proj.weight":   "perceiver.key_proj",
    "image_proj.cross_attn.value_proj.weight":  "perceiver.value_proj",
    "image_proj.cross_attn.out_proj.weight":   "perceiver.out_proj",
    "image_proj.norm.weight":                  "perceiver.norm_weight",
    "image_proj.norm.bias":                    "perceiver.norm_bias",
    "to_k_ip_stacked":                         "ip_k_stacked",
    "to_v_ip_stacked":                         "ip_v_stacked",
    "scale":                                   "ip_scale",
}

# Which tensors carry large linear weights (quantised in int8 mode; others kept F32)
_QUANTISED_TENSORS = {
    "perceiver.query_proj",
    "perceiver.key_proj",
    "perceiver.value_proj",
    "perceiver.out_proj",
    "ip_k_stacked",
    "ip_v_stacked",
}

# Tensors that are always kept in F32 (norms, scales, small biases)
_ALWAYS_F32 = {
    "perceiver.norm_weight",
    "perceiver.norm_bias",
    "ip_scale",
}

# Small tensors kept in F32 even for float16/bfloat16 export
_F32_OVERRIDE = {"perceiver.norm_weight", "perceiver.norm_bias", "ip_scale"}


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_safetensors_as_numpy(path: str, prefix: str = "") -> dict[str, np.ndarray]:
    """
    Load all tensors from a safetensors file as float32 numpy arrays.
    If prefix is non-empty (e.g. "ema."), only keys starting with that prefix
    are loaded; the prefix is stripped from the returned keys.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        print("Error: safetensors not installed. Run: pip install safetensors", file=sys.stderr)
        sys.exit(1)

    out: dict[str, np.ndarray] = {}
    with safe_open(path, framework="numpy") as f:
        for k in f.keys():
            if prefix and not k.startswith(prefix):
                continue
            stripped = k[len(prefix):] if prefix else k
            arr = f.get_tensor(stripped if not prefix else k)
            # Convert to float32 for manipulation; re-quantise later
            if arr.dtype in (np.float16,):
                arr = arr.astype(np.float32)
            elif arr.dtype == object:
                continue  # skip metadata blobs
            elif arr.dtype.kind not in ("f", "u", "i"):
                continue
            else:
                if arr.dtype != np.float32:
                    try:
                        arr = arr.astype(np.float32)
                    except Exception:
                        continue
            out[stripped] = arr
    return out


def _load_bfloat16_as_float32(path: str, prefix: str = "") -> dict[str, np.ndarray]:
    """
    Load safetensors using MLX (handles BF16 natively) and convert to F32 numpy.
    Falls back to pure-numpy path (which converts BF16 via uint16 reinterpretation).
    """
    try:
        import mlx.core as mx
        out: dict[str, np.ndarray] = {}
        sf = mx.load(path)
        for k, arr in sf.items():
            if prefix and not k.startswith(prefix):
                continue
            stripped = k[len(prefix):] if prefix else k
            out[stripped] = np.array(arr.astype(mx.float32))
        return out
    except ImportError:
        pass

    # Fallback: manual BF16→F32 via uint16 reinterpretation
    try:
        from safetensors import safe_open
    except ImportError:
        print("Error: neither mlx nor safetensors installed.", file=sys.stderr)
        sys.exit(1)

    out = {}
    with safe_open(path, framework="numpy") as f:
        for k in f.keys():
            if prefix and not k.startswith(prefix):
                continue
            stripped = k[len(prefix):] if prefix else k
            arr = f.get_tensor(stripped if not prefix else k)
            if str(arr.dtype) in ("uint16", "bfloat16"):
                # Reinterpret uint16 as BF16 by shifting to float32
                u16 = arr.view(np.uint16)
                f32 = np.zeros(u16.shape, dtype=np.float32)
                # BF16 occupies the high 16 bits of a float32
                f32.view(np.uint32)[:] = u16.astype(np.uint32) << 16
                out[stripped] = f32
            else:
                out[stripped] = arr.astype(np.float32)
    return out


def load_checkpoint(path: str, use_ema: bool) -> dict[str, np.ndarray]:
    """
    Load adapter weights from a training checkpoint as float32 numpy arrays.

    Handles both:
      - Direct adapter keys (e.g. "image_proj.query_tokens")
      - EMA-prefixed keys   (e.g. "ema.image_proj.query_tokens")

    Returns flat dict keyed by the UN-prefixed adapter key names.
    """
    print(f"Loading checkpoint: {path}")
    raw = _load_bfloat16_as_float32(path)
    all_keys = set(raw.keys())

    # Detect whether EMA keys are present
    has_live = any(k in _KEY_MAP for k in all_keys)
    has_ema  = any(k.startswith("ema.") and k[4:] in _KEY_MAP for k in all_keys)

    if use_ema:
        if not has_ema:
            raise ValueError(
                f"--use-ema requested but no 'ema.*' keys found in {path}.\n"
                f"  Available prefixes: {sorted({k.split('.')[0] for k in all_keys})[:10]}"
            )
        prefix = "ema."
    else:
        if not has_live:
            if has_ema:
                print("  NOTE: No live adapter keys found; using EMA keys automatically.")
                prefix = "ema."
            else:
                raise ValueError(
                    f"No recognisable adapter keys found in {path}.\n"
                    f"  Expected keys like: {list(_KEY_MAP)[:3]}\n"
                    f"  Found keys (first 10): {sorted(all_keys)[:10]}"
                )
        else:
            prefix = ""

    weights: dict[str, np.ndarray] = {}
    missing = []
    for ckpt_key, export_key in _KEY_MAP.items():
        lookup = prefix + ckpt_key
        if lookup in raw:
            weights[export_key] = raw[lookup]
        else:
            missing.append(ckpt_key)

    if missing:
        raise ValueError(
            f"Checkpoint is missing expected keys: {missing}\n"
            f"  (prefix={prefix!r}, total keys={len(raw)})"
        )

    src = "EMA" if prefix == "ema." else "live"
    print(f"  Loaded {len(weights)} tensors ({src} weights).")
    return weights


# ---------------------------------------------------------------------------
# Model dimension inference
# ---------------------------------------------------------------------------

def _infer_dims(weights: dict[str, np.ndarray]) -> dict[str, int]:
    """Infer model dimensions from tensor shapes."""
    qt = weights["perceiver.query_tokens"]  # [num_image_tokens, hidden_dim]
    kp = weights["perceiver.key_proj"]      # [hidden_dim, siglip_dim]
    ks = weights["ip_k_stacked"]            # [num_blocks, hidden_dim, hidden_dim]
    sc = weights["ip_scale"]               # [num_blocks]

    hidden_dim       = qt.shape[1]
    num_image_tokens = qt.shape[0]
    siglip_dim       = kp.shape[1]
    num_blocks       = ks.shape[0]
    # head_dim: standard Flux Klein 4B value; can't infer without num_heads
    num_heads        = 24 if hidden_dim == 3072 else (32 if hidden_dim == 4096 else 0)
    head_dim         = hidden_dim // num_heads if num_heads else 0

    # Infer double/single split from typical Flux Klein 4B layout
    num_double_blocks = 5  if num_blocks == 25 else num_blocks // 5
    num_single_blocks = num_blocks - num_double_blocks

    return {
        "num_blocks":        num_blocks,
        "num_double_blocks": num_double_blocks,
        "num_single_blocks": num_single_blocks,
        "hidden_dim":        hidden_dim,
        "head_dim":          head_dim,
        "num_heads":         num_heads,
        "num_image_tokens":  num_image_tokens,
        "siglip_dim":        siglip_dim,
    }


# ---------------------------------------------------------------------------
# Quantisation
# ---------------------------------------------------------------------------

def _quantise_int8_tensor(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Symmetric per-row INT8 quantisation.

    For 2D [M, N]:   scale[i]   = max(|arr[i,:]|) / 127
                     q[i,:]     = round(arr[i,:] / scale[i]).clip(-127, 127)
    For 3D [B,M,N]:  flatten to [B*M, N], quantise, reshape back.

    Returns:
      q:     INT8 array, same shape as arr
      scale: FLOAT32 array, shape [...leading dims..., leading_dim] per-row scales
    """
    orig_shape = arr.shape
    if arr.ndim == 3:
        B, M, N = arr.shape
        flat = arr.reshape(B * M, N)
    elif arr.ndim == 2:
        flat = arr
    else:
        raise ValueError(f"Expected 2D or 3D tensor for int8 quant, got shape {arr.shape}")

    rows, cols = flat.shape
    max_abs = np.abs(flat).max(axis=1, keepdims=True)  # [rows, 1]
    max_abs = np.where(max_abs == 0, 1.0, max_abs)     # avoid div-by-zero
    scale   = (max_abs / 127.0).astype(np.float32)     # [rows, 1]

    q = np.round(flat / scale).clip(-127, 127).astype(np.int8)  # [rows, cols]

    scale_flat = scale.reshape(rows).astype(np.float32)  # [rows]
    if arr.ndim == 3:
        return q.reshape(B, M, N), scale_flat.reshape(B, M)
    return q, scale_flat


def apply_quant(
    weights: dict[str, np.ndarray],
    quant: str,
) -> dict[str, np.ndarray]:
    """
    Apply quantisation to the weight dict.

    Returns a new dict with:
      - bfloat16: all tensors cast to BF16 (except ALWAYS_F32)
      - float16:  all tensors cast to F16  (except ALWAYS_F32)
      - int8:     large linear tensors as INT8 + companion <name>.scale F32;
                  small tensors (norms, biases, ip_scale) kept F32
    """
    if quant not in ("bfloat16", "float16", "int8"):
        raise ValueError(f"Unknown quant mode {quant!r}; choose: bfloat16, float16, int8")

    out: dict[str, np.ndarray] = {}

    for name, arr in weights.items():
        if name in _ALWAYS_F32 or name in _F32_OVERRIDE:
            out[name] = arr.astype(np.float32)
            continue

        if quant == "bfloat16":
            # Store as BF16 — MLX save_safetensors writes correct BF16 dtype header
            out[name] = _to_bfloat16_numpy(arr)
        elif quant == "float16":
            out[name] = arr.astype(np.float16)
        elif quant == "int8":
            if name in _QUANTISED_TENSORS:
                q, scale = _quantise_int8_tensor(arr)
                out[name] = q
                out[f"{name}.scale"] = scale
            else:
                out[name] = arr.astype(np.float32)

    return out


def _to_bfloat16_numpy(arr: np.ndarray) -> np.ndarray:
    """
    Convert float32 numpy array to bfloat16 representation stored as uint16.
    MLX's save_safetensors stores these with dtype tag "BF16".
    """
    try:
        import mlx.core as mx
        return np.array(mx.array(arr).astype(mx.bfloat16).view(mx.uint16))
    except ImportError:
        pass
    # Fallback: truncate float32 to upper 16 bits (round-to-zero BF16)
    f32 = arr.astype(np.float32)
    u32 = f32.view(np.uint32)
    # Round-to-nearest-even: add rounding_bias to the low 16 bits
    rounding_bias = (u32 >> 16) & 1  # add 1 to bit 16 if bit 16 is 1 (round to even)
    u32_rounded   = u32 + 0x7FFF + rounding_bias
    return (u32_rounded >> 16).astype(np.uint16)


# ---------------------------------------------------------------------------
# Bundle write
# ---------------------------------------------------------------------------

def _numpy_to_mlx(arr: np.ndarray):
    """Convert numpy array to MLX array for save_safetensors."""
    import mlx.core as mx
    if arr.dtype == np.uint16:
        # BF16 reinterpreted as uint16 — tell MLX it's bfloat16
        return mx.array(arr).view(mx.bfloat16)
    if arr.dtype == np.float16:
        return mx.array(arr).astype(mx.float16)
    if arr.dtype == np.float32:
        return mx.array(arr)
    if arr.dtype == np.int8:
        return mx.array(arr)
    return mx.array(arr.astype(np.float32))


def write_bundle(
    weights:   dict[str, np.ndarray],
    meta:      dict[str, Any],
    out_dir:   str,
) -> tuple[str, str]:
    """
    Write adapter_weights.safetensors + adapter_meta.json to out_dir.
    Returns (weights_path, meta_path).
    """
    import mlx.core as mx

    os.makedirs(out_dir, exist_ok=True)
    weights_path = os.path.join(out_dir, "adapter_weights.safetensors")
    meta_path    = os.path.join(out_dir, "adapter_meta.json")

    # Build MLX tensor dict (sorted for deterministic order)
    mlx_tensors: dict[str, mx.array] = {}
    for name in sorted(weights):
        mlx_tensors[name] = _numpy_to_mlx(weights[name])

    # Materialise all lazy arrays before writing
    mx.eval(*mlx_tensors.values())

    # Write weights
    mx.save_safetensors(weights_path, mlx_tensors)
    print(f"  Wrote {weights_path}  ({os.path.getsize(weights_path) / 1e6:.1f} MB)")

    # Add per-tensor shape/dtype info to metadata
    tensor_info: dict[str, dict] = {}
    for name, arr in weights.items():
        dtype_str = _numpy_dtype_to_str(arr.dtype)
        tensor_info[name] = {"shape": list(arr.shape), "dtype": dtype_str}
    meta["tensors"] = tensor_info

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Wrote {meta_path}")

    return weights_path, meta_path


def _numpy_dtype_to_str(dtype: np.dtype) -> str:
    if dtype == np.float32: return "F32"
    if dtype == np.float16: return "F16"
    if dtype == np.uint16:  return "BF16"
    if dtype == np.int8:    return "I8"
    return dtype.str


# ---------------------------------------------------------------------------
# Style-only post-processing
# ---------------------------------------------------------------------------

def apply_style_only(weights: dict[str, np.ndarray], num_double_blocks: int) -> None:
    """Zero ip_scale for double-stream blocks in-place (style-only inference mode)."""
    sc = weights["ip_scale"]
    sc[:num_double_blocks] = 0.0
    print(f"  Style-only: zeroed ip_scale[0:{num_double_blocks}] (double-stream blocks).")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_bundle(out_dir: str, quant: str) -> bool:
    """
    Re-open the exported bundle and verify structural integrity.
    Returns True if all checks pass.
    """
    import mlx.core as mx

    weights_path = os.path.join(out_dir, "adapter_weights.safetensors")
    meta_path    = os.path.join(out_dir, "adapter_meta.json")

    if not os.path.exists(weights_path):
        print(f"FAIL: {weights_path} not found", file=sys.stderr)
        return False
    if not os.path.exists(meta_path):
        print(f"FAIL: {meta_path} not found", file=sys.stderr)
        return False

    with open(meta_path) as f:
        meta = json.load(f)

    required_meta_keys = {
        "version", "adapter_type", "num_blocks", "hidden_dim",
        "num_image_tokens", "siglip_dim", "quant", "tensors",
    }
    missing_meta = required_meta_keys - set(meta.keys())
    if missing_meta:
        print(f"FAIL: adapter_meta.json missing keys: {missing_meta}", file=sys.stderr)
        return False

    loaded = mx.load(weights_path)

    # Check all expected tensors are present
    expected = set(_KEY_MAP.values())
    if quant == "int8":
        scale_keys = {f"{n}.scale" for n in _QUANTISED_TENSORS}
        expected = expected | scale_keys

    missing_tensors = expected - set(loaded.keys())
    if missing_tensors:
        print(f"FAIL: missing tensors: {missing_tensors}", file=sys.stderr)
        return False

    # Shape checks
    dims = meta
    errors = []

    qt = loaded["perceiver.query_tokens"]
    if qt.shape != (dims["num_image_tokens"], dims["hidden_dim"]):
        errors.append(f"perceiver.query_tokens shape {qt.shape}")

    ks = loaded["ip_k_stacked"]
    if ks.shape != (dims["num_blocks"], dims["hidden_dim"], dims["hidden_dim"]):
        errors.append(f"ip_k_stacked shape {ks.shape}")

    sc = loaded["ip_scale"]
    if sc.shape != (dims["num_blocks"],):
        errors.append(f"ip_scale shape {sc.shape}")

    if errors:
        for e in errors:
            print(f"FAIL: wrong shape — {e}", file=sys.stderr)
        return False

    # Numeric sanity: ip_scale values should be finite
    sc_np = np.array(sc.astype(mx.float32))
    if not np.all(np.isfinite(sc_np)):
        print("FAIL: ip_scale contains NaN/Inf", file=sys.stderr)
        return False

    # For int8: verify scale tensors are positive
    if quant == "int8":
        for name in _QUANTISED_TENSORS:
            skey = f"{name}.scale"
            if skey in loaded:
                s_np = np.array(loaded[skey].astype(mx.float32))
                if not np.all(s_np > 0):
                    print(f"FAIL: {skey} has non-positive values", file=sys.stderr)
                    return False

    # Quick dequant spot-check for int8: first perceiver weight
    if quant == "int8":
        q8 = np.array(loaded["perceiver.query_proj"])
        sc8 = np.array(loaded["perceiver.query_proj.scale"].astype(mx.float32))
        dq = q8.astype(np.float32) * sc8[:, None]
        if not np.all(np.isfinite(dq)):
            print("FAIL: perceiver.query_proj dequant has NaN/Inf", file=sys.stderr)
            return False

    print(f"  Validation PASSED  ({len(loaded)} tensors, {meta['quant']} quant)")
    return True


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export(
    checkpoint:      str,
    out_dir:         str,
    quant:           str       = "bfloat16",
    use_ema:         bool      = False,
    style_only:      bool      = False,
    step:            int | None = None,
    do_validate:     bool      = False,
    perceiver_heads: int | None = None,
) -> None:
    t0 = time.monotonic()

    # 1. Load weights
    weights_f32 = load_checkpoint(checkpoint, use_ema)

    # 2. Infer dimensions
    dims = _infer_dims(weights_f32)

    # 3. Style-only mode
    if style_only:
        apply_style_only(weights_f32, dims["num_double_blocks"])

    # 4. Infer training step from filename if not provided
    if step is None:
        name = os.path.basename(checkpoint)
        if name.startswith("step_") and name.endswith(".safetensors"):
            try:
                step = int(name[5:-len(".safetensors")])
            except ValueError:
                pass

    # 5. Quantise
    print(f"Applying quantisation: {quant}")
    weights_q = apply_quant(weights_f32, quant)

    # 6. Build metadata
    meta: dict[str, Any] = {
        "version":            EXPORT_VERSION,
        "adapter_type":       ADAPTER_TYPE,
        "model_target":       "flux-klein-4b" if dims["hidden_dim"] == 3072 else "flux-klein-9b",
        "iris_version":       "v2.7",
        **dims,
        "perceiver_heads":    perceiver_heads if perceiver_heads is not None else dims["num_heads"],
        "style_only":         style_only,
        "quant":              quant,
        "training_step":      step,
        "source_checkpoint":  os.path.basename(checkpoint),
        "export_time":        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # 7. Write bundle
    print(f"Writing bundle to {out_dir}/")
    write_bundle(weights_q, meta, out_dir)

    elapsed = time.monotonic() - t0
    print(f"Export complete in {elapsed:.1f}s")

    # 8. Optional validation
    if do_validate:
        print("Validating bundle ...")
        ok = validate_bundle(out_dir, quant)
        if not ok:
            sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export IP-Adapter checkpoint to iris.c bundle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Checkpoint Key Map")[0].strip(),
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to training checkpoint (.safetensors)",
    )
    p.add_argument(
        "--output", required=True,
        help="Output directory (created if absent)",
    )
    p.add_argument(
        "--quant",
        choices=["bfloat16", "float16", "int8"],
        default="bfloat16",
        help="Quantisation mode (default: bfloat16)",
    )
    p.add_argument(
        "--use-ema",
        action="store_true",
        help="Load EMA weights instead of live weights (recommended for inference)",
    )
    p.add_argument(
        "--style-only",
        action="store_true",
        help="Zero ip_scale for double-stream blocks (indices 0–4) for style-only mode",
    )
    p.add_argument(
        "--step", type=int, default=None,
        help="Override training step recorded in metadata (auto-detected from filename)",
    )
    p.add_argument(
        "--perceiver-heads", type=int, default=None,
        help="Number of heads in PerceiverResampler cross-attention (default: inferred as "
             "transformer num_heads, which is wrong if perceiver_heads differs — pass "
             "the value from your training config's adapter.perceiver_heads)",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="Validate the written bundle after export (re-reads, checks shapes/dtypes)",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    export(
        checkpoint      = args.checkpoint,
        out_dir         = args.output,
        quant           = args.quant,
        use_ema         = args.use_ema,
        style_only      = args.style_only,
        step            = args.step,
        do_validate     = args.validate,
        perceiver_heads = args.perceiver_heads,
    )


if __name__ == "__main__":
    main()
