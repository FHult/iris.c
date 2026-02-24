#!/usr/bin/env python3
"""
create_test_lora.py - Create a minimal XLabs-format LoRA for integration testing.

The LoRA has rank=2, covers double_blocks 0-4 (matching Klein 4B).
All weights are tiny random values so the LoRA won't affect output much.

Usage: python3 debug/create_test_lora.py --output /tmp/test_lora.safetensors
"""

import struct
import json
import argparse
import os
import random

random.seed(42)


def float_to_bf16(f):
    """Convert float32 to bfloat16 (truncate lower 16 bits)."""
    b = struct.pack('<f', f)
    return struct.unpack('<H', b[2:])[0]


def make_tensor_f32(shape):
    """Create a small random float32 tensor."""
    n = 1
    for d in shape:
        n *= d
    return [random.gauss(0, 0.01) for _ in range(n)]


def write_safetensors(path, tensors):
    """Write tensors dict {name: (shape, data_f32)} to a safetensors file."""
    # Build metadata
    metadata = {}
    offset = 0
    for name, (shape, data) in tensors.items():
        nbytes = len(data) * 4  # f32
        metadata[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes

    header_json = json.dumps(metadata)
    header_bytes = header_json.encode('utf-8')
    # Pad to 8-byte boundary
    while len(header_bytes) % 8 != 0:
        header_bytes += b' '

    with open(path, 'wb') as f:
        # 8-byte header size
        f.write(struct.pack('<Q', len(header_bytes)))
        f.write(header_bytes)
        # Tensor data
        for name, (shape, data) in tensors.items():
            for v in data:
                f.write(struct.pack('<f', v))

    total = 8 + len(header_bytes) + offset
    print(f"Written {path} ({len(tensors)} tensors, {total/1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='/tmp/test_lora_xlabs.safetensors')
    parser.add_argument('--rank', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=3072)
    parser.add_argument('--num-double', type=int, default=5)
    args = parser.parse_args()

    rank = args.rank
    hidden = args.hidden
    num_double = args.num_double

    tensors = {}
    for i in range(num_double):
        # img stream QKV (fused: lora1)
        tensors[f"double_blocks.{i}.processor.qkv_lora1.down.weight"] = (
            [rank, hidden], make_tensor_f32([rank, hidden]))
        tensors[f"double_blocks.{i}.processor.qkv_lora1.up.weight"] = (
            [hidden * 3, rank], make_tensor_f32([hidden * 3, rank]))

        # img stream proj
        tensors[f"double_blocks.{i}.processor.proj_lora1.down.weight"] = (
            [rank, hidden], make_tensor_f32([rank, hidden]))
        tensors[f"double_blocks.{i}.processor.proj_lora1.up.weight"] = (
            [hidden, rank], make_tensor_f32([hidden, rank]))

        # txt stream QKV (fused: lora2)
        tensors[f"double_blocks.{i}.processor.qkv_lora2.down.weight"] = (
            [rank, hidden], make_tensor_f32([rank, hidden]))
        tensors[f"double_blocks.{i}.processor.qkv_lora2.up.weight"] = (
            [hidden * 3, rank], make_tensor_f32([hidden * 3, rank]))

        # txt stream proj
        tensors[f"double_blocks.{i}.processor.proj_lora2.down.weight"] = (
            [rank, hidden], make_tensor_f32([rank, hidden]))
        tensors[f"double_blocks.{i}.processor.proj_lora2.up.weight"] = (
            [hidden, rank], make_tensor_f32([hidden, rank]))

    write_safetensors(args.output, tensors)
    print(f"Test XLabs LoRA created: rank={rank}, hidden={hidden}, {num_double} double blocks")


if __name__ == '__main__':
    main()
