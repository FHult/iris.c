"""
train/scripts/precompute_qwen3.py — Pre-compute and 4-bit quantise Qwen3 text embeddings.

Saves ~200ms per training step = ~6.7 hours over Stage 1 (105K steps).
Storage: ~143 GB at 4-bit quantised (vs ~2TB at BF16).

Run ONCE after build_shards.py + filter_shards.py complete.
Takes ~8 hours on M1 Max (1.55M images × ~18ms encode per caption).

Output: one .npz file per sample under data/qwen3_q4/{id}.npz
  - q:     uint8 [seq, dim//2] — packed 4-bit values (pairs of nibbles)
  - scale: float16 [seq, 1]   — per-token absmax scale

At training time, load with load_text_embed() (see bottom of this file).
The dequantization is CPU-trivial and done in the prefetch thread.

Quantisation format:
  4-bit signed (-8..7) per token, packed as pairs of nibbles into uint8.
  scale = abs(arr).max(-1) / 7.0   → per-token scale (float16)
  lo nibble = even index, hi nibble = odd index

CPU allocation:
  Qwen3 encode is GPU-bound on M1 Max; CPU pre-process feeds GPU batches.
  Use PERF_CORES workers for the outer shard loop (one process per shard group).

Reference: plans/ip-adapter-training.md §2.7
"""

import argparse
import glob
import io
import multiprocessing
import os
import sys
import tarfile

import numpy as np

try:
    import subprocess
    _PERF_CORES = int(subprocess.check_output(
        ["sysctl", "-n", "hw.perflevel0.logicalcpu"], text=True).strip())
except Exception:
    _PERF_CORES = os.cpu_count() or 4


# ---------------------------------------------------------------------------
# Quantisation helpers
# ---------------------------------------------------------------------------

def quantize_4bit_seq(arr: np.ndarray):
    """
    Per-token absmax 4-bit quantisation.
    arr: float32 [seq, dim]
    Returns:
      q_packed: uint8 [seq, dim//2]  — packed nibble pairs
      scale:    float16 [seq, 1]     — per-token absmax / 7
    """
    scale = np.abs(arr).max(axis=-1, keepdims=True) / 7.0
    scale = np.where(scale == 0, 1e-8, scale)  # avoid div by zero
    q = np.clip(np.round(arr / scale), -8, 7).astype(np.int8)
    # Pack pairs: lo nibble = even columns, hi nibble = odd columns
    q_packed = ((q[:, 0::2] & 0x0F) | ((q[:, 1::2] & 0x0F) << 4)).astype(np.uint8)
    return q_packed, scale.astype(np.float16)


def load_text_embed(npz_path: str) -> np.ndarray:
    """
    Dequantise a saved text embedding.
    Returns float16 [seq, dim].
    Called from the training prefetch thread (CPU-trivial operation).
    """
    d = np.load(npz_path)
    q = d["q"]  # uint8 [seq, dim//2]
    scale = d["scale"]  # float16 [seq, 1]
    lo = (q & 0x0F).astype(np.int8)   # even columns
    hi = ((q >> 4) & 0x0F).astype(np.int8)  # odd columns
    full = np.empty((q.shape[0], q.shape[1] * 2), dtype=np.int8)
    full[:, 0::2] = lo
    full[:, 1::2] = hi
    return (full.astype(np.float32) * scale.astype(np.float32)).astype(np.float16)


# ---------------------------------------------------------------------------
# Shard iteration
# ---------------------------------------------------------------------------

def iter_shard(shard_path: str):
    """Yield (id, caption) from a .tar shard."""
    try:
        with tarfile.open(shard_path) as tar:
            members = {m.name: m for m in tar.getmembers() if m.isfile()}
            keys = {}
            for name in members:
                stem, _, ext = name.rpartition(".")
                if stem not in keys:
                    keys[stem] = {}
                keys[stem][ext.lower()] = name

            for stem, exts in keys.items():
                txt_key = exts.get("txt") or exts.get("caption")
                if not txt_key:
                    continue
                txt = tar.extractfile(members[txt_key]).read().decode(
                    "utf-8", errors="replace"
                ).strip()
                yield stem, txt
    except Exception as e:
        print(f"Warning: {shard_path}: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _encode_single(model, tokenizer, rec_id, caption, output_dir):
    """Encode one caption. Returns True on success. Used as fallback from batch path."""
    import mlx.core as mx
    try:
        chat = [{"role": "user", "content": caption}]
        text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="mlx")
        outputs = model(inputs["input_ids"], output_hidden_states=True)
        h = outputs.hidden_states
        emb = mx.concatenate([h[9], h[18], h[27]], axis=-1)  # [1, seq, 7680]
        emb_np = np.array(emb[0])  # [seq, 7680]
        q_packed, scale = quantize_4bit_seq(emb_np.astype(np.float32))
        np.savez(os.path.join(output_dir, f"{rec_id}.npz"), q=q_packed, scale=scale)
        return True
    except Exception as e:
        print(f"  Skipping {rec_id}: {e}", file=sys.stderr)
        return False


def process_shard(args) -> dict:
    """
    Worker: encode all captions in one shard and save quantised embeddings.
    Loads text encoder once per worker process (not once per shard).
    Captions are processed in batches (sorted by length to minimise padding waste).
    """
    shard_path, output_dir, model_path, batch_size = args
    os.makedirs(output_dir, exist_ok=True)

    try:
        import mlx.core as mx
        from mlx_lm import load as mlx_lm_load
    except ImportError:
        print("mlx_lm not available. Run: pip install mlx-lm", file=sys.stderr)
        return {"shard": shard_path, "written": 0, "error": True}

    model, tokenizer = mlx_lm_load(model_path)
    model.eval()

    # Collect all pending (id, caption) pairs, skipping already-done
    written = 0
    pending = []
    for rec_id, caption in iter_shard(shard_path):
        if os.path.exists(os.path.join(output_dir, f"{rec_id}.npz")):
            written += 1
            continue
        pending.append((rec_id, caption))

    if not pending:
        return {"shard": shard_path, "written": written, "error": False}

    # Tokenize all captions; sort by length to minimise padding within each batch
    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)
    tokenized = []
    for rec_id, caption in pending:
        try:
            chat = [{"role": "user", "content": caption}]
            text = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            ids = tokenizer.encode(text)
            tokenized.append((rec_id, caption, ids))
        except Exception as e:
            print(f"  Skipping {rec_id} (tokenize): {e}", file=sys.stderr)
    tokenized.sort(key=lambda x: len(x[2]))

    # Process in batches with right-padding.
    # Causal attention ensures padding at the end does not affect earlier positions,
    # so hidden states at indices 0..seq_len-1 are identical to single-item inference.
    for i in range(0, len(tokenized), batch_size):
        batch = tokenized[i:i + batch_size]
        seq_lens = [len(ids) for _, _, ids in batch]
        max_len = max(seq_lens)
        padded = mx.array(np.array([
            ids + [pad_id] * (max_len - len(ids))
            for _, _, ids in batch
        ]))  # [B, max_len]

        try:
            outputs = model(padded, output_hidden_states=True)
            h = outputs.hidden_states  # list of [B, max_len, 2560]

            for j, (rec_id, _, _) in enumerate(batch):
                sl = seq_lens[j]
                emb_np = np.concatenate([
                    np.array(h[9][j, :sl]),    # [sl, 2560]
                    np.array(h[18][j, :sl]),   # [sl, 2560]
                    np.array(h[27][j, :sl]),   # [sl, 2560]
                ], axis=-1)                    # [sl, 7680]
                q_packed, scale = quantize_4bit_seq(emb_np.astype(np.float32))
                np.savez(os.path.join(output_dir, f"{rec_id}.npz"), q=q_packed, scale=scale)
                written += 1

        except Exception as e:
            print(f"  Batch {i // batch_size} failed ({e}), retrying single-item", file=sys.stderr)
            for rec_id, caption, _ in batch:
                if _encode_single(model, tokenizer, rec_id, caption, output_dir):
                    written += 1

    return {"shard": shard_path, "written": written, "error": False}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute 4-bit quantised Qwen3 text embeddings"
    )
    parser.add_argument(
        "--shards", required=True,
        help="Directory containing .tar shards"
    )
    parser.add_argument(
        "--output", default="data/qwen3_q4",
        help="Output directory for .npz files (default: data/qwen3_q4)"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-4B",
        help="HuggingFace model ID or local path for Qwen3 (default: Qwen/Qwen3-4B)"
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel processes (default 1; GPU-bound — 2+ processes contend for Metal)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Captions per encoder forward pass (default 4; sequences are right-padded "
             "and sorted by length to minimise wasted compute)"
    )
    args = parser.parse_args()

    shards = sorted(glob.glob(os.path.join(args.shards, "*.tar")))
    if not shards:
        print(f"No .tar files in {args.shards}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print(f"Pre-computing Qwen3 embeddings for {len(shards)} shards")
    print(f"  Model:   {args.model}")
    print(f"  Output:  {args.output}")
    print(f"  Workers:    {args.workers}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Storage estimate: ~{len(shards) * 0.46:.0f} GB")
    print()

    work_items = [(s, args.output, args.model, args.batch_size) for s in shards]

    # Use Pool with 2 workers: GPU is shared, more workers causes contention
    import time as _time
    results = []
    t_start = _time.time()
    t_last_hb = t_start
    interval_rates = []
    with multiprocessing.Pool(processes=args.workers) as pool:
        for done, result in enumerate(
            pool.imap_unordered(process_shard, work_items, chunksize=1), 1
        ):
            results.append(result)
            written_so_far = sum(r["written"] for r in results)
            errs_so_far = sum(1 for r in results if r["error"])
            t_now = _time.time()
            interval_time = t_now - t_last_hb
            if interval_time > 0:
                interval_rates.append(1.0 / interval_time)
            avg_rate = sum(interval_rates) / len(interval_rates) if interval_rates else 0
            eta = (len(work_items) - done) / avg_rate if avg_rate > 0 else 0
            t_last_hb = t_now
            err_str = f"  errors={errs_so_far}" if errs_so_far else ""
            print(
                f"  [{done}/{len(work_items)}] {written_so_far:,} embeddings"
                f"{err_str}  {avg_rate:.2f} shards/s  ETA {eta/60:.0f}m",
                flush=True,
            )

    total = sum(r["written"] for r in results)
    errors = sum(1 for r in results if r["error"])
    print(f"\nDone. {total:,} embeddings saved to {args.output}/")
    if errors:
        print(f"  {errors} shards had errors (check stderr)")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
