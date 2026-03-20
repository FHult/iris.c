"""
train/scripts/clip_dedup.py — CLIP-based deduplication via clip-retrieval + FAISS.

Removes near-duplicate images (cosine similarity > 0.95) from the training dataset.
Expected removal: 200–400K from LAION → ~1.6M unique images.

Steps:
  1. Embed all images with CLIP ViT-L/14 (~1.5h on M1 Max for 2M images)
  2. Build FAISS index (cosine similarity)
  3. Find duplicate pairs (threshold=0.95)
  4. Write duplicate_ids.txt blocklist → used by build_shards.py

Uses clip-retrieval with --num_prepro_workers PERF_CORES to saturate all P-cores
for the CPU decode/preprocess pipeline feeding GPU batching.

Usage:
    source train/.venv/bin/activate
    pip install clip-retrieval autofaiss

    # Step 1: embed (run under caffeinate, ~1.5h)
    python train/scripts/clip_dedup.py embed \\
        --shards /Volumes/IrisData/shards \\
        --embeddings /Volumes/IrisData/embeddings

    # Step 2: index + deduplicate (~20min)
    python train/scripts/clip_dedup.py dedup \\
        --embeddings /Volumes/IrisData/embeddings \\
        --output /Volumes/IrisData/dedup_ids

    # Or run both in sequence:
    python train/scripts/clip_dedup.py all \\
        --shards /Volumes/IrisData/shards \\
        --embeddings /Volumes/IrisData/embeddings \\
        --output /Volumes/IrisData/dedup_ids

Reference: plans/ip-adapter-training.md §2.3
"""

import argparse
import os
import subprocess
import sys

try:
    import subprocess as _sp
    _PERF_CORES = int(_sp.check_output(
        ["sysctl", "-n", "hw.perflevel0.logicalcpu"], text=True).strip())
except Exception:
    _PERF_CORES = os.cpu_count() or 4


def run_embed(shards_dir: str, embeddings_dir: str, batch_size: int = 256):
    """
    Run clip-retrieval inference to embed all training images.
    --num_prepro_workers PERF_CORES saturates all 8 P-cores for CPU decode/preprocess.
    """
    os.makedirs(embeddings_dir, exist_ok=True)
    cmd = [
        "clip-retrieval", "inference",
        "--input_dataset", shards_dir,
        "--output_folder", embeddings_dir,
        "--clip_model", "ViT-L/14",
        "--batch_size", str(batch_size),
        "--num_prepro_workers", str(_PERF_CORES),
        "--enable_metadata", "True",
    ]
    print(f"Running CLIP embedding with {_PERF_CORES} prepro workers...")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def run_index(embeddings_dir: str, index_dir: str, max_memory_gb: int = 12):
    """Build FAISS index from CLIP embeddings."""
    os.makedirs(index_dir, exist_ok=True)
    cmd = [
        "clip-retrieval", "index",
        "--embeddings_folder", embeddings_dir,
        "--index_folder", index_dir,
        "--max_index_memory_usage", f"{max_memory_gb}GB",
    ]
    print("Building FAISS index...")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def run_dedup(embeddings_dir: str, index_dir: str, output_dir: str, threshold: float = 0.95):
    """Find near-duplicate pairs and write a blocklist of IDs to remove."""
    os.makedirs(output_dir, exist_ok=True)
    dedup_dir = os.path.join(output_dir, "dedup_pairs")

    cmd = [
        "clip-retrieval", "deduplication",
        "--embeddings_folder", embeddings_dir,
        "--output_folder", dedup_dir,
        "--threshold", str(threshold),
    ]
    print(f"Finding duplicates (cosine similarity > {threshold})...")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    # Convert dedup pairs → flat blocklist of IDs to remove
    # Keep one image from each pair (the first); mark the second as duplicate.
    blocklist_path = os.path.join(output_dir, "duplicate_ids.txt")
    _write_blocklist(dedup_dir, blocklist_path)
    print(f"Blocklist written to {blocklist_path}")


def _write_blocklist(dedup_pairs_dir: str, output_path: str):
    """
    Parse clip-retrieval dedup output and write one duplicate ID per line.
    Each duplicate pair file contains (id_a, id_b, similarity) rows.
    We keep id_a and block id_b.
    """
    import glob
    blocked = set()
    pair_files = glob.glob(os.path.join(dedup_pairs_dir, "*.parquet"))
    if not pair_files:
        pair_files = glob.glob(os.path.join(dedup_pairs_dir, "*.csv"))

    for pf in pair_files:
        try:
            if pf.endswith(".parquet"):
                import pandas as pd
                df = pd.read_parquet(pf)
            else:
                import pandas as pd
                df = pd.read_csv(pf)
            # clip-retrieval dedup output has 'id_left', 'id_right' or similar
            id_col_right = None
            for col in ["id_right", "duplicate_id", "b_id"]:
                if col in df.columns:
                    id_col_right = col
                    break
            if id_col_right:
                blocked.update(df[id_col_right].astype(str).tolist())
        except Exception as e:
            print(f"Warning: could not parse {pf}: {e}", file=sys.stderr)

    with open(output_path, "w") as f:
        for rec_id in sorted(blocked):
            f.write(rec_id + "\n")

    print(f"  {len(blocked):,} duplicate IDs written to blocklist")


def main():
    parser = argparse.ArgumentParser(
        description="CLIP-based deduplication for training dataset"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # embed subcommand
    emb = subparsers.add_parser("embed", help="Embed all images with CLIP")
    emb.add_argument("--shards", required=True)
    emb.add_argument("--embeddings", required=True)
    emb.add_argument("--batch_size", type=int, default=256)

    # dedup subcommand
    dd = subparsers.add_parser("dedup", help="Index + find duplicates")
    dd.add_argument("--embeddings", required=True)
    dd.add_argument("--output", required=True,
                    help="Directory for duplicate_ids.txt")
    dd.add_argument("--threshold", type=float, default=0.95)
    dd.add_argument("--max_memory_gb", type=int, default=12)

    # all subcommand
    al = subparsers.add_parser("all", help="Embed + index + dedup in sequence")
    al.add_argument("--shards", required=True)
    al.add_argument("--embeddings", required=True)
    al.add_argument("--output", required=True)
    al.add_argument("--batch_size", type=int, default=256)
    al.add_argument("--threshold", type=float, default=0.95)

    args = parser.parse_args()

    if args.command == "embed":
        run_embed(args.shards, args.embeddings, args.batch_size)
    elif args.command == "dedup":
        index_dir = os.path.join(args.embeddings, "faiss_index")
        run_index(args.embeddings, index_dir, args.max_memory_gb)
        run_dedup(args.embeddings, index_dir, args.output, args.threshold)
    elif args.command == "all":
        run_embed(args.shards, args.embeddings, args.batch_size)
        index_dir = os.path.join(args.embeddings, "faiss_index")
        run_index(args.embeddings, index_dir)
        run_dedup(args.embeddings, index_dir, args.output, args.threshold)


if __name__ == "__main__":
    main()
