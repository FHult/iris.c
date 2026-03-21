"""
train/scripts/clip_dedup.py — CLIP-based deduplication via clip-retrieval + FAISS.

Removes near-duplicate images (cosine similarity > 0.95) from the training dataset.
Expected removal: 200–400K from LAION → ~1.6M unique images.

Steps (initial, chunk 1):
  1. Embed LAION images with CLIP ViT-L/14 (~1.5h on M1 Max for 2M images)
  2. Build FAISS pair index + find duplicate pairs (threshold=0.95)
  3. Write duplicate_ids.txt blocklist → used by build_shards.py
  4. After unified shards are built: build a persistent flat IP index over all shards
     (this is the cross-chunk dedup index, ~10min)

Steps (incremental, chunks 2–4):
  1. Embed new WDS chunk images
  2. Query against the persistent flat index → mark near-duplicates as blocked
  3. Append new IDs to blocklist (used by build_shards.py --blocklist)
  4. Add new embeddings to the persistent index

Usage:
    source train/.venv/bin/activate
    pip install clip-retrieval autofaiss faiss-cpu

    # ── Chunk 1: LAION intra-dedup ────────────────────────────────────────────
    python train/scripts/clip_dedup.py all \\
        --shards     train/data/raw/laion \\
        --embeddings train/data/embeddings/laion \\
        --output     train/data/dedup_ids

    # ── Chunk 1: build persistent cross-chunk dedup index ────────────────────
    #   (run after build_shards.py merges all sources)
    python train/scripts/clip_dedup.py build-index \\
        --shards     train/data/shards \\
        --embeddings train/data/embeddings/all \\
        --index      train/data/dedup_ids/dedup_index.faiss

    # ── Chunks 2–4: incremental cross-chunk dedup ─────────────────────────────
    python train/scripts/clip_dedup.py incremental \\
        --shards    train/data/raw/journeydb_wds_chunk2 \\
        --embeddings train/data/embeddings/chunk2 \\
        --index     train/data/dedup_ids/dedup_index.faiss \\
        --blocklist train/data/dedup_ids/duplicate_ids.txt

Reference: plans/ip-adapter-training.md §2.3
"""

import argparse
import glob
import os
import subprocess
import sys

try:
    import subprocess as _sp
    _PERF_CORES = int(_sp.check_output(
        ["sysctl", "-n", "hw.perflevel0.logicalcpu"], text=True).strip())
except Exception:
    _PERF_CORES = os.cpu_count() or 4


# ---------------------------------------------------------------------------
# clip-retrieval wrappers
# ---------------------------------------------------------------------------

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
    """Build FAISS index from CLIP embeddings (for intra-set dedup)."""
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

    blocklist_path = os.path.join(output_dir, "duplicate_ids.txt")
    _write_blocklist(dedup_dir, blocklist_path)
    print(f"Blocklist written to {blocklist_path}")


def _write_blocklist(dedup_pairs_dir: str, output_path: str):
    """
    Parse clip-retrieval dedup output and write one duplicate ID per line.
    Each duplicate pair file contains (id_a, id_b, similarity) rows.
    We keep id_a and block id_b.
    """
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


# ---------------------------------------------------------------------------
# Cross-chunk dedup helpers
# ---------------------------------------------------------------------------

def _load_embeddings_from_dir(embeddings_dir: str):
    """
    Load clip-retrieval inference output from a directory.
    Returns (ids: list[str], vecs: np.ndarray[N, D] float32, L2-normalised).
    Matches img_emb_*.npy files against metadata_*.parquet by index.
    """
    import numpy as np

    emb_files = sorted(glob.glob(os.path.join(embeddings_dir, "img_emb_*.npy")))
    if not emb_files:
        return [], np.zeros((0, 768), dtype=np.float32)

    all_vecs = []
    all_ids = []

    for emb_file in emb_files:
        idx = emb_file.rsplit("_", 1)[-1].replace(".npy", "")
        meta_file = os.path.join(embeddings_dir, f"metadata_{idx}.parquet")

        vecs = np.load(emb_file).astype(np.float32)
        all_vecs.append(vecs)

        if os.path.exists(meta_file):
            import pandas as pd
            df = pd.read_parquet(meta_file)
            key_col = next((c for c in ("key", "image_path", "url") if c in df.columns), None)
            if key_col:
                all_ids.extend(df[key_col].astype(str).tolist())
            else:
                all_ids.extend([f"{idx}_{i}" for i in range(len(vecs))])
        else:
            all_ids.extend([f"{idx}_{i}" for i in range(len(vecs))])

    vecs = np.vstack(all_vecs)
    # L2-normalise so IndexFlatIP gives cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs /= norms
    return all_ids, vecs


def _load_or_create_flat_index(index_path: str, dim: int = 768):
    """Load existing IndexFlatIP or create an empty one."""
    import faiss
    if os.path.exists(index_path):
        print(f"  Loading dedup index from {index_path} ...", flush=True)
        index = faiss.read_index(index_path)
        print(f"  Index contains {index.ntotal:,} vectors", flush=True)
    else:
        print(f"  Creating new dedup index (dim={dim})", flush=True)
        index = faiss.IndexFlatIP(dim)
    return index


def run_build_index(shards_dir: str, embeddings_dir: str, index_path: str,
                    batch_size: int = 256):
    """
    Embed all unified shards with CLIP and build the persistent flat IP index
    used for cross-chunk deduplication.

    Resume-safe: skips embedding only if the sentinel file (.embed_done) exists,
    indicating a previous run completed successfully.  A sentinel is written only
    after clip-retrieval inference exits cleanly, so a partially-written embedding
    dir is detected and re-embedded from scratch.
    """
    import faiss

    sentinel = os.path.join(embeddings_dir, ".embed_done")
    if os.path.exists(sentinel):
        existing_embs = glob.glob(os.path.join(embeddings_dir, "img_emb_*.npy"))
        print(f"Embeddings already complete ({len(existing_embs)} files) — skipping CLIP inference")
    else:
        # Remove any partial embedding files before re-running
        for stale in glob.glob(os.path.join(embeddings_dir, "img_emb_*.npy")):
            os.remove(stale)
        for stale in glob.glob(os.path.join(embeddings_dir, "metadata_*.parquet")):
            os.remove(stale)
        print(f"Embedding unified shards for cross-chunk dedup index...")
        run_embed(shards_dir, embeddings_dir, batch_size)
        open(sentinel, "w").close()  # mark embedding as complete

    print("Loading embeddings...")
    ids, vecs = _load_embeddings_from_dir(embeddings_dir)
    if len(ids) == 0:
        print("No embeddings found — nothing to index", file=sys.stderr)
        return

    print(f"  {len(ids):,} embeddings (dim={vecs.shape[1]})")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"  Saved dedup index → {index_path} ({index.ntotal:,} vectors)")


def run_incremental(new_shards: str, embeddings_dir: str, index_path: str,
                    blocklist_path: str, threshold: float = 0.95,
                    batch_size: int = 256):
    """
    Incremental cross-chunk deduplication:
      1. Embed new_shards with CLIP (skip if embeddings already present)
      2. Query every new embedding against the existing flat index
      3. Append IDs of near-duplicates (similarity >= threshold) to blocklist
      4. Add all new embeddings to the index (originals + duplicates alike,
         so future chunks are checked against the full corpus)
      5. Save the updated index

    build_shards.py --blocklist will then skip the flagged IDs when writing shards.
    """
    import faiss

    # 1. Embed
    existing_embs = glob.glob(os.path.join(embeddings_dir, "img_emb_*.npy"))
    if existing_embs:
        print(f"Embeddings already exist ({len(existing_embs)} files) — skipping CLIP inference")
    else:
        print(f"Embedding new shards from {new_shards} ...")
        run_embed(new_shards, embeddings_dir, batch_size)

    # 2. Load new embeddings
    new_ids, new_vecs = _load_embeddings_from_dir(embeddings_dir)
    if len(new_ids) == 0:
        print("No embeddings found in new shards — nothing to dedup", file=sys.stderr)
        return
    print(f"  {len(new_ids):,} new embeddings (dim={new_vecs.shape[1]})", flush=True)

    # 3. Load or create the persistent index
    index = _load_or_create_flat_index(index_path, dim=new_vecs.shape[1])

    # 4. Query for near-duplicates
    duplicate_ids: set[str] = set()
    if index.ntotal > 0:
        print(f"  Querying {len(new_ids):,} images against {index.ntotal:,} existing ...",
              flush=True)
        # Search in batches of 4096 to avoid peak memory
        BATCH = 4096
        for start in range(0, len(new_ids), BATCH):
            batch_vecs = new_vecs[start:start + BATCH]
            scores, _ = index.search(batch_vecs, k=1)
            for i, score_row in enumerate(scores):
                if score_row[0] >= threshold:
                    duplicate_ids.add(new_ids[start + i])
        print(f"  Found {len(duplicate_ids):,} near-duplicates (threshold={threshold})",
              flush=True)
    else:
        print("  Index is empty — all new images treated as unique (building initial index)")

    # 5. Append new duplicate IDs to blocklist
    if duplicate_ids:
        existing_blocked: set[str] = set()
        if os.path.exists(blocklist_path):
            with open(blocklist_path) as f:
                existing_blocked = {line.strip() for line in f if line.strip()}

        new_blocked = duplicate_ids - existing_blocked
        if new_blocked:
            os.makedirs(os.path.dirname(blocklist_path) or ".", exist_ok=True)
            with open(blocklist_path, "a") as f:
                for id_ in sorted(new_blocked):
                    f.write(id_ + "\n")
            print(f"  Added {len(new_blocked):,} IDs to blocklist ({blocklist_path})",
                  flush=True)

    # 6. Add all new embeddings to the index (so future chunks see this data)
    index.add(new_vecs)
    faiss.write_index(index, index_path)
    print(f"  Updated dedup index → {index_path} ({index.ntotal:,} total vectors)", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CLIP-based deduplication for training dataset"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # embed
    emb = subparsers.add_parser("embed", help="Embed images with CLIP ViT-L/14")
    emb.add_argument("--shards", required=True)
    emb.add_argument("--embeddings", required=True)
    emb.add_argument("--batch_size", type=int, default=256)

    # dedup (intra-set, for initial LAION pass)
    dd = subparsers.add_parser("dedup", help="Index + find intra-set duplicates")
    dd.add_argument("--embeddings", required=True)
    dd.add_argument("--output", required=True,
                    help="Directory for duplicate_ids.txt")
    dd.add_argument("--threshold", type=float, default=0.95)
    dd.add_argument("--max_memory_gb", type=int, default=12)

    # all (embed + intra dedup in one go, for LAION)
    al = subparsers.add_parser("all", help="Embed + index + dedup in sequence")
    al.add_argument("--shards", required=True)
    al.add_argument("--embeddings", required=True)
    al.add_argument("--output", required=True)
    al.add_argument("--batch_size", type=int, default=256)
    al.add_argument("--threshold", type=float, default=0.95)

    # build-index (after chunk 1 build_shards: embed all shards + build flat IP index)
    bi = subparsers.add_parser(
        "build-index",
        help="Embed unified shards and build persistent flat FAISS index for cross-chunk dedup"
    )
    bi.add_argument("--shards", required=True,
                    help="Unified shards directory (output of build_shards.py)")
    bi.add_argument("--embeddings", required=True,
                    help="Directory to store CLIP embeddings")
    bi.add_argument("--index", required=True,
                    help="Path to save the flat IP FAISS index (.faiss)")
    bi.add_argument("--batch_size", type=int, default=256)

    # incremental (chunks 2–4: embed new WDS, query index, update blocklist + index)
    inc = subparsers.add_parser(
        "incremental",
        help="Embed new chunk shards, find cross-chunk duplicates, update blocklist + index"
    )
    inc.add_argument("--shards", required=True,
                     help="New WDS chunk directory (output of convert_journeydb.py)")
    inc.add_argument("--embeddings", required=True,
                     help="Directory to store CLIP embeddings for this chunk")
    inc.add_argument("--index", required=True,
                     help="Path to the persistent flat IP FAISS index (.faiss)")
    inc.add_argument("--blocklist", required=True,
                     help="Path to duplicate_ids.txt (will be appended to)")
    inc.add_argument("--threshold", type=float, default=0.95)
    inc.add_argument("--batch_size", type=int, default=256)

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

    elif args.command == "build-index":
        run_build_index(args.shards, args.embeddings, args.index, args.batch_size)

    elif args.command == "incremental":
        run_incremental(
            args.shards, args.embeddings, args.index,
            args.blocklist, args.threshold, args.batch_size,
        )


if __name__ == "__main__":
    main()
