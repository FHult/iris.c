"""
train/scripts/clip_dedup.py — CLIP-based deduplication using open_clip + FAISS.

Removes near-duplicate images (cosine similarity > 0.95) from the training dataset.
Expected removal: 200–400K from LAION → ~1.6M unique images.

No clip-retrieval dependency. Uses open_clip (ViT-L-14-quickgelu, MPS on Apple Silicon)
for embedding and faiss-cpu for index / search.

Steps (initial, chunk 1):
  1. Embed LAION images with CLIP ViT-L/14 on MPS (~1.5h on M1 Max for 1.5M images)
  2. Find near-duplicate pairs with FAISS batched k-NN (threshold=0.95)
  3. Write duplicate_ids.txt blocklist → used by build_shards.py
  4. After unified shards are built: build a persistent flat IP index over all shards

Steps (incremental, chunks 2–4):
  1. Embed new WDS chunk images
  2. Query against the persistent flat index → mark near-duplicates as blocked
  3. Append new IDs to blocklist
  4. Add new embeddings to the persistent index

Usage:
    source train/.venv/bin/activate

    # ── Chunk 1: LAION intra-dedup ────────────────────────────────────────────
    python train/scripts/clip_dedup.py all \\
        --shards     train/data/raw/laion \\
        --embeddings train/data/embeddings/laion \\
        --output     train/data/dedup_ids

    # ── Chunk 1: build persistent cross-chunk dedup index ────────────────────
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
import io
import os
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

# Must be set before numpy/FAISS import; libomp on Apple Silicon crashes under
# multi-threaded barrier release (SIGSEGV in __kmp_hyper_barrier_release).
# KMP_DUPLICATE_LIB_OK suppresses abort when torch and faiss each bring their
# own libomp copy — safe here since we pin to 1 thread anyway.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

try:
    import subprocess as _sp
    _PERF_CORES = int(_sp.check_output(
        ["sysctl", "-n", "hw.perflevel0.logicalcpu"], text=True).strip())
except Exception:
    _PERF_CORES = os.cpu_count() or 4


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _clip_device() -> str:
    """Return 'mps' on Apple Silicon, 'cuda' if available, else 'cpu'."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _decode_shard(tar_path: str, preprocess) -> tuple:
    """
    Open one .tar shard and decode all JPEG images into preprocessed tensors.
    Runs in a background thread so I/O + CPU decode overlaps with GPU inference.
    Uses TurboJPEG when available (3-5x faster than PIL for JPEG decode).

    Two-phase design:
      Phase 1: sequential tar read (tarfile is not thread-safe)
      Phase 2: parallel decode+preprocess across P-cores via ThreadPoolExecutor

    Returns (keys, tensors) or (None, None) on hard error.
    """
    from PIL import Image as _PilImage
    try:
        from turbojpeg import TurboJPEG as _TJ, TJPF_RGB as _TJPF_RGB
        _tj = _TJ()
        def _decode_jpeg(data):
            return _PilImage.fromarray(_tj.decode(data, pixel_format=_TJPF_RGB))
    except ImportError:
        def _decode_jpeg(data):
            return _PilImage.open(io.BytesIO(data)).convert("RGB")

    # Phase 1: sequential tar read
    raw_items = []
    try:
        with tarfile.open(tar_path) as tf:
            members = {m.name: m for m in tf.getmembers()}
            jpg_keys = sorted(
                os.path.splitext(n)[0] for n in members if n.endswith(".jpg")
            )
            for key in jpg_keys:
                m = members.get(key + ".jpg")
                if m is None:
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                raw_items.append((key, f.read()))
    except Exception as e:
        print(f"  Warning: skipping {os.path.basename(tar_path)}: {e}")
        return None, None

    if not raw_items:
        return [], []

    # Phase 2: parallel decode + preprocess across P-cores
    def _process_one(key_raw):
        key, raw = key_raw
        try:
            return key, preprocess(_decode_jpeg(raw))
        except Exception:
            return None, None

    n_threads = min(_PERF_CORES, len(raw_items))
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        results = list(pool.map(_process_one, raw_items))

    keys = [k for k, t in results if k is not None]
    tensors = [t for k, t in results if t is not None]
    return keys, tensors


def run_embed(shards_dir: str, embeddings_dir: str, batch_size: int = 512):
    """
    Embed all WebDataset images with CLIP ViT-L/14 using open_clip.
    Forward pass runs on MPS (Apple Silicon GPU).
    Uses a background thread to prefetch+decode the next shard while the GPU
    processes the current one, overlapping CPU decode with GPU inference.
    Output per shard: img_emb_NNNN.npy (float32 L2-normalised) + metadata_NNNN.parquet.
    Resume-safe: skips shards whose .npy already exists.
    """
    import torch
    import open_clip
    import pandas as pd
    import time as _time

    device = torch.device(_clip_device())
    print(f"Loading CLIP ViT-L/14 on device={device} ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14-quickgelu", pretrained="openai"
    )
    model = model.to(device).half().eval()

    os.makedirs(embeddings_dir, exist_ok=True)
    tar_files = sorted(glob.glob(os.path.join(shards_dir, "*.tar")))

    # Pre-scan: accumulate counts for already-embedded shards; build pending list
    # so rate/ETA are computed only over shards that actually need work.
    total_embedded = 0
    pending = []  # (original_shard_idx, tar_path)
    for shard_idx, tar_path in enumerate(tar_files):
        out_emb  = os.path.join(embeddings_dir, f"img_emb_{shard_idx:04d}.npy")
        out_meta = os.path.join(embeddings_dir, f"metadata_{shard_idx:04d}.parquet")
        if os.path.exists(out_emb) and os.path.exists(out_meta):
            total_embedded += np.load(out_emb, mmap_mode="r").shape[0]
        else:
            pending.append((shard_idx, tar_path))

    print(
        f"Embedding {len(tar_files)} shards from {shards_dir}  "
        f"({len(pending)} to process, {len(tar_files)-len(pending)} already done) ..."
    )

    t_start = _time.time()
    t_last_hb = t_start
    interval_rates = []

    # Two background threads prefetch upcoming shards while the GPU processes the
    # current one.  A 2-deep queue absorbs I/O variance (e.g. a slow tar read on
    # one shard won't stall the GPU if the shard after it is already decoded).
    with ThreadPoolExecutor(max_workers=2) as loader:
        from collections import deque
        prefetch_q: deque = deque()
        # Seed the queue with the first two shards
        for k in range(min(2, len(pending))):
            prefetch_q.append(loader.submit(_decode_shard, pending[k][1], preprocess))

        for batch_idx, (shard_idx, tar_path) in enumerate(pending):
            out_emb  = os.path.join(embeddings_dir, f"img_emb_{shard_idx:04d}.npy")
            out_meta = os.path.join(embeddings_dir, f"metadata_{shard_idx:04d}.parquet")

            # Collect the front-of-queue result (already decoded in background)
            keys, tensors = prefetch_q.popleft().result()

            # Enqueue the shard that is 2 positions ahead
            next_k = batch_idx + 2
            if next_k < len(pending):
                prefetch_q.append(loader.submit(_decode_shard, pending[next_k][1], preprocess))

            if keys is None or not tensors:
                continue

            # GPU inference — runs while the next shard is being decoded above.
            # fp16 halves memory bandwidth; deferred CPU transfer avoids a GPU
            # sync after every batch.
            all_embs_gpu = []
            with torch.no_grad():
                for i in range(0, len(tensors), batch_size):
                    batch = torch.stack(tensors[i:i + batch_size]).to(device, dtype=torch.float16)
                    embs  = model.encode_image(batch)
                    embs  = embs / embs.norm(dim=-1, keepdim=True)
                    all_embs_gpu.append(embs)
            emb_arr = torch.cat(all_embs_gpu).float().cpu().numpy()
            np.save(out_emb, emb_arr)
            pd.DataFrame({"key": keys}).to_parquet(out_meta, index=False)

            total_embedded += len(keys)
            if (batch_idx + 1) % 10 == 0 or batch_idx == len(pending) - 1:
                t_now = _time.time()
                interval_time = t_now - t_last_hb
                if interval_time > 0:
                    interval_rates.append((batch_idx % 10 + 1) / interval_time)
                avg_rate = sum(interval_rates) / len(interval_rates) if interval_rates else 0
                eta = (len(pending) - batch_idx - 1) / avg_rate if avg_rate > 0 else 0
                t_last_hb = t_now
                print(
                    f"  [{batch_idx+1}/{len(pending)}] {total_embedded:,} images embedded"
                    f"  {avg_rate:.1f} shards/s  ETA {eta/60:.0f}m",
                    flush=True,
                )

    print(f"Done. {total_embedded:,} embeddings in {embeddings_dir}")


# ---------------------------------------------------------------------------
# Intra-set deduplication (LAION chunk 1)
# ---------------------------------------------------------------------------

def run_dedup(embeddings_dir: str, output_dir: str, threshold: float = 0.95):
    """
    Find near-duplicate pairs within the embedded set using batched FAISS k-NN search.
    Writes duplicate_ids.txt: one blocked ID per line (keep first occurrence, block rest).
    Works in O(N * k) with IndexFlatIP — exact cosine similarity, no approximation.
    """
    import faiss
    # OpenMP parallelism in FAISS is unstable on Apple Silicon (macOS) — crashes in
    # __kmp_suspend_initialize_thread with near-null pointer dereference.
    faiss.omp_set_num_threads(1)

    os.makedirs(output_dir, exist_ok=True)
    ids, vecs = _load_embeddings_from_dir(embeddings_dir)
    if len(ids) == 0:
        print("No embeddings found — nothing to dedup", file=sys.stderr)
        return

    N, D = vecs.shape
    print(f"Building FAISS IndexFlatIP for {N:,} vectors (dim={D}) ...")
    index = faiss.IndexFlatIP(D)
    index.add(vecs)

    print(f"Searching k=2 nearest neighbours (threshold={threshold}) ...")
    blocked: set[str] = set()
    BATCH = 4096
    n_batches = (N + BATCH - 1) // BATCH
    for b_idx, start in enumerate(range(0, N, BATCH), 1):
        batch = vecs[start:start + BATCH]
        scores, idxs = index.search(batch, k=2)
        for i, (score_row, idx_row) in enumerate(zip(scores, idxs)):
            global_i = start + i
            # k=0 is self-match; k=1 is nearest other
            if len(score_row) > 1 and score_row[1] >= threshold:
                j = idx_row[1]
                if j != global_i and j >= 0:
                    # Block the higher-index duplicate
                    blocked.add(ids[max(global_i, j)])
        if b_idx % 50 == 0 or b_idx == n_batches:
            print(
                f"  [{min(start + BATCH, N):,}/{N:,}] {len(blocked):,} duplicates found",
                flush=True,
            )

    blocklist_path = os.path.join(output_dir, "duplicate_ids.txt")
    with open(blocklist_path, "w") as f:
        for rec_id in sorted(blocked):
            f.write(rec_id + "\n")

    print(f"  {len(blocked):,} duplicate IDs → {blocklist_path}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_embeddings_from_dir(embeddings_dir: str):
    """
    Load img_emb_*.npy + metadata_*.parquet pairs.
    Returns (ids: list[str], vecs: np.ndarray[N, D] float32, L2-normalised).
    """
    emb_files = sorted(glob.glob(os.path.join(embeddings_dir, "img_emb_*.npy")))
    if not emb_files:
        return [], np.zeros((0, 768), dtype=np.float32)

    all_vecs = []
    all_ids  = []

    for emb_file in emb_files:
        idx_str  = emb_file.rsplit("_", 1)[-1].replace(".npy", "")
        meta_file = os.path.join(embeddings_dir, f"metadata_{idx_str}.parquet")

        vecs = np.load(emb_file).astype(np.float32)
        all_vecs.append(vecs)

        if os.path.exists(meta_file):
            import pandas as pd
            df = pd.read_parquet(meta_file)
            key_col = next((c for c in ("key", "image_path", "url") if c in df.columns), None)
            if key_col:
                all_ids.extend(df[key_col].astype(str).tolist())
            else:
                all_ids.extend([f"{idx_str}_{i}" for i in range(len(vecs))])
        else:
            all_ids.extend([f"{idx_str}_{i}" for i in range(len(vecs))])

    vecs = np.vstack(all_vecs)
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


# ---------------------------------------------------------------------------
# Cross-chunk dedup helpers
# ---------------------------------------------------------------------------

def run_build_index(shards_dir: str, embeddings_dir: str, index_path: str,
                    batch_size: int = 256):
    """
    Embed all unified shards with CLIP and build the persistent flat IP index.
    Resume-safe: skips embedding if .embed_done sentinel exists.
    """
    import faiss
    faiss.omp_set_num_threads(1)

    sentinel = os.path.join(embeddings_dir, ".embed_done")
    if os.path.exists(sentinel):
        existing_embs = glob.glob(os.path.join(embeddings_dir, "img_emb_*.npy"))
        print(f"Embeddings already complete ({len(existing_embs)} files) — skipping CLIP inference")
    else:
        # run_embed is resume-safe (skips shards whose .npy already exists),
        # so do NOT delete partial results here — let it pick up where it left off.
        print("Embedding unified shards for cross-chunk dedup index...")
        run_embed(shards_dir, embeddings_dir, batch_size)
        open(sentinel, "w").close()

    print("Loading embeddings...")
    ids, vecs = _load_embeddings_from_dir(embeddings_dir)
    if len(ids) == 0:
        print("No embeddings found — nothing to index", file=sys.stderr)
        return

    print(f"  {len(ids):,} embeddings (dim={vecs.shape[1]})")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    tmp_path = index_path + ".tmp"
    faiss.write_index(index, tmp_path)
    os.replace(tmp_path, index_path)
    print(f"  Saved dedup index → {index_path} ({index.ntotal:,} vectors)")


def run_incremental(new_shards: str, embeddings_dir: str, index_path: str,
                    blocklist_path: str, threshold: float = 0.95,
                    batch_size: int = 256):
    """
    Incremental cross-chunk deduplication:
      1. Embed new_shards (skip if embeddings already present)
      2. Query every new embedding against the existing flat index
      3. Append IDs of near-duplicates to blocklist
      4. Add all new embeddings to the index
      5. Save the updated index
    """
    import faiss
    faiss.omp_set_num_threads(1)

    existing_embs = glob.glob(os.path.join(embeddings_dir, "img_emb_*.npy"))
    if existing_embs:
        print(f"Embeddings already exist ({len(existing_embs)} files) — skipping CLIP inference")
    else:
        print(f"Embedding new shards from {new_shards} ...")
        run_embed(new_shards, embeddings_dir, batch_size)

    new_ids, new_vecs = _load_embeddings_from_dir(embeddings_dir)
    if len(new_ids) == 0:
        print("No embeddings found — nothing to dedup", file=sys.stderr)
        return
    print(f"  {len(new_ids):,} new embeddings (dim={new_vecs.shape[1]})", flush=True)

    index = _load_or_create_flat_index(index_path, dim=new_vecs.shape[1])

    duplicate_ids: set[str] = set()
    if index.ntotal > 0:
        print(f"  Querying {len(new_ids):,} images against {index.ntotal:,} existing ...",
              flush=True)
        BATCH = 4096
        n_new = len(new_ids)
        n_batches = (n_new + BATCH - 1) // BATCH
        for b_idx, start in enumerate(range(0, n_new, BATCH), 1):
            batch_vecs = new_vecs[start:start + BATCH]
            scores, _  = index.search(batch_vecs, k=1)
            for i, score_row in enumerate(scores):
                if score_row[0] >= threshold:
                    duplicate_ids.add(new_ids[start + i])
            if b_idx % 50 == 0 or b_idx == n_batches:
                print(
                    f"  [{min(start + BATCH, n_new):,}/{n_new:,}]"
                    f" {len(duplicate_ids):,} near-duplicates so far",
                    flush=True,
                )
        print(f"  Found {len(duplicate_ids):,} near-duplicates (threshold={threshold})",
              flush=True)
    else:
        print("  Index is empty — all new images treated as unique")

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

    index.add(new_vecs)
    tmp_path = index_path + ".tmp"
    faiss.write_index(index, tmp_path)
    os.replace(tmp_path, index_path)
    print(f"  Updated dedup index → {index_path} ({index.ntotal:,} total vectors)", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CLIP-based deduplication (open_clip + FAISS, no clip-retrieval)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # embed
    emb = subparsers.add_parser("embed", help="Embed images with CLIP ViT-L/14")
    emb.add_argument("--shards",      required=True)
    emb.add_argument("--embeddings",  required=True)
    emb.add_argument("--batch_size",  type=int, default=256)

    # dedup (intra-set, for initial LAION pass)
    dd = subparsers.add_parser("dedup", help="Find intra-set duplicates from embeddings")
    dd.add_argument("--embeddings",  required=True)
    dd.add_argument("--output",      required=True,
                    help="Directory for duplicate_ids.txt")
    dd.add_argument("--threshold",   type=float, default=0.95)

    # all (embed + intra dedup in one go, for LAION)
    al = subparsers.add_parser("all", help="Embed + dedup in sequence")
    al.add_argument("--shards",      required=True)
    al.add_argument("--embeddings",  required=True)
    al.add_argument("--output",      required=True)
    al.add_argument("--batch_size",  type=int, default=256)
    al.add_argument("--threshold",   type=float, default=0.95)

    # build-index
    bi = subparsers.add_parser(
        "build-index",
        help="Embed unified shards and build persistent flat FAISS index"
    )
    bi.add_argument("--shards",      required=True)
    bi.add_argument("--embeddings",  required=True)
    bi.add_argument("--index",       required=True)
    bi.add_argument("--batch_size",  type=int, default=512)

    # incremental
    inc = subparsers.add_parser(
        "incremental",
        help="Embed new chunk, find cross-chunk duplicates, update blocklist + index"
    )
    inc.add_argument("--shards",      required=True)
    inc.add_argument("--embeddings",  required=True)
    inc.add_argument("--index",       required=True)
    inc.add_argument("--blocklist",   required=True)
    inc.add_argument("--threshold",   type=float, default=0.95)
    inc.add_argument("--batch_size",  type=int, default=256)

    args = parser.parse_args()

    if args.command == "embed":
        run_embed(args.shards, args.embeddings, args.batch_size)

    elif args.command == "dedup":
        run_dedup(args.embeddings, args.output, args.threshold)

    elif args.command == "all":
        run_embed(args.shards, args.embeddings, args.batch_size)
        run_dedup(args.embeddings, args.output, args.threshold)

    elif args.command == "build-index":
        run_build_index(args.shards, args.embeddings, args.index, args.batch_size)

    elif args.command == "incremental":
        run_incremental(
            args.shards, args.embeddings, args.index,
            args.blocklist, args.threshold, args.batch_size,
        )


if __name__ == "__main__":
    main()
