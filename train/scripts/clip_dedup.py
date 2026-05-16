"""
train/scripts/clip_dedup.py — CLIP embedding + FAISS deduplication.

Three subcommands called sequentially by the orchestrator per chunk:

  embed       — compute CLIP image embeddings for all shards in --shards,
                write one {stem}.npz per shard into --embeddings
  build-index — load .npz files from --embeddings, extend (or create) the
                cumulative FAISS IndexFlatIP at --index; writes a
                {index}.ids sidecar (one record ID per line, matching add order)
  find-dups   — KNN search over the full index, write near-duplicate IDs
                to --out (the build_shards blocklist); append-safe

The index is cumulative: each chunk's build-index call extends it so
cross-chunk near-duplicates are caught when later chunks run find-dups.

Usage (called by orchestrator):
    clip_dedup.py embed       --shards <dir> --embeddings <dir>
    clip_dedup.py build-index --embeddings <dir> --index <path.faiss>
    clip_dedup.py find-dups   --index <path.faiss> --out <blocklist.txt>
"""

import argparse
import glob
import io
import json
import os
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Must be set before numpy/FAISS import on macOS to prevent libOMP crash.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_lib import (
    LOG_DIR, write_heartbeat, log_event, log_orch, now_iso,
)

try:
    import subprocess as _sp
    _PERF_CORES = int(_sp.check_output(
        ["sysctl", "-n", "hw.perflevel0.logicalcpu"], text=True).strip())
except Exception:
    _PERF_CORES = os.cpu_count() or 4

DUP_THRESHOLD = 0.95

# Module-level lock for FAISS index access in dedup_wds_tar.
_faiss_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Device + CLIP loading
# ---------------------------------------------------------------------------

def _clip_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


_clip_model      = None
_clip_preprocess = None
_clip_backend    = None
_mlx_embedder    = None


def _load_clip(backend: str = "auto") -> None:
    global _clip_model, _clip_preprocess, _clip_backend, _mlx_embedder
    if _clip_model is not None or _mlx_embedder is not None:
        return

    # At sustained workloads (≥512 images, B=16), MLX and open_clip/MPS are at parity
    # (~39 img/s on M1 Max).  Prefer open_clip in auto mode only because it's already
    # installed; fall through to MLX when open_clip/torch are absent.
    if backend in ("open_clip", "auto"):
        device = _clip_device()
        try:
            import torch, open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14-quickgelu", pretrained="openai"
            )
            model = model.to(device).half().eval()
            _clip_model, _clip_preprocess, _clip_backend = model, preprocess, "open_clip"
            log_orch(f"CLIP: ViT-L-14 via open_clip on {device}")
            return
        except ImportError:
            if backend == "open_clip":
                raise RuntimeError("open_clip not installed: pip install open-clip-torch")

    if backend in ("mlx", "auto"):
        try:
            from mlx_clip_embed import MLXCLIPEmbedder
            embedder = MLXCLIPEmbedder()
            embedder.load()
            _mlx_embedder = embedder
            _clip_backend = "mlx"
            log_orch("CLIP: ViT-L-14 via MLX (native Apple Silicon)")
            return
        except Exception as e:
            if backend == "mlx":
                raise RuntimeError(f"MLX CLIP backend failed: {e}") from e
            log_orch(f"CLIP: MLX unavailable ({e}), trying transformers", level="warning")

    if backend in ("transformers", "auto"):
        device = _clip_device()
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            model = model.to(device).eval()
            proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            _clip_model, _clip_preprocess, _clip_backend = model, proc, "transformers"
            log_orch(f"CLIP: ViT-L-14 via transformers on {device}")
            return
        except ImportError:
            if backend == "transformers":
                raise RuntimeError("transformers not installed: pip install transformers torch")

    raise RuntimeError(
        "No CLIP backend available. Install one of: "
        "mlx (recommended on Apple Silicon), open-clip-torch, or transformers+torch"
    )


# ---------------------------------------------------------------------------
# Shard decoding (parallelised, TurboJPEG when available)
# ---------------------------------------------------------------------------

def _decode_shard(tar_path: str, preprocess) -> tuple:
    """
    Open one WDS .tar shard, decode all images.
    Returns (ids: list[str], tensors: list[Tensor]) or (None, None) on error.
    Phase 1 is sequential (tarfile not thread-safe); phase 2 decodes in parallel.
    """
    from PIL import Image as _PIL
    try:
        from turbojpeg import TurboJPEG as _TJ, TJPF_RGB as _RGB
        _tj = _TJ()
        def _decode(data): return _PIL.fromarray(_tj.decode(data, pixel_format=_RGB))
    except ImportError:
        def _decode(data): return _PIL.open(io.BytesIO(data)).convert("RGB")

    raw_items = []
    try:
        with tarfile.open(tar_path) as tf:
            members = {m.name: m for m in tf.getmembers() if m.isfile()}
            keys: dict = {}
            for name in members:
                stem, _, ext = name.rpartition(".")
                if ext.lower() in ("jpg", "jpeg", "png"):
                    keys.setdefault(stem, name)
            for stem in sorted(keys):
                f = tf.extractfile(members[keys[stem]])
                if f:
                    raw_items.append((stem, f.read()))
    except Exception as e:
        log_orch(f"embed: skipping {os.path.basename(tar_path)}: {e}", level="warning")
        return None, None

    if not raw_items:
        return [], []

    def _proc(item):
        stem, raw = item
        try:
            return stem, preprocess(_decode(raw))
        except Exception:
            return None, None

    n_threads = min(_PERF_CORES, len(raw_items))
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        results = list(pool.map(_proc, raw_items))

    ids     = [k for k, t in results if k is not None]
    tensors = [t for k, t in results if t is not None]
    return ids, tensors


# ---------------------------------------------------------------------------
# embed subcommand
# ---------------------------------------------------------------------------

def _run_benchmark(shard_path: str, preferred_backend: str) -> int:
    """Time available CLIP backends on a sample shard and print throughput."""
    print(f"Benchmark: decoding {os.path.basename(shard_path)} ...", flush=True)
    ids, pil_images = _decode_shard(shard_path, lambda img: img)
    if not pil_images:
        print("No images decoded; aborting benchmark.", file=sys.stderr)
        return 1
    n = min(200, len(pil_images))
    pil_images = pil_images[:n]
    print(f"Benchmark: {n} images\n")
    print(f"{'Backend':<14} {'img/s':>8}  {'dim':>5}")
    print("-" * 32)

    def _time_mlx():
        from mlx_clip_embed import MLXCLIPEmbedder
        emb = MLXCLIPEmbedder()
        emb.load()
        emb.embed_batch(pil_images[:4])   # warmup JIT
        t0 = time.perf_counter()
        out = emb.embed_batch(pil_images)
        t1 = time.perf_counter()
        return out.shape[1], n / (t1 - t0)

    def _time_open_clip():
        import torch, open_clip
        device = _clip_device()
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14-quickgelu", pretrained="openai")
        model = model.to(device).half().eval()
        tensors = [preprocess(img) for img in pil_images]
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            model.encode_image(batch[:4].half())   # warmup
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.encode_image(batch.half()).float()
        t1 = time.perf_counter()
        return out.shape[1], n / (t1 - t0)

    for label, fn in [("mlx", _time_mlx), ("open_clip", _time_open_clip)]:
        try:
            dim, ips = fn()
            marker = "  <-- preferred" if label == preferred_backend else ""
            print(f"{label:<14} {ips:>8.0f}  {dim:>5}{marker}")
        except Exception as e:
            print(f"{label:<14} {'N/A':>8}  {'':>5}  ({e})")

    return 0


def cmd_embed(args) -> int:
    shard_dir = Path(args.shards)
    embed_dir = Path(args.embeddings)

    shards = sorted(shard_dir.glob("*.tar"))
    if not shards:
        log_orch(f"embed: no .tar files in {shard_dir}")
        return 0

    backend = getattr(args, "clip_backend", "auto")

    if getattr(args, "benchmark", False):
        return _run_benchmark(str(shards[0]), backend)

    embed_dir.mkdir(parents=True, exist_ok=True)
    _load_clip(backend)
    batch_size = args.batch_size

    # MLX takes raw PIL images; PyTorch backends need preprocessed tensors.
    _decode_preprocess = (lambda img: img) if _clip_backend == "mlx" else _clip_preprocess

    total = len(shards)
    done  = 0
    done_event = threading.Event()
    _chunk = getattr(args, "chunk", None)

    def _heartbeat():
        while not done_event.is_set():
            write_heartbeat("clip_dedup", _chunk, phase="embed",
                            done=done, total=total,
                            pct=round(done / total * 100, 1) if total else 100)
            time.sleep(30)

    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()

    try:
        # Prefetch next shard while GPU processes current one
        pending = [(i, p) for i, p in enumerate(shards)
                   if not (embed_dir / (p.stem + ".npz")).exists()]

        log_orch(f"embed: {total} shards total, {len(pending)} to process, backend={_clip_backend}")

        with ThreadPoolExecutor(max_workers=2) as loader:
            from collections import deque
            q: deque = deque()
            for k in range(min(2, len(pending))):
                q.append(loader.submit(_decode_shard, str(pending[k][1]), _decode_preprocess))

            for batch_i, (shard_idx, shard_path) in enumerate(pending):
                out_npz = embed_dir / (shard_path.stem + ".npz")
                ids, images = q.popleft().result()

                # Enqueue shard 2 ahead
                nxt = batch_i + 2
                if nxt < len(pending):
                    q.append(loader.submit(_decode_shard, str(pending[nxt][1]), _decode_preprocess))

                if ids is None or not images:
                    done += 1
                    continue

                all_embs = []
                if _clip_backend == "mlx":
                    for i in range(0, len(images), batch_size):
                        e = _mlx_embedder.embed_batch(images[i:i + batch_size])
                        all_embs.append(e)
                else:
                    import torch
                    device = next(_clip_model.parameters()).device
                    with torch.no_grad():
                        for i in range(0, len(images), batch_size):
                            b = torch.stack(images[i:i + batch_size]).to(device)
                            if _clip_backend == "open_clip":
                                e = _clip_model.encode_image(b.half()).float()
                            else:
                                e = _clip_model.get_image_features(pixel_values=b).float()
                            e = e / e.norm(dim=-1, keepdim=True)
                            all_embs.append(e.cpu().numpy())

                emb_arr = np.concatenate(all_embs, axis=0).astype(np.float32)
                tmp_stem = str(out_npz)[:-4] + ".tmp"
                np.savez(tmp_stem, ids=np.array(ids), embeddings=emb_arr)
                os.replace(tmp_stem + ".npz", out_npz)

                log_event("clip_dedup", "embed_shard",
                          shard=shard_path.name, n=len(ids))
                done += 1

    finally:
        done_event.set()

    write_heartbeat("clip_dedup", _chunk, phase="embed", done=total, total=total, pct=100)
    log_orch(f"embed: complete — embeddings in {embed_dir}")
    return 0


# ---------------------------------------------------------------------------
# build-index subcommand
# ---------------------------------------------------------------------------

def cmd_build_index(args) -> int:
    try:
        import faiss
        faiss.omp_set_num_threads(1)
    except ImportError:
        raise RuntimeError("build-index requires faiss-cpu: pip install faiss-cpu")

    embed_dir  = Path(args.embeddings)
    index_path = Path(args.index)
    ids_path   = index_path.with_suffix(".ids")

    npz_files = sorted(embed_dir.glob("*.npz"))
    if not npz_files:
        log_orch(f"build-index: no .npz files in {embed_dir}")
        return 0

    # Load existing state — cumulative across chunks
    if index_path.exists() and ids_path.exists():
        log_orch(f"build-index: extending existing index ({index_path.name})")
        index       = faiss.read_index(str(index_path))
        indexed_ids = set(ids_path.read_text().splitlines())
    else:
        index       = None
        indexed_ids = set()

    new_vecs: list = []
    new_ids:  list = []
    for npz_path in npz_files:
        data      = np.load(npz_path, allow_pickle=False)
        file_ids  = data["ids"].tolist()
        file_embs = data["embeddings"].astype(np.float32)
        for i, fid in enumerate(file_ids):
            if fid not in indexed_ids:
                new_ids.append(fid)
                new_vecs.append(file_embs[i])

    if not new_ids:
        log_orch("build-index: all embeddings already indexed")
        return 0

    vecs = np.stack(new_vecs, axis=0).astype(np.float32)
    dim  = vecs.shape[1]
    if index is None:
        index = faiss.IndexFlatIP(dim)

    index.add(vecs)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    with open(ids_path, "a") as f:
        for fid in new_ids:
            f.write(fid + "\n")

    log_event("clip_dedup", "index_built",
              added=len(new_ids), total=index.ntotal, dim=dim)
    log_orch(f"build-index: +{len(new_ids)} vectors; index total={index.ntotal}")
    return 0


# ---------------------------------------------------------------------------
# find-dups subcommand
# ---------------------------------------------------------------------------

def cmd_find_dups(args) -> int:
    try:
        import faiss
        faiss.omp_set_num_threads(1)
    except ImportError:
        raise RuntimeError("find-dups requires faiss-cpu: pip install faiss-cpu")

    index_path = Path(args.index)
    ids_path   = index_path.with_suffix(".ids")
    out_path   = Path(args.out)
    threshold  = args.threshold

    if not index_path.exists():
        log_orch(f"find-dups: index not found ({index_path}) — skipping", level="warning")
        return 0

    index   = faiss.read_index(str(index_path))
    all_ids = ids_path.read_text().splitlines()
    n       = index.ntotal

    if n == 0:
        log_orch("find-dups: empty index")
        return 0

    # Reconstruct stored vectors from the flat index
    d   = index.d
    all_vecs = np.zeros((n, d), dtype=np.float32)
    index.reconstruct_n(0, n, all_vecs)

    log_orch(f"find-dups: {n} vectors, threshold={threshold}")

    existing_dups: set = set()
    if out_path.exists():
        existing_dups = set(out_path.read_text().splitlines())

    k         = min(5, n)
    new_dups: list = []
    kept_set: set  = set()
    batch_size = 4096

    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        chunk = all_vecs[start:end]
        D, I  = index.search(chunk, k)

        for local_i in range(end - start):
            global_i = start + local_i
            fid = all_ids[global_i]
            if fid in existing_dups:
                continue
            kept_set.add(global_i)
            for rank in range(1, k):
                neighbor = int(I[local_i, rank])
                if neighbor < 0:
                    break
                if float(D[local_i, rank]) >= threshold:
                    nid = all_ids[neighbor]
                    if nid not in existing_dups and neighbor not in kept_set:
                        existing_dups.add(nid)
                        new_dups.append(nid)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if new_dups:
        with open(out_path, "a") as f:
            for fid in new_dups:
                f.write(fid + "\n")

    log_event("clip_dedup", "dups_found",
              new_dups=len(new_dups), total_dups=len(existing_dups), index_size=n)
    log_orch(f"find-dups: {len(new_dups)} new, {len(existing_dups)} total in blocklist")

    if getattr(args, "report_out", None):
        dedup_rate = round(len(existing_dups) / n * 100, 2) if n > 0 else 0.0
        report = {
            "ts":             now_iso(),
            "n_total":        n,
            "n_dups_total":   len(existing_dups),
            "n_new_dups":     len(new_dups),
            "n_kept":         n - len(existing_dups),
            "dedup_rate_pct": dedup_rate,
            "threshold":      threshold,
        }
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
        log_orch(f"find-dups: quality report → {report_path}")

    return 0


# ---------------------------------------------------------------------------
# Pool-level dedup (Track 1 + clean_wds_pool)
# ---------------------------------------------------------------------------

def dedup_wds_tar(
    tar_path: Path,
    index_path: Path,
    ids_path: Path,
    blocklist_path: Path,
    threshold: float = DUP_THRESHOLD,
    backend: str = "auto",
) -> tuple:
    """
    Embed all images in tar_path, search for near-duplicates in the existing
    FAISS index, rewrite the tar keeping only non-duplicate records, and
    extend the index with the new non-duplicate vectors.

    Args:
        tar_path:      Path to the WDS .tar file to dedup.
        index_path:    Path to the cumulative FAISS IndexFlatIP (.faiss).
        ids_path:      Path to the sidecar ID file (.ids) — one ID per line.
        blocklist_path:Path to the duplicate-IDs blocklist (append-safe).
        threshold:     Inner-product score >= this flags a record as duplicate.
        backend:       CLIP backend ("auto" | "open_clip" | "mlx" | "transformers").

    Returns:
        (records_in, records_out) — record counts before and after dedup.

    Thread-safety: acquires _faiss_lock around all index I/O so concurrent
    calls from multiple threads are safe.
    """
    try:
        import faiss
        faiss.omp_set_num_threads(1)
    except ImportError:
        raise RuntimeError("dedup_wds_tar requires faiss-cpu: pip install faiss-cpu")

    tar_path = Path(tar_path)
    index_path = Path(index_path)
    ids_path = Path(ids_path)
    blocklist_path = Path(blocklist_path)

    _load_clip(backend)
    _decode_preprocess = (lambda img: img) if _clip_backend == "mlx" else _clip_preprocess

    # Decode all images from the tar.
    ids, images = _decode_shard(str(tar_path), _decode_preprocess)
    if ids is None:
        raise RuntimeError(f"embed failed for {tar_path.name} — I/O error reading tar")
    if len(ids) == 0:
        return (0, 0)

    records_in = len(ids)

    # Compute CLIP embeddings.
    all_embs: list = []
    batch_size = 512
    if _clip_backend == "mlx":
        for i in range(0, len(images), batch_size):
            e = _mlx_embedder.embed_batch(images[i:i + batch_size])
            all_embs.append(e)
    else:
        import torch
        device = next(_clip_model.parameters()).device
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                b = torch.stack(images[i:i + batch_size]).to(device)
                if _clip_backend == "open_clip":
                    e = _clip_model.encode_image(b.half()).float()
                else:
                    e = _clip_model.get_image_features(pixel_values=b).float()
                e = e / e.norm(dim=-1, keepdim=True)
                all_embs.append(e.cpu().numpy())

    if _clip_backend == "mlx":
        import numpy as _np
        emb_arr = _np.concatenate([_np.array(e) for e in all_embs], axis=0).astype(np.float32)
        # L2-normalise for MLX path (open_clip/transformers paths already normalise above).
        norms = np.linalg.norm(emb_arr, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        emb_arr = (emb_arr / norms).astype(np.float32)
    else:
        emb_arr = np.concatenate(all_embs, axis=0).astype(np.float32)

    dim = emb_arr.shape[1]

    with _faiss_lock:
        # Load or create index.
        if index_path.exists() and ids_path.exists():
            index = faiss.read_index(str(index_path))
            existing_ids = ids_path.read_text().splitlines()
        else:
            index = faiss.IndexFlatIP(dim)
            existing_ids = []

        # Search existing index only (before adding new vectors).
        dup_stems: set = set()
        if index.ntotal > 0:
            D, _I = index.search(emb_arr, k=1)
            for local_i in range(len(ids)):
                if float(D[local_i, 0]) >= threshold:
                    dup_stems.add(ids[local_i])

        # Append dup IDs to blocklist.
        if dup_stems:
            blocklist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(blocklist_path, "a") as _bf:
                for fid in sorted(dup_stems):
                    _bf.write(fid + "\n")

        # Add non-duplicate vectors to index.
        keep_mask = [ids[i] not in dup_stems for i in range(len(ids))]
        kept_ids = [ids[i] for i in range(len(ids)) if keep_mask[i]]
        if kept_ids:
            kept_vecs = emb_arr[[i for i in range(len(ids)) if keep_mask[i]]]
            index.add(kept_vecs)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(index_path))
            with open(ids_path, "a") as _idf:
                for fid in kept_ids:
                    _idf.write(fid + "\n")
        elif not dup_stems:
            # No vectors at all (edge case: all decoded images had errors upstream).
            pass

    records_out = records_in - len(dup_stems)

    # Rewrite tar in place keeping only non-duplicate pairs.
    if dup_stems:
        tmp_path = tar_path.with_suffix(".tar.tmp")
        try:
            with tarfile.open(str(tar_path), "r") as src, \
                 tarfile.open(str(tmp_path), "w") as dst:
                for member in src.getmembers():
                    if not member.isfile():
                        continue
                    stem, _, _ = member.name.rpartition(".")
                    if stem in dup_stems:
                        continue
                    f = src.extractfile(member)
                    if f is not None:
                        dst.addfile(member, f)
            os.replace(str(tmp_path), str(tar_path))
        except Exception:
            try:
                os.unlink(str(tmp_path))
            except OSError:
                pass
            raise

    return (records_in, records_out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap  = argparse.ArgumentParser(description="CLIP deduplication pipeline (V2)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_emb = sub.add_parser("embed", help="Compute CLIP embeddings per shard")
    p_emb.add_argument("--shards",     required=True)
    p_emb.add_argument("--embeddings", required=True)
    p_emb.add_argument("--chunk",      type=int, default=None,
                       help="Chunk number (for heartbeat tracking by pipeline_status)")
    p_emb.add_argument("--batch-size", dest="batch_size", type=int, default=512)
    p_emb.add_argument("--clip-backend", dest="clip_backend",
                       choices=("auto", "mlx", "open_clip", "transformers"),
                       default="auto",
                       help="CLIP backend (default: auto — prefers open_clip>mlx>transformers)")
    p_emb.add_argument("--benchmark", action="store_true",
                       help="Time all available backends on the first shard and exit")

    p_idx = sub.add_parser("build-index", help="Build/extend cumulative FAISS index")
    p_idx.add_argument("--embeddings", required=True)
    p_idx.add_argument("--index",      required=True)

    p_dup = sub.add_parser("find-dups", help="Find near-duplicates, write blocklist")
    p_dup.add_argument("--index",      required=True)
    p_dup.add_argument("--out",        required=True)
    p_dup.add_argument("--threshold",  type=float, default=DUP_THRESHOLD)
    p_dup.add_argument("--report-out", dest="report_out", default=None, metavar="PATH",
                       help="Write JSON quality report to this file")

    args = ap.parse_args()
    handlers = {
        "embed":       cmd_embed,
        "build-index": cmd_build_index,
        "find-dups":   cmd_find_dups,
    }
    sys.exit(handlers[args.cmd](args))


if __name__ == "__main__":
    main()
