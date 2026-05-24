#!/usr/bin/env python3
"""
train/scripts/download_laion_coyo.py — One-time cold-pool download for LAION and COYO.

Downloads HF-hosted image-text datasets to cold_root/converted/{laion,coyo}/ as WebDataset
tar shards (paired {id}.jpg + {id}.txt per record). The cold pool survives hot disk
resets; downloader.py stages hot symlinks on first use per chunk.

Supported HF repo formats (auto-detected):
  • Parquet with image column — any HF dataset with "image" (PIL bytes) and a text
    column ("caption", "text", or "TEXT").  Records streamed and packed into WDS tars.
  • WebDataset (pre-built .tar files in the repo) — downloaded via snapshot_download()
    and moved into the cold pool as-is.  No conversion needed.

Usage:
    python download_laion_coyo.py [--dataset laion|coyo|all] [--config PATH]

Idempotent: re-running after completion exits immediately (reads .downloaded sentinel).
"""

import argparse
import io
import itertools
import json
import os
import sys
import tarfile
import threading
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_lib import (
    load_config, log_event, log_orch, write_heartbeat, now_iso, TRAIN_DIR,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RECORDS_PER_SHARD = 10_000       # records per output tar shard
MIN_IMAGE_PX      = 256          # skip images smaller than this in either dimension
MIN_CAPTION_WORDS = 5            # skip captions with fewer words

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _caption_col(features) -> Optional[str]:
    """Return the first recognised caption column name, or None."""
    for name in ("caption", "text", "TEXT", "long_caption", "description"):
        if name in features:
            return name
    return None


def _image_bytes_from_row(row, image_col: str) -> Optional[bytes]:
    """Extract JPEG bytes from a PIL-image column value."""
    img_val = row.get(image_col)
    if img_val is None:
        return None
    # HF datasets delivers PIL Images for Image() feature types
    try:
        from PIL import Image as PILImage
        if isinstance(img_val, dict) and "bytes" in img_val:
            raw = img_val["bytes"]
            if not raw:
                return None
            img = PILImage.open(io.BytesIO(raw)).convert("RGB")
        elif hasattr(img_val, "convert"):
            img = img_val.convert("RGB")
        else:
            return None
        w, h = img.size
        if w < MIN_IMAGE_PX or h < MIN_IMAGE_PX:
            return None
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    except Exception:
        return None


def _add_to_tar(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def _count_shard_records(path: Path) -> int:
    """Count .jpg members in a WDS tar shard; returns 0 on any error."""
    try:
        with tarfile.open(path, "r") as tf:
            return sum(1 for m in tf if m.name.endswith(".jpg"))
    except Exception:
        return 0


def _sentinel_path(out_dir: Path) -> Path:
    return out_dir / ".downloaded"


def _read_sentinel(out_dir: Path) -> Optional[dict]:
    p = _sentinel_path(out_dir)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return None


def _write_sentinel(out_dir: Path, hf_repo: str, shards: int, records: int) -> None:
    _sentinel_path(out_dir).write_text(json.dumps({
        "completed_at": now_iso(),
        "hf_repo": hf_repo,
        "shards": shards,
        "records": records,
    }))


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _repo_is_wds(repo_id: str) -> bool:
    """Return True if the HF repo stores pre-built WebDataset .tar files."""
    try:
        from huggingface_hub import list_repo_files
        for f in list_repo_files(repo_id, repo_type="dataset"):
            if f.endswith(".tar"):
                return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# WDS passthrough download (repo stores pre-built .tar files)
# ---------------------------------------------------------------------------

def _download_wds(repo_id: str, out_dir: Path, dataset: str) -> tuple[int, int]:
    """
    Download pre-built WDS tars from HF repo via snapshot_download.
    Returns (shard_count, record_count_estimate).
    """
    from huggingface_hub import snapshot_download

    log_orch(f"{dataset.upper()}: downloading WDS tars from {repo_id}")
    tmp_dir = out_dir.parent / f".{dataset}_hf_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(tmp_dir),
        ignore_patterns=["*.parquet", "*.json", "*.md", "README*"],
    )

    # Move downloaded tars into cold pool
    tars = sorted(tmp_dir.rglob("*.tar"))
    if not tars:
        raise RuntimeError(
            f"{dataset.upper()}: snapshot_download from {repo_id} returned no .tar files. "
            f"Verify the repo contains WebDataset shards or switch to parquet format."
        )
    for i, src in enumerate(tars):
        dst = out_dir / f"shard-{i:06d}.tar"
        src.rename(dst)
        log_orch(f"{dataset.upper()}: staged shard {i+1}/{len(tars)}: {dst.name}")

    # Clean up tmp dir
    import shutil
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    return len(tars), len(tars) * RECORDS_PER_SHARD  # estimate


# ---------------------------------------------------------------------------
# Parquet → WDS conversion
# ---------------------------------------------------------------------------

URL_LIST_SHARD_SIZE = 5_000      # records per shard for URL-list datasets
URL_LIST_WORKERS    = 128         # parallel HTTP fetches
URL_LIST_TIMEOUT    = 10         # per-URL HTTP timeout (seconds)


def _fetch_url_to_jpeg(url: str) -> Optional[bytes]:
    """Download a single URL, re-encode as JPEG bytes, or return None on failure."""
    import requests
    from PIL import Image as PILImage
    try:
        r = requests.get(url, timeout=URL_LIST_TIMEOUT, stream=True,
                         headers={"User-Agent": "iris-pipeline/1.0"})
        r.raise_for_status()
        raw = r.content
        if not raw:
            return None
        img = PILImage.open(io.BytesIO(raw)).convert("RGB")
        w, h = img.size
        if w < MIN_IMAGE_PX or h < MIN_IMAGE_PX:
            return None
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    except Exception:
        return None


def _download_url_list(repo_id: str, out_dir: Path, dataset: str,
                        url_col: str, cap_col: str) -> tuple[int, int]:
    """
    Stream a URL-list parquet dataset (e.g. laion-art) and write WDS shards
    by fetching each URL via HTTP. Returns (shard_count, record_count).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datasets import load_dataset

    log_orch(f"{dataset.upper()}: streaming URL-list from {repo_id} "
             f"(col URL='{url_col}', caption='{cap_col}', "
             f"workers={URL_LIST_WORKERS}, shard_size={URL_LIST_SHARD_SIZE})")

    ds = load_dataset(repo_id, split="train", streaming=True, trust_remote_code=False)

    # Resume logic: scan output dir for existing shards. Complete shards
    # (exactly URL_LIST_SHARD_SIZE records) are kept; partial shards beyond
    # the complete range are deleted. Resume index uses max(complete shard num) + 1
    # so a missing low-numbered shard (e.g. shard-000000) does not cause
    # the next write to overwrite a complete shard.
    import re as _re
    def _shard_num(path):
        m = _re.search(r"shard-(\d+)\.tar$", str(path))
        return int(m.group(1)) if m else -1

    existing_shards = sorted(out_dir.glob("shard-*.tar"))
    complete_paths = []
    partial_paths  = []
    for shard_path in existing_shards:
        n = _count_shard_records(shard_path)
        if n == URL_LIST_SHARD_SIZE:
            complete_paths.append(shard_path)
        else:
            partial_paths.append(shard_path)

    complete_nums = [_shard_num(p) for p in complete_paths if _shard_num(p) >= 0]
    resume_shards = (max(complete_nums) + 1) if complete_nums else 0

    partial_removed = 0
    for shard_path in partial_paths:
        if _shard_num(shard_path) >= resume_shards:
            shard_path.unlink()
            partial_removed += 1

    skip_rows = resume_shards * URL_LIST_SHARD_SIZE
    if resume_shards > 0:
        log_orch(f"{dataset.upper()}: [resume] found {len(complete_paths)} complete shards, "
                 f"shard_idx={resume_shards}, skipping {skip_rows} rows "
                 f"(removed {partial_removed} partial)")
        ds = itertools.islice(ds, skip_rows, None)
    else:
        log_orch(f"{dataset.upper()}: [resume] starting fresh "
                 f"(removed {partial_removed} partial shards)")

    shard_idx    = resume_shards
    rec_in_shard = 0
    total_recs   = skip_rows
    skipped      = 0
    seen         = 0

    current_tf: Optional[tarfile.TarFile] = None
    current_path: Optional[Path] = None

    def _open_shard():
        nonlocal current_tf, current_path, rec_in_shard
        current_path = out_dir / f"shard-{shard_idx:06d}.tar"
        current_tf   = tarfile.open(current_path, "w")
        rec_in_shard = 0

    def _close_shard():
        nonlocal current_tf, current_path, shard_idx
        if current_tf is not None:
            current_tf.close()
            log_orch(f"{dataset.upper()}: closed shard {shard_idx:06d} "
                     f"({rec_in_shard} records) → {current_path.name}")
            shard_idx += 1
            current_tf = None
            current_path = None

    _open_shard()

    # Stream rows; submit URL fetches in batches so memory stays bounded.
    BATCH = URL_LIST_WORKERS * 8
    pending_rows: list[tuple[str, str]] = []  # (url, caption)

    def _flush(batch: list[tuple[str, str]]) -> None:
        nonlocal rec_in_shard, total_recs, skipped
        if not batch:
            return
        with ThreadPoolExecutor(max_workers=URL_LIST_WORKERS) as pool:
            futs = {pool.submit(_fetch_url_to_jpeg, url): (url, cap) for url, cap in batch}
            for fut in as_completed(futs):
                url, cap = futs[fut]
                img_bytes = fut.result()
                if img_bytes is None:
                    skipped += 1
                    continue
                idx = total_recs % URL_LIST_SHARD_SIZE
                stem = f"{idx:08d}"
                _add_to_tar(current_tf, f"{stem}.jpg", img_bytes)
                _add_to_tar(current_tf, f"{stem}.txt", cap.encode("utf-8"))
                rec_in_shard += 1
                total_recs   += 1
                if rec_in_shard >= URL_LIST_SHARD_SIZE:
                    _close_shard()
                    _open_shard()

    hb_last = time.monotonic()
    for row in ds:
        seen += 1
        url = row.get(url_col)
        cap = str(row.get(cap_col, "")).strip()
        if not url or len(cap.split()) < MIN_CAPTION_WORDS:
            skipped += 1
            continue
        pending_rows.append((url, cap))
        if len(pending_rows) >= BATCH:
            _flush(pending_rows)
            pending_rows = []
            now = time.monotonic()
            if now - hb_last > 30:
                write_heartbeat("download_laion", chunk=None,
                                dataset=dataset, hf_repo=repo_id,
                                seen=seen, downloaded=total_recs,
                                skipped=skipped, shards=shard_idx,
                                done=total_recs, total=0, pct=0)
                hb_last = now

    _flush(pending_rows)

    if rec_in_shard > 0:
        _close_shard()
    elif current_tf is not None:
        current_tf.close()
        if current_path and current_path.exists():
            current_path.unlink()

    log_orch(f"{dataset.upper()}: URL-list complete — {shard_idx} shards, "
             f"{total_recs} records (seen {seen}, skipped {skipped})")
    return shard_idx, total_recs


def _download_parquet(repo_id: str, out_dir: Path, dataset: str) -> tuple[int, int]:
    """
    Stream a parquet-based HF dataset and write WDS tar shards to out_dir.
    Returns (shard_count, record_count).
    """
    from datasets import load_dataset

    log_orch(f"{dataset.upper()}: streaming parquet records from {repo_id}")

    ds = load_dataset(repo_id, split="train", streaming=True, trust_remote_code=False)
    features = ds.features if hasattr(ds, "features") and ds.features else {}

    # Detect columns
    image_col = "image" if "image" in features else None
    cap_col   = _caption_col(features)

    # URL-list datasets (laion-art etc.) have no 'image' column — route to HTTP downloader
    if image_col is None:
        url_col = next((c for c in ("URL", "url", "image_url") if c in features), None)
        if url_col is not None:
            if cap_col is None:
                raise RuntimeError(
                    f"{dataset.upper()}: URL-list dataset {repo_id} has no recognised caption column. "
                    f"Available columns: {list(features.keys())}"
                )
            return _download_url_list(repo_id, out_dir, dataset, url_col, cap_col)
        raise RuntimeError(
            f"{dataset.upper()}: no 'image' or URL column found in {repo_id}. "
            f"Available columns: {list(features.keys())}"
        )
    if cap_col is None:
        raise RuntimeError(
            f"{dataset.upper()}: no recognised caption column in {repo_id}. "
            f"Available columns: {list(features.keys())}"
        )

    log_orch(f"{dataset.upper()}: using columns image='{image_col}', caption='{cap_col}'")

    shard_idx   = 0
    rec_in_shard = 0
    total_recs  = 0
    skipped     = 0

    current_tf: Optional[tarfile.TarFile] = None
    current_path: Optional[Path] = None

    def _close_shard():
        nonlocal current_tf, current_path, shard_idx
        if current_tf is not None:
            current_tf.close()
            log_orch(f"{dataset.upper()}: closed shard {shard_idx:06d} "
                     f"({rec_in_shard} records) → {current_path.name}")
            shard_idx += 1
            current_tf = None
            current_path = None

    def _open_shard():
        nonlocal current_tf, current_path, rec_in_shard
        current_path = out_dir / f"shard-{shard_idx:06d}.tar"
        current_tf   = tarfile.open(current_path, "w")
        rec_in_shard = 0

    _open_shard()

    for row in ds:
        img_bytes = _image_bytes_from_row(row, image_col)
        if img_bytes is None:
            skipped += 1
            continue

        caption = str(row.get(cap_col, "")).strip()
        if len(caption.split()) < MIN_CAPTION_WORDS:
            skipped += 1
            continue

        rec_id = f"{shard_idx:06d}{rec_in_shard:04d}"
        _add_to_tar(current_tf, f"{rec_id}.jpg", img_bytes)
        _add_to_tar(current_tf, f"{rec_id}.txt", caption.encode("utf-8"))

        rec_in_shard += 1
        total_recs   += 1

        if rec_in_shard >= RECORDS_PER_SHARD:
            _close_shard()
            _open_shard()

    # Flush last (potentially partial) shard
    if rec_in_shard > 0:
        _close_shard()
    elif current_tf is not None:
        current_tf.close()
        # Remove empty shard file
        if current_path and current_path.exists():
            current_path.unlink()

    log_orch(f"{dataset.upper()}: wrote {shard_idx} shards, "
             f"{total_recs} records (skipped {skipped})")
    return shard_idx, total_recs


# ---------------------------------------------------------------------------
# Main per-dataset download orchestration
# ---------------------------------------------------------------------------

def download_dataset(dataset: str, hf_repo: str, cold_root: Path) -> None:
    """Download one dataset (laion or coyo) to cold_root/converted/{dataset}/."""
    out_dir = cold_root / "converted" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = _read_sentinel(out_dir)
    if existing is not None:
        log_orch(
            f"{dataset.upper()}: cold pool already complete "
            f"(repo={existing.get('hf_repo')}, shards={existing.get('shards')}, "
            f"records={existing.get('records')}, completed_at={existing.get('completed_at')})"
        )
        return

    log_event("download_laion_coyo", "start", dataset=dataset, hf_repo=hf_repo)
    t0 = time.monotonic()

    # Heartbeat thread
    done_shards = [0]
    stop_ev = threading.Event()

    def _hb_loop():
        while not stop_ev.wait(30):
            current = sum(1 for _ in out_dir.glob("shard-*.tar")) if out_dir.exists() else 0
            done_shards[0] = current
            write_heartbeat("download_laion_coyo", chunk=None,
                            dataset=dataset, hf_repo=hf_repo,
                            done=current, total=0, pct=0)
    hb = threading.Thread(target=_hb_loop, daemon=True)
    hb.start()

    try:
        is_wds = _repo_is_wds(hf_repo)
        if is_wds:
            shards, records = _download_wds(hf_repo, out_dir, dataset)
        else:
            shards, records = _download_parquet(hf_repo, out_dir, dataset)

        elapsed = round(time.monotonic() - t0)
        _write_sentinel(out_dir, hf_repo, shards, records)
        log_event("download_laion_coyo", "done",
                  dataset=dataset, hf_repo=hf_repo,
                  shards=shards, records=records, elapsed_secs=elapsed)
        log_orch(
            f"{dataset.upper()}: download complete — "
            f"{shards} shards, {records:,} records in {elapsed}s"
        )
    except Exception as e:
        log_event("download_laion_coyo", "error", dataset=dataset, error=str(e))
        log_orch(f"{dataset.upper()}: download failed: {e}", level="error")
        raise
    finally:
        stop_ev.set()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download LAION / COYO image-text datasets to cold pool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--dataset", choices=["laion", "coyo", "all"], default="all",
        help="Which dataset(s) to download (default: all)",
    )
    ap.add_argument(
        "--config",
        default=str(TRAIN_DIR / "configs" / "v2_pipeline.yaml"),
        help="Pipeline config YAML",
    )
    args = ap.parse_args()

    cfg      = load_config(args.config)
    sources  = cfg.get("data_sources", {})
    storage  = cfg.get("storage", {})
    cold_root = Path(os.environ.get("COLD_ROOT")
                     or storage.get("cold_root", "/Volumes/16TBCold"))

    if not cold_root.exists():
        print(f"ERROR: cold_root not mounted: {cold_root}", file=sys.stderr)
        return 1

    datasets_to_run = []
    if args.dataset in ("laion", "all"):
        repo = sources.get("laion_hf_repo", "")
        if repo:
            datasets_to_run.append(("laion", repo))
        else:
            log_orch("LAION: laion_hf_repo not set in config — skipping")

    if args.dataset in ("coyo", "all"):
        repo = sources.get("coyo_hf_repo", "")
        if repo:
            datasets_to_run.append(("coyo", repo))
        else:
            log_orch("COYO: coyo_hf_repo not set in config — skipping")

    if not datasets_to_run:
        log_orch("No datasets configured — nothing to do. "
                 "Set laion_hf_repo and/or coyo_hf_repo in data_sources config.")
        return 0

    for dataset, hf_repo in datasets_to_run:
        download_dataset(dataset, hf_repo, cold_root)

    return 0


if __name__ == "__main__":
    sys.exit(main())
