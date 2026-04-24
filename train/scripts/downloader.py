#!/usr/bin/env python3
"""
train/scripts/downloader.py — V2 per-source download worker.

Handles all download work for a single chunk. JourneyDB uses the
producer-consumer pattern (MLX-19): downloads one tgz at a time, signals
ready, so converter.py can process each tgz immediately and delete the raw
file before the next download starts. This keeps peak disk usage to ~2 tgzs
in-flight rather than all N simultaneously.

WikiArt is downloaded from HuggingFace datasets (parquet → images).
LAION and COYO: their source data (parquet manifests) must already exist in
raw/laion/ and raw/coyo/ — downloader verifies presence only.

Usage (called by orchestrator):
    python train/scripts/downloader.py --chunk 1 --config train/configs/v2_pipeline.yaml

Usage (standalone JDB only):
    python train/scripts/downloader.py --chunk 1 --jdb-only
"""

import argparse
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_lib import (
    DATA_ROOT, STAGING_DIR, TRAIN_DIR, LOG_DIR,
    write_heartbeat, log_event, log_orch, load_config, now_iso,
)


# ---------------------------------------------------------------------------
# JourneyDB — producer-consumer download (MLX-19)
# ---------------------------------------------------------------------------

JDB_REPO_ID  = "JourneyDB/JourneyDB"
JDB_ANNO_FILE = "data/train/train_anno_realease_repath.jsonl.tgz"

def jdb_tgz_filename(idx: int) -> str:
    return f"data/train/imgs/{idx:03d}.tgz"


def jdb_tgz_ranges(config: dict, scale: str = "all-in") -> dict[int, tuple[int, int]]:
    """Return {chunk: (start_tgz, end_tgz)} for the given scale."""
    raw = config.get("jdb", {}).get("tgz_ranges", {})
    # New format: nested by scale
    if scale in raw:
        return {int(k): tuple(v) for k, v in raw[scale].items()}
    # Legacy flat format or all-in fallback
    fallback = raw if raw and not any(isinstance(v, dict) for v in raw.values()) else {
        1: [0, 49], 2: [50, 99], 3: [100, 149], 4: [150, 201]
    }
    return {int(k): tuple(v) for k, v in fallback.items()}


_STALL_SECS = 600  # 10 min with no byte progress → abort


def _incomplete_bytes(cache_dir: Path) -> int:
    """Sum of sizes of all .incomplete files in cache_dir (HF in-progress downloads)."""
    total = 0
    try:
        for p in cache_dir.rglob("*.incomplete"):
            try:
                total += p.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return total


def _hf_download_file(repo_id: str, filename: str, local_dir: str) -> Path:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        local_dir=local_dir,
    )
    return Path(path)


def _hf_download_file_guarded(repo_id: str, filename: str, local_dir: str) -> Path:
    """Wraps _hf_download_file with a stall watchdog.

    If the HF .incomplete file stops growing for _STALL_SECS, sends SIGTERM to
    the current process so the orchestrator detects failure and retries.
    HF resumes from the partial file on next attempt.
    """
    import os
    import signal

    cache_dir = Path(local_dir) / ".cache" / "huggingface" / "download"
    stop_ev   = threading.Event()

    def _watchdog():
        last_size   = -1
        last_change = time.time()
        while not stop_ev.wait(60):
            size = _incomplete_bytes(cache_dir)
            if size != last_size:
                last_size   = size
                last_change = time.time()
            elif size > 0 and time.time() - last_change > _STALL_SECS:
                log_orch(
                    f"Download stall: {size / 1e9:.1f} GB downloaded, "
                    f"no progress for {_STALL_SECS}s — aborting for retry"
                )
                os.kill(os.getpid(), signal.SIGTERM)
                return

    t = threading.Thread(target=_watchdog, daemon=True)
    t.start()
    try:
        return _hf_download_file(repo_id, filename, local_dir)
    finally:
        stop_ev.set()


def download_jdb_annotation(raw_dir: Path) -> None:
    """Download annotation file if not already present."""
    anno_dest = raw_dir / "data" / "train" / "train_anno_realease_repath.jsonl.tgz"
    if anno_dest.exists():
        return
    log_orch("Downloading JDB annotation file...")
    _hf_download_file(JDB_REPO_ID, JDB_ANNO_FILE, str(raw_dir))
    log_orch(f"Annotation downloaded → {anno_dest}")


def run_jdb_download(chunk: int, config: dict, scale: str = "all-in") -> None:
    """Download JDB tgzs for the given chunk using producer-consumer pattern."""
    ranges = jdb_tgz_ranges(config, scale)
    start, end = ranges[chunk]
    tgz_range = range(start, end + 1)

    raw_dir = STAGING_DIR / f"chunk{chunk}" / "raw" / "journeydb"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sentinel_dir = raw_dir / ".tgz_state"
    sentinel_dir.mkdir(exist_ok=True)

    # Download annotation file first (converter needs it)
    download_jdb_annotation(raw_dir)

    total = len(tgz_range)
    log_orch(f"JDB chunk {chunk}: downloading {total} tgzs ({start:03d}–{end:03d})")

    done_event = threading.Event()

    def producer():
        for i in tgz_range:
            converted = sentinel_dir / f"{i:03d}.converted"
            ready     = sentinel_dir / f"{i:03d}.ready"
            if converted.exists():
                log_event("downloader", "skip", chunk=chunk, tgz=i,
                          reason="already_converted")
                continue
            if not ready.exists():
                log_event("downloader", "download_start", chunk=chunk, tgz=i)
                t0 = time.time()
                try:
                    _hf_download_file_guarded(JDB_REPO_ID, jdb_tgz_filename(i), str(raw_dir))
                    ready.touch()
                    elapsed = time.time() - t0
                    log_event("downloader", "download_done", chunk=chunk, tgz=i,
                              elapsed_sec=round(elapsed, 1))
                except Exception as e:
                    log_event("downloader", "download_error", chunk=chunk, tgz=i,
                              error=str(e))
                    raise
        done_event.set()

    def _heartbeat_loop():
        done_total = 0
        while not done_event.is_set():
            done_total = sum(
                1 for i in tgz_range
                if (sentinel_dir / f"{i:03d}.converted").exists()
            )
            write_heartbeat("downloader", chunk=chunk,
                            done=done_total, total=total,
                            pct=round(done_total / total * 100, 1))
            time.sleep(30)

    hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    hb_thread.start()

    producer()  # run synchronously; converter runs in separate orchestrator-launched process

    # Final heartbeat
    done_total = sum(1 for i in tgz_range
                     if (sentinel_dir / f"{i:03d}.converted").exists())
    write_heartbeat("downloader", chunk=chunk, done=done_total, total=total, pct=100)
    log_orch(f"JDB chunk {chunk}: all {total} tgzs downloaded")


# ---------------------------------------------------------------------------
# WikiArt — download from HuggingFace datasets
# ---------------------------------------------------------------------------

WIKIART_REPO = "huggan/wikiart"

def run_wikiart_download(chunk: int, config: dict) -> None:
    """Download WikiArt parquet files for the given chunk slice."""
    total_chunks = config.get("chunks", 4)
    out_dir = STAGING_DIR / f"chunk{chunk}" / "raw" / "wikiart"
    out_dir.mkdir(parents=True, exist_ok=True)

    done_sentinel = out_dir / ".download_done"
    if done_sentinel.exists():
        log_orch(f"WikiArt chunk {chunk}: already downloaded")
        return

    log_orch(f"WikiArt chunk {chunk}: downloading from HuggingFace ({WIKIART_REPO})")
    # Route HuggingFace intermediate cache to the SSD to avoid filling local disk.
    hf_cache = str(STAGING_DIR / "hf_cache")
    os.environ.setdefault("HF_DATASETS_CACHE", hf_cache)
    os.environ.setdefault("HF_HOME", str(STAGING_DIR / "hf_home"))
    try:
        from datasets import load_dataset
        ds = load_dataset(WIKIART_REPO, split="train", streaming=False)
        total = len(ds)
        start = (chunk - 1) * total // total_chunks
        end   = chunk * total // total_chunks
        slice_ds = ds.select(range(start, end))
        log_orch(f"WikiArt chunk {chunk}: saving records {start}–{end} ({end-start} images)")
        slice_ds.save_to_disk(str(out_dir))
        done_sentinel.touch()
        log_event("downloader", "wikiart_done", chunk=chunk,
                  records=end - start, start=start, end=end)
    except ImportError:
        log_orch("WikiArt download requires 'datasets' package — skipping", level="warning")
    except Exception as e:
        log_event("downloader", "wikiart_error", chunk=chunk, error=str(e))
        raise


# ---------------------------------------------------------------------------
# LAION / COYO — verify manifests exist (images fetched during build_shards)
# ---------------------------------------------------------------------------

def check_laion(config: dict) -> bool:
    laion_dir = DATA_ROOT / "raw" / "laion"
    if not laion_dir.exists() or not any(laion_dir.iterdir()):
        log_orch("WARNING: raw/laion/ is empty — LAION source will be skipped", level="warning")
        return False
    count = sum(1 for _ in laion_dir.glob("*.tar"))
    log_orch(f"LAION: {count} source tars present in raw/laion/")
    return True


def check_coyo(config: dict) -> bool:
    coyo_dir = DATA_ROOT / "raw" / "coyo"
    if not coyo_dir.exists() or not any(coyo_dir.iterdir()):
        log_orch("WARNING: raw/coyo/ is empty — COYO source will be skipped", level="warning")
        return False
    count = sum(1 for _ in coyo_dir.glob("*.tar"))
    log_orch(f"COYO: {count} source tars present in raw/coyo/")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="V2 download worker")
    ap.add_argument("--chunk",    type=int, required=True)
    ap.add_argument("--config",   default=str(TRAIN_DIR / "configs" / "v2_pipeline.yaml"))
    ap.add_argument("--scale",    default="all-in",
                    choices=["smoke", "small", "medium", "large", "all-in"])
    ap.add_argument("--jdb-only", action="store_true")
    ap.add_argument("--no-jdb",   action="store_true")
    args = ap.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        config = {}

    chunk = args.chunk
    scale = args.scale
    log_orch(f"Downloader starting for chunk {chunk} scale={scale}")

    if not args.no_jdb:
        run_jdb_download(chunk, config, scale)

    if not args.jdb_only:
        run_wikiart_download(chunk, config)
        check_laion(config)
        check_coyo(config)

    log_orch(f"Downloader complete for chunk {chunk}")


if __name__ == "__main__":
    main()
