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
    DATA_ROOT, STAGING_DIR, TRAIN_DIR, LOG_DIR, HF_CACHE_DIR,
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
    """Return {chunk: (start_tgz, end_tgz)} for the given scale, read from config."""
    raw = config.get("jdb", {}).get("tgz_ranges", {})
    if scale not in raw:
        available = list(raw.keys())
        raise ValueError(
            f"Scale '{scale}' not found in jdb.tgz_ranges. "
            f"Available: {available}. Check your pipeline config."
        )
    return {int(k): tuple(v) for k, v in raw[scale].items()}


def _hf_download_file(repo_id: str, filename: str, local_dir: str) -> Path:
    """Download a small HF file via huggingface_hub (annotation files, etc.)."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        local_dir=local_dir,
    )
    return Path(path)


def _curl_download_hf_file(repo_id: str, filename: str, local_dir: str) -> Path:
    """Download a large HF dataset file using curl with auto-resume and retry.

    Replaces the Python hf_hub_download + stall-watchdog pattern.  curl is
    significantly more reliable for multi-GB files against HuggingFace's CDN:

      -C -               auto-detect existing file size; issue Range: bytes=N-
                         so interrupted downloads resume without any .incomplete
                         file management.
      --speed-limit /    abort and retry if throughput drops below 10 KB/s for
      --speed-time       60 consecutive seconds — the stall case that the Python
                         HTTP stack never detects.
      --retry 15         reconnect and re-issue the range request automatically
      --retry-delay 5    on any connection error or speed-limit abort.
      -L                 follow HF → CDN redirect chain on every attempt.
    """
    import subprocess
    from huggingface_hub import hf_hub_url, get_token

    out_path = Path(local_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    url   = hf_hub_url(repo_id, filename, repo_type="dataset")
    token = get_token()

    cmd = [
        "curl", "-L",
        "--retry", "15",
        "--retry-delay", "5",
        "--retry-connrefused",
        "--speed-limit", "10240",   # bytes/s — abort if below this...
        "--speed-time",  "60",      # ...for this many consecutive seconds
        "-C", "-",                  # resume from current file size
        "-o", str(out_path),
    ]
    if token:
        cmd += ["--header", f"Authorization: Bearer {token}"]
    cmd.append(url)

    # cmd_fresh: same as cmd but without -C - so downloads start from byte 0.
    cmd_fresh = [c for c in cmd if c != "-C" and c != "-"]

    _MAX_FRESH_ATTEMPTS = 5
    for attempt in range(1, _MAX_FRESH_ATTEMPTS + 1):
        log_orch(f"curl: downloading {filename} → {out_path}"
                 + (f" (attempt {attempt})" if attempt > 1 else ""))
        # First attempt uses -C - to resume; subsequent attempts start fresh.
        run_cmd = cmd if attempt == 1 else cmd_fresh
        result = subprocess.run(run_cmd, check=False)

        # Exit code 33 = HTTP 416 Range Not Satisfiable: file already fully
        # downloaded, -C - tried to resume past EOF.  Treat as success.
        if result.returncode == 33:
            if not out_path.exists():
                raise RuntimeError(f"curl finished but output file missing: {out_path}")
            return out_path

        if result.returncode == 0:
            break

        # Exit code 18 = CURLE_PARTIAL_FILE: HF dropped the connection before
        # sending all bytes, or partial on disk is larger than the remote file.
        # Delete partial and retry from scratch.
        if result.returncode == 18:
            log_orch(f"curl exit 18 for {filename} — deleting partial and retrying "
                     f"({attempt}/{_MAX_FRESH_ATTEMPTS})", level="warning")
            try:
                out_path.unlink()
            except OSError:
                pass
            import time as _time
            _time.sleep(10)
            continue

        # Any other non-zero exit: give up immediately.
        raise RuntimeError(
            f"curl download failed (exit {result.returncode}): {filename}"
        )
    else:
        raise RuntimeError(
            f"curl download failed after {_MAX_FRESH_ATTEMPTS} attempts (exit 18): {filename}"
        )

    if not out_path.exists():
        raise RuntimeError(f"curl finished but output file missing: {out_path}")
    return out_path


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
                    _curl_download_hf_file(JDB_REPO_ID, jdb_tgz_filename(i), str(raw_dir))
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
    hf_cache_dir = HF_CACHE_DIR / "datasets"
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_cache_dir))
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))

    # Pre-count total parquet files for progress tracking
    try:
        from huggingface_hub import list_repo_files
        total_parquets = sum(
            1 for f in list_repo_files(WIKIART_REPO, repo_type="dataset")
            if f.endswith(".parquet")
        )
    except Exception:
        total_parquets = 0  # unknown — will show count without %

    # Shared phase state: "download" → "save"
    phase    = ["download"]
    save_n   = [0]
    save_tot = [0]
    stop_ev  = threading.Event()

    def _hb_loop():
        while not stop_ev.wait(15):
            if phase[0] == "download":
                done_p = sum(1 for _ in hf_cache_dir.rglob("*.parquet")) if hf_cache_dir.exists() else 0
                write_heartbeat("download_convert", chunk=chunk,
                                phase="wikiart_download",
                                done=done_p, total=total_parquets,
                                pct=round(done_p / total_parquets * 100) if total_parquets else 0)
            else:
                write_heartbeat("download_convert", chunk=chunk,
                                phase="wikiart_save",
                                done=save_n[0], total=save_tot[0],
                                pct=round(save_n[0] / save_tot[0] * 100) if save_tot[0] else 0)

    hb_thread = threading.Thread(target=_hb_loop, daemon=True)
    hb_thread.start()

    try:
        from datasets import load_dataset
        ds = load_dataset(WIKIART_REPO, split="train", streaming=False)
        total = len(ds)
        start = (chunk - 1) * total // total_chunks
        end   = chunk * total // total_chunks
        slice_ds = ds.select(range(start, end))
        log_orch(f"WikiArt chunk {chunk}: saving records {start}–{end} ({end-start} images)")
        phase[0]    = "save"
        save_tot[0] = end - start

        # Wrap save_to_disk with row-count heartbeat via a monitoring thread
        save_done_ev = threading.Event()
        def _save_monitor():
            while not save_done_ev.wait(15):
                arrow_files = list(out_dir.rglob("*.arrow")) if out_dir.exists() else []
                rows = sum(f.stat().st_size for f in arrow_files) // 4096  # rough estimate
                save_n[0] = min(rows, save_tot[0])
        mon = threading.Thread(target=_save_monitor, daemon=True)
        mon.start()

        slice_ds.save_to_disk(str(out_dir))
        save_done_ev.set()
        save_n[0] = save_tot[0]

        done_sentinel.touch()
        log_event("downloader", "wikiart_done", chunk=chunk,
                  records=end - start, start=start, end=end)
    except ImportError:
        log_orch("WikiArt download requires 'datasets' package — skipping", level="warning")
    except Exception as e:
        log_event("downloader", "wikiart_error", chunk=chunk, error=str(e))
        raise
    finally:
        stop_ev.set()


# ---------------------------------------------------------------------------
# LAION / COYO — verify cold pool and stage hot symlink on first use
# ---------------------------------------------------------------------------

def _stage_cold_pool_symlink(dataset: str, config: dict) -> Path:
    """
    Return the hot raw/{dataset}/ path, creating a symlink to the cold pool
    if the hot path is absent and the cold pool sentinel exists.
    """
    hot  = DATA_ROOT / "raw" / dataset
    cold_root_str = config.get("storage", {}).get("cold_root", "")
    if cold_root_str:
        cold = Path(cold_root_str) / "raw" / dataset
        if not hot.exists() and (cold / ".downloaded").exists():
            hot.parent.mkdir(parents=True, exist_ok=True)
            try:
                hot.symlink_to(cold)
                log_orch(f"{dataset.upper()}: staged cold pool symlink {hot} → {cold}")
            except FileExistsError:
                pass  # another process created it first
    return hot


def check_laion(config: dict) -> bool:
    laion_dir = _stage_cold_pool_symlink("laion", config)
    if not laion_dir.exists() or not any(laion_dir.glob("*.tar")):
        log_orch("WARNING: raw/laion/ is empty — LAION source will be skipped. "
                 "Run: train/.venv/bin/python train/scripts/download_laion_coyo.py --dataset laion",
                 level="warning")
        return False
    count = sum(1 for _ in laion_dir.glob("*.tar"))
    log_orch(f"LAION: {count} source tars present in raw/laion/")
    return True


def check_coyo(config: dict) -> bool:
    coyo_dir = _stage_cold_pool_symlink("coyo", config)
    if not coyo_dir.exists() or not any(coyo_dir.glob("*.tar")):
        log_orch("WARNING: raw/coyo/ is empty — COYO source will be skipped. "
                 "Run: train/.venv/bin/python train/scripts/download_laion_coyo.py --dataset coyo",
                 level="warning")
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
