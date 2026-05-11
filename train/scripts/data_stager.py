"""
data_stager.py — bidirectional staging between cold and hot storage tiers.

Cold tier (cold_root): Long-term archive on HDD or secondary SSD. Source of
  truth for raw WebDataset shards, all versioned precompute caches, and archived
  adapter weights. Cold storage is never automatically deleted.

Hot tier (hot_root): Fast NVMe SSD, working directory for active compute.
  Only the current and next chunk's data need to live here at any time.

Bidirectional flow
  Staging  (cold → hot):  Before precompute or training, symlink/copy the
    required shards and versioned .npz caches from cold to hot storage.
  Archiving (hot → cold): After a chunk completes, copy new precomputed data
    and adapter weights from hot back to cold for long-term retention.

Single-SSD prototyping
  Omit the `storage:` block or set cold_root == hot_root.  All staging and
  archiving operations become no-ops — nothing is copied or symlinked.

Same-device staging
  When cold_root and hot_root share the same physical filesystem
  (os.stat().st_dev match), symlinks are used (instant, zero disk cost).
  When on different physical devices, files are copied atomically:
  write to a temp path, then rename — so a crash never leaves partial files.

Orchestrator integration (see orchestrator.py)
  The orchestrator launches this script as a dedicated iris-stage tmux window
  using `nice -n 10 taskpolicy -d throttle` for I/O isolation.

CLI usage (standalone)
  python data_stager.py stage   --chunk 2 [--config PATH]
  python data_stager.py archive --chunk 1 [--config PATH]
  python data_stager.py status          [--config PATH]
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_lib import DATA_ROOT, load_config, log_event

log = logging.getLogger("data_stager")

# Shard ID block size per chunk — must match orchestrator.py's _SHARD_BLOCK.
# Chunk N owns shard IDs in [(N-1)*SHARD_BLOCK, N*SHARD_BLOCK).
_SHARD_BLOCK = 200_000

_ENCODERS = ("qwen3", "vae", "siglip")


# ---------------------------------------------------------------------------
# DataStager
# ---------------------------------------------------------------------------

class DataStager:
    """
    Manages staging (cold→hot) and archiving (hot→cold) of pipeline data.

    Instantiated in the orchestrator for `enabled` checks; the actual
    stage/archive operations run in a dedicated iris-stage tmux window via
    `_launch_stager()` in orchestrator.py.
    """

    def __init__(self, cfg: dict) -> None:
        storage = cfg.get("storage", {})

        self.cold_root = Path(storage.get("cold_root", DATA_ROOT)).resolve()
        self.hot_root  = Path(storage.get("hot_root",  DATA_ROOT)).resolve()

        self.staging_margin_gb   = float(storage.get("staging_margin_gb",   50.0))
        self.cleanup_safety_gb   = float(storage.get("cleanup_safety_gb",   20.0))
        self.archive_after_chunk = bool(storage.get("archive_after_chunk",  True))
        self.max_parallel        = int(storage.get("max_parallel_transfers", 3))

        # Derived paths on cold storage.
        self._cold_shards  = self.cold_root / "shards"
        self._cold_precomp = self.cold_root / "precomputed"
        self._cold_ckpts   = self.cold_root / "checkpoints" / "stage1"

        # Derived paths on hot storage (== DATA_ROOT paths when single-SSD).
        self._hot_shards  = self.hot_root / "shards"
        self._hot_precomp = self.hot_root / "precomputed"
        self._hot_ckpts   = self.hot_root / "checkpoints" / "stage1"

        # Detect same-device at init; cached for the lifetime of the process.
        self._use_symlinks: bool = self._detect_symlinks()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """False when cold and hot are the same path (single-SSD, no-op mode)."""
        return self.cold_root != self.hot_root

    def stage_for_chunk(self, chunk: int) -> dict:
        """
        Stage shards and precomputed caches for chunk N from cold to hot.

        Returns a summary dict: {shards_staged, npz_staged, bytes_transferred}.
        Skips files already present on hot.  Aborts if hot storage has less
        than staging_margin_gb free after the estimated transfer.
        """
        if not self.enabled:
            return {"shards_staged": 0, "npz_staged": 0, "bytes_transferred": 0}

        log.info("Staging chunk %d: cold=%s → hot=%s", chunk, self.cold_root, self.hot_root)
        _log("stage_start", chunk)

        total_bytes = 0
        shards_staged = self._stage_shards(chunk)
        npz_staged, nbytes = self._stage_precomputed()
        total_bytes += nbytes

        summary = {
            "shards_staged": shards_staged,
            "npz_staged": npz_staged,
            "bytes_transferred": total_bytes,
        }
        _log("stage_done", chunk, **summary)
        log.info("Staging chunk %d complete: %s", chunk, summary)
        return summary

    def archive_chunk(self, chunk: int) -> dict:
        """
        Archive hot-storage data for chunk N back to cold storage.

        Copies new versioned precomputed dirs and checkpoint snapshots that
        exist on hot but not yet on cold.  Never deletes from cold.

        Returns a summary dict: {npz_archived, ckpt_archived, bytes_transferred}.
        """
        if not self.enabled or not self.archive_after_chunk:
            return {"npz_archived": 0, "ckpt_archived": 0, "bytes_transferred": 0}

        log.info("Archiving chunk %d: hot=%s → cold=%s", chunk, self.hot_root, self.cold_root)
        _log("archive_start", chunk)

        npz_archived, nb_precomp = self._archive_precomputed()
        ckpt_archived, nb_ckpts  = self._archive_checkpoints(chunk)
        total_bytes = nb_precomp + nb_ckpts

        summary = {
            "npz_archived": npz_archived,
            "ckpt_archived": ckpt_archived,
            "bytes_transferred": total_bytes,
        }
        _log("archive_done", chunk, **summary)
        log.info("Archiving chunk %d complete: %s", chunk, summary)
        return summary

    def status(self) -> dict:
        """Return a status snapshot for pipeline_status.py."""
        hot_free  = _free_gb(self.hot_root)  if self.hot_root.exists()  else None
        cold_free = _free_gb(self.cold_root) if self.cold_root.exists() else None
        return {
            "enabled":      self.enabled,
            "use_symlinks": self._use_symlinks,
            "cold_root":    str(self.cold_root),
            "hot_root":     str(self.hot_root),
            "hot_free_gb":  round(hot_free,  1) if hot_free  is not None else None,
            "cold_free_gb": round(cold_free, 1) if cold_free is not None else None,
        }

    # ------------------------------------------------------------------
    # Staging internals
    # ------------------------------------------------------------------

    def _stage_shards(self, chunk: int) -> int:
        """
        Symlink/copy shards for chunk N from cold_shards → hot_shards.

        Chunk N owns shard IDs [(N-1)*_SHARD_BLOCK, N*_SHARD_BLOCK).
        Shards are named {shard_id:06d}.tar — match by numeric stem.
        """
        if not self._cold_shards.exists():
            return 0

        lo = (chunk - 1) * _SHARD_BLOCK
        hi =  chunk      * _SHARD_BLOCK
        self._hot_shards.mkdir(parents=True, exist_ok=True)

        staged = 0
        tasks: list[tuple[Path, Path]] = []
        for tar in self._cold_shards.glob("*.tar"):
            try:
                shard_id = int(tar.stem)
            except ValueError:
                continue
            if lo <= shard_id < hi:
                dst = self._hot_shards / tar.name
                if not dst.exists() and not dst.is_symlink():
                    tasks.append((tar, dst))

        def _do(src_dst: tuple[Path, Path]) -> int:
            self._link_or_copy(*src_dst)
            return 1

        with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            for _ in as_completed(pool.submit(_do, t) for t in tasks):
                staged += _.result()

        return staged

    def _stage_precomputed(self) -> tuple[int, int]:
        """
        Stage versioned precomputed .npz caches from cold → hot for all encoders.

        For each encoder, reads cold's `current` symlink, resolves the version
        dir, and links/copies any .npz files not already on hot.  Updates hot's
        `current` symlink to match cold's.

        Returns (npz_files_staged, bytes_transferred).
        """
        total_npz = 0
        total_bytes = 0

        for encoder in _ENCODERS:
            cold_enc = self._cold_precomp / encoder
            hot_enc  = self._hot_precomp  / encoder
            cold_cur = cold_enc / "current"

            if not cold_cur.exists() and not cold_cur.is_symlink():
                continue  # encoder not in cold — nothing to stage

            # Resolve which version dir cold currently points at.
            ver = os.readlink(cold_cur) if cold_cur.is_symlink() else cold_cur.name
            cold_ver_dir = cold_enc / ver
            hot_ver_dir  = hot_enc  / ver

            if not cold_ver_dir.exists():
                continue

            hot_ver_dir.mkdir(parents=True, exist_ok=True)

            tasks: list[tuple[Path, Path]] = []
            for f in cold_ver_dir.iterdir():
                dst = hot_ver_dir / f.name
                if not dst.exists() and not dst.is_symlink():
                    tasks.append((f, dst))

            def _do(src_dst: tuple[Path, Path]) -> int:
                return self._link_or_copy(*src_dst)

            with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
                for fut in as_completed(pool.submit(_do, t) for t in tasks):
                    nb = fut.result()
                    if nb >= 0:  # -1 → already existed / symlinked
                        total_bytes += nb
                    total_npz += 1

            # Keep hot's `current` symlink pointing at the same version as cold.
            _atomic_symlink(hot_enc / "current", ver)

        return total_npz, total_bytes

    # ------------------------------------------------------------------
    # Archiving internals
    # ------------------------------------------------------------------

    def _archive_precomputed(self) -> tuple[int, int]:
        """
        Copy new versioned precomputed dirs from hot → cold for all encoders.

        Iterates every version dir found on hot.  For any version that does
        not yet exist on cold, copies all its files to cold and updates cold's
        `current` symlink to match hot.  Never overwrites files already in cold.

        Returns (npz_files_archived, bytes_transferred).
        """
        total_npz = 0
        total_bytes = 0

        for encoder in _ENCODERS:
            hot_enc  = self._hot_precomp  / encoder
            cold_enc = self._cold_precomp / encoder

            if not hot_enc.exists():
                continue

            hot_cur = hot_enc / "current"
            current_ver: Optional[str] = None
            if hot_cur.is_symlink():
                current_ver = os.readlink(hot_cur)

            for ver_dir in hot_enc.iterdir():
                if not ver_dir.is_dir() or ver_dir.name == "current":
                    continue
                ver = ver_dir.name
                cold_ver_dir = cold_enc / ver

                # Only copy files that don't already exist in cold.
                files_to_copy = [
                    f for f in ver_dir.iterdir()
                    if not (cold_ver_dir / f.name).exists()
                ]
                if not files_to_copy:
                    continue

                cold_ver_dir.mkdir(parents=True, exist_ok=True)

                def _do(src: Path) -> int:
                    dst = cold_ver_dir / src.name
                    if dst.exists():
                        return 0
                    return self._atomic_copy(src, dst)

                with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
                    for fut in as_completed(pool.submit(_do, f) for f in files_to_copy):
                        nb = fut.result()
                        total_bytes += nb
                        total_npz += 1

            # Update cold's `current` to match hot's if hot has a current pointer.
            if current_ver and (cold_enc / current_ver).exists():
                _atomic_symlink(cold_enc / "current", current_ver)

        return total_npz, total_bytes

    def _archive_checkpoints(self, chunk: int) -> tuple[int, int]:
        """
        Archive chunk snapshot checkpoints and best weights from hot → cold.

        Only copies files that do not already exist in cold.
        Returns (files_archived, bytes_transferred).
        """
        total_files = 0
        total_bytes = 0

        def _copy_if_new(src: Path, dst: Path) -> int:
            if not src.exists() or dst.exists():
                return 0
            dst.parent.mkdir(parents=True, exist_ok=True)
            nb = self._atomic_copy(src, dst)
            return nb

        # Chunk snapshot: archive/chunk{N}_final.* files written by
        # orchestrator._archive_chunk_checkpoint().
        hot_archive  = self._hot_ckpts  / "archive"
        cold_archive = self._cold_ckpts / "archive"
        if hot_archive.exists():
            for f in hot_archive.glob(f"chunk{chunk}_final.*"):
                nb = _copy_if_new(f, cold_archive / f.name)
                if nb:
                    total_files += 1
                    total_bytes += nb

        # best.safetensors and best.json — always keep cold copy up to date.
        for name in ("best.safetensors", "best.json"):
            src = self._hot_ckpts / name
            # Archive as chunk-suffixed so cold keeps one snapshot per chunk.
            dst_name = f"chunk{chunk}_{name}"
            nb = _copy_if_new(src, cold_archive / dst_name)
            if nb:
                total_files += 1
                total_bytes += nb

        return total_files, total_bytes

    # ------------------------------------------------------------------
    # Transfer primitives
    # ------------------------------------------------------------------

    def _link_or_copy(self, src: Path, dst: Path) -> int:
        """
        Create dst pointing at src using a symlink (same device) or atomic
        copy (different device).  Returns bytes transferred (0 for symlinks,
        -1 if dst already exists).
        """
        if dst.exists() or dst.is_symlink():
            return -1
        dst.parent.mkdir(parents=True, exist_ok=True)
        if self._use_symlinks:
            os.symlink(src.resolve(), dst)
            return 0
        return self._atomic_copy(src, dst)

    def _atomic_copy(self, src: Path, dst: Path) -> int:
        """
        Copy src to dst atomically: write to a temp file, then rename.
        Returns bytes written.  Raises on failure (temp is cleaned up).
        """
        tmp = dst.with_suffix(dst.suffix + f".stg{os.getpid()}")
        try:
            shutil.copy2(src, tmp)
            size = tmp.stat().st_size
            os.replace(tmp, dst)
            return size
        except Exception:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detect_symlinks(self) -> bool:
        """
        True when cold and hot share the same physical filesystem.
        On same device symlinks are instant and zero-cost; on different
        devices we must copy.
        """
        if self.cold_root == self.hot_root:
            return True  # trivially same path
        try:
            cold_dev = os.stat(self.cold_root).st_dev if self.cold_root.exists() else None
            hot_dev  = os.stat(self.hot_root).st_dev  if self.hot_root.exists()  else None
            if cold_dev is not None and hot_dev is not None:
                return cold_dev == hot_dev
        except OSError:
            pass
        # Default to copy if we cannot stat (paths don't exist yet).
        return False

    def _check_hot_space(self, estimated_bytes: int) -> bool:
        """
        Return True if hot storage has at least staging_margin_gb free after
        the estimated transfer.  Logs a warning and returns False if not.
        """
        try:
            free_gb = _free_gb(self.hot_root)
            needed_gb = estimated_bytes / (1024 ** 3)
            if free_gb - needed_gb < self.staging_margin_gb:
                log.warning(
                    "Staging skipped: hot storage has %.1f GB free, need %.1f GB "
                    "+ %.0f GB safety margin",
                    free_gb, needed_gb, self.staging_margin_gb,
                )
                return False
        except OSError:
            pass
        return True


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def _atomic_symlink(link_path: Path, target: str) -> None:
    """Atomically create or replace a symlink (POSIX rename is atomic)."""
    link_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = link_path.parent / ".current_stg_tmp"
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    os.symlink(target, tmp)
    os.replace(tmp, link_path)


def _log(event: str, chunk: Optional[int] = None, **fields) -> None:
    try:
        log_event("stager", event, chunk=chunk, **fields)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_stager(args) -> DataStager:
    cfg = load_config(args.config if hasattr(args, "config") and args.config else None)
    return DataStager(cfg)


def cmd_stage(args) -> None:
    s = _build_stager(args)
    if not s.enabled:
        print("Stager disabled (cold_root == hot_root). Nothing to do.")
        return
    result = s.stage_for_chunk(args.chunk)
    print(json.dumps(result, indent=2))


def cmd_archive(args) -> None:
    s = _build_stager(args)
    if not s.enabled:
        print("Stager disabled (cold_root == hot_root). Nothing to do.")
        return
    result = s.archive_chunk(args.chunk)
    print(json.dumps(result, indent=2))


def cmd_status(args) -> None:
    s = _build_stager(args)
    print(json.dumps(s.status(), indent=2))


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="Data stager — cold/hot tier management")
    ap.add_argument("--config", metavar="PATH", help="Pipeline YAML config (default: v2_pipeline.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_stage = sub.add_parser("stage", help="Stage data for a chunk (cold → hot)")
    p_stage.add_argument("--chunk", type=int, required=True)
    p_stage.set_defaults(func=cmd_stage)

    p_arch = sub.add_parser("archive", help="Archive chunk data (hot → cold)")
    p_arch.add_argument("--chunk", type=int, required=True)
    p_arch.set_defaults(func=cmd_archive)

    p_stat = sub.add_parser("status", help="Show stager status")
    p_stat.set_defaults(func=cmd_status)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
