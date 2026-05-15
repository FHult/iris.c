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
import sqlite3
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_lib import (
    DATA_ROOT, load_config, log_event,
    write_heartbeat, mark_done, mark_error, has_error, clear_error,
    SHARD_BLOCK, RUN_METADATA_FILE, SHARD_SCORES_DB_PATH, ABLATION_DB_PATH,
    FLYWHEEL_DB_PATH, DEDUP_DIR,
)

log = logging.getLogger("data_stager")

_ENCODERS = ("qwen3", "vae", "siglip")


# ---------------------------------------------------------------------------
# Module-level transfer primitives (usable without instantiating DataStager)
# ---------------------------------------------------------------------------

def _same_device(p1: Path, p2: Path) -> bool:
    """True when p1 and p2 reside on the same filesystem device."""
    try:
        return os.stat(p1).st_dev == os.stat(p2).st_dev
    except OSError:
        return False


def _atomic_copy_file(src: Path, dst: Path) -> int:
    """Copy src to dst atomically (tmp + rename). Returns bytes written."""
    tmp = dst.with_suffix(dst.suffix + f".stg{os.getpid()}")
    try:
        shutil.copy2(src, tmp)
        size = tmp.stat().st_size
        os.replace(tmp, dst)
        return size
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

# Write a heartbeat to the sentinel dir every this many file completions so the
# orchestrator can detect a hung stager even during long multi-GB transfers.
_HEARTBEAT_EVERY = 100


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

        self.cold_root     = Path(storage.get("cold_root",     DATA_ROOT)).resolve()
        self.hot_root      = Path(storage.get("hot_root",      DATA_ROOT)).resolve()
        self.ultrahot_root = Path(storage.get("ultrahot_root", Path.home() / "ultrahot")).resolve()

        self.staging_margin_gb   = float(storage.get("staging_margin_gb",   50.0))
        self.cleanup_safety_gb   = float(storage.get("cleanup_safety_gb",   20.0))
        self.archive_after_chunk = bool(storage.get("archive_after_chunk",  True))
        self.max_parallel        = int(storage.get("max_parallel_transfers", 3))

        # Active prep tier: "hot" (default) or "ultrahot".
        # When "ultrahot", staging writes to and archiving reads from ultrahot_root.
        prep_tier = storage.get("data_prep_tier", "hot")
        self._prep_root = self.ultrahot_root if prep_tier == "ultrahot" else self.hot_root

        # Derived paths on cold storage.
        self._cold_shards   = self.cold_root / "shards"
        self._cold_precomp  = self.cold_root / "precomputed"
        self._cold_ckpts    = self.cold_root / "checkpoints" / "stage1"
        self._cold_weights  = self.cold_root / "weights"
        self._cold_metadata = self.cold_root / "metadata"

        # Derived paths on the active prep tier (hot or ultrahot).
        self._hot_shards  = self._prep_root / "shards"
        self._hot_precomp = self._prep_root / "precomputed"
        self._hot_ckpts   = self._prep_root / "checkpoints" / "stage1"

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

        Writes stage.done on success, stage.error on any failure.  The
        orchestrator polls these sentinels to gate training and retry on error.
        Clears any prior stage.error at the start of each attempt so a retry
        (after operator intervention) begins with a clean slate.
        """
        if not self.enabled:
            return {"shards_staged": 0, "npz_staged": 0, "bytes_transferred": 0}

        clear_error(chunk, "stage")
        write_heartbeat("stager", chunk, phase="stage", status="running")
        log.info("Staging chunk %d: cold=%s → prep=%s", chunk, self.cold_root, self._prep_root)
        _log("stage_start", chunk)

        try:
            total_bytes = 0
            shards_staged, nb_shards = self._stage_shards(chunk)
            npz_staged, nbytes = self._stage_precomputed(chunk)
            total_bytes += nb_shards + nbytes

            summary = {
                "shards_staged": shards_staged,
                "npz_staged": npz_staged,
                "bytes_transferred": total_bytes,
            }
            mark_done(chunk, "stage")
            write_heartbeat("stager", chunk, phase="stage", status="done", **summary)
            _log("stage_done", chunk, **summary)
            log.info("Staging chunk %d complete: %s", chunk, summary)
            return summary
        except Exception as exc:
            msg = str(exc)
            mark_error(chunk, "stage", msg)
            write_heartbeat("stager", chunk, phase="stage", status="error", error=msg)
            _log("stage_error", chunk, error=msg)
            raise

    def archive_chunk(self, chunk: int) -> dict:
        """
        Archive hot-storage data for chunk N back to cold storage.

        Copies new versioned precomputed dirs and checkpoint snapshots that
        exist on hot but not yet on cold.  Never deletes from cold.

        Returns a summary dict: {npz_archived, ckpt_archived, bytes_transferred}.

        Writes archive.done on success, archive.error on failure.  Archive
        errors dispatch an issue but do not block the pipeline — the orchestrator
        retries automatically until cold storage is updated.  Clears any prior
        archive.error at the start of each attempt.
        """
        if not self.enabled or not self.archive_after_chunk:
            return {"npz_archived": 0, "ckpt_archived": 0, "bytes_transferred": 0}

        clear_error(chunk, "archive")
        write_heartbeat("stager", chunk, phase="archive", status="running")
        log.info("Archiving chunk %d: prep=%s → cold=%s", chunk, self._prep_root, self.cold_root)
        _log("archive_start", chunk)

        try:
            npz_archived, nb_precomp = self._archive_precomputed(chunk)
            ckpt_archived, nb_ckpts  = self._archive_checkpoints(chunk)
            self._archive_dbs()
            total_bytes = nb_precomp + nb_ckpts

            summary = {
                "npz_archived": npz_archived,
                "ckpt_archived": ckpt_archived,
                "bytes_transferred": total_bytes,
            }
            mark_done(chunk, "archive")
            write_heartbeat("stager", chunk, phase="archive", status="done", **summary)
            _log("archive_done", chunk, **summary)
            log.info("Archiving chunk %d complete: %s", chunk, summary)
            return summary
        except Exception as exc:
            msg = str(exc)
            mark_error(chunk, "archive", msg)
            write_heartbeat("stager", chunk, phase="archive", status="error", error=msg)
            _log("archive_error", chunk, error=msg)
            raise

    def status(self) -> dict:
        """Return a status snapshot for pipeline_status.py."""
        prep_free = _free_gb(self._prep_root) if self._prep_root.exists() else None
        hot_free  = _free_gb(self.hot_root)   if self.hot_root.exists()   else None
        cold_free = _free_gb(self.cold_root)  if self.cold_root.exists()  else None
        return {
            "enabled":       self.enabled,
            "use_symlinks":  self._use_symlinks,
            "cold_root":     str(self.cold_root),
            "hot_root":      str(self.hot_root),
            "prep_root":     str(self._prep_root),
            "prep_free_gb":  round(prep_free, 1) if prep_free is not None else None,
            "hot_free_gb":   round(hot_free,  1) if hot_free  is not None else None,
            "cold_free_gb":  round(cold_free, 1) if cold_free is not None else None,
        }

    # ------------------------------------------------------------------
    # Staging internals
    # ------------------------------------------------------------------

    def _stage_shards(self, chunk: int) -> int:
        """
        Symlink/copy shards for chunk N from cold_shards → hot_shards.

        Chunk N owns shard IDs [(N-1)*SHARD_BLOCK, N*SHARD_BLOCK).
        Shards are named {shard_id:06d}.tar — match by numeric stem.
        """
        if not self._cold_shards.exists():
            return 0

        lo = (chunk - 1) * SHARD_BLOCK
        hi =  chunk      * SHARD_BLOCK
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

        if not self._use_symlinks and tasks:
            estimated = sum(_safe_size(src) for src, _ in tasks)
            if not self._check_hot_space(estimated):
                raise RuntimeError(
                    f"Insufficient hot-storage space to stage chunk {chunk} shards "
                    f"(~{estimated / 1e9:.1f} GB needed, margin={self.staging_margin_gb:.0f} GB)"
                )

        total_bytes = 0

        def _do(src_dst: tuple[Path, Path]) -> tuple[int, int]:
            nbytes = self._link_or_copy(*src_dst)
            return 1, max(0, nbytes)

        with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            for i, fut in enumerate(as_completed(pool.submit(_do, t) for t in tasks)):
                n, nb = fut.result()
                staged += n
                total_bytes += nb
                if i % _HEARTBEAT_EVERY == 0:
                    write_heartbeat("stager", chunk, phase="stage", status="running",
                                    shards_staged=staged, shards_total=len(tasks))

        return staged, total_bytes

    def _stage_precomputed(self, chunk: int) -> tuple[int, int]:
        """
        Stage versioned precomputed .npz caches from cold → hot for all encoders.

        Only stages .npz files whose shard stem falls within the chunk's shard
        range [(chunk-1)*SHARD_BLOCK, chunk*SHARD_BLOCK).  Non-.npz files
        (manifest.json, etc.) are staged unconditionally.  Updates hot's
        `current` symlink to match cold's.

        Returns (npz_files_staged, bytes_transferred).
        """
        total_npz = 0
        total_bytes = 0

        lo = (chunk - 1) * SHARD_BLOCK
        hi =  chunk      * SHARD_BLOCK

        def _in_chunk(f: Path) -> bool:
            """Return True if f should be staged for this chunk."""
            if f.suffix != ".npz":
                return True  # manifest.json and other metadata always staged
            try:
                shard_id = int(f.name.split("_")[0])
            except (ValueError, IndexError):
                return True  # unparseable name — stage it unconditionally
            return lo <= shard_id < hi

        # Pre-scan total transfer size before starting any copies so we can
        # enforce the hot-storage margin as a single upfront check.
        if not self._use_symlinks:
            estimated = 0
            for encoder in _ENCODERS:
                cold_cur = self._cold_precomp / encoder / "current"
                if not cold_cur.exists() and not cold_cur.is_symlink():
                    continue
                ver = os.readlink(cold_cur) if cold_cur.is_symlink() else cold_cur.name
                cold_ver_dir = self._cold_precomp / encoder / ver
                hot_ver_dir  = self._hot_precomp  / encoder / ver
                if cold_ver_dir.exists():
                    for f in cold_ver_dir.iterdir():
                        if _in_chunk(f) and not (hot_ver_dir / f.name).exists():
                            estimated += _safe_size(f)
            if estimated and not self._check_hot_space(estimated):
                raise RuntimeError(
                    f"Insufficient hot-storage space to stage chunk {chunk} precomputed caches "
                    f"(~{estimated / 1e9:.1f} GB needed, margin={self.staging_margin_gb:.0f} GB)"
                )

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
                if not _in_chunk(f):
                    continue
                dst = hot_ver_dir / f.name
                if not dst.exists() and not dst.is_symlink():
                    tasks.append((f, dst))

            def _do(src_dst: tuple[Path, Path]) -> int:
                return self._link_or_copy(*src_dst)

            with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
                for i, fut in enumerate(as_completed(pool.submit(_do, t) for t in tasks)):
                    nb = fut.result()
                    if nb >= 0:  # -1 → already existed / symlinked
                        total_bytes += nb
                    total_npz += 1
                    if i % _HEARTBEAT_EVERY == 0:
                        write_heartbeat("stager", chunk, phase="stage", status="running",
                                        npz_staged=total_npz, encoder=encoder)

            # Keep hot's `current` symlink pointing at the same version as cold.
            _atomic_symlink(hot_enc / "current", ver)

        return total_npz, total_bytes

    # ------------------------------------------------------------------
    # Archiving internals
    # ------------------------------------------------------------------

    def _archive_precomputed(self, chunk: int) -> tuple[int, int]:
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
                    for i, fut in enumerate(
                            as_completed(pool.submit(_do, f) for f in files_to_copy)):
                        nb = fut.result()
                        total_bytes += nb
                        total_npz += 1
                        if i % _HEARTBEAT_EVERY == 0:
                            write_heartbeat("stager", chunk, phase="archive",
                                            status="running", npz_archived=total_npz,
                                            encoder=encoder)

            # Update cold's `current` to match hot's if hot has a current pointer.
            if current_ver and (cold_enc / current_ver).exists():
                _atomic_symlink(cold_enc / "current", current_ver)

        return total_npz, total_bytes

    def _campaign_date(self) -> str:
        """Return campaign date string (YYYYMMDD) from run_metadata.json, or today."""
        try:
            meta = json.loads(RUN_METADATA_FILE.read_text())
            started_at = meta.get("started_at", "")
            if started_at:
                return started_at[:10].replace("-", "")
        except Exception:
            pass
        from datetime import date
        return date.today().strftime("%Y%m%d")

    def _archive_checkpoints(self, chunk: int) -> tuple[int, int]:
        """
        Archive chunk snapshot checkpoints from hot → cold campaign-structured layout.

        Cold layout:
          cold_root/weights/flywheel-YYYYMMDD/
            step_{N}.safetensors  ← from hot archive/chunk{chunk}_final.*
            step_{N}.json
            final.safetensors     ← copy of best.safetensors at chunk end
            final.json

        Also updates cold_root/weights/best/ per-metric symlinks.
        Returns (files_archived, bytes_transferred).
        """
        total_files = 0
        total_bytes = 0

        campaign = f"flywheel-{self._campaign_date()}"
        campaign_dir = self._cold_weights / campaign
        campaign_dir.mkdir(parents=True, exist_ok=True)

        def _copy_if_new(src: Path, dst: Path) -> int:
            if not src.exists() or dst.exists():
                return 0
            dst.parent.mkdir(parents=True, exist_ok=True)
            return self._atomic_copy(src, dst)

        # Chunk snapshot: archive/chunk{chunk}_final.{safetensors,json} from orchestrator.
        # Explicitly enumerate suffixes (not glob) to avoid matching .ema.safetensors,
        # which has the same .suffix and would map to the same dst_name.
        hot_archive = self._hot_ckpts / "archive"
        if hot_archive.exists():
            for suffix in (".safetensors", ".json"):
                src = hot_archive / f"chunk{chunk}_final{suffix}"
                nb = _copy_if_new(src, campaign_dir / src.name)
                if nb:
                    total_files += 1
                    total_bytes += nb

        # best.* snapshot — copy as final.* in this campaign
        for suffix in (".safetensors", ".json"):
            src = self._hot_ckpts / f"best{suffix}"
            nb = _copy_if_new(src, campaign_dir / f"final{suffix}")
            if nb:
                total_files += 1
                total_bytes += nb

        # Update per-metric best symlinks from best.json
        best_json_path = self._hot_ckpts / "best.json"
        if best_json_path.exists():
            try:
                best_meta = json.loads(best_json_path.read_text())
                final_ckpt = campaign_dir / "final.safetensors"
                if final_ckpt.exists():
                    self.update_best_symlinks(best_meta, final_ckpt)
            except Exception as exc:
                log.warning("update_best_symlinks failed: %s", exc)

        return total_files, total_bytes

    def update_best_symlinks(self, metrics: dict, ckpt_path: Path) -> None:
        """Update cold weights/best/{metric}.safetensors symlink if ckpt_path wins.

        metrics: dict with float values (e.g. {"cond_gap": 0.42, "loss": 0.18}).
        Higher is better for gap metrics, lower for loss.
        """
        best_dir = self._cold_weights / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        # Metrics where higher = better (all others treated as lower = better)
        higher_is_better = {"cond_gap", "ref_gap", "clip_i", "clip_t"}
        for metric, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            meta_path = best_dir / f"{metric}.json"
            try:
                current_best = json.loads(meta_path.read_text())["value"] if meta_path.exists() else None
            except Exception:
                current_best = None
            is_better = (
                current_best is None
                or (metric in higher_is_better and value > current_best)
                or (metric not in higher_is_better and value < current_best)
            )
            if is_better:
                # Relative symlink so it survives volume remounts.
                rel_target = os.path.relpath(ckpt_path.resolve(), best_dir)
                _atomic_symlink(best_dir / f"{metric}.safetensors", rel_target)
                meta_path.write_text(json.dumps(
                    {"value": value, "path": str(ckpt_path)}, indent=2))
                log.info("New best %s = %s → %s", metric, value, ckpt_path.name)

    def _archive_dbs(self) -> None:
        """Copy hot DBs and dedup state to cold metadata dir.

        SQLite files use the backup API for a WAL-consistent snapshot.
        Dedup files (FAISS index + blocklist) are plain files copied atomically.
        """
        self._cold_metadata.mkdir(parents=True, exist_ok=True)
        for src in (SHARD_SCORES_DB_PATH, ABLATION_DB_PATH, FLYWHEEL_DB_PATH):
            db_name = src.name
            if not src.exists():
                continue
            dst = self._cold_metadata / db_name
            tmp = dst.with_suffix(dst.suffix + f".stg{os.getpid()}")
            try:
                with sqlite3.connect(str(src)) as src_conn:
                    with sqlite3.connect(str(tmp)) as dst_conn:
                        src_conn.backup(dst_conn)
                os.replace(tmp, dst)
                log.info("Archived %s → %s", db_name, dst)
            except Exception as exc:
                tmp.unlink(missing_ok=True)
                log.warning("Failed to archive %s: %s", db_name, exc)

        # Archive dedup state: FAISS index + sidecar + blocklist
        for fname in ("dedup_index.faiss", "dedup_index.ids", "duplicate_ids.txt"):
            src = DEDUP_DIR / fname
            if not src.exists():
                continue
            try:
                _atomic_copy_file(src, self._cold_metadata / fname)
                log.info("Archived dedup %s → cold/metadata/", fname)
            except Exception as exc:
                log.warning("Failed to archive dedup %s: %s", fname, exc)

    def promote_to_ultrahot(self, chunk: int) -> dict:
        """Copy active checkpoint + precompute version to the Ultrahot tier.

        Writes atomically: all files go to ultrahot_root/.staging/, then the
        staging dir is renamed into place so the web app never sees partial state.
        Updates ultrahot_root/manifest.json with checkpoint path and version.

        Returns summary dict; raises on failure.
        """
        from datetime import datetime, timezone

        # Find the active checkpoint: prefer cold best/, fall back to hot best.
        best_cold = self._cold_weights / "best"
        hot_best  = self._hot_ckpts / "best.safetensors"
        ckpt_src: Optional[Path] = None
        if best_cold.exists():
            # Pick the cond_gap symlink if available, else any .safetensors.
            for name in ("cond_gap.safetensors", "loss.safetensors"):
                candidate = best_cold / name
                if candidate.exists():
                    ckpt_src = candidate.resolve()
                    break
            if ckpt_src is None:
                for f in sorted(best_cold.glob("*.safetensors")):
                    if f.exists():
                        ckpt_src = f.resolve()
                        break
        if ckpt_src is None and hot_best.exists():
            ckpt_src = hot_best.resolve()
        if ckpt_src is None:
            raise FileNotFoundError("No checkpoint found in cold best/ or hot best.safetensors")

        # Determine active precompute version per encoder.
        precomp_versions: dict[str, str] = {}
        for enc in _ENCODERS:
            cur = self._hot_precomp / enc / "current"
            if cur.is_symlink():
                precomp_versions[enc] = cur.resolve().name
            else:
                cold_cur = self._cold_precomp / enc / "current"
                if cold_cur.is_symlink():
                    precomp_versions[enc] = cold_cur.resolve().name

        uh = self.ultrahot_root
        staging = uh / ".staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)

        bytes_copied = 0

        # Copy checkpoint and its JSON sidecar.
        dst_ckpt = staging / "weights" / "current.safetensors"
        dst_ckpt.parent.mkdir(parents=True, exist_ok=True)
        bytes_copied += _atomic_copy_file(ckpt_src, dst_ckpt)
        ckpt_json = ckpt_src.with_suffix(".json")
        if ckpt_json.exists():
            bytes_copied += _atomic_copy_file(ckpt_json, dst_ckpt.with_suffix(".json"))

        # Copy precompute current dirs per encoder (or symlink if same device).
        precomp_copied = 0
        uh_same_dev = uh.exists() and self._prep_root.exists() and _same_device(uh, self._prep_root)
        for enc, ver in precomp_versions.items():
            src_ver = self._hot_precomp / enc / ver
            if not src_ver.exists():
                src_ver = self._cold_precomp / enc / ver
            if not src_ver.exists():
                continue
            dst_enc = staging / "precomputed" / enc
            dst_ver = dst_enc / ver
            dst_ver.mkdir(parents=True, exist_ok=True)
            if uh_same_dev:
                _atomic_symlink(dst_enc / "current", ver)
                for f in src_ver.glob("*"):
                    lnk = dst_ver / f.name
                    if not lnk.exists():
                        os.symlink(os.path.relpath(f.resolve(), dst_ver), lnk)
            else:
                with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
                    futs = {
                        pool.submit(_atomic_copy_file, f, dst_ver / f.name): f
                        for f in src_ver.glob("*") if f.is_file()
                    }
                    for fut in as_completed(futs):
                        try:
                            bytes_copied += fut.result()
                            precomp_copied += 1
                        except Exception as exc:
                            log.warning("Precompute copy failed for %s: %s", futs[fut], exc)
                _atomic_symlink(dst_enc / "current", ver)

        # Write manifest.
        manifest = {
            "checkpoint":        str(ckpt_src),
            "chunk":             chunk,
            "precompute_version": precomp_versions,
            "promoted_at":       datetime.now(timezone.utc).isoformat(),
            "bytes_copied":      bytes_copied,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # Atomic rename: .staging → live dirs.
        # Move each subdir of staging into place; rename manifest last.
        for subdir in ("weights", "precomputed"):
            src_sub = staging / subdir
            if not src_sub.exists():
                continue
            dst_sub = uh / subdir
            if dst_sub.exists():
                old = uh / f".old_{subdir}_{os.getpid()}"
                os.rename(dst_sub, old)
                shutil.rmtree(old, ignore_errors=True)
            os.rename(src_sub, dst_sub)
        manifest_dst = uh / "manifest.json"
        os.replace(staging / "manifest.json", manifest_dst)
        # Clean up empty staging dir.
        try:
            staging.rmdir()
        except OSError:
            pass

        summary = {
            "checkpoint": str(ckpt_src),
            "precompute_versions": precomp_versions,
            "bytes_copied": bytes_copied,
            "precomp_files_copied": precomp_copied,
        }
        log.info("Promoted chunk %d to Ultrahot tier: %s", chunk, ckpt_src.name)
        _log("promote_done", chunk, **summary)
        return summary

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
            os.symlink(os.path.relpath(src.resolve(), dst.parent), dst)
            return 0
        return self._atomic_copy(src, dst)

    def _atomic_copy(self, src: Path, dst: Path) -> int:
        """
        Copy src to dst atomically: write to a temp file, then rename.
        Returns bytes written.  Raises on failure (temp is cleaned up).
        """
        return _atomic_copy_file(src, dst)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detect_symlinks(self) -> bool:
        """
        True when cold and the active prep root share the same physical filesystem.
        On same device symlinks are instant and zero-cost; on different
        devices we must copy.
        """
        if self.cold_root == self._prep_root:
            return True  # trivially same path
        if self.cold_root.exists() and self._prep_root.exists():
            return _same_device(self.cold_root, self._prep_root)
        # Default to copy if we cannot stat (paths don't exist yet).
        return False

    def _check_hot_space(self, estimated_bytes: int) -> bool:
        """
        Return True if hot storage has at least staging_margin_gb free after
        the estimated transfer.  Logs a warning and returns False if not.

        Called before any file copies begin so that an insufficiently-sized hot
        volume is caught upfront rather than mid-transfer.  Callers raise
        RuntimeError on False so stage_for_chunk() writes stage.error.
        """
        try:
            free_gb   = _free_gb(self._prep_root)
            needed_gb = estimated_bytes / (1024 ** 3)
            if free_gb - needed_gb < self.staging_margin_gb:
                log.warning(
                    "Prep-tier storage space check failed: %.1f GB free, need %.1f GB "
                    "+ %.0f GB safety margin (%.1f GB available after transfer)",
                    free_gb, needed_gb, self.staging_margin_gb,
                    free_gb - needed_gb,
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


def _safe_size(path: Path) -> int:
    """Return file size in bytes, 0 on error (stat may fail for broken symlinks)."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _atomic_symlink(link_path: Path, target: str) -> None:
    """Atomically create or replace a symlink (POSIX rename is atomic)."""
    link_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = link_path.parent / f".{link_path.name}_tmp_{uuid.uuid4().hex[:8]}"
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
    try:
        result = s.stage_for_chunk(args.chunk)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        log.error("Stage chunk %d failed: %s", args.chunk, exc)
        sys.exit(1)


def cmd_archive(args) -> None:
    s = _build_stager(args)
    if not s.enabled:
        print("Stager disabled (cold_root == hot_root). Nothing to do.")
        return
    try:
        result = s.archive_chunk(args.chunk)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        log.error("Archive chunk %d failed: %s", args.chunk, exc)
        sys.exit(1)


def cmd_status(args) -> None:
    s = _build_stager(args)
    indent = None if getattr(args, "ai", False) else 2
    print(json.dumps(s.status(), indent=indent))


def cmd_promote(args) -> None:
    s = _build_stager(args)
    try:
        result = s.promote_to_ultrahot(args.chunk)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        log.error("Promote chunk %d to Ultrahot failed: %s", args.chunk, exc)
        sys.exit(1)


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

    p_prom = sub.add_parser("promote", help="Promote best checkpoint to Ultrahot tier")
    p_prom.add_argument("--chunk", type=int, required=True)
    p_prom.set_defaults(func=cmd_promote)

    p_stat = sub.add_parser("status", help="Show stager status")
    p_stat.add_argument("--ai", action="store_true",
                        help="Compact JSON for AI consumption")
    p_stat.set_defaults(func=cmd_status)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
