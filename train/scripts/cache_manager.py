"""
train/scripts/cache_manager.py — Versioned, content-addressable precompute cache.

Each encoder (qwen3, vae, siglip) gets a version directory whose name is a
short hash of the config fields that affect its output.  A `current` symlink
in each encoder dir points to the active version used by training.

Directory layout:
    PRECOMP_DIR/
      qwen3/
        v_a3f9c2/
          manifest.json
          000000_0000.npz ...
        current -> v_a3f9c2/    (POSIX symlink, atomically updated)
      vae/
        v_b17d44/ ...
      siglip/
        v_c9e012/ ...

Invalidation rules:
  - cache_dir exists + manifest.complete == True  → skip (fully cached)
  - cache_dir exists + manifest.complete == False → resume (partial run)
  - cache_dir does not exist                      → create and start fresh
  - Old version dirs survive until --clear-stale or explicit --clear-version.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ENCODERS = ("qwen3", "vae", "siglip")


# ---------------------------------------------------------------------------
# Version hash
# ---------------------------------------------------------------------------

def get_git_sha(repo_root: Optional[Path] = None) -> str:
    """Return 8-char short git SHA for HEAD, or '00000000' on failure."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            capture_output=True, text=True,
            cwd=str(repo_root or Path(__file__).parent),
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "00000000"


def version_hash(config_subset: dict, git_sha: str) -> str:
    """Return stable 'v_XXXXXX' identifier from config dict + git SHA."""
    blob = json.dumps(config_subset, sort_keys=True) + git_sha[:8]
    return "v_" + hashlib.sha256(blob.encode()).hexdigest()[:6]


def encoder_config_subset(encoder: str, cfg: dict) -> dict:
    """
    Extract the config fields that affect a given encoder's output.
    Used by the orchestrator to derive the version hash.

    cfg: top-level pipeline YAML dict (e.g. from load_config()).
    """
    model_cfg    = cfg.get("model", {})
    data_cfg     = cfg.get("data", {})
    if encoder == "vae":
        flux_model = model_cfg.get("flux_model", "flux-klein-4b")
        return {
            "flux_model": Path(flux_model).name,
            "image_size": data_cfg.get("image_size", 512),
        }
    if encoder == "qwen3":
        return {
            "qwen3_model": model_cfg.get("qwen3_model", "Qwen/Qwen3-4B"),
            "layers": [9, 18, 27],
        }
    if encoder == "siglip":
        return {
            "siglip_model": "google/siglip-so400m-patch14-384",
            "image_size": 384,
        }
    return {}


# ---------------------------------------------------------------------------
# PrecomputeCache
# ---------------------------------------------------------------------------

class PrecomputeCache:
    """
    Manages one versioned precompute cache directory for a single encoder.

    Typical workflow (precompute side):
        cache = PrecomputeCache(precomp_dir, "qwen3", config_subset, git_sha)
        cache.write_manifest_incomplete()
        # ... write .npz files into cache.cache_dir() ...
        cache.mark_complete(record_count=412800, shard_count=80)

    Typical workflow (promotion side):
        cache = PrecomputeCache(precomp_dir, "qwen3", config_subset, git_sha)
        cache.mark_complete(...)   # updates current symlink
    """

    def __init__(
        self,
        precomp_root: Path,
        encoder: str,
        config_subset: dict,
        git_sha: str,
    ) -> None:
        self._enc_dir = precomp_root / encoder
        self._encoder = encoder
        self._config  = config_subset
        self._git_sha = git_sha
        self._ver     = version_hash(config_subset, git_sha)

    def version(self) -> str:
        return self._ver

    def cache_dir(self) -> Path:
        return self._enc_dir / self._ver

    def is_complete(self) -> bool:
        m = self._read_manifest()
        return bool(m and m.get("complete"))

    def write_manifest_incomplete(self) -> None:
        """Create cache_dir and write manifest.json with complete=False."""
        self.cache_dir().mkdir(parents=True, exist_ok=True)
        self._write_manifest({"complete": False})

    def mark_complete(self, record_count: int, shard_count: int) -> None:
        """Update manifest to complete=True and atomically update `current` symlink."""
        self._write_manifest({
            "complete":     True,
            "record_count": record_count,
            "shard_count":  shard_count,
            "completed_at": _now_iso(),
        })
        _atomic_symlink(self._enc_dir / "current", self._ver)

    def all_records(self) -> set[str]:
        """Set of .npz stems present in cache_dir (for cache-hit checks)."""
        d = self.cache_dir()
        if not d.is_dir():
            return set()
        return {f.stem for f in d.iterdir() if f.suffix == ".npz"}

    def record_count(self) -> int:
        d = self.cache_dir()
        if not d.is_dir():
            return 0
        return sum(1 for f in d.iterdir() if f.suffix == ".npz")

    # ── Static helpers ──────────────────────────────────────────────────────

    @staticmethod
    def current_dir(precomp_root: Path, encoder: str) -> Optional[Path]:
        """
        Resolve the `current` symlink for encoder; return None if absent/broken.
        Following the symlink is transparent to all file I/O, but callers that
        want the canonical versioned path can call this first.
        """
        link = precomp_root / encoder / "current"
        try:
            if link.is_symlink():
                target = link.resolve()
                if target.is_dir():
                    return target
        except OSError:
            pass
        return None

    @staticmethod
    def effective_dir(precomp_root: Path, encoder: str) -> Optional[Path]:
        """
        Best available cache dir for encoder:
          1. Versioned current dir (current symlink resolved)
          2. Flat legacy dir (flat .npz files directly in enc_dir)
          3. None if nothing is available
        """
        cur = PrecomputeCache.current_dir(precomp_root, encoder)
        if cur:
            return cur
        flat = precomp_root / encoder
        if flat.is_dir() and any(flat.glob("*.npz")):
            return flat
        return None

    @staticmethod
    def list_versions(precomp_root: Path, encoder: str) -> list[dict]:
        """Return a list of dicts describing each version dir for encoder."""
        enc_dir = precomp_root / encoder
        if not enc_dir.is_dir():
            return []
        link = enc_dir / "current"
        current_name = os.readlink(str(link)) if link.is_symlink() else None
        versions: list[dict] = []
        for d in sorted(enc_dir.iterdir()):
            if not d.is_dir():
                continue
            info: dict = {
                "version":      d.name,
                "current":      d.name == current_name,
                "complete":     False,
                "record_count": 0,
            }
            try:
                m = json.loads((d / "manifest.json").read_text())
                info["complete"]     = bool(m.get("complete"))
                info["record_count"] = m.get("record_count", 0)
                info["created_at"]   = m.get("created_at", "?")
                info["completed_at"] = m.get("completed_at")
                info["config"]       = m.get("config", {})
            except Exception:
                info["record_count"] = sum(
                    1 for f in d.iterdir() if f.suffix == ".npz"
                )
            versions.append(info)
        return versions

    @staticmethod
    def clear(
        precomp_root: Path,
        encoder: str,
        version: Optional[str] = None,
        stale_only: bool = False,
    ) -> list[str]:
        """
        Delete version dir(s).  Returns list of deleted version names.

        stale_only=True: skip the current version.
        version=X:       delete only that specific version.
        Neither:         delete all version dirs.
        """
        import shutil
        enc_dir = precomp_root / encoder
        if not enc_dir.is_dir():
            return []
        link = enc_dir / "current"
        current_name = os.readlink(str(link)) if link.is_symlink() else None
        deleted: list[str] = []
        for d in list(enc_dir.iterdir()):
            if not d.is_dir():
                continue
            if version and d.name != version:
                continue
            if stale_only and d.name == current_name:
                continue
            shutil.rmtree(d)
            deleted.append(d.name)
        return deleted

    @staticmethod
    def migrate_legacy(precomp_root: Path, encoder: str) -> Optional[Path]:
        """
        Move flat .npz files in enc_dir into v_legacy/ and create a
        `current -> v_legacy` symlink.  No-op if no flat files present.
        """
        enc_dir = precomp_root / encoder
        if not enc_dir.is_dir():
            return None
        flat_files = [f for f in enc_dir.iterdir() if f.suffix == ".npz"]
        if not flat_files:
            return None
        legacy_dir = enc_dir / "v_legacy"
        legacy_dir.mkdir(exist_ok=True)
        for f in flat_files:
            f.rename(legacy_dir / f.name)
        done_file = enc_dir / ".precompute_done.json"
        if done_file.exists():
            done_file.rename(legacy_dir / ".precompute_done.json")
        _atomic_symlink(enc_dir / "current", "v_legacy")
        return legacy_dir

    # ── Private ─────────────────────────────────────────────────────────────

    def _write_manifest(self, extra: dict) -> None:
        manifest = {
            "version":    self._ver,
            "created_at": _now_iso(),
            "git_sha":    self._git_sha,
            "encoder":    self._encoder,
            "config":     self._config,
        }
        manifest.update(extra)
        tmp = self.cache_dir() / ".manifest_tmp.json"
        tmp.write_text(json.dumps(manifest, indent=2))
        tmp.rename(self.cache_dir() / "manifest.json")

    def _read_manifest(self) -> Optional[dict]:
        try:
            return json.loads((self.cache_dir() / "manifest.json").read_text())
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _atomic_symlink(link_path: Path, target: str) -> None:
    """Atomically create or replace a symlink (POSIX rename is atomic)."""
    tmp = link_path.parent / ".current_tmp"
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    os.symlink(target, tmp)
    os.replace(tmp, link_path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
