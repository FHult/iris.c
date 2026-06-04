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
import uuid
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


# Per-encoder code version. Bump an encoder's entry ONLY when a code change
# alters *that encoder's output* (e.g. a different qwen3 extraction-layer set, a
# VAE normalization fix). Unrelated commits must NOT change these.
#
# This is the PRECOMP-3 fix: the cache version is bound to encoder identity, not
# the whole-repo git SHA. Previously version_hash mixed in get_git_sha(), so any
# commit anywhere (orchestrator, doctor, docs) minted a brand-new version dir and
# forced a full re-precompute of the same shard pool — fragmenting the cache
# across SHAs and re-encoding identical shards every iteration. The encoder
# config + this code version fully determine an encoder's output, so they alone
# key the cache.
ENCODER_CODE_VERSION = {
    "qwen3":  "1",
    "vae":    "1",
    "siglip": "1",
}


def version_hash(config_subset: dict, git_sha: str = "") -> str:
    """Return stable 'v_XXXXXX' identifier from an encoder's config subset.

    The hash is bound to *encoder identity*: the model, the output-shaping config,
    and the per-encoder code version carried inside config_subset (see
    encoder_config_subset / ENCODER_CODE_VERSION). It deliberately does NOT mix in
    the repo git SHA — that made every unrelated commit produce a fresh version
    and re-precompute the same shards (PRECOMP-3). The git_sha argument is kept
    for call-site compatibility and is still recorded in the manifest for
    provenance, but it does not affect the cache key.
    """
    blob = json.dumps(config_subset, sort_keys=True)
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
            "code_version": ENCODER_CODE_VERSION["vae"],
        }
    if encoder == "qwen3":
        return {
            "qwen3_model": model_cfg.get("qwen3_model", "Qwen/Qwen3-4B"),
            "layers": [8, 17, 26],
            "think_tags": True,
            "code_version": ENCODER_CODE_VERSION["qwen3"],
        }
    if encoder == "siglip":
        return {
            "siglip_model": "google/siglip-so400m-patch14-384",
            "image_size": 384,
            "code_version": ENCODER_CODE_VERSION["siglip"],
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
          1. The `current` version — only if it actually holds records.
          2. The newest complete, non-empty version dir.
          3. Flat legacy dir (flat .npz files directly in enc_dir).
          4. None if nothing usable is available.

        An *empty* `current` (0 records) is deliberately skipped: that is the
        cache-clobber footgun — `current` pointing at an in-progress or stub
        version made every reader silently see 0 shards even though a fully
        populated version sat right next to it (the flywheel cache-clobber bug).
        A read path that returns an empty dir is never useful, so fall through to
        the newest complete version instead. (`any(glob)` short-circuits on the
        first file — cheap even on a 200k-file cold dir.)
        """
        cur = PrecomputeCache.current_dir(precomp_root, encoder)
        if cur is not None and any(cur.glob("*.npz")):
            return cur

        best: Optional[str] = None
        best_key = ""
        for v in PrecomputeCache.list_versions(precomp_root, encoder):
            if not v.get("complete") or v.get("record_count", 0) <= 0:
                continue
            key = v.get("completed_at") or v.get("created_at") or ""
            if best is None or key >= best_key:
                best, best_key = v["version"], key
        if best is not None:
            return precomp_root / encoder / best

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
            # Skip the `current` symlink (and any symlink): it resolves to a real
            # version dir, so following it here would double-count that version as
            # a phantom entry named "current".
            if d.is_symlink() or not d.is_dir():
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
    link_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = link_path.parent / f".{link_path.name}_tmp_{uuid.uuid4().hex[:8]}"
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    os.symlink(target, tmp)
    os.replace(tmp, link_path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Consolidation (PRECOMP-3 migration)
# ---------------------------------------------------------------------------

def _encoder_identity(config: Optional[dict]) -> dict:
    """Encoder-identity view of a manifest config, ignoring the code-version tag.

    Two version dirs describe the *same encoder output* if their configs match
    on everything except `code_version` (the historical git-SHA-keyed dirs had
    no code_version at all). Consolidation folds all such dirs together.
    """
    return {k: v for k, v in (config or {}).items() if k != "code_version"}


def _shard_prefix(npz_name: str) -> str:
    """'000007_0042.npz' -> '000007' (the tar-shard id), normalized to 6 digits."""
    stem = npz_name.split("_")[0]
    try:
        return f"{int(stem):06d}"
    except (ValueError, TypeError):
        return stem


def consolidate(
    precomp_root: Path,
    encoder: str,
    *,
    cfg: Optional[dict] = None,
    apply: bool = False,
) -> dict:
    """Fold every same-identity version dir for `encoder` into one canonical dir.

    The PRECOMP-3 key change (encoder identity instead of git SHA) means the
    historical per-SHA version dirs (e.g. v_d9a32b, v_c56d1c) all describe the
    same encoder. This unions their .npz (dedup by filename — the shared anchor
    shards appear in multiple dirs), writes them under the single canonical
    version, repoints `current`, and removes the now-redundant source dirs.

    Files are moved by hardlink (same filesystem → no data copy), so a dry run is
    free and the apply step never duplicates the ~hundreds of GB of npz.

    canonical config: from `cfg` (full pipeline config) if given, else inferred
    from the newest complete source manifest with the current code_version added.

    Returns a report dict; with apply=False nothing on disk changes.
    """
    enc_dir = Path(precomp_root) / encoder
    report: dict = {
        "encoder": encoder, "applied": apply, "target": None,
        "sources_merged": [], "skipped": [], "unique_npz": 0,
        "unique_shards": 0, "current_repointed": False,
    }
    if not enc_dir.is_dir():
        report["error"] = f"no encoder dir at {enc_dir}"
        return report

    versions = PrecomputeCache.list_versions(precomp_root, encoder)
    complete = [v for v in versions if v.get("complete") and v.get("record_count", 0) > 0]
    if not complete:
        report["error"] = "no complete, non-empty version dirs to consolidate"
        return report

    # canonical config + target version
    if cfg is not None:
        canon_cfg = encoder_config_subset(encoder, cfg)
    else:
        newest = max(complete, key=lambda v: v.get("completed_at") or v.get("created_at") or "")
        canon_cfg = dict(newest.get("config") or {})
        canon_cfg.setdefault("code_version", ENCODER_CODE_VERSION.get(encoder, "1"))
    canon_ident = _encoder_identity(canon_cfg)
    target_ver = version_hash(canon_cfg)
    target_dir = enc_dir / target_ver
    report["target"] = target_ver

    # select sources: same encoder identity (ignoring code_version)
    sources: list[str] = []
    for v in complete:
        if _encoder_identity(v.get("config")) == canon_ident:
            sources.append(v["version"])
        else:
            report["skipped"].append({"version": v["version"], "reason": "different encoder identity"})

    # union .npz by filename; dedup (shared anchor shards live in several dirs)
    seen: dict[str, Path] = {}
    shard_ids: set[str] = set()
    for ver in sources:
        vdir = enc_dir / ver
        for f in vdir.iterdir():
            if f.suffix != ".npz":
                continue
            shard_ids.add(_shard_prefix(f.name))
            if f.name not in seen:
                seen[f.name] = f
    report["unique_npz"] = len(seen)
    report["unique_shards"] = len(shard_ids)
    report["sources_merged"] = sources

    if not apply:
        return report

    # apply: hardlink the union into target, write manifest, repoint current,
    # then drop the redundant source dirs (target is preserved).
    target_dir.mkdir(parents=True, exist_ok=True)
    for name, src_path in seen.items():
        dst = target_dir / name
        if dst.exists():
            continue
        try:
            os.link(src_path, dst)
        except OSError:
            import shutil
            shutil.copy2(src_path, dst)

    cache = PrecomputeCache(precomp_root, encoder, canon_cfg, git_sha="")
    # ensure manifest reflects the canonical config/version even if target_dir
    # pre-existed under a stale manifest.
    cache.write_manifest_incomplete()
    cache.mark_complete(record_count=len(seen), shard_count=len(shard_ids))
    report["current_repointed"] = True

    import shutil
    for ver in sources:
        if ver == target_ver:
            continue
        shutil.rmtree(enc_dir / ver, ignore_errors=True)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_root() -> Optional[Path]:
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        from pipeline_lib import COLD_PRECOMPUTE_DIR  # type: ignore
        return Path(COLD_PRECOMPUTE_DIR)
    except Exception:
        return None


def main(argv: Optional[list] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Versioned precompute cache manager.")
    p.add_argument("--root", type=Path, default=_default_root(),
                   help="precompute root (default: cold COLD_PRECOMPUTE_DIR)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="list version dirs for an encoder")
    pl.add_argument("encoder", choices=ENCODERS)

    pc = sub.add_parser("consolidate",
                        help="fold same-identity version dirs into one canonical dir")
    pc.add_argument("encoder", choices=list(ENCODERS) + ["all"])
    pc.add_argument("--apply", action="store_true",
                    help="actually move files + repoint current (default: dry run)")

    args = p.parse_args(argv)
    if args.root is None:
        print("error: --root is required (could not resolve a default)", flush=True)
        return 2

    if args.cmd == "list":
        for v in PrecomputeCache.list_versions(args.root, args.encoder):
            star = " *current" if v.get("current") else ""
            print(f"{v['version']}  complete={v.get('complete')}  "
                  f"records={v.get('record_count')}{star}")
        return 0

    if args.cmd == "consolidate":
        encs = list(ENCODERS) if args.encoder == "all" else [args.encoder]
        rc = 0
        for enc in encs:
            r = consolidate(args.root, enc, apply=args.apply)
            mode = "APPLIED" if r["applied"] else "DRY-RUN"
            if r.get("error"):
                print(f"[{enc}] {mode}: {r['error']}")
                continue
            print(f"[{enc}] {mode}: target={r['target']}  "
                  f"merged={r['sources_merged']}  "
                  f"unique_npz={r['unique_npz']}  unique_shards={r['unique_shards']}"
                  + (f"  skipped={r['skipped']}" if r['skipped'] else ""))
        return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
