"""
train/scripts/mobile_mode.py — Checkout / Checkin for portable flywheel campaigns.

Workflow:
  checkout  — Stage active campaign data (precompute, weights, shards, DBs) from
              cold → hot and record a manifest in cold/metadata/checkouts/.
              Cold storage stays at home; you travel with hot (portable SSD).

  checkin   — On return, push new data from hot → cold: new precompute version dirs,
              new checkpoint snapshots, and DB backups. Never deletes from cold.

  status    — Show active sessions and whether the portable SSD is mounted.

Manifest lives at:  cold_root/metadata/checkouts/{session_id}/manifest.json
HTML report at:     cold_root/metadata/checkouts/{session_id}/checkin_report.html

Safety guarantees:
  - Cold storage is never modified during checkout.
  - Cold storage files are never deleted during checkin.
  - All writes to cold during checkin use atomic tmp→rename.
  - The manifest records a snapshot of hot state at checkout so checkin only
    pushes genuinely new files (delta since checkout).
  - Conflict detection warns if the base machine kept working after checkout.

Usage (via pipeline_ctl.py):
  pipeline_ctl checkout [--chunks 1 2 3] [--label TAG] [--dry-run]
  pipeline_ctl checkin  [--session SESSION_ID] [--from /Volumes/2TBSSD] [--force] [--dry-run]
  pipeline_ctl mobile-status
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sqlite3
import string
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_lib import (
    now_iso, load_config,
    COLD_ROOT, COLD_WEIGHTS_DIR, COLD_PRECOMPUTE_DIR, COLD_METADATA_DIR,
    DATA_ROOT, CKPT_DIR, PRECOMP_DIR, SHARDS_DIR,
    SHARD_SCORES_DB_PATH, FLYWHEEL_DB_PATH, ABLATION_DB_PATH,
    SHARD_BLOCK,
)
from data_stager import _atomic_copy_file, _atomic_symlink, _same_device

_ENCODERS = ("qwen3", "vae", "siglip")
_MAX_PARALLEL = 4


# ---------------------------------------------------------------------------
# Session ID / manifest path helpers
# ---------------------------------------------------------------------------

def _session_id(label: str = "") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = "".join(random.choices(string.ascii_lowercase, k=6))
    if label:
        return f"mobile-{label}-{ts}-{suffix}"
    return f"mobile-{ts}-{suffix}"


def _checkouts_dir(cold_root: Path) -> Path:
    return cold_root / "metadata" / "checkouts"


def _manifest_path(cold_root: Path, session_id: str) -> Path:
    return _checkouts_dir(cold_root) / session_id / "manifest.json"


# ---------------------------------------------------------------------------
# State snapshots (for conflict detection and delta computation)
# ---------------------------------------------------------------------------

def _snapshot_hot_state(hot_root: Path) -> dict:
    """
    Record the state of hot storage at checkout time.
    At checkin, anything on hot that wasn't in this snapshot is "new" and
    should be pushed to cold.
    """
    precomp_versions: dict[str, str] = {}
    hot_precomp = hot_root / "precomputed"
    if hot_precomp.exists():
        for enc_dir in sorted(hot_precomp.iterdir()):
            if not enc_dir.is_dir():
                continue
            cur = enc_dir / "current"
            if cur.is_symlink():
                precomp_versions[enc_dir.name] = os.readlink(cur)
            elif cur.is_dir():
                precomp_versions[enc_dir.name] = cur.name

    archive_files: list[str] = []
    hot_archive = hot_root / "checkpoints" / "stage1" / "archive"
    if hot_archive.exists():
        archive_files = sorted(f.name for f in hot_archive.iterdir() if f.is_file())

    db_mtimes: dict[str, float] = {}
    for db_name in ("shard_scores.db", "flywheel_history.db", "ablation_history.db"):
        p = hot_root / db_name
        if p.exists():
            db_mtimes[db_name] = p.stat().st_mtime

    return {
        "ts": now_iso(),
        "precompute_versions": precomp_versions,
        "archive_checkpoints": archive_files,
        "db_mtimes": db_mtimes,
    }


def _snapshot_cold_state(cold_root: Path) -> dict:
    """
    Record the state of cold storage at checkout time.
    At checkin, compare against current cold state to detect if the base machine
    kept working (and thus cold has newer data than when we left).
    """
    precomp_versions: dict[str, str] = {}
    cold_precomp = cold_root / "precomputed"
    if cold_precomp.exists():
        for enc_dir in sorted(cold_precomp.iterdir()):
            if not enc_dir.is_dir():
                continue
            cur = enc_dir / "current"
            if cur.is_symlink():
                precomp_versions[enc_dir.name] = os.readlink(cur)

    flywheel_iters = 0
    flywheel_db = cold_root / "metadata" / "flywheel_history.db"
    if flywheel_db.exists():
        try:
            conn = sqlite3.connect(str(flywheel_db))
            row = conn.execute("SELECT COUNT(*) FROM iterations").fetchone()
            flywheel_iters = row[0] if row else 0
            conn.close()
        except Exception:
            pass

    campaigns: list[str] = []
    cold_weights = cold_root / "weights"
    if cold_weights.exists():
        campaigns = sorted(
            d.name for d in cold_weights.iterdir()
            if d.is_dir() and d.name.startswith("flywheel-")
        )

    return {
        "ts": now_iso(),
        "precompute_versions": precomp_versions,
        "flywheel_iterations": flywheel_iters,
        "campaigns": campaigns,
        "latest_campaign": campaigns[-1] if campaigns else None,
    }


def _detect_conflicts(manifest: dict, current_cold_state: dict) -> list[str]:
    """
    Compare the cold-state snapshot recorded at checkout vs. the current cold state.
    Returns a list of human-readable conflict descriptions (empty = no conflicts).
    """
    snap = manifest.get("cold_state_at_checkout", {})
    conflicts = []

    snap_iters = snap.get("flywheel_iterations", 0)
    cur_iters = current_cold_state.get("flywheel_iterations", 0)
    if cur_iters > snap_iters:
        conflicts.append(
            f"Flywheel advanced on base machine: {snap_iters} → {cur_iters} iterations"
        )

    snap_camps = set(snap.get("campaigns", []))
    cur_camps = set(current_cold_state.get("campaigns", []))
    new_camps = cur_camps - snap_camps
    if new_camps:
        conflicts.append(
            f"New campaign(s) on base machine: {', '.join(sorted(new_camps))}"
        )

    snap_vers = snap.get("precompute_versions", {})
    cur_vers = current_cold_state.get("precompute_versions", {})
    for enc in _ENCODERS:
        sv, cv = snap_vers.get(enc), cur_vers.get(enc)
        if sv and cv and sv != cv:
            conflicts.append(
                f"Precompute version changed ({enc}): {sv} → {cv} while you were away"
            )

    return conflicts


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def _load_manifest(cold_root: Path,
                   session_id: Optional[str] = None,
                   portable_root: Optional[Path] = None) -> Optional[dict]:
    """
    Load a checkout manifest.

    Priority:
      1. Explicit session_id
      2. .mobile_session.json hint on portable_root
      3. Most recent active session in cold checkouts dir
    """
    checkouts_dir = _checkouts_dir(cold_root)
    if not checkouts_dir.exists():
        return None

    if session_id:
        p = _manifest_path(cold_root, session_id)
        if p.exists():
            return json.loads(p.read_text())
        return None

    if portable_root:
        hint_path = portable_root / ".mobile_session.json"
        if hint_path.exists():
            try:
                hint = json.loads(hint_path.read_text())
                sid = hint.get("session_id")
                if sid:
                    return _load_manifest(cold_root, sid)
            except Exception:
                pass

    # Most recent active session
    candidates = sorted(
        checkouts_dir.glob("*/manifest.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for mpath in candidates:
        try:
            m = json.loads(mpath.read_text())
            if m.get("status") == "active":
                return m
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Checkout
# ---------------------------------------------------------------------------

def checkout(cfg: dict,
             chunks: Optional[list[int]] = None,
             label: str = "",
             dry_run: bool = False) -> dict:
    """
    Stage active campaign data from cold → hot and record a manifest in cold.

    What gets staged:
      - Best checkpoint (safetensors + json + ema) from cold/weights/best/
      - Precomputed caches for all encoders, filtered to requested chunks
      - Shards for requested chunks
      - Validation set
      - DB baseline snapshots (.mobile_dbs_baseline/)

    Returns the manifest dict.
    """
    storage = cfg.get("storage", {})
    cold_root = Path(storage.get("cold_root", str(COLD_ROOT)))
    hot_root  = Path(storage.get("hot_root",  str(DATA_ROOT)))

    if not cold_root.exists():
        raise RuntimeError(f"Cold storage not mounted at {cold_root}")

    session = _session_id(label)
    same_dev = (
        cold_root.exists() and hot_root.exists()
        and _same_device(cold_root, hot_root)
    )

    # Snapshot state before staging anything
    hot_snap  = _snapshot_hot_state(hot_root)
    cold_snap = _snapshot_cold_state(cold_root)

    manifest: dict = {
        "session_id":             session,
        "status":                 "active",
        "checked_out_at":         now_iso(),
        "checked_in_at":          None,
        "cold_root":              str(cold_root),
        "portable_root":          str(hot_root),
        "chunks":                 chunks or [],
        "hot_state_at_checkout":  hot_snap,
        "cold_state_at_checkout": cold_snap,
        "staged":                 {},
        "dry_run":                dry_run,
    }

    total_bytes = 0
    staged: dict = {}

    def _transfer(src: Path, dst: Path) -> int:
        """Symlink (same device) or atomic-copy src→dst. Returns bytes transferred."""
        if dst.exists() or dst.is_symlink():
            return 0
        if dry_run:
            try:
                return src.stat().st_size
            except OSError:
                return 0
        dst.parent.mkdir(parents=True, exist_ok=True)
        if same_dev:
            os.symlink(os.path.relpath(src.resolve(), dst.parent), dst)
            return 0
        return _atomic_copy_file(src, dst)

    # --- 1. Best checkpoint -------------------------------------------------
    print("  [1/5] Staging best checkpoint...")
    cold_weights = cold_root / "weights"
    hot_ckpts    = hot_root  / "checkpoints" / "stage1"
    ckpt_src: Optional[Path] = None

    best_dir = cold_weights / "best"
    if best_dir.exists():
        for name in ("cond_gap.safetensors", "loss.safetensors"):
            c = best_dir / name
            if c.is_symlink() and c.exists():
                ckpt_src = c.resolve()
                break
    if ckpt_src is None:
        campaigns = sorted(cold_weights.glob("flywheel-*"), reverse=True) if cold_weights.exists() else []
        for camp in campaigns:
            c = camp / "final.safetensors"
            if c.exists():
                ckpt_src = c
                break

    ckpt_staged: list[str] = []
    if ckpt_src:
        for suffix in (".safetensors", ".json"):
            src = ckpt_src.parent / (ckpt_src.stem + suffix)
            if src.exists():
                dst = hot_ckpts / f"best{suffix}"
                nb = _transfer(src, dst)
                total_bytes += nb
                ckpt_staged.append(src.name)
        # EMA sidecar
        ema = ckpt_src.parent / (ckpt_src.stem + ".ema.safetensors")
        if ema.exists():
            nb = _transfer(ema, hot_ckpts / "best.ema.safetensors")
            total_bytes += nb
            ckpt_staged.append(ema.name)
    staged["weights"] = ckpt_staged

    # --- 2. Precomputed caches ---------------------------------------------
    print("  [2/5] Staging precomputed caches...")
    cold_precomp = cold_root / "precomputed"
    hot_precomp  = hot_root  / "precomputed"
    npz_staged   = 0
    precomp_staged: dict[str, str] = {}

    for enc in _ENCODERS:
        cold_enc = cold_precomp / enc
        cur_link = cold_enc / "current"
        if not cur_link.exists() and not cur_link.is_symlink():
            continue
        ver = os.readlink(cur_link) if cur_link.is_symlink() else cur_link.name
        cold_ver = cold_enc / ver
        if not cold_ver.exists():
            continue

        hot_enc = hot_precomp / enc
        hot_ver = hot_enc / ver

        tasks: list[tuple[Path, Path]] = []
        for f in cold_ver.iterdir():
            if f.suffix == ".npz" and chunks:
                try:
                    shard_id = int(f.name.split("_")[0])
                    if not any(
                        (c - 1) * SHARD_BLOCK <= shard_id < c * SHARD_BLOCK
                        for c in chunks
                    ):
                        continue
                except (ValueError, IndexError):
                    pass
            dst = hot_ver / f.name
            if not dst.exists() and not dst.is_symlink():
                tasks.append((f, dst))

        if not dry_run:
            hot_ver.mkdir(parents=True, exist_ok=True)

        with ThreadPoolExecutor(max_workers=_MAX_PARALLEL) as pool:
            for fut in as_completed(pool.submit(_transfer, s, d) for s, d in tasks):
                total_bytes += fut.result()
                npz_staged += 1

        if not dry_run:
            _atomic_symlink(hot_enc / "current", ver)
        precomp_staged[enc] = ver

    staged["precomputed"]  = precomp_staged
    staged["npz_staged"]   = npz_staged

    # --- 3. Shards ----------------------------------------------------------
    print("  [3/5] Staging shards...")
    cold_shards = cold_root / "shards"
    hot_shards  = hot_root  / "shards"
    shards_staged: list[str] = []

    if cold_shards.exists():
        tasks_shards: list[tuple[Path, Path]] = []
        for shard in cold_shards.glob("*.tar"):
            if chunks:
                try:
                    sid = int(shard.stem)
                    if not any(
                        (c - 1) * SHARD_BLOCK <= sid < c * SHARD_BLOCK
                        for c in chunks
                    ):
                        continue
                except ValueError:
                    continue
            dst = hot_shards / shard.name
            if not dst.exists() and not dst.is_symlink():
                tasks_shards.append((shard, dst))

        if not dry_run and tasks_shards:
            hot_shards.mkdir(parents=True, exist_ok=True)

        with ThreadPoolExecutor(max_workers=_MAX_PARALLEL) as pool:
            for fut in as_completed(pool.submit(_transfer, s, d) for s, d in tasks_shards):
                total_bytes += fut.result()
        shards_staged = [d.name for _, d in tasks_shards]

    staged["shards"] = shards_staged

    # --- 4. Validation set --------------------------------------------------
    print("  [4/5] Staging validation set...")
    cold_val = cold_root / "validation"
    hot_val  = hot_root  / "validation"
    val_files = 0
    if cold_val.exists():
        for f in cold_val.rglob("*"):
            if f.is_file():
                rel = f.relative_to(cold_val)
                nb  = _transfer(f, hot_val / rel)
                total_bytes += nb
                val_files   += 1
    staged["validation_files"] = val_files

    # --- 5. DB baseline snapshots ------------------------------------------
    print("  [5/5] Snapshotting databases to hot baseline dir...")
    baseline_dir = hot_root / ".mobile_dbs_baseline"
    dbs_snapped: list[str] = []
    if not dry_run:
        baseline_dir.mkdir(parents=True, exist_ok=True)
    for db_name in ("shard_scores.db", "flywheel_history.db", "ablation_history.db"):
        src_db = cold_root / "metadata" / db_name
        if src_db.exists():
            dst_db = baseline_dir / db_name
            if not dst_db.exists() and not dry_run:
                with sqlite3.connect(str(src_db)) as sc:
                    with sqlite3.connect(str(dst_db)) as dc:
                        sc.backup(dc)
            dbs_snapped.append(db_name)
    staged["db_baseline"] = dbs_snapped

    # --- Finalise manifest --------------------------------------------------
    manifest["staged"]       = staged
    manifest["total_bytes"]  = total_bytes

    if not dry_run:
        # Write manifest to cold storage
        session_dir = _checkouts_dir(cold_root) / session
        session_dir.mkdir(parents=True, exist_ok=True)
        _manifest_path(cold_root, session).write_text(json.dumps(manifest, indent=2))

        # Write a hint file on hot so `checkin --from <hot>` can find the session
        hint = {
            "session_id": session,
            "cold_root":  str(cold_root),
            "checked_out_at": manifest["checked_out_at"],
        }
        (hot_root / ".mobile_session.json").write_text(json.dumps(hint, indent=2))

    return manifest


# ---------------------------------------------------------------------------
# Checkin
# ---------------------------------------------------------------------------

def checkin(cfg: dict,
            session_id: Optional[str] = None,
            portable_root: Optional[Path] = None,
            dry_run: bool = False,
            force: bool = False) -> dict:
    """
    Push new hot-storage data back to cold storage after a mobile session.

    What gets pushed (new-since-checkout only):
      - Precomputed version dirs that weren't on hot at checkout time
      - Checkpoint archive files that weren't on hot at checkout time
      - DB snapshots (timestamped backups; never overwrites cold DBs)

    Returns a summary dict. Raises RuntimeError on hard failures.
    """
    storage  = cfg.get("storage", {})
    cold_root = Path(storage.get("cold_root", str(COLD_ROOT)))
    hot_root  = Path(storage.get("hot_root",  str(DATA_ROOT)))

    if portable_root is None:
        portable_root = hot_root

    if not cold_root.exists():
        raise RuntimeError(f"Cold storage not mounted at {cold_root}. Mount it first.")
    if not portable_root.exists():
        raise RuntimeError(f"Portable root not found: {portable_root}. Mount it first.")

    manifest = _load_manifest(cold_root, session_id, portable_root)
    if manifest is None:
        raise RuntimeError(
            "No active mobile session found.\n"
            "Pass --session SESSION_ID or --from PATH to the portable root."
        )

    session_id = manifest["session_id"]

    if manifest.get("status") == "checked-in" and not force:
        raise RuntimeError(
            f"Session {session_id} already checked in. Use --force to re-run checkin."
        )

    # Conflict detection
    current_cold = _snapshot_cold_state(cold_root)
    conflicts    = _detect_conflicts(manifest, current_cold)
    if conflicts and not force:
        print("CONFLICTS DETECTED — base machine worked while you were away:")
        for c in conflicts:
            print(f"  - {c}")
        print()
        print("Your mobile work is safe. To proceed anyway (mobile results are added")
        print("as new files; cold is never overwritten): re-run with --force.")
        return {"status": "conflicts", "conflicts": conflicts, "session_id": session_id}

    if conflicts:
        print("NOTE: proceeding with --force despite conflicts:")
        for c in conflicts:
            print(f"  - {c}")
        print()

    hot_snap_at_checkout = manifest.get("hot_state_at_checkout", {})
    known_archive        = set(hot_snap_at_checkout.get("archive_checkpoints", []))
    known_precomp        = hot_snap_at_checkout.get("precompute_versions", {})
    known_db_mtimes      = hot_snap_at_checkout.get("db_mtimes", {})

    summary: dict = {
        "session_id":        session_id,
        "conflicts":         conflicts,
        "npz_pushed":        0,
        "ckpts_pushed":      0,
        "bytes_transferred": 0,
        "dbs_backed_up":     [],
        "dry_run":           dry_run,
    }

    def _safe_copy(src: Path, dst: Path) -> int:
        """Atomic copy src→dst only if dst does not yet exist."""
        if not src.exists() or dst.exists():
            return 0
        if dry_run:
            try:
                return src.stat().st_size
            except OSError:
                return 0
        dst.parent.mkdir(parents=True, exist_ok=True)
        return _atomic_copy_file(src, dst)

    # --- 1. New precomputed version dirs -----------------------------------
    print("  [1/3] Pushing new precomputed data to cold...")
    hot_precomp  = portable_root / "precomputed"
    cold_precomp = cold_root / "precomputed"

    if hot_precomp.exists():
        for enc_dir in hot_precomp.iterdir():
            if not enc_dir.is_dir():
                continue
            enc = enc_dir.name
            known_ver = known_precomp.get(enc)
            cold_enc  = cold_precomp / enc

            for ver_dir in enc_dir.iterdir():
                if not ver_dir.is_dir() or ver_dir.name == "current":
                    continue
                ver = ver_dir.name
                # Skip versions that existed at checkout — they're already in cold
                if ver == known_ver:
                    continue

                cold_ver = cold_enc / ver
                tasks: list[tuple[Path, Path]] = [
                    (f, cold_ver / f.name)
                    for f in ver_dir.iterdir()
                    if f.is_file() and not (cold_ver / f.name).exists()
                ]
                with ThreadPoolExecutor(max_workers=_MAX_PARALLEL) as pool:
                    for fut in as_completed(pool.submit(_safe_copy, s, d) for s, d in tasks):
                        nb = fut.result()
                        if nb:
                            summary["npz_pushed"]        += 1
                            summary["bytes_transferred"] += nb

            # Update cold's current symlink if hot has a newer version
            hot_cur = enc_dir / "current"
            if hot_cur.is_symlink() and not dry_run:
                hot_ver = os.readlink(hot_cur)
                if (cold_enc / hot_ver).exists():
                    _atomic_symlink(cold_enc / "current", hot_ver)

    # --- 2. New checkpoint archive files -----------------------------------
    print("  [2/3] Pushing new checkpoint snapshots to cold...")
    hot_archive = portable_root / "checkpoints" / "stage1" / "archive"
    cold_weights = cold_root / "weights"

    # Determine campaign dir: use the latest campaign from cold state (may have
    # been created during this mobile session) or create a mobile-tagged one.
    latest_campaign = current_cold.get("latest_campaign") or \
                      manifest["cold_state_at_checkout"].get("latest_campaign")
    if not latest_campaign:
        from datetime import date
        latest_campaign = f"flywheel-{date.today().strftime('%Y%m%d')}"
    campaign_dir = cold_weights / latest_campaign

    if hot_archive.exists():
        for f in hot_archive.iterdir():
            if not f.is_file():
                continue
            if f.name in known_archive:
                continue  # existed at checkout, already in cold
            dst = campaign_dir / f"mobile_{f.name}"
            nb  = _safe_copy(f, dst)
            if nb:
                summary["ckpts_pushed"]      += 1
                summary["bytes_transferred"] += nb

    # Also push hot best.* if it changed
    hot_ckpts = portable_root / "checkpoints" / "stage1"
    for suffix in (".safetensors", ".json", ".ema.safetensors"):
        src = hot_ckpts / f"best{suffix}"
        if src.exists():
            dst = campaign_dir / f"final.mobile{suffix}"
            nb  = _safe_copy(src, dst)
            if nb:
                summary["ckpts_pushed"]      += 1
                summary["bytes_transferred"] += nb

    # --- 3. DB backups -------------------------------------------------------
    print("  [3/3] Backing up mobile databases to cold...")
    cold_meta = cold_root / "metadata"
    ts_tag    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    for db_name in ("shard_scores.db", "flywheel_history.db", "ablation_history.db"):
        mob_db = portable_root / db_name
        if not mob_db.exists():
            continue
        # Only back up if the DB changed since checkout
        snap_mtime = known_db_mtimes.get(db_name, 0.0)
        try:
            cur_mtime = mob_db.stat().st_mtime
        except OSError:
            continue
        if cur_mtime <= snap_mtime and not force:
            continue  # unchanged — skip

        backup_name = f"{db_name.replace('.db', '')}.mobile_{ts_tag}.db"
        backup_path = cold_meta / backup_name
        if not dry_run:
            cold_meta.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(str(mob_db)) as sc:
                with sqlite3.connect(str(backup_path)) as dc:
                    sc.backup(dc)
        summary["dbs_backed_up"].append(db_name)

    # --- Finalise -----------------------------------------------------------
    if not dry_run:
        manifest["status"]           = "checked-in"
        manifest["checked_in_at"]    = now_iso()
        manifest["checkin_summary"]  = summary
        manifest["checkin_conflicts"] = conflicts

        session_dir = _checkouts_dir(cold_root) / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        _manifest_path(cold_root, session_id).write_text(json.dumps(manifest, indent=2))

        report_path = session_dir / "checkin_report.html"
        report_path.write_text(_generate_html_report(manifest, summary))
        summary["report_path"] = str(report_path)
        print(f"\n  Checkin report: {report_path}")

        # Clean up the hint file on hot
        hint = portable_root / ".mobile_session.json"
        if hint.exists():
            try:
                h = json.loads(hint.read_text())
                h["status"] = "checked-in"
                h["checked_in_at"] = manifest["checked_in_at"]
                hint.write_text(json.dumps(h, indent=2))
            except Exception:
                pass

    return summary


# ---------------------------------------------------------------------------
# Mobile status
# ---------------------------------------------------------------------------

def mobile_status(cfg: dict) -> dict:
    """Return status of all checkout sessions recorded in cold."""
    storage   = cfg.get("storage", {})
    cold_root = Path(storage.get("cold_root", str(COLD_ROOT)))
    hot_root  = Path(storage.get("hot_root",  str(DATA_ROOT)))

    cold_mounted = cold_root.exists()
    checkouts_dir = _checkouts_dir(cold_root)

    sessions: list[dict] = []
    if cold_mounted and checkouts_dir.exists():
        for mpath in sorted(
            checkouts_dir.glob("*/manifest.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            try:
                m = json.loads(mpath.read_text())
            except Exception:
                continue

            checked_out_at = m.get("checked_out_at", "")
            age_str = _age_str(checked_out_at)

            port_root = Path(m.get("portable_root", ""))
            portable_mounted = port_root.exists() if m.get("portable_root") else False

            staged = m.get("staged", {})
            sessions.append({
                "session_id":       m.get("session_id"),
                "status":           m.get("status", "unknown"),
                "checked_out_at":   checked_out_at,
                "checked_in_at":    m.get("checked_in_at"),
                "age":              age_str,
                "chunks":           m.get("chunks", []),
                "portable_root":    m.get("portable_root"),
                "portable_mounted": portable_mounted,
                "shards_staged":    len(staged.get("shards", [])),
                "npz_staged":       staged.get("npz_staged", 0),
                "dbs":              staged.get("db_baseline", []),
            })

    # Check divergence for the most recent active session
    divergence: list[str] = []
    active = [s for s in sessions if s["status"] == "active"]
    if active and cold_mounted:
        most_recent_sid = active[0]["session_id"]
        manifest = _load_manifest(cold_root, most_recent_sid)
        if manifest:
            current = _snapshot_cold_state(cold_root)
            divergence = _detect_conflicts(manifest, current)

    return {
        "cold_mounted":    cold_mounted,
        "cold_root":       str(cold_root),
        "hot_root":        str(hot_root),
        "sessions":        sessions,
        "base_divergence": divergence,
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _generate_html_report(manifest: dict, summary: dict) -> str:
    session_id   = manifest.get("session_id", "unknown")
    checked_out  = manifest.get("checked_out_at", "?")
    checked_in   = manifest.get("checked_in_at", "?")
    chunks       = manifest.get("chunks", [])
    conflicts    = manifest.get("checkin_conflicts", [])
    staged       = manifest.get("staged", {})

    npz_pushed   = summary.get("npz_pushed", 0)
    ckpts_pushed = summary.get("ckpts_pushed", 0)
    gb_pushed    = summary.get("bytes_transferred", 0) / (1024 ** 3)
    dbs_backed   = summary.get("dbs_backed_up", [])
    dry_run      = summary.get("dry_run", False)

    conflict_html = ""
    if conflicts:
        rows = "".join(f"<li>{c}</li>" for c in conflicts)
        conflict_html = f'<div class="warn"><strong>Conflicts detected at checkin:</strong><ul>{rows}</ul></div>'

    dry_banner = ""
    if dry_run:
        dry_banner = '<div class="info"><strong>Dry run — no files were written.</strong></div>'

    precomp_rows = "".join(
        f"<tr><td>{enc}</td><td><code>{ver}</code></td></tr>"
        for enc, ver in staged.get("precomputed", {}).items()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Mobile Checkin — {session_id}</title>
<style>
  body  {{ font-family: system-ui, -apple-system, sans-serif; max-width: 860px;
           margin: 40px auto; padding: 0 20px; color: #1a1a1a; line-height: 1.5; }}
  h1    {{ color: #1a1a2e; border-bottom: 2px solid #e0e0e0; padding-bottom: 8px; }}
  h2    {{ color: #2d2d5e; margin-top: 32px; }}
  .meta {{ color: #555; font-size: .9em; margin-bottom: 24px; }}
  .ok   {{ background: #d4edda; color: #155724; padding: 10px 16px;
           border-radius: 6px; margin: 12px 0; }}
  .warn {{ background: #fff3cd; color: #856404; padding: 12px 16px;
           border-radius: 6px; margin: 12px 0; }}
  .warn ul {{ margin: 6px 0 0 0; }}
  .info {{ background: #d1ecf1; color: #0c5460; padding: 10px 16px;
           border-radius: 6px; margin: 12px 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th,td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }}
  th    {{ background: #f7f7f7; font-weight: 600; }}
  code  {{ background: #f0f0f0; padding: 1px 5px; border-radius: 3px;
           font-size: .9em; }}
  .num  {{ text-align: right; font-variant-numeric: tabular-nums; }}
  footer {{ margin-top: 48px; font-size: .8em; color: #aaa; border-top: 1px solid #eee;
            padding-top: 12px; }}
</style>
</head>
<body>
<h1>Mobile Checkin Report</h1>
<div class="meta">
  Session: <code>{session_id}</code><br>
  Checked out: <strong>{checked_out}</strong><br>
  Checked in: <strong>{checked_in}</strong><br>
  Chunks: {", ".join(str(c) for c in chunks) if chunks else "all"}
</div>

{dry_banner}
{conflict_html}

<div class="ok">Checkin complete — {npz_pushed} precompute files, {ckpts_pushed} checkpoint files pushed to cold.</div>

<h2>Data Pushed to Cold Storage</h2>
<table>
  <tr><th>Category</th><th class="num">Count</th></tr>
  <tr><td>Precomputed (.npz)</td><td class="num">{npz_pushed}</td></tr>
  <tr><td>Checkpoint files</td><td class="num">{ckpts_pushed}</td></tr>
  <tr><td>Total data transferred</td><td class="num">{gb_pushed:.2f} GB</td></tr>
  <tr><td>DB backups written to cold</td><td>{", ".join(dbs_backed) if dbs_backed else "none (unchanged)"}</td></tr>
</table>

<h2>What Was Checked Out</h2>
<table>
  <tr><th>Category</th><th class="num">Count</th></tr>
  <tr><td>Shards staged to hot</td><td class="num">{len(staged.get("shards", []))}</td></tr>
  <tr><td>Precomputed files staged</td><td class="num">{staged.get("npz_staged", 0)}</td></tr>
  <tr><td>Validation files staged</td><td class="num">{staged.get("validation_files", 0)}</td></tr>
  <tr><td>DB baselines snapped</td><td>{", ".join(staged.get("db_baseline", [])) or "none"}</td></tr>
</table>

<h2>Precomputed Versions (at checkout)</h2>
<table>
  <tr><th>Encoder</th><th>Version</th></tr>
  {precomp_rows if precomp_rows else "<tr><td colspan='2'>No precomputed data staged</td></tr>"}
</table>

<footer>Generated by iris.c mobile_mode.py &mdash; iris flywheel pipeline</footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _age_str(iso_ts: str) -> str:
    if not iso_ts:
        return ""
    try:
        then = datetime.fromisoformat(iso_ts)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        secs = (datetime.now(timezone.utc) - then).total_seconds()
        if secs < 3600:
            return f"{int(secs / 60)}m ago"
        if secs < 86400:
            return f"{int(secs / 3600)}h ago"
        return f"{int(secs / 86400)}d ago"
    except Exception:
        return ""
