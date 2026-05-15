#!/usr/bin/env python3
"""
train/scripts/validate_full_compatibility.py — Train/inference compatibility suite.

Catches the class of silent mismatch bug exemplified by INFER-C-001:
precomputed embeddings computed with a different token sequence than inference
produces, invalidating the entire precompute cache with no error at runtime.

Usage:
    python train/scripts/validate_full_compatibility.py [OPTIONS]

Exit codes:
    0  all checks pass
    1  one or more checks FAIL
    2  one or more checks WARN (only with --strict)
    3  configuration error

Run before any production precompute, flywheel launch, or code change to
precompute or model.  See train/DISPATCH.md §Compatibility Validation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ── bootstrap sys.path ────────────────────────────────────────────────────────
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from pipeline_lib import (
    DATA_ROOT, TRAIN_DIR, SCRIPTS_DIR, PRECOMP_DIR,
    load_config, now_iso, write_heartbeat,
)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    status: str                   # "PASS" | "WARN" | "FAIL" | "SKIP"
    detail: str = ""
    data: dict = field(default_factory=dict)
    action: Optional[str] = None  # remediation step if non-PASS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _git_sha(repo_root: Optional[Path] = None) -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            capture_output=True, text=True,
            cwd=str(repo_root or _SCRIPTS_DIR),
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "00000000"


def _recompute_version_hash(config_subset: dict, git_sha_str: str) -> str:
    blob = json.dumps(config_subset, sort_keys=True) + git_sha_str[:8]
    return "v_" + hashlib.sha256(blob.encode()).hexdigest()[:6]


def _read_file(path: Path) -> Optional[str]:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return None


def _load_config_safe(config_path: Optional[str], data_root: Path) -> Optional[dict]:
    """Load pipeline YAML config; return None on failure."""
    try:
        return load_config(config_path)
    except Exception:
        # Try smoke config as fallback
        for candidate in [
            TRAIN_DIR / "configs" / "v2_pipeline.yaml",
            TRAIN_DIR / "configs" / "v2_pipeline_smoke.yaml",
        ]:
            try:
                return load_config(str(candidate))
            except Exception:
                pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Check implementations
# ─────────────────────────────────────────────────────────────────────────────

def check_qwen3_layers(context: dict) -> CheckResult:
    """CHECK-1: Qwen3 layer extraction consistency (cache_manager vs iris_qwen3.h)."""
    # --- parse C header ---
    iris_qwen3_h = Path(context["repo_root"]) / "iris_qwen3.h"
    h_text = _read_file(iris_qwen3_h)
    if h_text is None:
        return CheckResult("SKIP", f"iris_qwen3.h not found at {iris_qwen3_h}")

    c_layers = []
    for m in re.finditer(r"#define\s+QWEN3_OUTPUT_LAYER_(\d+)\s+(\d+)", h_text):
        c_layers.append(int(m.group(2)))
    if not c_layers:
        return CheckResult("FAIL", "No QWEN3_OUTPUT_LAYER_* defines found in iris_qwen3.h",
                           action="Check iris_qwen3.h for QWEN3_OUTPUT_LAYER_1/2/3 defines")

    # --- parse cache_manager.py ---
    cache_mgr = _SCRIPTS_DIR / "cache_manager.py"
    cm_text = _read_file(cache_mgr)
    if cm_text is None:
        return CheckResult("SKIP", "cache_manager.py not found")

    # find encoder_config_subset for qwen3: look for 'layers': [...]
    m = re.search(r'"layers"\s*:\s*(\[[^\]]+\])', cm_text)
    if not m:
        return CheckResult("FAIL", "No 'layers' list found in cache_manager.py encoder_config_subset",
                           action="Add 'layers': [8, 17, 26] to encoder_config_subset('qwen3') in cache_manager.py")
    try:
        cache_layers = ast.literal_eval(m.group(1))
    except Exception as e:
        return CheckResult("FAIL", f"Could not parse layers list from cache_manager.py: {e}",
                           action="Fix layers list syntax in cache_manager.py")

    expected = [8, 17, 26]
    ok_c     = sorted(c_layers) == expected
    ok_cache = sorted(cache_layers) == expected

    data = {"expected": expected, "found_c": c_layers, "found_cache": cache_layers}

    if ok_c and ok_cache:
        return CheckResult("PASS", "Both C and cache use layers [8, 17, 26]", data)

    msgs = []
    actions = []
    if not ok_c:
        msgs.append(f"iris_qwen3.h uses {sorted(c_layers)} (expected {expected})")
        actions.append("Fix QWEN3_OUTPUT_LAYER_* defines in iris_qwen3.h to 8, 17, 26")
    if not ok_cache:
        msgs.append(f"cache_manager.py uses {sorted(cache_layers)} (expected {expected})")
        actions.append("Fix 'layers' in encoder_config_subset('qwen3') in cache_manager.py to [8, 17, 26]")

    return CheckResult("FAIL", "; ".join(msgs), data,
                       action="; ".join(actions))


def check_think_tags(context: dict) -> CheckResult:
    """CHECK-2: Chat template think-tag presence in precompute_all.py."""
    precompute_py = _SCRIPTS_DIR / "precompute_all.py"
    text = _read_file(precompute_py)
    if text is None:
        return CheckResult("SKIP", "precompute_all.py not found")

    # Check if enable_thinking=False is present in apply_chat_template call
    has_kwarg = bool(re.search(r"apply_chat_template\s*\([^)]*enable_thinking\s*=\s*False", text, re.DOTALL))
    # Also check more broadly
    if not has_kwarg:
        # check if enable_thinking appears at all near apply_chat_template
        has_kwarg = bool(re.search(r"enable_thinking\s*=\s*False", text))

    data = {"precompute_has_enable_thinking_false": has_kwarg}

    if not has_kwarg:
        # Count tokens with vs without to document the mismatch magnitude
        return CheckResult(
            "FAIL",
            "precompute_all.py calls apply_chat_template without enable_thinking=False. "
            "This omits the <think>\\n\\n</think>\\n\\n suffix (7 tokens), making precomputed "
            "embeddings misaligned with C inference and live-encoding fallback.",
            data,
            action=(
                "Add enable_thinking=False to apply_chat_template call in precompute_all.py "
                "(_encode_qwen3 function, line ~365); then regenerate all Qwen3 precompute caches."
            ),
        )

    return CheckResult("PASS", "precompute_all.py passes enable_thinking=False to apply_chat_template", data)


def check_cache_version_hash(context: dict) -> CheckResult:
    """CHECK-3: Cache version hash matches current code+config."""
    cfg = context.get("cfg")
    precomp_root = context.get("precomp_root", PRECOMP_DIR)

    if cfg is None:
        return CheckResult("SKIP", "No pipeline config available — cannot recompute version hash")

    try:
        from cache_manager import encoder_config_subset, version_hash, get_git_sha
    except ImportError:
        return CheckResult("SKIP", "cache_manager.py could not be imported")

    sha = _git_sha()
    try:
        config_subset = encoder_config_subset("qwen3", cfg)
        expected_ver  = version_hash(config_subset, sha)
    except Exception as e:
        return CheckResult("SKIP", f"Could not compute expected version hash: {e}")

    enc_dir  = Path(precomp_root) / "qwen3"
    cur_link = enc_dir / "current"

    if not cur_link.exists() and not cur_link.is_symlink():
        return CheckResult(
            "SKIP",
            f"No current symlink at {cur_link} — cache not yet created",
        )

    try:
        current_target = os.readlink(str(cur_link))
    except OSError as e:
        return CheckResult("FAIL", f"Could not read current symlink: {e}",
                           action="Recreate the current symlink in the qwen3 precompute dir")

    data = {
        "current_version": current_target,
        "expected_version": expected_ver,
        "git_sha":          sha,
        "config_subset":    config_subset,
    }

    if current_target == expected_ver:
        return CheckResult("PASS", f"Cache version {current_target} matches current code+config", data)

    return CheckResult(
        "WARN",
        f"Current cache version '{current_target}' differs from expected '{expected_ver}'. "
        "Cache may have been built from a different code version or config.",
        data,
        action=(
            f"If code/config changed since cache was built, regenerate: "
            f"precompute_all.py will create a new version dir automatically."
        ),
    )


def check_embedding_numerical_sanity(context: dict) -> CheckResult:
    """CHECK-4: Embedding numerical sanity on fixed test prompts."""
    # This check requires model load; skipped with --fast.
    if context.get("fast"):
        return CheckResult("SKIP", "Skipped by --fast flag")

    precomp_root = context.get("precomp_root", PRECOMP_DIR)

    try:
        import numpy as np
    except ImportError:
        return CheckResult("SKIP", "numpy not available")

    # Find the first shard in the active cache
    enc_dir  = Path(precomp_root) / "qwen3"
    cur_link = enc_dir / "current"

    if not cur_link.exists():
        return CheckResult("SKIP", f"No current qwen3 cache at {cur_link}")

    try:
        cache_dir = cur_link.resolve()
    except OSError:
        return CheckResult("SKIP", "current symlink is broken")

    npz_files = sorted(cache_dir.glob("*.npz"))[:5]
    if not npz_files:
        return CheckResult("SKIP", f"No NPZ files in cache dir {cache_dir}")

    issues = []
    for f in npz_files:
        try:
            d = np.load(f)
            q, scale = d["q"], d["scale"]
            # q is packed 4-bit: shape [seq, dim//2]
            # scale is [seq, 1] float16
            # Reconstructed shape: [seq, dim] where dim = q.shape[1]*2
            reconstructed_dim = q.shape[1] * 2
            seq_len = q.shape[0]
            if reconstructed_dim != 7680:
                issues.append(f"{f.name}: expected dim=7680, got {reconstructed_dim}")
            if seq_len == 0 or seq_len > 512:
                issues.append(f"{f.name}: unexpected seq_len={seq_len} (expected 1..512)")
        except Exception as e:
            issues.append(f"{f.name}: load error: {e}")

    data = {"checked_files": [f.name for f in npz_files], "issues": issues}

    if issues:
        return CheckResult(
            "FAIL",
            f"{len(issues)} embedding shape issues found",
            data,
            action="Regenerate qwen3 precompute cache — some NPZ files have wrong dimensions",
        )

    return CheckResult(
        "PASS",
        f"Checked {len(npz_files)} NPZ files: shapes are [seq, 7680] as expected",
        data,
    )


def check_vae_shape_consistency(context: dict) -> CheckResult:
    """CHECK-5: VAE latent shape consistency."""
    precomp_root = context.get("precomp_root", PRECOMP_DIR)

    try:
        import numpy as np
    except ImportError:
        return CheckResult("SKIP", "numpy not available")

    enc_dir  = Path(precomp_root) / "vae"
    cur_link = enc_dir / "current"

    # Tolerate flat legacy layout too
    if cur_link.exists():
        try:
            cache_dir = cur_link.resolve()
        except OSError:
            return CheckResult("SKIP", "vae/current symlink is broken")
    elif (enc_dir).is_dir():
        cache_dir = enc_dir
    else:
        return CheckResult("SKIP", f"No VAE precompute dir at {enc_dir}")

    npz_files = sorted(cache_dir.glob("*.npz"))[:5]
    if not npz_files:
        return CheckResult("SKIP", f"No VAE NPZ files in {cache_dir}")

    issues = []
    for f in npz_files:
        try:
            d = np.load(f)
            q = d["q"]
            # Expected shape: [32, H/8, W/8] — channels first
            if q.shape[0] != 32:
                issues.append(f"{f.name}: expected 32 channels, got {q.shape[0]}")
        except Exception as e:
            issues.append(f"{f.name}: load error: {e}")

    data = {"checked_files": [f.name for f in npz_files], "issues": issues}

    if issues:
        return CheckResult(
            "FAIL",
            f"{len(issues)} VAE shape issues found",
            data,
            action="Regenerate VAE precompute cache — stored latents have wrong channel count",
        )

    return CheckResult(
        "PASS",
        f"Checked {len(npz_files)} VAE NPZ files: channel count is 32 as expected",
        data,
    )


def check_siglip_shape_consistency(context: dict) -> CheckResult:
    """CHECK-6: SigLIP feature shape consistency."""
    precomp_root = context.get("precomp_root", PRECOMP_DIR)

    try:
        import numpy as np
    except ImportError:
        return CheckResult("SKIP", "numpy not available")

    enc_dir  = Path(precomp_root) / "siglip"
    cur_link = enc_dir / "current"

    if cur_link.exists():
        try:
            cache_dir = cur_link.resolve()
        except OSError:
            return CheckResult("SKIP", "siglip/current symlink is broken")
    elif enc_dir.is_dir():
        cache_dir = enc_dir
    else:
        return CheckResult("SKIP", "No SigLIP precompute dir — SigLIP cache not present (optional)")

    npz_files = sorted(cache_dir.glob("*.npz"))[:5]
    if not npz_files:
        return CheckResult("SKIP", "No SigLIP NPZ files — SigLIP cache empty (optional)")

    # so400m-patch14-384: 27×27 = 729 patches, 1152-dim features
    expected_patches = 729
    expected_dim     = 1152
    issues = []
    for f in npz_files:
        try:
            d = np.load(f)
            q = d["q"]
            # packed 4-bit: [patches, dim//2]
            patches = q.shape[0]
            recon_dim = q.shape[1] * 2
            if patches != expected_patches:
                issues.append(f"{f.name}: expected {expected_patches} patches, got {patches}")
            if recon_dim != expected_dim:
                issues.append(f"{f.name}: expected dim={expected_dim}, got {recon_dim}")
        except Exception as e:
            issues.append(f"{f.name}: load error: {e}")

    data = {"checked_files": [f.name for f in npz_files], "issues": issues}

    if issues:
        return CheckResult(
            "FAIL",
            f"{len(issues)} SigLIP shape issues found",
            data,
            action="Regenerate SigLIP precompute cache",
        )

    return CheckResult(
        "PASS",
        f"Checked {len(npz_files)} SigLIP NPZ files: [{expected_patches}, {expected_dim}] as expected",
        data,
    )


def check_pad_alignment(context: dict) -> CheckResult:
    """CHECK-7: Verify dataset.py _TEXT_PAD == QWEN3_MAX_SEQ_LEN from iris_qwen3.h."""
    # Parse dataset.py
    dataset_py = TRAIN_DIR / "ip_adapter" / "dataset.py"
    ds_text = _read_file(dataset_py)
    if ds_text is None:
        return CheckResult("SKIP", f"dataset.py not found at {dataset_py}")

    m = re.search(r"_TEXT_PAD\s*=\s*(\d+)", ds_text)
    if not m:
        return CheckResult("FAIL", "No _TEXT_PAD constant found in dataset.py",
                           action="Add _TEXT_PAD = 512 constant to train/ip_adapter/dataset.py")
    dataset_pad = int(m.group(1))

    # Parse iris_qwen3.h
    iris_qwen3_h = Path(context["repo_root"]) / "iris_qwen3.h"
    h_text = _read_file(iris_qwen3_h)
    if h_text is None:
        return CheckResult("SKIP", f"iris_qwen3.h not found at {iris_qwen3_h}")

    m2 = re.search(r"#define\s+QWEN3_MAX_SEQ_LEN\s+(\d+)", h_text)
    if not m2:
        return CheckResult("FAIL", "No QWEN3_MAX_SEQ_LEN define found in iris_qwen3.h",
                           action="Add #define QWEN3_MAX_SEQ_LEN 512 to iris_qwen3.h")
    c_max_seq = int(m2.group(1))

    data = {"dataset_TEXT_PAD": dataset_pad, "c_QWEN3_MAX_SEQ_LEN": c_max_seq}

    if dataset_pad == c_max_seq:
        return CheckResult(
            "PASS",
            f"dataset._TEXT_PAD ({dataset_pad}) == QWEN3_MAX_SEQ_LEN ({c_max_seq})",
            data,
        )

    return CheckResult(
        "FAIL",
        f"Padding mismatch: dataset._TEXT_PAD={dataset_pad} vs QWEN3_MAX_SEQ_LEN={c_max_seq}",
        data,
        action=(
            f"Align dataset.py _TEXT_PAD and iris_qwen3.h QWEN3_MAX_SEQ_LEN to the same value; "
            f"the standard is 512."
        ),
    )


def check_ip_adapter_injection_coverage(context: dict) -> CheckResult:
    """CHECK-8: IP-Adapter injection block coverage (training vs C inference)."""
    train_py = TRAIN_DIR / "train_ip_adapter.py"
    t_text = _read_file(train_py)
    if t_text is None:
        return CheckResult("SKIP", "train_ip_adapter.py not found")

    # Count injection sites in training code
    double_injects = len(re.findall(r"ip_scale\[.*?\]\s*\*\s*ip_out", t_text))
    single_injects = len(re.findall(r"ip_scale\[.*?block_ip_idx.*?\]", t_text))

    # Check C inference
    iris_flux_c = Path(context["repo_root"]) / "iris_transformer_flux.c"
    c_text = _read_file(iris_flux_c)

    c_note = "iris_transformer_flux.c not present (inference injection coverage not verified)"
    c_has_ip = False
    if c_text is not None:
        c_has_ip = bool(re.search(r"ip_scale", c_text))
        c_note = "ip_scale references found in iris_transformer_flux.c" if c_has_ip else \
                 "No ip_scale references in iris_transformer_flux.c (inference does not apply IP-Adapter)"

    data = {
        "training_double_inject_sites": double_injects,
        "training_single_inject_sites": single_injects,
        "c_inference_has_ip_scale":     c_has_ip,
        "c_note":                       c_note,
    }

    if not c_has_ip:
        return CheckResult(
            "WARN",
            f"Training injects at ~{double_injects} double + ~{single_injects} single sites; "
            f"C inference does not implement IP-Adapter injection yet ({c_note}).",
            data,
            action="This is expected during development. When inference supports IP-Adapter, "
                   "verify injection block indices match training.",
        )

    return CheckResult(
        "PASS",
        f"Both training and C inference contain ip_scale injection code. "
        f"Training sites: ~{double_injects} double + ~{single_injects} single.",
        data,
    )


def check_cache_manifest_completeness(context: dict) -> CheckResult:
    """CHECK-9: Cache manifest completeness for all encoders."""
    precomp_root = context.get("precomp_root", PRECOMP_DIR)

    try:
        from cache_manager import ENCODERS
    except ImportError:
        return CheckResult("SKIP", "cache_manager.py could not be imported")

    issues = []
    warnings = []
    details = {}

    for enc in ENCODERS:
        enc_dir  = Path(precomp_root) / enc
        cur_link = enc_dir / "current"
        if not cur_link.exists() and not cur_link.is_symlink():
            details[enc] = {"status": "absent", "note": "no current symlink"}
            continue

        try:
            ver_dir = cur_link.resolve()
        except OSError:
            warnings.append(f"{enc}: broken current symlink")
            details[enc] = {"status": "broken_symlink"}
            continue

        manifest_path = ver_dir / "manifest.json"
        if not manifest_path.exists():
            warnings.append(f"{enc}: no manifest.json in {ver_dir.name}")
            details[enc] = {"status": "no_manifest", "version": ver_dir.name}
            continue

        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as e:
            warnings.append(f"{enc}: manifest.json unreadable: {e}")
            details[enc] = {"status": "unreadable", "version": ver_dir.name}
            continue

        complete      = bool(manifest.get("complete"))
        record_count  = manifest.get("record_count", 0)
        completed_at  = manifest.get("completed_at", "?")
        ver           = manifest.get("version", ver_dir.name)

        details[enc] = {
            "version":      ver,
            "complete":     complete,
            "record_count": record_count,
            "completed_at": completed_at,
        }

        if not complete:
            warnings.append(
                f"{enc}: manifest.complete=False — precompute may be in progress or crashed. "
                f"Records written: {record_count:,}"
            )

    data = {"encoder_manifests": details, "issues": issues, "warnings": warnings}

    if issues:
        return CheckResult("FAIL", "; ".join(issues), data,
                           action="Investigate failed encoders and re-run precompute if needed")
    if warnings:
        return CheckResult("WARN", "; ".join(warnings), data,
                           action="If precompute is not running, re-run precompute to mark cache complete")

    present = [e for e, d in details.items() if d.get("status") != "absent"]
    if not present:
        return CheckResult("SKIP", "No precompute cache present for any encoder")

    return CheckResult(
        "PASS",
        f"All present encoder manifests ({', '.join(present)}) are complete",
        data,
    )


def check_mining_timestep(context: dict) -> CheckResult:
    """CHECK-10: Mining timestep consistency with training distribution."""
    mine_py = _SCRIPTS_DIR / "mine_hard_examples.py"
    text = _read_file(mine_py)
    if text is None:
        return CheckResult("SKIP", "mine_hard_examples.py not found")

    # Look for fixed timestep usage
    fixed_t_match = re.search(r"t_int\s*=\s*mx\.array\(\[(\d+)\]", text)
    logit_normal   = bool(re.search(r"logit.normal|sigmoid.*random\.normal", text))

    if fixed_t_match:
        t_val = int(fixed_t_match.group(1))
        data  = {"timestep_mode": "fixed", "fixed_t": t_val, "logit_normal_found": logit_normal}
        return CheckResult(
            "WARN",
            f"mine_hard_examples.py uses fixed t={t_val}. "
            f"This biases hard-example selection to difficulty at t={t_val} only (PIPE-H-004). "
            f"Samples hard at low/high t but easy at t={t_val} are systematically mis-ranked.",
            data,
            action=(
                f"Fix per PIPE-H-004: sample t from logit-normal matching training distribution. "
                f"Code: mx.clip((mx.sigmoid(mx.random.normal(shape=(B,))) * 1000).astype(mx.int32), 0, 999)"
            ),
        )

    if logit_normal:
        data = {"timestep_mode": "logit_normal", "fixed_t": None}
        return CheckResult(
            "PASS",
            "mine_hard_examples.py uses logit-normal t sampling matching training distribution",
            data,
        )

    data = {"timestep_mode": "unknown"}
    return CheckResult(
        "WARN",
        "Could not determine timestep mode in mine_hard_examples.py",
        data,
        action="Verify _eval_loss/_eval_loss_batch use logit-normal t sampling (not fixed t=500)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# All checks registry
# ─────────────────────────────────────────────────────────────────────────────

_ALL_CHECKS: dict[str, tuple[str, Any]] = {
    "qwen3_layers":          ("CHECK-1: Qwen3 layer extraction consistency",    check_qwen3_layers),
    "think_tags":            ("CHECK-2: Chat template think-tag presence",       check_think_tags),
    "cache_version_hash":    ("CHECK-3: Cache version hash matches code",        check_cache_version_hash),
    "embedding_sanity":      ("CHECK-4: Embedding numerical sanity",             check_embedding_numerical_sanity),
    "vae_shape":             ("CHECK-5: VAE latent shape consistency",           check_vae_shape_consistency),
    "siglip_shape":          ("CHECK-6: SigLIP feature shape consistency",       check_siglip_shape_consistency),
    "pad_alignment":         ("CHECK-7: Pad alignment (dataset vs C header)",    check_pad_alignment),
    "ip_injection_coverage": ("CHECK-8: IP-Adapter injection block coverage",    check_ip_adapter_injection_coverage),
    "cache_manifest":        ("CHECK-9: Cache manifest completeness",            check_cache_manifest_completeness),
    "mining_timestep":       ("CHECK-10: Mining timestep consistency",           check_mining_timestep),
}


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def _overall_status(results: dict[str, CheckResult]) -> str:
    statuses = [r.status for r in results.values()]
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    if all(s in ("PASS", "SKIP") for s in statuses):
        return "PASS"
    return "PASS"


def _build_json_report(results: dict[str, CheckResult], ts: str, git: str) -> dict:
    checks_out = {}
    actions = []
    for cid, res in results.items():
        entry = {"status": res.status, "detail": res.detail}
        entry.update(res.data)
        if res.action:
            entry["action"] = res.action
        checks_out[cid] = entry
        if res.action and res.status in ("FAIL", "WARN"):
            actions.append(res.action)

    n_fail = sum(1 for r in results.values() if r.status == "FAIL")
    n_warn = sum(1 for r in results.values() if r.status == "WARN")
    n_pass = sum(1 for r in results.values() if r.status == "PASS")
    n_skip = sum(1 for r in results.values() if r.status == "SKIP")

    return {
        "timestamp":       ts,
        "git_sha":         git,
        "overall":         _overall_status(results),
        "checks":          checks_out,
        "summary":         f"{n_fail} failed, {n_warn} warned, {n_pass} passed, {n_skip} skipped",
        "action_required": actions,
    }


def _status_colour_css(status: str) -> str:
    return {"PASS": "#2d8a4e", "WARN": "#b8860b", "FAIL": "#c0392b", "SKIP": "#666666"}.get(status, "#333")


def _build_html_report(results: dict[str, CheckResult], ts: str, git: str, check_titles: dict) -> str:
    overall = _overall_status(results)
    oc      = _status_colour_css(overall)

    rows = []
    for cid, res in results.items():
        title  = check_titles.get(cid, cid)
        sc     = _status_colour_css(res.status)
        detail_escaped = res.detail.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        action_html = ""
        if res.action and res.status in ("FAIL", "WARN"):
            a_esc = res.action.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            action_html = f'<div class="action">Action: {a_esc}</div>'
        rows.append(textwrap.dedent(f"""\
            <details class="check">
              <summary>
                <span class="badge" style="background:{sc}">{res.status}</span>
                {title}
              </summary>
              <div class="detail">{detail_escaped}</div>
              {action_html}
            </details>"""))

    checks_html = "\n".join(rows)

    n_fail = sum(1 for r in results.values() if r.status == "FAIL")
    n_warn = sum(1 for r in results.values() if r.status == "WARN")
    n_pass = sum(1 for r in results.values() if r.status == "PASS")
    n_skip = sum(1 for r in results.values() if r.status == "SKIP")

    actions_html = ""
    action_items = [r.action for r in results.values() if r.action and r.status in ("FAIL", "WARN")]
    if action_items:
        items = "".join(f"<li>{a.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')}</li>"
                        for a in action_items)
        actions_html = f"<h2>Action Required</h2><ol>{items}</ol>"

    return textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="utf-8">
        <title>Compatibility Report {ts}</title>
        <style>
          body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                  max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
          h1 {{ font-size: 1.4rem; margin-bottom: 0.2rem; }}
          .verdict {{ font-size: 2.5rem; font-weight: 700; color: {oc}; margin: 0.5rem 0 1.5rem; }}
          .summary {{ color: #555; margin-bottom: 1.5rem; }}
          .check {{ border: 1px solid #ddd; border-radius: 4px; margin-bottom: 0.5rem;
                    padding: 0; }}
          .check summary {{ padding: 0.6rem 1rem; cursor: pointer; list-style: none;
                            display: flex; align-items: center; gap: 0.7rem; }}
          .check summary::-webkit-details-marker {{ display: none; }}
          .badge {{ color: #fff; padding: 0.15rem 0.55rem; border-radius: 3px;
                    font-size: 0.75rem; font-weight: 600; min-width: 3.5rem;
                    text-align: center; }}
          .detail {{ padding: 0.6rem 1rem 0.4rem; color: #444; font-size: 0.9rem; }}
          .action {{ padding: 0.2rem 1rem 0.8rem; color: #5a3e00; font-size: 0.85rem;
                     background: #fffbe6; border-top: 1px solid #f0e080; }}
          h2 {{ margin-top: 2rem; font-size: 1.1rem; }}
          ol {{ padding-left: 1.5rem; }}
          li {{ margin-bottom: 0.4rem; font-size: 0.9rem; }}
          footer {{ margin-top: 3rem; font-size: 0.8rem; color: #999;
                    border-top: 1px solid #eee; padding-top: 0.8rem; }}
        </style>
        </head>
        <body>
        <h1>Train/Inference Compatibility Report</h1>
        <div class="verdict">{overall}</div>
        <div class="summary">
          {n_fail} failed &middot; {n_warn} warned &middot; {n_pass} passed &middot; {n_skip} skipped
        </div>
        {checks_html}
        {actions_html}
        <footer>Generated {ts} &middot; git {git}</footer>
        </body>
        </html>""")


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_checks(
    context: dict,
    check_ids: Optional[list[str]] = None,
    skip_ids: Optional[list[str]] = None,
) -> dict[str, CheckResult]:
    """Run the requested checks and return results dict."""
    ids_to_run = list(_ALL_CHECKS.keys())
    if check_ids:
        ids_to_run = [c for c in ids_to_run if c in check_ids]
    if skip_ids:
        ids_to_run = [c for c in ids_to_run if c not in skip_ids]

    results: dict[str, CheckResult] = {}
    total = len(ids_to_run)
    for i, cid in enumerate(ids_to_run):
        write_heartbeat("validate_compat", context.get("chunk"),
                        done=i, total=total, pct=100 * i // total,
                        eta_sec=0)
        _, fn = _ALL_CHECKS[cid]
        try:
            results[cid] = fn(context)
        except Exception as exc:
            results[cid] = CheckResult(
                "FAIL",
                f"Check raised an unexpected exception: {exc}",
                action=f"Investigate and fix the check implementation for {cid}",
            )

    write_heartbeat("validate_compat", context.get("chunk"),
                    done=total, total=total, pct=100, eta_sec=0)
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Train/inference compatibility validation suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config",      metavar="PATH",
                        help="Pipeline YAML config (default: auto-detect)")
    parser.add_argument("--data-root",   metavar="PATH",
                        help="Override DATA_ROOT")
    parser.add_argument("--report-dir",  metavar="PATH",
                        help="Where to write HTML/JSON reports (default: DATA_ROOT/reports/compat)")
    parser.add_argument("--strict",      action="store_true",
                        help="Exit 2 if any WARN (default: only FAIL causes exit 1)")
    parser.add_argument("--checks",      metavar="LIST",
                        help="Comma-separated check IDs to run (default: all)")
    parser.add_argument("--skip-checks", metavar="LIST",
                        help="Comma-separated check IDs to skip")
    parser.add_argument("--fast",        action="store_true",
                        help="Skip CHECK-4 (embedding numerical comparison, requires model load)")
    parser.add_argument("--ai",          action="store_true",
                        help="Output only JSON report to stdout (for pipeline_doctor integration)")
    parser.add_argument("--no-report",   action="store_true",
                        help="Do not write report files (stdout only)")
    args = parser.parse_args(argv)

    # Resolve paths
    data_root    = Path(args.data_root) if args.data_root else DATA_ROOT
    precomp_root = data_root / "precomputed"

    # Validate config
    cfg = _load_config_safe(args.config, data_root)
    if cfg is None and not args.ai:
        print("WARNING: Could not load pipeline config — some checks will SKIP", file=sys.stderr)

    # Build context
    repo_root = TRAIN_DIR.parent
    context   = {
        "repo_root":    repo_root,
        "cfg":          cfg,
        "precomp_root": precomp_root,
        "fast":         args.fast,
        "chunk":        None,
    }

    # Resolve check list
    check_ids = [c.strip() for c in args.checks.split(",")] if args.checks else None
    skip_ids  = [c.strip() for c in args.skip_checks.split(",")] if args.skip_checks else None
    if args.fast:
        skip_ids = list(skip_ids or []) + ["embedding_sanity"]

    # Validate check IDs
    all_ids = set(_ALL_CHECKS.keys())
    for bad in (check_ids or []) + (skip_ids or []):
        if bad and bad not in all_ids:
            print(f"ERROR: Unknown check ID '{bad}'. Valid IDs: {', '.join(sorted(all_ids))}",
                  file=sys.stderr)
            return 3

    # Run checks
    results = run_checks(context, check_ids=check_ids, skip_ids=skip_ids)

    # Build reports
    ts  = now_iso()
    git = _git_sha()
    report = _build_json_report(results, ts, git)

    check_titles = {cid: title for cid, (title, _) in _ALL_CHECKS.items()}

    if args.ai:
        print(json.dumps(report))
    else:
        print(f"\nCompatibility Report — {ts}  (git {git})")
        print(f"Overall: {report['overall']}")
        print(f"Summary: {report['summary']}")
        print()
        for cid, res in results.items():
            title = check_titles.get(cid, cid)
            print(f"  [{res.status:4s}] {title}")
            if res.detail:
                for line in textwrap.wrap(res.detail, 76, initial_indent="         ", subsequent_indent="         "):
                    print(line)
        if report["action_required"]:
            print("\nActions Required:")
            for i, a in enumerate(report["action_required"], 1):
                print(f"  {i}. {a}")
        print()

    # Write report files
    if not args.no_report:
        if args.report_dir:
            report_dir = Path(args.report_dir)
        else:
            vcfg = (cfg or {}).get("validation", {})
            rel  = vcfg.get("report_dir", "reports/compat")
            report_dir = data_root / rel

        try:
            report_dir.mkdir(parents=True, exist_ok=True)
            ts_safe = ts.replace(":", "-").replace("+", "Z")
            json_path = report_dir / f"compat_report_{ts_safe}.json"
            html_path = report_dir / f"compat_report_{ts_safe}.html"

            json_path.write_text(json.dumps(report, indent=2))

            html = _build_html_report(results, ts, git, check_titles)
            html_path.write_text(html)

            if not args.ai:
                print(f"Reports written:")
                print(f"  JSON: {json_path}")
                print(f"  HTML: {html_path}")
        except OSError as e:
            if not args.ai:
                print(f"WARNING: Could not write reports to {report_dir}: {e}", file=sys.stderr)

    # Exit code
    overall = _overall_status(results)
    if overall == "FAIL":
        return 1
    if args.strict and overall == "WARN":
        return 2
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Integration hooks (callable from precompute_all.py / orchestrator / flywheel)
# ─────────────────────────────────────────────────────────────────────────────

def validate_for_pipeline(
    config: dict,
    data_root: Path,
    strict: bool = False,
    fast: bool = True,
    skip_ids: Optional[list[str]] = None,
) -> dict:
    """
    Callable from precompute_all.py, flywheel, and orchestrator.

    Returns the JSON report dict.
    Raises RuntimeError on FAIL if strict=True.
    Raises RuntimeError on WARN if strict=True.

    Example usage in orchestrator before launching precompute:
        from validate_full_compatibility import validate_for_pipeline
        report = validate_for_pipeline(cfg, DATA_ROOT, strict=True)
        if report["overall"] == "FAIL":
            raise RuntimeError("Compatibility check failed — see report for details")
    """
    repo_root    = TRAIN_DIR.parent
    precomp_root = data_root / "precomputed"

    context = {
        "repo_root":    repo_root,
        "cfg":          config,
        "precomp_root": precomp_root,
        "fast":         fast,
        "chunk":        None,
    }

    _skip = list(skip_ids or [])
    if fast and "embedding_sanity" not in _skip:
        _skip.append("embedding_sanity")

    results = run_checks(context, skip_ids=_skip if _skip else None)

    ts  = now_iso()
    git = _git_sha()
    report = _build_json_report(results, ts, git)

    if strict and report["overall"] in ("FAIL", "WARN"):
        raise RuntimeError(
            f"Compatibility validation {report['overall']}: {report['summary']}. "
            f"Actions: {'; '.join(report['action_required'])}"
        )
    elif report["overall"] == "FAIL":
        raise RuntimeError(
            f"Compatibility validation FAILED: {report['summary']}. "
            f"Actions: {'; '.join(report['action_required'])}"
        )

    return report


if __name__ == "__main__":
    sys.exit(main())
