"""
train/tests/test_cache_manager.py — precompute cache versioning logic.

Covers the pure version-hash functions (GROK-TEST-6) that decide which
precompute cache dir an encoder's output lands in. These invariants are what
keep the flywheel's per-iter caches correct: a config change that affects an
encoder's output MUST change its version hash (else stale latents are reused),
and a cosmetic change (dict key order, flux_model parent dir) MUST NOT.

Pure: no GPU, no I/O, no DB — just dict → hash math.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cm = _load("cache_manager")


# ---------------------------------------------------------------------------
# version_hash
# ---------------------------------------------------------------------------

class TestVersionHash:
    def test_format(self):
        h = cm.version_hash({"a": 1}, "deadbeef")
        assert h.startswith("v_")
        assert len(h) == 2 + 6                  # "v_" + 6 hex chars
        assert all(c in "0123456789abcdef" for c in h[2:])

    def test_deterministic(self):
        cfg = {"flux_model": "x", "image_size": 512}
        assert cm.version_hash(cfg, "abc12345") == cm.version_hash(cfg, "abc12345")

    def test_stable_under_key_order(self):
        # json.dumps(sort_keys=True) → insertion order must not matter.
        a = cm.version_hash({"image_size": 512, "flux_model": "x"}, "sha00000")
        b = cm.version_hash({"flux_model": "x", "image_size": 512}, "sha00000")
        assert a == b

    def test_config_change_changes_hash(self):
        base = cm.version_hash({"image_size": 512}, "sha00000")
        assert cm.version_hash({"image_size": 768}, "sha00000") != base

    def test_git_sha_changes_hash(self):
        cfg = {"image_size": 512}
        assert cm.version_hash(cfg, "11111111") != cm.version_hash(cfg, "22222222")

    def test_only_first_8_sha_chars_matter(self):
        # version_hash uses git_sha[:8]; chars beyond 8 must not affect the result.
        cfg = {"image_size": 512}
        assert cm.version_hash(cfg, "abcdef12") == cm.version_hash(cfg, "abcdef12ZZZZ")


# ---------------------------------------------------------------------------
# encoder_config_subset
# ---------------------------------------------------------------------------

class TestEncoderConfigSubset:
    def test_vae_strips_flux_model_dir(self):
        # Same model basename via different parent dirs → identical subset → same
        # version. The model is identified by name, not path.
        a = cm.encoder_config_subset("vae", {"model": {"flux_model": "/a/b/flux-klein-model"}})
        b = cm.encoder_config_subset("vae", {"model": {"flux_model": "flux-klein-model"}})
        assert a == b
        assert a["flux_model"] == "flux-klein-model"

    def test_vae_image_size_versions_separately(self):
        # 512 vs 768 must produce different subsets → different version dirs
        # (this is the Stage 2/3 high-res cache separation mechanism).
        s512 = cm.encoder_config_subset("vae", {"data": {"image_size": 512}})
        s768 = cm.encoder_config_subset("vae", {"data": {"image_size": 768}})
        assert s512 != s768
        assert cm.version_hash(s512, "x") != cm.version_hash(s768, "x")

    def test_vae_defaults(self):
        s = cm.encoder_config_subset("vae", {})
        assert s["image_size"] == 512
        assert s["flux_model"] == "flux-klein-4b"

    def test_qwen3_subset_includes_layers_and_think_tags(self):
        s = cm.encoder_config_subset("qwen3", {})
        assert s["layers"] == [8, 17, 26]
        assert s["think_tags"] is True
        assert s["qwen3_model"] == "Qwen/Qwen3-4B"

    def test_siglip_subset_is_fixed(self):
        s = cm.encoder_config_subset("siglip", {"data": {"image_size": 999}})
        # SigLIP always resizes to 384 regardless of training image_size.
        assert s["image_size"] == 384
        assert "siglip" in s["siglip_model"]

    def test_unknown_encoder_is_empty(self):
        assert cm.encoder_config_subset("bogus", {"model": {}}) == {}

    def test_end_to_end_vae_version_stability(self):
        # Two configs that differ only in flux_model parent dir + key order must
        # land in the SAME version dir; image_size change must NOT.
        cfg_a = {"model": {"flux_model": "/vol/x/flux-klein-model"},
                 "data": {"image_size": 512}}
        cfg_b = {"data": {"image_size": 512},
                 "model": {"flux_model": "flux-klein-model"}}
        va = cm.version_hash(cm.encoder_config_subset("vae", cfg_a), "sha00000")
        vb = cm.version_hash(cm.encoder_config_subset("vae", cfg_b), "sha00000")
        assert va == vb
