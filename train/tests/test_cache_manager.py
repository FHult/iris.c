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

    def test_git_sha_does_not_change_hash(self):
        # PRECOMP-3: the cache key must be bound to encoder identity, NOT the repo
        # git SHA. An unrelated commit (different SHA, same config) must resolve to
        # the SAME version dir — otherwise every commit re-precomputes the pool.
        cfg = {"image_size": 512}
        assert cm.version_hash(cfg, "11111111") == cm.version_hash(cfg, "22222222")

    def test_git_sha_fully_ignored(self):
        # git_sha is accepted for call-site compatibility but never affects the key.
        cfg = {"image_size": 512}
        assert cm.version_hash(cfg, "") == cm.version_hash(cfg, "deadbeefcafe")

    def test_code_version_changes_hash(self):
        # A per-encoder code-version bump (encoding semantics changed) MUST change
        # the version, so stale latents are not silently reused.
        base = cm.version_hash({"qwen3_model": "Qwen/Qwen3-4B", "code_version": "1"})
        bumped = cm.version_hash({"qwen3_model": "Qwen/Qwen3-4B", "code_version": "2"})
        assert base != bumped


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

    def test_subset_carries_code_version(self):
        for enc in ("qwen3", "vae", "siglip"):
            s = cm.encoder_config_subset(enc, {})
            assert s["code_version"] == cm.ENCODER_CODE_VERSION[enc]

    def test_unrelated_commit_keeps_version(self):
        # The real PRECOMP-3 scenario: same encoder config, two different repo
        # SHAs (a commit happened between iters) → identical version dir.
        cfg = {"model": {"qwen3_model": "Qwen/Qwen3-4B"}}
        s = cm.encoder_config_subset("qwen3", cfg)
        assert cm.version_hash(s, "31e647a4") == cm.version_hash(s, "5d36078c")


# ---------------------------------------------------------------------------
# consolidate (PRECOMP-3 migration)
# ---------------------------------------------------------------------------

import json as _json


def _make_version(enc_dir: Path, ver: str, shard_ids, recs_per_shard, config,
                  complete=True, completed_at="2026-01-01T00:00:00+00:00"):
    """Create a fake version dir with npz files + manifest. shard_ids: list of int."""
    vdir = enc_dir / ver
    vdir.mkdir(parents=True, exist_ok=True)
    n = 0
    for sid in shard_ids:
        for r in range(recs_per_shard):
            (vdir / f"{sid:06d}_{r:04d}.npz").write_bytes(b"x")
            n += 1
    manifest = {"version": ver, "encoder": enc_dir.name, "config": config,
                "complete": complete, "record_count": n,
                "shard_count": len(list(shard_ids)), "completed_at": completed_at,
                "created_at": completed_at}
    (vdir / "manifest.json").write_text(_json.dumps(manifest))
    return vdir


class TestConsolidate:
    def _qwen_cfg(self, code_version=None):
        c = {"qwen3_model": "Qwen/Qwen3-4B", "layers": [8, 17, 26], "think_tags": True}
        if code_version is not None:
            c["code_version"] = code_version
        return c

    def test_dry_run_reports_union_no_disk_change(self, tmp_path):
        enc = tmp_path / "qwen3"
        # iter10: shards 0,1,2 ; iter11: shards 1,2,3  → union 0,1,2,3 (overlap 1,2)
        _make_version(enc, "v_iter10", [0, 1, 2], 2, self._qwen_cfg(),
                      completed_at="2026-06-02T00:00:00+00:00")
        _make_version(enc, "v_iter11", [1, 2, 3], 2, self._qwen_cfg(),
                      completed_at="2026-06-03T00:00:00+00:00")
        r = cm.consolidate(tmp_path, "qwen3", apply=False)
        assert r["unique_shards"] == 4
        assert r["unique_npz"] == 8           # 4 shards * 2 recs, dedup by name
        assert set(r["sources_merged"]) == {"v_iter10", "v_iter11"}
        assert not r["applied"]
        # nothing moved: both source dirs intact, no target dir, current absent
        assert (enc / "v_iter10").is_dir() and (enc / "v_iter11").is_dir()
        assert not (enc / "current").is_symlink()

    def test_apply_unions_and_repoints_current(self, tmp_path):
        enc = tmp_path / "qwen3"
        _make_version(enc, "v_iter10", [0, 1, 2], 2, self._qwen_cfg(),
                      completed_at="2026-06-02T00:00:00+00:00")
        _make_version(enc, "v_iter11", [1, 2, 3], 2, self._qwen_cfg(),
                      completed_at="2026-06-03T00:00:00+00:00")
        r = cm.consolidate(tmp_path, "qwen3", apply=True)
        target = r["target"]
        tdir = enc / target
        # union present in target, deduped
        npz = sorted(f.name for f in tdir.glob("*.npz"))
        assert len(npz) == 8
        shard_prefixes = {f.split("_")[0] for f in npz}
        assert shard_prefixes == {"000000", "000001", "000002", "000003"}
        # current repointed at target, manifest complete + correct counts
        assert _os_readlink(enc / "current") == target
        m = _json.loads((tdir / "manifest.json").read_text())
        assert m["complete"] is True and m["record_count"] == 8 and m["shard_count"] == 4
        # redundant source dirs removed (unless they were the target)
        for src in ("v_iter10", "v_iter11"):
            if src != target:
                assert not (enc / src).exists()

    def test_skips_different_identity(self, tmp_path):
        enc = tmp_path / "vae"
        _make_version(enc, "v_512", [0, 1], 2,
                      {"flux_model": "flux-klein-4b", "image_size": 512},
                      completed_at="2026-06-02T00:00:00+00:00")
        _make_version(enc, "v_768", [2, 3], 2,
                      {"flux_model": "flux-klein-4b", "image_size": 768},
                      completed_at="2026-06-03T00:00:00+00:00")
        r = cm.consolidate(tmp_path, "vae", apply=False)
        # newest (768) is canonical; 512 has a different identity → skipped
        assert r["unique_shards"] == 2
        assert any(s["version"] == "v_512" for s in r["skipped"])

    def test_ignores_empty_stub_versions(self, tmp_path):
        enc = tmp_path / "qwen3"
        _make_version(enc, "v_stub", [], 0, self._qwen_cfg())        # 0 records
        _make_version(enc, "v_real", [0, 1], 3, self._qwen_cfg())
        r = cm.consolidate(tmp_path, "qwen3", apply=False)
        assert "v_stub" not in r["sources_merged"]
        assert r["unique_shards"] == 2

    def test_no_complete_versions_reports_error(self, tmp_path):
        enc = tmp_path / "qwen3"
        _make_version(enc, "v_part", [0], 2, self._qwen_cfg(), complete=False)
        r = cm.consolidate(tmp_path, "qwen3", apply=False)
        assert "error" in r and not r["applied"]


def _os_readlink(link: Path) -> str:
    import os
    return os.readlink(str(link))


# ---------------------------------------------------------------------------
# current-symlink resolution — regression guards for the footguns this session
# surfaced. The original flywheel cache-clobber bug AND the stale-hot-dir
# confusion both traced to one root cause: `current` pointing at an empty
# (0-record) version dir. These pin the behavior so it can't drift silently.
# ---------------------------------------------------------------------------

class TestCurrentSymlinkResolution:
    def test_list_versions_skips_current_symlink(self, tmp_path):
        # `current` is a symlink to a real version dir; list_versions must not
        # surface it as a phantom version (it did, double-counting records and —
        # worse — becoming a fake "complete" source that consolidate would try to
        # rmtree *through the symlink*).
        enc = tmp_path / "qwen3"
        _make_version(enc, "v_real", [0, 1], 2, {"x": 1})
        cm._atomic_symlink(enc / "current", "v_real")
        vers = cm.PrecomputeCache.list_versions(tmp_path, "qwen3")
        assert [v["version"] for v in vers] == ["v_real"]      # no "current" entry
        assert vers[0]["current"] is True

    def test_effective_dir_returns_populated_current(self, tmp_path):
        enc = tmp_path / "qwen3"
        _make_version(enc, "v_real", [0, 1], 2, {"x": 1})
        cm._atomic_symlink(enc / "current", "v_real")
        assert cm.PrecomputeCache.effective_dir(tmp_path, "qwen3") == enc / "v_real"

    def test_effective_dir_empty_current_falls_through_to_complete(self, tmp_path):
        # The hardened behavior: when `current` resolves to an existing but EMPTY
        # (0-record) version dir, effective_dir must SKIP it and fall through to
        # the newest complete, non-empty version. This is the fix for the
        # flywheel cache-clobber bug — an empty `current` no longer silently
        # yields 0 shards when a populated version sits right next to it.
        enc = tmp_path / "qwen3"
        empty = enc / "v_empty"
        empty.mkdir(parents=True)
        (empty / "manifest.json").write_text(
            _json.dumps({"version": "v_empty", "complete": True, "record_count": 0}))
        _make_version(enc, "v_full", [0, 1], 2, {"x": 1})
        cm._atomic_symlink(enc / "current", "v_empty")
        eff = cm.PrecomputeCache.effective_dir(tmp_path, "qwen3")
        assert eff == enc / "v_full"                           # fell through, not empty
        assert list(eff.glob("*.npz"))

    def test_effective_dir_empty_current_no_complete_returns_none(self, tmp_path):
        # Empty current and nothing else usable → None (honest "no cache"),
        # rather than a path to an empty dir.
        enc = tmp_path / "qwen3"
        empty = enc / "v_empty"
        empty.mkdir(parents=True)
        (empty / "manifest.json").write_text(
            _json.dumps({"version": "v_empty", "complete": True, "record_count": 0}))
        cm._atomic_symlink(enc / "current", "v_empty")
        assert cm.PrecomputeCache.effective_dir(tmp_path, "qwen3") is None

    def test_effective_dir_picks_newest_complete_when_current_empty(self, tmp_path):
        # Two complete versions, empty current → newest (by completed_at) wins.
        enc = tmp_path / "qwen3"
        empty = enc / "v_empty"; empty.mkdir(parents=True)
        (empty / "manifest.json").write_text(
            _json.dumps({"version": "v_empty", "complete": True, "record_count": 0}))
        _make_version(enc, "v_old", [0], 2, {"x": 1},
                      completed_at="2026-06-01T00:00:00+00:00")
        _make_version(enc, "v_new", [1], 2, {"x": 1},
                      completed_at="2026-06-03T00:00:00+00:00")
        cm._atomic_symlink(enc / "current", "v_empty")
        assert cm.PrecomputeCache.effective_dir(tmp_path, "qwen3") == enc / "v_new"

    def test_current_dir_none_when_dangling(self, tmp_path):
        # A truly dangling symlink (target removed) → current_dir returns None and
        # effective_dir falls through (here: to nothing, since no flat npz).
        enc = tmp_path / "qwen3"
        enc.mkdir(parents=True)
        cm._atomic_symlink(enc / "current", "v_gone")
        assert cm.PrecomputeCache.current_dir(tmp_path, "qwen3") is None
        assert cm.PrecomputeCache.effective_dir(tmp_path, "qwen3") is None

    def test_consolidate_does_not_consume_or_delete_current(self, tmp_path):
        # consolidate must not treat the `current` symlink as a source version,
        # and must leave a working `current` (repointed) after apply.
        enc = tmp_path / "qwen3"
        _make_version(enc, "v_a", [0, 1], 2,
                      {"qwen3_model": "Q", "code_version": "1"})
        cm._atomic_symlink(enc / "current", "v_a")
        r = cm.consolidate(tmp_path, "qwen3", apply=True)
        assert "current" not in r["sources_merged"]
        assert (enc / "current").is_symlink()
        assert cm.PrecomputeCache.current_dir(tmp_path, "qwen3") is not None
