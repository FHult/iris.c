"""train/tests/test_config_schema.py — GROK-T-7 config validator.

Pure/hermetic: validates in-memory dicts and one tempfile. No GPU, no live config.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import config_schema as cs


def _sev(issues, severity):
    return [i for i in issues if i["severity"] == severity]


def test_clean_pipeline_config_passes():
    cfg = {"scale": "large", "chunks": 4, "training": {"steps": {"large": {1: 1000}}},
           "model": {}, "anomaly": {}}
    assert cs.validate_config(cfg) == []


def test_unknown_top_level_warns():
    issues = cs.validate_config({"trainng": {}, "scale": "small"})  # typo'd "training"
    w = _sev(issues, "WARNING")
    assert any(i["path"] == "trainng" for i in w)


def test_must_be_mapping_error():
    issues = cs.validate_config({"training": "oops_a_string"})
    e = _sev(issues, "ERROR")
    assert any(i["path"] == "training" for i in e)


def test_non_dict_root_error():
    issues = cs.validate_config(["not", "a", "mapping"])
    assert len(issues) == 1 and issues[0]["severity"] == "ERROR"


def test_flywheel_health_nested_typo_warns():
    # valid section, typo'd inner key that would silently use the default.
    cfg = {"flywheel_health": {"stall_precompute_secs": 3600, "max_campagn_days": 7}}
    w = _sev(cs.validate_config(cfg), "WARNING")
    assert any(i["path"] == "flywheel_health.max_campagn_days" for i in w)


def test_flywheel_health_known_keys_clean():
    cfg = {"flywheel_health": {"stall_precompute_secs": 3600, "stall_train_secs": 600,
                               "logs_max_gb": 5, "max_campaign_days": 7,
                               "consecutive_fail_warn": 2, "consecutive_fail_crit": 3}}
    assert cs.validate_config(cfg) == []


def test_flywheel_missing_required_warns():
    issues = cs.validate_config({"flywheel": {"steps_per_iteration": 1000}})
    paths = {i["path"] for i in issues}
    assert "flywheel.max_iterations" in paths and "flywheel.n_shards" in paths


def test_flywheel_complete_clean():
    cfg = {"flywheel": {"max_iterations": 15, "n_shards": 40, "steps_per_iteration": 1000}}
    assert cs.validate_config(cfg) == []


def test_validate_file_roundtrip(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("scale: small\nbogus_key: 1\n")
    issues = cs.validate_file(str(p))
    assert any(i["path"] == "bogus_key" for i in issues)


def test_validate_file_bad_yaml(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("key: [unclosed\n")
    issues = cs.validate_file(str(p))
    assert issues and issues[0]["severity"] == "ERROR"


def test_format_issues_ok():
    assert "passed validation" in cs.format_issues([], "x.yaml")
