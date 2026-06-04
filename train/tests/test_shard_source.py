"""
train/tests/test_shard_source.py — provenance.json -> canonical source label.

Guards the fix for the shard-source data gap: shard_index.db left multi-source
shards "unknown" and shard_scores.db stamped everything "journeydb". Both now
derive the label from provenance.json via shard_source.py. Pure unit tests:
JSON in, label out, no DB.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SCRIPTS = Path(__file__).parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ss = _load("shard_source")


def _prov(tmp, stem, types):
    p = tmp / f"{stem}.provenance.json"
    p.write_text(json.dumps({
        "shard_id": f"shard-{stem}",
        "sources": [{"type": t, "tgz": i} for i, t in enumerate(types)],
    }))
    return p


class TestNormalize:
    def test_jdb_maps_to_journeydb(self):
        assert ss.normalize_source_type("jdb") == "journeydb"

    def test_passthrough_known(self):
        assert ss.normalize_source_type("coyo") == "coyo"
        assert ss.normalize_source_type("LAION") == "laion"   # case-insensitive

    def test_blank_is_unknown(self):
        assert ss.normalize_source_type("") == "unknown"
        assert ss.normalize_source_type(None) == "unknown"


class TestSourceFromProvenance:
    def test_pure_jdb_is_journeydb(self, tmp_path):
        p = _prov(tmp_path, "000000", ["jdb", "jdb", "jdb"])
        assert ss.source_from_provenance(p) == "journeydb"

    def test_pure_coyo(self, tmp_path):
        p = _prov(tmp_path, "000868", ["coyo"])
        assert ss.source_from_provenance(p) == "coyo"

    def test_mixed_is_sorted_combined_tag(self, tmp_path):
        # 7x jdb + 1x wikiart (the real 000745 shape) -> deterministic combined tag
        p = _prov(tmp_path, "000745", ["jdb"] * 7 + ["wikiart"])
        assert ss.source_from_provenance(p) == "journeydb+wikiart"

    def test_three_way_mix_sorted(self, tmp_path):
        p = _prov(tmp_path, "000700", ["wikiart", "coyo", "laion", "coyo"])
        assert ss.source_from_provenance(p) == "coyo+laion+wikiart"

    def test_order_independent(self, tmp_path):
        a = _prov(tmp_path, "a", ["wikiart", "jdb"])
        b = _prov(tmp_path, "b", ["jdb", "wikiart"])
        assert ss.source_from_provenance(a) == ss.source_from_provenance(b) == "journeydb+wikiart"

    def test_missing_file_is_none(self, tmp_path):
        assert ss.source_from_provenance(tmp_path / "nope.provenance.json") is None

    def test_no_sources_is_none(self, tmp_path):
        p = tmp_path / "x.provenance.json"
        p.write_text(json.dumps({"shard_id": "x", "sources": []}))
        assert ss.source_from_provenance(p) is None

    def test_unknown_types_dropped(self, tmp_path):
        # unknown/blank types are dropped; a real type still resolves
        p = _prov(tmp_path, "y", ["coyo", "", None])
        assert ss.source_from_provenance(p) == "coyo"

    def test_malformed_json_is_none(self, tmp_path):
        p = tmp_path / "z.provenance.json"
        p.write_text("{not json")
        assert ss.source_from_provenance(p) is None


class TestSourceForTar:
    def test_resolves_sibling_provenance(self, tmp_path):
        _prov(tmp_path, "000123", ["coyo", "laion"])
        assert ss.source_for_tar(tmp_path / "000123.tar") == "coyo+laion"

    def test_missing_sibling_is_none(self, tmp_path):
        assert ss.source_for_tar(tmp_path / "000999.tar") is None
