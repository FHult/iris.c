"""
train/tests/test_ablation_report.py — ablation HTML report rendering (ABL-4).

The report is a large str.format() template; a single unbalanced brace silently
breaks rendering (KeyError / ValueError at format time) or leaves an unformatted
{placeholder} in the output. It had no test. These render synthetic results and
assert the template fully formats and the ABL-4 Pareto scatter + hover tooltip
are present.

Pure: no GPU, no DB, no real trials — just dict → HTML string.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from ablation_harness import _render_html


def _result(i, combo, rg, cg, fl, sc, pareto):
    return {
        "id": i, "combo_id": combo, "ref_gap": rg, "cond_gap": cg,
        "final_loss": fl, "score": sc, "is_pareto": pareto,
        "params": {"lr": 1e-4, "style_loss_weight": 0.07},
        "verdict": "OK" if sc is not None else "CRASH",
        "elapsed_secs": 1200, "ts": "2026-06-03T10:00",
    }


def _render(results):
    return _render_html(results, "m", steps=500, ts="2026-06-03",
                        total_elapsed=3600, run_dir_name="run_x",
                        objective={"ref_gap": 1.0, "cond_gap": 1.0})


def _no_unformatted(html):
    # A surviving {placeholder} means a brace-escaping bug in the template.
    for token in ("{pareto_data}", "{show_pareto}", "{js_series}",
                  "{trend_data}", "{best_config_yaml}"):
        assert token not in html, f"unformatted placeholder: {token}"


class TestAblationReport:
    def test_renders_with_pareto_and_hover(self):
        results = [
            _result(1, "lr1e-4_style0.07", 0.10, 0.25, 0.42, 0.30, 1),
            _result(2, "lr2e-4_style0.05", 0.05, 0.30, 0.40, 0.28, 1),
            _result(3, "lr1e-4_style0.00", 0.02, 0.10, 0.55, 0.12, 0),
        ]
        html = _render(results)
        _no_unformatted(html)
        # ABL-4 elements present.
        assert "paretoTip" in html                 # tooltip div
        assert "cv.onmousemove" in html            # hover handler
        assert "cv.onmouseleave" in html
        assert "drawPareto" in html
        assert "hover any point" in html           # updated chart label
        assert 'show_pareto="true"' not in html    # value injected, not literal
        # Each trial's data is embedded for the scatter.
        assert "lr1e-4_style0.07" in html

    def test_few_points_disables_pareto_but_still_renders(self):
        # <3 scored points → show_pareto false; must still format cleanly.
        results = [_result(1, "only_one", 0.1, 0.2, 0.4, 0.3, 1)]
        html = _render(results)
        _no_unformatted(html)
        assert "<html" in html

    def test_crashed_trial_does_not_break_render(self):
        results = [
            _result(1, "ok_a", 0.10, 0.25, 0.42, 0.30, 1),
            _result(2, "ok_b", 0.05, 0.30, 0.40, 0.28, 1),
            _result(3, "ok_c", 0.02, 0.10, 0.55, 0.12, 0),
            _result(4, "crashed", None, None, None, None, 0),   # null metrics
        ]
        html = _render(results)
        _no_unformatted(html)
        assert "crashed" in html
