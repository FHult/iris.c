"""
train/tests/test_collect_metrics.py — flywheel metric parsing from trainer logs.

Pins the contract the whole campaign stands on (code review M1): the regexes that
turn trainer stdout into the metrics dict, and the rule that a held-out VAL
cond_gap SUPERSEDES the in-training (train-batch) gap for everything downstream
(champion selection, shard attribution), with the train value preserved under
cond_gap_train. Hermetic — writes a synthetic log to tmp.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from flywheel_lib import collect_metrics_from_log

LOG_BODY = """\
step      990/1,000  loss 0.4422 (avg 0.5964)  lr 1.05e-05  0.09 steps/s  ETA 0h01m
  loss_cond=0.4960  loss_null=0.4963  gap=+0.0003 (+0.1%)  [n=333/167]
  loss_ref: self=0.4959 [n=216]  cross=0.4961 [n=117]  gap=+0.0002
  ip_scale: mean=0.7820  double=0.0000  single=0.9775  range=[0.0000, 0.9799]
step      1,000/1,000  loss 0.7391 (avg 0.6096)  lr 1.08e-06  0.11 steps/s  ETA 0h00m

Training complete. EMA weights: /tmp/x/best.safetensors
"""

VAL_LINE = "VAL loss_cond=0.4012 loss_null=0.4391 cond_gap=+0.0379 [n=58] (held-out)\n"


def _parse(tmp_path, text):
    p = tmp_path / "trainer.log"
    p.write_text(text)
    return collect_metrics_from_log(p)


def test_train_batch_metrics_without_val(tmp_path):
    m = _parse(tmp_path, LOG_BODY)
    assert m["step"] == 1000
    assert m["loss_smooth"] == 0.6096
    assert m["cond_gap"] == 0.0003          # train-batch gap is all there is
    assert m["ref_gap"] == 0.0002
    assert m["ip_scale_single"] == 0.9775
    assert "val_cond_gap" not in m


def test_val_cond_gap_supersedes_train_gap(tmp_path):
    m = _parse(tmp_path, LOG_BODY + VAL_LINE)
    assert m["val_cond_gap"] == 0.0379
    assert m["val_loss_cond"] == 0.4012
    assert m["val_loss_null"] == 0.4391
    assert m["val_n_pairs"] == 58
    # the held-out value is what downstream consumers see as cond_gap
    assert m["cond_gap"] == 0.0379
    # the train-batch value survives for diagnostics
    assert m["cond_gap_train"] == 0.0003


def test_val_negative_gap_parses(tmp_path):
    m = _parse(tmp_path, "VAL loss_cond=0.5000 loss_null=0.4800 cond_gap=-0.0200 [n=10] (held-out)\n")
    assert m["cond_gap"] == -0.02
    assert "cond_gap_train" not in m        # no train gap in this log


def test_val_unavailable_line_is_ignored(tmp_path):
    m = _parse(tmp_path, LOG_BODY + "VAL held-out cond_gap unavailable (no val set or no SigLIP pairs)\n")
    assert m["cond_gap"] == 0.0003          # falls back to the train-batch gap
    assert "val_cond_gap" not in m
