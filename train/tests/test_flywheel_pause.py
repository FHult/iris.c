"""
train/tests/test_flywheel_pause.py — `pause --free-gpu` control-flow (orchestrator).

Pins: a free-gpu pause kills the registered GPU subprocess and (when allowed) raises
_RestartIteration on resume so the iteration re-runs; allow_restart=False just waits and
returns; a plain pause neither kills nor raises. Hermetic — fake proc, no real GPU/tmux,
sleep stubbed to simulate resume.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import orchestrator as o


class FakeProc:
    def __init__(self):
        self.terminated = self.killed = False
        self._alive = True

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self.terminated = True
        self._alive = False

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True
        self._alive = False


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Redirect the control file, stub tmux/log/sleep; sleep removes the control file
    (simulates `resume`) so wait loops terminate immediately."""
    ctl = tmp_path / "flywheel_control.json"
    monkeypatch.setattr(o, "FLYWHEEL_CONTROL_FILE", ctl)
    monkeypatch.setattr(o, "tmux_window_exists", lambda *_a, **_k: False)
    monkeypatch.setattr(o, "log_orch", lambda *_a, **_k: None)
    monkeypatch.setattr(o.time, "sleep", lambda *_a: ctl.unlink(missing_ok=True))
    o._FLYWHEEL_GPU_PROC[0] = None
    yield ctl
    o._FLYWHEEL_GPU_PROC[0] = None


def _write(ctl, action, **kw):
    ctl.write_text(json.dumps({"action": action, **kw}))


class TestFreeGpuPause:
    def test_kills_and_restarts(self, harness):
        proc = FakeProc()
        o._FLYWHEEL_GPU_PROC[0] = proc
        _write(harness, "pause", free_gpu=True)
        with pytest.raises(o._RestartIteration):
            o._check_flywheel_control("t", None)          # allow_restart default True
        assert proc.terminated, "free-gpu pause must kill the GPU subprocess"

    def test_allow_restart_false_waits_no_raise(self, harness):
        proc = FakeProc()
        o._FLYWHEEL_GPU_PROC[0] = proc
        _write(harness, "pause", free_gpu=True)
        o._check_flywheel_control("t", None, allow_restart=False)  # returns, no raise
        assert proc.terminated, "GPU still freed even when not restarting"

    def test_plain_pause_no_kill_no_raise(self, harness):
        proc = FakeProc()
        o._FLYWHEEL_GPU_PROC[0] = proc
        _write(harness, "pause")                          # no free_gpu
        o._check_flywheel_control("t", None)              # returns normally
        assert not proc.terminated, "cooperative pause must NOT kill the GPU subprocess"

    def test_no_control_file_is_noop(self, harness):
        o._check_flywheel_control("t", None)              # file absent → returns immediately

    def test_interruptible_wait_raises_on_free_gpu(self, harness):
        proc = FakeProc()
        _write(harness, "pause", free_gpu=True)
        with pytest.raises(o._RestartIteration):
            o._interruptible_proc_wait(proc, "t", None, poll=0)
        assert proc.terminated
        assert o._FLYWHEEL_GPU_PROC[0] is None, "holder cleared on exit"
