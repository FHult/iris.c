"""
Flask API tests for server.py.

These tests exercise the web server endpoints using Flask's built-in test
client. No model, no binary, and no running server process are required.

Run:
    cd web && python -m pytest tests/ -v
    # or from project root:
    python -m pytest web/tests/ -v
"""

import base64
import json
import sys
import time
from pathlib import Path

import pytest

# Add the web directory to the path so we can import server.py
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def client(tmp_path):
    """Set up a Flask test client with isolated in-memory state."""
    import server as srv

    # Point I/O dirs to a temp directory so tests are isolated
    srv.OUTPUT_DIR = tmp_path
    srv.THUMB_DIR = tmp_path / "thumbs"
    srv.HISTORY_FILE = tmp_path / "history.json"

    # No real model or binary
    srv.model_dir_path = tmp_path / "model"
    srv.flux_server = None

    # Clear shared state between tests
    srv.history.clear()
    srv.history_by_id.clear()
    with srv.job_queue_lock:
        srv.job_queue.clear()
    with srv.jobs_lock:
        srv.jobs.clear()

    srv.app.config["TESTING"] = True
    with srv.app.test_client() as c:
        yield c

    # Clean up after test
    srv.history.clear()
    srv.history_by_id.clear()
    with srv.job_queue_lock:
        srv.job_queue.clear()
    with srv.jobs_lock:
        srv.jobs.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_history(n=1):
    """Insert n fake completed jobs into the server history."""
    import server as srv

    jobs = []
    for i in range(n):
        job_id = f"test{i:06d}"
        job = srv.Job(job_id, prompt=f"test prompt {i}",
                      width=512, height=512, steps=4)
        job.seed = 1000 + i
        job.status = "complete"
        job.output_path = srv.OUTPUT_DIR / f"{job_id}.png"
        srv.history.insert(0, job)
        srv.history_by_id[job_id] = job
        jobs.append(job)
    return jobs


def _tiny_png_b64():
    """Return a 1x1 white PNG as a base64 data-URI."""
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe"
        b"\xdc\xccY\xe7\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


# ---------------------------------------------------------------------------
# /history  GET
# ---------------------------------------------------------------------------

class TestHistoryGet:
    def test_empty_history(self, client):
        r = client.get("/history")
        assert r.status_code == 200
        assert r.get_json() == []

    def test_history_with_items(self, client):
        _seed_history(3)
        r = client.get("/history")
        data = r.get_json()
        assert r.status_code == 200
        assert len(data) == 3

    def test_history_item_shape(self, client):
        _seed_history(1)
        r = client.get("/history")
        item = r.get_json()[0]
        required_fields = {"id", "prompt", "width", "height", "steps",
                           "seed", "image_url", "thumb_url", "favorited"}
        assert required_fields.issubset(set(item.keys()))

    def test_history_image_url_format(self, client):
        _seed_history(1)
        r = client.get("/history")
        item = r.get_json()[0]
        assert item["image_url"].startswith("/image/")
        assert item["thumb_url"].startswith("/thumb/")


# ---------------------------------------------------------------------------
# /history/<id>  DELETE
# ---------------------------------------------------------------------------

class TestHistoryDelete:
    def test_delete_existing(self, client):
        jobs = _seed_history(1)
        job_id = jobs[0].id
        r = client.delete(f"/history/{job_id}")
        assert r.status_code == 200
        assert r.get_json()["status"] == "deleted"
        # Item is gone from history
        r2 = client.get("/history")
        assert r2.get_json() == []

    def test_delete_nonexistent(self, client):
        r = client.delete("/history/doesnotexist")
        assert r.status_code == 404

    def test_delete_reduces_count(self, client):
        jobs = _seed_history(3)
        client.delete(f"/history/{jobs[0].id}")
        r = client.get("/history")
        assert len(r.get_json()) == 2


# ---------------------------------------------------------------------------
# /history/<id>/favorite  POST
# ---------------------------------------------------------------------------

class TestFavoriteToggle:
    def test_toggle_on(self, client):
        jobs = _seed_history(1)
        job_id = jobs[0].id
        r = client.post(f"/history/{job_id}/favorite")
        assert r.status_code == 200
        assert r.get_json()["favorited"] is True

    def test_toggle_off(self, client):
        jobs = _seed_history(1)
        job_id = jobs[0].id
        client.post(f"/history/{job_id}/favorite")  # on
        r = client.post(f"/history/{job_id}/favorite")  # off
        assert r.get_json()["favorited"] is False

    def test_toggle_nonexistent(self, client):
        r = client.post("/history/doesnotexist/favorite")
        assert r.status_code == 404

    def test_reflected_in_history(self, client):
        jobs = _seed_history(1)
        job_id = jobs[0].id
        client.post(f"/history/{job_id}/favorite")
        r = client.get("/history")
        item = r.get_json()[0]
        assert item["favorited"] is True


# ---------------------------------------------------------------------------
# /generate  POST — validation only (no binary, no model)
# ---------------------------------------------------------------------------

class TestGenerateValidation:
    def test_missing_prompt(self, client):
        r = client.post("/generate", json={"width": 512, "height": 512, "steps": 4})
        assert r.status_code == 400
        assert "prompt" in r.get_json().get("error", "").lower()

    def test_empty_prompt(self, client):
        r = client.post("/generate",
                        json={"prompt": "   ", "width": 512, "height": 512, "steps": 4})
        assert r.status_code == 400

    def test_width_not_divisible_by_16(self, client):
        r = client.post("/generate",
                        json={"prompt": "test", "width": 513, "height": 512, "steps": 4})
        assert r.status_code == 400

    def test_height_not_divisible_by_16(self, client):
        r = client.post("/generate",
                        json={"prompt": "test", "width": 512, "height": 513, "steps": 4})
        assert r.status_code == 400

    def test_width_too_small(self, client):
        r = client.post("/generate",
                        json={"prompt": "test", "width": 32, "height": 512, "steps": 4})
        assert r.status_code == 400

    def test_width_too_large(self, client):
        r = client.post("/generate",
                        json={"prompt": "test", "width": 2048, "height": 512, "steps": 4})
        assert r.status_code == 400

    def test_height_too_small(self, client):
        r = client.post("/generate",
                        json={"prompt": "test", "width": 512, "height": 32, "steps": 4})
        assert r.status_code == 400

    def test_height_too_large(self, client):
        r = client.post("/generate",
                        json={"prompt": "test", "width": 512, "height": 2048, "steps": 4})
        assert r.status_code == 400

    def test_steps_too_small(self, client):
        r = client.post("/generate",
                        json={"prompt": "test", "width": 512, "height": 512, "steps": 0})
        assert r.status_code == 400

    def test_steps_too_large(self, client):
        r = client.post("/generate",
                        json={"prompt": "test", "width": 512, "height": 512, "steps": 999})
        assert r.status_code == 400

    def test_invalid_reference_image(self, client):
        r = client.post("/generate", json={
            "prompt": "test", "width": 512, "height": 512, "steps": 4,
            "reference_images": ["data:image/png;base64,notvalidbase64!!!!!"],
        })
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# /queue/reorder  POST
# ---------------------------------------------------------------------------

class TestQueueReorder:
    def test_empty_queue_reorder(self, client):
        r = client.post("/queue/reorder", json={"order": []})
        assert r.status_code == 200
        assert r.get_json()["status"] == "ok"

    def test_reorder_with_jobs(self, client):
        import server as srv
        # Manually add fake queued entries
        job_a = srv.Job("aaa", prompt="a", width=512, height=512, steps=4)
        job_b = srv.Job("bbb", prompt="b", width=512, height=512, steps=4)
        with srv.job_queue_lock:
            srv.job_queue.append({"job": job_a})
            srv.job_queue.append({"job": job_b})

        # Reverse the order
        r = client.post("/queue/reorder", json={"order": ["bbb", "aaa"]})
        assert r.status_code == 200
        with srv.job_queue_lock:
            assert srv.job_queue[0]["job"].id == "bbb"
            assert srv.job_queue[1]["job"].id == "aaa"

    def test_reorder_unknown_ids_ignored(self, client):
        """Unknown IDs in the order list should be ignored gracefully."""
        r = client.post("/queue/reorder", json={"order": ["nonexistent", "also_fake"]})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# /available-loras  GET
# ---------------------------------------------------------------------------

class TestAvailableLoras:
    def test_returns_structure(self, client):
        r = client.get("/available-loras")
        assert r.status_code == 200
        data = r.get_json()
        assert "loras" in data
        assert "curated" in data

    def test_no_loras_when_dir_missing(self, client):
        r = client.get("/available-loras")
        data = r.get_json()
        assert data["loras"] == []

    def test_curated_list_populated(self, client):
        r = client.get("/available-loras")
        curated = r.get_json()["curated"]
        assert len(curated) > 0

    def test_curated_item_shape(self, client):
        r = client.get("/available-loras")
        for item in r.get_json()["curated"]:
            assert "id" in item
            assert "name" in item
            assert "filename" in item
            assert "downloaded" in item
            assert isinstance(item["downloaded"], bool)

    def test_downloaded_flag_reflects_filesystem(self, client, tmp_path):
        import server as srv
        # Create a fake LoRA file in the expected location
        lora_dir = tmp_path / "model" / "loras"
        lora_dir.mkdir(parents=True)
        first_curated = srv.CURATED_LORAS[0]
        (lora_dir / first_curated["filename"]).write_bytes(b"\x00" * 100)
        srv.model_dir_path = tmp_path / "model"

        r = client.get("/available-loras")
        data = r.get_json()

        # The local loras list should contain the file
        assert any(item["filename"] == first_curated["filename"]
                   for item in data["loras"])
        # The curated entry should be marked downloaded
        curated_entry = next(c for c in data["curated"]
                             if c["id"] == first_curated["id"])
        assert curated_entry["downloaded"] is True


# ---------------------------------------------------------------------------
# /server-status  GET
# ---------------------------------------------------------------------------

class TestServerStatus:
    def test_returns_structure(self, client):
        r = client.get("/server-status")
        assert r.status_code == 200
        data = r.get_json()
        assert "ready" in data
        assert "busy" in data
        assert "queue_length" in data

    def test_no_server_not_ready(self, client):
        r = client.get("/server-status")
        data = r.get_json()
        assert data["ready"] is False
        assert data["busy"] is False
        assert data["queue_length"] == 0


# ---------------------------------------------------------------------------
# /model-info  GET
# ---------------------------------------------------------------------------

class TestModelInfo:
    def test_returns_503_when_not_ready(self, client):
        r = client.get("/model-info")
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# /active-jobs  GET
# ---------------------------------------------------------------------------

class TestActiveJobs:
    def test_returns_structure(self, client):
        r = client.get("/active-jobs")
        assert r.status_code == 200
        data = r.get_json()
        assert "running" in data
        assert "queued" in data

    def test_empty_when_no_jobs(self, client):
        r = client.get("/active-jobs")
        data = r.get_json()
        assert data["running"] is None
        assert data["queued"] == []


# ---------------------------------------------------------------------------
# /style-presets  GET
# ---------------------------------------------------------------------------

class TestStylePresets:
    def test_returns_presets(self, client):
        r = client.get("/style-presets")
        assert r.status_code == 200
        data = r.get_json()
        assert "presets" in data
        assert "step_guidance" in data
        assert len(data["presets"]) > 0

    def test_preset_item_shape(self, client):
        r = client.get("/style-presets")
        for _key, preset in r.get_json()["presets"].items():
            assert "name" in preset
            assert "description" in preset
            assert "recommended_steps" in preset


# ---------------------------------------------------------------------------
# /enhance-prompt  POST
# ---------------------------------------------------------------------------

class TestEnhancePrompt:
    def test_missing_prompt(self, client):
        r = client.post("/enhance-prompt", json={})
        assert r.status_code == 400

    def test_empty_prompt(self, client):
        r = client.post("/enhance-prompt", json={"prompt": ""})
        assert r.status_code == 400

    def test_returns_enhanced(self, client):
        r = client.post("/enhance-prompt", json={"prompt": "a cat"})
        assert r.status_code == 200
        data = r.get_json()
        assert "enhanced" in data
        assert "original" in data
        assert data["original"] == "a cat"

    def test_style_preset_applied(self, client):
        import server as srv
        if not srv.STYLE_PRESETS:
            pytest.skip("No style presets defined")
        key = next(iter(srv.STYLE_PRESETS))
        preset = srv.STYLE_PRESETS[key]
        r = client.post("/enhance-prompt",
                        json={"prompt": "a cat", "style": key})
        assert r.status_code == 200
        data = r.get_json()
        assert preset["suffix"] in data["enhanced"]
        assert data["recommended_steps"] == preset["recommended_steps"]

    def test_no_auto_enhance(self, client):
        r = client.post("/enhance-prompt",
                        json={"prompt": "a cat", "auto_enhance": False})
        data = r.get_json()
        assert data["enhanced"] == "a cat"


# ---------------------------------------------------------------------------
# /save-crop  POST
# ---------------------------------------------------------------------------

class TestSaveCrop:
    def test_missing_image(self, client):
        r = client.post("/save-crop", json={"prompt": "test"})
        assert r.status_code == 400

    def test_saves_and_returns_url(self, client, tmp_path):
        import server as srv
        srv.OUTPUT_DIR = tmp_path
        r = client.post("/save-crop", json={
            "image": _tiny_png_b64(),
            "prompt": "cropped cat",
            "width": 1,
            "height": 1,
            "seed": 42,
        })
        assert r.status_code == 200
        data = r.get_json()
        assert "job_id" in data
        assert data["image_url"].startswith("/image/")

    def test_appears_in_history(self, client, tmp_path):
        import server as srv
        srv.OUTPUT_DIR = tmp_path
        r = client.post("/save-crop", json={
            "image": _tiny_png_b64(),
            "prompt": "cropped cat",
            "width": 1, "height": 1,
        })
        job_id = r.get_json()["job_id"]
        history_r = client.get("/history")
        ids = [item["id"] for item in history_r.get_json()]
        assert job_id in ids


# ---------------------------------------------------------------------------
# /status/<job_id>  GET
# ---------------------------------------------------------------------------

class TestJobStatus:
    def test_unknown_job(self, client):
        r = client.get("/status/nonexistent")
        assert r.status_code == 404

    def test_active_job_found(self, client):
        import server as srv
        job = srv.Job("activejob1", prompt="test", width=512, height=512, steps=4)
        job.status = "complete"
        with srv.jobs_lock:
            srv.jobs["activejob1"] = job

        r = client.get("/status/activejob1")
        assert r.status_code == 200
        data = r.get_json()
        assert data["status"] == "complete"
        assert "image_url" in data
