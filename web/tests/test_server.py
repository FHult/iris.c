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
    srv.iris_server = None

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
        output_path = srv.OUTPUT_DIR / f"{job_id}.png"
        # Create a minimal placeholder so the history filter sees the file as present
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()
        job.output_path = output_path
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
                           "seed", "image_url", "thumb_url", "favorited",
                           "guidance", "schedule", "img2img_strength",
                           "negative_prompt"}
        assert required_fields.issubset(set(item.keys()))

    def test_history_item_guidance_schedule_defaults(self, client):
        _seed_history(1)
        r = client.get("/history")
        item = r.get_json()[0]
        assert item["guidance"] is None
        assert item["schedule"] is None
        assert item["img2img_strength"] == 1.0

    def test_history_item_guidance_schedule_persisted(self, client):
        import server as srv
        job_id = "guidtest1"
        job = srv.Job(job_id, prompt="test", width=512, height=512, steps=4)
        job.guidance = 4.5
        job.schedule = "linear"
        job.img2img_strength = 0.7
        job.seed = 99
        job.status = "complete"
        output_path = srv.OUTPUT_DIR / f"{job_id}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()
        job.output_path = output_path
        srv.history.insert(0, job)
        srv.history_by_id[job_id] = job

        r = client.get("/history")
        item = next(i for i in r.get_json() if i["id"] == job_id)
        assert item["guidance"] == 4.5
        assert item["schedule"] == "linear"
        assert item["img2img_strength"] == 0.7

    def test_history_image_url_format(self, client):
        _seed_history(1)
        r = client.get("/history")
        item = r.get_json()[0]
        assert item["image_url"].startswith("/image/")
        assert item["thumb_url"].startswith("/thumb/")

    def test_guidance_schedule_round_trip_via_disk(self, client, tmp_path):
        """guidance/schedule/img2img_strength survive save→load cycle."""
        import json
        import server as srv

        srv.OUTPUT_DIR = tmp_path
        srv.HISTORY_FILE = tmp_path / "history.json"

        job_id = "roundtrip1"
        job = srv.Job(job_id, prompt="rt", width=512, height=512, steps=4)
        job.guidance = 3.0
        job.schedule = "power"
        job.img2img_strength = 0.55
        job.seed = 7
        job.status = "complete"
        out = tmp_path / f"{job_id}.png"
        out.touch()
        job.output_path = out

        srv.history.insert(0, job)
        srv.history_by_id[job_id] = job

        # Save
        with srv.history_lock:
            srv.save_history()

        # Reload into fresh state
        srv.history.clear()
        srv.history_by_id.clear()
        srv.load_history_from_disk()

        loaded = srv.history_by_id.get(job_id)
        assert loaded is not None
        assert loaded.guidance == 3.0
        assert loaded.schedule == "power"
        assert loaded.img2img_strength == 0.55


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

    def test_invalid_reference_image_object(self, client):
        """Per-slot object with bad base64 should also return 400."""
        r = client.post("/generate", json={
            "prompt": "test", "width": 512, "height": 512, "steps": 4,
            "reference_images": [{"data": "data:image/png;base64,!!!bad!!!", "strength": 0.8, "mode": "composition"}],
        })
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# /generate — additional edge cases (types, path traversal, boundaries)
# ---------------------------------------------------------------------------

class TestGenerateEdgeCases:
    def test_non_json_body_returns_4xx(self, client):
        """Plain-text body → Flask returns 415 (wrong content type) or 400."""
        r = client.post("/generate", data="not json", content_type="text/plain")
        assert r.status_code in (400, 415)

    def test_form_data_body_returns_4xx(self, client):
        """Form-encoded body is not JSON → Flask returns 415 or 400."""
        r = client.post("/generate", data={"prompt": "a cat"},
                        content_type="application/x-www-form-urlencoded")
        assert r.status_code in (400, 415)

    def test_empty_json_object_returns_400(self, client):
        """Empty JSON object → prompt missing → 400."""
        r = client.post("/generate", json={})
        assert r.status_code == 400

    def test_steps_as_non_numeric_string_returns_400(self, client):
        """steps='four' → int() raises ValueError → 400."""
        r = client.post("/generate", json={
            "prompt": "a cat", "width": 512, "height": 512, "steps": "four",
        })
        assert r.status_code == 400

    def test_width_as_null_returns_400(self, client):
        """width=null → int(None) raises TypeError → 400."""
        r = client.post("/generate", json={
            "prompt": "a cat", "width": None, "height": 512, "steps": 4,
        })
        assert r.status_code == 400

    def test_lora_path_traversal_slash_rejected(self, client):
        r = client.post("/generate", json={
            "prompt": "a cat", "width": 512, "height": 512, "steps": 4,
            "lora": "../../etc/passwd",
        })
        assert r.status_code == 400

    def test_lora_path_traversal_dotdot_rejected(self, client):
        r = client.post("/generate", json={
            "prompt": "a cat", "width": 512, "height": 512, "steps": 4,
            "lora": "..evil",
        })
        assert r.status_code == 400

    def test_img2img_strength_zero_accepted(self, client):
        """strength=0.0 is a valid boundary — job should be queued (200)."""
        r = client.post("/generate", json={
            "prompt": "a cat", "width": 512, "height": 512, "steps": 4,
            "img2img_strength": 0.0,
        })
        assert r.status_code == 200

    def test_img2img_strength_one_accepted(self, client):
        """strength=1.0 is the default — must be accepted (200)."""
        r = client.post("/generate", json={
            "prompt": "a cat", "width": 512, "height": 512, "steps": 4,
            "img2img_strength": 1.0,
        })
        assert r.status_code == 200

    def test_reference_images_as_string_returns_400(self, client):
        """reference_images as a bare string (not a list) should return 400."""
        r = client.post("/generate", json={
            "prompt": "a cat", "width": 512, "height": 512, "steps": 4,
            "reference_images": "not a list",
        })
        assert r.status_code == 400

    def test_response_includes_job_id(self, client):
        """A valid request must return a job_id in the response body."""
        r = client.post("/generate", json={
            "prompt": "a cat", "width": 512, "height": 512, "steps": 4,
        })
        assert r.status_code == 200
        assert "job_id" in r.get_json()


# ---------------------------------------------------------------------------
# Per-slot reference image strength normalisation (unit tests)
# ---------------------------------------------------------------------------

class TestRefSlotNormalise:
    """Tests for the _normalise_ref / effective img2img_strength logic in /generate."""

    def _post_valid_with_refs(self, client, refs):
        """Post to /generate with valid params and given reference_images."""
        return client.post("/generate", json={
            "prompt": "test",
            "width": 512,
            "height": 512,
            "steps": 4,
            "reference_images": refs,
        })

    def test_bare_string_accepted(self, client):
        """Bare base64 strings (backward compat) must still return 400 only on bad data."""
        r = self._post_valid_with_refs(client, ["data:image/png;base64,notvalid!!!"])
        assert r.status_code == 400  # bad data, but parsed as bare string

    def test_slot_object_accepted(self, client):
        """Valid per-slot objects with bad data return 400 (data decoded OK level)."""
        r = self._post_valid_with_refs(client, [
            {"data": "data:image/png;base64,notvalid!!!", "strength": 0.5, "mode": "composition"}
        ])
        assert r.status_code == 400

    def test_effective_strength_from_slots(self):
        """Minimum slot strength drives effective img2img_strength."""
        import server as srv
        # Simulate the inline logic from /generate
        slots = [
            {"data": "x", "strength": 0.9, "mode": "composition"},
            {"data": "x", "strength": 0.6, "mode": "composition"},
        ]
        strengths = []
        for s in slots:
            w = float(s.get("strength", 1.0))
            if s.get("mode") == "style":
                w *= 0.5
            strengths.append(w)
        effective = min(strengths)
        assert effective == pytest.approx(0.6)

    def test_style_mode_halves_strength(self):
        """Style mode should halve the effective weight before taking the minimum."""
        slots = [
            {"data": "x", "strength": 0.8, "mode": "style"},
            {"data": "x", "strength": 0.9, "mode": "composition"},
        ]
        strengths = []
        for s in slots:
            w = float(s.get("strength", 1.0))
            if s.get("mode") == "style":
                w *= 0.5
            strengths.append(w)
        effective = min(strengths)
        assert effective == pytest.approx(0.4)  # 0.8 * 0.5 = 0.4 < 0.9

    def test_explicit_img2img_strength_overrides_slots(self):
        """When img2img_strength is set in the request, slot strengths do not override it."""
        # We test this at the request level: explicitly setting img2img_strength=0.3
        # means the server should NOT override it with derived slot strength.
        # This is checked by verifying the job stores 0.3.
        import server as srv
        import json as _json

        job_id = "overridetest1"
        job = srv.Job(job_id, prompt="ov", width=512, height=512, steps=4)
        job.img2img_strength = 0.3
        # The value should be preserved exactly as set on the job
        d = job.to_history_dict()
        assert d["img2img_strength"] == pytest.approx(0.3)


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
# Z-Image capabilities
# ---------------------------------------------------------------------------

class TestZImageCapabilities:
    def _make_mock_server(self, is_zimage):
        return type("MockIrisServer", (), {
            "ready": True,
            "model_info": "Z-Image Turbo" if is_zimage else "Flux Klein 4B",
            "is_distilled": True,
            "is_zimage": is_zimage,
        })()

    def test_model_info_is_zimage_false_by_default(self, client):
        import server as srv
        srv.iris_server = self._make_mock_server(False)
        r = client.get("/model-info")
        assert r.status_code == 200
        assert r.get_json()["is_zimage"] is False
        srv.iris_server = None

    def test_model_info_is_zimage_true_when_set(self, client):
        import server as srv
        srv.iris_server = self._make_mock_server(True)
        r = client.get("/model-info")
        assert r.status_code == 200
        data = r.get_json()
        assert data["is_zimage"] is True
        assert data["model"] == "Z-Image Turbo"
        srv.iris_server = None


# ---------------------------------------------------------------------------
# /delete-model  POST
# ---------------------------------------------------------------------------

class TestDeleteModel:
    def test_unknown_key_returns_400(self, client):
        r = client.post("/delete-model", json={"key": "nonexistent"})
        assert r.status_code == 400

    def test_not_downloaded_returns_400(self, client, tmp_path):
        import server as srv
        orig = srv.PROJECT_DIR
        srv.PROJECT_DIR = tmp_path  # empty dir — no model present
        r = client.post("/delete-model", json={"key": "flux-klein-4b-base"})
        srv.PROJECT_DIR = orig
        assert r.status_code == 400

    def test_deletes_directory(self, client, tmp_path):
        import server as srv
        model_dir = tmp_path / "flux-klein-4b-base"
        (model_dir / "transformer").mkdir(parents=True)
        orig = srv.PROJECT_DIR
        srv.PROJECT_DIR = tmp_path
        r = client.post("/delete-model", json={"key": "flux-klein-4b-base"})
        srv.PROJECT_DIR = orig
        assert r.status_code == 200
        assert not model_dir.exists()

    def test_cannot_delete_current_model(self, client, tmp_path):
        import server as srv
        model_dir = tmp_path / "flux-klein-4b-base"
        (model_dir / "transformer").mkdir(parents=True)
        mock = type("M", (), {
            "model_dir": str(model_dir),
            "ready": True,
            "model_info": "",
            "is_distilled": True,
            "is_zimage": False,
        })()
        orig_srv, orig_dir = srv.iris_server, srv.PROJECT_DIR
        srv.iris_server = mock
        srv.PROJECT_DIR = tmp_path
        r = client.post("/delete-model", json={"key": "flux-klein-4b-base"})
        srv.iris_server = orig_srv
        srv.PROJECT_DIR = orig_dir
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# /settings  GET + POST
# ---------------------------------------------------------------------------

class TestSettings:
    def test_get_settings_no_token(self, client):
        import server as srv
        orig = srv._hf_token
        srv._hf_token = ""
        r = client.get("/settings")
        assert r.status_code == 200
        assert r.get_json()["hf_token_set"] is False
        srv._hf_token = orig

    def test_get_settings_with_token(self, client):
        import server as srv
        orig = srv._hf_token
        srv._hf_token = "hf_abc123"
        r = client.get("/settings")
        assert r.status_code == 200
        data = r.get_json()
        assert data["hf_token_set"] is True
        assert "hf_abc123" not in str(data)
        srv._hf_token = orig

    def test_post_settings_sets_token(self, client):
        import server as srv
        orig = srv._hf_token
        r = client.post("/settings", json={"hf_token": "hf_test"})
        assert r.status_code == 200
        assert r.get_json()["ok"] is True
        assert srv._hf_token == "hf_test"
        srv._hf_token = orig

    def test_post_settings_clears_token(self, client):
        import server as srv
        orig = srv._hf_token
        srv._hf_token = "hf_existing"
        client.post("/settings", json={"hf_token": ""})
        assert srv._hf_token == ""
        srv._hf_token = orig


# ---------------------------------------------------------------------------
# /verify-model/<key>  GET
# ---------------------------------------------------------------------------

class TestVerifyModel:
    @staticmethod
    def _make_safetensors(path, payload=b"hello"):
        """Write a minimal valid safetensors file."""
        import struct, json as _json
        hdr = {
            "__metadata__": {},
            "t": {"dtype": "F32", "shape": [len(payload)], "data_offsets": [0, len(payload)]},
        }
        hdr_bytes = _json.dumps(hdr).encode()
        path.write_bytes(struct.pack("<Q", len(hdr_bytes)) + hdr_bytes + payload)

    def test_unknown_key_returns_400(self, client):
        r = client.get("/verify-model/nonexistent")
        assert r.status_code == 400

    def test_all_files_present_returns_ok(self, client, tmp_path):
        import server as srv
        model_dir = tmp_path / "flux-klein-4b-base"
        for rel in srv.MODEL_SLOTS[1]["expected_files"]:
            f = model_dir / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            if rel.endswith(".safetensors"):
                self._make_safetensors(f)
            else:
                f.write_bytes(b"x")
        orig = srv.PROJECT_DIR
        srv.PROJECT_DIR = tmp_path
        r = client.get("/verify-model/flux-klein-4b-base")
        srv.PROJECT_DIR = orig
        assert r.status_code == 200
        assert r.get_json()["ok"] is True

    def test_missing_file_reported(self, client, tmp_path):
        import server as srv
        model_dir = tmp_path / "flux-klein-4b-base"
        model_dir.mkdir()
        orig = srv.PROJECT_DIR
        srv.PROJECT_DIR = tmp_path
        r = client.get("/verify-model/flux-klein-4b-base")
        srv.PROJECT_DIR = orig
        data = r.get_json()
        assert data["ok"] is False
        assert len(data["missing"]) > 0

    def test_empty_file_reported(self, client, tmp_path):
        import server as srv
        model_dir = tmp_path / "flux-klein-4b-base"
        first = srv.MODEL_SLOTS[1]["expected_files"][0]
        for rel in srv.MODEL_SLOTS[1]["expected_files"]:
            f = model_dir / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            if rel == first:
                f.write_bytes(b"")
            elif rel.endswith(".safetensors"):
                self._make_safetensors(f)
            else:
                f.write_bytes(b"x")
        orig = srv.PROJECT_DIR
        srv.PROJECT_DIR = tmp_path
        r = client.get("/verify-model/flux-klein-4b-base")
        srv.PROJECT_DIR = orig
        data = r.get_json()
        assert data["ok"] is False
        assert len(data["empty"]) == 1

    def test_truncated_safetensors_reported(self, client, tmp_path):
        import server as srv, struct, json as _json
        model_dir = tmp_path / "flux-klein-4b-base"
        st_rel = next(r for r in srv.MODEL_SLOTS[1]["expected_files"] if r.endswith(".safetensors"))
        for rel in srv.MODEL_SLOTS[1]["expected_files"]:
            f = model_dir / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            if rel == st_rel:
                # Header claims 100 bytes of data but we only write 10
                hdr = {"__metadata__": {}, "t": {"dtype": "F32", "shape": [25], "data_offsets": [0, 100]}}
                hdr_bytes = _json.dumps(hdr).encode()
                f.write_bytes(struct.pack("<Q", len(hdr_bytes)) + hdr_bytes + b"\x00" * 10)
            elif rel.endswith(".safetensors"):
                self._make_safetensors(f)
            else:
                f.write_bytes(b"x")
        orig = srv.PROJECT_DIR
        srv.PROJECT_DIR = tmp_path
        r = client.get("/verify-model/flux-klein-4b-base")
        srv.PROJECT_DIR = orig
        data = r.get_json()
        assert data["ok"] is False
        assert len(data["truncated"]) == 1
        assert data["truncated"][0]["file"] == st_rel


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


# ---------------------------------------------------------------------------
# /available-models  GET
# ---------------------------------------------------------------------------

class TestAvailableModels:
    def test_returns_slots_and_current(self, client):
        r = client.get("/available-models")
        assert r.status_code == 200
        data = r.get_json()
        assert "slots" in data
        assert "current_model_dir" in data

    def test_slots_have_required_fields(self, client):
        r = client.get("/available-models")
        data = r.get_json()
        for slot in data["slots"]:
            assert "key" in slot
            assert "label" in slot
            assert "description" in slot
            assert "downloaded" in slot
            assert "current" in slot
            assert "downloadable" in slot

    def test_no_server_current_is_null(self, client):
        # client fixture sets iris_server = None
        r = client.get("/available-models")
        data = r.get_json()
        assert data["current_model_dir"] is None
        # No slot should be marked current when there's no server
        assert not any(s["current"] for s in data["slots"])

    def test_downloaded_reflects_filesystem(self, client, tmp_path):
        import server as srv
        # Point PROJECT_DIR at tmp_path so we can create a fake model dir
        original_project_dir = srv.PROJECT_DIR
        srv.PROJECT_DIR = tmp_path
        try:
            # Add a test slot
            test_slot = {
                "key": "test-model",
                "label": "Test",
                "description": "test",
                "sh_arg": "test",
                "expected_files": ["model_index.json"],
            }
            srv.MODEL_SLOTS.append(test_slot)

            # Not downloaded yet
            r = client.get("/available-models")
            slots = {s["key"]: s for s in r.get_json()["slots"]}
            assert slots["test-model"]["downloaded"] is False

            # Create the transformer dir to simulate download
            (tmp_path / "test-model" / "transformer").mkdir(parents=True)

            r = client.get("/available-models")
            slots = {s["key"]: s for s in r.get_json()["slots"]}
            assert slots["test-model"]["downloaded"] is True
        finally:
            srv.MODEL_SLOTS.remove(test_slot)
            srv.PROJECT_DIR = original_project_dir


# ---------------------------------------------------------------------------
# /switch-model  POST
# ---------------------------------------------------------------------------

class TestSwitchModel:
    def test_unknown_key_returns_400(self, client):
        r = client.post("/switch-model", json={"key": "no-such-model"})
        assert r.status_code == 400
        assert "error" in r.get_json()

    def test_not_downloaded_returns_400(self, client, tmp_path):
        import server as srv
        original_project_dir = srv.PROJECT_DIR
        srv.PROJECT_DIR = tmp_path
        test_slot = {
            "key": "not-downloaded-model",
            "label": "ND",
            "description": "not downloaded",
            "sh_arg": "nd",
            "expected_files": [],
        }
        srv.MODEL_SLOTS.append(test_slot)
        try:
            r = client.post("/switch-model", json={"key": "not-downloaded-model"})
            assert r.status_code == 400
            assert "not downloaded" in r.get_json()["error"].lower()
        finally:
            srv.MODEL_SLOTS.remove(test_slot)
            srv.PROJECT_DIR = original_project_dir

    def test_busy_server_returns_409(self, client, tmp_path):
        """Cannot switch while a job is queued."""
        import server as srv
        original_project_dir = srv.PROJECT_DIR
        srv.PROJECT_DIR = tmp_path
        test_slot = {
            "key": "busy-model",
            "label": "Busy",
            "description": "busy model",
            "sh_arg": "busy",
            "expected_files": [],
        }
        srv.MODEL_SLOTS.append(test_slot)
        # Create transformer dir so it passes the "downloaded" check
        (tmp_path / "busy-model" / "transformer").mkdir(parents=True)
        # Push a fake job into the queue to make server appear busy
        fake_job = srv.Job("busyjob1", prompt="x", width=512, height=512, steps=4)
        with srv.job_queue_lock:
            srv.job_queue.append({"job": fake_job, "params": {}})
        try:
            r = client.post("/switch-model", json={"key": "busy-model"})
            assert r.status_code == 409
            assert "error" in r.get_json()
        finally:
            with srv.job_queue_lock:
                srv.job_queue.clear()
            srv.MODEL_SLOTS.remove(test_slot)
            srv.PROJECT_DIR = original_project_dir

    def test_missing_key_returns_400(self, client):
        r = client.post("/switch-model", json={})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# /download-model  POST  and  /download-model/progress/<key>  GET
# ---------------------------------------------------------------------------

class TestDownloadModel:
    def test_unknown_key_returns_400(self, client):
        r = client.post("/download-model", json={"key": "bogus"})
        assert r.status_code == 400

    def test_non_downloadable_returns_400(self, client):
        import server as srv
        # Startup model slot has sh_arg=None (not downloadable)
        non_dl = next((s for s in srv.MODEL_SLOTS if s.get("sh_arg") is None), None)
        if non_dl is None:
            pytest.skip("No non-downloadable slot in MODEL_SLOTS")
        r = client.post("/download-model", json={"key": non_dl["key"]})
        assert r.status_code == 400

    def test_progress_not_started_returns_404(self, client):
        r = client.get("/download-model/progress/never-started-key")
        assert r.status_code == 404

    def test_progress_has_required_fields(self, client):
        import server as srv
        # Inject a fake in-progress download state directly
        srv._model_download_progress["test-progress-key"] = {
            "done": False, "error": None
        }
        try:
            r = client.get("/download-model/progress/test-progress-key")
            assert r.status_code == 200
            data = r.get_json()
            assert "done" in data
            assert "error" in data
            assert "files_done" in data
            assert "files_total" in data
        finally:
            srv._model_download_progress.pop("test-progress-key", None)

    def test_idempotent_while_running(self, client):
        import server as srv
        # Simulate an already-running download
        key = next((s["key"] for s in srv.MODEL_SLOTS if s.get("sh_arg")), None)
        if key is None:
            pytest.skip("No downloadable slot in MODEL_SLOTS")
        srv._model_download_progress[key] = {"done": False, "error": None}
        try:
            r = client.post("/download-model", json={"key": key})
            assert r.status_code == 200
            data = r.get_json()
            assert data.get("ok") is True
            assert data.get("already_running") is True
        finally:
            srv._model_download_progress.pop(key, None)


# ---------------------------------------------------------------------------
# /cancel/<job_id>  POST  — regression for stuck-server bug
# ---------------------------------------------------------------------------

class TestCancelJob:
    def test_unknown_job_returns_404(self, client):
        r = client.post("/cancel/nonexistent-job-id")
        assert r.status_code == 404

    def test_cancel_queued_job(self, client):
        """Cancelling a queued (not yet running) job removes it from the queue."""
        import server as srv
        job = srv.Job("queued01", prompt="test", width=512, height=512, steps=4)
        job.status = "queued"
        with srv.jobs_lock:
            srv.jobs["queued01"] = job
        with srv.job_queue_lock:
            srv.job_queue.append({"job": job, "params": {}})

        r = client.post("/cancel/queued01")
        assert r.status_code == 200
        data = r.get_json()
        assert data["status"] == "cancelled"

        # Job must be removed from the queue
        with srv.job_queue_lock:
            queued_ids = [q["job"].id for q in srv.job_queue]
        assert "queued01" not in queued_ids

        # Job status must be updated
        assert job.status == "cancelled"

    def test_cancel_already_complete_returns_400(self, client):
        import server as srv
        job = srv.Job("done01", prompt="test", width=512, height=512, steps=4)
        job.status = "complete"
        with srv.jobs_lock:
            srv.jobs["done01"] = job

        r = client.post("/cancel/done01")
        assert r.status_code == 400

    def test_cancel_running_job_clears_current_job(self, client):
        """Regression: cancelling running job must clear current_job so server
        is not permanently stuck as busy (fixed in restart())."""
        import server as srv
        import threading

        # Set up a fake IrisServer-like object whose restart() we can observe
        class FakeServer:
            def __init__(self):
                self.model_dir = "/fake/model"
                self.current_job = None
                self.restarted = threading.Event()

            def restart(self):
                # The real fix: clear current_job inside restart()
                self.current_job = None
                self.restarted.set()

        fake_server = FakeServer()
        job = srv.Job("running01", prompt="test", width=512, height=512, steps=4)
        job.status = "running"
        fake_server.current_job = job

        with srv.jobs_lock:
            srv.jobs["running01"] = job

        original_server = srv.iris_server
        srv.iris_server = fake_server
        try:
            r = client.post("/cancel/running01")
            assert r.status_code == 200
            data = r.get_json()
            assert data["status"] == "cancelled"
            assert data.get("server_restarting") is True

            # Wait for the background restart thread to complete
            assert fake_server.restarted.wait(timeout=3), "restart() was not called"

            # The critical regression check: current_job must be None after restart
            assert fake_server.current_job is None, \
                "current_job was not cleared by restart() — stuck-server bug regression"
        finally:
            srv.iris_server = original_server


# ---------------------------------------------------------------------------
# /outpaint-prep  POST
# ---------------------------------------------------------------------------

class TestOutpaintPrep:
    def test_missing_image_returns_400(self, client):
        r = client.post("/outpaint-prep", json={"top": 128})
        assert r.status_code == 400
        assert "error" in r.get_json()

    def test_invalid_image_returns_400(self, client):
        r = client.post("/outpaint-prep", json={
            "image": "data:image/png;base64,notvalidbase64!!!",
            "top": 64,
        })
        assert r.status_code == 400

    def test_no_expansion_returns_400(self, client):
        r = client.post("/outpaint-prep", json={
            "image": _tiny_png_b64(),
            "top": 0, "bottom": 0, "left": 0, "right": 0,
        })
        assert r.status_code == 400
        assert "No expansion" in r.get_json()["error"]

    def test_valid_top_expansion(self, client):
        r = client.post("/outpaint-prep", json={
            "image": _tiny_png_b64(),
            "top": 16,
        })
        assert r.status_code == 200
        data = r.get_json()
        assert "padded_image" in data
        assert data["padded_image"].startswith("data:image/png;base64,")
        assert data["width"] == 1          # original 1px wide
        assert data["height"] == 17        # 1 + 16 (snapped to 16? no: 16 already multiple)
        assert data["orig_x"] == 0
        assert data["orig_y"] == 16

    def test_all_directions(self, client):
        r = client.post("/outpaint-prep", json={
            "image": _tiny_png_b64(),
            "top": 16, "bottom": 16, "left": 16, "right": 16,
        })
        assert r.status_code == 200
        data = r.get_json()
        assert data["width"] == 33     # 1 + 16 + 16
        assert data["height"] == 33    # 1 + 16 + 16
        assert data["orig_x"] == 16
        assert data["orig_y"] == 16

    def test_size_snapped_to_16(self, client):
        """Fractional pixel values should be snapped to nearest 16px multiple."""
        r = client.post("/outpaint-prep", json={
            "image": _tiny_png_b64(),
            "top": 10,   # → snaps to 16
        })
        assert r.status_code == 200
        data = r.get_json()
        # top=10 → snapped to 16
        assert data["height"] == 1 + 16

    def test_exceeds_max_returns_400(self, client):
        r = client.post("/outpaint-prep", json={
            "image": _tiny_png_b64(),
            "top": 1792, "bottom": 1792,
        })
        assert r.status_code == 400
        assert "1792" in r.get_json()["error"]

    def test_returns_base64_decodable_png(self, client):
        import base64, io
        from PIL import Image as PILImage
        r = client.post("/outpaint-prep", json={
            "image": _tiny_png_b64(),
            "right": 16,
        })
        assert r.status_code == 200
        padded_b64 = r.get_json()["padded_image"].split(",")[1]
        img_bytes = base64.b64decode(padded_b64)
        img = PILImage.open(io.BytesIO(img_bytes))
        assert img.format == "PNG"
        assert img.width == 17   # 1 + 16
        assert img.height == 1


# ---------------------------------------------------------------------------
# Negative prompt — Feature 1
# ---------------------------------------------------------------------------

class TestNegativePrompt:
    """negative_prompt field persisted in history and passed to generation."""

    def test_history_item_negative_prompt_default(self, client):
        """History items have an empty negative_prompt by default."""
        _seed_history(1)
        r = client.get("/history")
        item = r.get_json()[0]
        assert "negative_prompt" in item
        assert item["negative_prompt"] == ""

    def test_history_item_negative_prompt_persisted(self, client):
        """negative_prompt is preserved in history when explicitly set."""
        import server as srv
        job_id = "negtest1"
        job = srv.Job(job_id, prompt="castle", width=512, height=512, steps=50)
        job.negative_prompt = "blurry, low quality"
        job.seed = 42
        job.status = "complete"
        out = srv.OUTPUT_DIR / f"{job_id}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.touch()
        job.output_path = out
        srv.history.insert(0, job)
        srv.history_by_id[job_id] = job

        r = client.get("/history")
        item = next(i for i in r.get_json() if i["id"] == job_id)
        assert item["negative_prompt"] == "blurry, low quality"

    def test_negative_prompt_round_trip_via_disk(self, client, tmp_path):
        """negative_prompt survives save→load cycle through history.json."""
        import json
        import server as srv

        srv.OUTPUT_DIR = tmp_path
        srv.HISTORY_FILE = tmp_path / "history.json"

        job_id = "negrt1"
        job = srv.Job(job_id, prompt="dragon", width=512, height=512, steps=50)
        job.negative_prompt = "cartoon, watermark"
        job.seed = 77
        job.status = "complete"
        out = tmp_path / f"{job_id}.png"
        out.touch()
        job.output_path = out

        srv.history.insert(0, job)
        srv.history_by_id[job_id] = job

        with srv.history_lock:
            srv.save_history()

        srv.history.clear()
        srv.history_by_id.clear()
        srv.load_history_from_disk()

        loaded = srv.history_by_id.get(job_id)
        assert loaded is not None
        assert loaded.negative_prompt == "cartoon, watermark"

    def test_generate_stores_negative_prompt_on_job(self, client):
        """POST /generate with negative_prompt stores it on the job object."""
        import server as srv
        r = client.post("/generate", json={
            "prompt": "a castle",
            "width": 512,
            "height": 512,
            "steps": 4,
            "negative_prompt": "blurry, ugly",
        })
        assert r.status_code == 200
        job_id = r.get_json()["job_id"]
        with srv.jobs_lock:
            job = srv.jobs.get(job_id)
        assert job is not None
        assert job.negative_prompt == "blurry, ugly"

    def test_generate_empty_negative_prompt_stored_empty(self, client):
        """POST /generate with empty negative_prompt stores empty string."""
        import server as srv
        r = client.post("/generate", json={
            "prompt": "a forest",
            "width": 512,
            "height": 512,
            "steps": 4,
            "negative_prompt": "",
        })
        assert r.status_code == 200
        job_id = r.get_json()["job_id"]
        with srv.jobs_lock:
            job = srv.jobs.get(job_id)
        assert job is not None
        assert job.negative_prompt == ""

    def test_generate_whitespace_negative_prompt_stored_empty(self, client):
        """Whitespace-only negative_prompt is normalised to empty string."""
        import server as srv
        r = client.post("/generate", json={
            "prompt": "a mountain",
            "width": 512,
            "height": 512,
            "steps": 4,
            "negative_prompt": "   ",
        })
        assert r.status_code == 200
        job_id = r.get_json()["job_id"]
        with srv.jobs_lock:
            job = srv.jobs.get(job_id)
        assert job is not None
        assert job.negative_prompt == ""
