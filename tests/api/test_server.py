"""Tests the HTTP layer (auth, job lifecycle, error handling) without paying for real
model loads or inference -- `run_analysis` is monkeypatched to a fast fake. Whether the
*pipeline itself* produces correct results is covered by the video_engine test suite;
this file is about whether the API around it behaves (rejects unauthenticated writes,
returns a job id, reaches "done", 404s on an unknown id), which is a different, real
risk from the pipeline's own correctness.
"""
from __future__ import annotations

import importlib
import io
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("THIRD_UMPIRE_API_KEY", "test-key")
    monkeypatch.setenv("THIRD_UMPIRE_MAX_UPLOAD_MB", "1")

    import api.server as server_module

    importlib.reload(server_module)  # picks up the monkeypatched env vars at import time

    def fake_run_analysis(video_path, source_label, match_id, innings, over, ball):
        return {
            "source_label": source_label,
            "match_id": match_id,
            "innings": innings,
            "over_number": over,
            "ball_number": ball,
            "frames_processed": 1,
            "people_tracked": 0,
            "pose_frames": 0,
            "shot_classification": "unknown",
            "event_notes": "fake result for a test",
            "detection_confidence": {},
            "timings": {},
        }

    monkeypatch.setattr(server_module, "run_analysis", fake_run_analysis)
    try:
        yield TestClient(server_module.app)
    finally:
        server_module._executor.shutdown(wait=False)


def _tiny_upload():
    return {"file": ("clip.mp4", io.BytesIO(b"not a real video, just test bytes"), "video/mp4")}


def test_health_needs_no_key(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_create_job_without_api_key_is_rejected(client):
    res = client.post("/jobs", files=_tiny_upload())
    assert res.status_code == 401


def test_create_job_with_wrong_api_key_is_rejected(client):
    res = client.post("/jobs", files=_tiny_upload(), headers={"X-API-Key": "wrong"})
    assert res.status_code == 401


def test_create_job_and_poll_to_completion(client):
    res = client.post(
        "/jobs",
        files=_tiny_upload(),
        data={"match_id": "m1", "innings": "1", "over": "2", "ball": "3"},
        headers={"X-API-Key": "test-key"},
    )
    assert res.status_code == 200
    job_id = res.json()["job_id"]
    assert res.json()["status"] == "queued"

    # The fake runs synchronously fast, but it's still on a background thread -- poll
    # briefly rather than assuming it's already done the instant the response returns.
    deadline = time.time() + 5
    job = None
    while time.time() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.05)

    assert job is not None
    assert job["status"] == "done", job
    assert job["result"]["shot_classification"] == "unknown"
    assert job["result"]["match_id"] == "m1"
    assert job["result"]["over_number"] == 2


def test_unknown_job_id_is_404(client):
    res = client.get("/jobs/does-not-exist")
    assert res.status_code == 404


def test_oversized_upload_is_rejected(client):
    big = io.BytesIO(b"x" * (2 * 1024 * 1024))  # 2MB against the fixture's 1MB limit
    res = client.post(
        "/jobs", files={"file": ("clip.mp4", big, "video/mp4")}, headers={"X-API-Key": "test-key"}
    )
    assert res.status_code == 413
