"""Proves the API keeps answering while a job is burning a core.

This is the property the production incident was *blamed* on and, it turns out, never
lost -- see `api.job_executor` for the measurements that sent the investigation to
`api.frame_budget` instead. It is still worth holding down, and worth being precise about
what each half of this file shows.

`test_..._thread_worker` runs the pipeline on a **thread**, in the same interpreter as the
HTTP server, with a fake that is a plain Python loop -- the least GIL-friendly work this
pipeline does (`_best_matching_track`'s IoU comparisons, the keypoint zips over
`.tolist()`), not the most. That is the *old* architecture under its worst case, and it
measures far worse than expected: **median `/health` latency around 540 ms**, against
2.4 ms on the same idle server. Not a few unlucky outliers -- the typical request. A
request crosses the event loop and anyio's worker pool, and every one of those hand-offs
has to win the GIL back from a thread that wants it.

That number is worth being precise about in both directions. It is about 200x worse than
idle, so "the thread version was fine" is not a supportable claim. It is also about 540
milliseconds, and an edge proxy waits tens of seconds, so it is not what killed the
container either. `api.frame_budget` is what killed the container. So this test asserts
what is actually true of the thread version -- every request still answered, and nowhere
near a timeout -- rather than a responsiveness it did not have.

`test_..._sibling_process` is the shape that ships: the server serving HTTP while a
*different process* saturates a core. No monkeypatched fake can cross into a real child,
so the burn is started directly through `JobExecutor` rather than through `POST /jobs` --
the HTTP layer is doing exactly what it does in production either way, which is the thing
under test. There the degradation is gone entirely, which is the measured case for the
process boundary independent of the crash-isolation argument.
"""
from __future__ import annotations

import importlib
import io
import time

import pytest
from fastapi.testclient import TestClient

from api import supabase_auth
from api.job_executor import JobExecutor
from tests.api import worker_probes

KEY = {"X-API-Key": "test-key"}

# Long enough that the sampling below sits squarely inside the job rather than around its
# edges: sampling costs a couple of seconds, mostly in those 700 ms outliers.
BURN_SECONDS = 6.0
# Six, not sixty: on a thread worker each of these costs about half a second, so a larger
# sample would outlast the job it is supposed to be measuring *during* and quietly turn
# into a measurement of an idle server. The thread-worker tests assert on the peak and on
# every call returning 200, neither of which needs a large sample.
SAMPLE_CALLS = 6

# What "responsive" means on an unloaded interpreter. Measured at about 2.4 ms; the
# sibling-process case has to stay in this territory, which is the point of that test.
MAX_ACCEPTABLE_MEDIAN = 0.1
# A tripwire for something structural, not a benchmark. Measured worst case on a thread
# worker is ~0.74 s, and an edge proxy waits tens of seconds, so there is a lot of room
# between "degraded" and "gone".
MAX_ACCEPTABLE_PEAK = 3.0


@pytest.fixture
def make_client(monkeypatch):
    """Builds a TestClient whose fake pipeline burns CPU for a chosen number of seconds."""

    def _make(burn_seconds: float, mode: str = "thread"):
        monkeypatch.setenv("THIRD_UMPIRE_API_KEY", "test-key")
        monkeypatch.setenv("THIRD_UMPIRE_JOB_EXECUTOR", mode)
        supabase_auth.clear_cache()

        import api.pipeline_runner as pipeline_runner
        import api.server as server_module

        importlib.reload(server_module)
        monkeypatch.setattr(server_module, "probe_video", lambda path: (1280, 720, 25.0, 120))

        def cpu_bound_pipeline(*_args, **_kwargs):
            end = time.perf_counter() + burn_seconds
            total = 0
            while time.perf_counter() < end:
                for i in range(10_000):
                    total += i * i
            return {"source_label": "burned", "total": total}

        monkeypatch.setattr(pipeline_runner, "run_analysis", cpu_bound_pipeline)
        clients.append(server_module)
        return server_module, TestClient(server_module.app)

    clients: list = []
    try:
        yield _make
    finally:
        for server_module in clients:
            server_module._executor.shutdown(wait=False)


def _start_job(http):
    res = http.post(
        "/jobs",
        files={"file": ("clip.mp4", io.BytesIO(b"pretend footage"), "video/mp4")},
        headers=KEY,
    )
    assert res.status_code == 200
    return res.json()["job_id"]


def _latencies(http, path, calls=SAMPLE_CALLS):
    """`calls` requests to `path`, sorted fastest to slowest.

    A fixed number of calls rather than a fixed time budget: sampling for N seconds would
    quietly shrink the sample exactly when latency got interesting, which is backwards.
    """
    out = []
    for _ in range(calls):
        started = time.perf_counter()
        res = http.get(path, headers=KEY)
        out.append(time.perf_counter() - started)
        assert res.status_code == 200, res.text
    return sorted(out)


def _median(values):
    return values[len(values) // 2]


def test_every_request_is_still_answered_by_a_thread_worker(make_client):
    """The old architecture under the worst workload it ever faced: degraded, not down.

    `_latencies` asserts a 200 on every call, so the load-bearing claim here is that none
    of the twelve requests failed or hung. The peak assertion is what separates "slower"
    from "an edge proxy gave up" -- the distinction the incident turned on.
    """
    _server, http = make_client(BURN_SECONDS)
    job_id = _start_job(http)
    assert http.get(f"/jobs/{job_id}", headers=KEY).json()["status"] == "running"

    latencies = _latencies(http, "/health")

    assert latencies[-1] < MAX_ACCEPTABLE_PEAK, f"peak {latencies[-1]:.3f}s"
    # The job was still running throughout, so the measurement means something.
    assert http.get(f"/jobs/{job_id}", headers=KEY).json()["status"] == "running"


def test_another_jobs_status_is_readable_while_one_is_running(make_client):
    """The poll that came back as a Railway 502 during the incident. Reading job state
    must not queue behind the job that is running -- a second job's status is answerable
    while the first is mid-analysis."""
    _server, http = make_client(BURN_SECONDS)
    first = _start_job(http)
    second = _start_job(http)
    assert http.get(f"/jobs/{second}", headers=KEY).json()["status"] == "queued"
    assert http.get(f"/jobs/{first}", headers=KEY).json()["status"] == "running"

    # The two assertions above already establish that this sample begins with one job
    # mid-analysis and another behind it; re-asserting that afterwards would only be
    # asserting that sampling finished before the job did, which is a fact about how slow
    # this test client is under load and not about the server.
    latencies = _latencies(http, f"/jobs/{second}")

    assert latencies[-1] < MAX_ACCEPTABLE_PEAK


def test_health_is_unaffected_by_a_cpu_bound_sibling_process(make_client):
    """The shipping shape: the pipeline's CPU load is in another process entirely.

    Compared against this same server's *idle* latency rather than against an absolute
    number, because the claim is specifically that a busy worker process costs the HTTP
    layer nothing -- not merely that it stays under some threshold.
    """
    _server, http = make_client(0.0)
    idle = _latencies(http, "/health", calls=20)

    burner = JobExecutor(mode="process")
    try:
        running = burner.submit(worker_probes.burn_cpu, BURN_SECONDS / 2)
        time.sleep(0.5)  # let the child actually start burning
        busy = _latencies(http, "/health", calls=20)
        running.result(timeout=120)
    finally:
        burner.shutdown(wait=True)

    assert _median(busy) < MAX_ACCEPTABLE_MEDIAN
    # Generous slack so this is a real-signal test and not a benchmark: what would fail it
    # is the pipeline finding its way back into the server's interpreter.
    assert _median(busy) < _median(idle) + 0.05, (
        f"a busy sibling process slowed /health: idle median {_median(idle):.4f}s, "
        f"busy median {_median(busy):.4f}s"
    )


def test_a_queued_job_is_promoted_to_running_when_the_worker_frees_up(make_client):
    """The status transition that used to be the worker's own first act, and now has to be
    the parent's -- the worker is in another process and cannot see job state at all."""
    _server, http = make_client(3.0)
    first = _start_job(http)
    second = _start_job(http)

    assert http.get(f"/jobs/{second}", headers=KEY).json()["status"] == "queued"

    deadline = time.time() + 30
    while time.time() < deadline:
        if http.get(f"/jobs/{first}", headers=KEY).json()["status"] == "done":
            break
        time.sleep(0.05)

    assert http.get(f"/jobs/{first}", headers=KEY).json()["status"] == "done"
    assert http.get(f"/jobs/{second}", headers=KEY).json()["status"] in ("running", "done")
