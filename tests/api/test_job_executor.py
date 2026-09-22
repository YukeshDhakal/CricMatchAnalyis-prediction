"""Tests the real process boundary, with real subprocesses.

Every other API test runs the worker on a thread, because a monkeypatched fake pipeline
cannot cross into a child process. That makes this file the only place the thing that
actually ships gets exercised, so it uses no fakes at the executor level: it starts real
`spawn`ed workers, kills one the way the kernel would, and asserts on what comes back.

The four properties worth holding down, all of them things the thread pool gave for free
and a process pool does not:

1. Arguments and results survive pickling (a `Path` in, a dict out).
2. The worker process is *reused* between jobs, so the lazy model singleton loads once --
   this is the entire reason `max_workers=1` is allowed to keep a singleton at all.
3. A worker that dies without returning fails one job, and the next job gets a working
   pool. `ProcessPoolExecutor` does not do this on its own; left alone, one killed worker
   makes every subsequent job fail forever.
4. A pipeline exception comes back as a readable message even when the exception object
   itself cannot be pickled.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import BrokenExecutor
from pathlib import Path

import pytest

from api.job_executor import JobExecutor, run_pipeline_job
from tests.api import worker_probes


@pytest.fixture
def executor():
    ex = JobExecutor(mode="process")
    try:
        yield ex
    finally:
        ex.shutdown(wait=False)


def test_the_real_worker_signature_crosses_the_process_boundary(executor, tmp_path):
    """`run_pipeline_job` itself, with its real arguments, through a real subprocess.

    Both directions have to pickle: a `Path`, three `int | None`s, a `str` and a `bool`
    going out, and a result dict coming back. The clip does not exist, which makes this
    cheap -- `run_analysis` probes the file before it touches a model, so nothing loads
    torch -- while still proving the failure arrives as a readable per-job outcome rather
    than as a dead worker.
    """
    outcome = executor.submit(
        run_pipeline_job, tmp_path / "missing.mp4", "missing.mp4", "m1", 1, 2, 3, False
    ).result(timeout=300)

    assert outcome["ok"] is False
    assert "missing.mp4" in outcome["error"]


def test_the_worker_process_is_reused_so_models_load_once(executor, tmp_path):
    """Two jobs back to back land in the same process, and the second does not reload.

    This is the claim the whole design rests on: `ProcessPoolExecutor(max_workers=1)`
    keeps its worker alive between submissions, so `pipeline_runner._Components` --
    module-level state that does *not* inherit from the parent under `spawn` -- loads on
    the first job in the child and is still there for the second.

    Both halves are asserted. `loads == 1` alone would also pass if each job ran in a
    brand new process (each would report its own first load), so the pid is checked too:
    same pid, one load, and the load attributed to that pid.
    """
    first = executor.submit(worker_probes.load_and_report, tmp_path).result(timeout=120)
    second = executor.submit(worker_probes.load_and_report, tmp_path).result(timeout=120)

    assert second["pid"] == first["pid"], "the pool started a second worker"
    assert second["loads"] == 1, "the models were loaded again for the second job"
    assert second["loaded_by_pid"] == second["pid"]
    assert executor.generation == 1, "the pool was rebuilt between jobs"


def test_the_worker_runs_in_a_different_process_from_the_server(executor, tmp_path):
    import os

    result = executor.submit(worker_probes.load_and_report, tmp_path).result(timeout=120)
    assert result["pid"] != os.getpid()


def test_a_killed_worker_fails_one_job_and_the_next_one_still_runs(executor, tmp_path):
    """The failure mode that would have been *worse* than the bug being fixed.

    A worker killed mid-job (OOM killer, segfault) leaves `ProcessPoolExecutor`
    permanently broken: the in-flight future raises `BrokenProcessPool`, and so does
    every `submit` after it. Without `discard_if_broken`, one bad clip would take every
    subsequent job down with it and the only cure would be restarting the container --
    turning a crash that used to be loud and self-healing into a quiet, permanent one.
    """
    doomed = executor.submit(worker_probes.die_hard, tmp_path)
    with pytest.raises(BrokenExecutor):
        doomed.result(timeout=120)

    assert executor.discard_if_broken(doomed.exception()) is True

    survivor = executor.submit(worker_probes.load_and_report, tmp_path).result(timeout=120)
    assert survivor["loads"] == 1
    assert executor.generation == 2, "a fresh pool should have been built"


def test_submitting_to_a_pool_whose_worker_already_died_recovers(executor, tmp_path):
    """The same recovery, but with nobody having inspected the broken future first.

    `discard_if_broken` is driven by the completion callback in `api.server`; if that
    path were ever missed, `submit` itself has to cope rather than propagate a failure
    caused entirely by the previous job.
    """
    doomed = executor.submit(worker_probes.die_hard, tmp_path)
    with pytest.raises(BrokenExecutor):
        doomed.result(timeout=120)

    # Deliberately no discard_if_broken() call here.
    recovered = executor.submit(worker_probes.load_and_report, tmp_path).result(timeout=120)
    assert recovered["loads"] == 1


def test_an_unpicklable_exception_still_comes_back_as_a_message(monkeypatch):
    """`run_pipeline_job` flattens failures in the child, where the exception still exists.

    Raised across the boundary instead, this one would fail to unpickle and the parent
    would report a `TypeError` about reconstructing the error rather than the error.
    """
    import api.pipeline_runner as pipeline_runner

    monkeypatch.setattr(pipeline_runner, "run_analysis", worker_probes.raise_unpicklable)
    outcome = run_pipeline_job(Path("clip.mp4"), "clip.mp4", None, None, None, None, False)

    assert outcome["ok"] is False
    assert "the real reason" in outcome["error"]


def test_a_pipeline_failure_is_an_outcome_not_an_exception(monkeypatch, tmp_path):
    """The ordinary failure path: a job that raises is reported, not propagated."""
    import api.pipeline_runner as pipeline_runner

    def boom(*_args, **_kwargs):
        raise ValueError("could not open video file")

    monkeypatch.setattr(pipeline_runner, "run_analysis", boom)
    outcome = run_pipeline_job(tmp_path / "c.mp4", "c.mp4", None, None, None, None, False)

    assert outcome == {"ok": False, "error": "ValueError: could not open video file"}


def test_a_successful_run_is_wrapped_in_an_ok_outcome(monkeypatch, tmp_path):
    import api.pipeline_runner as pipeline_runner

    monkeypatch.setattr(
        pipeline_runner, "run_analysis", lambda *a, **k: {"shot_classification": "drive"}
    )
    outcome = run_pipeline_job(tmp_path / "c.mp4", "c.mp4", "m1", 1, 2, 3, True)

    assert outcome == {"ok": True, "result": {"shot_classification": "drive"}}


@pytest.mark.skipif(
    (os.cpu_count() or 1) < 2, reason="needs more than one core to separate the two cases"
)
def test_a_process_worker_leaves_the_parent_more_responsive_than_a_thread_worker():
    """The measurement behind choosing a process, written down as an assertion.

    A worker saturating a core with pure-Python work -- the least GIL-friendly thing this
    pipeline does -- is run both ways while the parent times how long a 5 ms sleep really
    takes. Measured repeatedly while writing this: 10.2 ms median against a thread worker,
    0.50 ms against a process worker, a twentyfold difference that held across runs.

    The assertion is deliberately loose (a factor of two, and 5 ms of absolute slack)
    because the honest conclusion from these numbers is *not* that the thread version was
    breaking anything. Ten milliseconds does not get a container killed; `api.frame_budget`
    explains what did. This test exists so that the process boundary's benefit is a
    measured fact rather than a plausible story, and so it would be noticed if some future
    change put the pipeline back in the server's interpreter.
    """
    def worst_and_median(mode: str) -> tuple[float, float]:
        ex = JobExecutor(mode=mode)
        try:
            running = ex.submit(worker_probes.burn_cpu, 3.0)
            time.sleep(0.3)  # let it get going
            stalls = []
            for _ in range(200):
                started = time.perf_counter()
                time.sleep(0.005)
                stalls.append(time.perf_counter() - started - 0.005)
            running.result(timeout=120)
            stalls.sort()
            return stalls[-1], stalls[len(stalls) // 2]
        finally:
            ex.shutdown(wait=True)

    _thread_worst, thread_median = worst_and_median("thread")
    _process_worst, process_median = worst_and_median("process")

    assert process_median < 0.005, f"process worker stalled the parent {process_median:.4f}s"
    assert process_median * 2 < thread_median + 0.005, (
        f"expected a process worker to be clearly gentler on the parent; got "
        f"thread={thread_median:.4f}s process={process_median:.4f}s"
    )


def test_an_unknown_mode_is_rejected_at_construction():
    """A typo in `THIRD_UMPIRE_JOB_EXECUTOR` should fail loudly at start-up rather than
    silently picking a default -- "it ran on a thread all along" is not something anyone
    would notice from the outside."""
    with pytest.raises(ValueError, match="process.*thread"):
        JobExecutor(mode="subprocess")
