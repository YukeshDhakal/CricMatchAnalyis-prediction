"""The one worker that runs the CV pipeline, and the process boundary around it.

**What this is not a fix for.** The production incident that prompted this module was a
container killed mid-job, and the first hypothesis was that the pipeline running as a
*thread* starved uvicorn's asyncio event loop through the GIL until Railway's edge gave
up. That was measured and it is not what happens: with a pipeline-shaped CPU-bound
workload (4K `cv2.cvtColor`, a torch conv forward, `to_tensor` on 4K frames, and a plain
Python IoU loop) pinned in a `ThreadPoolExecutor`, the worst event-loop tick overshoot
observed was **51 ms**, and the median was under 6. CPython preempts a pure-Python thread
every 5 ms (`sys.getswitchinterval`) and OpenCV and torch release the GIL around their
native calls, so a background thread delays the event loop by milliseconds, not by the
tens of seconds an edge proxy waits. The real cause was memory -- see `api.frame_budget`.

**So why a separate process anyway.** Not for latency; for blast radius. Everything this
worker runs is native code operating on attacker-supplied video: OpenCV decoders,
Ultralytics, torch. When something in that stack dies hard -- a segfault on a malformed
container, or the kernel's OOM killer picking the process with the largest resident set --
a thread takes the whole service down with it, because the HTTP server shares its address
space. In a child process the same death is a failed job: the parent keeps serving, the
future comes back broken, that one job is marked `error` with a real message, and the next
job gets a fresh worker. That is the difference between "the service restarted and every
in-flight job vanished" and "one clip failed".

This is defence in depth and it is deliberately second in line. A process boundary would
not have saved the 4K clip on its own -- it would have moved the OOM kill from the server
to the worker, which is better, but the job would still have failed every time. The bound
in `api.frame_budget` is what makes the job succeed; this is what keeps a future unhandled
crash from taking the API with it.

**One worker, still.** Unchanged from the thread-pool version and for the same reason:
`KeypointRcnnPoseEstimator` and `YoloDetector` are real models, and two concurrent
analyses on a CPU host contend rather than parallelise.

**`spawn` on every platform, including Linux.** `fork` is the default on Linux and is the
wrong choice here twice over. The pool's worker is created lazily on the first submission,
which is long after uvicorn has started its own threads, and forking a multi-threaded
process gives the child a memory image containing locks held by threads that do not exist
in it -- torch's OpenMP pool is a well-known way to deadlock exactly like this. It would
also inherit the parent's loaded modules, which sounds like a saving and is really a way
for the child's state to depend on whatever the HTTP process happened to have touched.
`forkserver` avoids both, but only exists on POSIX, and using it in production while tests
on Windows exercise `spawn` would mean the thing being tested is not the thing shipping.
`spawn` is uniform, so the worker start-up path is the same one the test suite runs.

The cost of `spawn` is that the child re-imports torch and re-loads the models on its
first job. That is a one-time cost per worker, not per job -- `_Components` in
`api.pipeline_runner` is a lazy singleton and the single worker process is reused across
submissions, so the second job and every job after it reuses the models already resident
in the child. `tests/api/test_job_executor.py` asserts that rather than assuming it.
"""
from __future__ import annotations

import multiprocessing
import os
import threading
from concurrent.futures import BrokenExecutor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

DEFAULT_MODE = "process"


def run_pipeline_job(
    video_path: Path,
    source_label: str,
    match_id: str | None,
    innings: int | None,
    over: int | None,
    ball: int | None,
    include_frames: bool,
) -> dict[str, Any]:
    """The worker entrypoint, run in the child process.

    **Returns an outcome, never raises.** Anything raised here would have to survive
    pickling to reach the parent, and a good deal of what this stack raises does not: an
    exception whose `__init__` takes arguments its `args` tuple does not reproduce fails
    to unpickle, and the parent then sees an unrelated error about the error. Flattening
    to a string in the child, where the real exception still exists, is the only way to be
    sure the caller learns what actually went wrong.

    Resolved through the module rather than imported at the top so that patching
    `api.pipeline_runner.run_analysis` in a test reaches this function -- which is what
    lets the HTTP tests drive this exact worker with a fake pipeline instead of a
    near-copy of it.
    """
    from api import pipeline_runner

    try:
        result = pipeline_runner.run_analysis(
            video_path, source_label, match_id, innings, over, ball, include_frames
        )
    except Exception as exc:  # noqa: BLE001 -- a failed job is a reportable outcome
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "result": result}


class JobExecutor:
    """A single-worker executor that replaces itself when its worker dies.

    `concurrent.futures.ProcessPoolExecutor` is permanently unusable once a worker exits
    uncleanly: every pending future resolves to `BrokenProcessPool` and so does every
    subsequent `submit`. Left alone that turns one killed worker into a service where
    every job from then on fails until someone restarts the container -- strictly worse
    than the thread pool this replaced, which at least died visibly. So a broken pool is
    discarded and rebuilt, and the cost of a crash stays with the job that caused it.
    """

    def __init__(self, mode: str | None = None) -> None:
        self.mode = (mode or os.environ.get("THIRD_UMPIRE_JOB_EXECUTOR", DEFAULT_MODE)).lower()
        if self.mode not in ("process", "thread"):
            raise ValueError(
                f"THIRD_UMPIRE_JOB_EXECUTOR must be 'process' or 'thread', got {self.mode!r}."
            )
        self._lock = threading.Lock()
        self._executor: ProcessPoolExecutor | ThreadPoolExecutor | None = None
        self._generation = 0

    def _build(self) -> ProcessPoolExecutor | ThreadPoolExecutor:
        if self.mode == "thread":
            return ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")
        return ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn")
        )

    @property
    def generation(self) -> int:
        """How many times a worker pool has been built. Only interesting to tests, which
        use it to prove a pool was *not* rebuilt between two jobs (and therefore that the
        models were not reloaded)."""
        with self._lock:
            return self._generation

    def submit(self, fn: Callable[..., Any], *args: Any) -> Future:
        """Queue `fn`, rebuilding the pool first if the previous worker died.

        A pool can be found broken either here (it died while idle, so `submit` itself
        raises) or later through a future. Both paths land on `_discard`; this one retries
        once so a caller never sees a failure caused purely by the *previous* job's crash.
        """
        with self._lock:
            if self._executor is None:
                self._executor = self._build()
                self._generation += 1
            try:
                return self._executor.submit(fn, *args)
            except BrokenExecutor:
                self._executor.shutdown(wait=False)
                self._executor = self._build()
                self._generation += 1
                return self._executor.submit(fn, *args)

    def discard_if_broken(self, exc: BaseException | None) -> bool:
        """Drop the pool if `exc` (a future's exception) means its worker is gone.

        Returns whether it did, so the caller can say so in the job's error message --
        "the worker process died" is a materially different thing to tell someone than a
        pipeline error, and it is the one that means "try again".
        """
        if not isinstance(exc, BrokenExecutor):
            return False
        with self._lock:
            broken, self._executor = self._executor, None
        if broken is not None:
            broken.shutdown(wait=False)
        return True

    def shutdown(self, wait: bool = False) -> None:
        with self._lock:
            current, self._executor = self._executor, None
        if current is not None:
            current.shutdown(wait=wait)
