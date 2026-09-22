"""Picklable functions for `test_job_executor.py` to run in a real child process.

Not a `test_*` module on purpose: pytest must not collect it, but a spawned worker must
be able to *import* it by name, which is the whole point. Everything here has to be
importable from a fresh interpreter with only `sys.path` carried over, so it deliberately
imports nothing from this project.
"""
from __future__ import annotations

import os
import time


class _FakeComponents:
    """The same shape as `api.pipeline_runner._Components`: module-level state, populated
    on first use, reused after.

    This is the thing that has to survive between two submissions for the process pool to
    be worth having. Testing it with a counter rather than the real models is not a
    shortcut around the question -- what is in doubt is whether *the worker process is
    reused*, not whether torch can load weights, and a counter answers that in
    milliseconds instead of by downloading 226 MB of Keypoint R-CNN into CI.
    """

    _instance: dict | None = None
    loads: int = 0

    @classmethod
    def get(cls) -> dict:
        if cls._instance is None:
            cls.loads += 1
            cls._instance = {"loaded_by_pid": os.getpid()}
        return cls._instance


def load_and_report(_ignored: object = None) -> dict:
    """Returns the worker's pid and how many times it has 'loaded the models'.

    A second call landing in the same process reports the same pid and `loads == 1`.
    A second call landing in a *new* process reports a different pid and `loads == 1`
    again -- so the pid is the fact that distinguishes reuse from a reload, and both are
    asserted together.
    """
    components = _FakeComponents.get()
    return {
        "pid": os.getpid(),
        "loads": _FakeComponents.loads,
        "loaded_by_pid": components["loaded_by_pid"],
    }


def die_hard(_ignored: object = None) -> dict:
    """Leave without unwinding, the way a segfault or an OOM kill leaves.

    `os._exit` skips `finally` blocks, atexit handlers and any chance to return a value --
    exactly the exit the worker takes when the kernel kills it, and exactly the one
    `ProcessPoolExecutor` turns into a permanently broken pool if nobody intervenes.
    """
    os._exit(1)


def raise_unpicklable(*_args: object, **_kwargs: object) -> dict:
    """Raise something that cannot make the trip home as an exception object.

    `__init__` demands two arguments but `args` only carries one, so unpickling it raises
    `TypeError` inside the parent's result handling and the original message is lost. This
    is why `run_pipeline_job` flattens failures to a string in the child rather than
    letting them propagate.
    """

    class Awkward(Exception):
        def __init__(self, a, b):  # noqa: D107
            super().__init__(a)
            self.b = b

    raise Awkward("the real reason", "extra")


def burn_cpu(seconds: float) -> dict:
    """Saturate a core with pure-Python work -- the least GIL-friendly thing this
    pipeline does (the IoU and keypoint loops), not the most."""
    end = time.perf_counter() + float(seconds)
    total = 0
    while time.perf_counter() < end:
        for i in range(10_000):
            total += i * i
    return {"pid": os.getpid(), "total": total}
