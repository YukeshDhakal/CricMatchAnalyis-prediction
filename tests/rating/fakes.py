"""Fakes for the rating layer's swappable interfaces.

Same pattern as `tests/fakes.py` -- hand-written classes implementing the real ABC,
no mocking framework, scripted return values -- and here for the same reason that
module exists: so the suite never needs the thing being faked to actually be running.
`tests/fakes.py` keeps the video-engine tests off a GPU and model weights; these keep
the suggestion tests off a live Ollama server.

They live in `tests/rating/` rather than in `tests/fakes.py` deliberately. That module
imports `video_engine.detection.base`, whose package `__init__` eagerly imports
`ultralytics` (and therefore torch); putting these there would make the rating tests --
which are pure arithmetic over dataclasses and touch no CV code at all -- drag in the
entire computer-vision dependency stack to run.
"""
from __future__ import annotations

from rating.contracts import NoteDraft, PerformanceFlag
from rating.llm import NoteWriter


class FakeNoteWriter(NoteWriter):
    """A `NoteWriter` that returns scripted bodies instead of calling a model.

    Records every flag it was handed in `seen`, so a test can assert on what the
    pipeline *passed to* the writer. That is the more important half of PRD 2.4's
    guarantee than what came back: a writer can only cite what it was given, so a flag
    arriving without its stat or its citations is a failure in the rule layer no
    amount of validation downstream can fix.

    `bodies` is consumed in order and the last entry repeats once exhausted, so a
    one-entry list covers a whole batch.
    """

    def __init__(self, bodies: list[str], source: str = "llm") -> None:
        self._bodies = list(bodies)
        self._source = source
        self.seen: list[PerformanceFlag] = []

    def write(self, flag: PerformanceFlag) -> NoteDraft:
        self.seen.append(flag)
        index = min(len(self.seen) - 1, len(self._bodies) - 1)
        return NoteDraft(title=f"note for {flag.metric.value}", body=self._bodies[index], source=self._source)


class FakeOllamaTransport:
    """A stand-in for `rating.llm.OllamaClient`'s HTTP transport.

    Replays scripted `/api/generate` responses so the client's citation validation,
    retry and template-fallback behaviour can be exercised without a server. A `str`
    entry becomes `{"response": <str>}`, a `dict` is returned as-is (for malformed
    payloads), and an `Exception` entry is raised (for transport failures). An
    exhausted script raises, which the client treats like any other failure.

    Records each request payload in `requests`, so a test can assert the prompt
    actually carried the citations -- a model can only echo a token it was given.
    """

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.requests: list[dict] = []

    def __call__(self, url: str, payload: dict, timeout_s: float) -> dict:
        self.requests.append(payload)
        if not self._responses:
            raise ConnectionError("no scripted response left")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        if isinstance(nxt, str):
            return {"response": nxt}
        return nxt
