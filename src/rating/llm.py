"""Note-writers: turn an already-computed `PerformanceFlag` into coaching prose.

**This module never decides anything.** It receives a flag whose stat, baseline,
delta and `DeliveryRef` citations were all computed deterministically by
`rating.suggestions` from real rows, and its only job is phrasing. That boundary is
the mechanism behind PRD 2.4's "no black-box verdicts -- every AI-derived suggestion
must cite its underlying clip and stat so an analyst can verify it": an LLM that
cannot originate a number or a citation cannot produce an unverifiable claim, no
matter what it hallucinates, because anything it originates is rejected before it
reaches a `CoachingSuggestion`.

Two implementations of one small interface (`NoteWriter`), swappable at the call site
the way `video_engine`'s `Detector`/`PoseEstimator` are:

* `TemplateNoteWriter` -- the PRD's original curated template library. Deterministic,
  dependency-free, always available.
* `OllamaClient` -- a local LLM (PRD deviation, agreed with the user), with
  `TemplateNoteWriter` as its fallback. It degrades to the template when the server
  is unreachable, times out, returns junk, drops a citation, or invents a number.
  The suggestion engine therefore never hard-fails because Ollama isn't running.

**Model choice.** The default is `phi3.5` (Microsoft's 3.8B instruct model), not the
`llama3.2:1b` that happened to be pulled on the dev machine. For this job -- follow
constrained formatting rules exactly and don't embellish -- instruction adherence
matters far more than breadth, and in the <=4GB-VRAM class Phi-3.5-mini leads on the
closest available proxies for it (~69% MMLU / ~86% GSM8K, vs ~63%/~77% for Llama 3.2
3B, ~65%/~79% for Qwen2.5 3B, ~52%/~40% for Gemma 2 2B, ~49% MMLU for Llama 3.2 1B).
It is also MIT-licensed (no redistribution restrictions to reason about, unlike the
Llama and Qwen community licenses), ~2.2GB on disk, and natively available as
`ollama pull phi3.5`. **Llama 3.2 3B is the documented alternative** if Meta's
ecosystem is preferred; `llama3.2:1b` remains a supported swap-in for low-resource
machines. Change either with the `THIRD_UMPIRE_OLLAMA_MODEL` env var or the `model`
constructor argument -- nothing here is hardcoded to one model.

Those are borrowed leaderboard numbers, and no published benchmark measures this
exact task (coaching-note generation with mandatory verbatim citation echo). The
evidence that actually matters for this deployment is
`scripts/eval_llm_notewriter.py`, which runs synthetic flags through a live Ollama
and measures citation-echo and number-fidelity pass rates on this machine.
"""
from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence

from .contracts import METRIC_SPECS, Metric, NoteDraft, PerformanceFlag

__all__ = [
    "NoteWriter",
    "TemplateNoteWriter",
    "OllamaClient",
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_OLLAMA_MODEL",
    "ALTERNATIVE_OLLAMA_MODELS",
    "OLLAMA_BASE_URL_ENV",
    "OLLAMA_MODEL_ENV",
    "build_prompt",
    "missing_citations",
    "unsupported_numbers",
    "write_notes",
]

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

# See this module's docstring for the comparison behind this default.
DEFAULT_OLLAMA_MODEL = "phi3.5"

# Documented swap-ins, in preference order after the default. `llama3.2:3b` is the
# alternative for a Meta-ecosystem preference; `llama3.2:1b` is the low-resource
# option (weakest instruction adherence of the three -- expect it to fall back to the
# template more often, which is a degradation in phrasing quality, never in accuracy).
ALTERNATIVE_OLLAMA_MODELS = ("llama3.2:3b", "qwen2.5:3b", "llama3.2:1b")

OLLAMA_BASE_URL_ENV = "THIRD_UMPIRE_OLLAMA_URL"
OLLAMA_MODEL_ENV = "THIRD_UMPIRE_OLLAMA_MODEL"

# Numbers in a generated note are matched against the input values at these rounding
# precisions. A model that writes "8.4" for an input of 8.42 is rephrasing, not
# inventing; one that writes "9.1" is inventing.
_ACCEPTED_ROUNDINGS = (0, 1, 2)

_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


class NoteWriter(ABC):
    """Turns one `PerformanceFlag` into the prose half of a `CoachingSuggestion`.

    Deliberately tiny, and deliberately takes the whole flag rather than pre-formatted
    strings: an implementation may need the phase, the sample size or the baseline
    source, and a writer that had to be handed exactly the right pre-rendered fields
    would push formatting decisions back into the rule layer.

    An implementation must never add a fact the flag doesn't contain. `OllamaClient`
    enforces that on its model's output; a hand-written implementation is on its
    author's honour, which is why `NoteDraft.source` records which one ran.
    """

    @abstractmethod
    def write(self, flag: PerformanceFlag) -> NoteDraft:
        """Return the note for `flag`. Must not raise -- a writer that can't produce a
        note should degrade to something deterministic, because a suggestion pipeline
        that dies on one unphrasable flag loses every other finding in the batch."""


def _pct(value: float) -> float:
    """A relative delta as a signed percentage, 1dp -- the form every note quotes it in."""
    return round(100 * value, 1)


def _phase_clause(flag: PerformanceFlag) -> str:
    return f"in the {flag.phase} overs" if flag.phase else "across the innings"


def _direction_word(flag: PerformanceFlag) -> str:
    return "above" if flag.delta > 0 else "below"


class TemplateNoteWriter(NoteWriter):
    """PRD 3.2's curated coaching-note template library.

    Small on purpose. This is the floor the whole suggestion engine stands on -- it is
    what runs when Ollama is down, when `requests` isn't installed, when the model
    drops a citation, and in every test -- so it has no dependencies, no network, no
    randomness, and no way to fail. Its phrasing is blunt rather than good; that is
    the correct trade for a fallback, and `OllamaClient` exists precisely to do better
    when it can.

    Templates are keyed by `(Metric, is_concern)` because the coaching point inverts
    with the sign: a low dot-ball percentage and a high one are different notes, not
    the same note with a different adjective. An unlisted key falls through to
    `_generic_body`, which is still correct and still cited -- a new `Metric` should
    degrade to plain phrasing, never to an exception in the middle of a batch.
    """

    _TITLES: dict[tuple[Metric, bool], str] = {
        (Metric.STRIKE_RATE, True): "Scoring rate below baseline",
        (Metric.STRIKE_RATE, False): "Scoring rate ahead of baseline",
        (Metric.ECONOMY_RATE, True): "Leaking runs against baseline",
        (Metric.ECONOMY_RATE, False): "Economy ahead of baseline",
        (Metric.DOT_BALL_PERCENT, True): "Too many dot balls",
        (Metric.DOT_BALL_PERCENT, False): "Rotating strike well",
        (Metric.BOUNDARY_PERCENT, True): "Boundary rate below baseline",
        (Metric.BOUNDARY_PERCENT, False): "Boundary rate ahead of baseline",
        (Metric.PHASE_STRIKE_RATE, True): "Phase scoring rate below baseline",
        (Metric.PHASE_STRIKE_RATE, False): "Phase scoring rate ahead of baseline",
        (Metric.PHASE_ECONOMY_RATE, True): "Phase economy below baseline",
        (Metric.PHASE_ECONOMY_RATE, False): "Phase economy ahead of baseline",
    }

    _BODIES: dict[tuple[Metric, bool], str] = {
        (Metric.STRIKE_RATE, True): (
            "{player} struck at {actual} {phase_clause}, {gap}% below the baseline of {baseline} "
            "({baseline_source}). Look at strike rotation off the front foot before "
            "adding risk. Evidence: {citations}"
        ),
        (Metric.STRIKE_RATE, False): (
            "{player} struck at {actual} {phase_clause}, {gap}% above the baseline of {baseline} "
            "({baseline_source}). Worth reviewing what is working here before the next "
            "match. Evidence: {citations}"
        ),
        (Metric.ECONOMY_RATE, True): (
            "{player} went at {actual} runs per over {phase_clause}, {gap}% above the baseline of "
            "{baseline} ({baseline_source}). Review length and field alignment on "
            "the deliveries below. Evidence: {citations}"
        ),
        (Metric.ECONOMY_RATE, False): (
            "{player} went at {actual} runs per over {phase_clause}, {gap}% better than the "
            "baseline of {baseline} ({baseline_source}). Evidence: {citations}"
        ),
        (Metric.DOT_BALL_PERCENT, True): (
            "{player} left {actual}% of balls faced scoreless {phase_clause}, against a baseline "
            "of {baseline}% ({baseline_source}). Work on singles into the gaps rather "
            "than waiting for a boundary ball. Evidence: {citations}"
        ),
        (Metric.DOT_BALL_PERCENT, False): (
            "{player} left only {actual}% of balls faced scoreless {phase_clause}, against a "
            "baseline of {baseline}% ({baseline_source}). Evidence: {citations}"
        ),
        (Metric.BOUNDARY_PERCENT, True): (
            "{player} found the boundary off {actual}% of balls faced {phase_clause}, against "
            "a baseline of {baseline}% ({baseline_source}). Check shot selection against the "
            "deliveries below. Evidence: {citations}"
        ),
        (Metric.BOUNDARY_PERCENT, False): (
            "{player} found the boundary off {actual}% of balls faced {phase_clause}, against "
            "a baseline of {baseline}% ({baseline_source}). Evidence: {citations}"
        ),
    }

    def title_for(self, flag: PerformanceFlag) -> str:
        """The deterministic title for `flag`.

        Public because `OllamaClient` uses it too: a suggestion's title is never
        model-written, only its body is. A title is what an analyst scans a list by,
        so it is the last thing that should vary run to run, and pinning it also means
        the LLM has exactly one output to produce and one to be validated.
        """
        spec = METRIC_SPECS[flag.metric]
        title = self._TITLES.get((flag.metric, flag.is_concern))
        if title is not None:
            return title if flag.phase is None else f"{title} ({flag.phase})"
        direction = _direction_word(flag)
        scope = flag.phase or "innings"
        return f"{spec.label.capitalize()} {direction} baseline ({scope})"

    def write(self, flag: PerformanceFlag) -> NoteDraft:
        citations = ", ".join(flag.citation_tokens()) or "no delivery-level citation available"
        fields = {
            "player": flag.player,
            "actual": flag.actual,
            "baseline": flag.baseline,
            "gap": abs(_pct(flag.relative_delta)),
            "baseline_source": flag.baseline_source,
            "phase_clause": _phase_clause(flag),
            "citations": citations,
        }
        template = self._BODIES.get((flag.metric, flag.is_concern))
        body = template.format(**fields) if template else self._generic_body(flag, fields)
        return NoteDraft(title=self.title_for(flag), body=body, source="template")

    @staticmethod
    def _generic_body(flag: PerformanceFlag, fields: dict) -> str:
        spec = METRIC_SPECS[flag.metric]
        return (
            f"{flag.player}'s {spec.label} was {flag.actual} {spec.unit} "
            f"{fields['phase_clause']}, {fields['gap']}% {_direction_word(flag)} the baseline of "
            f"{flag.baseline} ({flag.baseline_source}). Evidence: {fields['citations']}"
        )


def build_prompt(flag: PerformanceFlag) -> str:
    """The prompt for `flag`: the computed facts, then the rules for restating them.

    Everything the model is allowed to say is listed above the rules, and the citation
    tokens are given as literal strings to copy rather than as a description of what
    to cite. That is the difference between a model *echoing* a citation and a model
    *composing* one -- only the first is verifiable, and `missing_citations` checks
    for exactly that echo.

    The number rule is the other half: with no arithmetic to do and no rounding asked
    for, any number in the output that isn't in the input is a fabrication rather than
    a paraphrase, which is what makes `unsupported_numbers` a usable check rather than
    a source of false alarms.
    """
    spec = METRIC_SPECS[flag.metric]
    tokens = flag.citation_tokens()
    citations = " ".join(tokens) if tokens else "(none available)"
    reading = "worse than expected" if flag.is_concern else "better than expected"

    return (
        "You are a cricket performance analyst's writing assistant. Restate one "
        "already-computed finding as a short coaching note.\n"
        "\n"
        "FACTS (the only facts you may use):\n"
        f"- Player: {flag.player}\n"
        f"- Metric: {spec.label} ({spec.unit})\n"
        f"- Phase: {flag.phase or 'whole innings'}\n"
        f"- Measured value: {flag.actual}\n"
        f"- Baseline value: {flag.baseline} (baseline is the {flag.baseline_source})\n"
        f"- Difference vs baseline: {flag.delta} ({_pct(flag.relative_delta)}%)\n"
        f"- Reading: this is {reading} for the player\n"
        f"- Sample size: {flag.sample_size} balls\n"
        f"- Citations to copy exactly: {citations}\n"
        "\n"
        "RULES:\n"
        "1. Two or three sentences, 60 words maximum, addressed to a coach.\n"
        "2. Use ONLY the numbers listed above. Do not calculate, convert, estimate, "
        "round or invent any other number.\n"
        "3. Reproduce every citation above character for character, square brackets "
        "included. Do not reword, renumber, reformat or split them.\n"
        "4. Output the note text only: no preamble, no heading, no bullet points, no "
        "closing commentary.\n"
    )


def missing_citations(text: str, flag: PerformanceFlag) -> tuple[str, ...]:
    """Citation tokens from `flag` that don't appear verbatim in `text`.

    A plain substring check, on purpose. Anything fuzzier -- normalising whitespace,
    accepting a reordered token, matching on the match_id alone -- would accept a
    citation the model retyped, and a retyped citation is one the model could have
    retyped wrong. Empty means the note is fully cited; non-empty means it fails
    PRD 2.4 and must not reach an analyst as-is.
    """
    return tuple(token for token in flag.citation_tokens() if token not in text)


def unsupported_numbers(text: str, flag: PerformanceFlag) -> tuple[str, ...]:
    """Numbers in `text` that aren't traceable to `flag`'s inputs.

    Citation tokens are stripped out before scanning, since their digits (innings,
    over, ball, and any digits inside a match_id) are legitimate and already
    validated by `missing_citations`.

    Each remaining number must match one of the flag's own values at 0, 1 or 2 decimal
    places, or that value's magnitude -- a model writing "8.4" for 8.42, or "12%
    below" for a -12.0% delta, is rephrasing. Digits that were in the prompt as text
    (the metric's unit, the baseline description) count as input too. A number that
    appears nowhere in the input is asserting something no row supports, which is the
    exact failure PRD 2.4 forbids, so it is rejected even though the prose around it
    may read perfectly well.

    Best-effort by design: it catches invented *quantities*, not invented *claims*
    ("his footwork collapsed" has no number in it and passes). Deterministic
    generation and a tight prompt are what limit the latter; this check is the
    backstop for the numeric half, which is the half an analyst is most likely to
    take on trust.
    """
    for token in flag.citation_tokens():
        text = text.replace(token, " ")

    allowed = _allowed_number_strings(flag)
    found = _NUMBER_PATTERN.findall(text)
    return tuple(n for n in found if n.lstrip("0") not in allowed and n not in allowed)


def _allowed_number_strings(flag: PerformanceFlag) -> set[str]:
    sources = [
        flag.actual,
        flag.baseline,
        flag.delta,
        abs(flag.delta),
        _pct(flag.relative_delta),
        abs(_pct(flag.relative_delta)),
        float(flag.sample_size),
    ]
    spec = METRIC_SPECS[flag.metric]
    allowed: set[str] = set()
    # Digits that arrive as *text* in the prompt and get quoted back verbatim are input,
    # not invention. The metric's unit is the one that matters in practice: "runs per
    # 100 balls" is the name of the scale, and flagging its 100 as a fabricated
    # statistic was a false positive frequent enough to reject otherwise-perfect notes.
    # The baseline description ("the cohort mean over 24 matches") is the same case.
    for text in (flag.baseline_source, spec.unit, spec.label, flag.phase or ""):
        for token in _NUMBER_PATTERN.findall(text):
            allowed.add(token)
            allowed.add(token.lstrip("0") or "0")
    for value in sources:
        for places in _ACCEPTED_ROUNDINGS:
            rounded = round(value, places)
            for rendered in (f"{rounded:.{places}f}", str(rounded), str(abs(rounded))):
                allowed.add(rendered)
                allowed.add(rendered.lstrip("-"))
                allowed.add(rendered.lstrip("-").lstrip("0") or "0")
    return allowed


def _requests_transport(url: str, payload: dict, timeout_s: float) -> dict:
    """POST `payload` as JSON to `url` and return the decoded response.

    `requests` is imported here rather than at module scope so that importing
    `rating` -- and therefore running the whole test suite, which uses fakes and the
    template writer -- never depends on an HTTP library being installed. An
    `ImportError` raised here is caught by `OllamaClient.write` exactly like a
    connection error, and falls back to the template.
    """
    import requests  # noqa: PLC0415 -- deliberately lazy; see docstring

    response = requests.post(url, json=payload, timeout=timeout_s)
    response.raise_for_status()
    return response.json()


class OllamaClient(NoteWriter):
    """Writes coaching notes with a locally-hosted Ollama model, with hard guardrails.

    The contract this upholds, in order, for every note:

    1. The **stat and the citations come from the flag**, never from the model. They
       are formatted into the prompt by `build_prompt` and are never parsed back out
       of the response -- the response only supplies sentences.
    2. The model's output must **echo every citation verbatim** (`missing_citations`).
    3. The model's output must contain **no number the flag didn't supply**
       (`unsupported_numbers`), unless `reject_unsupported_numbers=False`.
    4. Any failure at any step -- unreachable server, timeout, HTTP error, malformed
       JSON, empty text, missing citation, invented number, `requests` not installed
       -- retries up to `max_attempts` and then returns the `TemplateNoteWriter`
       draft. **The suggestion engine never hard-fails because the LLM isn't there.**

    The failure mode is therefore always "less fluent phrasing", never "wrong number"
    or "no suggestion", and `NoteDraft.source` tells the caller which happened.

    `base_url` and `model` fall back to the `THIRD_UMPIRE_OLLAMA_URL` /
    `THIRD_UMPIRE_OLLAMA_MODEL` env vars and then to the module defaults, so the
    deployment target is configurable without a code change. `transport` is injectable
    so the guardrail logic above can be tested against scripted responses without a
    live server -- see `tests/rating/test_llm.py`.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: float = 30.0,
        max_attempts: int = 2,
        fallback: Optional[NoteWriter] = None,
        transport: Optional[Callable[[str, dict, float], dict]] = None,
        reject_unsupported_numbers: bool = True,
        temperature: float = 0.1,
    ) -> None:
        self.base_url = (base_url or os.environ.get(OLLAMA_BASE_URL_ENV) or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
        self.model = model or os.environ.get(OLLAMA_MODEL_ENV) or DEFAULT_OLLAMA_MODEL
        self.timeout_s = timeout_s
        self.max_attempts = max(1, max_attempts)
        self.fallback = fallback or TemplateNoteWriter()
        self._transport = transport or _requests_transport
        self.reject_unsupported_numbers = reject_unsupported_numbers
        self.temperature = temperature
        # Why the last attempt was rejected, for the eval script and for anyone
        # debugging a suddenly template-only run. Not part of NoteDraft: it describes
        # the writer's last call, not the note a caller ended up with.
        self.last_rejection: Optional[str] = None

    @property
    def generate_url(self) -> str:
        return f"{self.base_url}/api/generate"

    def write(self, flag: PerformanceFlag) -> NoteDraft:
        title = (
            self.fallback.title_for(flag)
            if isinstance(self.fallback, TemplateNoteWriter)
            else self.fallback.write(flag).title
        )
        self.last_rejection = None

        for _ in range(self.max_attempts):
            text = self._generate(flag)
            if text is None:
                continue
            problem = self.rejection_reason(text, flag)
            if problem is None:
                return NoteDraft(title=title, body=text, source="llm")
            self.last_rejection = problem

        return self.fallback.write(flag)

    def rejection_reason(self, text: str, flag: PerformanceFlag) -> Optional[str]:
        """Why `text` can't be used as `flag`'s note, or `None` if it can.

        Separate from `write` so the eval script can score a model's raw output
        without re-running generation, and so a test can assert on the reason rather
        than only on the fact that a fallback happened.
        """
        if not text.strip():
            return "empty response"
        missing = missing_citations(text, flag)
        if missing:
            return f"citation(s) not echoed verbatim: {', '.join(missing)}"
        if self.reject_unsupported_numbers:
            invented = unsupported_numbers(text, flag)
            if invented:
                return f"number(s) not present in the input: {', '.join(invented)}"
        return None

    def _generate(self, flag: PerformanceFlag) -> Optional[str]:
        """One generation attempt. `None` on any transport or decoding failure --
        the distinction between "couldn't reach the model" and "the model said
        something unusable" doesn't change what happens next (retry, then template),
        so `write` treats them the same."""
        payload = {
            "model": self.model,
            "prompt": build_prompt(flag),
            "stream": False,
            # Near-greedy: two runs over the same flag should differ as little as
            # possible, because a suggestion an analyst rejected shouldn't come back
            # reworded on the next run and read as a new finding.
            "options": {"temperature": self.temperature},
        }
        try:
            response = self._transport(self.generate_url, payload, self.timeout_s)
        except Exception as exc:  # noqa: BLE001 -- any failure degrades to the template
            self.last_rejection = f"{type(exc).__name__}: {exc}"
            return None

        if isinstance(response, (str, bytes)):
            try:
                response = json.loads(response)
            except ValueError:
                self.last_rejection = "response was not JSON"
                return None
        if not isinstance(response, dict):
            self.last_rejection = "response was not a JSON object"
            return None

        text = response.get("response")
        if not isinstance(text, str):
            self.last_rejection = "response JSON had no 'response' string"
            return None
        return text.strip()


def write_notes(flags: Sequence[PerformanceFlag], writer: NoteWriter) -> list[NoteDraft]:
    """`writer.write` across `flags`, in order. Convenience only -- kept here rather
    than in `suggestions` so a caller can draft notes without building suggestions."""
    return [writer.write(flag) for flag in flags]
