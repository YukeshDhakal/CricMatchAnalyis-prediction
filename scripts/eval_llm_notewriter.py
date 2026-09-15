"""Measure whether the local LLM note-writer actually honours its citation contract.

PRD 2.4's constraint -- "no black-box verdicts: every AI-derived suggestion must cite
its underlying clip and stat so an analyst can verify it" -- is enforced in code by
`rating.llm.OllamaClient`, which rejects a note that drops a citation or invents a
number. Whether a *given model* usually passes that check or usually trips it is a
different question, and it is not answerable from a leaderboard: the published
benchmarks behind the model choice (MMLU, GSM8K) measure reasoning, and no public
benchmark measures sports-coaching note generation with mandatory verbatim citation
echo. So the evidence for this deployment has to be measured here.

What it reports, over a spread of synthetic flags built the same way the unit tests
build theirs:

  * **Citation fidelity** -- the share of generations reproducing every citation token
    character for character. This is the number that matters: below ~90% the model is
    mostly falling back to templates and is not earning its latency.
  * **Number fidelity** -- the share introducing no number the flag didn't supply.
  * **Accepted** -- generations passing both, i.e. what an analyst would actually read
    as model-written prose.
  * **Latency** -- per-generation wall time, logged rather than asserted. This is one
    developer machine, not a benchmark rig.

A failure here is a quality result, not an outage: every rejected generation still
produces a correct, fully cited note from `TemplateNoteWriter`. That is why this is a
script and an opt-in test rather than part of the default suite -- the suite must not
depend on a model being pulled or a server being up.

Run:  python scripts/eval_llm_notewriter.py
      python scripts/eval_llm_notewriter.py --model llama3.2:3b --runs 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ingestion.contracts import DeliveryRef  # noqa: E402
from rating.contracts import Metric, PerformanceFlag  # noqa: E402
from rating.llm import (  # noqa: E402
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    OllamaClient,
    missing_citations,
    unsupported_numbers,
)


def eval_flags() -> list[PerformanceFlag]:
    """A spread of flags covering every metric, both directions, one and two citations,
    and phase and whole-innings scope.

    Two-citation cases are in deliberately: echoing one token is easier than echoing
    two without merging, reordering or renumbering them, and a real suggestion often
    carries three (`suggestions.MAX_CITATIONS_PER_FLAG`).
    """
    ref = DeliveryRef
    specs = [
        (Metric.ECONOMY_RATE, "death", 11.2, 8.4, 18, True, [ref("m1", 2, 18, 1)]),
        (Metric.ECONOMY_RATE, None, 9.1, 7.6, 24, True, [ref("m1", 2, 4, 3), ref("m1", 2, 11, 5)]),
        (Metric.ECONOMY_RATE, "powerplay", 5.2, 7.6, 24, False, [ref("m2", 1, 2, 2)]),
        (Metric.STRIKE_RATE, None, 92.5, 131.0, 40, True, [ref("m2", 1, 7, 4)]),
        (Metric.STRIKE_RATE, "middle", 71.0, 118.0, 22, True, [ref("m2", 1, 9, 1), ref("m2", 1, 12, 6)]),
        (Metric.STRIKE_RATE, "death", 210.0, 148.0, 15, False, [ref("m3", 1, 19, 2)]),
        (Metric.DOT_BALL_PERCENT, None, 51.0, 34.5, 45, True, [ref("m3", 2, 3, 1), ref("m3", 2, 8, 2)]),
        (Metric.DOT_BALL_PERCENT, "powerplay", 22.0, 36.0, 30, False, [ref("m3", 1, 1, 1)]),
        (Metric.BOUNDARY_PERCENT, None, 6.2, 14.8, 38, True, [ref("m4", 1, 5, 5)]),
        (Metric.BOUNDARY_PERCENT, "death", 31.0, 19.5, 16, False, [ref("m4", 2, 17, 3)]),
        (Metric.PHASE_STRIKE_RATE, "middle", 84.0, 112.0, 28, True, [ref("m5", 1, 10, 4)]),
        (Metric.PHASE_ECONOMY_RATE, "death", 13.5, 9.2, 12, True, [ref("m5", 2, 19, 6), ref("m5", 2, 20, 1)]),
        (Metric.PHASE_ECONOMY_RATE, "middle", 6.1, 7.4, 36, False, [ref("m5", 2, 9, 2)]),
    ]

    flags = []
    for metric, phase, actual, baseline, sample, is_concern, citations in specs:
        delta = round(actual - baseline, 2)
        flags.append(
            PerformanceFlag(
                player="A Sharma",
                metric=metric,
                phase=phase,
                actual=actual,
                baseline=baseline,
                delta=delta,
                relative_delta=round(delta / baseline, 4),
                is_concern=is_concern,
                severity=round(abs(delta / baseline), 4),
                sample_size=sample,
                baseline_source="cohort mean over 24 matches, 31 players",
                citations=tuple(citations),
                citations_have_video=True,
            )
        )
    return flags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument("--runs", type=int, default=1, help="generations per flag")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--verbose", action="store_true", help="print every generated note")
    args = parser.parse_args()

    # max_attempts=1: this measures the model's first-pass fidelity, not how often a
    # retry rescues it. The shipped client retries; this deliberately doesn't.
    client = OllamaClient(base_url=args.base_url, model=args.model, timeout_s=args.timeout, max_attempts=1)
    flags = eval_flags()

    print(f"Ollama:  {client.generate_url}")
    print(f"Model:   {client.model}")
    print(f"Cases:   {len(flags)} flags x {args.runs} run(s)\n")

    total = cited = numeric = accepted = 0
    latencies: list[float] = []
    failures: list[str] = []

    for _ in range(args.runs):
        for flag in flags:
            total += 1
            started = time.perf_counter()
            text = client._generate(flag)  # noqa: SLF001 -- the eval needs the raw generation
            latencies.append(time.perf_counter() - started)

            if text is None:
                failures.append(f"{flag.metric.value}: transport failure -- {client.last_rejection}")
                if len(failures) >= total:  # nothing has succeeded yet: likely no server
                    print(f"  ! {client.last_rejection}")
                continue

            dropped = missing_citations(text, flag)
            invented = unsupported_numbers(text, flag)
            cited += not dropped
            numeric += not invented
            accepted += not dropped and not invented

            if args.verbose:
                print(f"  [{flag.metric.value}] {text}\n")
            if dropped:
                failures.append(f"{flag.metric.value}: dropped citation(s) {', '.join(dropped)}")
            if invented:
                failures.append(f"{flag.metric.value}: invented number(s) {', '.join(invented)}")

    if not total:
        return 1

    def pct(n: int) -> str:
        return f"{100 * n / total:5.1f}%  ({n}/{total})"

    print("-" * 60)
    print(f"Citation fidelity : {pct(cited)}")
    print(f"Number fidelity   : {pct(numeric)}")
    print(f"Accepted (both)   : {pct(accepted)}")
    if latencies:
        ordered = sorted(latencies)
        print(
            f"Latency           : mean {sum(ordered) / len(ordered):.2f}s  "
            f"median {ordered[len(ordered) // 2]:.2f}s  max {ordered[-1]:.2f}s"
        )
    print("-" * 60)

    if failures:
        print(f"\n{len(failures)} rejection(s) -- each of these fell back to a correct template note:")
        for line in failures[:20]:
            print(f"  - {line}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")

    # Non-zero only when the model is unusable for this job (majority rejected). A few
    # rejections are an expected, safely-handled outcome, not a broken build.
    return 0 if accepted * 2 >= total else 2


if __name__ == "__main__":
    raise SystemExit(main())
