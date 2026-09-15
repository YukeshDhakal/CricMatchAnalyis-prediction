"""Opt-in fidelity check against a *live* Ollama server.

Everything else under `tests/rating/` runs against fakes and must never need a model.
This module is the exception, and it skips itself unless both of these are true:

  * `THIRD_UMPIRE_LLM_EVAL=1` is set -- so a normal `pytest` run, CI included, never
    waits on a local server that may not exist; and
  * the configured Ollama server answers, with the configured model pulled.

It exists because the citation contract is the one thing about this integration that
borrowed benchmarks cannot speak to: MMLU and GSM8K say nothing about whether a model
will reproduce `[clip m1 i1 o18.3]` character for character. The same measurement with
a fuller report lives in `scripts/eval_llm_notewriter.py`; this is the subset worth
pinning as a test so a model or prompt change that destroys citation fidelity fails
loudly rather than quietly degrading every note to a template.

Run it with:  THIRD_UMPIRE_LLM_EVAL=1 pytest tests/rating/test_llm_citation_fidelity.py -v
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from rating.llm import OllamaClient, missing_citations, unsupported_numbers

# Thresholds, not aspirations. Below these the model is falling back to templates often
# enough that it isn't earning its latency -- which is a quality regression to
# investigate, never a correctness failure: a rejected generation still yields a
# correct, fully cited template note.
MIN_CITATION_FIDELITY = 0.85
MIN_NUMBER_FIDELITY = 0.70


def _eval_flags():
    """The eval set from `scripts/eval_llm_notewriter.py`, so the script and this test
    can never drift apart.

    `scripts/` isn't a package, so it gets added to `sys.path` -- here, inside the
    opt-in path, rather than at import time, so a skipped run doesn't reorder module
    resolution for the rest of the session.
    """
    scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from eval_llm_notewriter import eval_flags  # noqa: PLC0415 -- only on the opt-in path

    return eval_flags()


def _live_client() -> OllamaClient:
    if os.environ.get("THIRD_UMPIRE_LLM_EVAL") != "1":
        pytest.skip("set THIRD_UMPIRE_LLM_EVAL=1 to run the live-Ollama fidelity check")

    client = OllamaClient(timeout_s=60.0, max_attempts=1)
    probe = client._generate(_eval_flags()[0])  # noqa: SLF001
    if probe is None:
        pytest.skip(f"Ollama unreachable at {client.generate_url}: {client.last_rejection}")
    return client


@pytest.fixture(scope="module")
def fidelity() -> dict:
    client = _live_client()
    flags = _eval_flags()
    cited = numeric = generated = 0
    latencies: list[float] = []

    for flag in flags:
        started = time.perf_counter()
        text = client._generate(flag)  # noqa: SLF001
        latencies.append(time.perf_counter() - started)
        if text is None:
            continue
        generated += 1
        cited += not missing_citations(text, flag)
        numeric += not unsupported_numbers(text, flag)

    return {
        "model": client.model,
        "flags": len(flags),
        "generated": generated,
        "cited": cited,
        "numeric": numeric,
        "latencies": latencies,
    }


def test_every_flag_produced_a_generation(fidelity):
    assert fidelity["generated"] == fidelity["flags"], "the server answered the probe but not every case"


def test_citations_survive_generation_verbatim(fidelity):
    rate = fidelity["cited"] / fidelity["generated"]
    assert rate >= MIN_CITATION_FIDELITY, (
        f"{fidelity['model']} echoed every citation in only {rate:.0%} of notes "
        f"({fidelity['cited']}/{fidelity['generated']}); below {MIN_CITATION_FIDELITY:.0%} most "
        f"suggestions fall back to templates -- try another model from "
        f"rating.llm.ALTERNATIVE_OLLAMA_MODELS or revisit the prompt"
    )


def test_generated_notes_do_not_introduce_numbers_the_flag_never_supplied(fidelity):
    rate = fidelity["numeric"] / fidelity["generated"]
    assert rate >= MIN_NUMBER_FIDELITY, (
        f"{fidelity['model']} invented a number in {1 - rate:.0%} of notes "
        f"({fidelity['generated'] - fidelity['numeric']}/{fidelity['generated']})"
    )


def test_latency_is_reported_not_asserted(fidelity, capsys):
    """Logged for the record. No threshold: this is one developer laptop, and a slow
    note is a slow note, not a wrong one."""
    latencies = sorted(fidelity["latencies"])
    with capsys.disabled():
        print(
            f"\n{fidelity['model']}: mean {sum(latencies) / len(latencies):.2f}s, "
            f"median {latencies[len(latencies) // 2]:.2f}s, max {latencies[-1]:.2f}s "
            f"over {len(latencies)} generations"
        )
    assert latencies
