import pytest

from ingestion.contracts import DeliveryRef
from rating.contracts import Metric, PerformanceFlag, citation_token
from rating.llm import (
    ALTERNATIVE_OLLAMA_MODELS,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_BASE_URL_ENV,
    OLLAMA_MODEL_ENV,
    OllamaClient,
    TemplateNoteWriter,
    build_prompt,
    missing_citations,
    unsupported_numbers,
)

from .fakes import FakeOllamaTransport

_REF = DeliveryRef(match_id="m1", innings=1, over=18, ball=3)
_TOKEN = citation_token(_REF)


def _flag(**overrides) -> PerformanceFlag:
    base = dict(
        player="Bob",
        metric=Metric.ECONOMY_RATE,
        phase="death",
        actual=10.0,
        baseline=6.0,
        delta=4.0,
        relative_delta=0.6667,
        is_concern=True,
        severity=0.6667,
        sample_size=12,
        baseline_source="cohort mean over 3 match(es)",
        citations=(_REF,),
        citations_have_video=True,
    )
    base.update(overrides)
    return PerformanceFlag(**base)


def test_the_citation_token_is_a_single_copyable_string_not_prose():
    assert _TOKEN == "[clip m1 i1 o18.3]"


def test_the_template_writer_cites_every_delivery_and_invents_no_number():
    flag = _flag(citations=(_REF, DeliveryRef("m1", 1, 19, 1)))

    draft = TemplateNoteWriter().write(flag)

    assert draft.source == "template"
    assert missing_citations(draft.body, flag) == ()
    assert unsupported_numbers(draft.body, flag) == ()


def test_the_template_writer_still_cites_a_metric_it_has_no_curated_wording_for():
    """A new `Metric` must degrade to plain phrasing, never to an exception in the
    middle of a batch or to an uncited note."""
    flag = _flag(metric=Metric.PHASE_ECONOMY_RATE)

    draft = TemplateNoteWriter().write(flag)

    assert missing_citations(draft.body, flag) == ()
    assert "economy rate" in draft.body


def test_the_prompt_hands_the_model_the_citation_to_copy_rather_than_describing_it():
    prompt = build_prompt(_flag())

    assert _TOKEN in prompt
    assert "character for character" in prompt
    assert "10.0" in prompt and "6.0" in prompt


def test_a_note_that_echoes_every_citation_is_accepted_as_model_written():
    flag = _flag()
    transport = FakeOllamaTransport(
        [f"Bob conceded 10.0 an over at the death, 4.0 above the 6.0 baseline. Review it: {_TOKEN}"]
    )
    client = OllamaClient(transport=transport)

    draft = client.write(flag)

    assert draft.source == "llm"
    assert _TOKEN in draft.body
    assert transport.requests[0]["model"] == DEFAULT_OLLAMA_MODEL
    assert transport.requests[0]["stream"] is False


def test_the_title_is_never_model_written_even_when_the_body_is():
    flag = _flag()
    client = OllamaClient(transport=FakeOllamaTransport([f"Anything at all. {_TOKEN}"]))

    assert client.write(flag).title == TemplateNoteWriter().title_for(flag)


def test_a_note_that_drops_a_citation_is_rejected_and_the_template_answers_instead():
    flag = _flag()
    transport = FakeOllamaTransport(["Bob went at 10.0 an over at the death. Tighten the yorkers."] * 2)
    client = OllamaClient(transport=transport)

    draft = client.write(flag)

    assert draft.source == "template"
    assert _TOKEN in draft.body, "the fallback is still fully cited"
    assert "not echoed verbatim" in client.last_rejection


def test_a_reworded_citation_counts_as_dropped():
    """Anything fuzzier than an exact substring would accept a citation the model
    retyped, and a retyped citation is one it could have retyped wrong."""
    flag = _flag()
    client = OllamaClient(transport=FakeOllamaTransport(["Over 18, ball 3 of innings 1 in match m1."] * 2))

    assert client.write(flag).source == "template"


def test_a_number_the_flag_never_supplied_is_rejected():
    flag = _flag()
    transport = FakeOllamaTransport([f"Bob went at 10.0 an over, up from his career 7.5 average. {_TOKEN}"] * 2)
    client = OllamaClient(transport=transport)

    draft = client.write(flag)

    assert draft.source == "template"
    assert "7.5" in client.last_rejection


def test_rounding_an_input_number_is_rephrasing_and_stays_accepted():
    flag = _flag(actual=10.42)
    client = OllamaClient(transport=FakeOllamaTransport([f"Bob went at 10.4 an over at the death. {_TOKEN}"]))

    assert client.write(flag).source == "llm"


def test_the_number_check_can_be_turned_off_without_weakening_the_citation_check():
    flag = _flag()
    loose = OllamaClient(
        transport=FakeOllamaTransport([f"Bob's 7.5 career average is the context here. {_TOKEN}"] * 2),
        reject_unsupported_numbers=False,
    )
    still_strict = OllamaClient(
        transport=FakeOllamaTransport(["Bob's 7.5 career average, no citation."] * 2),
        reject_unsupported_numbers=False,
    )

    assert loose.write(flag).source == "llm"
    assert still_strict.write(flag).source == "template"


def test_an_unreachable_server_falls_back_instead_of_raising():
    flag = _flag()
    transport = FakeOllamaTransport([ConnectionError("connection refused"), ConnectionError("connection refused")])
    client = OllamaClient(transport=transport)

    draft = client.write(flag)

    assert draft.source == "template"
    assert missing_citations(draft.body, flag) == ()
    assert "ConnectionError" in client.last_rejection


def test_a_malformed_payload_falls_back_instead_of_raising():
    client = OllamaClient(transport=FakeOllamaTransport([{"unexpected": "shape"}] * 2))

    assert client.write(_flag()).source == "template"


def test_one_bad_attempt_is_retried_before_giving_up_on_the_model():
    flag = _flag()
    transport = FakeOllamaTransport(
        ["no citation here at all", f"Bob went at 10.0 an over at the death. {_TOKEN}"]
    )
    client = OllamaClient(transport=transport)

    draft = client.write(flag)

    assert draft.source == "llm"
    assert len(transport.requests) == 2


def test_an_empty_response_is_rejected():
    client = OllamaClient(transport=FakeOllamaTransport(["   ", "  "]))

    assert client.write(_flag()).source == "template"


def test_the_default_model_is_phi35_with_llama_documented_as_the_alternative():
    """Chosen for instruction adherence in the <=4GB-VRAM class and an MIT licence,
    not because it happened to be pulled -- see `rating.llm`'s module docstring."""
    assert DEFAULT_OLLAMA_MODEL == "phi3.5"
    assert "llama3.2:3b" in ALTERNATIVE_OLLAMA_MODELS
    assert "llama3.2:1b" in ALTERNATIVE_OLLAMA_MODELS, "the low-resource swap-in must stay supported"


def test_the_endpoint_and_model_are_configurable_and_never_hardcoded(monkeypatch):
    monkeypatch.setenv(OLLAMA_BASE_URL_ENV, "http://gpu-box:11434/")
    monkeypatch.setenv(OLLAMA_MODEL_ENV, "llama3.2:1b")

    from_env = OllamaClient()
    explicit = OllamaClient(base_url="http://other:1234", model="qwen2.5:3b")

    assert from_env.base_url == "http://gpu-box:11434"  # trailing slash normalised
    assert from_env.model == "llama3.2:1b"
    assert from_env.generate_url == "http://gpu-box:11434/api/generate"
    assert explicit.base_url == "http://other:1234", "an explicit argument beats the env var"
    assert explicit.model == "qwen2.5:3b"


def test_the_defaults_apply_when_nothing_is_configured(monkeypatch):
    monkeypatch.delenv(OLLAMA_BASE_URL_ENV, raising=False)
    monkeypatch.delenv(OLLAMA_MODEL_ENV, raising=False)

    client = OllamaClient()

    assert client.base_url == DEFAULT_OLLAMA_BASE_URL
    assert client.model == DEFAULT_OLLAMA_MODEL


def test_the_client_works_without_an_http_library_installed(monkeypatch):
    """`requests` is imported lazily so that importing `rating` -- and running this
    whole suite -- never depends on an HTTP library being present. An ImportError has
    to degrade exactly like a connection error."""
    import builtins

    real_import = builtins.__import__

    def no_requests(name, *args, **kwargs):
        if name == "requests":
            raise ImportError("No module named 'requests'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_requests)
    flag = _flag()

    draft = OllamaClient().write(flag)

    assert draft.source == "template"
    assert missing_citations(draft.body, flag) == ()


def test_unsupported_numbers_ignores_digits_inside_a_citation_token():
    flag = _flag()

    assert unsupported_numbers(f"See {_TOKEN}", flag) == ()


def test_the_metrics_own_unit_is_not_mistaken_for_an_invented_statistic():
    """"Runs per 100 balls" is the name of the scale, not a claim about the player.
    Found by running the live eval: it was rejecting otherwise-perfect notes."""
    flag = _flag(metric=Metric.STRIKE_RATE, actual=92.5, baseline=131.0, delta=-38.5, relative_delta=-0.2939)

    assert unsupported_numbers(f"Bob struck at 92.5 runs per 100 balls. {_TOKEN}", flag) == ()
    assert unsupported_numbers(f"Bob struck at 92.5, down from a career 77.0. {_TOKEN}", flag) == ("77.0",)


@pytest.mark.parametrize("body", ["", "   \n  "])
def test_missing_citations_reports_every_uncited_token(body):
    flag = _flag(citations=(_REF, DeliveryRef("m1", 1, 19, 1)))

    assert len(missing_citations(body, flag)) == 2
