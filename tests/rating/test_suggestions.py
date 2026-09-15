from dataclasses import replace

from ingestion.contracts import DeliveryRef
from rating.contracts import Metric, PerformanceFlag
from rating.engine import compute_baseline
from rating.llm import TemplateNoteWriter
from rating.suggestions import flag_player, rank_flags, suggest_for_player, write_suggestions

from .fakes import FakeNoteWriter
from .helpers import match_lines, over_of

# A 12-ball scoring pattern with dots, singles and one boundary: 12 runs, 4 dots, one
# four. Used for the cohort so the baseline has a non-zero value on every metric --
# a zero baseline is skipped by the flagger (a relative deviation from zero is
# undefined), which would quietly make half these tests vacuous.
_TYPICAL_OVER_PAIR = [0, 1, 1, 0, 4, 1, 0, 1, 1, 0, 1, 2]


def _cohort():
    """One reference match: Ann faces 12 middle-over balls and 12 death-over balls off
    Carl, scoring the typical pattern in each. Gives a cohort strike rate of 100, a
    dot-ball rate of 33.33%, and a run a ball in both phases."""
    rows = over_of(_TYPICAL_OVER_PAIR, match_id="m0", striker="Ann", bowler="Carl", phase="middle") + over_of(
        _TYPICAL_OVER_PAIR, match_id="m0", start_ball=12, striker="Ann", bowler="Carl", phase="death"
    )
    return compute_baseline(match_lines(rows, "2026-01-01"), rows)


def test_a_player_sitting_on_the_cohort_baseline_is_not_flagged_at_all():
    baseline = _cohort()
    rows = over_of(_TYPICAL_OVER_PAIR, match_id="m1", striker="Bob", bowler="Dex", phase="middle")

    assert flag_player(rows, "Bob", baseline) == []


def test_a_strike_rate_below_baseline_is_flagged_as_a_concern_with_real_delivery_citations():
    baseline = _cohort()
    # 2 runs off 12: the first six balls are dots, then a single, then more dots.
    rows = over_of([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0], match_id="m1", striker="Bob", bowler="Dex")

    flags = flag_player(rows, "Bob", baseline)
    strike_rate = next(f for f in flags if f.metric is Metric.STRIKE_RATE)

    assert strike_rate.is_concern is True
    assert strike_rate.actual == 16.67
    assert strike_rate.baseline == 100.0
    assert strike_rate.delta == -83.33
    assert strike_rate.sample_size == 12
    assert strike_rate.baseline_source == baseline.source
    # Real refs off the real rows, not composed strings.
    assert all(isinstance(ref, DeliveryRef) for ref in strike_rate.citations)
    assert all(ref.match_id == "m1" for ref in strike_rate.citations)
    assert strike_rate.citation_tokens()[0] == "[clip m1 i1 o0.1]"


def test_a_dot_ball_concern_cites_the_dot_balls_and_not_the_expensive_ones():
    """The evidence direction is a property of the metric, not of whether the number
    went up or down: too *many* dot balls is evidenced by the cheapest deliveries,
    while too *many* runs conceded is evidenced by the dearest. Deriving one from the
    other gets dot-ball percentage exactly backwards.
    """
    baseline = _cohort()
    rows = over_of([0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 6], match_id="m1", striker="Bob", bowler="Dex")

    flags = flag_player(rows, "Bob", baseline)
    dots = next(f for f in flags if f.metric is Metric.DOT_BALL_PERCENT)

    assert dots.is_concern is True
    cited = {(ref.over, ref.ball) for ref in dots.citations}
    assert (0, 5) not in cited and (1, 6) not in cited, "the two sixes are not evidence of a dot-ball problem"
    assert cited == {(0, 1), (0, 2), (0, 3)}


def test_an_economy_concern_cites_the_most_expensive_deliveries():
    baseline = _cohort()
    rows = over_of([1, 1, 6, 1, 4, 1, 1, 1, 1, 1, 1, 1], match_id="m1", striker="Bob", bowler="Dex", phase="death")

    flags = flag_player(rows, "Dex", baseline)
    economy = next(f for f in flags if f.metric is Metric.ECONOMY_RATE)

    assert economy.is_concern is True
    assert economy.actual == 10.0  # 20 runs off two overs
    assert [(ref.over, ref.ball) for ref in economy.citations][:2] == [(0, 3), (0, 5)]


def test_citations_prefer_deliveries_that_actually_have_footage():
    """PRD 2.4 asks for the clip an analyst can watch. A ball-by-ball row is still
    verifiable but it isn't a clip, so a covered delivery wins over a marginally more
    extreme uncovered one while video coverage is partial."""
    baseline = _cohort()
    rows = over_of([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0], match_id="m1", striker="Bob", bowler="Dex")
    rows = [replace(d, video_available=(d.over == 1 and d.ball == 4)) for d in rows]

    flags = flag_player(rows, "Bob", baseline)
    strike_rate = next(f for f in flags if f.metric is Metric.STRIKE_RATE)

    assert strike_rate.citations_have_video is True
    assert [(ref.over, ref.ball) for ref in strike_rate.citations] == [(1, 4)]


def test_citations_fall_back_to_ball_by_ball_refs_and_say_so_when_nothing_has_footage():
    baseline = _cohort()
    rows = over_of([0] * 12, match_id="m1", striker="Bob", bowler="Dex")

    strike_rate = next(f for f in flag_player(rows, "Bob", baseline) if f.metric is Metric.STRIKE_RATE)

    assert strike_rate.citations, "an uncovered delivery is still a verifiable citation"
    assert strike_rate.citations_have_video is False


def test_a_spell_shorter_than_the_minimum_sample_is_not_flagged_at_all():
    baseline = _cohort()
    rows = over_of([0] * 4, match_id="m1", striker="Bob", bowler="Dex")

    assert flag_player(rows, "Bob", baseline) == []


def _flag(metric: Metric, relative_delta: float, sample_size: int, player: str = "Bob") -> PerformanceFlag:
    from rating.suggestions import _severity

    return PerformanceFlag(
        player=player,
        metric=metric,
        phase=None,
        actual=1.0,
        baseline=1.0,
        delta=relative_delta,
        relative_delta=relative_delta,
        is_concern=relative_delta < 0,
        severity=_severity(relative_delta, sample_size),
        sample_size=sample_size,
        baseline_source="cohort",
    )


def test_a_thin_sample_is_discounted_so_it_cannot_top_the_ranking_on_noise_alone():
    """An 80%-off-baseline reading over six balls is one bad over; a 40% one over a
    full 30 is a pattern. Without the sample discount the top of every list would be
    the smallest samples, which is where the noise lives."""
    noisy = _flag(Metric.STRIKE_RATE, -0.80, sample_size=6)
    solid = _flag(Metric.BOUNDARY_PERCENT, -0.40, sample_size=30)

    assert [f.metric for f in rank_flags([noisy, solid])] == [Metric.BOUNDARY_PERCENT, Metric.STRIKE_RATE]
    assert noisy.severity == 0.16  # 0.80 * (6/30)
    assert solid.severity == 0.4  # 0.40 * min(1, 30/30)


def test_rank_flags_breaks_every_tie_and_repeats_exactly():
    tied = [
        _flag(Metric.STRIKE_RATE, -0.50, 30, player="Zoe"),
        _flag(Metric.STRIKE_RATE, -0.50, 30, player="Abe"),
        _flag(Metric.ECONOMY_RATE, 0.50, 30, player="Abe"),
        _flag(Metric.DOT_BALL_PERCENT, -0.50, 30, player="Abe"),
    ]

    orders = [[(f.metric, f.player) for f in rank_flags(list(reversed(tied)) if n % 2 else tied)] for n in range(20)]

    assert all(order == orders[0] for order in orders), "input order must not change the ranking"
    # concerns before strengths, then metric name, then player
    assert orders[0] == [
        (Metric.DOT_BALL_PERCENT, "Abe"),
        (Metric.STRIKE_RATE, "Abe"),
        (Metric.STRIKE_RATE, "Zoe"),
        (Metric.ECONOMY_RATE, "Abe"),
    ]


def test_suggestions_take_their_citations_from_the_flag_and_never_from_the_note_text():
    """The machine-readable citation list is layer 1's output. Even a note that passed
    validation is prose, and a UI turning citations into clip links must not be parsing
    sentences to find them."""
    baseline = _cohort()
    rows = over_of([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0], match_id="m1", striker="Bob", bowler="Dex")
    flags = flag_player(rows, "Bob", baseline)
    writer = FakeNoteWriter(["a note that mentions no citation whatsoever"])

    suggestions = write_suggestions(flags, writer=writer)

    assert suggestions, "the flags should have produced suggestions"
    for suggestion in suggestions:
        assert suggestion.citations == suggestion.flag.citations
        assert suggestion.citations, "every suggestion carries its evidence"


def test_the_note_writer_only_ever_receives_already_computed_flags():
    baseline = _cohort()
    rows = over_of([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0], match_id="m1", striker="Bob", bowler="Dex")
    flags = flag_player(rows, "Bob", baseline)
    writer = FakeNoteWriter(["note"])

    write_suggestions(flags, writer=writer)

    assert [f.metric for f in writer.seen] == [f.metric for f in rank_flags(flags)]
    assert all(f.baseline_source == baseline.source for f in writer.seen)


def test_suggestions_start_pending_and_are_ranked_from_one():
    baseline = _cohort()
    rows = over_of([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0], match_id="m1", striker="Bob", bowler="Dex")

    suggestions = suggest_for_player(rows, "Bob", baseline, writer=FakeNoteWriter(["note"]))

    assert [s.rank for s in suggestions] == list(range(1, len(suggestions) + 1))
    assert {s.status for s in suggestions} == {"pending"}


def test_suggest_for_player_keeps_only_concerns_by_default_but_can_return_strengths():
    baseline = _cohort()
    # 36 off 12 is well above the cohort: a strength on strike rate and boundary rate.
    rows = over_of([6] * 12, match_id="m1", striker="Bob", bowler="Dex")

    concerns = suggest_for_player(rows, "Bob", baseline, writer=FakeNoteWriter(["note"]))
    everything = suggest_for_player(rows, "Bob", baseline, writer=FakeNoteWriter(["note"]), concerns_only=False)

    assert all(s.flag.is_concern for s in concerns)
    assert any(not s.flag.is_concern for s in everything)
    assert len(everything) > len(concerns)


def test_write_suggestions_defaults_to_the_deterministic_template_writer():
    """A caller who hasn't thought about the LLM gets correct, cited output rather than
    a silent dependency on a server being up."""
    baseline = _cohort()
    rows = over_of([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0], match_id="m1", striker="Bob", bowler="Dex")

    suggestions = write_suggestions(flag_player(rows, "Bob", baseline))

    assert {s.note_source for s in suggestions} == {"template"}
    for suggestion in suggestions:
        for token in suggestion.citation_tokens():
            assert token in suggestion.body


def test_limit_caps_after_ranking_so_the_most_severe_findings_survive():
    baseline = _cohort()
    rows = over_of([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0], match_id="m1", striker="Bob", bowler="Dex")
    flags = flag_player(rows, "Bob", baseline)

    capped = write_suggestions(flags, writer=TemplateNoteWriter(), limit=1)

    assert len(capped) == 1
    assert capped[0].flag.metric is rank_flags(flags)[0].metric
