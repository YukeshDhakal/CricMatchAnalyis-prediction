"""Coaching suggestions (PRD 2.2 stage 6 / PRD 3.2-3.3): flag -> rank -> write.

Three layers, in strict order, and the order is the point:

1. **Flagging** (`flag_player`) -- deterministic arithmetic over real `FusedDelivery`
   rows against a `RatingBaseline`. Produces `PerformanceFlag`s carrying the metric,
   the measured value, the baseline, the delta and real `DeliveryRef` citations.
   *No LLM is involved and none can be.*
2. **Ranking** (`rank_flags`) -- a documented deterministic scoring formula. This is
   the MVP stand-in for PRD 3.2's "learned ranker"; see that function for why a model
   isn't trained here yet and what would have to exist first.
3. **Note-writing** (`write_suggestions`) -- hands each ranked flag to a
   `rating.llm.NoteWriter` for phrasing only. The writer receives the flag's already
   computed fields and the already chosen citations; it cannot originate either.

That split is how PRD 2.4's "no black-box verdicts -- every AI-derived suggestion must
cite its underlying clip and stat so an analyst can verify it" is actually enforced
rather than merely intended: every number and every citation in a suggestion was
produced by layer 1, and `rating.llm` rejects a generated note that doesn't carry them
through unchanged.

Scoring conventions match `fusion.pipeline` and `player_reports.stats` exactly.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

from fusion.contracts import FusedDelivery
from ingestion.contracts import DeliveryRef

from video_engine.calibration import STUMP_HALF_WIDTH_M

from .contracts import (
    METRIC_SPECS,
    PITCH_METRIC_BANDS,
    STUMP_LINE_TOLERANCE_M,
    CoachingSuggestion,
    Metric,
    PerformanceFlag,
    RatingBaseline,
)
from .engine import balls_faced, legal_balls_bowled
from .llm import NoteWriter, TemplateNoteWriter

__all__ = [
    "FLAG_RELATIVE_THRESHOLD",
    "MIN_BALLS_FOR_FLAG",
    "MAX_CITATIONS_PER_FLAG",
    "SEVERITY_FULL_CONFIDENCE_BALLS",
    "flag_player",
    "rank_flags",
    "write_suggestions",
    "suggest_for_player",
]

# How far off baseline a metric has to be before it's worth an analyst's attention.
# 15% is a judgement call, not a fitted value: tight enough that a real slump surfaces,
# loose enough that normal match-to-match noise doesn't produce a flag for every player
# on every metric. Exposed as a parameter on `flag_player` precisely because the right
# number depends on the competition and nobody has measured it yet.
FLAG_RELATIVE_THRESHOLD = 0.15

# Below this many balls, a rate is noise and a flag on it is a false alarm dressed up
# with a citation -- worse than no flag, because the citation makes it look checked.
MIN_BALLS_FOR_FLAG = 6

# Citations per flag. Three is enough for an analyst to see a pattern rather than one
# unlucky ball, and few enough that the note stays readable and the LLM has a short,
# exactly-copyable citation list (see `rating.llm.build_prompt`).
MAX_CITATIONS_PER_FLAG = 3

# The sample size at which `severity` stops being discounted -- roughly five overs.
SEVERITY_FULL_CONFIDENCE_BALLS = 30

_UNCHARGED_EXTRAS = ("byes", "legbyes")


def _runs_charged(delivery: FusedDelivery) -> int:
    if delivery.extras_type in _UNCHARGED_EXTRAS:
        return delivery.runs_total - delivery.runs_extras
    return delivery.runs_total


def _severity(relative_delta: float, sample_size: int) -> float:
    """The ranking score: relative deviation, discounted for a thin sample.

    `abs(relative_delta) * min(1, sample_size / SEVERITY_FULL_CONFIDENCE_BALLS)`.

    Relative rather than absolute so metrics in different units compete fairly -- 2.0
    runs per over and 2.0 runs per 100 balls are not the same size of problem. The
    sample discount is what stops a 6-ball cameo 80% off baseline from outranking a
    120-ball slump 25% off it; without it the top of every list would be the
    smallest samples, which is where the noise lives.
    """
    confidence = min(1.0, sample_size / SEVERITY_FULL_CONFIDENCE_BALLS)
    return round(abs(relative_delta) * confidence, 6)


def _runs_off_bat(delivery: FusedDelivery) -> int:
    return delivery.runs_batter


def _pick_citations(
    rows: Sequence[FusedDelivery],
    high_runs_first: bool,
    runs_of: Callable[[FusedDelivery], int],
) -> tuple[tuple[DeliveryRef, ...], bool]:
    """The deliveries to cite for a flag, chosen deterministically.

    Deliveries with footage are preferred over those without, because PRD 2.4 asks for
    the *clip* an analyst can watch and a ball-by-ball row alone isn't one. The
    trade-off is real and deliberate: the cited balls are then the most extreme among
    the *covered* deliveries rather than among all of them, which is the right call
    while video coverage is partial (`BroadcastFeedAdapter` isn't implemented, so most
    matches have no footage at all). When nothing in the set has video, it falls back
    to plain `DeliveryRef`s and reports that through the second return value, so the
    caller can say so rather than implying a clip exists.

    Ordering is (runs, over, ball, innings) with runs descending when
    `high_runs_first` -- fully specified, so the same rows always cite the same balls
    no matter what order they arrived in.
    """
    if not rows:
        return (), False

    with_video = [d for d in rows if d.video_available]
    candidates = with_video or list(rows)
    has_video = bool(with_video)

    sign = -1 if high_runs_first else 1
    ordered = sorted(candidates, key=lambda d: (sign * runs_of(d), d.over, d.ball, d.innings))
    return tuple(d.delivery for d in ordered[:MAX_CITATIONS_PER_FLAG]), has_video


def _make_flag(
    player: str,
    metric: Metric,
    phase: Optional[str],
    actual: float,
    baseline_value: float,
    sample_size: int,
    baseline_source: str,
    rows: Sequence[FusedDelivery],
    runs_of: Callable[[FusedDelivery], int],
    threshold: float,
) -> Optional[PerformanceFlag]:
    """One flag, or `None` if the metric is within `threshold` of baseline.

    A zero or negative baseline yields `None` rather than a flag: a relative deviation
    from zero is undefined, and reporting one as "infinitely off baseline" would put
    a meaningless finding at the top of every ranking.
    """
    if baseline_value <= 0:
        return None
    delta = actual - baseline_value
    # Rounded before severity is derived from it, so an analyst recomputing the ranking
    # score by hand from the flag's own published fields gets the flag's own severity.
    # A severity computed from more precision than the flag exposes is unauditable for
    # the sake of digits nobody reads.
    relative_delta = round(delta / baseline_value, 4)
    if abs(relative_delta) < threshold:
        return None

    spec = METRIC_SPECS[metric]
    is_concern = (delta < 0) if spec.higher_is_better else (delta > 0)
    # Cite the deliveries that produced the deviation, whichever way it went: if the
    # metric came in above baseline, that's the balls that push it up, and vice versa.
    # Which balls those are is a property of the metric, not of its direction of
    # goodness -- see `MetricSpec.evidence_is_high_runs`.
    cite_high_runs = spec.evidence_is_high_runs if delta > 0 else not spec.evidence_is_high_runs
    citations, has_video = _pick_citations(rows, high_runs_first=cite_high_runs, runs_of=runs_of)

    return PerformanceFlag(
        player=player,
        metric=metric,
        phase=phase,
        actual=round(actual, 2),
        baseline=round(baseline_value, 2),
        delta=round(delta, 2),
        relative_delta=relative_delta,
        is_concern=is_concern,
        severity=_severity(relative_delta, sample_size),
        sample_size=sample_size,
        baseline_source=baseline_source,
        citations=citations,
        citations_have_video=has_video,
    )


def _pitch_flags(
    bowled: Sequence[FusedDelivery],
    player: str,
    baseline: RatingBaseline,
    threshold: float,
    min_balls: int,
) -> list[Optional[PerformanceFlag]]:
    """Pitch-length/line flags for a bowler, or nothing when there's no geometry.

    **This emits nothing today and that is correct, not a stub.** Both preconditions
    fail: `baseline.pitch` is `None` because no cohort has calibrated geometry, and no
    delivery passes `has_pitch_geometry()` because no ball or stumps detector exists to
    produce a bounce point (see `video_engine.calibration`). The arithmetic below is
    real and is exercised by tests that construct rows carrying geometry by hand, the
    same way `tests/rating/helpers.py` constructs every other row.

    The guard is deliberately *two* conditions rather than one. A cohort baseline can
    carry geometry while this particular player's window doesn't (their matches weren't
    filmed), and flagging a player's "0% good length" against a real cohort rate would
    turn missing footage into a damning finding -- the same trap
    `engine.cohort_pitch_baseline` documents for its denominator, one layer down.

    Rates are over the player's legal deliveries **that carry geometry**, so
    `sample_size` is that covered count and not their full workload. That matters for
    ranking: `_severity` discounts by sample size, so a flag resting on eight filmed
    deliveries out of forty bowled is correctly outranked by one resting on forty.
    """
    pitch = baseline.pitch
    if pitch is None:
        return []

    rows = [d for d in bowled if d.has_pitch_geometry()]
    if len(rows) < min_balls:
        return []

    flags: list[Optional[PerformanceFlag]] = []
    for metric, baseline_value in (
        (Metric.GOOD_LENGTH_PERCENT, pitch.good_length_percent),
        (Metric.SHORT_BALL_PERCENT, pitch.short_ball_percent),
        (Metric.FULL_BALL_PERCENT, pitch.full_ball_percent),
    ):
        bands = PITCH_METRIC_BANDS[metric]
        in_band = [d for d in rows if d.pitch_length.value in bands]
        flags.append(
            _make_flag(
                player, metric, None, 100 * len(in_band) / len(rows), baseline_value,
                len(rows), baseline.source, rows, _runs_charged, threshold,
            )
        )

    # Line is scored over the rows that have one. A full toss has a length band and no
    # bounce point, so it has no line and is not evidence either way about line control.
    with_line = [d for d in rows if d.pitch_line_m is not None]
    if len(with_line) >= min_balls:
        corridor = STUMP_HALF_WIDTH_M + STUMP_LINE_TOLERANCE_M
        on_line = [d for d in with_line if abs(d.pitch_line_m) <= corridor]
        flags.append(
            _make_flag(
                player, Metric.STUMP_LINE_PERCENT, None,
                100 * len(on_line) / len(with_line), pitch.stump_line_percent,
                len(with_line), baseline.source, with_line, _runs_charged, threshold,
            )
        )
    return flags


def flag_player(
    deliveries: Sequence[FusedDelivery],
    player: str,
    baseline: RatingBaseline,
    threshold: float = FLAG_RELATIVE_THRESHOLD,
    min_balls: int = MIN_BALLS_FOR_FLAG,
) -> list[PerformanceFlag]:
    """Every metric where `player` deviates from `baseline` by more than `threshold`.

    PRD 3.2's rule/threshold layer. Pure arithmetic over `deliveries` -- no model, no
    heuristics over pixel data, nothing that can't be recomputed by hand from the
    ball-by-ball table.

    Covers both whole-innings metrics and per-phase ones. A per-phase flag is the more
    actionable of the two ("his death-overs economy is the problem", not "his economy
    is the problem"), which is why phase metrics are emitted alongside the overall
    ones rather than instead of them -- both are true, and ranking decides which the
    analyst sees first.

    Returns flags in a deterministic order (metric declaration order, then phase) so
    that two identical inputs produce two identical lists before ranking touches them.
    Strengths are included as well as concerns; `suggest_for_player` is where the
    concerns-only filter lives, because a flag is a finding and a suggestion is advice.
    """
    flags: list[Optional[PerformanceFlag]] = []
    faced = balls_faced(deliveries, player)
    bowled = legal_balls_bowled(deliveries, player)

    batter_runs = _runs_off_bat
    bowler_runs = _runs_charged

    if len(faced) >= min_balls:
        runs = sum(d.runs_batter for d in faced)
        flags.append(
            _make_flag(
                player, Metric.STRIKE_RATE, None, 100 * runs / len(faced),
                baseline.batting_strike_rate, len(faced), baseline.source, faced, batter_runs, threshold,
            )
        )
        if baseline.dot_ball_percent is not None:
            dots = [d for d in faced if d.runs_total == 0]
            flags.append(
                _make_flag(
                    player, Metric.DOT_BALL_PERCENT, None, 100 * len(dots) / len(faced),
                    baseline.dot_ball_percent, len(faced), baseline.source, faced, batter_runs, threshold,
                )
            )
        if baseline.boundary_percent is not None:
            boundaries = [d for d in faced if d.runs_batter in (4, 6)]
            flags.append(
                _make_flag(
                    player, Metric.BOUNDARY_PERCENT, None, 100 * len(boundaries) / len(faced),
                    baseline.boundary_percent, len(faced), baseline.source, faced, batter_runs, threshold,
                )
            )

    if len(bowled) >= min_balls:
        conceded = sum(_runs_charged(d) for d in bowled)
        flags.append(
            _make_flag(
                player, Metric.ECONOMY_RATE, None, conceded / (len(bowled) / 6),
                baseline.economy_rate, len(bowled), baseline.source, bowled, bowler_runs, threshold,
            )
        )

    flags.extend(_pitch_flags(bowled, player, baseline, threshold, min_balls))

    for phase_baseline in baseline.phases:
        phase = phase_baseline.phase
        phase_faced = [d for d in faced if d.phase == phase]
        if len(phase_faced) >= min_balls and phase_baseline.runs_per_ball > 0:
            runs = sum(d.runs_batter for d in phase_faced)
            flags.append(
                _make_flag(
                    player, Metric.PHASE_STRIKE_RATE, phase, 100 * runs / len(phase_faced),
                    100 * phase_baseline.runs_per_ball, len(phase_faced), baseline.source,
                    phase_faced, batter_runs, threshold,
                )
            )
        phase_bowled = [d for d in bowled if d.phase == phase]
        if len(phase_bowled) >= min_balls and phase_baseline.runs_conceded_per_ball > 0:
            conceded = sum(_runs_charged(d) for d in phase_bowled)
            flags.append(
                _make_flag(
                    player, Metric.PHASE_ECONOMY_RATE, phase, conceded / (len(phase_bowled) / 6),
                    6 * phase_baseline.runs_conceded_per_ball, len(phase_bowled), baseline.source,
                    phase_bowled, bowler_runs, threshold,
                )
            )

    return [f for f in flags if f is not None]


def rank_flags(flags: Sequence[PerformanceFlag]) -> list[PerformanceFlag]:
    """`flags` ordered most-worth-reading first.

    **This is the MVP stand-in for PRD 3.2's "learned ranker", and is deliberately not
    a model.** Training one needs labelled data about which suggestions analysts
    actually found useful, and that data doesn't exist yet -- it is exactly what the
    `status` field on `CoachingSuggestion` ("pending"/"accepted"/"edited"/"rejected")
    is there to accumulate once a review UI exists. Fitting a ranker before then would
    mean inventing the labels, which would make the ordering look learned while being
    no better than this formula and considerably harder to explain to the analyst
    reading it.

    The ordering is `severity` descending (see `_severity`: relative deviation,
    discounted for thin samples), with ties broken by concerns before strengths, then
    metric name, then phase, then player. Every tie-break is total and data-independent,
    so repeated calls over identical input can't reorder -- the same discipline
    `player_reports.last_n_match_ids` applies to match windows, for the same reason.

    Swapping in a real ranker later means replacing this function and nothing else:
    nothing upstream or downstream depends on how the order was arrived at.
    """
    return sorted(
        flags,
        key=lambda f: (-f.severity, not f.is_concern, f.metric.value, f.phase or "", f.player),
    )


def write_suggestions(
    flags: Sequence[PerformanceFlag],
    writer: Optional[NoteWriter] = None,
    limit: Optional[int] = None,
) -> list[CoachingSuggestion]:
    """Rank `flags` and phrase each one into a `CoachingSuggestion`.

    `writer` defaults to `TemplateNoteWriter` rather than to `OllamaClient`: the
    deterministic writer is the one that always works, so a caller who hasn't thought
    about the LLM gets correct, cited output instead of a silent dependency on a
    server being up. Passing `OllamaClient()` opts into the nicer phrasing, and that
    client falls back to this same template on any failure.

    `limit` caps the list *after* ranking, so capping keeps the most severe findings
    rather than an arbitrary subset.

    Citations are copied onto the suggestion from the flag, never from the note text:
    even a note that passed `rating.llm`'s validation is prose, and the machine-readable
    citation list a UI turns into clip links must come from layer 1.
    """
    note_writer = writer or TemplateNoteWriter()
    ranked = rank_flags(flags)
    if limit is not None:
        ranked = ranked[:limit]

    suggestions: list[CoachingSuggestion] = []
    for position, flag in enumerate(ranked, start=1):
        draft = note_writer.write(flag)
        suggestions.append(
            CoachingSuggestion(
                player=flag.player,
                title=draft.title,
                body=draft.body,
                flag=flag,
                citations=flag.citations,
                rank=position,
                note_source=draft.source,
                status="pending",  # PRD 3.3's human review hasn't happened yet, by definition
            )
        )
    return suggestions


def suggest_for_player(
    deliveries: Sequence[FusedDelivery],
    player: str,
    baseline: RatingBaseline,
    writer: Optional[NoteWriter] = None,
    threshold: float = FLAG_RELATIVE_THRESHOLD,
    min_balls: int = MIN_BALLS_FOR_FLAG,
    limit: Optional[int] = None,
    concerns_only: bool = True,
) -> list[CoachingSuggestion]:
    """The whole stage-6 pipeline for one player: flag -> rank -> write.

    `concerns_only` defaults True because this function's output is *advice*, and a
    note about something already going well isn't advice. Set it False to keep the
    strengths too -- they are computed either way (`flag_player` returns both), so
    nothing is recomputed by asking for them.
    """
    flags = flag_player(deliveries, player, baseline, threshold=threshold, min_balls=min_balls)
    if concerns_only:
        flags = [f for f in flags if f.is_concern]
    return write_suggestions(flags, writer=writer, limit=limit)
