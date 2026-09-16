"""Composite player rating (PRD 2.2 stage 5 / PRD 3.1): four weighted pillars over a
rolling window of `fusion.PlayerMatchStats`, with per-delivery `FusedDelivery` rows
supplying the phase breakdown the Game impact pillar needs.

Scoring conventions (what counts as a ball faced/bowled, which wickets credit the
bowler, which extras aren't charged to them) mirror `fusion.pipeline` and
`player_reports.stats` exactly. A rating and the stat line it was computed from must
never disagree about what those mean.

Two things are deliberately *not* here, both because there is no honest input for
them today rather than because they were forgotten:

* **Technique never gets a score.** `_technique_score` returns `None`
  unconditionally; see its docstring and `contracts.PlayerRating`.
* **The baseline is a cohort/season mean, not PRD 3.1's matchup baseline.** See
  `compute_baseline` and `contracts.RatingBaseline`.
"""
from __future__ import annotations

from statistics import pstdev
from typing import Iterable, Mapping, Optional, Sequence

import pandas as pd

from fusion.contracts import FusedDelivery, PlayerMatchStats
from video_engine.calibration import STUMP_HALF_WIDTH_M

from .contracts import (
    DEFAULT_PHASE_WEIGHT,
    PHASE_IMPACT_WEIGHTS,
    PITCH_METRIC_BANDS,
    ROLE_PROFILES,
    STUMP_LINE_TOLERANCE_M,
    Metric,
    PhaseBaseline,
    Pillar,
    PillarScores,
    PitchBaseline,
    PlayerRating,
    PlayerRole,
    RatingBaseline,
    RoleProfile,
)

__all__ = [
    "balls_faced",
    "baseline_from_deliveries",
    "cohort_pitch_baseline",
    "compute_baseline",
    "legal_balls_bowled",
    "pillars_as_dict",
    "rate_player",
    "rate_players",
    "RATIO_SCORE_SPREAD",
    "BASELINE_SCORE",
]

# The 0-100 pillar scale's anchor: exactly-at-baseline scores 50, so a composite of 50
# means "an average player in this cohort", not "half marks". See `_ratio_score`.
BASELINE_SCORE = 50.0

# How far off baseline saturates the pillar scale. 0.5 == a player 50% better than the
# cohort on a metric scores 100 on it, 50% worse scores 0.
#
# Picked, not fitted -- like `PHASE_IMPACT_WEIGHTS`, there is no labelled data in this
# repo to fit it against. It is exposed as a module constant (rather than buried in
# `_ratio_score`) so a later calibration pass has one number to change, and so a test
# can state the mapping it depends on instead of hard-coding magic outputs.
RATIO_SCORE_SPREAD = 0.5

# A pillar needs at least this many deliveries behind it before it's scored at all.
# Below it, a single boundary swings a rate far enough that the "score" is noise --
# and a noisy pillar that gets weight is worse than an absent one that gets
# redistributed, because nothing downstream can tell it was noise.
MIN_BALLS_FOR_PILLAR = 6

# Consistency is a spread measure, so it needs at least two innings to have a spread.
MIN_INNINGS_FOR_CONSISTENCY = 2

_WIDES = ("wides",)
_ILLEGAL_DELIVERIES = ("wides", "noballs")
_UNCHARGED_EXTRAS = ("byes", "legbyes")


def _phase_weight(phase: str) -> float:
    return PHASE_IMPACT_WEIGHTS.get(phase, DEFAULT_PHASE_WEIGHT)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _ratio_score(actual: float, expected: float, higher_is_better: bool) -> Optional[float]:
    """Map `actual` against `expected` onto the 0-100 pillar scale, 50 == at baseline.

    Linear in the *relative* gap, so the scale behaves the same whether the metric is
    a strike rate in the 120s or an economy rate in the 8s -- which is what lets four
    pillars in four different units be averaged together at all.

    Returns `None` when `expected` is non-positive: a ratio against a zero baseline is
    undefined, and the caller must treat that as an unmeasured pillar (redistributing
    its weight) rather than silently scoring it 0 or 50.
    """
    if expected <= 0:
        return None
    ratio = actual / expected
    edge = (ratio - 1.0) if higher_is_better else (1.0 - ratio)
    return round(_clamp(BASELINE_SCORE + BASELINE_SCORE * (edge / RATIO_SCORE_SPREAD), 0.0, 100.0), 2)


def _runs_charged_to_bowler(delivery: FusedDelivery) -> int:
    """Runs off this delivery charged to the bowler -- byes/leg-byes aren't, same
    convention as `fusion.pipeline` and `player_reports.bowling_summary`."""
    if delivery.extras_type in _UNCHARGED_EXTRAS:
        return delivery.runs_total - delivery.runs_extras
    return delivery.runs_total


def balls_faced(deliveries: Iterable[FusedDelivery], player: str) -> list[FusedDelivery]:
    """Deliveries `player` faced as striker. A wide isn't a ball faced; a no-ball is."""
    return [d for d in deliveries if d.striker == player and d.extras_type not in _WIDES]


def legal_balls_bowled(deliveries: Iterable[FusedDelivery], player: str) -> list[FusedDelivery]:
    """Legal deliveries `player` bowled -- wides and no-balls excluded, as in `fusion`."""
    return [d for d in deliveries if d.bowler == player and d.extras_type not in _ILLEGAL_DELIVERIES]


def compute_baseline(
    history: Sequence[PlayerMatchStats],
    fused: Sequence[FusedDelivery] = (),
    source: Optional[str] = None,
) -> Optional[RatingBaseline]:
    """The cohort mean every pillar is scored against.

    **This is a season/cohort baseline, not PRD 3.1's matchup baseline**, and the
    reason is a data gap rather than a shortcut: a matchup baseline needs
    `bowling_style`/`batting_style` metadata to generalise "this batter vs left-arm
    orthodox", and Cricsheet's player registry carries none -- the same gap
    `prediction.contracts.MatchupFeatures` documents and declines to fake. Pooling
    every player in the sample is the most specific comparison the ingested data
    actually supports. Scope it by passing a narrower `history` (one competition, one
    season) and describing that scope in `source`.

    Pooling is over totals, not a mean of per-player rates, so a player with 200 balls
    weighs more than one with 4 -- an unweighted mean of rates would let a single
    4-ball cameo at a strike rate of 300 drag the whole cohort baseline up.

    `fused` is optional and only adds the delivery-level fields (`dot_ball_percent`
    and the per-phase rates). Without it those stay `None`/empty and the Game impact
    pillar goes unmeasured rather than guessed -- phase lives on the delivery row, not
    on `PlayerMatchStats`.

    Returns `None` for an empty `history`: there is no such thing as a baseline over
    no players, and returning a zero-filled one would make every later ratio undefined
    in a way the caller couldn't see.
    """
    if not history:
        return None

    total_runs = sum(s.runs_scored for s in history)
    total_balls_faced = sum(s.balls_faced for s in history)
    total_boundaries = sum(s.fours + s.sixes for s in history)
    total_conceded = sum(s.runs_conceded for s in history)
    total_balls_bowled = sum(s.balls_bowled for s in history)
    total_wickets = sum(s.wickets_taken for s in history)

    overs_bowled = total_balls_bowled / 6
    players = {s.player for s in history}
    matches = {s.match_id for s in history}

    # Kept short because it is quoted inline in coaching notes, not just logged --
    # a sentence-long description makes every generated and templated note unreadable.
    described = source or f"cohort mean over {len(matches)} matches, {len(players)} players"

    return RatingBaseline(
        source=described,
        players_in_cohort=len(players),
        matches_in_cohort=len(matches),
        batting_strike_rate=round(100 * total_runs / total_balls_faced, 2) if total_balls_faced else 0.0,
        economy_rate=round(total_conceded / overs_bowled, 2) if overs_bowled else 0.0,
        wickets_per_over=round(total_wickets / overs_bowled, 4) if overs_bowled else 0.0,
        boundary_percent=round(100 * total_boundaries / total_balls_faced, 2) if total_balls_faced else None,
        dot_ball_percent=_cohort_dot_ball_percent(fused),
        phases=_cohort_phase_baselines(fused),
        pitch=cohort_pitch_baseline(fused),
    )


def cohort_pitch_baseline(fused: Sequence[FusedDelivery]) -> Optional[PitchBaseline]:
    """Cohort pitch-length/line rates, or `None` when no row carries real geometry.

    **Returns `None` for every input available today**, because
    `FusedDelivery.has_pitch_geometry()` is False on every row a real pipeline produces
    -- there is no ball or stumps detector and so no calibrated bounce point (see
    `video_engine.calibration`). It is implemented rather than stubbed because it is
    pure arithmetic over a declared field, and because the rate it computes is what
    decides whether `suggestions.flag_player` can emit a pitch flag at all.

    The denominator is legal deliveries *that carry geometry*, not all legal deliveries.
    Mixing the two would silently report a bowler's good-length percentage as near zero
    whenever most of their deliveries simply weren't filmed -- an artefact of video
    coverage presented as a bowling problem, which is precisely the kind of number this
    repo refuses to produce. `PitchBaseline.deliveries_with_geometry` carries that
    denominator so the coverage is visible downstream.
    """
    rows = [
        d for d in fused
        if d.extras_type not in _ILLEGAL_DELIVERIES and d.has_pitch_geometry()
    ]
    if not rows:
        return None

    total = len(rows)
    with_line = [d for d in rows if d.pitch_line_m is not None]
    in_corridor = sum(
        1 for d in with_line
        if abs(d.pitch_line_m) <= STUMP_HALF_WIDTH_M + STUMP_LINE_TOLERANCE_M
    )
    return PitchBaseline(
        good_length_percent=_band_percent(rows, Metric.GOOD_LENGTH_PERCENT),
        short_ball_percent=_band_percent(rows, Metric.SHORT_BALL_PERCENT),
        full_ball_percent=_band_percent(rows, Metric.FULL_BALL_PERCENT),
        # Line is reported over the rows that actually have a line, which can be fewer
        # than `total`: a full toss has a real length classification and no bounce point,
        # so no line. Zero when none of them do -- not None, because `PitchBaseline`
        # exists at all only when some geometry was present, and a nullable field inside
        # it would reintroduce the partial-population trap that class documents.
        stump_line_percent=round(100 * in_corridor / len(with_line), 2) if with_line else 0.0,
        deliveries_with_geometry=total,
    )


def _band_percent(rows: Sequence[FusedDelivery], metric: Metric) -> float:
    """Percentage of `rows` whose `pitch_length` falls in `metric`'s bands."""
    bands = PITCH_METRIC_BANDS[metric]
    return round(100 * sum(1 for d in rows if d.pitch_length.value in bands) / len(rows), 2)


def _cohort_dot_ball_percent(fused: Sequence[FusedDelivery]) -> Optional[float]:
    """Dot balls as a % of balls faced across the cohort, or `None` without delivery rows.

    A "dot" here is a delivery that produced no runs at all (not merely none off the
    bat) -- that's the ball a batter is actually asked to account for.
    """
    faced = [d for d in fused if d.extras_type not in _WIDES]
    if not faced:
        return None
    return round(100 * sum(1 for d in faced if d.runs_total == 0) / len(faced), 2)


def _cohort_phase_baselines(fused: Sequence[FusedDelivery]) -> tuple[PhaseBaseline, ...]:
    """One `PhaseBaseline` per phase present in `fused`, in `PHASE_IMPACT_WEIGHTS`
    order first and then any unrecognised phase alphabetically -- a stable order, so
    two identical inputs can't produce two differently-ordered baselines."""
    if not fused:
        return ()

    present = {d.phase for d in fused}
    known = [p for p in PHASE_IMPACT_WEIGHTS if p in present]
    unknown = sorted(present - set(PHASE_IMPACT_WEIGHTS))

    baselines: list[PhaseBaseline] = []
    for phase in known + unknown:
        faced = [d for d in fused if d.phase == phase and d.extras_type not in _WIDES]
        legal = [d for d in fused if d.phase == phase and d.extras_type not in _ILLEGAL_DELIVERIES]
        if not faced and not legal:
            continue
        baselines.append(
            PhaseBaseline(
                phase=phase,
                runs_per_ball=round(sum(d.runs_batter for d in faced) / len(faced), 4) if faced else 0.0,
                runs_conceded_per_ball=(
                    round(sum(_runs_charged_to_bowler(d) for d in legal) / len(legal), 4) if legal else 0.0
                ),
                balls_in_cohort=len(faced),
            )
        )
    return tuple(baselines)


def baseline_from_deliveries(
    deliveries: "pd.DataFrame", source: Optional[str] = None
) -> Optional[RatingBaseline]:
    """`compute_baseline`'s answer, computed straight off the ball-by-ball frame.

    **Identical output, different cost.** `compute_baseline(history, fused)` needs its
    caller to have built a `PlayerMatchStats` for every (player, match) and a
    `FusedDelivery` for every ball first -- roughly 1.8M dataclass instances for a
    full-history warehouse -- and then does nothing with them but sum. Every figure on
    `RatingBaseline` is a *pooled total*, so the per-player and per-delivery objects
    cancel out of the arithmetic entirely and can be skipped. This does the same sums
    vectorised, which is the difference between minutes and milliseconds at warehouse
    scale.

    That this is genuinely the same number and not a close-enough reimplementation is
    the load-bearing claim, so it is pinned by
    `test_baseline_from_deliveries_matches_compute_baseline_exactly` rather than argued
    for here: the test builds both from the same rows and asserts field equality,
    including the rounding. Every convention below is the one `fusion.pipeline` and
    `player_reports.stats` use -- a wide is not a ball faced, wides and no-balls are not
    legal balls bowled, byes and leg-byes are not charged to the bowler, run-outs are
    not credited to them.

    `compute_baseline` is not deprecated by this and is still the right entry point when
    a caller already holds `PlayerMatchStats` (the tests do, and so does any path that
    has fused for another reason). This is the entry point when the caller holds rows.

    **One field is structurally out of reach here: `pitch`.** The ball-by-ball table is
    ingestion's, and pitch geometry is video-derived -- there is no column in this frame
    that could carry it, so this always returns `pitch=None`. The equality with
    `compute_baseline` is therefore exact *for a cohort whose fused rows carry no pitch
    geometry*, which is every cohort today (no ball or stumps detector exists; see
    `video_engine.calibration`) but will not be forever. When a detector lands, a caller
    that needs pitch baselines must fuse its cohort and use `compute_baseline`, or this
    function must grow a way to read geometry from wherever it comes to be stored. The
    test that pins the equality asserts on that precondition rather than assuming it, so
    this will fail loudly rather than drift.
    """
    if deliveries is None or deliveries.empty:
        return None

    faced = deliveries[deliveries["extras_type"] != "wides"]
    legal = deliveries[~deliveries["extras_type"].isin(_ILLEGAL_DELIVERIES)]

    total_runs = int(deliveries["runs_batter"].sum())
    total_balls_faced = int(len(faced))
    total_boundaries = int(deliveries["runs_batter"].isin((4, 6)).sum())

    uncharged = deliveries.loc[
        deliveries["extras_type"].isin(_UNCHARGED_EXTRAS), "runs_extras"
    ].sum()
    total_conceded = int(deliveries["runs_total"].sum() - uncharged)
    total_balls_bowled = int(len(legal))
    total_wickets = int(_credited_wickets(deliveries).sum())

    overs_bowled = total_balls_bowled / 6
    players = pd.concat([deliveries["striker"], deliveries["bowler"]]).nunique()
    matches = deliveries["match_id"].nunique()

    described = source or f"cohort mean over {matches} matches, {players} players"

    return RatingBaseline(
        source=described,
        players_in_cohort=int(players),
        matches_in_cohort=int(matches),
        batting_strike_rate=round(100 * total_runs / total_balls_faced, 2) if total_balls_faced else 0.0,
        economy_rate=round(total_conceded / overs_bowled, 2) if overs_bowled else 0.0,
        wickets_per_over=round(total_wickets / overs_bowled, 4) if overs_bowled else 0.0,
        boundary_percent=round(100 * total_boundaries / total_balls_faced, 2) if total_balls_faced else None,
        dot_ball_percent=(
            round(100 * int((faced["runs_total"] == 0).sum()) / total_balls_faced, 2)
            if total_balls_faced
            else None
        ),
        phases=_phase_baselines_from_deliveries(deliveries, faced, legal),
    )


def _credited_wickets(deliveries: "pd.DataFrame") -> "pd.Series":
    """Boolean mask of deliveries that credit the bowler with a wicket.

    Mirrors `fusion.pipeline`'s `d.wicket_player_out and d.wicket_kind != "run out"`.
    The truthiness there matters: a missing `wicket_player_out` round-trips out of
    Parquet as `None` (falsy) but an *empty string* would be falsy too, so both are
    excluded here rather than only nulls.
    """
    out = deliveries["wicket_player_out"]
    return out.notna() & (out.astype(str) != "") & (deliveries["wicket_kind"] != "run out")


def _phase_baselines_from_deliveries(
    deliveries: "pd.DataFrame", faced: "pd.DataFrame", legal: "pd.DataFrame"
) -> tuple[PhaseBaseline, ...]:
    """Vectorised `_cohort_phase_baselines`, including its phase ordering.

    Order is `PHASE_IMPACT_WEIGHTS`' declaration order for the phases present, then any
    unrecognised phase alphabetically -- the same stable order, for the same reason: two
    identical inputs must not produce two differently-ordered baselines.
    """
    if deliveries.empty:
        return ()

    present = set(deliveries["phase"].dropna().unique())
    known = [p for p in PHASE_IMPACT_WEIGHTS if p in present]
    unknown = sorted(present - set(PHASE_IMPACT_WEIGHTS))

    faced_runs = faced.groupby("phase")["runs_batter"].sum()
    faced_balls = faced.groupby("phase").size()

    charged = legal["runs_total"] - legal["runs_extras"].where(
        legal["extras_type"].isin(_UNCHARGED_EXTRAS), 0
    )
    conceded = charged.groupby(legal["phase"]).sum()
    legal_balls = legal.groupby("phase").size()

    baselines: list[PhaseBaseline] = []
    for phase in known + unknown:
        n_faced = int(faced_balls.get(phase, 0))
        n_legal = int(legal_balls.get(phase, 0))
        if not n_faced and not n_legal:
            continue
        baselines.append(
            PhaseBaseline(
                phase=phase,
                runs_per_ball=round(float(faced_runs.get(phase, 0)) / n_faced, 4) if n_faced else 0.0,
                runs_conceded_per_ball=(
                    round(float(conceded.get(phase, 0)) / n_legal, 4) if n_legal else 0.0
                ),
                balls_in_cohort=n_faced,
            )
        )
    return tuple(baselines)


def _window_for(history: Sequence[PlayerMatchStats], player: str, window_size: int) -> list[PlayerMatchStats]:
    """`player`'s last `window_size` matches, most recent first.

    Ties on `match_date` break by `match_id` descending -- the same deterministic rule
    `fusion.refresh_rolling_summary` and `player_reports.last_n_match_ids` use. Same
    reason: same-date ties are common in T20 tournaments, and an unstable sort would
    let repeated calls against identical data rate the player off a different window.
    """
    mine = [s for s in history if s.player == player]
    return sorted(mine, key=lambda s: (s.match_date, s.match_id), reverse=True)[:window_size]


def _technique_score(deliveries: Sequence[FusedDelivery], player: str) -> Optional[float]:
    """PRD 3.1's Technique pillar. **Returns `None`. Always. For every player.**

    Not a stub that someone forgot to finish -- a refusal. Technique means footwork,
    backlift and release point, and the only inputs for those would be
    `FusedDelivery`'s `wrist_speed_ms`, `bowling_elbow_extension_deg`,
    `ball_release_speed_kmh` and `bat_swing_speed_kmh`, every one of which is `None`
    today because no calibrated extractor produces them (see
    `fusion.contracts.FusedDelivery`). What `video_engine` *does* produce is wrist
    displacement in raw pixel space with no camera calibration behind it, which cannot
    be turned into a real-world joint angle or speed no matter how it's scaled.

    So this deliberately does not read those fields at all, even if a caller populates
    them. A score derived from uncalibrated pixel displacement would look exactly like
    a real Technique score to everything downstream -- the composite, the UI, an
    analyst -- while meaning nothing, and that is a worse failure than an absent
    pillar, which at least announces itself through `PlayerRating.unmeasured_pillars`.

    When a calibrated extractor lands in `video_engine/pose/`, this is the one
    function to implement: the composite machinery already handles a non-`None`
    Technique with no other change, because `rate_player` reads the pillar set from
    `PillarScores.measured()` rather than from a hard-coded list.
    """
    return None


def _execution_score(
    window: Sequence[PlayerMatchStats], baseline: RatingBaseline, profile: RoleProfile
) -> Optional[float]:
    """PRD 3.1's Execution pillar: outcome vs. the cohort's expectation.

    Batting roles are scored on strike rate against the cohort strike rate. Bowling
    roles blend two components with equal weight -- economy (lower is better) and
    wickets per over (higher is better) -- because PRD 3.1 describes this pillar as
    "runs/wickets relative to matchup baseline" and economy alone would rate a
    defensive bowler who never takes a wicket identically to a strike bowler with the
    same economy. The wickets component drops out when the cohort took no wickets at
    all, rather than scoring everyone 0 against an impossible baseline.

    `None` when the player has fewer than `MIN_BALLS_FOR_PILLAR` balls on the relevant
    side of the ball in the window.
    """
    if profile.is_batting_role:
        runs = sum(s.runs_scored for s in window)
        balls = sum(s.balls_faced for s in window)
        if balls < MIN_BALLS_FOR_PILLAR:
            return None
        return _ratio_score(100 * runs / balls, baseline.batting_strike_rate, higher_is_better=True)

    balls = sum(s.balls_bowled for s in window)
    if balls < MIN_BALLS_FOR_PILLAR:
        return None
    overs = balls / 6
    economy = sum(s.runs_conceded for s in window) / overs
    components = [_ratio_score(economy, baseline.economy_rate, higher_is_better=False)]
    if baseline.wickets_per_over > 0:
        wickets_per_over = sum(s.wickets_taken for s in window) / overs
        components.append(_ratio_score(wickets_per_over, baseline.wickets_per_over, higher_is_better=True))

    scored = [c for c in components if c is not None]
    if not scored:
        return None
    return round(sum(scored) / len(scored), 2)


def _game_impact_score(
    deliveries: Sequence[FusedDelivery], player: str, baseline: RatingBaseline, profile: RoleProfile
) -> Optional[float]:
    """PRD 3.1's Game impact pillar: phase-adjusted contribution.

    Phase enters twice, deliberately, because it answers two different questions:

    * The *expectation* is computed per phase, so a batter isn't penalised for facing
      a hard phase -- each ball is compared against what the cohort scores in that
      phase (`PhaseBaseline`).
    * Both the actual and the expectation are then weighted by `PHASE_IMPACT_WEIGHTS`,
      so a run at the death counts for more than one in the middle overs.

    Because the same weights appear in both the numerator and the denominator, the
    score answers "did this player beat a phase-appropriate baseline, counted most
    heavily in the phases that matter most" -- not "did they bat in the death overs",
    which is a selection fact, not a performance one.

    `None` when there are no delivery rows for the player, or fewer than
    `MIN_BALLS_FOR_PILLAR` of them, or when the cohort baseline has no usable rate to
    compare against. Phase lives only on `FusedDelivery`, so a stats-only caller
    legitimately gets an unmeasured pillar here and the weight redistributes.
    """
    batting = profile.is_batting_role
    rows = balls_faced(deliveries, player) if batting else legal_balls_bowled(deliveries, player)
    if len(rows) < MIN_BALLS_FOR_PILLAR:
        return None

    weights = [_phase_weight(d.phase) for d in rows]
    total_weight = sum(weights)
    if total_weight <= 0:
        return None

    expectations = [_expected_runs_per_ball(baseline, d.phase, batting=batting) for d in rows]
    if any(e is None for e in expectations):
        return None

    runs = [float(d.runs_batter if batting else _runs_charged_to_bowler(d)) for d in rows]
    actual = sum(r * w for r, w in zip(runs, weights)) / total_weight
    expected = sum(e * w for e, w in zip(expectations, weights)) / total_weight
    # More runs scored is good; more runs conceded is not -- hence the role flag doubles
    # as the direction of the comparison.
    return _ratio_score(actual, expected, higher_is_better=batting)


def _expected_runs_per_ball(baseline: RatingBaseline, phase: str, batting: bool) -> Optional[float]:
    """The cohort's runs-per-ball in `phase`, falling back to its whole-innings rate.

    The fallback matters for short windows: a player can face a phase the cohort
    sample happens not to cover, and falling back to the overall rate is a weaker but
    still honest comparison. It is only `None` when the cohort has no rate at all on
    that side of the ball, which is a genuinely unmeasurable pillar.
    """
    phase_baseline = baseline.phase_baseline(phase)
    if phase_baseline is not None:
        value = phase_baseline.runs_per_ball if batting else phase_baseline.runs_conceded_per_ball
        if value > 0:
            return value

    fallback = baseline.batting_strike_rate / 100 if batting else baseline.economy_rate / 6
    return fallback if fallback > 0 else None


def _consistency_score(window: Sequence[PlayerMatchStats], profile: RoleProfile) -> Optional[float]:
    """PRD 3.1's Consistency pillar: variance across the recent innings in the window.

    Scored from the coefficient of variation (population standard deviation over the
    mean) of per-match runs for batting roles and per-match economy for bowling roles.
    CV rather than raw standard deviation because it is unit-free: a batter averaging
    60 with a spread of 15 is more consistent than one averaging 20 with a spread of
    12, and raw sigma says the opposite.

    `score = 100 * (1 - cv)`, clamped -- so a perfectly level player scores 100, one
    whose spread equals their mean scores 0. Only the *spread* is measured here; being
    consistently poor scores well on this pillar and badly on Execution, which is the
    intended division of labour between the two.

    `None` with fewer than `MIN_INNINGS_FOR_CONSISTENCY` relevant innings (a spread
    needs two points) or when the mean is non-positive (CV undefined).
    """
    if profile.is_batting_role:
        values = [float(s.runs_scored) for s in window if s.balls_faced > 0]
    else:
        values = [float(s.economy_rate) for s in window if s.balls_bowled > 0]

    if len(values) < MIN_INNINGS_FOR_CONSISTENCY:
        return None
    mean = sum(values) / len(values)
    if mean <= 0:
        return None
    return round(_clamp(100 * (1 - pstdev(values) / mean), 0.0, 100.0), 2)


def rate_player(
    history: Sequence[PlayerMatchStats],
    player: str,
    role: PlayerRole,
    baseline: RatingBaseline,
    fused: Sequence[FusedDelivery] = (),
    window_size: int = 5,
) -> Optional[PlayerRating]:
    """`player`'s composite rating over their last `window_size` matches in `history`.

    Signature mirrors `fusion.refresh_rolling_summary` -- full history in, one player
    named, window applied internally -- so the two can be driven off the same list
    without the caller pre-slicing it differently for each.

    `fused` should be the delivery rows for those same matches; it is only used by the
    Game impact pillar. Omitting it doesn't produce a wrong rating, it produces a
    rating with Game impact unmeasured and its weight redistributed, flagged as such
    in `PlayerRating.unmeasured_pillars`.

    Returns `None` when `player` has no matches in `history` at all -- the same
    "unknown player" case `refresh_rolling_summary` returns `None` for. A player who
    *does* appear but whose every pillar is unmeasurable gets a real `PlayerRating`
    with `composite=None` and `insufficient_data=True` instead, because "we have this
    player and can't score them" and "we've never seen this player" are different
    answers and the caller may well handle them differently.
    """
    profile = ROLE_PROFILES[role]
    window = _window_for(history, player, window_size)
    if not window:
        return None

    window_ids = {s.match_id for s in window}
    window_deliveries = [d for d in fused if d.match_id in window_ids]

    pillars = PillarScores(
        technique=_technique_score(window_deliveries, player),
        execution=_execution_score(window, baseline, profile),
        game_impact=_game_impact_score(window_deliveries, player, baseline, profile),
        consistency=_consistency_score(window, profile),
    )

    measured = pillars.measured()
    weights_used = profile.weights.restricted_to(measured)
    unmeasured = tuple(p for p in Pillar if p not in measured)

    if not measured:
        composite = None
    else:
        scores = pillars_as_dict(pillars)
        composite = round(sum(weights_used.as_dict()[p] * scores[p] for p in measured), 2)

    return PlayerRating(
        player=player,
        role=role,
        composite=composite,
        pillars=pillars,
        weights_used=weights_used,
        unmeasured_pillars=unmeasured,
        insufficient_data=composite is None,
        matches_in_sample=len(window),
        baseline_source=baseline.source,
    )


def pillars_as_dict(pillars: PillarScores) -> dict[Pillar, Optional[float]]:
    """`PillarScores` keyed by `Pillar`, so weights and scores can be zipped by key
    rather than by field order -- adding a pillar shouldn't be able to silently
    misalign the two."""
    return {
        Pillar.TECHNIQUE: pillars.technique,
        Pillar.EXECUTION: pillars.execution,
        Pillar.GAME_IMPACT: pillars.game_impact,
        Pillar.CONSISTENCY: pillars.consistency,
    }


def rate_players(
    history: Sequence[PlayerMatchStats],
    roles: Mapping[str, PlayerRole],
    baseline: RatingBaseline,
    fused: Sequence[FusedDelivery] = (),
    window_size: int = 5,
) -> list[PlayerRating]:
    """`rate_player` for every player in `roles`, in player-name order.

    Keyed off `roles` rather than off everyone in `history` because role is an input
    the data can't supply (see `contracts.PlayerRole`): a player with no role assigned
    can't be rated, and guessing one from batting position would be an invention.
    Players named in `roles` but absent from `history` are skipped, matching
    `fusion.refresh_all_rolling_summaries`' handling of a player with no history.
    """
    ratings = ((player, rate_player(history, player, roles[player], baseline, fused, window_size))
               for player in sorted(roles))
    return [rating for _, rating in ratings if rating is not None]
