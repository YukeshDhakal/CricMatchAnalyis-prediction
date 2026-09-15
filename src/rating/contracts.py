"""Data contracts for the rating + coaching-suggestion layer (PRD 2.2 stages 5-6).

Stage 5 turns `fusion`'s `PlayerMatchStats`/`FusedDelivery` rows into a composite
`PlayerRating` built from PRD 3.1's four weighted pillars. Stage 6 turns the same
rows into ranked, clip-cited `CoachingSuggestion`s (PRD 3.2/3.3).

Two things in here are deliberate deviations from the PRD, both recorded in the
docstrings of the contracts they affect rather than only in a commit message:

1. **The Technique pillar has no producer and is never scored.** PRD 3.1 weights it
   at 20-30% depending on role, but every input it would need -- `FusedDelivery`'s
   `wrist_speed_ms`, `bowling_elbow_extension_deg`, `ball_release_speed_kmh`,
   `bat_swing_speed_kmh` -- is `None` today and will stay `None` until a *calibrated*
   biomechanics extractor lands in `video_engine/pose/` (see
   `fusion.contracts.FusedDelivery`'s docstring for why the existing pixel-space
   heuristic doesn't count). `PlayerRating` handles that by redistribution, not by
   scoring it 0 -- see that class's docstring for the decision and its cost.

2. **Ranking is a documented deterministic formula, not the PRD's "learned ranker".**
   See `suggestions.rank_flags`. There is no labelled analyst-feedback data to train
   a ranker on yet (the `status` field on `CoachingSuggestion` is where that data
   would eventually come from), so training one now would mean inventing labels.

Contracts only -- no scoring, flagging, or I/O happens in this module. Same division
of labor as `fusion/contracts.py` vs `fusion/pipeline.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ingestion.contracts import DeliveryRef

__all__ = [
    "Pillar",
    "PlayerRole",
    "PillarWeights",
    "RoleProfile",
    "ROLE_PROFILES",
    "PHASE_IMPACT_WEIGHTS",
    "DEFAULT_PHASE_WEIGHT",
    "PhaseBaseline",
    "RatingBaseline",
    "PillarScores",
    "PlayerRating",
    "Metric",
    "MetricSpec",
    "METRIC_SPECS",
    "PerformanceFlag",
    "NoteDraft",
    "CoachingSuggestion",
    "SUGGESTION_STATUSES",
    "citation_token",
]


class Pillar(str, Enum):
    """PRD 3.1's four rating pillars. `str`-valued for the same reason `ShotType` is --
    it has to survive a round trip through JSON or a DataFrame column without a codec.
    """

    TECHNIQUE = "technique"
    EXECUTION = "execution"
    GAME_IMPACT = "game_impact"
    CONSISTENCY = "consistency"


class PlayerRole(str, Enum):
    """The role whose weight table a player is rated under.

    Only the two roles PRD 3.1 actually tabulates exist here. Roles are *not* inferred
    from the data -- there is no batting-order or bowling-style metadata in the
    ingested Cricsheet schema to infer them from (the same gap
    `prediction.contracts.MatchupFeatures` documents and declines to fake), so the
    caller passes the role in. Adding a role means adding one `RoleProfile` to
    `ROLE_PROFILES` and nothing else: the engine reads weights only through that table.
    """

    TOP_ORDER_BATTER = "top_order_batter"
    DEATH_OVERS_BOWLER = "death_overs_bowler"


@dataclass(frozen=True)
class PillarWeights:
    """One row of PRD 3.1's weight table. Weights are fractions and should sum to 1.0."""

    technique: float
    execution: float
    game_impact: float
    consistency: float

    def as_dict(self) -> dict[Pillar, float]:
        return {
            Pillar.TECHNIQUE: self.technique,
            Pillar.EXECUTION: self.execution,
            Pillar.GAME_IMPACT: self.game_impact,
            Pillar.CONSISTENCY: self.consistency,
        }

    def total(self) -> float:
        return self.technique + self.execution + self.game_impact + self.consistency

    def restricted_to(self, pillars: frozenset[Pillar]) -> "PillarWeights":
        """Zero every pillar outside `pillars` and rescale the rest to sum to 1.0.

        This is the mechanical half of the Technique decision described on
        `PlayerRating`: a pillar with no measurable input is taken out of the scoring
        entirely instead of being scored 0, which would silently drag every composite
        down by that pillar's weight.

        Returns an all-zero `PillarWeights` when nothing survives -- the caller
        (`engine.rate_player`) reads that as "insufficient data" rather than dividing
        by zero.
        """
        kept = {p: w for p, w in self.as_dict().items() if p in pillars}
        surviving = sum(kept.values())
        if surviving <= 0:
            return PillarWeights(0.0, 0.0, 0.0, 0.0)
        scaled = {p: w / surviving for p, w in kept.items()}
        return PillarWeights(
            technique=scaled.get(Pillar.TECHNIQUE, 0.0),
            execution=scaled.get(Pillar.EXECUTION, 0.0),
            game_impact=scaled.get(Pillar.GAME_IMPACT, 0.0),
            consistency=scaled.get(Pillar.CONSISTENCY, 0.0),
        )


@dataclass(frozen=True)
class RoleProfile:
    """Everything role-specific the engine needs: the weight row plus which side of the
    ball the role is scored on.

    `is_batting_role` picks which stat feeds Execution/Game impact/Consistency -- a
    top-order batter is scored on scoring rate, a death-overs bowler on runs conceded
    and wickets taken. An all-rounder role would need both sides scored and blended;
    that is deliberately not modelled, because there is no defensible split weight to
    pick without real analyst input.
    """

    role: PlayerRole
    weights: PillarWeights
    is_batting_role: bool


# PRD 3.1's weight table, verbatim. Tunable per role -- this dict is the single place
# weights live, so adding a role is a one-entry change here and never a code change in
# `engine.py`.
ROLE_PROFILES: dict[PlayerRole, RoleProfile] = {
    PlayerRole.TOP_ORDER_BATTER: RoleProfile(
        role=PlayerRole.TOP_ORDER_BATTER,
        weights=PillarWeights(technique=0.30, execution=0.30, game_impact=0.25, consistency=0.15),
        is_batting_role=True,
    ),
    PlayerRole.DEATH_OVERS_BOWLER: RoleProfile(
        role=PlayerRole.DEATH_OVERS_BOWLER,
        weights=PillarWeights(technique=0.20, execution=0.35, game_impact=0.35, consistency=0.10),
        is_batting_role=False,
    ),
}


# Phase multipliers for PRD 3.1's "phase-adjusted contribution" pillar. A run scored
# (or conceded) at the death is worth more than one in the middle overs, and the
# powerplay sits between the two because of the fielding restrictions.
#
# These are a deliberate, tunable starting point -- not fitted values. Nothing in this
# repo has the labelled outcome data you would need to fit them (that means regressing
# phase-segmented contribution against match results across a season), so presenting
# them as calibrated would be exactly the kind of false precision `FusedDelivery`
# refuses for its biomechanics fields. Keys are `ingestion.rules`' phase strings.
PHASE_IMPACT_WEIGHTS: dict[str, float] = {
    "powerplay": 1.25,
    "middle": 1.00,
    "death": 1.50,
}

# A phase string with no entry above falls back to this rather than raising: an
# unrecognised phase shouldn't take a whole rating down, it should score as neutral.
DEFAULT_PHASE_WEIGHT = 1.00


@dataclass(frozen=True)
class PhaseBaseline:
    """Cohort reference rates for one phase.

    Held as a tuple of these on `RatingBaseline` rather than as a `dict` field so the
    contract stays hashable and genuinely immutable the way every other frozen
    contract in this repo is -- a `dict` field on a frozen dataclass is still mutable
    in place, which would let a caller edit a baseline that a `PerformanceFlag`
    already cited.
    """

    phase: str
    runs_per_ball: float  # runs off the bat per ball faced -- the batting expectation
    # Runs charged to the bowler per legal ball. Not the same number as
    # `runs_per_ball`: byes/leg-byes score for the batting side but aren't charged to
    # the bowler, and wides/no-balls add runs without adding a legal ball. Keeping
    # both means neither side of the ball is scored against the other side's
    # expectation. Conventions match `fusion.pipeline` exactly.
    runs_conceded_per_ball: float
    balls_in_cohort: int


@dataclass(frozen=True)
class RatingBaseline:
    """The "expectation" half of PRD 3.1's Execution pillar: what an average player in
    this sample does, so one player's numbers can be scored as a ratio against it.

    **Which baseline this is, and why.** PRD 3.1 asks for a *matchup* baseline (this
    batter against this class of bowling). That is not computable from what is
    ingested today: a matchup baseline needs `bowling_style`/`batting_style` player
    metadata, and Cricsheet's registry has none -- the exact gap
    `prediction.contracts.MatchupFeatures` already documents. So this is a
    **cohort/season baseline**: the pooled mean across every player in the
    `PlayerMatchStats` sample handed to `engine.compute_baseline`. Feed it one
    competition's matches and it is that competition's season average.

    `source` carries a human-readable description of the cohort so it can be echoed
    into a `CoachingSuggestion`. PRD 2.4 forbids black-box verdicts, and "15% below
    baseline" *is* a black-box verdict unless the analyst can see which baseline.

    Boundary/dot/phase rates are `None` or empty whenever the producing rows weren't
    supplied -- dot-ball rate and phase rates live on `FusedDelivery`, not on
    `PlayerMatchStats`, so a stats-only baseline legitimately has neither.
    """

    source: str
    players_in_cohort: int
    matches_in_cohort: int

    batting_strike_rate: float  # runs per 100 balls faced
    economy_rate: float  # runs conceded per over
    wickets_per_over: float

    boundary_percent: Optional[float] = None  # fours+sixes as a % of balls faced
    dot_ball_percent: Optional[float] = None  # needs delivery rows; None without them
    phases: tuple[PhaseBaseline, ...] = ()

    def phase_baseline(self, phase: str) -> Optional[PhaseBaseline]:
        """The `PhaseBaseline` for `phase`, or `None` if the cohort had no balls in it."""
        return next((p for p in self.phases if p.phase == phase), None)


@dataclass(frozen=True)
class PillarScores:
    """Each pillar on a 0-100 scale where **50 means "exactly at baseline"**, not "half
    marks".

    A pillar is `None` when it has no measurable input for this player -- never 0, for
    the same reason `FusedDelivery` leaves unproduced fields `None`: 0 is a real score
    meaning "as bad as this scale goes", and the two must not be confusable by
    anything downstream.

    `technique` is `None` for every player today. See `PlayerRating`.
    """

    technique: Optional[float]
    execution: Optional[float]
    game_impact: Optional[float]
    consistency: Optional[float]

    def measured(self) -> frozenset[Pillar]:
        """The pillars that actually got a score."""
        pairs = (
            (Pillar.TECHNIQUE, self.technique),
            (Pillar.EXECUTION, self.execution),
            (Pillar.GAME_IMPACT, self.game_impact),
            (Pillar.CONSISTENCY, self.consistency),
        )
        return frozenset(pillar for pillar, value in pairs if value is not None)


@dataclass(frozen=True)
class PlayerRating:
    """One player's composite rating over a window of matches, plus the four pillar
    sub-scores it was built from.

    **The Technique decision (PRD 3.1's 20-30% pillar).** Technique is currently
    unmeasurable -- see this module's docstring. Of the two honest options, this module
    takes **(a) proportional weight redistribution**: a pillar with no measurable input
    has its weight removed and the remaining weights rescaled to sum to 1.0
    (`PillarWeights.restricted_to`), and the composite is computed from what is left.

    Why (a) and not (b) "return `None` / insufficient data": Technique is `None` for
    *every* player, and will be until a calibrated extractor exists. So (b) would make
    the entire rating engine return `None` for every input -- dead code that can't be
    tested against anything real, and nothing for the rest of the pipeline to be built
    against. Redistribution keeps the three pillars that *do* have real producers
    (Execution, Game impact, Consistency) usable today.

    The cost of (a), stated plainly because it is a real cost: a redistributed
    composite is **not** comparable to a future four-pillar composite, and it silently
    over-weights Execution unless the reader knows Technique is missing. That is what
    `unmeasured_pillars` and `weights_used` are for -- they are not debug fields. Any
    surface that shows `composite` must also show `unmeasured_pillars`, and no caller
    should persist a composite for later comparison without it.

    Option (b) is still reachable and is not dead: when *nothing* is measurable,
    `composite` is `None` and `insufficient_data` is True, rather than a number
    fabricated out of an empty weight set.

    Scores use `PillarScores`' 0-100 scale (50 == baseline), rounded to 2dp the way
    every other derived figure in `fusion.pipeline` is.
    """

    player: str
    role: PlayerRole
    composite: Optional[float]
    pillars: PillarScores

    # Weights after redistribution -- i.e. what actually produced `composite`, not the
    # role's table row. Compare against `ROLE_PROFILES[role].weights` to see what moved.
    weights_used: PillarWeights
    unmeasured_pillars: tuple[Pillar, ...]
    insufficient_data: bool

    matches_in_sample: int
    baseline_source: str


class Metric(str, Enum):
    """The vocabulary of things `suggestions.flag_player` can flag.

    An enum rather than free strings because three separate layers key off these
    names -- flagging computes them, ranking compares them, and `rating.llm`'s
    template library looks phrasing up by them. A typo'd metric name in any one of
    those would produce a silently unphrased or unrankable suggestion, which is
    exactly the failure mode a closed vocabulary removes.

    Every metric here is computable today from `FusedDelivery` rows plus a
    `RatingBaseline`. Nothing is reserved-but-unpopulated in this enum -- the
    unpopulated-shape pattern lives on `FusedDelivery`, not here, because a flag that
    can never fire is just dead branching in the rule layer.
    """

    STRIKE_RATE = "strike_rate"
    ECONOMY_RATE = "economy_rate"
    DOT_BALL_PERCENT = "dot_ball_percent"
    BOUNDARY_PERCENT = "boundary_percent"
    PHASE_STRIKE_RATE = "phase_strike_rate"
    PHASE_ECONOMY_RATE = "phase_economy_rate"


@dataclass(frozen=True)
class MetricSpec:
    """How to read and describe one `Metric`.

    `higher_is_better` is what turns a signed delta into `PerformanceFlag.is_concern`
    -- it is the only thing that knows a high economy rate is bad news and a high
    strike rate is good news, and it lives here so the flagging layer, the ranker and
    the note-writers can't each hold a different opinion about it.

    `label`/`unit` exist so generated prose and template prose describe a metric
    identically. If the LLM says "scoring rate" and the fallback template says
    "strike rate" for the same flag, an analyst comparing two notes can't tell whether
    they're looking at the same finding.
    """

    metric: "Metric"
    label: str
    unit: str
    higher_is_better: bool
    is_batting: bool
    # Whether the deliveries that push this metric *up* are the high-scoring ones.
    # Needed because it doesn't follow from `higher_is_better`: dot-ball percentage and
    # strike rate both concern a batter, but a bad dot-ball figure is evidenced by the
    # cheapest balls and a bad strike rate by... also the cheapest balls, while a bad
    # economy rate is evidenced by the most expensive. Deriving the evidence direction
    # from the metric's direction gets dot-ball percentage exactly backwards, so it is
    # stated per metric instead. See `suggestions._make_flag`.
    evidence_is_high_runs: bool


METRIC_SPECS: dict["Metric", MetricSpec] = {
    Metric.STRIKE_RATE: MetricSpec(
        metric=Metric.STRIKE_RATE, label="strike rate", unit="runs per 100 balls",
        higher_is_better=True, is_batting=True, evidence_is_high_runs=True,
    ),
    Metric.ECONOMY_RATE: MetricSpec(
        metric=Metric.ECONOMY_RATE, label="economy rate", unit="runs per over",
        higher_is_better=False, is_batting=False, evidence_is_high_runs=True,
    ),
    # More dot balls is worse for the batter whose flag this is -- the mirror-image
    # bowling view of the same deliveries would want the opposite sign, which is why
    # `is_batting` is part of the spec and not inferred from the name.
    Metric.DOT_BALL_PERCENT: MetricSpec(
        metric=Metric.DOT_BALL_PERCENT, label="dot-ball percentage", unit="% of balls faced",
        higher_is_better=False, is_batting=True, evidence_is_high_runs=False,
    ),
    Metric.BOUNDARY_PERCENT: MetricSpec(
        metric=Metric.BOUNDARY_PERCENT, label="boundary percentage", unit="% of balls faced",
        higher_is_better=True, is_batting=True, evidence_is_high_runs=True,
    ),
    Metric.PHASE_STRIKE_RATE: MetricSpec(
        metric=Metric.PHASE_STRIKE_RATE, label="strike rate", unit="runs per 100 balls",
        higher_is_better=True, is_batting=True, evidence_is_high_runs=True,
    ),
    Metric.PHASE_ECONOMY_RATE: MetricSpec(
        metric=Metric.PHASE_ECONOMY_RATE, label="economy rate", unit="runs per over",
        higher_is_better=False, is_batting=False, evidence_is_high_runs=True,
    ),
}


@dataclass(frozen=True)
class PerformanceFlag:
    """One deterministic "this number is off baseline" finding -- PRD 3.2's rule layer.

    Every field here is computed arithmetic over real `FusedDelivery`/`PlayerMatchStats`
    rows and real `DeliveryRef`s. **No LLM ever produces or edits a `PerformanceFlag`.**
    That separation is the whole mechanism behind PRD 2.4's "no black-box verdicts":
    the stat and the citation are established here, and the note-writer downstream is
    only ever allowed to rephrase them (see `rating.llm`).

    `citations` are the specific deliveries that drove the delta, chosen
    deterministically (see `suggestions.flag_player`) so the same input always cites
    the same balls. `citations_have_video` says whether those deliveries actually have
    footage behind them -- a `DeliveryRef` without video is still verifiable against
    the ball-by-ball table, but it is not the "underlying clip" PRD 2.4 asks for, and
    an analyst should be able to tell the difference at a glance rather than clicking
    through to find out.

    `is_concern` is what makes a flag actionable rather than merely notable: a strike
    rate 20% *above* baseline and one 20% below are both flags, but only one of them
    is something to coach away from.
    """

    player: str
    metric: Metric
    phase: Optional[str]  # None for a whole-innings metric
    actual: float
    baseline: float
    delta: float  # actual - baseline, in the metric's own units
    relative_delta: float  # delta / baseline, the unit-free quantity ranking uses
    is_concern: bool
    severity: float  # ranking score; see suggestions.rank_flags for the formula
    sample_size: int  # balls behind `actual` -- small samples are noisy, see severity
    baseline_source: str
    citations: tuple[DeliveryRef, ...] = ()
    citations_have_video: bool = False

    def citation_tokens(self) -> tuple[str, ...]:
        """The exact strings a generated note must echo back. See `citation_token`."""
        return tuple(citation_token(ref) for ref in self.citations)


def citation_token(ref: DeliveryRef) -> str:
    """A delivery's citation, as one compact, unambiguous, copy-pasteable token.

    Deliberately not prose. `rating.llm` validates a generated note by checking these
    substrings survived verbatim, so the format has to be something a language model
    will copy rather than paraphrase: "over 4 ball 2 of the first innings" invites
    rewording, `[clip m1 i1 o4.2]` does not.

    Changing this format changes what an already-stored `CoachingSuggestion` body can
    be re-validated against, so treat it as a persisted format, not an implementation
    detail.
    """
    return f"[clip {ref.match_id} i{ref.innings} o{ref.over}.{ref.ball}]"


@dataclass(frozen=True)
class NoteDraft:
    """The prose half of a suggestion, from whichever `rating.llm.NoteWriter` produced it.

    `source` is provenance, not decoration: it records whether an analyst is reading
    model-generated phrasing or the deterministic template, which is the difference
    between a sentence that needs checking and one that doesn't.
    """

    title: str
    body: str
    source: str  # "llm" | "template"


# The human-in-the-loop review states from PRD 3.3. No UI consumes these yet -- the
# field exists so the app layer can add review without a contract change, and so the
# accept/edit/reject stream it produces becomes the labelled data PRD 3.2's "learned
# ranker" would eventually need (see suggestions.rank_flags).
SUGGESTION_STATUSES = ("pending", "accepted", "edited", "rejected")


@dataclass(frozen=True)
class CoachingSuggestion:
    """One ranked, cited coaching note -- the rating layer's user-facing output.

    Carries the whole `flag` rather than copies of its numbers, so the suggestion can
    never drift out of sync with the arithmetic that justified it: `title`/`body` are
    presentation, `flag` is the evidence, and an analyst checking the claim reads the
    flag, not the sentence.

    `citations` is duplicated up from `flag.citations` because it is load-bearing at
    this level too -- it is what `rating.llm` validated the body against, and what a
    UI would turn into clip links.
    """

    player: str
    title: str
    body: str
    flag: PerformanceFlag
    citations: tuple[DeliveryRef, ...]
    rank: int  # 1-based position from suggestions.rank_flags
    note_source: str  # "llm" | "template", mirrors NoteDraft.source
    status: str = "pending"  # one of SUGGESTION_STATUSES

    def citation_tokens(self) -> tuple[str, ...]:
        return tuple(citation_token(ref) for ref in self.citations)
