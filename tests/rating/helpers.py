"""Hand-built `FusedDelivery`/`PlayerMatchStats` rows for the rating tests.

Same approach as `tests/fusion/test_pipeline.py`'s `_delivery` helper and
`tests/fakes.py`: real contract objects assembled by hand, no mocking framework and
no fixtures loaded from disk. The rating math is arithmetic over these rows, so a
test that can state its input in three lines can also state the exact number it
expects out -- which is the point of building them this way rather than sampling
real data.
"""
from __future__ import annotations

from typing import Sequence

from fusion.contracts import FusedDelivery, PlayerMatchStats
from ingestion.contracts import DeliveryRef


def fused(**overrides) -> FusedDelivery:
    """One `FusedDelivery`. `runs_total` defaults to `runs_batter + runs_extras` so a
    test that only cares about runs off the bat doesn't have to keep them in sync by
    hand -- an easy way to accidentally write a row that couldn't exist."""
    base = dict(
        match_id="m1",
        innings=1,
        over=0,
        ball=1,
        phase="middle",
        striker="Bob",
        non_striker="Nobody",
        bowler="Carl",
        runs_batter=0,
        runs_extras=0,
        extras_type=None,
        wicket_player_out=None,
        wicket_kind=None,
        video_available=False,
    )
    base.update(overrides)
    base.setdefault("runs_total", base["runs_batter"] + base["runs_extras"])
    ref = DeliveryRef(
        match_id=base["match_id"], innings=base["innings"], over=base["over"], ball=base["ball"]
    )
    return FusedDelivery(delivery=ref, **base)


def over_of(runs: Sequence[int], start_ball: int = 0, **common) -> list[FusedDelivery]:
    """One delivery per entry in `runs`, numbered sequentially from `start_ball`.

    Over/ball are derived (six legal balls to an over) rather than passed in, so a
    20-ball innings is one line and every delivery still gets a distinct, realistic
    `DeliveryRef` -- citations are only meaningful if the refs are distinguishable.
    """
    deliveries = []
    for offset, run in enumerate(runs):
        index = start_ball + offset
        deliveries.append(fused(over=index // 6, ball=index % 6 + 1, runs_batter=run, **common))
    return deliveries


def stats(**overrides) -> PlayerMatchStats:
    """One `PlayerMatchStats`. Rates are derived from the totals unless overridden, for
    the same "can't build an impossible row by accident" reason as `fused`."""
    base = dict(
        player="Bob",
        match_id="m1",
        match_date="2026-01-01",
        runs_scored=0,
        balls_faced=0,
        fours=0,
        sixes=0,
        balls_bowled=0,
        runs_conceded=0,
        wickets_taken=0,
    )
    base.update(overrides)
    if "batting_strike_rate" not in base:
        base["batting_strike_rate"] = (
            round(100 * base["runs_scored"] / base["balls_faced"], 2) if base["balls_faced"] else 0.0
        )
    if "economy_rate" not in base:
        base["economy_rate"] = (
            round(base["runs_conceded"] / (base["balls_bowled"] / 6), 2) if base["balls_bowled"] else 0.0
        )
    return PlayerMatchStats(**base)


def match_lines(deliveries: Sequence[FusedDelivery], match_date: str = "2026-01-01") -> list[PlayerMatchStats]:
    """The `PlayerMatchStats` lines `deliveries` aggregate to.

    Goes through the real `fusion.aggregate_player_match_stats` rather than
    recomputing totals in the test. A test that did its own arithmetic could agree
    with itself while disagreeing with the pipeline the rating engine is actually fed
    from, which would hide exactly the kind of convention drift (wides, no-balls,
    uncredited run-outs) the fusion layer is careful about.
    """
    from fusion.pipeline import aggregate_player_match_stats

    return aggregate_player_match_stats(list(deliveries), match_date=match_date)
