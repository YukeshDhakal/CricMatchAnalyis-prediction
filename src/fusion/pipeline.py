"""Fusion pipeline: join ingestion's ball-by-ball truth with video_engine's per-delivery
CV output into `FusedDelivery` rows, then aggregate those into per-match and
rolling-N-match player stats.

Scoring conventions (what counts as a ball faced/bowled, which wickets credit the
bowler, which extras don't count against them) intentionally mirror
`player_reports.stats` exactly -- a fusion-derived stat and a pure ball-by-ball one
should never disagree about what these mean, only about whether CV signal is layered
on top.

**Fusing many matches goes through `fuse_matches`, not a loop over
`fuse_match_deliveries`.** The single-match function selects its rows with a boolean
mask over the frame it is handed, which costs one full scan of that frame per call --
fine for one match, quadratic when a caller loops it over every match in the warehouse.
At 8,026 matches / 1.8M deliveries that loop measured ~69 ms per match (~550 s total),
against ~6 ms per match when the same warehouse held 230 matches: the per-call cost
grows with the *warehouse*, not with the match. `fuse_matches` groups once and is
linear in the rows it is given. See `fuse_matches` for the full note.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Optional

import pandas as pd

from ingestion.contracts import Delivery
from video_engine.contracts import DeliveryAnalysis

from .contracts import FusedDelivery, PlayerMatchStats, PlayerRollingSummary

# ICC's legal limit for bowling-arm elbow extension during the delivery stride.
_ILLEGAL_ACTION_ELBOW_DEG = 15.0

__all__ = [
    "fuse_match_deliveries",
    "fuse_matches",
    "aggregate_player_match_stats",
    "refresh_rolling_summary",
    "refresh_all_rolling_summaries",
]


def _delivery_from_row(row) -> Delivery:
    """One ball-by-ball `Delivery` from one `itertuples` row.

    Split out so `fuse_match_deliveries` and `fuse_matches` build rows through exactly
    the same code -- two row-builders that drifted apart would let the single-match and
    many-match paths disagree about the same delivery, which is the one thing a
    performance rewrite of this module must not be able to do.
    """
    return Delivery(
        match_id=row.match_id,
        innings=row.innings,
        over=row.over,
        ball=row.ball,
        batting_team=row.batting_team,
        bowling_team=row.bowling_team,
        striker=row.striker,
        non_striker=row.non_striker,
        bowler=row.bowler,
        runs_batter=row.runs_batter,
        runs_extras=row.runs_extras,
        runs_total=row.runs_total,
        extras_type=row.extras_type,
        wicket_player_out=row.wicket_player_out,
        wicket_kind=row.wicket_kind,
        phase=row.phase,
    )


def _fuse_rows(rows: pd.DataFrame, by_ref: dict) -> list[FusedDelivery]:
    """`FusedDelivery` per row of an already-selected frame, joined to `by_ref` analyses."""
    fused: list[FusedDelivery] = []
    for row in rows.itertuples():
        delivery = _delivery_from_row(row)
        analysis = by_ref.get(delivery.ref())
        if analysis is None:
            fused.append(FusedDelivery.from_delivery_only(delivery))
            continue

        fused.append(
            FusedDelivery(
                delivery=delivery.ref(),
                match_id=delivery.match_id,
                innings=delivery.innings,
                over=delivery.over,
                ball=delivery.ball,
                phase=delivery.phase,
                striker=delivery.striker,
                non_striker=delivery.non_striker,
                bowler=delivery.bowler,
                runs_batter=delivery.runs_batter,
                runs_extras=delivery.runs_extras,
                runs_total=delivery.runs_total,
                extras_type=delivery.extras_type,
                wicket_player_out=delivery.wicket_player_out,
                wicket_kind=delivery.wicket_kind,
                video_available=True,
                shot_type=analysis.event.shot_type,
                release_frame=analysis.event.release_frame,
                contact_frame=analysis.event.contact_frame,
                # Carried through, not derived. `DeliveryEvent.pitch_point` is `None` on
                # every real analysis today (no ball/stumps detector, no calibration --
                # see `video_engine.calibration`), so these resolve to None/UNKNOWN; the
                # join is written now so a detector landing later needs no change here.
                pitch_length_m=analysis.event.pitch_point.length_m if analysis.event.pitch_point else None,
                pitch_line_m=analysis.event.pitch_point.line_m if analysis.event.pitch_point else None,
                pitch_length=analysis.event.pitch_length,
            )
        )
    return fused


def fuse_match_deliveries(
    deliveries: pd.DataFrame, match_id: str, analyses: list[DeliveryAnalysis] = ()
) -> list[FusedDelivery]:
    """One `FusedDelivery` per ball-by-ball row for `match_id`. Video-derived fields are
    filled in wherever `analyses` has a `DeliveryAnalysis` for that exact delivery;
    everything else falls back to `FusedDelivery.from_delivery_only` -- e.g. because
    `BroadcastFeedAdapter` isn't implemented, or only some deliveries in the match were
    ever uploaded.

    Costs one scan of `deliveries` to select the match's rows, so **don't call this in a
    loop over many matches** -- use `fuse_matches`, which groups once. Handing this a
    frame already narrowed to the match (e.g. by
    `ParquetStore.read_deliveries(match_ids=[match_id])`) makes the scan free.
    """
    match_rows = deliveries[deliveries["match_id"] == match_id]
    return _fuse_rows(match_rows, {a.delivery: a for a in analyses})


def fuse_matches(
    deliveries: pd.DataFrame,
    match_ids: Optional[Iterable[str]] = None,
    analyses: Iterable[DeliveryAnalysis] = (),
) -> dict[str, list[FusedDelivery]]:
    """`{match_id: [FusedDelivery, ...]}` for every requested match, in one pass.

    The many-match counterpart to `fuse_match_deliveries`, and the reason it exists is
    complexity rather than convenience. Selecting one match's rows with
    `deliveries[deliveries["match_id"] == match_id]` scans the whole frame; doing that
    once per match is O(matches x deliveries). That was invisible while the warehouse
    held one competition (230 matches, ~52k deliveries, ~6 ms a match) and became the
    dominant cost as soon as it held several (8,026 matches, 1.8M deliveries, ~69 ms a
    match -- the per-match cost grew 12x because the *frame* grew, not the match).
    Grouping once is O(deliveries).

    `match_ids=None` fuses every match present in `deliveries`. Passing an explicit
    collection also fixes the output order, and match ids with no rows are simply
    absent from the result rather than mapping to an empty list -- "this match has no
    ball-by-ball rows" and "this match wasn't asked for" should not look identical to
    the caller.

    Returning a dict keyed by match rather than one flat list is deliberate: nearly
    every consumer needs rows grouped by match anyway (`aggregate_player_match_stats`
    takes one match's rows, and `match_date` is per match), and a flat list would make
    each of them re-group it.
    """
    by_ref = {a.delivery: a for a in analyses}
    if match_ids is not None:
        wanted = set(match_ids)
        if not wanted:
            return {}
        deliveries = deliveries[deliveries["match_id"].isin(wanted)]

    if deliveries.empty:
        return {}
    return {
        str(match_id): _fuse_rows(rows, by_ref)
        for match_id, rows in deliveries.groupby("match_id", sort=False)
    }


def fuse_and_aggregate(
    deliveries: pd.DataFrame,
    match_dates: Mapping[str, str],
    analyses: Iterable[DeliveryAnalysis] = (),
) -> tuple[list[PlayerMatchStats], list[FusedDelivery]]:
    """`(history, fused_rows)` for the matches in `match_dates`, in one grouped pass.

    The exact pair `rating.engine.compute_baseline` and `rate_player` consume, built
    without the caller having to write the group-by-then-aggregate loop itself -- which
    is where the quadratic scan kept getting reintroduced (see `fuse_matches`).

    `match_dates` doubles as the match selection *and* the date lookup
    `aggregate_player_match_stats` needs, so there is no way to fuse a match whose date
    the caller forgot to supply -- a `PlayerMatchStats` with an empty `match_date`
    sorts unpredictably in every last-N window downstream.
    """
    grouped = fuse_matches(deliveries, match_dates.keys(), analyses)
    history: list[PlayerMatchStats] = []
    fused_all: list[FusedDelivery] = []
    for match_id, rows in grouped.items():
        fused_all.extend(rows)
        history.extend(aggregate_player_match_stats(rows, match_dates[match_id]))
    return history, fused_all


def aggregate_player_match_stats(fused: list[FusedDelivery], match_date: str) -> list[PlayerMatchStats]:
    """One `PlayerMatchStats` per player who batted or bowled in `fused`'s match."""
    if not fused:
        return []
    match_id = fused[0].match_id
    players = {d.striker for d in fused} | {d.bowler for d in fused}
    return [_player_match_stats(player, match_id, match_date, fused) for player in sorted(players)]


def _player_match_stats(player: str, match_id: str, match_date: str, fused: list[FusedDelivery]) -> PlayerMatchStats:
    faced = [d for d in fused if d.striker == player]
    balls_faced_rows = [d for d in faced if d.extras_type != "wides"]  # a wide isn't a ball faced
    runs_scored = sum(d.runs_batter for d in faced)
    balls_faced = len(balls_faced_rows)

    bowled = [d for d in fused if d.bowler == player]
    legal = [d for d in bowled if d.extras_type not in ("wides", "noballs")]
    balls_bowled = len(legal)
    byes_and_legbyes = sum(d.runs_extras for d in bowled if d.extras_type in ("byes", "legbyes"))
    runs_conceded = sum(d.runs_total for d in bowled) - byes_and_legbyes
    wickets = [d for d in bowled if d.wicket_player_out and d.wicket_kind != "run out"]  # run outs aren't credited

    # CV biomechanics: only over deliveries this player bowled that actually have video
    # and a populated value -- both are None today (see FusedDelivery's docstring), so
    # these all resolve to None until a calibrated extractor exists.
    wrist_rows = [d for d in bowled if d.video_available and d.wrist_speed_ms is not None]
    elbow_rows = [d for d in bowled if d.video_available and d.bowling_elbow_extension_deg is not None]
    avg_wrist = round(sum(d.wrist_speed_ms for d in wrist_rows) / len(wrist_rows), 2) if wrist_rows else None
    max_wrist = max((d.wrist_speed_ms for d in wrist_rows), default=None)
    avg_elbow = round(sum(d.bowling_elbow_extension_deg for d in elbow_rows) / len(elbow_rows), 2) if elbow_rows else None
    max_elbow = max((d.bowling_elbow_extension_deg for d in elbow_rows), default=None)
    illegal_flag = None if max_elbow is None else max_elbow > _ILLEGAL_ACTION_ELBOW_DEG

    return PlayerMatchStats(
        player=player,
        match_id=match_id,
        match_date=match_date,
        runs_scored=runs_scored,
        balls_faced=balls_faced,
        fours=sum(1 for d in faced if d.runs_batter == 4),
        sixes=sum(1 for d in faced if d.runs_batter == 6),
        batting_strike_rate=round(100 * runs_scored / balls_faced, 2) if balls_faced else 0.0,
        balls_bowled=balls_bowled,
        runs_conceded=runs_conceded,
        wickets_taken=len(wickets),
        economy_rate=round(runs_conceded / (balls_bowled / 6), 2) if balls_bowled else 0.0,
        avg_wrist_speed_ms=avg_wrist,
        max_wrist_speed_ms=max_wrist,
        avg_bowling_elbow_extension_deg=avg_elbow,
        illegal_bowling_action_flag=illegal_flag,
    )


def refresh_rolling_summary(
    history: list[PlayerMatchStats], player: str, window_size: int = 5, last_updated: str = ""
) -> Optional[PlayerRollingSummary]:
    """Last `window_size` matches for `player` out of their full `history`, most recent
    first. Ties on `match_date` break by `match_id` descending -- the same deterministic
    rule `player_reports.last_n_match_ids` uses, so repeated calls against identical data
    can't silently return a different window (same-date ties are common; see that
    function's docstring). Returns `None` if `player` has no match stats at all.
    """
    mine = [s for s in history if s.player == player]
    if not mine:
        return None
    window = sorted(mine, key=lambda s: (s.match_date, s.match_id), reverse=True)[:window_size]

    total_runs = sum(s.runs_scored for s in window)
    total_balls = sum(s.balls_faced for s in window)
    total_wickets = sum(s.wickets_taken for s in window)
    total_runs_conceded = sum(s.runs_conceded for s in window)
    total_balls_bowled = sum(s.balls_bowled for s in window)

    wrist_speeds = [s.avg_wrist_speed_ms for s in window if s.avg_wrist_speed_ms is not None]
    max_wrists = [s.max_wrist_speed_ms for s in window if s.max_wrist_speed_ms is not None]
    elbow_avgs = [s.avg_bowling_elbow_extension_deg for s in window if s.avg_bowling_elbow_extension_deg is not None]

    return PlayerRollingSummary(
        player=player,
        window_size=window_size,
        matches_in_sample=len(window),
        last_updated=last_updated,
        total_runs=total_runs,
        avg_runs=round(total_runs / len(window), 2),
        strike_rate=round(100 * total_runs / total_balls, 2) if total_balls else 0.0,
        highest_score=max((s.runs_scored for s in window), default=0),
        boundary_percent=round(100 * sum(s.fours + s.sixes for s in window) / total_balls, 2) if total_balls else 0.0,
        total_wickets=total_wickets,
        bowling_average=round(total_runs_conceded / total_wickets, 2) if total_wickets else None,
        economy_rate=round(total_runs_conceded / (total_balls_bowled / 6), 2) if total_balls_bowled else 0.0,
        avg_wrist_speed_ms=round(sum(wrist_speeds) / len(wrist_speeds), 2) if wrist_speeds else None,
        max_wrist_speed_ms=max(max_wrists, default=None),
        avg_bowling_elbow_extension_deg=round(sum(elbow_avgs) / len(elbow_avgs), 2) if elbow_avgs else None,
    )


def refresh_all_rolling_summaries(
    history: list[PlayerMatchStats], window_size: int = 5, last_updated: str = ""
) -> list[PlayerRollingSummary]:
    """`refresh_rolling_summary` for every player who has at least one match in `history`."""
    players = sorted({s.player for s in history})
    summaries = (refresh_rolling_summary(history, p, window_size, last_updated) for p in players)
    return [s for s in summaries if s is not None]
