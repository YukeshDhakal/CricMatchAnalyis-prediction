"""Stats-only player performance reports, computed directly from the ingested
ball-by-ball table (no video-derived signals -- there's no trained CV model or
real match footage yet, so this covers only the PRD's "Execution" and part of
"Game impact" pillars, not "Technique" or a composite rating).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def last_n_match_ids(deliveries: pd.DataFrame, matches: pd.DataFrame, player: str, n: int = 5) -> list[str]:
    """Match IDs the player batted, bowled, or was a non-striker in, most recent first.

    Ties on date are common -- T20 World Cups often play several matches on the same
    day -- so date alone isn't a unique sort key. Break ties by match_id (descending)
    for a deterministic order: pandas' default sort is quicksort, which is *not*
    stable, so relying on date alone could return a different match on repeated calls
    against the exact same data.
    """
    played_in = deliveries[
        (deliveries["striker"] == player) | (deliveries["non_striker"] == player) | (deliveries["bowler"] == player)
    ]["match_id"].unique()

    match_dates = matches.set_index("match_id")["dates"].apply(lambda d: d[0] if len(d) else "")
    played_dates = match_dates.loc[match_dates.index.intersection(played_in)]
    ordered = played_dates.rename("date").reset_index().sort_values(["date", "match_id"], ascending=False)
    return ordered["match_id"].head(n).tolist()


def _filter_matches(deliveries: pd.DataFrame, match_ids: Optional[list[str]]) -> pd.DataFrame:
    if match_ids is None:
        return deliveries
    return deliveries[deliveries["match_id"].isin(match_ids)]


def batting_summary(deliveries: pd.DataFrame, player: str, match_ids: Optional[list[str]] = None) -> dict:
    """Runs/balls faced/strike rate. A wide doesn't count as a ball faced; a no-ball
    does -- standard scoring convention."""
    faced = _filter_matches(deliveries, match_ids)
    faced = faced[faced["striker"] == player]

    balls_faced = faced[faced["extras_type"] != "wides"]
    per_innings_runs = faced.groupby(["match_id", "innings"])["runs_batter"].sum()
    dismissals = faced[faced["wicket_player_out"] == player]

    total_runs = int(faced["runs_batter"].sum())
    n_balls_faced = len(balls_faced)

    return {
        "innings": int(per_innings_runs.shape[0]),
        "runs": total_runs,
        "balls_faced": n_balls_faced,
        "fours": int((faced["runs_batter"] == 4).sum()),
        "sixes": int((faced["runs_batter"] == 6).sum()),
        "dismissals": int(dismissals.shape[0]),
        "highest_score": int(per_innings_runs.max()) if not per_innings_runs.empty else 0,
        "strike_rate": round(100 * total_runs / n_balls_faced, 2) if n_balls_faced else 0.0,
    }


def bowling_summary(deliveries: pd.DataFrame, player: str, match_ids: Optional[list[str]] = None) -> dict:
    """Wickets exclude run-outs (not credited to the bowler); byes/leg-byes aren't
    charged as runs conceded -- standard scoring convention."""
    bowled = _filter_matches(deliveries, match_ids)
    bowled = bowled[bowled["bowler"] == player]

    legal = bowled[~bowled["extras_type"].isin(["wides", "noballs"])]
    balls_bowled = len(legal)

    byes_and_legbyes = bowled[bowled["extras_type"].isin(["byes", "legbyes"])]["runs_extras"].sum()
    runs_conceded = int(bowled["runs_total"].sum() - byes_and_legbyes)

    wickets = bowled[bowled["wicket_player_out"].notna() & (bowled["wicket_kind"] != "run out")]

    overs_for_economy = balls_bowled / 6
    return {
        "overs": f"{balls_bowled // 6}.{balls_bowled % 6}",
        "runs_conceded": runs_conceded,
        "wickets": int(wickets.shape[0]),
        "economy": round(runs_conceded / overs_for_economy, 2) if overs_for_economy else 0.0,
    }
