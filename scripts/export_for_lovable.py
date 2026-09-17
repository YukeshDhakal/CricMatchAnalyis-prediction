"""One-off export: compute real Overview/Player Performance/Rating & Suggestions data
using the actual pipeline (same functions app/streamlit_app.py calls) and print it as
JSON, for loading into the Lovable project's Supabase tables.

Not wired into any automated sync -- see DECISIONS.md's Lovable UI entry for why the
load step is manual (via the Lovable MCP's query_database) rather than handing this
script real Supabase credentials.

Usage: python scripts/export_for_lovable.py > /tmp/export.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from ingestion.storage import ParquetStore
from player_reports.stats import batting_summary, bowling_summary, last_n_match_ids
from rating.contracts import PlayerRole
from rating.engine import rate_player
from rating.llm import TemplateNoteWriter
from rating.pipeline import cohort_baseline, prepare_rating, resolve_scope
from rating.suggestions import flag_player, write_suggestions

WINDOW = 5
PLAYERS = [
    ("V Kohli", PlayerRole.TOP_ORDER_BATTER),
    ("JJ Bumrah", PlayerRole.DEATH_OVERS_BOWLER),
]


def _clean(obj):
    """Recursively make dataclass/enum/NaN values JSON-safe."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "value") and hasattr(obj, "name") and not isinstance(obj, (int, str)):
        return obj.value
    if isinstance(obj, float) and pd.isna(obj):
        return None
    return obj


def main() -> None:
    store = ParquetStore()
    matches = store.read_matches()
    deliveries = store.read_deliveries()

    overview = {
        "matches": int(len(matches)),
        "deliveries": int(len(deliveries)),
        "competitions": int(matches["competition"].nunique()),
        "players_seen": int(pd.concat([deliveries["striker"], deliveries["bowler"]]).nunique()),
    }

    recent = matches.tail(10)
    recent_matches = [
        {
            "match_id": row.match_id,
            "competition": row.competition,
            "teams": list(row.teams),
            "venue": row.venue,
            "outcome_winner": row.outcome_winner,
            "match_date": str(row.dates[0]) if row.dates else None,
        }
        for row in recent.itertuples()
    ]

    players_out = []
    batting_out = []
    bowling_out = []
    ratings_out = []
    suggestions_out = []

    for player, role in PLAYERS:
        match_ids = last_n_match_ids(deliveries, matches, player, n=WINDOW)
        window_label = f"last {WINDOW} matches"

        bat = batting_summary(deliveries, player, match_ids)
        bowl = bowling_summary(deliveries, player, match_ids)

        per_match = (
            deliveries[deliveries["match_id"].isin(match_ids) & (deliveries["striker"] == player)]
            .groupby("match_id")["runs_batter"]
            .sum()
        )
        per_match_runs = [{"match_id": mid, "runs": int(r)} for mid, r in per_match.items()]

        players_out.append({"player": player, "role": role.value})
        batting_out.append(
            {"player": player, "window_label": window_label, "stats": _clean(bat), "per_match_runs": per_match_runs}
        )
        bowling_out.append({"player": player, "window_label": window_label, "stats": _clean(bowl)})

        scope = resolve_scope(store, matches, player, WINDOW)
        baseline = cohort_baseline(store, scope[0])

        if baseline is None:
            print(f"# WARN: no baseline for {player}, skipping rating", file=sys.stderr)
            continue

        inputs = prepare_rating(store, matches, player, window_size=WINDOW, baseline=baseline, scope=scope)
        history = list(inputs.window_history)
        fused_all = list(inputs.window_fused)
        rating = rate_player(history, player, role, baseline, fused_all, window_size=WINDOW)
        if rating is None:
            print(f"# WARN: rate_player returned None for {player}", file=sys.stderr)
            continue

        ratings_out.append(
            {
                "player": player,
                "role": role.value,
                "composite": rating.composite,
                "technique": rating.pillars.technique,
                "execution": rating.pillars.execution,
                "game_impact": rating.pillars.game_impact,
                "consistency": rating.pillars.consistency,
                "unmeasured_pillars": [p.value for p in rating.unmeasured_pillars],
                "weights_used": _clean(rating.weights_used.as_dict()) if hasattr(rating.weights_used, "as_dict") else _clean(rating.weights_used.__dict__),
                "matches_in_sample": rating.matches_in_sample,
                "baseline_source": rating.baseline_source,
                "insufficient_data": rating.insufficient_data,
            }
        )

        all_flags = flag_player(fused_all, player, baseline)
        weakness_flags = [f for f in all_flags if f.is_concern]
        strength_flags = [f for f in all_flags if not f.is_concern]
        writer = TemplateNoteWriter()
        weaknesses = write_suggestions(weakness_flags, writer=writer, limit=5)
        strengths = write_suggestions(strength_flags, writer=writer, limit=3)

        for s in [*weaknesses, *strengths]:
            suggestions_out.append(
                {
                    "player": s.player,
                    "rank": s.rank,
                    "title": s.title,
                    "body": s.body,
                    "metric": s.flag.metric.value,
                    "phase": s.flag.phase,
                    "actual": s.flag.actual,
                    "baseline": s.flag.baseline,
                    "delta": s.flag.delta,
                    "relative_delta": s.flag.relative_delta,
                    "is_concern": s.flag.is_concern,
                    "severity": s.flag.severity,
                    "sample_size": s.flag.sample_size,
                    "baseline_source": s.flag.baseline_source,
                    "citations": [
                        {"match_id": c.match_id, "innings": c.innings, "over": c.over, "ball": c.ball}
                        for c in s.citations
                    ],
                    "citations_have_video": s.flag.citations_have_video,
                    "note_source": s.note_source,
                    "status": s.status,
                }
            )

    print(
        json.dumps(
            {
                "overview": overview,
                "recent_matches": recent_matches,
                "players": players_out,
                "batting_summaries": batting_out,
                "bowling_summaries": bowling_out,
                "ratings": ratings_out,
                "suggestions": suggestions_out,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
