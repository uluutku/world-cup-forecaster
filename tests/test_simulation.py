import pandas as pd

from worldcup_predictor.experiments import blind_team_leaderboard
from worldcup_predictor.simulation import infer_groups


def test_infer_groups_from_round_robin_graph():
    frame = pd.DataFrame(
        [
            {"date": "2026-01-01", "home_team": "A", "away_team": "B"},
            {"date": "2026-01-02", "home_team": "A", "away_team": "C"},
            {"date": "2026-01-03", "home_team": "B", "away_team": "C"},
            {"date": "2026-01-01", "home_team": "D", "away_team": "E"},
            {"date": "2026-01-02", "home_team": "D", "away_team": "F"},
            {"date": "2026-01-03", "home_team": "E", "away_team": "F"},
        ]
    )
    groups = infer_groups(frame)
    assert set(groups["A"]) == {"A", "B", "C"}
    assert set(groups["B"]) == {"D", "E", "F"}


def test_blind_leaderboard_scores_both_team_perspectives():
    forecasts = pd.DataFrame(
        [
            {
                "home_team": "A",
                "away_team": "B",
                "home_score": 2,
                "away_score": 0,
                "is_completed": True,
                "p_home": 0.6,
                "p_draw": 0.25,
                "p_away": 0.15,
            }
        ]
    )
    leaderboard = blind_team_leaderboard(forecasts)
    assert set(leaderboard["team"]) == {"A", "B"}
    assert leaderboard["accuracy"].eq(1.0).all()
