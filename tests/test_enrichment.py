import pandas as pd

from worldcup_predictor.enrichment import (
    add_ranking_features,
    add_scorer_features,
)


def test_scorer_features_update_only_after_observed_match():
    matches = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-01-01"),
                "home_team": "A",
                "away_team": "B",
                "is_completed": True,
            },
            {
                "date": pd.Timestamp("2020-02-01"),
                "home_team": "A",
                "away_team": "B",
                "is_completed": True,
            },
        ]
    )
    goals = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-01-01"),
                "home_team": "A",
                "away_team": "B",
                "team": "A",
                "scorer": "Player A",
                "minute": 20,
                "own_goal": False,
                "penalty": False,
            }
        ]
    )
    enriched = add_scorer_features(pd.DataFrame(index=range(2)), matches, goals)
    assert enriched.iloc[0]["scorer_depth_diff"] == 0
    assert enriched.iloc[1]["scorer_depth_diff"] > 0


def test_ranking_merge_never_uses_future_release():
    features = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-06-01"),
                "home_team": "A",
                "away_team": "B",
            }
        ]
    )
    rankings = pd.DataFrame(
        [
            {"date": pd.Timestamp("2020-01-01"), "team": "A", "total_points": 1100},
            {"date": pd.Timestamp("2020-01-01"), "team": "B", "total_points": 900},
            {"date": pd.Timestamp("2020-12-01"), "team": "A", "total_points": 1800},
        ]
    )
    enriched = add_ranking_features(features, rankings)
    assert enriched.iloc[0]["fifa_points_diff"] == 0.5

