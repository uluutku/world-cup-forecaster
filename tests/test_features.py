import pandas as pd

from worldcup_predictor.features import SequentialFeatureBuilder


def test_features_are_pre_match_and_state_updates_after_result():
    rows = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-01-01"),
                "home_team": "A",
                "away_team": "B",
                "home_score": 3,
                "away_score": 0,
                "tournament": "Friendly",
                "country": "A",
                "neutral": False,
                "is_completed": True,
            },
            {
                "date": pd.Timestamp("2020-02-01"),
                "home_team": "A",
                "away_team": "B",
                "home_score": 1,
                "away_score": 1,
                "tournament": "Friendly",
                "country": "A",
                "neutral": False,
                "is_completed": True,
            },
        ]
    )
    transformed = SequentialFeatureBuilder().transform(rows)
    assert transformed.iloc[0]["elo_diff"] == 0
    assert transformed.iloc[1]["elo_diff"] > 0
    assert transformed.iloc[0]["target"] == 2
    assert transformed.iloc[1]["target"] == 1


def test_future_fixture_does_not_update_state():
    builder = SequentialFeatureBuilder()
    rows = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-01"),
                "home_team": "A",
                "away_team": "B",
                "home_score": float("nan"),
                "away_score": float("nan"),
                "tournament": "FIFA World Cup",
                "country": "United States",
                "neutral": True,
                "is_completed": False,
            }
        ]
    )
    builder.transform(rows)
    assert builder.states["A"].matches == 0
    assert builder.states["B"].matches == 0

