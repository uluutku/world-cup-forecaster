import pandas as pd
import pytest

from worldcup_predictor.conditions import add_condition_features
from worldcup_predictor.lineup_data import (
    add_announced_lineup_features,
)
from worldcup_predictor.squad_analytics import squad_analytics_2026


def test_condition_features_expose_missingness_without_nan():
    features = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2022-11-20"),
                "home_team": "Qatar",
                "away_team": "Ecuador",
            }
        ]
    )
    conditions = pd.DataFrame(
        columns=["date", "home_team", "away_team"]
    )
    enriched = add_condition_features(features, conditions)
    assert enriched.iloc[0]["weather_missing"] == 1
    assert enriched.iloc[0]["travel_missing"] == 1
    assert enriched.iloc[0]["weather_temperature"] == 0


def test_announced_lineup_features_join_only_the_exact_fixture():
    features = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2022-11-20"),
                "home_team": "A",
                "away_team": "B",
            },
            {
                "date": pd.Timestamp("2022-11-21"),
                "home_team": "A",
                "away_team": "B",
            },
        ]
    )
    lineups = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2022-11-20"),
                "home_team": "A",
                "away_team": "B",
                "team": "A",
                "lineup_continuity": 0.8,
                "lineup_changes": 0.2,
                "lineup_experience": 5.0,
                "lineup_debut_share": 0.1,
                "lineup_keeper_change": 0.0,
                "defender_count": 4,
                "midfielder_count": 4,
                "forward_count": 2,
            },
            {
                "date": pd.Timestamp("2022-11-20"),
                "home_team": "A",
                "away_team": "B",
                "team": "B",
                "lineup_continuity": 0.5,
                "lineup_changes": 0.5,
                "lineup_experience": 2.0,
                "lineup_debut_share": 0.3,
                "lineup_keeper_change": 1.0,
                "defender_count": 5,
                "midfielder_count": 3,
                "forward_count": 2,
            },
        ]
    )
    enriched = add_announced_lineup_features(features, lineups)
    assert enriched.iloc[0]["lineup_continuity_diff"] == pytest.approx(0.3)
    assert enriched.iloc[0]["lineup_defender_count_diff"] == -1
    assert enriched.iloc[0]["lineup_data_missing"] == 0
    assert enriched.iloc[1]["lineup_data_missing"] == 1


def test_squad_analytics_builds_current_snapshot_without_official_pdf():
    squads = [
        {
            "name": "Example",
            "fifa_code": "EXP",
            "group": "A",
            "players": [
                {
                    "number": 1,
                    "pos": "GK",
                    "name": "Keeper",
                    "date_of_birth": "1995-01-01",
                    "club": {"name": "Club A", "country": "ENG"},
                },
                {
                    "number": 9,
                    "pos": "FW",
                    "name": "Forward",
                    "date_of_birth": "2003-01-01",
                    "club": {"name": "Club B", "country": "EXP"},
                },
            ],
        }
    ]
    players, teams = squad_analytics_2026(squads)
    assert len(players) == 2
    assert teams.iloc[0]["players"] == 2
    assert teams.iloc[0]["top_five_league_share"] == 0.5
