from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .config import STATSBOMB_SUMMARY_PATH

STATSBOMB_FEATURE_COLUMNS = [
    "event_xg_diff",
    "event_xg_against_diff",
    "event_xg_balance_diff",
    "event_shots_diff",
    "event_shot_quality_diff",
    "event_possession_diff",
    "event_pass_completion_diff",
    "event_pressures_diff",
    "event_counterpress_diff",
    "event_set_piece_xg_diff",
    "event_transition_xg_diff",
    "event_final_third_diff",
    "event_data_missing",
]

INTERNATIONAL_SELECTION = {
    ("FIFA World Cup", "2018"),
    ("FIFA World Cup", "2022"),
    ("UEFA Euro", "2020"),
    ("UEFA Euro", "2024"),
    ("Copa America", "2024"),
    ("African Cup of Nations", "2023"),
}


def build_statsbomb_summary(
    refresh: bool = False, workers: int = 12
) -> pd.DataFrame:
    if STATSBOMB_SUMMARY_PATH.exists() and not refresh:
        return pd.read_parquet(STATSBOMB_SUMMARY_PATH)
    competitions = _get_json(
        "https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json"
    )
    selected = [
        item
        for item in competitions
        if (item["competition_name"], item["season_name"])
        in INTERNATIONAL_SELECTION
    ]
    matches = []
    for item in selected:
        url = (
            "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"
            f"matches/{item['competition_id']}/{item['season_id']}.json"
        )
        for match in _get_json(url):
            matches.append(
                {
                    "match_id": match["match_id"],
                    "date": pd.Timestamp(match["match_date"]),
                    "competition": item["competition_name"],
                    "season": item["season_name"],
                    "home_team": match["home_team"]["home_team_name"],
                    "away_team": match["away_team"]["away_team_name"],
                }
            )
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_summarize_match, match): match for match in matches
        }
        for future in as_completed(futures):
            try:
                rows.extend(future.result())
            except Exception:
                continue
    frame = pd.DataFrame(rows).sort_values(["date", "team"]).reset_index(drop=True)
    STATSBOMB_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(STATSBOMB_SUMMARY_PATH, index=False)
    return frame


def _summarize_match(match: dict[str, object]) -> list[dict[str, object]]:
    events = _get_json(
        "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"
        f"events/{match['match_id']}.json",
        retries=4,
    )
    teams = [match["home_team"], match["away_team"]]
    stats = {team: _empty_stats() for team in teams}
    possession_counts = {team: 0 for team in teams}
    pass_attempts = {team: 0 for team in teams}
    for event in events:
        if int(event.get("period", 1)) > 4:
            continue
        team = event.get("team", {}).get("name")
        if team not in stats:
            continue
        event_type = event.get("type", {}).get("name")
        possession_team = event.get("possession_team", {}).get("name")
        if possession_team in possession_counts:
            possession_counts[possession_team] += 1
        if event_type == "Shot":
            shot = event.get("shot", {})
            xg = float(shot.get("statsbomb_xg", 0) or 0)
            stats[team]["xg"] += xg
            stats[team]["shots"] += 1
            pattern = event.get("play_pattern", {}).get("name", "")
            shot_type = shot.get("type", {}).get("name", "")
            if pattern in {"From Corner", "From Free Kick"} or shot_type in {
                "Free Kick",
                "Corner",
            }:
                stats[team]["set_piece_xg"] += xg
            if pattern in {"From Counter", "From Keeper"}:
                stats[team]["transition_xg"] += xg
        elif event_type == "Pass":
            pass_attempts[team] += 1
            if "outcome" not in event.get("pass", {}):
                stats[team]["completed_passes"] += 1
            end_location = event.get("pass", {}).get("end_location")
            if end_location and end_location[0] >= 80:
                stats[team]["final_third_entries"] += 1
        elif event_type == "Carry":
            end_location = event.get("carry", {}).get("end_location")
            if end_location and end_location[0] >= 80:
                stats[team]["final_third_entries"] += 1
        elif event_type == "Pressure":
            stats[team]["pressures"] += 1
            stats[team]["counterpressures"] += int(
                bool(event.get("counterpress", False))
            )
    total_possessions = max(sum(possession_counts.values()), 1)
    output = []
    for team, opponent in ((teams[0], teams[1]), (teams[1], teams[0])):
        team_stats = stats[team]
        opponent_stats = stats[opponent]
        shots = max(team_stats["shots"], 1)
        output.append(
            {
                **match,
                "team": team,
                "opponent": opponent,
                "xg": team_stats["xg"],
                "xg_against": opponent_stats["xg"],
                "shots": team_stats["shots"],
                "shot_quality": team_stats["xg"] / shots,
                "possession": possession_counts[team] / total_possessions,
                "pass_completion": team_stats["completed_passes"]
                / max(pass_attempts[team], 1),
                "pressures": team_stats["pressures"] / 100.0,
                "counterpressures": team_stats["counterpressures"] / 25.0,
                "set_piece_xg": team_stats["set_piece_xg"],
                "transition_xg": team_stats["transition_xg"],
                "final_third_entries": team_stats["final_third_entries"] / 50.0,
            }
        )
    return output


def _empty_stats() -> dict[str, float]:
    return {
        "xg": 0.0,
        "shots": 0.0,
        "completed_passes": 0.0,
        "pressures": 0.0,
        "counterpressures": 0.0,
        "set_piece_xg": 0.0,
        "transition_xg": 0.0,
        "final_third_entries": 0.0,
    }


def add_statsbomb_features(
    features: pd.DataFrame, summary: pd.DataFrame
) -> pd.DataFrame:
    metrics = [
        "xg",
        "xg_against",
        "shots",
        "shot_quality",
        "possession",
        "pass_completion",
        "pressures",
        "counterpressures",
        "set_piece_xg",
        "transition_xg",
        "final_third_entries",
    ]
    profiles = summary.sort_values(["team", "date"]).copy()
    for metric in metrics:
        profiles[f"profile_{metric}"] = profiles.groupby("team")[metric].transform(
            lambda values: values.ewm(alpha=0.28, adjust=False).mean()
        )
    profiles["profile_matches"] = profiles.groupby("team").cumcount() + 1
    profile_columns = [f"profile_{metric}" for metric in metrics] + [
        "profile_matches"
    ]
    frame = features.reset_index(drop=True).copy()
    frame["_row"] = np.arange(len(frame))
    home = _asof_profiles(frame, profiles, "home_team", "home", profile_columns)
    away = _asof_profiles(frame, profiles, "away_team", "away", profile_columns)
    frame = frame.merge(home, on="_row", how="left").merge(away, on="_row", how="left")
    mapping = {
        "event_xg_diff": "xg",
        "event_xg_against_diff": "xg_against",
        "event_shots_diff": "shots",
        "event_shot_quality_diff": "shot_quality",
        "event_possession_diff": "possession",
        "event_pass_completion_diff": "pass_completion",
        "event_pressures_diff": "pressures",
        "event_counterpress_diff": "counterpressures",
        "event_set_piece_xg_diff": "set_piece_xg",
        "event_transition_xg_diff": "transition_xg",
        "event_final_third_diff": "final_third_entries",
    }
    for feature, metric in mapping.items():
        frame[feature] = (
            frame[f"home_profile_{metric}"].fillna(0)
            - frame[f"away_profile_{metric}"].fillna(0)
        )
    frame["event_xg_balance_diff"] = (
        frame["event_xg_diff"] - frame["event_xg_against_diff"]
    )
    frame["event_data_missing"] = (
        frame[["home_profile_matches", "away_profile_matches"]]
        .isna()
        .any(axis=1)
        .astype(float)
    )
    return frame.drop(
        columns=["_row"]
        + [f"{side}_{column}" for side in ("home", "away") for column in profile_columns]
    )


def _asof_profiles(
    features: pd.DataFrame,
    profiles: pd.DataFrame,
    team_column: str,
    prefix: str,
    profile_columns: list[str],
) -> pd.DataFrame:
    left = features[["_row", "date", team_column]].rename(
        columns={team_column: "team"}
    )
    right = profiles[["team", "date"] + profile_columns].copy()
    merged = pd.merge_asof(
        left.sort_values("date"),
        right.sort_values("date"),
        on="date",
        by="team",
        direction="backward",
        allow_exact_matches=False,
    )
    return merged[["_row"] + profile_columns].rename(
        columns={column: f"{prefix}_{column}" for column in profile_columns}
    )


def _get_json(url: str, retries: int = 3):
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": "WorldCupForecaster/2.0"})
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.2 * (attempt + 1))
    return []
