from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .config import LINEUP_SUMMARY_PATH

LINEUP_FEATURE_COLUMNS = [
    "lineup_continuity_diff",
    "lineup_changes_diff",
    "lineup_experience_diff",
    "lineup_debut_share_diff",
    "lineup_keeper_change_diff",
    "lineup_defender_count_diff",
    "lineup_midfielder_count_diff",
    "lineup_forward_count_diff",
    "lineup_data_missing",
]


def build_statsbomb_lineup_summary(
    event_summary: pd.DataFrame,
    refresh: bool = False,
    workers: int = 12,
) -> pd.DataFrame:
    """Build pre-kickoff lineup descriptors from StatsBomb lineups, without using the result."""
    if LINEUP_SUMMARY_PATH.exists() and not refresh:
        return pd.read_parquet(LINEUP_SUMMARY_PATH)
    matches = (
        event_summary[
            [
                "match_id",
                "date",
                "competition",
                "season",
                "home_team",
                "away_team",
            ]
        ]
        .drop_duplicates("match_id")
        .sort_values(["date", "match_id"])
    )
    raw_rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_fetch_match_lineups, row._asdict()): row.match_id
            for row in matches.itertuples(index=False)
        }
        for future in as_completed(futures):
            try:
                raw_rows.extend(future.result())
            except Exception:
                continue
    raw = pd.DataFrame(raw_rows)
    if raw.empty:
        return raw
    enriched = _sequential_lineup_features(raw)
    LINEUP_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(LINEUP_SUMMARY_PATH, index=False)
    return enriched


def _fetch_match_lineups(match: dict[str, object]) -> list[dict[str, object]]:
    payload = _get_json(
        "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"
        f"lineups/{int(match['match_id'])}.json"
    )
    rows = []
    for team in payload:
        starters = []
        roles = []
        keeper = None
        for player in team.get("lineup", []):
            starting_positions = [
                position
                for position in player.get("positions", [])
                if position.get("start_reason") == "Starting XI"
                and position.get("from") == "00:00"
            ]
            if not starting_positions:
                continue
            starters.append(int(player["player_id"]))
            role = str(starting_positions[0].get("position", ""))
            roles.append(role)
            if "Goalkeeper" in role:
                keeper = int(player["player_id"])
        rows.append(
            {
                **match,
                "team": team["team_name"],
                "starter_ids": json.dumps(sorted(starters)),
                "keeper_id": keeper,
                "starter_count": len(starters),
                "defender_count": sum(_role_group(role) == "defender" for role in roles),
                "midfielder_count": sum(
                    _role_group(role) == "midfielder" for role in roles
                ),
                "forward_count": sum(_role_group(role) == "forward" for role in roles),
            }
        )
    return rows


def _sequential_lineup_features(raw: pd.DataFrame) -> pd.DataFrame:
    previous: dict[str, set[int]] = {}
    previous_keeper: dict[str, int | None] = {}
    player_starts: dict[tuple[str, int], int] = {}
    rows = []
    for row in raw.sort_values(["date", "match_id", "team"]).itertuples(index=False):
        starters = set(json.loads(row.starter_ids))
        old = previous.get(row.team)
        overlap = len(starters & old) if old else 0
        experience = [player_starts.get((row.team, player), 0) for player in starters]
        rows.append(
            {
                **row._asdict(),
                "lineup_continuity": overlap / max(len(starters), 1),
                "lineup_changes": (len(starters) - overlap) / max(len(starters), 1),
                "lineup_experience": float(np.mean(experience)) if experience else 0.0,
                "lineup_debut_share": (
                    float(np.mean(np.asarray(experience) == 0)) if experience else 1.0
                ),
                "lineup_keeper_change": float(
                    row.team in previous_keeper
                    and previous_keeper[row.team] != row.keeper_id
                ),
            }
        )
        previous[row.team] = starters
        previous_keeper[row.team] = row.keeper_id
        for player in starters:
            key = (row.team, player)
            player_starts[key] = player_starts.get(key, 0) + 1
    return pd.DataFrame(rows)


def add_announced_lineup_features(
    features: pd.DataFrame,
    lineup_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Join starting-XI features that are valid only after lineups are announced."""
    frame = features.reset_index(drop=True).copy()
    match_rows = lineup_summary[
        ["date", "home_team", "away_team", "team"]
        + [
            "lineup_continuity",
            "lineup_changes",
            "lineup_experience",
            "lineup_debut_share",
            "lineup_keeper_change",
            "defender_count",
            "midfielder_count",
            "forward_count",
        ]
    ]
    metrics = [
        "lineup_continuity",
        "lineup_changes",
        "lineup_experience",
        "lineup_debut_share",
        "lineup_keeper_change",
        "defender_count",
        "midfielder_count",
        "forward_count",
    ]
    home = match_rows.loc[
        match_rows["team"].eq(match_rows["home_team"])
    ].rename(columns={name: f"home_{name}" for name in metrics})
    home = home.drop(columns="team").drop_duplicates(
        ["date", "home_team", "away_team"]
    )
    away = match_rows.loc[
        match_rows["team"].eq(match_rows["away_team"])
    ].rename(columns={name: f"away_{name}" for name in metrics})
    away = away.drop(columns="team").drop_duplicates(
        ["date", "home_team", "away_team"]
    )
    frame = frame.merge(
        home[["date", "home_team", "away_team"] + [f"home_{name}" for name in metrics]],
        on=["date", "home_team", "away_team"],
        how="left",
    )
    frame = frame.merge(
        away[["date", "home_team", "away_team"] + [f"away_{name}" for name in metrics]],
        on=["date", "home_team", "away_team"],
        how="left",
    )
    for metric in metrics:
        output = (
            f"{metric}_diff"
            if metric.startswith("lineup_")
            else f"lineup_{metric}_diff"
        )
        frame[output] = frame[f"home_{metric}"].fillna(0) - frame[
            f"away_{metric}"
        ].fillna(0)
    frame["lineup_data_missing"] = (
        frame[["home_lineup_continuity", "away_lineup_continuity"]]
        .isna()
        .any(axis=1)
        .astype(float)
    )
    return frame.drop(
        columns=[
            f"{side}_{metric}"
            for side in ("home", "away")
            for metric in metrics
        ]
    )


def _role_group(role: str) -> str:
    value = role.lower()
    if "back" in value or "center back" in value:
        return "defender"
    if "midfield" in value or "wing" in value:
        return "midfielder"
    if "forward" in value or "striker" in value:
        return "forward"
    return "other"


def _get_json(url: str, retries: int = 4):
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": "WorldCupIntelligence/2.0"})
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.2 * (attempt + 1))
    return []
