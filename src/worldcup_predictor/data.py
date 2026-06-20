from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from .config import (
    FIFA_RANKINGS_URL,
    FJELSTUL_BASE_URL,
    GOALSCORERS_URL,
    RAW_DIR,
    RESULTS_URL,
    SHOOTOUTS_URL,
    SQUAD_2026_PATH,
)

REQUIRED_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "country",
    "neutral",
}


def download_results(force: bool = False, url: str = RESULTS_URL) -> Path:
    """Download the maintained international results dataset."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    destination = RAW_DIR / "results.csv"
    if force or not destination.exists():
        urlretrieve(url, destination)
    return destination


def download_auxiliary_data(force: bool = False) -> dict[str, Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    sources = {
        "goalscorers": (GOALSCORERS_URL, RAW_DIR / "goalscorers.csv"),
        "shootouts": (SHOOTOUTS_URL, RAW_DIR / "shootouts.csv"),
        "fifa_rankings": (FIFA_RANKINGS_URL, RAW_DIR / "fifa_rankings.csv"),
    }
    paths = {}
    for name, (url, destination) in sources.items():
        if force or not destination.exists():
            urlretrieve(url, destination)
        paths[name] = destination
    return paths


def load_auxiliary_data(
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = download_auxiliary_data(force=refresh)
    goalscorers = pd.read_csv(paths["goalscorers"], parse_dates=["date"])
    goalscorers["own_goal"] = goalscorers["own_goal"].astype(bool)
    goalscorers["penalty"] = goalscorers["penalty"].astype(bool)
    shootouts = pd.read_csv(paths["shootouts"], parse_dates=["date"])
    rankings = pd.read_csv(paths["fifa_rankings"], parse_dates=["date"])
    rankings["total_points"] = pd.to_numeric(rankings["total_points"], errors="coerce")
    return goalscorers, shootouts, rankings


def load_world_cup_player_data(
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = ("players.csv", "squads.csv", "player_appearances.csv")
    paths = {}
    for filename in files:
        destination = RAW_DIR / f"worldcup_{filename}"
        if refresh or not destination.exists():
            urlretrieve(f"{FJELSTUL_BASE_URL}/{filename}", destination)
        paths[filename] = destination
    players = pd.read_csv(paths["players.csv"])
    players["birth_date"] = pd.to_datetime(players["birth_date"], errors="coerce")
    squads = pd.read_csv(paths["squads.csv"])
    appearances = pd.read_csv(
        paths["player_appearances.csv"], parse_dates=["match_date"]
    )
    return players, squads, appearances


def load_world_cup_reference_data(
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for filename in ("matches.csv", "stadiums.csv"):
        destination = RAW_DIR / f"worldcup_{filename}"
        if refresh or not destination.exists():
            urlretrieve(f"{FJELSTUL_BASE_URL}/{filename}", destination)
        paths[filename] = destination
    matches = pd.read_csv(paths["matches.csv"], parse_dates=["match_date"])
    stadiums = pd.read_csv(paths["stadiums.csv"])
    return matches, stadiums


def load_2026_squads(refresh: bool = False) -> list[dict[str, object]]:
    SQUAD_2026_PATH.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not SQUAD_2026_PATH.exists():
        urlretrieve(
            "https://raw.githubusercontent.com/openfootball/worldcup.json/"
            "master/2026/worldcup.squads.json",
            SQUAD_2026_PATH,
        )
    import json

    return json.loads(SQUAD_2026_PATH.read_text(encoding="utf-8"))


def load_results(path: str | Path | None = None, refresh: bool = False) -> pd.DataFrame:
    source = Path(path) if path else download_results(force=refresh)
    frame = pd.read_csv(source, parse_dates=["date"])
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    frame = frame.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    frame["neutral"] = frame["neutral"].astype(bool)
    frame["is_completed"] = frame[["home_score", "away_score"]].notna().all(axis=1)
    return frame


def split_completed_fixtures(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed = frame.loc[frame["is_completed"]].copy()
    fixtures = frame.loc[~frame["is_completed"]].copy()
    return completed, fixtures


def world_cup_2026(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[
        frame["tournament"].eq("FIFA World Cup") & frame["date"].dt.year.eq(2026)
    ].copy()
