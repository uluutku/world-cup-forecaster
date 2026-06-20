from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, SequentialFeatureBuilder
from .model import MatchEnsemble


def blind_team_tournament_forecast(
    matches: pd.DataFrame,
    team: str,
    tournament: str,
    year: int,
    training_years: int = 28,
    seed: int = 2026,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Forecast a team's full schedule from a frozen state before its first match."""
    schedule = matches.loc[
        matches["tournament"].eq(tournament)
        & matches["date"].dt.year.eq(year)
        & (matches["home_team"].eq(team) | matches["away_team"].eq(team))
    ].sort_values("date")
    if schedule.empty:
        raise ValueError(f"No {year} {tournament} fixtures found for {team}.")
    cutoff = schedule["date"].min()
    history = matches.loc[matches["is_completed"] & (matches["date"] < cutoff)].copy()
    builder = SequentialFeatureBuilder()
    features = builder.transform(history)
    train = features.loc[
        features["is_completed"]
        & (features["date"] >= cutoff - pd.DateOffset(years=training_years))
    ]
    model = MatchEnsemble(random_state=seed).fit(train, tune=True)

    rows = []
    outcome_names = np.array(["Away win", "Draw", "Home win"])
    for match in schedule.itertuples():
        row = pd.DataFrame(
            [
                builder.make_features(
                    match.home_team,
                    match.away_team,
                    match.date,
                    match.tournament,
                    match.country,
                    bool(match.neutral),
                )
            ]
        )
        probabilities = model.predict_proba(row[FEATURE_COLUMNS])[0]
        team_is_home = match.home_team == team
        opponent = match.away_team if team_is_home else match.home_team
        team_win_probability = probabilities[2] if team_is_home else probabilities[0]
        opponent_win_probability = probabilities[0] if team_is_home else probabilities[2]
        team_view = np.array([opponent_win_probability, probabilities[1], team_win_probability])
        team_pick = np.array([f"{opponent} win", "Draw", f"{team} win"])[
            int(team_view.argmax())
        ]
        model_class = int(probabilities.argmax())
        actual_class = None
        actual_outcome_probability = None
        correct = None
        actual_result = None
        if match.is_completed:
            actual_class = (
                2
                if match.home_score > match.away_score
                else 0
                if match.home_score < match.away_score
                else 1
            )
            actual_outcome_probability = float(probabilities[actual_class])
            correct = model_class == actual_class
            team_score = match.home_score if team_is_home else match.away_score
            opponent_score = match.away_score if team_is_home else match.home_score
            actual_result = (
                f"{team} win"
                if team_score > opponent_score
                else f"{opponent} win"
                if team_score < opponent_score
                else "Draw"
            )
        rows.append(
            {
                "date": match.date,
                "home_team": match.home_team,
                "away_team": match.away_team,
                "opponent": opponent,
                "home_score": match.home_score,
                "away_score": match.away_score,
                "is_completed": bool(match.is_completed),
                "p_home": probabilities[2],
                "p_draw": probabilities[1],
                "p_away": probabilities[0],
                "p_team_win": team_win_probability,
                "p_opponent_win": opponent_win_probability,
                "model_pick": outcome_names[model_class],
                "team_view_pick": team_pick,
                "actual_outcome_probability": actual_outcome_probability,
                "actual_result": actual_result,
                "correct_pick": correct,
            }
        )
    metadata = {
        "cutoff": cutoff.isoformat(),
        "last_training_match": train["date"].max().isoformat(),
        "training_matches": int(len(train)),
        "training_years": training_years,
        "blend_weights": model.blend_weights.tolist(),
        "temperature": model.temperature,
        "dixon_coles_rho": model.dc_rho,
        "state_updates_from_hidden_matches": 0,
    }
    return pd.DataFrame(rows), metadata


def blind_tournament_forecast(
    matches: pd.DataFrame,
    tournament: str,
    year: int,
    training_years: int = 28,
    seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Freeze before a tournament and forecast every scheduled match without state updates."""
    schedule = matches.loc[
        matches["tournament"].eq(tournament) & matches["date"].dt.year.eq(year)
    ].sort_values(["date", "home_team", "away_team"])
    if schedule.empty:
        raise ValueError(f"No {year} {tournament} fixtures found.")
    cutoff = schedule["date"].min()
    history = matches.loc[matches["is_completed"] & (matches["date"] < cutoff)].copy()
    builder = SequentialFeatureBuilder()
    features = builder.transform(history)
    train = features.loc[
        features["is_completed"]
        & (features["date"] >= cutoff - pd.DateOffset(years=training_years))
    ]
    model = MatchEnsemble(random_state=seed).fit(train, tune=True)
    outcome_names = np.array(["Away win", "Draw", "Home win"])
    rows = []
    for match in schedule.itertuples():
        feature_row = pd.DataFrame(
            [
                builder.make_features(
                    match.home_team,
                    match.away_team,
                    match.date,
                    match.tournament,
                    match.country,
                    bool(match.neutral),
                )
            ]
        )
        probabilities = model.predict_proba(feature_row[FEATURE_COLUMNS])[0]
        actual_class = None
        actual_probability = None
        correct = None
        actual_outcome = None
        if match.is_completed:
            actual_class = (
                2
                if match.home_score > match.away_score
                else 0
                if match.home_score < match.away_score
                else 1
            )
            actual_probability = float(probabilities[actual_class])
            correct = int(probabilities.argmax()) == actual_class
            actual_outcome = outcome_names[actual_class]
        rows.append(
            {
                "date": match.date,
                "home_team": match.home_team,
                "away_team": match.away_team,
                "home_score": match.home_score,
                "away_score": match.away_score,
                "is_completed": bool(match.is_completed),
                "p_home": probabilities[2],
                "p_draw": probabilities[1],
                "p_away": probabilities[0],
                "model_pick": outcome_names[int(probabilities.argmax())],
                "actual_outcome": actual_outcome,
                "actual_outcome_probability": actual_probability,
                "correct_pick": correct,
            }
        )
    forecasts = pd.DataFrame(rows)
    leaderboard = blind_team_leaderboard(forecasts)
    completed = forecasts.loc[forecasts["is_completed"]]
    actual = np.where(
        completed["home_score"] > completed["away_score"],
        2,
        np.where(completed["home_score"] < completed["away_score"], 0, 1),
    ).astype(int)
    probabilities = completed[["p_away", "p_draw", "p_home"]].to_numpy()
    observed = np.eye(3)[actual]
    metadata = {
        "cutoff": cutoff.isoformat(),
        "last_training_match": train["date"].max().isoformat(),
        "training_matches": int(len(train)),
        "training_years": training_years,
        "scheduled_matches": int(len(forecasts)),
        "scored_matches": int(len(completed)),
        "accuracy": float((probabilities.argmax(axis=1) == actual).mean()),
        "log_loss": float(
            -np.log(np.clip(probabilities[np.arange(len(actual)), actual], 1e-12, 1)).mean()
        ),
        "brier": float(np.mean(np.sum((probabilities - observed) ** 2, axis=1))),
        "blend_weights": model.blend_weights.tolist(),
        "temperature": model.temperature,
        "dixon_coles_rho": model.dc_rho,
        "state_updates_from_tournament": 0,
    }
    return forecasts, leaderboard, metadata


def blind_team_leaderboard(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Score frozen forecasts from each team's win/draw/loss perspective."""
    rows = []
    completed = forecasts.loc[forecasts["is_completed"]]
    for match in completed.itertuples():
        home_result = (
            2
            if match.home_score > match.away_score
            else 0
            if match.home_score < match.away_score
            else 1
        )
        for team, opponent, probabilities, result in (
            (
                match.home_team,
                match.away_team,
                np.array([match.p_away, match.p_draw, match.p_home]),
                home_result,
            ),
            (
                match.away_team,
                match.home_team,
                np.array([match.p_home, match.p_draw, match.p_away]),
                2 - home_result if home_result != 1 else 1,
            ),
        ):
            observed = np.eye(3)[result]
            rows.append(
                {
                    "team": team,
                    "opponent": opponent,
                    "correct": int(probabilities.argmax()) == result,
                    "log_loss": -np.log(np.clip(probabilities[result], 1e-12, 1)),
                    "brier": np.sum((probabilities - observed) ** 2),
                    "actual_probability": probabilities[result],
                }
            )
    return (
        pd.DataFrame(rows)
        .groupby("team", as_index=False)
        .agg(
            matches=("correct", "size"),
            accuracy=("correct", "mean"),
            log_loss=("log_loss", "mean"),
            brier=("brier", "mean"),
            mean_actual_probability=("actual_probability", "mean"),
        )
        .sort_values(["accuracy", "log_loss"], ascending=[False, True])
        .reset_index(drop=True)
    )
