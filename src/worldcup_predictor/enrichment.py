from __future__ import annotations

from collections import Counter, defaultdict, deque

import numpy as np
import pandas as pd

SCORER_FEATURE_COLUMNS = [
    "scorer_depth_diff",
    "scorer_form_diff",
    "star_dependency_diff",
    "penalty_reliance_diff",
    "late_goal_share_diff",
    "scorer_continuity_diff",
]

RANKING_FEATURE_COLUMNS = [
    "fifa_points_diff",
    "fifa_momentum_diff",
    "fifa_rank_missing",
]

SHOOTOUT_FEATURE_COLUMNS = [
    "shootout_experience_diff",
    "shootout_win_rate_diff",
]

SQUAD_FEATURE_COLUMNS = [
    "squad_age_diff",
    "squad_age_spread_diff",
    "returning_player_share_diff",
    "prior_wc_appearances_diff",
    "veteran_share_diff",
    "squad_data_missing",
]

ENRICHED_FEATURE_COLUMNS = (
    SCORER_FEATURE_COLUMNS
    + RANKING_FEATURE_COLUMNS
    + SHOOTOUT_FEATURE_COLUMNS
    + SQUAD_FEATURE_COLUMNS
)

TEAM_ALIASES = {
    "USA": "United States",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Türkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic",
    "Curaçao": "Curaçao",
    "Cabo Verde": "Cape Verde",
}


def add_scorer_features(
    features: pd.DataFrame,
    matches: pd.DataFrame,
    goalscorers: pd.DataFrame,
) -> pd.DataFrame:
    """Add pre-match scorer-depth and concentration features without future events."""
    ordered_matches = matches.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    events = goalscorers.copy()
    events["scorer"] = events["scorer"].fillna("Unknown scorer")
    event_lookup = {
        (date, home, away): group.to_dict(orient="records")
        for (date, home, away), group in events.groupby(
            ["date", "home_team", "away_team"], sort=False
        )
    }
    histories: dict[str, deque[dict[str, object]]] = defaultdict(deque)
    rows = []
    for match in ordered_matches.itertuples():
        date = match.date
        for team in (match.home_team, match.away_team):
            history = histories[team]
            while history and (date - history[0]["date"]).days > 730:
                history.popleft()
        home_profile = _scorer_profile(histories[match.home_team], date)
        away_profile = _scorer_profile(histories[match.away_team], date)
        rows.append(
            {
                "scorer_depth_diff": home_profile["depth"] - away_profile["depth"],
                "scorer_form_diff": home_profile["form"] - away_profile["form"],
                "star_dependency_diff": home_profile["star_dependency"]
                - away_profile["star_dependency"],
                "penalty_reliance_diff": home_profile["penalty_reliance"]
                - away_profile["penalty_reliance"],
                "late_goal_share_diff": home_profile["late_share"]
                - away_profile["late_share"],
                "scorer_continuity_diff": home_profile["continuity"]
                - away_profile["continuity"],
            }
        )
        if match.is_completed:
            for event in event_lookup.get(
                (match.date, match.home_team, match.away_team), []
            ):
                if event["team"] not in histories:
                    histories[event["team"]] = deque()
                histories[event["team"]].append(
                    {
                        "date": match.date,
                        "scorer": event["scorer"],
                        "penalty": bool(event["penalty"]),
                        "own_goal": bool(event["own_goal"]),
                        "minute": event["minute"],
                    }
                )
    enriched = features.reset_index(drop=True).copy()
    return pd.concat([enriched, pd.DataFrame(rows)], axis=1)


def _scorer_profile(history: deque[dict[str, object]], date: pd.Timestamp) -> dict[str, float]:
    valid = [
        event
        for event in history
        if not event["own_goal"] and (date - event["date"]).days <= 730
    ]
    recent_year = [event for event in valid if (date - event["date"]).days <= 365]
    recent_half = [event for event in valid if (date - event["date"]).days <= 180]
    previous_year = [
        event for event in valid if 365 < (date - event["date"]).days <= 730
    ]
    scorers = Counter(event["scorer"] for event in valid)
    recent_scorers = {event["scorer"] for event in recent_year}
    previous_scorers = {event["scorer"] for event in previous_year}
    total = len(valid)
    known_minutes = [
        float(event["minute"])
        for event in valid
        if pd.notna(event["minute"]) and str(event["minute"]).replace(".", "", 1).isdigit()
    ]
    return {
        "depth": float(np.log1p(len(recent_scorers))),
        "form": float(np.log1p(len(recent_half)) / 3.0),
        "star_dependency": max(scorers.values()) / total if total else 0.0,
        "penalty_reliance": (
            sum(bool(event["penalty"]) for event in valid) / total if total else 0.0
        ),
        "late_share": (
            sum(minute >= 75 for minute in known_minutes) / len(known_minutes)
            if known_minutes
            else 0.0
        ),
        "continuity": (
            len(recent_scorers & previous_scorers) / len(recent_scorers | previous_scorers)
            if recent_scorers | previous_scorers
            else 0.0
        ),
    }


def add_ranking_features(features: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    normalized = rankings[["team", "date", "total_points"]].dropna().copy()
    normalized["team"] = normalized["team"].replace(TEAM_ALIASES)
    normalized = normalized.sort_values(["date", "team"]).drop_duplicates(
        ["team", "date"], keep="last"
    )
    frame = features.reset_index(drop=True).copy()
    frame["_row"] = np.arange(len(frame))
    home = _asof_team_values(frame, normalized, "home_team", "home")
    away = _asof_team_values(frame, normalized, "away_team", "away")
    frame = frame.merge(home, on="_row", how="left").merge(away, on="_row", how="left")
    frame["fifa_points_diff"] = (
        frame["home_points"].fillna(1000) - frame["away_points"].fillna(1000)
    ) / 400.0
    frame["fifa_momentum_diff"] = (
        frame["home_momentum"].fillna(0) - frame["away_momentum"].fillna(0)
    ) / 100.0
    frame["fifa_rank_missing"] = (
        frame["home_points"].isna().astype(float)
        + frame["away_points"].isna().astype(float)
    ) / 2.0
    return frame.drop(
        columns=[
            "_row",
            "home_points",
            "home_momentum",
            "away_points",
            "away_momentum",
        ]
    )


def _asof_team_values(
    features: pd.DataFrame,
    rankings: pd.DataFrame,
    team_column: str,
    prefix: str,
) -> pd.DataFrame:
    left = features[["_row", "date", team_column]].rename(columns={team_column: "team"})
    current = pd.merge_asof(
        left.sort_values("date"),
        rankings.sort_values("date"),
        on="date",
        by="team",
        direction="backward",
    ).rename(columns={"total_points": f"{prefix}_points"})
    prior_left = left.copy()
    prior_left["date"] = prior_left["date"] - pd.DateOffset(years=1)
    prior = pd.merge_asof(
        prior_left.sort_values("date"),
        rankings.sort_values("date"),
        on="date",
        by="team",
        direction="backward",
    ).rename(columns={"total_points": f"{prefix}_prior"})
    output = current[["_row", f"{prefix}_points"]].merge(
        prior[["_row", f"{prefix}_prior"]], on="_row", how="left"
    )
    output[f"{prefix}_momentum"] = (
        output[f"{prefix}_points"] - output[f"{prefix}_prior"]
    )
    return output[["_row", f"{prefix}_points", f"{prefix}_momentum"]]


def add_shootout_features(
    features: pd.DataFrame,
    matches: pd.DataFrame,
    shootouts: pd.DataFrame,
) -> pd.DataFrame:
    ordered_matches = matches.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    lookup = {
        (row.date, row.home_team, row.away_team): row.winner
        for row in shootouts.itertuples()
    }
    attempts: Counter[str] = Counter()
    wins: Counter[str] = Counter()
    rows = []
    for match in ordered_matches.itertuples():
        home_attempts, away_attempts = attempts[match.home_team], attempts[match.away_team]
        rows.append(
            {
                "shootout_experience_diff": (
                    np.log1p(home_attempts) - np.log1p(away_attempts)
                )
                / 3.0,
                "shootout_win_rate_diff": (
                    wins[match.home_team] / home_attempts if home_attempts else 0.5
                )
                - (wins[match.away_team] / away_attempts if away_attempts else 0.5),
            }
        )
        winner = lookup.get((match.date, match.home_team, match.away_team))
        if winner:
            attempts[match.home_team] += 1
            attempts[match.away_team] += 1
            wins[winner] += 1
    return pd.concat(
        [features.reset_index(drop=True), pd.DataFrame(rows)], axis=1
    )


def build_enriched_features(
    base_features: pd.DataFrame,
    matches: pd.DataFrame,
    goalscorers: pd.DataFrame,
    shootouts: pd.DataFrame,
    rankings: pd.DataFrame,
    players: pd.DataFrame | None = None,
    squads: pd.DataFrame | None = None,
    appearances: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = add_scorer_features(base_features, matches, goalscorers)
    frame = add_ranking_features(frame, rankings)
    frame = add_shootout_features(frame, matches, shootouts)
    if players is not None and squads is not None and appearances is not None:
        frame = add_world_cup_squad_features(
            frame, players, squads, appearances
        )
    else:
        for column in SQUAD_FEATURE_COLUMNS[:-1]:
            frame[column] = 0.0
        frame["squad_data_missing"] = 1.0
    return frame


def add_world_cup_squad_features(
    features: pd.DataFrame,
    players: pd.DataFrame,
    squads: pd.DataFrame,
    appearances: pd.DataFrame,
) -> pd.DataFrame:
    """Add tournament-announced squad composition and prior World Cup experience."""
    player_birth = players.set_index("player_id")["birth_date"].to_dict()
    appearances = appearances.loc[
        appearances["tournament_name"].str.contains("Men's World Cup", na=False)
    ].copy()
    appearance_dates = (
        appearances.groupby("player_id")["match_date"]
        .apply(lambda values: sorted(values.dropna().tolist()))
        .to_dict()
    )
    profiles: dict[tuple[int, str], dict[str, float]] = {}
    men_squads = squads.loc[
        squads["tournament_name"].str.contains("Men's World Cup", na=False)
    ].copy()
    men_squads["year"] = men_squads["tournament_id"].str.extract(r"(\d{4})").astype(int)
    for (year, team), group in men_squads.groupby(["year", "team_name"]):
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        ages = []
        prior_counts = []
        for player_id in group["player_id"].dropna():
            birth_date = player_birth.get(player_id)
            if pd.notna(birth_date):
                ages.append((cutoff - birth_date).days / 365.25)
            prior_counts.append(
                sum(date < cutoff for date in appearance_dates.get(player_id, []))
            )
        profiles[(year, team)] = {
            "age": float(np.mean(ages)) if ages else 0.0,
            "age_spread": float(np.std(ages)) if ages else 0.0,
            "returning": float(np.mean(np.asarray(prior_counts) > 0))
            if prior_counts
            else 0.0,
            "prior_appearances": float(np.mean(prior_counts)) if prior_counts else 0.0,
            "veteran_share": float(np.mean(np.asarray(prior_counts) >= 5))
            if prior_counts
            else 0.0,
        }
    rows = []
    for match in features.itertuples():
        is_world_cup = match.tournament == "FIFA World Cup"
        year = match.date.year
        home = profiles.get((year, match.home_team)) if is_world_cup else None
        away = profiles.get((year, match.away_team)) if is_world_cup else None
        home = home or {}
        away = away or {}
        rows.append(
            {
                "squad_age_diff": (home.get("age", 0) - away.get("age", 0)) / 5.0,
                "squad_age_spread_diff": (
                    home.get("age_spread", 0) - away.get("age_spread", 0)
                )
                / 3.0,
                "returning_player_share_diff": home.get("returning", 0)
                - away.get("returning", 0),
                "prior_wc_appearances_diff": (
                    home.get("prior_appearances", 0)
                    - away.get("prior_appearances", 0)
                )
                / 5.0,
                "veteran_share_diff": home.get("veteran_share", 0)
                - away.get("veteran_share", 0),
                "squad_data_missing": float(not home or not away),
            }
        )
    return pd.concat(
        [features.reset_index(drop=True), pd.DataFrame(rows)], axis=1
    )


def data_coverage_report(
    matches: pd.DataFrame,
    goalscorers: pd.DataFrame,
    rankings: pd.DataFrame,
    shootouts: pd.DataFrame,
    squads: pd.DataFrame | None = None,
) -> pd.DataFrame:
    completed = matches.loc[matches["is_completed"]]
    scorer_keys = goalscorers[["date", "home_team", "away_team"]].drop_duplicates()
    match_keys = completed[["date", "home_team", "away_team"]]
    scorer_covered = match_keys.merge(
        scorer_keys.assign(covered=True),
        on=["date", "home_team", "away_team"],
        how="left",
    )["covered"].notna()
    rows = [
            {
                "source": "International results",
                "signal_family": "scores, venue, competition",
                "rows": len(matches),
                "first_date": matches["date"].min(),
                "last_date": matches["date"].max(),
                "coverage": 1.0,
                "access": "open",
            },
            {
                "source": "Goalscorers",
                "signal_family": "players, scorer depth, penalties, timing",
                "rows": len(goalscorers),
                "first_date": goalscorers["date"].min(),
                "last_date": goalscorers["date"].max(),
                "coverage": float(scorer_covered.mean()),
                "access": "open",
            },
            {
                "source": "Historical FIFA ranking",
                "signal_family": "external team strength",
                "rows": len(rankings),
                "first_date": rankings["date"].min(),
                "last_date": rankings["date"].max(),
                "coverage": float(rankings["total_points"].notna().mean()),
                "access": "open, scraped from FIFA",
            },
            {
                "source": "Shootouts",
                "signal_family": "penalty experience",
                "rows": len(shootouts),
                "first_date": shootouts["date"].min(),
                "last_date": shootouts["date"].max(),
                "coverage": float(len(shootouts) / max(len(completed), 1)),
                "access": "open",
            },
        ]
    if squads is not None:
        rows.append(
            {
                "source": "Fjelstul World Cup Database",
                "signal_family": "squads, age, positions, player experience",
                "rows": len(squads),
                "first_date": pd.Timestamp("1930-01-01"),
                "last_date": pd.Timestamp("2022-12-18"),
                "coverage": float(
                    squads["tournament_name"]
                    .str.contains("Men's World Cup", na=False)
                    .mean()
                ),
                "access": "open",
            }
        )
    return pd.DataFrame(rows)
