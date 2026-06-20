from __future__ import annotations

from dataclasses import dataclass, field
from math import log1p

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "elo_diff",
    "elo_mean",
    "rating_confidence_diff",
    "rating_uncertainty",
    "volatility_diff",
    "form_diff",
    "momentum_diff",
    "goals_for_diff",
    "goals_against_diff",
    "goal_balance_diff",
    "attack_trend_diff",
    "defence_trend_diff",
    "clean_sheet_diff",
    "scoring_rate_diff",
    "consistency_diff",
    "win_rate_diff",
    "sos_diff",
    "prestige_diff",
    "major_form_diff",
    "experience_diff",
    "activity_diff",
    "rest_diff",
    "h2h_form",
    "h2h_goal_balance",
    "h2h_experience",
    "neutral",
    "home_advantage",
    "host_home",
    "host_away",
    "importance",
    "is_world_cup",
    "year",
]

MAJOR_TOURNAMENT_TOKENS = (
    "FIFA World Cup",
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "CONCACAF",
    "Oceania Nations Cup",
)


def tournament_importance(name: str) -> float:
    if name == "FIFA World Cup":
        return 1.0
    if "World Cup qualification" in name or "World Cup qualifier" in name:
        return 0.82
    if any(token in name for token in MAJOR_TOURNAMENT_TOKENS):
        return 0.88
    if "Nations League" in name:
        return 0.67
    if name == "Friendly":
        return 0.28
    return 0.52


@dataclass
class TeamState:
    elo: float = 1500.0
    rating_sigma: float = 300.0
    volatility: float = 0.12
    form: float = 0.5
    momentum_short: float = 0.5
    momentum_long: float = 0.5
    goals_for: float = 1.25
    goals_against: float = 1.25
    attack_long: float = 1.25
    defence_long: float = 1.25
    clean_sheet_rate: float = 0.25
    scoring_rate: float = 0.75
    result_variance: float = 0.20
    win_rate: float = 0.33
    sos: float = 1500.0
    prestige: float = 1500.0
    major_form: float = 0.5
    matches: int = 0
    last_date: pd.Timestamp | None = None
    recent_dates: list[pd.Timestamp] = field(default_factory=list)


@dataclass
class HeadToHeadState:
    home_points: float = 0.5
    home_goal_balance: float = 0.0
    matches: int = 0


class SequentialFeatureBuilder:
    """Build pre-match features while updating state only after each result."""

    def __init__(self, alpha: float = 0.12):
        self.alpha = alpha
        self.states: dict[str, TeamState] = {}
        self.h2h: dict[tuple[str, str], HeadToHeadState] = {}

    def _state(self, team: str) -> TeamState:
        if team not in self.states:
            self.states[team] = TeamState()
        return self.states[team]

    @staticmethod
    def _rest_days(state: TeamState, date: pd.Timestamp) -> float:
        if state.last_date is None:
            return 30.0
        return float(np.clip((date - state.last_date).days, 2, 90))

    @staticmethod
    def _canonical_h2h(home: str, away: str) -> tuple[tuple[str, str], bool]:
        ordered = tuple(sorted((home, away)))
        return ordered, home == ordered[0]

    def make_features(
        self,
        home: str,
        away: str,
        date: pd.Timestamp,
        tournament: str,
        country: str,
        neutral: bool,
    ) -> dict[str, float]:
        hs, aws = self._state(home), self._state(away)
        key, home_is_first = self._canonical_h2h(home, away)
        h2h = self.h2h.get(key, HeadToHeadState())
        h2h_form = h2h.home_points if home_is_first else 1.0 - h2h.home_points
        h2h_balance = h2h.home_goal_balance * (1 if home_is_first else -1)
        importance = tournament_importance(tournament)
        home_rest = self._rest_days(hs, date)
        away_rest = self._rest_days(aws, date)
        home_activity = sum((date - seen).days <= 365 for seen in hs.recent_dates)
        away_activity = sum((date - seen).days <= 365 for seen in aws.recent_dates)
        return {
            "elo_diff": (hs.elo - aws.elo) / 400.0,
            "elo_mean": (hs.elo + aws.elo - 3000.0) / 400.0,
            "rating_confidence_diff": (aws.rating_sigma - hs.rating_sigma) / 300.0,
            "rating_uncertainty": (hs.rating_sigma + aws.rating_sigma) / 600.0,
            "volatility_diff": hs.volatility - aws.volatility,
            "form_diff": hs.form - aws.form,
            "momentum_diff": (hs.momentum_short - hs.momentum_long)
            - (aws.momentum_short - aws.momentum_long),
            "goals_for_diff": hs.goals_for - aws.goals_for,
            "goals_against_diff": hs.goals_against - aws.goals_against,
            "goal_balance_diff": (hs.goals_for - hs.goals_against)
            - (aws.goals_for - aws.goals_against),
            "attack_trend_diff": (hs.goals_for - hs.attack_long)
            - (aws.goals_for - aws.attack_long),
            "defence_trend_diff": (hs.goals_against - hs.defence_long)
            - (aws.goals_against - aws.defence_long),
            "clean_sheet_diff": hs.clean_sheet_rate - aws.clean_sheet_rate,
            "scoring_rate_diff": hs.scoring_rate - aws.scoring_rate,
            "consistency_diff": aws.result_variance - hs.result_variance,
            "win_rate_diff": hs.win_rate - aws.win_rate,
            "sos_diff": (hs.sos - aws.sos) / 400.0,
            "prestige_diff": (hs.prestige - aws.prestige) / 400.0,
            "major_form_diff": hs.major_form - aws.major_form,
            "experience_diff": log1p(hs.matches) - log1p(aws.matches),
            "activity_diff": (home_activity - away_activity) / 12.0,
            "rest_diff": (home_rest - away_rest) / 30.0,
            "h2h_form": h2h_form - 0.5,
            "h2h_goal_balance": h2h_balance,
            "h2h_experience": log1p(h2h.matches) / 3.0,
            "neutral": float(neutral),
            "home_advantage": float(not neutral),
            "host_home": float(home == country),
            "host_away": float(away == country),
            "importance": importance,
            "is_world_cup": float(tournament == "FIFA World Cup"),
            "year": (date.year - 2000) / 25.0,
        }

    def update(
        self,
        home: str,
        away: str,
        date: pd.Timestamp,
        home_goals: int,
        away_goals: int,
        tournament: str,
        neutral: bool,
    ) -> None:
        hs, aws = self._state(home), self._state(away)
        old_home_elo, old_away_elo = hs.elo, aws.elo
        for state in (hs, aws):
            if state.last_date is not None:
                inactivity = max((date - state.last_date).days, 0)
                state.rating_sigma = min(
                    350.0,
                    float(np.sqrt(state.rating_sigma**2 + 0.55 * inactivity)),
                )
        if home_goals > away_goals:
            home_result = 1.0
            home_points, away_points = 1.0, 0.0
        elif home_goals < away_goals:
            home_result = 0.0
            home_points, away_points = 0.0, 1.0
        else:
            home_result = 0.5
            home_points = away_points = 0.5

        home_bonus = 0.0 if neutral else 80.0
        expected_home = 1.0 / (1.0 + 10 ** ((old_away_elo - old_home_elo - home_bonus) / 400))
        margin = abs(home_goals - away_goals)
        margin_multiplier = 1.0 if margin <= 1 else 1.5 if margin == 2 else 1.75
        k = 18.0 + 32.0 * tournament_importance(tournament)
        uncertainty_scale = np.clip((hs.rating_sigma + aws.rating_sigma) / 360.0, 0.65, 1.65)
        surprise = home_result - expected_home
        delta = k * margin_multiplier * uncertainty_scale * surprise
        hs.elo += delta
        aws.elo -= delta
        information = expected_home * (1.0 - expected_home)
        hs.rating_sigma = max(42.0, hs.rating_sigma * np.sqrt(max(0.90, 1 - 0.08 * information)))
        aws.rating_sigma = max(42.0, aws.rating_sigma * np.sqrt(max(0.90, 1 - 0.08 * information)))

        a = self.alpha
        for state, gf, ga, points, opponent_elo in (
            (hs, home_goals, away_goals, home_points, old_away_elo),
            (aws, away_goals, home_goals, away_points, old_home_elo),
        ):
            state.form = (1 - a) * state.form + a * points
            state.momentum_short = 0.72 * state.momentum_short + 0.28 * points
            state.momentum_long = 0.94 * state.momentum_long + 0.06 * points
            state.goals_for = (1 - a) * state.goals_for + a * gf
            state.goals_against = (1 - a) * state.goals_against + a * ga
            state.attack_long = 0.95 * state.attack_long + 0.05 * gf
            state.defence_long = 0.95 * state.defence_long + 0.05 * ga
            state.clean_sheet_rate = (1 - a) * state.clean_sheet_rate + a * float(ga == 0)
            state.scoring_rate = (1 - a) * state.scoring_rate + a * float(gf > 0)
            squared_surprise = (points - state.form) ** 2
            state.result_variance = (1 - a) * state.result_variance + a * squared_surprise
            state.win_rate = (1 - a) * state.win_rate + a * float(points == 1.0)
            state.sos = (1 - a) * state.sos + a * opponent_elo
            performance_rating = opponent_elo + 400.0 * (points - 0.5)
            state.prestige = 0.90 * state.prestige + 0.10 * performance_rating
            state.volatility = 0.90 * state.volatility + 0.10 * abs(surprise)
            if any(token in tournament for token in MAJOR_TOURNAMENT_TOKENS):
                state.major_form = (1 - a) * state.major_form + a * points
            state.matches += 1
            state.last_date = date
            state.recent_dates.append(date)
            state.recent_dates = [
                seen for seen in state.recent_dates if (date - seen).days <= 730
            ]

        key, home_is_first = self._canonical_h2h(home, away)
        pair = self.h2h.setdefault(key, HeadToHeadState())
        first_points = home_points if home_is_first else away_points
        first_balance = (home_goals - away_goals) * (1 if home_is_first else -1)
        pair.home_points = (1 - a) * pair.home_points + a * first_points
        pair.home_goal_balance = (1 - a) * pair.home_goal_balance + a * first_balance
        pair.matches += 1

    def transform(self, matches: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for match in matches.sort_values(["date", "home_team", "away_team"]).itertuples():
            features = self.make_features(
                match.home_team,
                match.away_team,
                match.date,
                match.tournament,
                match.country,
                bool(match.neutral),
            )
            row = {
                **features,
                "date": match.date,
                "home_team": match.home_team,
                "away_team": match.away_team,
                "tournament": match.tournament,
                "home_score": match.home_score,
                "away_score": match.away_score,
                "is_completed": bool(match.is_completed),
            }
            if match.is_completed:
                home_score, away_score = int(match.home_score), int(match.away_score)
                row["target"] = 2 if home_score > away_score else 0 if home_score < away_score else 1
                self.update(
                    match.home_team,
                    match.away_team,
                    match.date,
                    home_score,
                    away_score,
                    match.tournament,
                    bool(match.neutral),
                )
            else:
                row["target"] = np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    def ratings_frame(self) -> pd.DataFrame:
        return (
            pd.DataFrame(
                [
                    {
                        "team": team,
                        "elo": state.elo,
                        "rating_sigma": state.rating_sigma,
                        "power_low": state.elo - 1.64 * state.rating_sigma,
                        "power_high": state.elo + 1.64 * state.rating_sigma,
                        "volatility": state.volatility,
                        "form": state.form,
                        "momentum": state.momentum_short - state.momentum_long,
                        "attack": state.goals_for,
                        "defence": state.goals_against,
                        "attack_trend": state.goals_for - state.attack_long,
                        "defence_trend": state.goals_against - state.defence_long,
                        "clean_sheet_rate": state.clean_sheet_rate,
                        "scoring_rate": state.scoring_rate,
                        "consistency": 1.0 - state.result_variance,
                        "schedule_strength": state.sos,
                        "prestige": state.prestige,
                        "matches": state.matches,
                    }
                    for team, state in self.states.items()
                ]
            )
            .sort_values("elo", ascending=False)
            .reset_index(drop=True)
        )
