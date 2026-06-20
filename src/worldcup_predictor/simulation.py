from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, SequentialFeatureBuilder
from .model import MatchEnsemble

THIRD_PLACE_SLOTS = {
    "74": set("ABCDF"),
    "77": set("CDFGH"),
    "79": set("CE FHI".replace(" ", "")),
    "80": set("EHIJK"),
    "81": set("BEFIJ"),
    "82": set("AEHIJ"),
    "85": set("EFGIJ"),
    "87": set("DEIJL"),
}

GROUP_ANCHORS = {
    "A": "Mexico",
    "B": "Canada",
    "C": "Brazil",
    "D": "United States",
    "E": "Germany",
    "F": "Netherlands",
    "G": "Belgium",
    "H": "Spain",
    "I": "France",
    "J": "Argentina",
    "K": "Portugal",
    "L": "England",
}


def infer_groups(fixtures: pd.DataFrame) -> dict[str, list[str]]:
    """Infer 12 groups as connected components of the round-robin fixture graph."""
    adjacency: dict[str, set[str]] = defaultdict(set)
    first_seen: dict[str, int] = {}
    for index, match in enumerate(fixtures.sort_values("date").itertuples()):
        adjacency[match.home_team].add(match.away_team)
        adjacency[match.away_team].add(match.home_team)
        first_seen.setdefault(match.home_team, index)
        first_seen.setdefault(match.away_team, index)
    components: list[list[str]] = []
    unseen = set(adjacency)
    while unseen:
        root = min(unseen, key=lambda team: first_seen[team])
        stack, component = [root], []
        while stack:
            team = stack.pop()
            if team not in unseen:
                continue
            unseen.remove(team)
            component.append(team)
            stack.extend(adjacency[team] & unseen)
        components.append(sorted(component, key=lambda team: first_seen[team]))
    components.sort(key=lambda teams: min(first_seen[t] for t in teams))
    anchored = {
        letter: component
        for letter, anchor in GROUP_ANCHORS.items()
        for component in components
        if anchor in component
    }
    if len(anchored) == 12:
        return anchored
    return {chr(65 + i): teams for i, teams in enumerate(components)}


def forecast_fixtures(model: MatchEnsemble, features: pd.DataFrame) -> pd.DataFrame:
    fixture_features = features.loc[~features["is_completed"]].copy()
    probabilities = model.predict_proba(fixture_features[FEATURE_COLUMNS])
    components = model.component_probabilities(fixture_features[FEATURE_COLUMNS])
    home_xg, away_xg = model.predict_goals(fixture_features[FEATURE_COLUMNS])
    output = fixture_features[["date", "home_team", "away_team", "tournament"]].copy()
    output[["p_away", "p_draw", "p_home"]] = probabilities
    output["home_xg"] = home_xg
    output["away_xg"] = away_xg
    output["total_xg"] = home_xg + away_xg
    output["entropy"] = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-12, 1)), axis=1
    )
    output["model_disagreement"] = np.std(
        np.stack(list(components.values()), axis=0), axis=0
    ).mean(axis=1)
    output["upset_index"] = 1.0 - probabilities.max(axis=1)
    output["confidence"] = probabilities.max(axis=1)
    for name, values in components.items():
        output[f"{name}_home"] = values[:, 2]
        output[f"{name}_draw"] = values[:, 1]
        output[f"{name}_away"] = values[:, 0]
    output["most_likely"] = np.array(["Away win", "Draw", "Home win"])[
        probabilities.argmax(axis=1)
    ]
    return output.reset_index(drop=True)


def _standings(
    teams: list[str], matches: list[tuple[str, str, int, int]], rng: np.random.Generator
) -> tuple[list[str], dict[str, list[int]]]:
    table = {team: [0, 0, 0] for team in teams}  # points, goal difference, goals for
    for home, away, hg, ag in matches:
        table[home][1] += hg - ag
        table[away][1] += ag - hg
        table[home][2] += hg
        table[away][2] += ag
        if hg > ag:
            table[home][0] += 3
        elif hg < ag:
            table[away][0] += 3
        else:
            table[home][0] += 1
            table[away][0] += 1
    ranking = sorted(
        teams,
        key=lambda team: (*table[team], rng.random()),
        reverse=True,
    )
    return ranking, table


def _assign_thirds(
    third_teams: list[tuple[str, str]], rng: np.random.Generator
) -> dict[str, str]:
    slots = list(THIRD_PLACE_SLOTS)
    rng.shuffle(slots)
    candidates = third_teams.copy()
    rng.shuffle(candidates)

    def search(index: int, assigned: dict[str, str], remaining: list[tuple[str, str]]):
        if index == len(slots):
            return assigned
        slot = slots[index]
        for candidate_index, (group, team) in enumerate(remaining):
            if group in THIRD_PLACE_SLOTS[slot]:
                result = search(
                    index + 1,
                    {**assigned, slot: team},
                    remaining[:candidate_index] + remaining[candidate_index + 1 :],
                )
                if result is not None:
                    return result
        return None

    assignment = search(0, {}, candidates)
    if assignment is None:
        # Eligibility placeholders cannot represent every combination without FIFA's lookup table.
        return {slot: team for slot, (_, team) in zip(sorted(slots), candidates)}
    return assignment


def _knockout_winner(
    team_a: str,
    team_b: str,
    win_lookup: dict[tuple[str, str], float],
    rng: np.random.Generator,
) -> str:
    return team_a if rng.random() < win_lookup[(team_a, team_b)] else team_b


def simulate_tournament(
    model: MatchEnsemble,
    builder: SequentialFeatureBuilder,
    tournament: pd.DataFrame,
    fixture_predictions: pd.DataFrame,
    simulations: int = 5000,
    seed: int = 2026,
) -> pd.DataFrame:
    groups = infer_groups(tournament)
    team_group = {team: group for group, teams in groups.items() for team in teams}
    completed = tournament.loc[tournament["is_completed"]]
    pending = tournament.loc[~tournament["is_completed"]]
    prediction_lookup = {
        (row.home_team, row.away_team): (row.home_xg, row.away_xg)
        for row in fixture_predictions.itertuples()
    }
    counts = {
        team: {"round_32": 0, "quarterfinal": 0, "semifinal": 0, "final": 0, "champion": 0}
        for team in team_group
    }
    rng = np.random.default_rng(seed)
    teams = list(team_group)
    matchup_rows = []
    matchup_keys = []
    for team_a in teams:
        for team_b in teams:
            if team_a == team_b:
                continue
            matchup_keys.append((team_a, team_b))
            matchup_rows.append(
                builder.make_features(
                    team_a,
                    team_b,
                    pd.Timestamp("2026-07-01"),
                    "FIFA World Cup",
                    "United States",
                    True,
                )
            )
    matchup_probabilities = model.predict_proba(pd.DataFrame(matchup_rows)[FEATURE_COLUMNS])
    win_lookup = {}
    for key, probabilities in zip(matchup_keys, matchup_probabilities):
        decisive_probability = probabilities[2] / (probabilities[2] + probabilities[0])
        win_lookup[key] = float(decisive_probability)

    fixed_matches: dict[str, list[tuple[str, str, int, int]]] = defaultdict(list)
    for row in completed.itertuples():
        fixed_matches[team_group[row.home_team]].append(
            (row.home_team, row.away_team, int(row.home_score), int(row.away_score))
        )

    for _ in range(simulations):
        group_matches = {group: list(matches) for group, matches in fixed_matches.items()}
        for row in pending.itertuples():
            home_lambda, away_lambda = prediction_lookup[(row.home_team, row.away_team)]
            hg, ag = int(rng.poisson(home_lambda)), int(rng.poisson(away_lambda))
            group_matches.setdefault(team_group[row.home_team], []).append(
                (row.home_team, row.away_team, hg, ag)
            )
        ranked_tables = {
            group: _standings(teams, group_matches.get(group, []), rng)
            for group, teams in groups.items()
        }
        rankings = {group: value[0] for group, value in ranked_tables.items()}
        thirds = [
            (group, ranking[2], ranked_tables[group][1][ranking[2]])
            for group, ranking in rankings.items()
        ]
        thirds.sort(
            key=lambda item: (*item[2], builder.states[item[1]].elo, rng.random()),
            reverse=True,
        )
        best_thirds = [(group, team) for group, team, _ in thirds[:8]]
        third_slots = _assign_thirds(best_thirds, rng)

        pairings = [
            (rankings["A"][1], rankings["B"][1]),
            (rankings["E"][0], third_slots["74"]),
            (rankings["F"][0], rankings["C"][1]),
            (rankings["C"][0], rankings["F"][1]),
            (rankings["I"][0], third_slots["77"]),
            (rankings["E"][1], rankings["I"][1]),
            (rankings["A"][0], third_slots["79"]),
            (rankings["L"][0], third_slots["80"]),
            (rankings["D"][0], third_slots["81"]),
            (rankings["G"][0], third_slots["82"]),
            (rankings["K"][1], rankings["L"][1]),
            (rankings["H"][0], rankings["J"][1]),
            (rankings["B"][0], third_slots["85"]),
            (rankings["J"][0], rankings["H"][1]),
            (rankings["K"][0], third_slots["87"]),
            (rankings["D"][1], rankings["G"][1]),
        ]
        qualifiers = {team for pairing in pairings for team in pairing}
        for team in qualifiers:
            counts[team]["round_32"] += 1

        round_16 = [
            _knockout_winner(a, b, win_lookup, rng)
            for a, b in pairings
        ]
        quarterfinalists = [
            _knockout_winner(round_16[i], round_16[i + 1], win_lookup, rng)
            for i in range(0, 16, 2)
        ]
        for team in quarterfinalists:
            counts[team]["quarterfinal"] += 1
        semifinalists = [
            _knockout_winner(
                quarterfinalists[i],
                quarterfinalists[i + 1],
                win_lookup,
                rng,
            )
            for i in range(0, 8, 2)
        ]
        for team in semifinalists:
            counts[team]["semifinal"] += 1
        finalists = [
            _knockout_winner(
                semifinalists[i],
                semifinalists[i + 1],
                win_lookup,
                rng,
            )
            for i in range(0, 4, 2)
        ]
        for team in finalists:
            counts[team]["final"] += 1
        champion = _knockout_winner(
            finalists[0],
            finalists[1],
            win_lookup,
            rng,
        )
        counts[champion]["champion"] += 1

    rows = []
    for team, stages in counts.items():
        rows.append(
            {
                "team": team,
                "group": team_group[team],
                **{stage: value / simulations for stage, value in stages.items()},
                "elo": builder.states[team].elo,
            }
        )
    return pd.DataFrame(rows).sort_values("champion", ascending=False).reset_index(drop=True)


def conditional_fixture_scenarios(
    model: MatchEnsemble,
    builder: SequentialFeatureBuilder,
    tournament: pd.DataFrame,
    fixture_predictions: pd.DataFrame,
    home_team: str,
    away_team: str,
    scenarios: dict[str, str],
    focus_team: str,
    simulations: int = 3000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Estimate tournament paths after forcing a future fixture outcome.

    The score process is integrated out conditional on each requested outcome.
    Public outputs expose only win, draw, and loss conditions.
    """
    fixture_mask = tournament["home_team"].eq(home_team) & tournament["away_team"].eq(
        away_team
    )
    if fixture_mask.sum() != 1:
        raise ValueError(f"Expected exactly one fixture: {home_team} vs {away_team}.")
    fixture = tournament.loc[fixture_mask].iloc[0]
    feature_row = pd.DataFrame(
        [
            builder.make_features(
                home_team,
                away_team,
                fixture["date"],
                fixture["tournament"],
                fixture["country"],
                bool(fixture["neutral"]),
            )
        ]
    )
    score_probabilities = model.score_matrix(
        feature_row[FEATURE_COLUMNS], max_goals=7
    )[0]
    remaining_predictions = fixture_predictions.loc[
        ~(
            fixture_predictions["home_team"].eq(home_team)
            & fixture_predictions["away_team"].eq(away_team)
        )
    ].copy()
    rows = []
    for index, (scenario, outcome) in enumerate(scenarios.items()):
        home_grid, away_grid = np.indices(score_probabilities.shape)
        mask = {
            "home": home_grid > away_grid,
            "draw": home_grid == away_grid,
            "away": home_grid < away_grid,
        }[outcome]
        candidates = np.argwhere(mask)
        weights = score_probabilities[mask]
        weights = weights / weights.sum()
        rng = np.random.default_rng(seed + index)
        allocations = rng.multinomial(simulations, weights)
        stage_totals = {
            "round_32": 0.0,
            "quarterfinal": 0.0,
            "semifinal": 0.0,
            "final": 0.0,
            "champion": 0.0,
        }
        for score_index, ((home_score, away_score), count) in enumerate(
            zip(candidates, allocations)
        ):
            if count == 0:
                continue
            conditioned = tournament.copy()
            conditioned.loc[
                fixture_mask, ["home_score", "away_score", "is_completed"]
            ] = [int(home_score), int(away_score), True]
            simulation = simulate_tournament(
                model,
                builder,
                conditioned,
                remaining_predictions,
                simulations=int(count),
                seed=seed + index * 100 + score_index,
            )
            result = simulation.loc[simulation["team"].eq(focus_team)].iloc[0]
            for stage in stage_totals:
                stage_totals[stage] += float(result[stage]) * count
        rows.append(
            {
                "scenario": scenario,
                **{
                    stage: value / simulations
                    for stage, value in stage_totals.items()
                },
            }
        )
    return pd.DataFrame(rows)
