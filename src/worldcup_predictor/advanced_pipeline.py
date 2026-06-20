from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from .conditions import (
    TRAVEL_FEATURE_COLUMNS,
    WEATHER_FEATURE_COLUMNS,
    add_condition_features,
    build_world_cup_conditions,
)
from .config import (
    ABLATION_PATH,
    ADVANCED_MATRIX_PATH,
    ADVANCED_REPORT_PATH,
    ARCHITECTURE_BENCHMARK_PATH,
    ARTIFACT_DIR,
    BENCHMARK_PATH,
    BLIND_LEADERBOARD_PATH,
    DATA_COVERAGE_PATH,
    ENRICHMENT_BENCHMARK_PATH,
    EVENT_BENCHMARK_PATH,
    FEATURES_PATH,
    FIXTURES_PATH,
    LINEUP_BENCHMARK_PATH,
    REPORT_PATH,
    SIMULATION_PATH,
    SQUAD_2026_ANALYTICS_PATH,
    SQUAD_2026_PLAYERS_PATH,
    TURKEY_BLIND_PATH,
    TURKEY_PATH,
    TURKEY_SCENARIOS_PATH,
    TOURNAMENT_BLIND_PATH,
    WEATHER_BENCHMARK_PATH,
)
from .data import (
    load_2026_squads,
    load_world_cup_reference_data,
)
from .evaluation import architecture_benchmark, enrichment_benchmark
from .event_data import (
    STATSBOMB_FEATURE_COLUMNS,
    add_statsbomb_features,
    build_statsbomb_summary,
)
from .features import FEATURE_COLUMNS
from .lineup_data import (
    LINEUP_FEATURE_COLUMNS,
    add_announced_lineup_features,
    build_statsbomb_lineup_summary,
)
from .model import MatchEnsemble
from .reporting import generate_unified_visual_report
from .squad_analytics import OFFICIAL_SQUAD_PDF, squad_analytics_2026

OFFICIAL_PDF_PATH = Path("data/raw/fifa_2026_squads.pdf")


def run_advanced_pipeline(
    refresh: bool = False,
    rerun_gpu: bool = False,
) -> dict[str, object]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(FEATURES_PATH)
    features["date"] = pd.to_datetime(features["date"])

    reference_matches, _ = load_world_cup_reference_data(refresh=refresh)
    conditions = build_world_cup_conditions(reference_matches, refresh=refresh)
    conditioned = add_condition_features(features, conditions)
    weather = _weather_benchmark(conditioned)

    event_summary = build_statsbomb_summary(refresh=refresh)
    event_features = add_statsbomb_features(conditioned, event_summary)
    lineup_summary = build_statsbomb_lineup_summary(
        event_summary, refresh=refresh
    )
    advanced_features = add_announced_lineup_features(
        event_features, lineup_summary
    )
    lineup_results, lineup_predictions = _lineup_combination_benchmark(
        advanced_features
    )
    lineup_results.to_csv(LINEUP_BENCHMARK_PATH, index=False)

    event_results = lineup_results.loc[
        lineup_results["configuration"].isin(
            ["Baseline scores + state", "+ StatsBomb event dynamics"]
        )
    ].copy()
    event_results.to_csv(EVENT_BENCHMARK_PATH, index=False)

    if rerun_gpu or not ARCHITECTURE_BENCHMARK_PATH.exists():
        architecture_benchmark(features, FEATURE_COLUMNS).to_csv(
            ARCHITECTURE_BENCHMARK_PATH, index=False
        )
    architectures = pd.read_csv(ARCHITECTURE_BENCHMARK_PATH)

    squads = load_2026_squads(refresh=refresh)
    OFFICIAL_PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not OFFICIAL_PDF_PATH.exists():
        urlretrieve(OFFICIAL_SQUAD_PDF, OFFICIAL_PDF_PATH)
    squad_players, squad_teams = squad_analytics_2026(
        squads, OFFICIAL_PDF_PATH
    )
    squad_players.to_csv(SQUAD_2026_PLAYERS_PATH, index=False)
    squad_teams.to_csv(SQUAD_2026_ANALYTICS_PATH, index=False)

    matrix = _experiment_matrix(
        weather, architectures, lineup_results, lineup_predictions
    )
    matrix.to_csv(ADVANCED_MATRIX_PATH, index=False)
    report = _build_report(
        matrix,
        architectures,
        event_summary,
        lineup_summary,
        squad_players,
        squad_teams,
        conditions,
    )
    ADVANCED_REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    visual_paths = _regenerate_unified_visuals(
        matrix, architectures, squad_teams, squad_players
    )
    report["visual_assets"] = [
        str(path.relative_to(Path.cwd()).as_posix()) for path in visual_paths
    ]
    ADVANCED_REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return report


def _weather_benchmark(conditioned: pd.DataFrame) -> pd.DataFrame:
    if WEATHER_BENCHMARK_PATH.exists():
        return pd.read_csv(WEATHER_BENCHMARK_PATH)
    configurations = {
        "Baseline scores + state": FEATURE_COLUMNS,
        "+ travel context": FEATURE_COLUMNS + TRAVEL_FEATURE_COLUMNS,
        "+ oracle weather": FEATURE_COLUMNS + WEATHER_FEATURE_COLUMNS,
        "+ oracle weather + travel": (
            FEATURE_COLUMNS + WEATHER_FEATURE_COLUMNS + TRAVEL_FEATURE_COLUMNS
        ),
    }
    result = enrichment_benchmark(conditioned, configurations)
    result.to_csv(WEATHER_BENCHMARK_PATH, index=False)
    return result


def _lineup_combination_benchmark(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    configurations = {
        "Baseline scores + state": FEATURE_COLUMNS,
        "+ StatsBomb event dynamics": FEATURE_COLUMNS + STATSBOMB_FEATURE_COLUMNS,
        "+ announced lineup": FEATURE_COLUMNS + LINEUP_FEATURE_COLUMNS,
        "+ event + announced lineup": (
            FEATURE_COLUMNS + STATSBOMB_FEATURE_COLUMNS + LINEUP_FEATURE_COLUMNS
        ),
        "+ event + lineup + travel": (
            FEATURE_COLUMNS
            + STATSBOMB_FEATURE_COLUMNS
            + LINEUP_FEATURE_COLUMNS
            + TRAVEL_FEATURE_COLUMNS
        ),
        "+ event + lineup + oracle weather + travel": (
            FEATURE_COLUMNS
            + STATSBOMB_FEATURE_COLUMNS
            + LINEUP_FEATURE_COLUMNS
            + WEATHER_FEATURE_COLUMNS
            + TRAVEL_FEATURE_COLUMNS
        ),
    }
    completed = features.dropna(subset=["target"]).copy()
    test = completed.loc[
        completed["tournament"].eq("FIFA World Cup")
        & completed["date"].dt.year.eq(2022)
    ]
    cutoff = test["date"].min()
    train = completed.loc[
        (completed["date"] < cutoff)
        & (completed["date"] >= cutoff - pd.DateOffset(years=28))
    ]
    rows = []
    predictions = []
    y = test["target"].astype(int).to_numpy()
    for name, columns in configurations.items():
        model = MatchEnsemble(
            random_state=2022, feature_columns=columns
        ).fit(train, tune=True)
        probability = model.predict_proba(test)
        metrics = _probability_metrics(y, probability)
        rows.append(
            {
                "edition": 2022,
                "configuration": name,
                "features": len(columns),
                **metrics,
            }
        )
        ledger = test[
            ["date", "home_team", "away_team", "target"]
        ].copy()
        ledger["configuration"] = name
        ledger[["p_away", "p_draw", "p_home"]] = probability
        predictions.append(ledger)
    result = pd.DataFrame(rows)
    baseline = result.loc[
        result["configuration"].eq("Baseline scores + state")
    ].iloc[0]
    for metric in ("log_loss", "accuracy", "brier"):
        result[f"{metric}_delta"] = result[metric] - baseline[metric]
    return result.sort_values("log_loss").reset_index(drop=True), pd.concat(
        predictions, ignore_index=True
    )


def _probability_metrics(
    target: np.ndarray, probability: np.ndarray
) -> dict[str, float]:
    one_hot = np.eye(3)[target]
    return {
        "log_loss": float(log_loss(target, probability, labels=[0, 1, 2])),
        "accuracy": float(accuracy_score(target, probability.argmax(axis=1))),
        "brier": float(np.mean(np.sum((probability - one_hot) ** 2, axis=1))),
    }


def _paired_bootstrap(
    predictions: pd.DataFrame,
    configuration: str,
    draws: int = 4000,
) -> tuple[float, float]:
    baseline = predictions.loc[
        predictions["configuration"].eq("Baseline scores + state")
    ].sort_values(["date", "home_team", "away_team"])
    candidate = predictions.loc[
        predictions["configuration"].eq(configuration)
    ].sort_values(["date", "home_team", "away_team"])
    target = baseline["target"].astype(int).to_numpy()
    baseline_probability = baseline[
        ["p_away", "p_draw", "p_home"]
    ].to_numpy()
    candidate_probability = candidate[
        ["p_away", "p_draw", "p_home"]
    ].to_numpy()
    rng = np.random.default_rng(2026)
    deltas = []
    for _ in range(draws):
        index = rng.integers(0, len(target), len(target))
        deltas.append(
            log_loss(
                target[index],
                candidate_probability[index],
                labels=[0, 1, 2],
            )
            - log_loss(
                target[index],
                baseline_probability[index],
                labels=[0, 1, 2],
            )
        )
    low, high = np.quantile(deltas, [0.025, 0.975])
    return float(low), float(high)


def _experiment_matrix(
    weather: pd.DataFrame,
    architectures: pd.DataFrame,
    lineup: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for name, frame in weather.groupby("configuration"):
        rows.append(
            {
                "stage": "Environment",
                "candidate": name,
                "evaluation_scope": "2006–2022 · 5 World Cups",
                "folds": frame["edition"].nunique(),
                "log_loss": frame["log_loss"].mean(),
                "accuracy": frame["accuracy"].mean(),
                "log_loss_delta": frame["log_loss_delta"].mean(),
                "accuracy_delta": frame["accuracy_delta"].mean(),
                "interval_low": np.nan,
                "interval_high": np.nan,
                "availability": (
                    "Retrospective reanalysis"
                    if "weather" in name
                    else "Pre-match derivable"
                ),
            }
        )
    for name, frame in architectures.groupby("architecture"):
        rows.append(
            {
                "stage": "Architecture",
                "candidate": name,
                "evaluation_scope": "2006–2022 · 5 World Cups",
                "folds": frame["edition"].nunique(),
                "log_loss": frame["log_loss"].mean(),
                "accuracy": frame["accuracy"].mean(),
                "log_loss_delta": frame["log_loss_delta"].mean(),
                "accuracy_delta": frame["accuracy_delta"].mean(),
                "interval_low": np.nan,
                "interval_high": np.nan,
                "availability": str(frame["device"].mode().iloc[0]).upper(),
            }
        )
    for row in lineup.itertuples(index=False):
        low, high = _paired_bootstrap(predictions, row.configuration)
        rows.append(
            {
                "stage": "Event + lineup",
                "candidate": row.configuration,
                "evaluation_scope": "2022 World Cup · coverage-limited",
                "folds": 1,
                "log_loss": row.log_loss,
                "accuracy": row.accuracy,
                "log_loss_delta": row.log_loss_delta,
                "accuracy_delta": row.accuracy_delta,
                "interval_low": low,
                "interval_high": high,
                "availability": (
                    "After official XI release"
                    if "lineup" in row.configuration
                    else "Pre-match event history"
                ),
            }
        )
    matrix = pd.DataFrame(rows).drop_duplicates(
        ["stage", "candidate"], keep="last"
    )
    matrix["decision"] = matrix.apply(_decision, axis=1)
    matrix["log_loss_basis_points"] = matrix["log_loss_delta"] * 10_000
    matrix["accuracy_points"] = matrix["accuracy_delta"] * 100
    return matrix.sort_values(
        ["stage", "log_loss", "candidate"]
    ).reset_index(drop=True)


def _decision(row: pd.Series) -> str:
    if row["candidate"] in {
        "Baseline scores + state",
        "Calibrated hybrid",
    }:
        return "Production baseline"
    if row["folds"] < 5:
        return "Coverage-limited"
    if row["availability"] == "Retrospective reanalysis":
        return "Oracle only"
    if row["log_loss_delta"] < -0.001 and row["accuracy_delta"] >= 0:
        return "Promotion candidate"
    return "Rejected"


def _build_report(
    matrix: pd.DataFrame,
    architectures: pd.DataFrame,
    event_summary: pd.DataFrame,
    lineup_summary: pd.DataFrame,
    squad_players: pd.DataFrame,
    squad_teams: pd.DataFrame,
    conditions: pd.DataFrame,
) -> dict[str, object]:
    turkey = squad_teams.loc[squad_teams["fifa_code"].eq("TUR")].iloc[0]
    return {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "runtime": {
            "python": platform.python_version(),
            **_gpu_runtime(),
        },
        "coverage": {
            "weather_world_cup_matches": int(
                conditions["weather_temperature"].notna().sum()
            ),
            "weather_total_matches": int(len(conditions)),
            "statsbomb_team_matches": int(len(event_summary)),
            "statsbomb_matches": int(event_summary["match_id"].nunique()),
            "statsbomb_teams": int(event_summary["team"].nunique()),
            "announced_lineup_team_matches": int(len(lineup_summary)),
            "official_2026_squad_players": int(len(squad_players)),
            "official_squad_join_rate": float(
                squad_players["caps"].notna().mean()
            ),
        },
        "architecture": _records(
            architectures.groupby("architecture")
            .agg(
                log_loss=("log_loss", "mean"),
                accuracy=("accuracy", "mean"),
                training_seconds=("training_seconds", "mean"),
            )
            .reset_index()
            .sort_values("log_loss")
        ),
        "decisions": _records(matrix),
        "turkey_squad": {
            key: _json_value(turkey[key])
            for key in [
                "mean_age",
                "total_caps",
                "total_international_goals",
                "top_five_league_share",
                "domestic_club_share",
                "club_diversity",
                "official_stats_coverage",
            ]
        },
        "methodology": {
            "weather": (
                "ERA5 reanalysis is an oracle ceiling. It is never eligible for "
                "live promotion because exact observed weather was unavailable "
                "before historical kickoff."
            ),
            "lineups": (
                "Current starting-XI features are used only in the announced-lineup "
                "model, after official lineups become available."
            ),
            "promotion_gate": (
                "Five chronological World Cup folds, lower log loss, no accuracy "
                "regression, acceptable calibration, and pre-kickoff availability."
            ),
        },
    }


def _gpu_runtime() -> dict[str, object]:
    try:
        import torch

        return {
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        }
    except ImportError:
        return {
            "torch": None,
            "cuda_available": False,
            "cuda_device": None,
        }


def _regenerate_unified_visuals(
    matrix: pd.DataFrame,
    architectures: pd.DataFrame,
    squad_teams: pd.DataFrame,
    squad_players: pd.DataFrame,
) -> list[Path]:
    base_report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    return generate_unified_visual_report(
        pd.DataFrame(base_report["backtests"]),
        pd.read_csv(BENCHMARK_PATH),
        pd.DataFrame(base_report["reliability"]),
        pd.read_csv(ABLATION_PATH),
        pd.read_csv(SIMULATION_PATH),
        pd.read_csv(FIXTURES_PATH, parse_dates=["date"]),
        pd.DataFrame(base_report["ratings"]),
        pd.read_csv(TURKEY_PATH, parse_dates=["date"]),
        pd.read_csv(TURKEY_SCENARIOS_PATH),
        pd.read_csv(TURKEY_BLIND_PATH, parse_dates=["date"]),
        pd.read_csv(TOURNAMENT_BLIND_PATH, parse_dates=["date"]),
        pd.read_csv(BLIND_LEADERBOARD_PATH),
        pd.read_csv(ENRICHMENT_BENCHMARK_PATH),
        pd.read_csv(DATA_COVERAGE_PATH),
        matrix,
        architectures,
        squad_teams,
        squad_players,
    )


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run staged environment, event, lineup, squad, and GPU research."
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--rerun-gpu", action="store_true")
    args = parser.parse_args()
    report = run_advanced_pipeline(
        refresh=args.refresh, rerun_gpu=args.rerun_gpu
    )
    print(json.dumps(report["coverage"], indent=2))


if __name__ == "__main__":
    main()
