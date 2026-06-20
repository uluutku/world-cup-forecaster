from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from .analytics import team_state_embedding
from .config import (
    ABLATION_PATH,
    ARTIFACT_DIR,
    BACKTEST_PATH,
    BLIND_LEADERBOARD_PATH,
    BENCHMARK_PATH,
    DATA_COVERAGE_PATH,
    DRIFT_PATH,
    ENRICHMENT_BENCHMARK_PATH,
    FEATURES_PATH,
    FIXTURES_PATH,
    MODEL_PATH,
    RAW_DIR,
    REPORT_PATH,
    SIMULATION_PATH,
    TURKEY_PATH,
    TURKEY_BLIND_PATH,
    TURKEY_SCENARIOS_PATH,
    TOURNAMENT_BLIND_PATH,
)
from .data import (
    load_auxiliary_data,
    load_results,
    load_world_cup_player_data,
    world_cup_2026,
)
from .enrichment import (
    RANKING_FEATURE_COLUMNS,
    SCORER_FEATURE_COLUMNS,
    SHOOTOUT_FEATURE_COLUMNS,
    SQUAD_FEATURE_COLUMNS,
    build_enriched_features,
    data_coverage_report,
)
from .evaluation import (
    bootstrap_intervals,
    confidence_reliability,
    conformal_diagnostics,
    confusion_table,
    drift_table,
    enrichment_benchmark,
    feature_ablation,
    reliability_table,
    world_cup_walk_forward,
)
from .features import FEATURE_COLUMNS, SequentialFeatureBuilder
from .experiments import blind_team_tournament_forecast, blind_tournament_forecast
from .model import MatchEnsemble
from .reporting import generate_visual_report
from .simulation import (
    conditional_fixture_scenarios,
    forecast_fixtures,
    simulate_tournament,
)


def run_pipeline(
    refresh: bool = False,
    simulations: int = 5000,
    run_backtest: bool = True,
) -> dict[str, object]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    matches = load_results(refresh=refresh)
    goalscorers, shootouts, rankings = load_auxiliary_data(refresh=refresh)
    players, squads, player_appearances = load_world_cup_player_data(
        refresh=refresh
    )
    builder = SequentialFeatureBuilder()
    features = builder.transform(matches)
    enriched_features = build_enriched_features(
        features,
        matches,
        goalscorers,
        shootouts,
        rankings,
        players,
        squads,
        player_appearances,
    )
    coverage = data_coverage_report(
        matches, goalscorers, rankings, shootouts, squads
    )
    completed = features.loc[features["is_completed"]].copy()
    recent = completed.loc[completed["date"] >= completed["date"].max() - pd.DateOffset(years=28)]

    model = MatchEnsemble().fit(recent, tune=True)

    fixture_rows = features.loc[
        ~features["is_completed"]
        & features["tournament"].eq("FIFA World Cup")
        & features["date"].dt.year.eq(2026)
    ]
    tournament = world_cup_2026(matches)
    fixture_predictions = forecast_fixtures(model, fixture_rows)
    simulation = simulate_tournament(
        model,
        builder,
        tournament,
        fixture_predictions,
        simulations=simulations,
    )
    turkey_schedule = tournament.loc[
        tournament["home_team"].eq("Turkey") | tournament["away_team"].eq("Turkey")
    ].copy()
    turkey_schedule = turkey_schedule.merge(
        fixture_predictions,
        on=["date", "home_team", "away_team", "tournament"],
        how="left",
        suffixes=("", "_forecast"),
    )
    turkey_schedule["status"] = turkey_schedule["is_completed"].map(
        {True: "Final", False: "Forecast"}
    )
    turkey_scenarios = conditional_fixture_scenarios(
        model,
        builder,
        tournament,
        fixture_predictions,
        home_team="United States",
        away_team="Turkey",
        scenarios={
            "Türkiye win": "away",
            "Draw": "draw",
            "Türkiye loss": "home",
        },
        focus_team="Turkey",
        simulations=max(3000, simulations // 2),
    )
    tournament_blind, blind_leaderboard, tournament_blind_metadata = (
        blind_tournament_forecast(
            matches,
            tournament="FIFA World Cup",
            year=2026,
        )
    )
    # Preserve the original Türkiye baseline protocol: its state freezes immediately
    # before Türkiye's first match, while the tournament-wide audit freezes before
    # the opening match for every team.
    turkey_blind, turkey_blind_metadata = blind_team_tournament_forecast(
        matches,
        team="Turkey",
        tournament="FIFA World Cup",
        year=2026,
    )

    backtests = pd.DataFrame()
    backtest_predictions = pd.DataFrame()
    benchmarks = pd.DataFrame()
    if run_backtest:
        backtests, backtest_predictions, benchmarks = world_cup_walk_forward(features)

    reliability = (
        reliability_table(backtest_predictions)
        if not backtest_predictions.empty
        else pd.DataFrame()
    )
    confidence_curve = (
        confidence_reliability(backtest_predictions)
        if not backtest_predictions.empty
        else pd.DataFrame()
    )
    bootstrap = (
        bootstrap_intervals(backtest_predictions)
        if not backtest_predictions.empty
        else {}
    )
    conformal_summary, conformal_matches = (
        conformal_diagnostics(backtest_predictions)
        if not backtest_predictions.empty
        else ({}, pd.DataFrame())
    )
    ablation = feature_ablation(features) if run_backtest else pd.DataFrame()
    drift = drift_table(features)
    ratings = builder.ratings_frame()
    embedding = team_state_embedding(ratings, simulation["team"].tolist())
    enrichment_configurations = {
        "Baseline scores + state": FEATURE_COLUMNS,
        "+ FIFA ranking": FEATURE_COLUMNS + RANKING_FEATURE_COLUMNS,
        "+ scorer/player dynamics": FEATURE_COLUMNS + SCORER_FEATURE_COLUMNS,
        "+ shootout dynamics": FEATURE_COLUMNS + SHOOTOUT_FEATURE_COLUMNS,
        "+ World Cup squad composition": FEATURE_COLUMNS + SQUAD_FEATURE_COLUMNS,
        "+ rankings + players": FEATURE_COLUMNS
        + RANKING_FEATURE_COLUMNS
        + SCORER_FEATURE_COLUMNS,
        "+ rankings + shootouts": FEATURE_COLUMNS
        + RANKING_FEATURE_COLUMNS
        + SHOOTOUT_FEATURE_COLUMNS,
        "+ players + shootouts": FEATURE_COLUMNS
        + SCORER_FEATURE_COLUMNS
        + SHOOTOUT_FEATURE_COLUMNS,
        "+ rankings + squad composition": FEATURE_COLUMNS
        + RANKING_FEATURE_COLUMNS
        + SQUAD_FEATURE_COLUMNS,
        "+ all open enrichment": FEATURE_COLUMNS
        + RANKING_FEATURE_COLUMNS
        + SCORER_FEATURE_COLUMNS
        + SHOOTOUT_FEATURE_COLUMNS,
        "+ all open + squad composition": FEATURE_COLUMNS
        + RANKING_FEATURE_COLUMNS
        + SCORER_FEATURE_COLUMNS
        + SHOOTOUT_FEATURE_COLUMNS
        + SQUAD_FEATURE_COLUMNS,
    }
    enrichment_results = (
        enrichment_benchmark(enriched_features, enrichment_configurations)
        if run_backtest
        else pd.DataFrame()
    )
    enrichment_summary = (
        enrichment_results.groupby("configuration", as_index=False)
        .agg(
            log_loss=("log_loss", "mean"),
            accuracy=("accuracy", "mean"),
            brier=("brier", "mean"),
            rps=("rps", "mean"),
            ece=("ece", "mean"),
            log_loss_delta=("log_loss_delta", "mean"),
            accuracy_delta=("accuracy_delta", "mean"),
        )
        .sort_values("log_loss")
        if not enrichment_results.empty
        else pd.DataFrame()
    )

    raw_path = RAW_DIR / "results.csv"
    data_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()

    report = {
        "experiment": {
            "id": f"wci-{pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%SZ')}",
            "data_sha256": data_sha256,
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "random_seed": 2026,
            "training_window_years": 28,
            "simulation_runs": simulations,
        },
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "data": {
            "matches": int(len(matches)),
            "completed_matches": int(matches["is_completed"].sum()),
            "first_match": matches["date"].min().date().isoformat(),
            "last_result": matches.loc[matches["is_completed"], "date"].max().date().isoformat(),
            "future_fixtures": int((~matches["is_completed"]).sum()),
        },
        "model": {
            "blend_weights": {
                "linear": float(model.blend_weights[0]),
                "gradient_boosting": float(model.blend_weights[1]),
                "poisson": float(model.blend_weights[2]),
            },
            "temperature": model.temperature,
            "dixon_coles_rho": model.dc_rho,
        },
        "backtests": backtests.to_dict(orient="records"),
        "benchmark_ladder": benchmarks.to_dict(orient="records"),
        "bootstrap_intervals": bootstrap,
        "conformal": {
            **conformal_summary,
            "evaluated_matches": int(len(conformal_matches)),
        },
        "ablation": ablation.to_dict(orient="records"),
        "drift": drift.to_dict(orient="records"),
        "data_coverage": coverage.assign(
            first_date=lambda frame: frame["first_date"].astype(str),
            last_date=lambda frame: frame["last_date"].astype(str),
        ).to_dict(orient="records"),
        "enrichment_benchmark": enrichment_results.to_dict(orient="records"),
        "enrichment_summary": enrichment_summary.to_dict(orient="records"),
        "reliability": reliability.to_dict(orient="records"),
        "confidence_reliability": confidence_curve.to_dict(orient="records"),
        "confusion_matrix": (
            confusion_table(backtest_predictions) if not backtest_predictions.empty else []
        ),
        "feature_importance": (
            model.feature_importance_.to_dict(orient="records")
            if model.feature_importance_ is not None
            else []
        ),
        "ratings": ratings.head(150).to_dict(orient="records"),
        "team_embedding": embedding.to_dict(orient="records"),
        "turkey_blind_experiment": turkey_blind_metadata,
        "tournament_blind_experiment": tournament_blind_metadata,
    }

    joblib.dump({"model": model, "builder": builder}, MODEL_PATH)
    features.to_parquet(FEATURES_PATH, index=False)
    fixture_predictions.to_csv(FIXTURES_PATH, index=False)
    simulation.to_csv(SIMULATION_PATH, index=False)
    turkey_schedule.to_csv(TURKEY_PATH, index=False)
    turkey_scenarios.to_csv(TURKEY_SCENARIOS_PATH, index=False)
    turkey_blind.to_csv(TURKEY_BLIND_PATH, index=False)
    tournament_blind.to_csv(TOURNAMENT_BLIND_PATH, index=False)
    blind_leaderboard.to_csv(BLIND_LEADERBOARD_PATH, index=False)
    enrichment_results.to_csv(ENRICHMENT_BENCHMARK_PATH, index=False)
    coverage.to_csv(DATA_COVERAGE_PATH, index=False)
    backtest_predictions.to_csv(BACKTEST_PATH, index=False)
    benchmarks.to_csv(BENCHMARK_PATH, index=False)
    ablation.to_csv(ABLATION_PATH, index=False)
    drift.to_csv(DRIFT_PATH, index=False)
    if run_backtest:
        visual_paths = generate_visual_report(
            backtests,
            benchmarks,
            reliability,
            ablation,
            simulation,
            fixture_predictions,
            ratings,
            turkey_schedule,
            turkey_scenarios,
            turkey_blind,
            tournament_blind,
            blind_leaderboard,
            enrichment_results,
            coverage,
        )
        report["visual_assets"] = [str(path.relative_to(Path.cwd())) for path in visual_paths]
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate World Cup Intelligence.")
    parser.add_argument("--refresh", action="store_true", help="Refresh the source dataset.")
    parser.add_argument("--simulations", type=int, default=5000)
    parser.add_argument("--skip-backtest", action="store_true")
    args = parser.parse_args()
    report = run_pipeline(
        refresh=args.refresh,
        simulations=args.simulations,
        run_backtest=not args.skip_backtest,
    )
    print(json.dumps(report["model"], indent=2))
    print(f"Artifacts written to {Path(ARTIFACT_DIR).resolve()}")


if __name__ == "__main__":
    main()
