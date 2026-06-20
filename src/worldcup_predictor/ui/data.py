from __future__ import annotations

import json

import joblib
import pandas as pd
import streamlit as st

from ..artifacts import ensure_runtime_artifacts
from ..config import (
    ABLATION_PATH,
    ADVANCED_MATRIX_PATH,
    ADVANCED_REPORT_PATH,
    ARCHITECTURE_BENCHMARK_PATH,
    BACKTEST_PATH,
    BENCHMARK_PATH,
    BLIND_LEADERBOARD_PATH,
    DATA_COVERAGE_PATH,
    DRIFT_PATH,
    ENRICHMENT_BENCHMARK_PATH,
    FIXTURES_PATH,
    LINEUP_BENCHMARK_PATH,
    MODEL_PATH,
    REPORT_PATH,
    SIMULATION_PATH,
    SQUAD_2026_ANALYTICS_PATH,
    SQUAD_2026_PLAYERS_PATH,
    TOURNAMENT_BLIND_PATH,
    TURKEY_BLIND_PATH,
    TURKEY_PATH,
    TURKEY_SCENARIOS_PATH,
)

REQUIRED_PATHS = [
    MODEL_PATH,
    REPORT_PATH,
    FIXTURES_PATH,
    SIMULATION_PATH,
    BACKTEST_PATH,
    BENCHMARK_PATH,
    ABLATION_PATH,
    DRIFT_PATH,
    ENRICHMENT_BENCHMARK_PATH,
    DATA_COVERAGE_PATH,
    ADVANCED_REPORT_PATH,
    ADVANCED_MATRIX_PATH,
    ARCHITECTURE_BENCHMARK_PATH,
    LINEUP_BENCHMARK_PATH,
    SQUAD_2026_ANALYTICS_PATH,
    SQUAD_2026_PLAYERS_PATH,
    TOURNAMENT_BLIND_PATH,
    BLIND_LEADERBOARD_PATH,
    TURKEY_BLIND_PATH,
    TURKEY_PATH,
    TURKEY_SCENARIOS_PATH,
]


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_outputs() -> dict[str, object]:
    return {
        "report": json.loads(REPORT_PATH.read_text(encoding="utf-8")),
        "fixtures": pd.read_csv(FIXTURES_PATH, parse_dates=["date"]),
        "simulation": pd.read_csv(SIMULATION_PATH),
        "backtest_predictions": pd.read_csv(
            BACKTEST_PATH, parse_dates=["date"]
        ),
        "benchmarks": pd.read_csv(BENCHMARK_PATH),
        "ablation": pd.read_csv(ABLATION_PATH),
        "drift": pd.read_csv(DRIFT_PATH),
        "enrichment": pd.read_csv(ENRICHMENT_BENCHMARK_PATH),
        "coverage": pd.read_csv(DATA_COVERAGE_PATH),
        "advanced_report": json.loads(
            ADVANCED_REPORT_PATH.read_text(encoding="utf-8")
        ),
        "advanced_matrix": pd.read_csv(ADVANCED_MATRIX_PATH),
        "architectures": pd.read_csv(ARCHITECTURE_BENCHMARK_PATH),
        "lineup_benchmark": pd.read_csv(LINEUP_BENCHMARK_PATH),
        "squad_teams": pd.read_csv(SQUAD_2026_ANALYTICS_PATH),
        "squad_players": pd.read_csv(
            SQUAD_2026_PLAYERS_PATH, parse_dates=["birth_date"]
        ),
        "tournament_blind": pd.read_csv(
            TOURNAMENT_BLIND_PATH, parse_dates=["date"]
        ),
        "blind_leaderboard": pd.read_csv(BLIND_LEADERBOARD_PATH),
        "turkey": pd.read_csv(TURKEY_PATH, parse_dates=["date"]),
        "turkey_blind": pd.read_csv(
            TURKEY_BLIND_PATH, parse_dates=["date"]
        ),
        "turkey_scenarios": pd.read_csv(TURKEY_SCENARIOS_PATH),
    }


def load_runtime() -> tuple[dict[str, object], dict[str, object]]:
    if not all(path.exists() for path in REQUIRED_PATHS):
        errors = ensure_runtime_artifacts()
        if errors:
            st.error(
                "Runtime artifacts are incomplete. Install the versioned "
                "GitHub Release bundle or train locally."
            )
            st.code("\n".join(errors))
            st.stop()
    return load_bundle(), load_outputs()
