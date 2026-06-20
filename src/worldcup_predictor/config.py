from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACT_DIR = ROOT / "artifacts"
ASSET_DIR = ROOT / "assets"
VISUAL_DIR = ASSET_DIR / "visuals"

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/results.csv"
)
GOALSCORERS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/goalscorers.csv"
)
SHOOTOUTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/shootouts.csv"
)
FIFA_RANKINGS_URL = (
    "https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/"
    "master/ranking_fifa_historical.csv"
)
FJELSTUL_BASE_URL = (
    "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv"
)

RANDOM_SEED = 2026
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
FEATURES_PATH = ARTIFACT_DIR / "features.parquet"
REPORT_PATH = ARTIFACT_DIR / "report.json"
FIXTURES_PATH = ARTIFACT_DIR / "predictions_2026.csv"
SIMULATION_PATH = ARTIFACT_DIR / "simulation_2026.csv"
BACKTEST_PATH = ARTIFACT_DIR / "backtest_predictions.csv"
BENCHMARK_PATH = ARTIFACT_DIR / "benchmark_ladder.csv"
ABLATION_PATH = ARTIFACT_DIR / "feature_ablation.csv"
DRIFT_PATH = ARTIFACT_DIR / "feature_drift.csv"
TURKEY_PATH = ARTIFACT_DIR / "turkiye_2026.csv"
TURKEY_SCENARIOS_PATH = ARTIFACT_DIR / "turkiye_scenarios.csv"
TURKEY_BLIND_PATH = ARTIFACT_DIR / "turkiye_blind_forecast.csv"
TOURNAMENT_BLIND_PATH = ARTIFACT_DIR / "world_cup_2026_blind_forecast.csv"
BLIND_LEADERBOARD_PATH = ARTIFACT_DIR / "blind_team_leaderboard.csv"
ENRICHMENT_BENCHMARK_PATH = ARTIFACT_DIR / "enrichment_benchmark.csv"
DATA_COVERAGE_PATH = ARTIFACT_DIR / "data_coverage.csv"
WEATHER_BENCHMARK_PATH = ARTIFACT_DIR / "weather_benchmark.csv"
ARCHITECTURE_BENCHMARK_PATH = ARTIFACT_DIR / "architecture_benchmark.csv"
CONDITIONS_PATH = RAW_DIR / "world_cup_match_conditions.parquet"
STATSBOMB_SUMMARY_PATH = RAW_DIR / "statsbomb_international_summary.parquet"
EVENT_BENCHMARK_PATH = ARTIFACT_DIR / "event_benchmark.csv"
SQUAD_2026_PATH = RAW_DIR / "worldcup_2026_squads.json"
SQUAD_2026_ANALYTICS_PATH = ARTIFACT_DIR / "squad_2026_analytics.csv"
SQUAD_2026_PLAYERS_PATH = ARTIFACT_DIR / "squad_2026_players.csv"
LINEUP_SUMMARY_PATH = RAW_DIR / "statsbomb_lineup_summary.parquet"
LINEUP_BENCHMARK_PATH = ARTIFACT_DIR / "lineup_benchmark.csv"
ADVANCED_MATRIX_PATH = ARTIFACT_DIR / "advanced_experiment_matrix.csv"
ADVANCED_REPORT_PATH = ARTIFACT_DIR / "advanced_report.json"
RUNTIME_MANIFEST_PATH = ROOT / "release" / "runtime-manifest.json"
