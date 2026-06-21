from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "release" / "runtime-manifest.json"

RUNTIME_FILES = [
    "artifacts/model.joblib",
    "artifacts/report.json",
    "artifacts/advanced_report.json",
    "artifacts/predictions_2026.csv",
    "artifacts/simulation_2026.csv",
    "artifacts/backtest_predictions.csv",
    "artifacts/benchmark_ladder.csv",
    "artifacts/blind_team_leaderboard.csv",
    "artifacts/data_coverage.csv",
    "artifacts/enrichment_benchmark.csv",
    "artifacts/feature_ablation.csv",
    "artifacts/feature_drift.csv",
    "artifacts/weather_benchmark.csv",
    "artifacts/architecture_benchmark.csv",
    "artifacts/event_benchmark.csv",
    "artifacts/lineup_benchmark.csv",
    "artifacts/advanced_experiment_matrix.csv",
    "artifacts/squad_2026_analytics.csv",
    "artifacts/squad_2026_players.csv",
    "artifacts/turkiye_2026.csv",
    "artifacts/turkiye_blind_forecast.csv",
    "artifacts/turkiye_scenarios.csv",
    "artifacts/world_cup_2026_blind_forecast.csv",
]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v2.0.0")
    args = parser.parse_args()
    missing = [path for path in RUNTIME_FILES if not (ROOT / path).exists()]
    if missing:
        raise SystemExit("Missing runtime artifacts:\n" + "\n".join(missing))
    entries = [
        {
            "path": path,
            "bytes": (ROOT / path).stat().st_size,
            "sha256": file_hash(ROOT / path),
        }
        for path in RUNTIME_FILES
    ]
    manifest = {
        "version": args.version,
        "generated_from": "versioned pipeline outputs",
        "files": entries,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    archive_path = dist / f"world-cup-forecaster-{args.version}.zip"
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in RUNTIME_FILES:
            archive.write(ROOT / path, path)
        archive.write(MANIFEST_PATH, "release/runtime-manifest.json")
    checksum_path = archive_path.with_suffix(".zip.sha256")
    checksum_path.write_text(
        f"{file_hash(archive_path)}  {archive_path.name}\n", encoding="utf-8"
    )
    print(archive_path)


if __name__ == "__main__":
    main()
