from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import MODEL_PATH, REPORT_PATH, SIMULATION_PATH
from .features import FEATURE_COLUMNS

app = FastAPI(
    title="World Cup Forecaster API",
    description="Calibrated match probabilities and 2026 tournament forecasts.",
    version="2.0.0",
)


class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    neutral: bool = True
    home_strength_shock: float = Field(default=0, ge=-250, le=250)
    away_strength_shock: float = Field(default=0, ge=-250, le=250)


class OutcomeProbabilities(BaseModel):
    away_win: float
    draw: float
    home_win: float


class ScoreForecast(BaseModel):
    expected_home_goals: float
    expected_away_goals: float
    most_likely_scoreline: str
    scoreline_probability: float


class MatchResponse(BaseModel):
    home_team: str
    away_team: str
    probabilities: OutcomeProbabilities
    score_forecast: ScoreForecast
    model_disagreement: float
    experiment_id: str


@lru_cache(maxsize=1)
def runtime() -> tuple[object, object, dict[str, object]]:
    if not MODEL_PATH.exists() or not REPORT_PATH.exists():
        raise RuntimeError("Model artifacts are unavailable. Run the training pipeline first.")
    bundle = joblib.load(MODEL_PATH)
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    return bundle["model"], bundle["builder"], report


@app.get("/health")
def health() -> dict[str, object]:
    ready = MODEL_PATH.exists() and REPORT_PATH.exists()
    response: dict[str, object] = {"status": "ready" if ready else "not_ready"}
    if ready:
        _, _, report = runtime()
        response["experiment_id"] = report["experiment"]["id"]
        response["last_result"] = report["data"]["last_result"]
    return response


@app.get("/v1/teams")
def teams() -> list[str]:
    _, builder, _ = runtime()
    return sorted(builder.states)


@app.get("/v1/tournament")
def tournament() -> list[dict[str, object]]:
    if not SIMULATION_PATH.exists():
        raise HTTPException(status_code=503, detail="Tournament artifact unavailable.")
    return pd.read_csv(SIMULATION_PATH).to_dict(orient="records")


@app.post("/v1/predict", response_model=MatchResponse)
def predict(request: MatchRequest) -> MatchResponse:
    if request.home_team == request.away_team:
        raise HTTPException(status_code=422, detail="Teams must be different.")
    model, builder, report = runtime()
    unknown = [
        team
        for team in (request.home_team, request.away_team)
        if team not in builder.states
    ]
    if unknown:
        raise HTTPException(status_code=404, detail=f"Unknown team(s): {', '.join(unknown)}")
    row = pd.DataFrame(
        [
            builder.make_features(
                request.home_team,
                request.away_team,
                pd.Timestamp("2026-07-01"),
                "FIFA World Cup",
                "United States",
                request.neutral,
            )
        ]
    )
    shock = request.home_strength_shock - request.away_strength_shock
    row["elo_diff"] += shock / 400.0
    row["prestige_diff"] += shock / 800.0
    probabilities = model.predict_proba(row[FEATURE_COLUMNS])[0]
    components = model.component_probabilities(row[FEATURE_COLUMNS])
    disagreement = float(np.std(np.stack(list(components.values())), axis=0).mean())
    home_goals, away_goals = model.predict_goals(row[FEATURE_COLUMNS])
    matrix = model.score_matrix(row[FEATURE_COLUMNS])[0]
    likely_home, likely_away = (int(value) for value in np.unravel_index(int(matrix.argmax()), matrix.shape))
    return MatchResponse(
        home_team=request.home_team,
        away_team=request.away_team,
        probabilities=OutcomeProbabilities(
            away_win=float(probabilities[0]),
            draw=float(probabilities[1]),
            home_win=float(probabilities[2]),
        ),
        score_forecast=ScoreForecast(
            expected_home_goals=float(home_goals[0]),
            expected_away_goals=float(away_goals[0]),
            most_likely_scoreline=f"{likely_home}–{likely_away}",
            scoreline_probability=float(matrix[likely_home, likely_away]),
        ),
        model_disagreement=disagreement,
        experiment_id=str(report["experiment"]["id"]),
    )


def run() -> None:
    import uvicorn

    uvicorn.run("worldcup_predictor.api:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
