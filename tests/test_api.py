import numpy as np

from fastapi.testclient import TestClient

from worldcup_predictor import api
from worldcup_predictor.features import FEATURE_COLUMNS


class DummyBuilder:
    states = {"Brazil": object(), "Argentina": object()}

    @staticmethod
    def make_features(*args, **kwargs):
        return {feature: 0.0 for feature in FEATURE_COLUMNS}


class DummyModel:
    @staticmethod
    def predict_proba(frame):
        return np.array([[0.28, 0.25, 0.47]])

    @staticmethod
    def component_probabilities(frame):
        return {
            "linear": np.array([[0.30, 0.25, 0.45]]),
            "boost": np.array([[0.25, 0.25, 0.50]]),
            "dixon_coles": np.array([[0.29, 0.26, 0.45]]),
        }

    @staticmethod
    def predict_goals(frame):
        return np.array([1.6]), np.array([1.1])

    @staticmethod
    def score_matrix(frame, max_goals=7):
        matrix = np.full((1, max_goals + 1, max_goals + 1), 0.001)
        matrix[0, 2, 1] = 0.4
        return matrix / matrix.sum(axis=(1, 2), keepdims=True)


def test_health_has_stable_contract():
    response = TestClient(api.app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"ready", "not_ready"}


def test_prediction_contract_and_probability_simplex(monkeypatch):
    monkeypatch.setattr(
        api,
        "runtime",
        lambda: (
            DummyModel(),
            DummyBuilder(),
            {"experiment": {"id": "wci-test"}},
        ),
    )
    response = TestClient(api.app).post(
        "/v1/predict",
        json={
            "home_team": "Brazil",
            "away_team": "Argentina",
            "neutral": True,
            "home_strength_shock": 25,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    probabilities = payload["probabilities"]
    assert abs(sum(probabilities.values()) - 1.0) < 1e-8
    assert set(probabilities) == {"away_win", "draw", "home_win"}
    assert set(payload) == {
        "away_team",
        "experiment_id",
        "home_team",
        "model_disagreement",
        "probabilities",
        "score_forecast",
    }
    score_forecast = payload["score_forecast"]
    assert set(score_forecast) == {
        "expected_home_goals",
        "expected_away_goals",
        "most_likely_scoreline",
        "scoreline_probability",
    }
    assert score_forecast["most_likely_scoreline"] == "2–1"
    assert 0.0 <= score_forecast["scoreline_probability"] <= 1.0
