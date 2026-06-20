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
    }
