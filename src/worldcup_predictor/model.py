from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_SEED
from .features import FEATURE_COLUMNS

OUTCOMES = ["Away win", "Draw", "Home win"]


def poisson_outcome_probabilities(
    home_lambda: np.ndarray,
    away_lambda: np.ndarray,
    max_goals: int = 10,
    rho: float = 0.0,
) -> np.ndarray:
    """Convert Dixon-Coles-adjusted Poisson rates into away/draw/home probabilities."""
    matrix = dixon_coles_score_matrix(home_lambda, away_lambda, max_goals, rho)
    away_win = np.triu(matrix, k=1).sum(axis=(1, 2))
    draw = np.diagonal(matrix, axis1=1, axis2=2).sum(axis=1)
    home_win = np.tril(matrix, k=-1).sum(axis=(1, 2))
    probabilities = np.column_stack([away_win, draw, home_win])
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def dixon_coles_score_matrix(
    home_lambda: np.ndarray,
    away_lambda: np.ndarray,
    max_goals: int = 10,
    rho: float = 0.0,
) -> np.ndarray:
    home_lambda = np.clip(np.asarray(home_lambda, dtype=float), 0.05, 6.5)
    away_lambda = np.clip(np.asarray(away_lambda, dtype=float), 0.05, 6.5)
    goals = np.arange(max_goals + 1)
    from math import factorial

    denominators = np.array([factorial(int(g)) for g in goals])
    home_p = np.exp(-home_lambda[:, None]) * home_lambda[:, None] ** goals / denominators
    away_p = np.exp(-away_lambda[:, None]) * away_lambda[:, None] ** goals / denominators
    matrix = home_p[:, :, None] * away_p[:, None, :]
    if rho:
        matrix[:, 0, 0] *= np.clip(1 - home_lambda * away_lambda * rho, 0.05, 2.0)
        matrix[:, 0, 1] *= np.clip(1 + home_lambda * rho, 0.05, 2.0)
        matrix[:, 1, 0] *= np.clip(1 + away_lambda * rho, 0.05, 2.0)
        matrix[:, 1, 1] *= np.clip(1 - rho, 0.05, 2.0)
    return matrix / matrix.sum(axis=(1, 2), keepdims=True)


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    observed = np.eye(3)[y_true.astype(int)]
    return float(np.mean(np.sum((probabilities - observed) ** 2, axis=1)))


def ranked_probability_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    observed = np.eye(3)[y_true.astype(int)]
    return float(np.mean(np.sum((np.cumsum(probabilities, axis=1)[:, :-1] -
                                 np.cumsum(observed, axis=1)[:, :-1]) ** 2, axis=1) / 2))


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    adjusted = np.power(np.clip(probabilities, 1e-9, 1), 1.0 / temperature)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


@dataclass
class Evaluation:
    log_loss: float
    brier: float
    rps: float
    accuracy: float
    baseline_log_loss: float
    expected_calibration_error: float
    sharpness: float


class MatchEnsemble:
    """Calibrated blend of linear, nonlinear, and generative score models."""

    def __init__(
        self,
        random_state: int = RANDOM_SEED,
        feature_columns: list[str] | None = None,
    ):
        self.random_state = random_state
        self.feature_columns = list(feature_columns or FEATURE_COLUMNS)
        self.linear = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.7, max_iter=1500),
        )
        self.boost = HistGradientBoostingClassifier(
            learning_rate=0.055,
            max_iter=260,
            max_leaf_nodes=24,
            l2_regularization=1.5,
            min_samples_leaf=28,
            random_state=random_state,
        )
        self.home_goals = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.05,
            max_iter=220,
            max_leaf_nodes=20,
            l2_regularization=2.0,
            min_samples_leaf=30,
            random_state=random_state,
        )
        self.away_goals = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.05,
            max_iter=220,
            max_leaf_nodes=20,
            l2_regularization=2.0,
            min_samples_leaf=30,
            random_state=random_state + 1,
        )
        self.blend_weights = np.array([0.25, 0.50, 0.25])
        self.temperature = 1.0
        self.dc_rho = 0.0
        self.feature_importance_: pd.DataFrame | None = None

    @staticmethod
    def _sample_weight(dates: pd.Series, importance: pd.Series) -> np.ndarray:
        age_years = (dates.max() - dates).dt.days.to_numpy() / 365.25
        recency = np.power(0.5, age_years / 8.0)
        return recency * (0.65 + importance.to_numpy())

    def fit(self, frame: pd.DataFrame, tune: bool = True) -> MatchEnsemble:
        data = frame.dropna(subset=["target", "home_score", "away_score"]).copy()
        data = data.sort_values("date")
        if len(data) < 500:
            raise ValueError("At least 500 completed matches are required to fit the model.")

        split = int(len(data) * 0.82) if tune else len(data)
        train = data.iloc[:split]
        calibration = data.iloc[split:] if tune else data.iloc[-min(1000, len(data)) :]
        self._fit_estimators(train)

        if tune and len(calibration) >= 100:
            x_cal = calibration[self.feature_columns]
            y_cal = calibration["target"].astype(int).to_numpy()
            self.dc_rho = self._fit_dixon_coles_rho(calibration)
            components = self._component_probabilities(x_cal)
            self.blend_weights, self.temperature = self._optimize_blend(
                components, y_cal
            )

            result = permutation_importance(
                self.boost,
                x_cal,
                y_cal,
                scoring="neg_log_loss",
                n_repeats=3,
                random_state=self.random_state,
            )
            self.feature_importance_ = (
                pd.DataFrame(
                    {
                        "feature": self.feature_columns,
                        "importance": result.importances_mean,
                    }
                )
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
            # After hyperparameters and calibration are selected chronologically, use every
            # available result for the production estimators.
            self._fit_estimators(data)
        return self

    @staticmethod
    def _optimize_blend(
        components: tuple[np.ndarray, ...], y_true: np.ndarray
    ) -> tuple[np.ndarray, float]:
        stacked = np.stack(components, axis=0)

        def objective(parameters: np.ndarray) -> float:
            logits = parameters[:3].copy()
            logits -= logits.max()
            weights = np.exp(logits) / np.exp(logits).sum()
            temperature = float(np.exp(parameters[3]))
            blended = np.tensordot(weights, stacked, axes=(0, 0))
            calibrated = apply_temperature(blended, temperature)
            regularization = 0.001 * np.sum((weights - 1 / 3) ** 2)
            return float(
                log_loss(y_true, calibrated, labels=[0, 1, 2]) + regularization
            )

        starts = (
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([1.5, -0.5, -0.5, 0.0]),
            np.array([-0.5, 1.5, -0.5, 0.0]),
            np.array([-0.5, -0.5, 1.5, 0.0]),
        )
        best = None
        for start in starts:
            result = minimize(
                objective,
                start,
                method="L-BFGS-B",
                bounds=[(-5, 5), (-5, 5), (-5, 5), (np.log(0.55), np.log(1.8))],
            )
            if best is None or result.fun < best.fun:
                best = result
        logits = best.x[:3] - best.x[:3].max()
        weights = np.exp(logits) / np.exp(logits).sum()
        return weights, float(np.exp(best.x[3]))

    def _fit_estimators(self, data: pd.DataFrame) -> None:
        x = data[self.feature_columns]
        y = data["target"].astype(int)
        weights = self._sample_weight(data["date"], data["importance"])
        self.linear.fit(x, y, logisticregression__sample_weight=weights)
        self.boost.fit(x, y, sample_weight=weights)
        self.home_goals.fit(x, data["home_score"], sample_weight=weights)
        self.away_goals.fit(x, data["away_score"], sample_weight=weights)

    def _fit_dixon_coles_rho(self, calibration: pd.DataFrame) -> float:
        home_lambda = np.clip(
            self.home_goals.predict(calibration[self.feature_columns]), 0.05, 6.5
        )
        away_lambda = np.clip(
            self.away_goals.predict(calibration[self.feature_columns]), 0.05, 6.5
        )
        home_score = calibration["home_score"].astype(int).to_numpy()
        away_score = calibration["away_score"].astype(int).to_numpy()
        valid = (home_score <= 10) & (away_score <= 10)
        best_rho, best_nll = 0.0, float("inf")
        for rho in np.arange(-0.18, 0.181, 0.01):
            matrices = dixon_coles_score_matrix(home_lambda, away_lambda, 10, float(rho))
            likelihood = matrices[
                np.arange(len(matrices))[valid], home_score[valid], away_score[valid]
            ]
            nll = -float(np.log(np.clip(likelihood, 1e-12, 1)).mean())
            if nll < best_nll:
                best_rho, best_nll = float(rho), nll
        return best_rho

    def _component_probabilities(self, x: pd.DataFrame) -> tuple[np.ndarray, ...]:
        linear = self.linear.predict_proba(x)
        boost = self.boost.predict_proba(x)
        home_lambda = self.home_goals.predict(x)
        away_lambda = self.away_goals.predict(x)
        poisson = poisson_outcome_probabilities(
            home_lambda, away_lambda, rho=self.dc_rho
        )
        return linear, boost, poisson

    def component_probabilities(self, x: pd.DataFrame) -> dict[str, np.ndarray]:
        linear, boost, poisson = self._component_probabilities(
            x[self.feature_columns]
        )
        return {"linear": linear, "boost": boost, "dixon_coles": poisson}

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        components = self._component_probabilities(x[self.feature_columns])
        blended = sum(w * p for w, p in zip(self.blend_weights, components))
        return apply_temperature(blended, self.temperature)

    def predict_goals(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.clip(self.home_goals.predict(x[self.feature_columns]), 0.05, 6.5),
            np.clip(self.away_goals.predict(x[self.feature_columns]), 0.05, 6.5),
        )

    def evaluate(self, frame: pd.DataFrame) -> Evaluation:
        data = frame.dropna(subset=["target"])
        y = data["target"].astype(int).to_numpy()
        probabilities = self.predict_proba(data)
        baseline = np.tile(np.bincount(y, minlength=3) / len(y), (len(y), 1))
        return Evaluation(
            log_loss=float(log_loss(y, probabilities, labels=[0, 1, 2])),
            brier=multiclass_brier(y, probabilities),
            rps=ranked_probability_score(y, probabilities),
            accuracy=float(accuracy_score(y, probabilities.argmax(axis=1))),
            baseline_log_loss=float(log_loss(y, baseline, labels=[0, 1, 2])),
            expected_calibration_error=expected_calibration_error(y, probabilities),
            sharpness=float(np.mean(np.max(probabilities, axis=1))),
        )

    def score_matrix(self, x: pd.DataFrame, max_goals: int = 7) -> np.ndarray:
        home_lambda, away_lambda = self.predict_goals(x)
        return dixon_coles_score_matrix(
            home_lambda, away_lambda, max_goals=max_goals, rho=self.dc_rho
        )


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y_true
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            error += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return float(error)
