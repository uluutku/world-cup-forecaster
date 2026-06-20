from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .architectures import GpuXGBoostClassifier, ResidualTabularNetwork
from .features import FEATURE_COLUMNS
from .model import (
    MatchEnsemble,
    expected_calibration_error,
    multiclass_brier,
    ranked_probability_score,
)

FEATURE_GROUPS = {
    "rating uncertainty": [
        "rating_confidence_diff",
        "rating_uncertainty",
        "volatility_diff",
        "consistency_diff",
    ],
    "recent dynamics": [
        "form_diff",
        "momentum_diff",
        "attack_trend_diff",
        "defence_trend_diff",
        "activity_diff",
        "rest_diff",
    ],
    "score process": [
        "goals_for_diff",
        "goals_against_diff",
        "goal_balance_diff",
        "clean_sheet_diff",
        "scoring_rate_diff",
    ],
    "opponent context": [
        "sos_diff",
        "prestige_diff",
        "major_form_diff",
        "experience_diff",
    ],
    "venue and history": [
        "h2h_form",
        "h2h_goal_balance",
        "h2h_experience",
        "neutral",
        "home_advantage",
        "host_home",
        "host_away",
        "importance",
        "is_world_cup",
    ],
}


def _metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1, 2])),
        "brier": multiclass_brier(y, probabilities),
        "rps": ranked_probability_score(y, probabilities),
        "accuracy": float(accuracy_score(y, probabilities.argmax(axis=1))),
        "ece": expected_calibration_error(y, probabilities),
        "sharpness": float(probabilities.max(axis=1).mean()),
        "entropy": float(
            np.mean(-np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1)), axis=1))
        ),
    }


def world_cup_walk_forward(
    features: pd.DataFrame,
    editions: tuple[int, ...] = (2006, 2010, 2014, 2018, 2022),
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, float]] = []
    predictions: list[pd.DataFrame] = []
    benchmarks: list[dict[str, float | str]] = []
    selected_features = feature_columns or FEATURE_COLUMNS
    completed = features.dropna(subset=["target"]).copy()
    for edition in editions:
        test = completed.loc[
            completed["tournament"].eq("FIFA World Cup")
            & completed["date"].dt.year.eq(edition)
        ]
        if test.empty:
            continue
        cutoff = test["date"].min()
        train = completed.loc[
            (completed["date"] < cutoff)
            & (completed["date"] >= cutoff - pd.DateOffset(years=28))
        ]
        model = MatchEnsemble(
            random_state=edition, feature_columns=selected_features
        ).fit(train, tune=True)
        y = test["target"].astype(int).to_numpy()
        probabilities = model.predict_proba(test)
        metrics = _metrics(y, probabilities)

        prior = np.bincount(train["target"].astype(int), minlength=3).astype(float)
        prior /= prior.sum()
        prior_probabilities = np.tile(prior, (len(test), 1))
        elo_features = ["elo_diff", "home_advantage", "neutral"]
        elo_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, max_iter=1000),
        )
        elo_model.fit(train[elo_features], train["target"].astype(int))
        benchmark_probabilities = {
            "Historical prior": prior_probabilities,
            "Elo-only logistic": elo_model.predict_proba(test[elo_features]),
            "Linear state model": model.component_probabilities(test)["linear"],
            "Nonlinear boosting": model.component_probabilities(test)["boost"],
            "Dixon-Coles goals": model.component_probabilities(test)["dixon_coles"],
            "Calibrated ensemble": probabilities,
        }
        for name, model_probabilities in benchmark_probabilities.items():
            benchmarks.append(
                {"edition": edition, "model": name, **_metrics(y, model_probabilities)}
            )

        summaries.append(
            {
                "edition": edition,
                "matches": len(test),
                **metrics,
                "baseline_log_loss": _metrics(y, prior_probabilities)["log_loss"],
            }
        )
        fold = test[
            ["date", "home_team", "away_team", "home_score", "away_score", "target"]
        ].copy()
        fold[["p_away", "p_draw", "p_home"]] = probabilities
        components = model.component_probabilities(test)
        for component, values in components.items():
            fold[f"{component}_confidence"] = values.max(axis=1)
        fold["model_disagreement"] = np.std(
            np.stack(list(components.values()), axis=0), axis=0
        ).mean(axis=1)
        fold["edition"] = edition
        predictions.append(fold)
    return (
        pd.DataFrame(summaries),
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(benchmarks),
    )


def enrichment_benchmark(
    features: pd.DataFrame,
    configurations: dict[str, list[str]],
    editions: tuple[int, ...] = (2006, 2010, 2014, 2018, 2022),
) -> pd.DataFrame:
    """Evaluate feature-family combinations on identical tournament cutoffs."""
    completed = features.dropna(subset=["target"]).copy()
    rows = []
    for edition in editions:
        test = completed.loc[
            completed["tournament"].eq("FIFA World Cup")
            & completed["date"].dt.year.eq(edition)
        ]
        if test.empty:
            continue
        cutoff = test["date"].min()
        train = completed.loc[
            (completed["date"] < cutoff)
            & (completed["date"] >= cutoff - pd.DateOffset(years=28))
        ]
        y = test["target"].astype(int).to_numpy()
        for configuration, columns in configurations.items():
            model = MatchEnsemble(
                random_state=edition,
                feature_columns=columns,
            ).fit(train, tune=True)
            probabilities = model.predict_proba(test)
            rows.append(
                {
                    "edition": edition,
                    "configuration": configuration,
                    "features": len(columns),
                    **_metrics(y, probabilities),
                }
            )
    result = pd.DataFrame(rows)
    baseline = result.loc[
        result["configuration"].eq("Baseline scores + state"),
        ["edition", "log_loss", "accuracy", "brier", "rps", "ece"],
    ].rename(
        columns={
            "log_loss": "baseline_log_loss",
            "accuracy": "baseline_accuracy",
            "brier": "baseline_brier",
            "rps": "baseline_rps",
            "ece": "baseline_ece",
        }
    )
    result = result.merge(baseline, on="edition", how="left")
    result["log_loss_delta"] = result["log_loss"] - result["baseline_log_loss"]
    result["accuracy_delta"] = result["accuracy"] - result["baseline_accuracy"]
    result["brier_delta"] = result["brier"] - result["baseline_brier"]
    result["rps_delta"] = result["rps"] - result["baseline_rps"]
    result["ece_delta"] = result["ece"] - result["baseline_ece"]
    return result.sort_values(["edition", "log_loss"]).reset_index(drop=True)


def architecture_benchmark(
    features: pd.DataFrame,
    feature_columns: list[str],
    editions: tuple[int, ...] = (2006, 2010, 2014, 2018, 2022),
    include_neural: bool = True,
) -> pd.DataFrame:
    """Compare calibrated model families on identical chronological folds."""
    completed = features.dropna(subset=["target"]).copy()
    rows = []
    for edition in editions:
        test = completed.loc[
            completed["tournament"].eq("FIFA World Cup")
            & completed["date"].dt.year.eq(edition)
        ]
        if test.empty:
            continue
        cutoff = test["date"].min()
        train = completed.loc[
            (completed["date"] < cutoff)
            & (completed["date"] >= cutoff - pd.DateOffset(years=28))
        ]
        y = test["target"].astype(int).to_numpy()
        candidates: list[tuple[str, object]] = [
            (
                "Calibrated hybrid",
                MatchEnsemble(
                    random_state=edition, feature_columns=feature_columns
                ),
            ),
            (
                "CUDA XGBoost",
                GpuXGBoostClassifier(
                    feature_columns, random_state=edition, use_gpu=True
                ),
            ),
        ]
        if include_neural:
            candidates.append(
                (
                    "CUDA residual network",
                    ResidualTabularNetwork(
                        feature_columns, random_state=edition, use_gpu=True
                    ),
                )
            )
        for name, candidate in candidates:
            candidate.fit(train)
            probabilities = candidate.predict_proba(test)
            metadata = getattr(candidate, "metadata", None)
            rows.append(
                {
                    "edition": edition,
                    "architecture": name,
                    "features": len(feature_columns),
                    "device": metadata.device if metadata else "cpu",
                    "training_seconds": (
                        metadata.training_seconds if metadata else np.nan
                    ),
                    "parameters": metadata.parameters if metadata else np.nan,
                    "temperature": (
                        metadata.temperature
                        if metadata
                        else getattr(candidate, "temperature", 1.0)
                    ),
                    **_metrics(y, probabilities),
                }
            )
    result = pd.DataFrame(rows)
    baseline = result.loc[
        result["architecture"].eq("Calibrated hybrid"),
        ["edition", "log_loss", "accuracy"],
    ].rename(
        columns={
            "log_loss": "baseline_log_loss",
            "accuracy": "baseline_accuracy",
        }
    )
    result = result.merge(baseline, on="edition", how="left")
    result["log_loss_delta"] = result["log_loss"] - result["baseline_log_loss"]
    result["accuracy_delta"] = result["accuracy"] - result["baseline_accuracy"]
    return result.sort_values(["edition", "log_loss"]).reset_index(drop=True)


def reliability_table(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    rows = []
    for class_id, probability_column in enumerate(("p_away", "p_draw", "p_home")):
        values = predictions[[probability_column, "target"]].copy()
        values["bin"] = pd.cut(
            values[probability_column],
            np.linspace(0, 1, bins + 1),
            include_lowest=True,
        )
        for interval, group in values.groupby("bin", observed=True):
            rows.append(
                {
                    "outcome": ("Away win", "Draw", "Home win")[class_id],
                    "bin": str(interval),
                    "predicted": group[probability_column].mean(),
                    "observed": (group["target"] == class_id).mean(),
                    "count": len(group),
                }
            )
    return pd.DataFrame(rows)


def confidence_reliability(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    probabilities = predictions[["p_away", "p_draw", "p_home"]].to_numpy()
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == predictions["target"].to_numpy()
    frame = pd.DataFrame({"confidence": confidence, "correct": correct})
    frame["bin"] = pd.cut(
        frame["confidence"], np.linspace(0, 1, bins + 1), include_lowest=True
    )
    return (
        frame.groupby("bin", observed=True)
        .agg(confidence=("confidence", "mean"), accuracy=("correct", "mean"), count=("correct", "size"))
        .reset_index()
        .assign(bin=lambda value: value["bin"].astype(str))
    )


def bootstrap_intervals(
    predictions: pd.DataFrame, draws: int = 2000, seed: int = 2026
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    samples = {"log_loss": [], "brier": [], "rps": [], "accuracy": [], "ece": []}
    for _ in range(draws):
        chunks = []
        for _, edition in predictions.groupby("edition"):
            indices = rng.integers(0, len(edition), len(edition))
            chunks.append(edition.iloc[indices])
        sample = pd.concat(chunks, ignore_index=True)
        y = sample["target"].astype(int).to_numpy()
        probabilities = sample[["p_away", "p_draw", "p_home"]].to_numpy()
        values = _metrics(y, probabilities)
        for name in samples:
            samples[name].append(values[name])
    output = {}
    for name, values in samples.items():
        low, median, high = np.quantile(values, [0.025, 0.5, 0.975])
        output[name] = {
            "low": float(low),
            "median": float(median),
            "high": float(high),
        }
    return output


def conformal_diagnostics(
    predictions: pd.DataFrame, alpha: float = 0.10, adaptation_rate: float = 0.025
) -> tuple[dict[str, float], pd.DataFrame]:
    """Adaptive conformal sets updated sequentially without future leakage."""
    rows = []
    editions = sorted(predictions["edition"].unique())
    calibration = predictions.loc[predictions["edition"] == editions[0]].sort_values("date")
    cal_probabilities = calibration[["p_away", "p_draw", "p_home"]].to_numpy()
    cal_y = calibration["target"].astype(int).to_numpy()
    scores = list(1.0 - cal_probabilities[np.arange(len(cal_y)), cal_y])
    adaptive_alpha = alpha
    evaluation = predictions.loc[predictions["edition"] > editions[0]].sort_values("date")
    for match in evaluation.itertuples():
        recent_scores = np.asarray(scores[-320:])
        level = min(
            1.0,
            np.ceil((len(recent_scores) + 1) * (1 - adaptive_alpha))
            / len(recent_scores),
        )
        threshold = float(np.quantile(recent_scores, level, method="higher"))
        probabilities = np.array([match.p_away, match.p_draw, match.p_home])
        prediction_set = (1.0 - probabilities) <= threshold
        target = int(match.target)
        covered = bool(prediction_set[target])
        rows.append(
            {
                "edition": int(match.edition),
                "home_team": match.home_team,
                "away_team": match.away_team,
                "set_size": int(prediction_set.sum()),
                "covered": covered,
                "threshold": threshold,
                "adaptive_alpha": adaptive_alpha,
            }
        )
        scores.append(1.0 - probabilities[target])
        adaptive_alpha = float(
            np.clip(
                adaptive_alpha + adaptation_rate * (alpha - float(not covered)),
                0.01,
                0.35,
            )
        )
    diagnostics = pd.DataFrame(rows)
    summary = {
        "target_coverage": 1.0 - alpha,
        "empirical_coverage": float(diagnostics["covered"].mean()),
        "average_set_size": float(diagnostics["set_size"].mean()),
        "single_outcome_rate": float((diagnostics["set_size"] == 1).mean()),
        "latest_threshold": float(diagnostics["threshold"].iloc[-1]),
        "final_adaptive_alpha": adaptive_alpha,
    }
    return summary, diagnostics


def feature_ablation(
    features: pd.DataFrame, edition: int = 2022
) -> pd.DataFrame:
    completed = features.dropna(subset=["target"]).copy()
    test = completed.loc[
        completed["tournament"].eq("FIFA World Cup")
        & completed["date"].dt.year.eq(edition)
    ]
    cutoff = test["date"].min()
    train = completed.loc[
        (completed["date"] < cutoff)
        & (completed["date"] >= cutoff - pd.DateOffset(years=28))
    ]
    configurations: dict[str, list[str]] = {"Full feature system": FEATURE_COLUMNS}
    for group, columns in FEATURE_GROUPS.items():
        configurations[f"Without {group}"] = [
            column for column in FEATURE_COLUMNS if column not in columns
        ]
    configurations["Ratings only"] = [
        "elo_diff",
        "elo_mean",
        "rating_confidence_diff",
        "rating_uncertainty",
        "home_advantage",
        "neutral",
    ]
    rows = []
    y_train = train["target"].astype(int)
    y_test = test["target"].astype(int).to_numpy()
    age_years = (train["date"].max() - train["date"]).dt.days.to_numpy() / 365.25
    weights = np.power(0.5, age_years / 8.0) * (0.65 + train["importance"].to_numpy())
    for name, columns in configurations.items():
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=220,
            max_leaf_nodes=22,
            l2_regularization=1.5,
            min_samples_leaf=28,
            random_state=edition,
        )
        estimator.fit(train[columns], y_train, sample_weight=weights)
        probabilities = estimator.predict_proba(test[columns])
        rows.append({"configuration": name, **_metrics(y_test, probabilities)})
    result = pd.DataFrame(rows).sort_values("log_loss")
    full_loss = float(
        result.loc[result["configuration"].eq("Full feature system"), "log_loss"].iloc[0]
    )
    result["log_loss_delta"] = result["log_loss"] - full_loss
    return result.reset_index(drop=True)


def drift_table(
    features: pd.DataFrame,
    reference_end: str = "2022-01-01",
    current_start: str = "2024-01-01",
) -> pd.DataFrame:
    reference = features.loc[
        features["is_completed"]
        & (features["date"] < pd.Timestamp(reference_end))
        & (features["date"] >= pd.Timestamp(reference_end) - pd.DateOffset(years=8))
    ]
    current = features.loc[
        features["is_completed"] & (features["date"] >= pd.Timestamp(current_start))
    ]
    rows = []
    for column in FEATURE_COLUMNS:
        edges = np.unique(reference[column].quantile(np.linspace(0, 1, 11)).to_numpy())
        if len(edges) < 3:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        expected = np.histogram(reference[column], bins=edges)[0].astype(float)
        actual = np.histogram(current[column], bins=edges)[0].astype(float)
        expected = np.clip(expected / expected.sum(), 1e-4, 1)
        actual = np.clip(actual / actual.sum(), 1e-4, 1)
        psi = float(np.sum((actual - expected) * np.log(actual / expected)))
        rows.append(
            {
                "feature": column,
                "psi": psi,
                "status": "material" if psi >= 0.25 else "watch" if psi >= 0.10 else "stable",
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def confusion_table(predictions: pd.DataFrame) -> list[list[int]]:
    probabilities = predictions[["p_away", "p_draw", "p_home"]].to_numpy()
    return confusion_matrix(
        predictions["target"],
        probabilities.argmax(axis=1),
        labels=[0, 1, 2],
    ).tolist()
