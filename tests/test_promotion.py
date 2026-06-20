from pathlib import Path

import pandas as pd

from worldcup_predictor.promotion import promotion_report


def test_promotion_gate_requires_quality_coverage_and_availability(
    tmp_path: Path,
):
    matrix = pd.DataFrame(
        [
            {
                "candidate": "valid",
                "folds": 5,
                "log_loss_delta": -0.01,
                "accuracy_delta": 0.01,
                "availability": "Pre-match derivable",
            },
            {
                "candidate": "oracle",
                "folds": 5,
                "log_loss_delta": -0.02,
                "accuracy_delta": 0.01,
                "availability": "Retrospective reanalysis",
            },
        ]
    )
    matrix_path = tmp_path / "matrix.csv"
    matrix.to_csv(matrix_path, index=False)
    policy_path = tmp_path / "policy.toml"
    policy_path.write_text(
        "\n".join(
            [
                "minimum_folds = 5",
                "minimum_log_loss_improvement = 0.001",
                "minimum_accuracy_delta = 0.0",
                'allowed_availability = ["Pre-match derivable"]',
                'blocked_availability = ["Retrospective reanalysis"]',
            ]
        ),
        encoding="utf-8",
    )
    report = promotion_report(matrix_path, policy_path)
    assert report[0]["promoted"] is True
    assert report[1]["promoted"] is False
