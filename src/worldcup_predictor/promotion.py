from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import pandas as pd

from .config import ADVANCED_MATRIX_PATH, ROOT


def evaluate_candidate(
    row: pd.Series,
    policy: dict[str, object],
) -> dict[str, object]:
    reasons = []
    if int(row["folds"]) < int(policy["minimum_folds"]):
        reasons.append("insufficient chronological folds")
    if float(row["log_loss_delta"]) > -float(
        policy["minimum_log_loss_improvement"]
    ):
        reasons.append("log-loss improvement below threshold")
    if float(row["accuracy_delta"]) < float(policy["minimum_accuracy_delta"]):
        reasons.append("accuracy regression")
    if str(row["availability"]) in policy["blocked_availability"]:
        reasons.append("information unavailable at normal pre-match inference")
    return {
        "candidate": row["candidate"],
        "promoted": not reasons,
        "reasons": reasons,
    }


def promotion_report(
    matrix_path: Path = ADVANCED_MATRIX_PATH,
    policy_path: Path = ROOT / "configs" / "promotion.toml",
) -> list[dict[str, object]]:
    matrix = pd.read_csv(matrix_path)
    policy = tomllib.loads(policy_path.read_text(encoding="utf-8"))
    candidates = matrix.loc[
        ~matrix["candidate"].isin(
            ["Baseline scores + state", "Calibrated hybrid"]
        )
    ]
    return [
        evaluate_candidate(row, policy)
        for _, row in candidates.iterrows()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the model promotion gate.")
    parser.add_argument("--matrix", type=Path, default=ADVANCED_MATRIX_PATH)
    parser.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "configs" / "promotion.toml",
    )
    args = parser.parse_args()
    report = promotion_report(args.matrix, args.policy)
    print(json.dumps(report, indent=2))
    if any(item["promoted"] for item in report):
        raise SystemExit(10)


if __name__ == "__main__":
    main()
