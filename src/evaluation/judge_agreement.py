from __future__ import annotations

import json

import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from src.config import (
    DATA_GOLDEN,
    RESULTS_DIR,
    get_logger,
)


logger = get_logger(__name__)

JUDGE_RESULTS_PATH = RESULTS_DIR / "judge_results.csv"
HUMAN_RATINGS_PATH = DATA_GOLDEN / "human_ratings.csv"
AGREEMENT_OUTPUT_PATH = RESULTS_DIR / "judge_human_agreement.json"

NUMERIC_FIELDS = [
    "correctness",
    "relevance",
    "grounding",
    "completeness",
    "helpfulness",
]


def main() -> None:
    if not JUDGE_RESULTS_PATH.exists():
        status = {
            "status": "PENDING",
            "reason": "No judge results found.",
            "how_to_generate": (
                "Run python -m src.evaluation.judge first."
            ),
        }

        print(
            json.dumps(
                status,
                indent=2,
            )
        )

        AGREEMENT_OUTPUT_PATH.write_text(
            json.dumps(
                status,
                indent=2,
            )
        )

        return

    if not HUMAN_RATINGS_PATH.exists():
        status = {
            "human_agreement": "PENDING",
            "reason": "No human ratings file found.",
            "how_to_generate": (
                f"Create data/golden/human_ratings.csv "
                f"with columns: "
                f"id,{','.join(NUMERIC_FIELDS)} — "
                f"score the same ids in judge_results.csv, "
                f"then re-run this script."
            ),
        }

        print(
            json.dumps(
                status,
                indent=2,
            )
        )

        AGREEMENT_OUTPUT_PATH.write_text(
            json.dumps(
                status,
                indent=2,
            )
        )

        return

    judge_df = pd.read_csv(
        JUDGE_RESULTS_PATH
    )

    human_df = pd.read_csv(
        HUMAN_RATINGS_PATH
    )

    merged = judge_df.merge(
        human_df,
        on="id",
        suffixes=(
            "_judge",
            "_human",
        ),
    )

    if len(merged) == 0:
        print(
            "No matching ids between "
            "judge_results.csv and human_ratings.csv."
        )
        return

    results = {}

    for field in NUMERIC_FIELDS:
        j_col = f"{field}_judge"
        h_col = f"{field}_human"

        if (
            j_col not in merged.columns
            or h_col not in merged.columns
        ):
            continue

        pair = merged[
            [j_col, h_col]
        ].dropna()

        if len(pair) < 2:
            results[field] = {
                "status": "INSUFFICIENT_DATA",
                "n": len(pair),
            }
            continue

        corr, p = spearmanr(
            pair[j_col],
            pair[h_col],
        )

        kappa = cohen_kappa_score(
            pair[j_col].round().astype(int),
            pair[h_col].round().astype(int),
        )

        results[field] = {
            "n_examples": len(pair),
            "spearman_correlation": round(
                float(corr),
                4,
            ),
            "spearman_p_value": round(
                float(p),
                4,
            ),
            "cohens_kappa": round(
                float(kappa),
                4,
            ),
        }

    output = {
        "status": "COMPLETE",
        "n_matched_examples": len(merged),
        "per_metric": results,
    }

    AGREEMENT_OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print(
        json.dumps(
            output,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()