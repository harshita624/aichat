from __future__ import annotations

import argparse
import csv
import json

import numpy as np
import pandas as pd

from src.config import (
    DATA_GOLDEN,
    DATA_PROCESSED,
    SELECTED_BRAND_PATH,
    conversations_path_for,
    get_logger,
)
from src.escalation.policy import HIGH_RISK_KEYWORDS, ANGER_KEYWORDS

logger = get_logger(__name__)

TAXONOMY_PATH = DATA_PROCESSED / "taxonomy.json"
GOLDEN_TEMPLATE_PATH = DATA_GOLDEN / "golden_set_template.csv"
INTENT_MODEL_PATH = DATA_PROCESSED / "intent_model.joblib"

COLUMNS = [
    "id",
    "customer_message",
    "intent",
    "expected_action",
    "reference_resolution",
    "difficulty",
    "notes",
]


def _load_conversations() -> tuple[pd.DataFrame, str]:
    if not SELECTED_BRAND_PATH.exists():
        raise FileNotFoundError(
            "No selected brand found. Run the data pipeline first."
        )

    brand = json.loads(
        SELECTED_BRAND_PATH.read_text()
    )["brand"]

    conv_path = conversations_path_for(brand)

    if not conv_path.exists():
        raise FileNotFoundError(
            f"No conversations file at {conv_path}. "
            "Run src.data.conversations first."
        )

    df = (
        pd.read_csv(conv_path)
        .dropna(subset=["customer_message"])
        .reset_index(drop=True)
    )

    return df, brand


def _assign_clusters(
    df: pd.DataFrame,
) -> pd.Series:
    if not INTENT_MODEL_PATH.exists():
        logger.warning(
            "No trained intent model found yet — "
            "sampling without intent stratification."
        )
        return pd.Series(
            ["unclustered"] * len(df)
        )

    import joblib

    artifacts = joblib.load(
        INTENT_MODEL_PATH
    )

    vectorizer = artifacts["vectorizer"]
    kmeans = artifacts["kmeans"]

    X = vectorizer.transform(
        df["customer_message"].astype(str)
    )

    cluster_ids = kmeans.predict(X)

    if TAXONOMY_PATH.exists():
        taxonomy = json.loads(
            TAXONOMY_PATH.read_text()
        )

        cluster_to_name = {
            i["cluster_id"]: i["name"]
            for i in taxonomy["intents"]
        }

        return pd.Series(
            [
                cluster_to_name.get(
                    c,
                    f"cluster_{c}",
                )
                for c in cluster_ids
            ]
        )

    return pd.Series(
        [
            f"cluster_{c}"
            for c in cluster_ids
        ]
    )


def sample_candidates(
    df: pd.DataFrame,
    n_total: int,
    seed: int,
) -> pd.DataFrame:
    df = df.copy()

    df["_cluster"] = _assign_clusters(
        df
    )

    df["_len"] = (
        df["customer_message"]
        .astype(str)
        .str.len()
    )

    df["_high_risk"] = (
        df["customer_message"]
        .astype(str)
        .str.contains(
            HIGH_RISK_KEYWORDS
        )
    )

    df["_angry"] = (
        df["customer_message"]
        .astype(str)
        .str.contains(
            ANGER_KEYWORDS
        )
    )

    picked_indices: set[int] = set()

    clusters = df["_cluster"].unique().tolist()

    per_cluster_budget = max(
        2,
        int(
            n_total
            * 0.5
            / max(
                len(clusters),
                1,
            )
        ),
    )

    for c in clusters:
        pool = df[
            (df["_cluster"] == c)
            & (~df.index.isin(picked_indices))
        ]

        take = pool.sample(
            min(
                per_cluster_budget,
                len(pool),
            ),
            random_state=seed,
        )

        picked_indices.update(
            take.index.tolist()
        )

    risky_pool = df[
        (df["_high_risk"] | df["_angry"])
        & (~df.index.isin(picked_indices))
    ]

    take = risky_pool.sample(
        min(
            max(
                5,
                int(n_total * 0.1),
            ),
            len(risky_pool),
        ),
        random_state=seed,
    )

    picked_indices.update(
        take.index.tolist()
    )

    short_pool = df[
        (df["_len"] <= 20)
        & (~df.index.isin(picked_indices))
    ]

    take = short_pool.sample(
        min(
            max(
                3,
                int(n_total * 0.05),
            ),
            len(short_pool),
        ),
        random_state=seed,
    )

    picked_indices.update(
        take.index.tolist()
    )

    long_pool = df[
        (df["_len"] >= 200)
        & (~df.index.isin(picked_indices))
    ]

    take = long_pool.sample(
        min(
            max(
                3,
                int(n_total * 0.05),
            ),
            len(long_pool),
        ),
        random_state=seed,
    )

    picked_indices.update(
        take.index.tolist()
    )

    remaining = (
        n_total
        - len(picked_indices)
    )

    if remaining > 0:
        pool = df[
            ~df.index.isin(picked_indices)
        ]

        take = pool.sample(
            min(
                remaining,
                len(pool),
            ),
            random_state=seed,
        )

        picked_indices.update(
            take.index.tolist()
        )

    result = df.loc[
        sorted(picked_indices)
    ].sample(
        frac=1,
        random_state=seed,
    )

    return result.head(
        n_total
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__
    )

    ap.add_argument(
        "--n",
        type=int,
        default=200,
        help=(
            "Target number of candidate examples "
            "(150-250 recommended)."
        ),
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = ap.parse_args()

    df, brand = _load_conversations()

    logger.info(
        f"Sampling {args.n} candidate examples "
        f"from {len(df)} real {brand} conversations..."
    )

    candidates = sample_candidates(
        df,
        args.n,
        args.seed,
    )

    rows = []

    for i, (_, row) in enumerate(
        candidates.iterrows(),
        start=1,
    ):
        rows.append(
            {
                "id": f"G{i:03d}",
                "customer_message": row[
                    "customer_message"
                ],
                "intent": "",
                "expected_action": "",
                "reference_resolution": "",
                "difficulty": "",
                "notes": (
                    f"source_conversation_id="
                    f"{row.get('conversation_id', '')}"
                ),
            }
        )

    GOLDEN_TEMPLATE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        GOLDEN_TEMPLATE_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=COLUMNS,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\nWrote {len(rows)} REAL candidate messages "
        f"to {GOLDEN_TEMPLATE_PATH}"
    )

    print(
        "\nintent / expected_action / "
        "reference_resolution / difficulty are "
        "intentionally left BLANK — that's the "
        "annotation work only you can do. See "
        "data/golden/annotation_guidelines.md for "
        "how to fill each column in. 'notes' already "
        "has the source conversation_id in case you "
        "want the full original thread while annotating.\n"
    )

    print(
        "Once every row has a non-empty 'intent', run:\n"
        "    python -m src.evaluation.evaluate\n"
    )


if __name__ == "__main__":
    main()