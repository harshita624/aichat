from __future__ import annotations

import json

import pandas as pd

from src.config import (
    DATA_RAW,
    INSPECTION_SUMMARY_PATH,
    BRAND_FREQUENCIES_PATH,
    SELECTED_BRAND_PATH,
    get_logger,
)
from src.data.inspect import main as run_inspect


logger = get_logger(__name__)

MIN_RESOLVABLE_PAIRS = 300

MAX_RESOLVABLE_PAIRS = 40000


def _load_inspection() -> dict:
    if not INSPECTION_SUMMARY_PATH.exists():
        logger.info(
            "No inspection summary found yet — running inspection first."
        )
        run_inspect()

    if not INSPECTION_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Dataset inspection could not be completed. Make sure "
            "data/raw/ contains the downloaded CSV, then re-run "
            "python -m src.data.inspect."
        )

    return json.loads(
        INSPECTION_SUMMARY_PATH.read_text()
    )


def _is_numeric_id(value) -> bool:
    try:
        int(str(value).strip())
        return True
    except (ValueError, TypeError):
        return False


def _normalize_tweet_ids(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    return numeric.dropna().astype("int64").astype(str)


def compute_brand_candidates() -> pd.DataFrame:
    inspection = _load_inspection()

    main_info = inspection["main_dataset"]
    cols = main_info["detected_columns"]

    required = [
        "id_column",
        "text_column",
        "author_column",
        "inbound_column",
        "in_response_to_column",
    ]

    missing = [
        r for r in required
        if not cols.get(r)
    ]

    if missing:
        raise ValueError(
            "Cannot run brand exploration — required columns "
            f"not auto-detected: {missing}. Check "
            "data/processed/dataset_inspection.json manually."
        )

    file_path = DATA_RAW / main_info["file"]

    logger.info(
        f"Loading full dataset from {file_path} "
        "(this may take a minute)..."
    )

    id_c = cols["id_column"]
    text_c = cols["text_column"]
    author_c = cols["author_column"]
    inbound_c = cols["inbound_column"]
    reply_c = cols["in_response_to_column"]

    usecols = [
        id_c,
        text_c,
        author_c,
        inbound_c,
        reply_c,
    ]

    df = pd.read_csv(
        file_path,
        usecols=usecols,
        low_memory=False,
    )

    df["_inbound_bool"] = (
        df[inbound_c]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
            ]
        )
    )

    inbound_df = df[
        df["_inbound_bool"]
    ].copy()

    outbound_df = df[
        ~df["_inbound_bool"]
    ].copy()

    inbound_df["_normalized_tweet_id"] = (
        _normalize_tweet_ids(
            inbound_df[id_c]
        )
    )

    outbound_df["_normalized_reply_id"] = (
        _normalize_tweet_ids(
            outbound_df[reply_c]
        )
    )

    inbound_id_set = set(
        inbound_df[
            "_normalized_tweet_id"
        ]
    )

    outbound_df["_is_brand_like"] = (
        ~outbound_df[author_c].apply(
            _is_numeric_id
        )
    )

    brand_rows = outbound_df[
        outbound_df["_is_brand_like"]
    ]

    records = []

    for brand, group in brand_rows.groupby(
        author_c
    ):

        total_outbound = len(group)

        replied_ids = group[
            "_normalized_reply_id"
        ]

        resolvable_mask = (
            replied_ids.isin(
                inbound_id_set
            )
        )

        resolvable_pairs = int(
            resolvable_mask.sum()
        )

        resolvable_reply_ids = set(
            replied_ids[
                resolvable_mask
            ]
        )

        distinct_customers = inbound_df[
            inbound_df[
                "_normalized_tweet_id"
            ].isin(
                resolvable_reply_ids
            )
        ][author_c].nunique()

        records.append(
            {
                "brand": brand,
                "total_outbound_tweets": total_outbound,
                "resolvable_pairs": resolvable_pairs,
                "distinct_customers_helped": int(
                    distinct_customers
                ),
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "brand",
                "total_outbound_tweets",
                "resolvable_pairs",
                "distinct_customers_helped",
            ]
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            "resolvable_pairs",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def select_brand(
    candidates: pd.DataFrame,
) -> dict:
    eligible = candidates[
        (
            candidates[
                "resolvable_pairs"
            ]
            >= MIN_RESOLVABLE_PAIRS
        )
        &
        (
            candidates[
                "resolvable_pairs"
            ]
            <= MAX_RESOLVABLE_PAIRS
        )
    ].copy()

    if eligible.empty:
        raise ValueError(
            f"No brand met the eligibility window "
            f"({MIN_RESOLVABLE_PAIRS}-"
            f"{MAX_RESOLVABLE_PAIRS} resolvable pairs). "
            "Check data/processed/brand_frequencies.csv "
            "and adjust the thresholds in "
            "src/data/explore.py if needed."
        )

    eligible = eligible.sort_values(
        [
            "resolvable_pairs",
            "distinct_customers_helped",
        ],
        ascending=False,
    )

    top = eligible.iloc[0]

    why = (
        f"Selected from {len(candidates)} candidate "
        "brand accounts. "
        f"{top['brand']} had "
        f"{int(top['resolvable_pairs'])} "
        "resolvable customer-agent pairs "
        f"(within the {MIN_RESOLVABLE_PAIRS}-"
        f"{MAX_RESOLVABLE_PAIRS} eligibility "
        "window used to balance data sufficiency "
        "against local-dev practicality) and "
        f"{int(top['distinct_customers_helped'])} "
        "distinct customers helped — the highest "
        "among eligible candidates, used as a proxy "
        "for issue diversity."
    )

    return {
        "brand": top["brand"],
        "why": why,
        "num_conversations": int(
            top["resolvable_pairs"]
        ),
        "num_customer_messages": int(
            top["resolvable_pairs"]
        ),
        "num_agent_responses": int(
            top["resolvable_pairs"]
        ),
        "total_outbound_tweets_by_brand": int(
            top["total_outbound_tweets"]
        ),
    }


def main() -> None:
    candidates = compute_brand_candidates()

    if candidates.empty:
        print(
            "No brand-like candidate accounts found. "
            "This likely means column detection picked "
            "the wrong author/inbound columns — check "
            "data/processed/dataset_inspection.json."
        )
        return

    candidates.to_csv(
        BRAND_FREQUENCIES_PATH,
        index=False,
    )

    print(
        f"\nBrand candidate frequencies saved to "
        f"{BRAND_FREQUENCIES_PATH}"
    )

    print(
        "\nTop 15 candidates by resolvable pairs:"
    )

    print(
        candidates.head(15).to_string(
            index=False
        )
    )

    selection = select_brand(
        candidates
    )

    SELECTED_BRAND_PATH.write_text(
        json.dumps(
            selection,
            indent=2,
        )
    )

    print(
        "\n" + "=" * 60
    )

    print(
        f"Selected brand: "
        f"{selection['brand']}"
    )

    print(
        f"Why: {selection['why']}"
    )

    print(
        f"Number of conversations: "
        f"{selection['num_conversations']}"
    )

    print(
        f"Number of customer messages: "
        f"{selection['num_customer_messages']}"
    )

    print(
        f"Number of agent responses: "
        f"{selection['num_agent_responses']}"
    )

    print(
        "=" * 60
    )

    print(
        f"\nSaved selection to "
        f"{SELECTED_BRAND_PATH}"
    )


if __name__ == "__main__":
    main()