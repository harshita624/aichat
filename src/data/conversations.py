from __future__ import annotations

import argparse
import json

import pandas as pd

from src.config import (
    CLEANED_DATA_PATH,
    INSPECTION_SUMMARY_PATH,
    SELECTED_BRAND_PATH,
    conversations_path_for,
    get_logger,
)

logger = get_logger(__name__)

MAX_CONTEXT_DEPTH = 3

DIAGNOSTIC_FAILURE_THRESHOLD = 0.5
N_DIAGNOSTIC_SAMPLES = 5


def _resolve_brand(explicit_brand: str | None) -> str:
    if explicit_brand:
        return explicit_brand
    if not SELECTED_BRAND_PATH.exists():
        raise FileNotFoundError(
            "No brand specified and no data/processed/selected_brand.json found. "
            "Run python -m src.data.explore first, or pass --brand explicitly."
        )
    return json.loads(SELECTED_BRAND_PATH.read_text())["brand"]


def _normalize_id_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.astype("Int64").astype(str).replace("<NA>", "")


def deduplicate_conversations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    before = len(df)
    dupe_mask = df.duplicated(
        subset=["customer_message", "agent_response"],
        keep="first",
    )
    n_dupes = int(dupe_mask.sum())
    df = df[~dupe_mask]
    logger.info(
        f"Conversation-level dedup: {before} -> {len(df)} rows "
        f"({n_dupes} exact duplicate (customer_message, agent_response) pairs removed)."
    )
    return df


def _print_link_failure_diagnostics(
    brand_outbound: pd.DataFrame,
    by_id: pd.DataFrame,
    brand: str,
) -> None:
    failing = brand_outbound[
        (brand_outbound["_reply_str"] != "")
        & (~brand_outbound["_reply_str"].isin(by_id.index))
    ]
    n_failing = len(failing)
    n_total = len(brand_outbound)

    if n_total == 0 or n_failing / n_total < DIAGNOSTIC_FAILURE_THRESHOLD:
        return

    logger.warning(
        f"{n_failing}/{n_total} ({n_failing / n_total * 100:.1f}%) of {brand}'s "
        f"outbound tweets had a non-empty in_response_to id that could not be "
        f"found in the cleaned corpus. This is high enough to indicate an "
        f"upstream data problem rather than normal noise. Sample failures:"
    )

    for _, row in failing.head(N_DIAGNOSTIC_SAMPLES).iterrows():
        logger.warning(
            f"  agent tweet id={row['_id_str']!r} points to reply_to={row['_reply_str']!r}, "
            f"which is not present in the cleaned corpus's id index at all."
        )

    logger.warning(
        "If this persists after re-running the full pipeline from "
        "src.data.clean onward, compare a few of the reply_to ids above "
        "against data/raw/ directly to confirm they exist there — that "
        "would point to a NEW issue distinct from the dedup-before-linking "
        "bug this diagnostic was added to catch."
    )


def build_conversations(
    df: pd.DataFrame,
    cols: dict,
    brand: str,
) -> pd.DataFrame:
    id_c, author_c, inbound_c, reply_c, ts_c = (
        cols["id_column"],
        cols["author_column"],
        cols["inbound_column"],
        cols["in_response_to_column"],
        cols.get("timestamp_column"),
    )

    df = df.copy()
    df["_inbound_bool"] = (
        df[inbound_c]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )
    df["_id_str"] = _normalize_id_series(df[id_c])
    df["_reply_str"] = _normalize_id_series(df[reply_c])

    df = df[df["_id_str"] != ""]

    by_id = df.set_index("_id_str", drop=False)

    brand_outbound = df[
        (~df["_inbound_bool"])
        & (df[author_c] == brand)
    ]

    _print_link_failure_diagnostics(
        brand_outbound,
        by_id,
        brand,
    )

    records = []
    skipped_no_link = 0
    skipped_link_not_inbound = 0
    skipped_empty_text = 0

    for _, agent_row in brand_outbound.iterrows():
        reply_to_id = agent_row["_reply_str"]

        if reply_to_id == "" or reply_to_id not in by_id.index:
            skipped_no_link += 1
            continue

        customer_row = by_id.loc[reply_to_id]

        if isinstance(customer_row, pd.DataFrame):
            customer_row = customer_row.iloc[0]

        if not bool(customer_row["_inbound_bool"]):
            skipped_link_not_inbound += 1
            continue

        agent_text = agent_row.get("clean_text", "")
        customer_text = customer_row.get("clean_text", "")

        if not agent_text or not customer_text:
            skipped_empty_text += 1
            continue

        context_turns = [customer_text]
        cursor = customer_row
        depth = 0

        while depth < MAX_CONTEXT_DEPTH:
            parent_id = cursor.get("_reply_str", "")

            if parent_id == "" or parent_id not in by_id.index:
                break

            parent_row = by_id.loc[parent_id]

            if isinstance(parent_row, pd.DataFrame):
                parent_row = parent_row.iloc[0]

            parent_text = parent_row.get("clean_text", "")

            if parent_text:
                context_turns.insert(0, parent_text)

            cursor = parent_row
            depth += 1

        records.append({
            "conversation_id": f"{brand}_{agent_row['_id_str']}",
            "brand": brand,
            "customer_message": customer_text,
            "agent_response": agent_text,
            "full_context": "\n".join(context_turns),
            "timestamp": (
                agent_row.get(ts_c, None)
                if ts_c
                else None
            ),
        })

    logger.info(
        f"Built {len(records)} conversation pairs for {brand}. Skipped: "
        f"{skipped_no_link} (no valid reply link), {skipped_link_not_inbound} "
        f"(linked tweet wasn't inbound), {skipped_empty_text} (empty text)."
    )

    result_df = pd.DataFrame(records)
    result_df = deduplicate_conversations(result_df)

    return result_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--brand",
        default=None,
        help="Override the brand from selected_brand.json",
    )
    args = parser.parse_args()

    if not CLEANED_DATA_PATH.exists():
        print(
            "No cleaned data found. Run: "
            "python -m src.data.clean first."
        )
        return

    if not INSPECTION_SUMMARY_PATH.exists():
        print(
            "No inspection summary found. Run: "
            "python -m src.data.inspect first."
        )
        return

    brand = _resolve_brand(args.brand)

    cols = json.loads(
        INSPECTION_SUMMARY_PATH.read_text()
    )["main_dataset"]["detected_columns"]

    logger.info(
        f"Loading cleaned data to build conversations for brand: {brand}"
    )

    df = pd.read_csv(
        CLEANED_DATA_PATH,
        low_memory=False,
    )

    conversations = build_conversations(
        df,
        cols,
        brand,
    )

    if conversations.empty:
        print(
            f"No conversation pairs could be reconstructed for {brand}. "
            f"Check data/processed/brand_frequencies.csv and consider "
            f"re-running python -m src.data.explore."
        )
        return

    out_path = conversations_path_for(brand)

    conversations.to_csv(
        out_path,
        index=False,
    )

    print(
        f"Saved {len(conversations)} conversation pairs to {out_path}"
    )


if __name__ == "__main__":
    main()