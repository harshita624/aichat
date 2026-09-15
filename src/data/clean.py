from __future__ import annotations

import html
import json
import re

import pandas as pd

from src.config import (
    DATA_RAW,
    INSPECTION_SUMMARY_PATH,
    CLEANED_DATA_PATH,
    get_logger,
)

logger = get_logger(__name__)

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
WHITESPACE_PATTERN = re.compile(r"\s+")

MIN_TEXT_LENGTH = 3


def clean_text(raw: str) -> str:
    if not isinstance(raw, str):
        return ""

    text = html.unescape(raw)
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text


def clean_dataframe(
    df: pd.DataFrame,
    text_column: str,
) -> pd.DataFrame:
    before = len(df)

    df = df.copy()
    df["clean_text"] = df[text_column].apply(clean_text)
    df = df[
        df["clean_text"].str.len() >= MIN_TEXT_LENGTH
    ]

    after = len(df)

    logger.info(
        f"Cleaning: {before} -> {after} rows "
        f"(removed {before - after} "
        f"empty/too-short messages after cleaning; "
        f"NO duplicate-text removal happens at this stage — "
        f"see deduplicate_conversations() in "
        f"src/data/conversations.py for where that now runs)."
    )

    return df


def main() -> None:
    if not INSPECTION_SUMMARY_PATH.exists():
        print(
            "No inspection summary found. "
            "Run: python -m src.data.inspect first."
        )
        return

    inspection = json.loads(
        INSPECTION_SUMMARY_PATH.read_text()
    )

    main_info = inspection["main_dataset"]
    cols = main_info["detected_columns"]
    text_col = cols.get("text_column")

    if not text_col:
        print(
            "No text column detected — cannot clean. "
            "Check data/processed/dataset_inspection.json."
        )
        return

    file_path = DATA_RAW / main_info["file"]

    logger.info(
        f"Loading {file_path} for cleaning "
        "(this may take a minute)..."
    )

    df = pd.read_csv(
        file_path,
        low_memory=False,
    )

    cleaned = clean_dataframe(
        df,
        text_col,
    )

    cleaned.to_csv(
        CLEANED_DATA_PATH,
        index=False,
    )

    print(
        f"Saved {len(cleaned)} cleaned rows to "
        f"{CLEANED_DATA_PATH}"
    )


if __name__ == "__main__":
    main()