from __future__ import annotations

import json

import pandas as pd

from src.config import (
    DATA_SAMPLE,
    SELECTED_BRAND_PATH,
    conversations_path_for,
    get_logger,
)


logger = get_logger(__name__)

SAMPLE_SIZE = 300


def main() -> None:
    if not SELECTED_BRAND_PATH.exists():
        print(
            "No selected brand found. "
            "Run: python -m src.data.explore first."
        )
        return

    brand = json.loads(
        SELECTED_BRAND_PATH.read_text()
    )["brand"]

    conv_path = conversations_path_for(
        brand
    )

    if not conv_path.exists():
        print(
            f"No conversations file found at {conv_path}. "
            "Run: python -m src.data.conversations first."
        )
        return

    df = pd.read_csv(
        conv_path
    )

    n = min(
        SAMPLE_SIZE,
        len(df),
    )

    sample = df.sample(
        n=n,
        random_state=42,
    )

    out_path = (
        DATA_SAMPLE
        / f"sample_{brand.lower()}.csv"
    )

    sample.to_csv(
        out_path,
        index=False,
    )

    print(
        f"Saved {n}-row development sample to {out_path}"
    )

    print(
        "Reminder: this sample is for local iteration only — "
        "never used in evaluation."
    )


if __name__ == "__main__":
    main()