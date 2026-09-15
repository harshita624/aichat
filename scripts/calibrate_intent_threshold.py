from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import (
    DATA_PROCESSED,
    SELECTED_BRAND_PATH,
    conversations_path_for,
    get_logger,
)
from src.intents.classifier import IntentClassifier

logger = get_logger(__name__)

TAXONOMY_PATH = DATA_PROCESSED / "taxonomy.json"

CANDIDATE_THRESHOLDS = [
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.12,
    0.15,
    0.20,
    0.25,
    0.30,
]


def _print_distribution(
    label: str,
    arr: np.ndarray,
) -> None:
    print(
        f"\n{'=' * 60}\n"
        f"{label} ({len(arr)} messages)\n"
        f"{'=' * 60}"
    )

    if len(arr) == 0:
        print(
            "  (no messages in this subset)"
        )
        return

    for p in [
        0,
        10,
        25,
        50,
        75,
        90,
        95,
        99,
        100,
    ]:
        print(
            f"  p{p:<3d}: "
            f"{np.percentile(arr, p):.4f}"
        )

    print(
        f"  mean: {arr.mean():.4f}   "
        f"std: {arr.std():.4f}"
    )


def main() -> None:
    if not SELECTED_BRAND_PATH.exists():
        print(
            "No selected brand found. "
            "Run the data pipeline first."
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
            f"No conversations file at {conv_path}."
        )
        return

    if not TAXONOMY_PATH.exists():
        print(
            "No taxonomy.json found. "
            "Run python -m src.intents.taxonomy first."
        )
        return

    taxonomy = json.loads(
        TAXONOMY_PATH.read_text(
            encoding="utf-8"
        )
    )

    eligible_intents = {
        i["name"]
        for i in taxonomy["intents"]
        if i["auto_handle_eligible"]
    }

    print(
        f"Currently auto_handle_eligible intents "
        f"({len(eligible_intents)}): "
        f"{sorted(eligible_intents)}"
    )

    df = (
        pd.read_csv(conv_path)
        .dropna(subset=["customer_message"])
    )

    texts = (
        df["customer_message"]
        .astype(str)
        .tolist()
    )

    print(
        f"\nScoring {len(texts)} real customer messages "
        f"from {brand}'s conversation corpus..."
    )

    clf = IntentClassifier()

    margins = []
    intents = []

    for t in texts:
        result = clf.predict(t)
        margins.append(
            result["confidence"]
        )
        intents.append(
            result["intent"]
        )

    margin_arr = np.array(
        margins
    )

    intent_arr = np.array(
        intents
    )

    eligible_mask = np.isin(
        intent_arr,
        list(eligible_intents),
    )

    _print_distribution(
        "ALL MESSAGES "
        "(includes permanently-ineligible clusters)",
        margin_arr,
    )

    _print_distribution(
        "ELIGIBLE-ONLY MESSAGES "
        "(the population MIN_INTENT_CONFIDENCE actually gates)",
        margin_arr[eligible_mask],
    )

    print(
        f"\n{'=' * 60}\n"
        f"OF THE {eligible_mask.sum():,} "
        f"ELIGIBLE-CLUSTER MESSAGES, "
        f"HOW MANY CLEAR EACH THRESHOLD\n"
        f"{'=' * 60}"
    )

    eligible_margins = margin_arr[
        eligible_mask
    ]

    for t in CANDIDATE_THRESHOLDS:
        n_pass = int(
            (eligible_margins >= t).sum()
        )

        pct_of_eligible = (
            n_pass
            / max(
                len(eligible_margins),
                1,
            )
            * 100
        )

        pct_of_all = (
            n_pass
            / len(margin_arr)
            * 100
        )

        print(
            f"  threshold {t:.2f}: "
            f"{n_pass:>5,} / "
            f"{len(eligible_margins):,} "
            f"eligible messages pass "
            f"({pct_of_eligible:5.1f}% of eligible, "
            f"{pct_of_all:4.1f}% of ALL traffic)"
        )

    print(
        f"\n{'=' * 60}\n"
        "PER-ELIGIBLE-INTENT MARGIN PERCENTILES\n"
        f"{'=' * 60}"
    )

    for name in sorted(
        eligible_intents
    ):
        vals = margin_arr[
            intent_arr == name
        ]

        if len(vals) == 0:
            continue

        print(
            f"\n{name} (n={len(vals)}):"
        )

        for p in [
            25,
            50,
            75,
            90,
        ]:
            print(
                f"    p{p}: "
                f"{np.percentile(vals, p):.4f}"
            )

    print(
        "\nPick MIN_INTENT_CONFIDENCE from the "
        "ELIGIBLE-ONLY table above, not the "
        "ALL-MESSAGES one -- ineligible-cluster "
        "messages escalate unconditionally already, "
        "so including them just understates how much "
        "of your genuinely well-clustered traffic "
        "could actually auto-handle. Validate the "
        "final choice once src.evaluation.evaluate "
        "can run against your annotated golden set."
    )


if __name__ == "__main__":
    main()