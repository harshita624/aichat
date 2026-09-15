from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    DATA_GOLDEN,
    DATA_PROCESSED,
    SELECTED_BRAND_PATH,
    conversations_path_for,
    get_logger,
)


logger = get_logger(__name__)

NEAR_DUP_THRESHOLD = 0.92
NEAR_DUP_MAX_ROWS = 5000


@dataclass
class LeakageReport:
    exact_duplicates: int
    near_duplicates_sampled: int
    near_duplicate_sample_size: int
    train_test_text_overlap: int
    golden_retrieval_overlap: int
    conversation_level_overlap: int
    temporal_leakage_detected: bool
    temporal_check_status: str
    notes: list[str]


def check_exact_duplicates(
    df: pd.DataFrame,
    text_col: str = "customer_message",
) -> int:
    return int(
        df.duplicated(
            subset=[text_col]
        ).sum()
    )


def check_near_duplicates(
    df: pd.DataFrame,
    text_col: str = "customer_message",
    threshold: float = NEAR_DUP_THRESHOLD,
) -> tuple[int, int]:
    sample = (
        df[text_col]
        .dropna()
        .astype(str)
    )

    sample_size = min(
        len(sample),
        NEAR_DUP_MAX_ROWS,
    )

    sample = (
        sample.sample(
            n=sample_size,
            random_state=42,
        )
        if len(sample) > sample_size
        else sample
    )

    if len(sample) < 2:
        return 0, len(sample)

    vec = TfidfVectorizer(
        min_df=1
    ).fit_transform(sample)

    sims = cosine_similarity(vec)

    n = sims.shape[0]
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= threshold:
                count += 1

    return count, len(sample)


def check_split_overlap(
    train_texts: pd.Series,
    test_texts: pd.Series,
) -> int:
    return len(
        set(
            train_texts
            .dropna()
            .astype(str)
        )
        &
        set(
            test_texts
            .dropna()
            .astype(str)
        )
    )


def check_golden_retrieval_overlap(
    golden_df: pd.DataFrame,
    retrieval_df: pd.DataFrame,
    golden_text_col: str = "customer_message",
    retrieval_text_col: str = "customer_message",
) -> int:
    return len(
        set(
            golden_df[golden_text_col]
            .dropna()
            .astype(str)
        )
        &
        set(
            retrieval_df[retrieval_text_col]
            .dropna()
            .astype(str)
        )
    )


def check_conversation_level_overlap(
    train_ids: set[str],
    test_ids: set[str],
) -> int:
    return len(
        train_ids & test_ids
    )


def check_temporal_leakage(
    df: pd.DataFrame,
    timestamp_col: str | None,
    train_idx,
    test_idx,
) -> tuple[bool, str]:
    if (
        timestamp_col is None
        or timestamp_col not in df.columns
    ):
        return (
            False,
            "SKIPPED: no reliable timestamp column available.",
        )

    try:
        ts = pd.to_datetime(
            df[timestamp_col],
            errors="coerce",
        )
    except Exception:
        return (
            False,
            "SKIPPED: timestamp column could not be parsed.",
        )

    if ts.isna().mean() > 0.5:
        return (
            False,
            "SKIPPED: more than half of timestamps failed to parse — unreliable.",
        )

    train_max = ts.loc[train_idx].max()
    test_min = ts.loc[test_idx].min()

    if pd.isna(train_max) or pd.isna(test_min):
        return (
            False,
            "SKIPPED: insufficient valid timestamps in split.",
        )

    leak = test_min < train_max

    status = (
        f"CHECKED: train max={train_max}, test min={test_min}. "
        f"{'LEAK DETECTED.' if leak else 'No temporal leak detected.'}"
    )

    return bool(leak), status


def exclude_self_from_results(
    results: list[dict],
    exclude_conversation_id: str | None = None,
    exclude_message_text: str | None = None,
) -> list[dict]:
    if (
        not exclude_conversation_id
        and not exclude_message_text
    ):
        return results

    filtered = []

    for r in results:
        if (
            exclude_conversation_id
            and r.get("conversation_id")
            == exclude_conversation_id
        ):
            continue

        if (
            exclude_message_text
            and r.get("historical_customer_message")
            == exclude_message_text
        ):
            continue

        filtered.append(r)

    return filtered


def build_report() -> LeakageReport:
    notes = []

    if not SELECTED_BRAND_PATH.exists():
        raise FileNotFoundError(
            "No selected brand found. Run the data pipeline first."
        )

    brand = json.loads(
        SELECTED_BRAND_PATH.read_text()
    )["brand"]

    conv_path = conversations_path_for(
        brand
    )

    if not conv_path.exists():
        raise FileNotFoundError(
            f"No conversations file at {conv_path}. "
            "Run src.data.conversations first."
        )

    df = pd.read_csv(conv_path)

    exact_dupes = check_exact_duplicates(
        df
    )

    near_dupes, sample_size = check_near_duplicates(
        df
    )

    train_df = df.sample(
        frac=0.8,
        random_state=42,
    )

    test_df = df.drop(
        train_df.index
    )

    split_overlap = check_split_overlap(
        train_df["customer_message"],
        test_df["customer_message"],
    )

    conv_overlap = check_conversation_level_overlap(
        set(train_df["conversation_id"]),
        set(test_df["conversation_id"]),
    )

    ts_col = (
        "timestamp"
        if "timestamp" in df.columns
        else None
    )

    temporal_leak, temporal_status = check_temporal_leakage(
        df,
        ts_col,
        train_df.index,
        test_df.index,
    )

    golden_path = (
        DATA_GOLDEN
        / "golden_set_template.csv"
    )

    golden_overlap = -1

    if golden_path.exists():
        golden_df = pd.read_csv(
            golden_path
        )

        annotated = golden_df[
            golden_df.get(
                "intent",
                pd.Series(dtype=str),
            ).notna()
        ]

        if len(annotated) > 0:
            golden_overlap = check_golden_retrieval_overlap(
                annotated,
                df,
            )
        else:
            notes.append(
                "Golden set exists but is not yet annotated — "
                "overlap check skipped."
            )
    else:
        notes.append(
            "Golden set template not found — "
            "overlap check skipped."
        )

    if exact_dupes > 0:
        notes.append(
            f"{exact_dupes} exact duplicate customer_message rows found."
        )

    if split_overlap > 0:
        notes.append(
            f"{split_overlap} text overlaps in an 80/20 random split — "
            "supports conversation-level splitting over random row splitting."
        )

    return LeakageReport(
        exact_dupes,
        near_dupes,
        sample_size,
        split_overlap,
        max(golden_overlap, 0),
        conv_overlap,
        temporal_leak,
        temporal_status,
        notes,
    )


def main() -> None:
    report = build_report()

    print(
        json.dumps(
            asdict(report),
            indent=2,
        )
    )

    out_path = (
        DATA_PROCESSED
        / "leakage_report.json"
    )

    out_path.write_text(
        json.dumps(
            asdict(report),
            indent=2,
        )
    )

    print(
        f"\nSaved to {out_path}"
    )


if __name__ == "__main__":
    main()