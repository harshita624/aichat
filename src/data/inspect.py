from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATA_RAW, INSPECTION_SUMMARY_PATH, get_logger

logger = get_logger(__name__)

SAMPLE_ROWS = 5000


def find_dataset_files() -> list[Path]:
    if not DATA_RAW.exists():
        return []
    return sorted(DATA_RAW.rglob("*.csv"))


def count_rows_fast(path: Path) -> int:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        total_lines = sum(1 for _ in f)
    return max(total_lines - 1, 0)


def _score_column_name(name: str, keywords: list[str]) -> int:
    lowered = name.lower()
    return sum(1 for kw in keywords if kw in lowered)


def detect_columns(df: pd.DataFrame) -> dict[str, str | None]:
    columns = list(df.columns)

    def contains_any(name: str, keywords: list[str]) -> bool:
        lowered = name.lower()
        return any(kw in lowered for kw in keywords)

    def best_match(
        keywords: list[str],
        exclude_keywords: list[str] | None = None,
    ) -> str | None:
        exclude_keywords = exclude_keywords or []
        scored = []
        for c in columns:
            if contains_any(c, exclude_keywords):
                continue
            score = _score_column_name(c, keywords)
            if score > 0:
                scored.append((c, score, len(c)))
        if not scored:
            return None
        scored.sort(key=lambda x: (-x[1], x[2]))
        return scored[0][0]

    in_response_to_col = best_match(
        ["in_response_to", "reply_to", "parent_id"]
    )
    response_ids_col = best_match(
        ["response_tweet", "reply_ids", "replies"]
    )
    author_col = best_match(
        ["author", "user_id", "username", "from_user"]
    )
    inbound_col = best_match(
        ["inbound", "direction", "is_customer"]
    )
    timestamp_col = best_match(
        ["created_at", "timestamp", "created", "date", "time"]
    )

    id_col = best_match(
        ["tweet_id", "id"],
        exclude_keywords=[
            "response",
            "author",
            "user",
            "parent",
            "reply",
        ],
    )

    text_col = best_match(
        ["text", "message", "body", "content"]
    )

    if text_col is None:
        object_cols = [
            c for c in columns if df[c].dtype == object
        ]
        if object_cols:
            avg_lengths = {
                c: df[c].dropna().astype(str).str.len().mean()
                for c in object_cols
            }
            text_col = (
                max(avg_lengths, key=avg_lengths.get)
                if avg_lengths
                else None
            )

    return {
        "id_column": id_col,
        "text_column": text_col,
        "author_column": author_col,
        "inbound_column": inbound_col,
        "in_response_to_column": in_response_to_col,
        "response_ids_column": response_ids_col,
        "timestamp_column": timestamp_col,
    }


def inspect_file(path: Path) -> dict[str, Any]:
    logger.info(f"Inspecting {path} ...")
    sample_df = pd.read_csv(
        path,
        nrows=SAMPLE_ROWS,
        low_memory=False,
    )
    total_rows = count_rows_fast(path)

    detected = detect_columns(sample_df)
    null_counts = sample_df.isnull().sum().to_dict()
    dtypes = {
        c: str(sample_df[c].dtype)
        for c in sample_df.columns
    }
    sample_rows = sample_df.head(3).to_dict(
        orient="records"
    )

    return {
        "file": str(path.relative_to(DATA_RAW)),
        "total_rows_estimate": total_rows,
        "sample_rows_loaded": len(sample_df),
        "columns": list(sample_df.columns),
        "dtypes": dtypes,
        "null_counts_in_sample": null_counts,
        "detected_columns": detected,
        "sample_records": sample_rows,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"\n=== {summary['file']} ===")
    print(
        f"Total rows (estimate): "
        f"{summary['total_rows_estimate']:,}"
    )
    print(f"Columns: {summary['columns']}")
    print("\nDetected column roles:")

    for role, col in summary["detected_columns"].items():
        print(
            f"  {role:24s} -> "
            f"{col if col else 'NOT DETECTED'}"
        )

    missing = [
        role
        for role, col in summary["detected_columns"].items()
        if col is None
    ]

    if missing:
        print(
            f"\nWARNING: could not confidently detect: {missing}. "
            "Downstream scripts relying on these will fail with an "
            "explicit error rather than guessing — inspect the raw "
            "file manually if this happens."
        )


def main() -> None:
    files = find_dataset_files()

    if not files:
        print(
            "No CSV files found under data/raw/.\n\n"
            "Download the dataset first:\n"
            "  kaggle datasets download -d "
            "thoughtvector/customer-support-on-twitter "
            "-p data/raw --unzip\n\n"
            "(Requires kaggle.json set up first — see PART 1 setup "
            "instructions.)\n"
            "After downloading, re-run: "
            "python -m src.data.inspect"
        )
        return

    all_summaries = []

    for f in files:
        try:
            summary = inspect_file(f)
        except Exception as e:
            print(f"Could not read {f}: {e}")
            continue

        print_summary(summary)
        all_summaries.append(summary)

    if not all_summaries:
        print("No files could be successfully inspected.")
        return

    main_summary = max(
        all_summaries,
        key=lambda s: s["total_rows_estimate"],
    )

    print(
        f"\nLargest file (assumed main dataset): "
        f"{main_summary['file']}"
    )

    output = {
        "main_dataset": main_summary,
        "all_files_inspected": all_summaries,
    }

    INSPECTION_SUMMARY_PATH.write_text(
        json.dumps(
            output,
            indent=2,
            default=str,
        )
    )

    print(
        f"\nSaved inspection summary to "
        f"{INSPECTION_SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()