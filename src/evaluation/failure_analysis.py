from __future__ import annotations

import json
import sys

from src.config import RESULTS_DIR, get_logger


logger = get_logger(__name__)

METRICS_PATH = RESULTS_DIR / "metrics.json"
FAILURE_ANALYSIS_PATH = RESULTS_DIR / "failure_analysis.md"
TOP_N_FAILURES = 5


try:
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace",
    )
except AttributeError:
    pass


def _hypothesize(
    example: dict,
) -> tuple[str, str]:
    intent_wrong = (
        example["true_intent"]
        != example["predicted_intent"]
    )

    action_wrong = (
        example["true_action"]
        != example["predicted_decision"]
    )

    if intent_wrong and action_wrong:
        return (
            "Intent misclassification likely cascaded into a wrong "
            "escalation decision, since intent confidence feeds the "
            "escalation policy directly.",
            "Check whether this message's true intent is under-represented "
            "in the clustering input, or genuinely overlaps multiple intents.",
        )

    if intent_wrong:
        return (
            "Intent was misclassified, but escalation happened to still be correct.",
            "Add more representative examples of the true intent, or review whether "
            "this message is genuinely ambiguous between two intents.",
        )

    if action_wrong:
        return (
            "Intent was correct, but escalation thresholds produced the wrong decision anyway.",
            "Review this example's retrieval similarity and grounding result — the "
            "escalation thresholds may need retuning on validation data.",
        )

    return (
        "Both intent and action were correct at the classification level — likely a "
        "reply-quality issue instead. See results/judge_results.csv for this id.",
        "Review judge scores for correctness/grounding on this specific example.",
    )


def main() -> None:
    if not METRICS_PATH.exists():
        content = (
            "STATUS: PENDING\n"
            "REASON: Evaluation has not been completed.\n"
        )

        print(content)

        FAILURE_ANALYSIS_PATH.write_text(
            content,
            encoding="utf-8",
        )

        return

    metrics = json.loads(
        METRICS_PATH.read_text()
    )

    if metrics.get("status") == "PENDING":
        content = (
            f"STATUS: PENDING\n"
            f"REASON: {metrics.get('reason', 'Golden set not annotated.')}\n"
        )

        print(content)

        FAILURE_ANALYSIS_PATH.write_text(
            content,
            encoding="utf-8",
        )

        return

    examples = metrics[
        "final_agent"
    ][
        "per_example_results"
    ]

    failures = [
        e
        for e in examples
        if e["true_intent"]
        != e["predicted_intent"]
        or e["true_action"]
        != e["predicted_decision"]
    ]

    if not failures:
        content = (
            "No failures found — the final agent matched golden labels "
            "on every example.\n"
        )

        print(content)

        FAILURE_ANALYSIS_PATH.write_text(
            content,
            encoding="utf-8",
        )

        return

    lines = [
        "# Failure Analysis\n",
        f"Found {len(failures)}/{len(examples)} golden examples "
        f"as failures. Showing top "
        f"{min(TOP_N_FAILURES, len(failures))}.\n",
    ]

    for i, ex in enumerate(
        failures[:TOP_N_FAILURES],
        start=1,
    ):
        why, fix = _hypothesize(
            ex
        )

        lines += [
            f"## Failure {i}: {ex['id']}\n",
            f"**Example:** {ex['customer_message']}\n",
            f"**Expected:** intent={ex['true_intent']}, "
            f"action={ex['true_action']}\n",
            f"**Actual:** intent={ex['predicted_intent']}, "
            f"action={ex['predicted_decision']}\n",
            f"**Why it failed:** {why}\n",
            f"**Potential fix:** {fix}\n",
        ]

    content = "\n".join(
        lines
    )

    FAILURE_ANALYSIS_PATH.write_text(
        content,
        encoding="utf-8",
    )

    print(content)


if __name__ == "__main__":
    main()