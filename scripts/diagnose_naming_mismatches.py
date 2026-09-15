from __future__ import annotations

import json
from collections import Counter

from src.config import RESULTS_DIR

METRICS_PATH = RESULTS_DIR / "metrics.json"


def _is_naming_variant(
    true_intent: str,
    predicted_intent: str,
) -> bool:
    if true_intent == predicted_intent:
        return False

    t_base = true_intent.split("__")[0]
    p_base = predicted_intent.split("__")[0]

    return t_base == p_base


def main() -> None:
    if not METRICS_PATH.exists():
        print(
            "No results/metrics.json found. "
            "Run python -m src.evaluation.evaluate first."
        )
        return

    metrics = json.loads(
        METRICS_PATH.read_text(
            encoding="utf-8"
        )
    )

    if metrics.get("status") != "COMPLETE":
        print(
            "Evaluation hasn't completed yet."
        )
        return

    examples = metrics[
        "final_agent"
    ][
        "per_example_results"
    ]

    buckets = Counter()
    naming_examples = []
    confusion_examples = []

    for ex in examples:
        true_i = ex["true_intent"]
        pred_i = ex["predicted_intent"]
        true_a = ex["true_action"]
        pred_a = ex["predicted_decision"]

        if true_i == pred_i and true_a == pred_a:
            buckets["correct"] += 1

        elif _is_naming_variant(
            true_i,
            pred_i,
        ):
            buckets["naming_artifact"] += 1
            naming_examples.append(
                (
                    ex["id"],
                    true_i,
                    pred_i,
                )
            )

        elif true_i == pred_i:
            buckets["escalation_only"] += 1

        else:
            buckets["genuine_confusion"] += 1
            confusion_examples.append(
                (
                    ex["id"],
                    true_i,
                    pred_i,
                )
            )

    total = len(examples)

    print(
        f"Out of {total} golden examples:\n"
    )

    for label in [
        "correct",
        "naming_artifact",
        "escalation_only",
        "genuine_confusion",
    ]:
        n = buckets[label]

        print(
            f"  {label:20s}: "
            f"{n:>4d} "
            f"({n / total * 100:5.1f}%)"
        )

    if naming_examples:
        print(
            "\nSample naming-artifact mismatches "
            "(true -> predicted):"
        )

        for id_, t, p in naming_examples[:10]:
            print(
                f"  {id_}: {t!r} -> {p!r}"
            )

    if confusion_examples:
        true_intents_confused = Counter(
            t
            for _, t, _ in confusion_examples
        )

        print(
            "\nTrue intents most often genuinely "
            "confused (top 8):"
        )

        for intent, count in (
            true_intents_confused.most_common(8)
        ):
            print(
                f"  {intent}: {count}"
            )

    corrected_accuracy = (
        buckets["correct"]
        + buckets["naming_artifact"]
    ) / total

    print(
        f"\nIf naming artifacts were normalized away, "
        f"intent accuracy would read ~"
        f"{corrected_accuracy:.4f} instead of the "
        f"reported "
        f"{metrics['final_agent']['intent']['accuracy']}."
    )


if __name__ == "__main__":
    main()