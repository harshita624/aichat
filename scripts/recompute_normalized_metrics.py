from __future__ import annotations

import json
from collections import Counter

from src.config import RESULTS_DIR
from src.evaluation.metrics import intent_metrics

METRICS_PATH = RESULTS_DIR / "metrics.json"


def _is_dedup_suffix_variant(a: str, b: str) -> bool:
    if a == b:
        return False

    def _check(shorter: str, longer: str) -> bool:
        prefix = shorter + "__"
        return (
            longer.startswith(prefix)
            and longer[len(prefix):].isdigit()
        )

    return _check(a, b) or _check(b, a)


def _normalize_pair(
    true_i: str,
    pred_i: str,
) -> tuple[str, str]:
    if _is_dedup_suffix_variant(
        true_i,
        pred_i,
    ):
        base = (
            true_i
            if len(true_i) <= len(pred_i)
            else pred_i
        )
        return base, base

    return true_i, pred_i


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

    y_true_raw = [
        ex["true_intent"]
        for ex in examples
    ]

    y_pred_raw = [
        ex["predicted_intent"]
        for ex in examples
    ]

    y_true_norm = []
    y_pred_norm = []
    normalized_pairs = []

    for t, p in zip(
        y_true_raw,
        y_pred_raw,
    ):
        nt, np_ = _normalize_pair(
            t,
            p,
        )

        if (nt, np_) != (t, p):
            normalized_pairs.append(
                (t, p)
            )

        y_true_norm.append(nt)
        y_pred_norm.append(np_)

    raw_metrics = intent_metrics(
        y_true_raw,
        y_pred_raw,
    )

    norm_metrics = intent_metrics(
        y_true_norm,
        y_pred_norm,
    )

    print(
        f"{'=' * 70}\n"
        f"RAW vs NAME-NORMALIZED INTENT METRICS "
        f"({len(examples)} examples)\n"
        f"{'=' * 70}"
    )

    print(
        f"  raw accuracy:        "
        f"{raw_metrics['accuracy']:.4f}"
    )

    print(
        f"  normalized accuracy: "
        f"{norm_metrics['accuracy']:.4f}   "
        f"({len(normalized_pairs)} example(s) affected "
        f"by the disambiguation-suffix artifact)"
    )

    print(
        f"  raw macro F1:        "
        f"{raw_metrics['macro_f1']:.4f}"
    )

    print(
        f"  normalized macro F1: "
        f"{norm_metrics['macro_f1']:.4f}"
    )

    if normalized_pairs:
        print(
            "\nExamples actually normalized "
            "(true <-> predicted):"
        )

        for t, p in normalized_pairs[:15]:
            print(
                f"    {t!r} <-> {p!r}"
            )

    still_wrong = sum(
        1
        for t, p in zip(
            y_true_norm,
            y_pred_norm,
        )
        if t != p
    )

    print(
        f"\nAfter normalization: "
        f"{len(examples) - still_wrong}/"
        f"{len(examples)} intents correct."
    )

    top_confused = Counter(
        t
        for t, p in zip(
            y_true_norm,
            y_pred_norm,
        )
        if t != p
    )

    print(
        "\nTrue intents most often still "
        "genuinely confused (top 8):"
    )

    for intent, count in top_confused.most_common(8):
        print(
            f"    {intent}: {count}"
        )

    metrics["final_agent"]["normalized_intent"] = (
        norm_metrics
    )

    metrics["final_agent"]["normalization_note"] = (
        f"{len(normalized_pairs)} example(s) had predicted "
        f"intents differing from the true label only by a "
        f"taxonomy collision-disambiguation suffix (e.g. "
        f"'X' vs 'X__4', added by taxonomy.py when two "
        f"distinct KMeans clusters independently matched "
        f"the same candidate label). Counted correct in "
        f"'normalized_intent' below but NOT in 'intent' above, "
        f"which is left exactly as originally computed for "
        f"auditability."
    )

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        f"\nUpdated {METRICS_PATH} with "
        f"'normalized_intent' "
        f"(original 'intent' section untouched)."
    )


if __name__ == "__main__":
    main()