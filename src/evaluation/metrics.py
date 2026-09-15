from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def intent_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    p, r, f, s = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    per_intent = {
        label: {
            "precision": round(float(pi), 4),
            "recall": round(float(ri), 4),
            "f1": round(float(fi), 4),
            "support": int(si),
        }
        for label, pi, ri, fi, si in zip(
            labels,
            p,
            r,
            f,
            s,
        )
    }

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    return {
        "accuracy": round(float(accuracy), 4),
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_intent": per_intent,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels,
        "n_examples": len(y_true),
    }


def escalation_metrics(
    y_true_escalate: list[bool],
    y_pred_escalate: list[bool],
) -> dict:
    precision = precision_score(
        y_true_escalate,
        y_pred_escalate,
        zero_division=0,
    )

    recall = recall_score(
        y_true_escalate,
        y_pred_escalate,
        zero_division=0,
    )

    f1 = f1_score(
        y_true_escalate,
        y_pred_escalate,
        zero_division=0,
    )

    y_true_arr = np.array(y_true_escalate)
    y_pred_arr = np.array(y_pred_escalate)

    should_auto = ~y_true_arr

    false_escalations = int(
        np.sum(
            should_auto & y_pred_arr
        )
    )

    false_escalation_rate = (
        false_escalations
        / max(int(should_auto.sum()), 1)
    )

    should_escalate = y_true_arr

    missed_escalations = int(
        np.sum(
            should_escalate & ~y_pred_arr
        )
    )

    missed_escalation_rate = (
        missed_escalations
        / max(int(should_escalate.sum()), 1)
    )

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "false_escalation_rate": round(
            float(false_escalation_rate),
            4,
        ),
        "false_escalation_count": false_escalations,
        "missed_escalation_rate": round(
            float(missed_escalation_rate),
            4,
        ),
        "missed_escalation_count": missed_escalations,
        "n_examples": len(y_true_escalate),
    }


def save_confusion_matrix_plot(
    cm: list[list[int]],
    labels: list[str],
    output_path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    cm_arr = np.array(cm)

    fig, ax = plt.subplots(
        figsize=(
            max(6, len(labels) * 0.8),
            max(5, len(labels) * 0.7),
        )
    )

    im = ax.imshow(
        cm_arr,
        cmap="Blues",
    )

    ax.set_xticks(
        range(len(labels))
    )

    ax.set_yticks(
        range(len(labels))
    )

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(labels)

    ax.set_xlabel(
        "Predicted intent"
    )

    ax.set_ylabel(
        "True intent"
    )

    ax.set_title(
        "Intent Classification Confusion Matrix"
    )

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                str(cm_arr[i, j]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm_arr[i, j] > cm_arr.max() / 2
                    else "black"
                ),
            )

    fig.colorbar(
        im,
        ax=ax,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=120,
    )

    plt.close(fig)