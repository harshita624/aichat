from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from src.agent import SupportAgent
from src.baselines.tfidf import TfidfBaseline, TFIDF_MODEL_PATH
from src.baselines.trivial import TrivialBaseline
from src.config import (
    DATA_GOLDEN,
    RESULTS_DIR,
    SETTINGS,
    get_logger,
)
from src.evaluation.leakage import (
    build_report as build_leakage_report,
)
from src.evaluation.metrics import (
    escalation_metrics,
    intent_metrics,
    save_confusion_matrix_plot,
)
from src.generation.prompts import SYSTEM_PROMPT


logger = get_logger(__name__)

GOLDEN_PATH = DATA_GOLDEN / "golden_set.csv"
METRICS_OUTPUT_PATH = RESULTS_DIR / "metrics.json"
CONFUSION_MATRIX_OUTPUT_PATH = RESULTS_DIR / "confusion_matrix.png"

_SOURCE_ID_PATTERN = re.compile(
    r"source_conversation_id=(\S+)"
)

EVAL_CACHE_PATH = RESULTS_DIR / "agent_eval_cache.json"


def _extract_source_conversation_id(
    notes,
) -> str | None:
    if not isinstance(notes, str):
        return None

    match = _SOURCE_ID_PATTERN.search(
        notes
    )

    return match.group(1) if match else None


def _current_cache_fingerprint() -> dict:
    provider = SETTINGS.llm_provider

    model = {
        "groq": SETTINGS.groq_model,
        "openai": SETTINGS.openai_model,
        "gemini": SETTINGS.gemini_model,
    }.get(
        provider,
        "unconfigured",
    )

    return {
        "provider": provider,
        "model": model,
        "prompt_fingerprint": hashlib.sha256(
            SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest()[:10],
        "min_intent_confidence": SETTINGS.min_intent_confidence,
        "min_retrieval_similarity": SETTINGS.min_retrieval_similarity,
    }


def _load_cache(
    path: Path,
) -> dict:
    if not path.exists():
        return {}

    try:
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except (
        json.JSONDecodeError,
        OSError,
    ) as e:
        logger.warning(
            f"Could not read {path} ({e}) — "
            f"starting with an empty cache."
        )
        return {}

    current_fp = _current_cache_fingerprint()

    if data.get("fingerprint") != current_fp:
        logger.info(
            f"Evaluation cache at {path} was built under "
            f"different settings (provider/model/prompt/threshold "
            f"changed since then) -- starting fresh rather than "
            f"mixing results from two configurations. Old cache is "
            f"left on disk in case you want to inspect it."
        )
        return {}

    return data.get(
        "rows",
        {},
    )


def _save_cache(
    path: Path,
    rows: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "fingerprint": _current_cache_fingerprint(),
        "rows": rows,
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def _load_golden_set() -> pd.DataFrame | None:
    if not GOLDEN_PATH.exists():
        return None

    df = pd.read_csv(
        GOLDEN_PATH
    )

    df = df[
        ~df["id"]
        .astype(str)
        .str.startswith("EXAMPLE")
    ]

    df = df[
        df["intent"].notna()
        & (
            df["intent"]
            .astype(str)
            .str.strip()
            != ""
        )
    ]

    return df if len(df) > 0 else None


def _run_trivial(
    golden_df: pd.DataFrame,
) -> dict:
    baseline = TrivialBaseline()

    y_true_i = []
    y_pred_i = []
    y_true_e = []
    y_pred_e = []

    for _, row in golden_df.iterrows():
        pred = baseline.predict(
            row["customer_message"]
        )

        y_true_i.append(
            row["intent"]
        )

        y_pred_i.append(
            pred["intent"]
        )

        y_true_e.append(
            row["expected_action"]
            == "ESCALATE"
        )

        y_pred_e.append(
            pred["decision"]
            == "ESCALATE"
        )

    return {
        "intent": intent_metrics(
            y_true_i,
            y_pred_i,
        ),
        "escalation": escalation_metrics(
            y_true_e,
            y_pred_e,
        ),
    }


def _run_tfidf(
    golden_df: pd.DataFrame,
) -> dict:
    if not TFIDF_MODEL_PATH.exists():
        return {
            "status": "PENDING",
            "reason": (
                "TF-IDF baseline not trained. "
                "Run python -m src.baselines.tfidf."
            ),
        }

    baseline = TfidfBaseline.load()

    y_true_i = []
    y_pred_i = []

    for _, row in golden_df.iterrows():
        pred = baseline.predict_intent(
            row["customer_message"]
        )

        y_true_i.append(
            row["intent"]
        )

        y_pred_i.append(
            pred["intent"]
        )

    return {
        "intent": intent_metrics(
            y_true_i,
            y_pred_i,
        )
    }


def _run_final_agent(
    golden_df: pd.DataFrame,
    use_cache: bool = True,
) -> dict:
    agent = SupportAgent()

    cached_rows = (
        _load_cache(
            EVAL_CACHE_PATH
        )
        if use_cache
        else {}
    )

    if (
        not use_cache
        and EVAL_CACHE_PATH.exists()
    ):
        logger.info(
            "Ignoring existing evaluation cache (--fresh)."
        )

    y_true_i = []
    y_pred_i = []
    y_true_e = []
    y_pred_e = []
    per_example = []

    n_missing_source_id = 0
    n_from_cache = 0

    for _, row in golden_df.iterrows():
        row_id = str(
            row["id"]
        )

        source_id = _extract_source_conversation_id(
            row.get(
                "notes",
                "",
            )
        )

        if source_id is None:
            n_missing_source_id += 1

        cached = cached_rows.get(
            row_id
        )

        if (
            cached is not None
            and cached.get("customer_message")
            == row["customer_message"]
        ):
            entry = cached
            n_from_cache += 1

        else:
            result = agent.handle(
                row["customer_message"],
                exclude_conversation_id=source_id,
                exclude_message_text=row[
                    "customer_message"
                ],
            )

            entry = {
                "customer_message": row[
                    "customer_message"
                ],
                "predicted_intent": result[
                    "intent"
                ],
                "predicted_decision": result[
                    "decision"
                ],
                "generated_reply": result[
                    "generated_reply"
                ],
                "retrieved_evidence": result[
                    "retrieved_evidence"
                ],
                "escalation_reason": result[
                    "escalation_reason"
                ],
            }

            cached_rows[row_id] = entry

            _save_cache(
                EVAL_CACHE_PATH,
                cached_rows,
            )

        y_true_i.append(
            row["intent"]
        )

        y_pred_i.append(
            entry["predicted_intent"]
        )

        y_true_e.append(
            row["expected_action"]
            == "ESCALATE"
        )

        y_pred_e.append(
            entry["predicted_decision"]
            == "ESCALATE"
        )

        per_example.append(
            {
                "id": row["id"],
                "customer_message": row[
                    "customer_message"
                ],
                "true_intent": row[
                    "intent"
                ],
                "predicted_intent": entry[
                    "predicted_intent"
                ],
                "true_action": row[
                    "expected_action"
                ],
                "predicted_decision": entry[
                    "predicted_decision"
                ],
                "generated_reply": entry[
                    "generated_reply"
                ],
                "retrieved_evidence": entry[
                    "retrieved_evidence"
                ],
                "reference_resolution": row.get(
                    "reference_resolution",
                    "",
                ),
                "escalation_reason": entry[
                    "escalation_reason"
                ],
            }
        )

    if n_from_cache:
        logger.info(
            f"Reused {n_from_cache}/{len(golden_df)} rows "
            f"from a previous interrupted run "
            f"({EVAL_CACHE_PATH}) instead of re-calling the agent."
        )

    if n_missing_source_id > 0:
        logger.warning(
            f"{n_missing_source_id}/{len(golden_df)} golden rows "
            f"had no parseable source_conversation_id in their "
            f"'notes' column. exclude_message_text still protects "
            f"these from exact-text leakage, but "
            f"exclude_conversation_id could not be applied "
            f"specifically for them -- check that their 'notes' "
            f"column actually contains 'source_conversation_id=...' "
            f"if this count is high."
        )

    return {
        "intent": intent_metrics(
            y_true_i,
            y_pred_i,
        ),
        "escalation": escalation_metrics(
            y_true_e,
            y_pred_e,
        ),
        "per_example_results": per_example,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__
    )

    ap.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Ignore any existing evaluation cache and re-run "
            "every golden example against the agent from scratch. "
            "Note this happens AUTOMATICALLY anyway if the provider, "
            "model, prompt, or thresholds changed since the cache "
            "was written -- use this flag to force it even when "
            "nothing changed."
        ),
    )

    args = ap.parse_args()

    golden_df = _load_golden_set()

    if golden_df is None:
        status = {
            "status": "PENDING",
            "reason": (
                "Golden set has not been manually annotated."
            ),
            "how_to_generate": (
                "Complete data/golden/golden_set_template.csv per "
                "data/golden/annotation_guidelines.md, then re-run: "
                "python -m src.evaluation.evaluate"
            ),
        }

        print(
            json.dumps(
                status,
                indent=2,
            )
        )

        METRICS_OUTPUT_PATH.write_text(
            json.dumps(
                status,
                indent=2,
            ),
            encoding="utf-8",
        )

        return

    logger.info(
        f"Loaded {len(golden_df)} annotated golden examples."
    )

    leakage_report = build_leakage_report()

    if leakage_report.golden_retrieval_overlap > 0:
        logger.warning(
            f"{leakage_report.golden_retrieval_overlap} golden examples "
            f"overlap verbatim with the retrieval corpus (expected, by "
            f"construction -- see leakage.py). exclude_conversation_id/"
            f"exclude_message_text below are what actually prevent this "
            f"from inflating retrieval-dependent metrics."
        )

    trivial_results = _run_trivial(
        golden_df
    )

    tfidf_results = _run_tfidf(
        golden_df
    )

    final_results = _run_final_agent(
        golden_df,
        use_cache=not args.fresh,
    )

    save_confusion_matrix_plot(
        final_results["intent"][
            "confusion_matrix"
        ],
        final_results["intent"][
            "confusion_matrix_labels"
        ],
        CONFUSION_MATRIX_OUTPUT_PATH,
    )

    output = {
        "status": "COMPLETE",
        "n_golden_examples": len(
            golden_df
        ),
        "leakage_check": {
            "exact_duplicates": (
                leakage_report.exact_duplicates
            ),
            "golden_retrieval_overlap": (
                leakage_report.golden_retrieval_overlap
            ),
            "temporal_check_status": (
                leakage_report.temporal_check_status
            ),
        },
        "trivial_baseline": trivial_results,
        "tfidf_baseline": tfidf_results,
        "final_agent": final_results,
        "note": (
            "Metrics above are automatically computed from "
            "golden-set labels. Reply-quality scores are NOT "
            "included — see judge.py / judge_results.csv, "
            "which are LLM-judge scores, not ground truth."
        ),
    }

    METRICS_OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        f"\nSaved to {METRICS_OUTPUT_PATH}"
    )

    print(
        f"Final agent intent accuracy: "
        f"{final_results['intent']['accuracy']}"
    )

    print(
        f"Final agent macro F1: "
        f"{final_results['intent']['macro_f1']}"
    )

    print(
        f"Escalation F1: "
        f"{final_results['escalation']['f1']}"
    )

    print(
        f"Missed escalation rate: "
        f"{final_results['escalation']['missed_escalation_rate']}"
    )


if __name__ == "__main__":
    main()