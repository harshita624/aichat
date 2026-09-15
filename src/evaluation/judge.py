from __future__ import annotations

import csv
import json

from src.config import RESULTS_DIR, SETTINGS, get_logger
from src.generation.generator import (
    _call_gemini,
    _call_groq,
    _call_openai,
)


logger = get_logger(__name__)

METRICS_PATH = RESULTS_DIR / "metrics.json"
JUDGE_RESULTS_PATH = RESULTS_DIR / "judge_results.csv"


JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator of customer support responses. You will be \
given a customer message, historical evidence the response was based on, the generated reply, and \
(if available) a human-written reference resolution.

Score the reply 1-5 (5=best) on: correctness, relevance, grounding, completeness, helpfulness.
Also list any unsupported_claims (numbers/dates/promises not backed by evidence).

Respond with ONLY this JSON shape, no markdown:
{"correctness": 1-5, "relevance": 1-5, "grounding": 1-5, "completeness": 1-5, "helpfulness": 1-5,
"unsupported_claims": ["strings, empty if none"], "judge_reasoning": "one or two short sentences"}
"""


JUDGE_REQUIRED_KEYS = {
    "correctness",
    "relevance",
    "grounding",
    "completeness",
    "helpfulness",
}


def _parse_judge_json(raw: str) -> dict | None:
    cleaned = (
        raw.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )

    try:
        parsed = json.loads(cleaned)

    except json.JSONDecodeError:
        start, end = (
            cleaned.find("{"),
            cleaned.rfind("}"),
        )

        if (
            start == -1
            or end == -1
            or end < start
        ):
            return None

        try:
            parsed = json.loads(
                cleaned[start:end + 1]
            )

        except json.JSONDecodeError:
            return None

    if not JUDGE_REQUIRED_KEYS.issubset(
        parsed.keys()
    ):
        logger.warning(
            f"Judge response missing keys: "
            f"{JUDGE_REQUIRED_KEYS - parsed.keys()}"
        )
        return None

    return parsed


def _build_judge_prompt(example: dict) -> str:
    evidence_text = "\n".join(
        f"- Customer: {e.get('historical_customer_message', '')} "
        f"| Agent: {e.get('historical_agent_response', '')}"
        for e in example.get(
            "retrieved_evidence",
            [],
        )
    ) or "No evidence was retrieved."

    return (
        f"Customer message: {example['customer_message']}\n\n"
        f"Retrieved evidence:\n{evidence_text}\n\n"
        f"Reference resolution: "
        f"{example.get('reference_resolution', 'Not provided.')}\n\n"
        f"Generated reply to evaluate: "
        f"{example['generated_reply']}\n\n"
        f"Return only the JSON object."
    )


def judge_single(
    example: dict,
) -> dict | None:
    if not SETTINGS.llm_configured():
        return None

    prompt = _build_judge_prompt(
        example
    )

    try:
        if SETTINGS.llm_provider == "groq":
            raw = _call_groq(
                JUDGE_SYSTEM_PROMPT,
                prompt,
            )

        elif SETTINGS.llm_provider == "openai":
            raw = _call_openai(
                JUDGE_SYSTEM_PROMPT,
                prompt,
            )

        elif SETTINGS.llm_provider == "gemini":
            raw = _call_gemini(
                JUDGE_SYSTEM_PROMPT,
                prompt,
            )

        else:
            return None

    except Exception as e:
        logger.error(
            f"Judge call failed for "
            f"{example.get('id')}: {e}"
        )
        return None

    return _parse_judge_json(raw)


def _load_existing_results() -> dict[str, dict]:
    if not JUDGE_RESULTS_PATH.exists():
        return {}

    existing = {}

    try:
        with open(
            JUDGE_RESULTS_PATH,
            "r",
            newline="",
            encoding="utf-8",
        ) as f:
            for row in csv.DictReader(f):
                existing[row["id"]] = row

    except (
        OSError,
        csv.Error,
        KeyError,
    ) as e:
        logger.warning(
            f"Could not read existing judge results at "
            f"{JUDGE_RESULTS_PATH} ({e}); starting fresh."
        )
        return {}

    return existing


def _save_results(
    rows: list[dict],
) -> None:
    if not rows:
        return

    fieldnames = [
        "id",
        "status",
        "correctness",
        "relevance",
        "grounding",
        "completeness",
        "helpfulness",
        "unsupported_claims",
        "judge_reasoning",
    ]

    with open(
        JUDGE_RESULTS_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not METRICS_PATH.exists():
        print(
            "No results/metrics.json found. "
            "Run python -m src.evaluation.evaluate first."
        )
        return

    metrics = json.loads(
        METRICS_PATH.read_text()
    )

    if metrics.get("status") == "PENDING":
        print(
            "STATUS: PENDING\n"
            "REASON: Golden set not annotated yet, "
            "so evaluation hasn't run."
        )
        return

    if not SETTINGS.llm_configured():
        print(
            f"STATUS: PENDING\n"
            f"REASON: No LLM provider configured "
            f"(LLM_PROVIDER={SETTINGS.llm_provider})."
        )

        print(
            "HOW TO GENERATE: Set an API key in .env, "
            "then re-run this script."
        )

        return

    examples = metrics[
        "final_agent"
    ][
        "per_example_results"
    ]

    existing = _load_existing_results()

    n_reusable = sum(
        1
        for ex in examples
        if existing.get(
            ex["id"],
            {},
        ).get("status") == "OK"
    )

    logger.info(
        f"Judging {len(examples)} examples "
        f"with {SETTINGS.llm_provider} "
        f"({n_reusable} already judged successfully "
        f"and will be reused)..."
    )

    rows_by_id: dict[str, dict] = {}

    for ex in examples:
        prior = existing.get(
            ex["id"]
        )

        if (
            prior
            and prior.get("status") == "OK"
        ):
            rows_by_id[
                ex["id"]
            ] = prior

    rows = list(
        rows_by_id.values()
    )

    n_done = 0

    for ex in examples:
        n_done += 1

        if n_done % 25 == 0:
            print(
                f"  ...{n_done}/{len(examples)}"
            )

        if ex["id"] in rows_by_id:
            continue

        judgment = judge_single(
            ex
        )

        if judgment is None:
            row = {
                "id": ex["id"],
                "status": "JUDGE_FAILED",
                "correctness": None,
                "relevance": None,
                "grounding": None,
                "completeness": None,
                "helpfulness": None,
                "unsupported_claims": None,
                "judge_reasoning": None,
            }

        else:
            row = {
                "id": ex["id"],
                "status": "OK",
                "correctness": judgment.get(
                    "correctness"
                ),
                "relevance": judgment.get(
                    "relevance"
                ),
                "grounding": judgment.get(
                    "grounding"
                ),
                "completeness": judgment.get(
                    "completeness"
                ),
                "helpfulness": judgment.get(
                    "helpfulness"
                ),
                "unsupported_claims": "; ".join(
                    judgment.get(
                        "unsupported_claims",
                        [],
                    )
                ),
                "judge_reasoning": judgment.get(
                    "judge_reasoning"
                ),
            }

        rows_by_id[
            ex["id"]
        ] = row

        rows = list(
            rows_by_id.values()
        )

        _save_results(
            rows
        )

    n_ok = sum(
        1
        for r in rows
        if r["status"] == "OK"
    )

    print(
        f"Judged {n_ok}/{len(rows)} successfully. "
        f"Saved to {JUDGE_RESULTS_PATH}"
    )

    print(
        "Reminder: LLM-judge scores, not ground truth. "
        "See judge_agreement.py to compare against humans."
    )


if __name__ == "__main__":
    main()