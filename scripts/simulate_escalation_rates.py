from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from src.config import (
    DATA_PROCESSED,
    SELECTED_BRAND_PATH,
    SETTINGS,
    conversations_path_for,
    get_logger,
)
from src.data.clean import clean_text
from src.escalation.policy import EscalationPolicy
from src.generation.generator import ResponseGenerator
from src.generation.grounding import GroundingValidator
from src.intents.classifier import IntentClassifier
from src.retrieval.retriever import Retriever

logger = get_logger(__name__)

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
]


def _message_key(message: str) -> str:
    return hashlib.sha256(
        message.encode("utf-8")
    ).hexdigest()


def _load_cache(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        return json.loads(
            path.read_text()
        )
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(
            f"Could not read cache at {path} ({e}); "
            "starting with an empty cache."
        )
        return {}


def _save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = path.with_suffix(".tmp")

    tmp_path.write_text(
        json.dumps(
            cache,
            indent=2,
            default=str,
        )
    )

    tmp_path.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__
    )

    ap.add_argument(
        "--n",
        type=int,
        default=200,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    ap.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help=(
            "Optional seconds to sleep after each FRESH "
            "(non-cached) pipeline call, to reduce the "
            "chance of hitting Groq's rate limit in the "
            "first place. Has no effect on cached messages. "
            "Default 0 = no change to current behavior."
        ),
    )

    ap.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help=(
            "Override the cache file location. Defaults to "
            "DATA_PROCESSED/escalation_simulation_cache_<brand>.json."
        ),
    )

    args = ap.parse_args()

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

    cache_path = (
        args.cache_path
        or (
            DATA_PROCESSED
            / f"escalation_simulation_cache_{brand}.json"
        )
    )

    cache = _load_cache(
        cache_path
    )

    df = (
        pd.read_csv(conv_path)
        .dropna(subset=["customer_message"])
    )

    sample = (
        df["customer_message"]
        .astype(str)
        .sample(
            min(args.n, len(df)),
            random_state=args.seed,
        )
    )

    n_already_cached = sum(
        1
        for m in sample
        if _message_key(m) in cache
    )

    n_to_fetch = (
        len(sample) - n_already_cached
    )

    print(
        f"Loading agent components -- "
        f"{n_already_cached}/{len(sample)} messages "
        f"already cached, {n_to_fetch} will make real "
        f"LLM calls..."
    )

    classifier = IntentClassifier()
    retriever = Retriever()
    generator = ResponseGenerator()
    grounding_validator = GroundingValidator()
    policy = EscalationPolicy()

    records = []
    cache_hits = 0
    cache_misses = 0

    for i, message in enumerate(
        sample,
        start=1,
    ):
        if i % 25 == 0:
            print(
                f"  ...{i}/{len(sample)} "
                f"(cache hits so far: {cache_hits})"
            )

        key = _message_key(
            message
        )

        cached = cache.get(
            key
        )

        if cached is not None:
            records.append(
                cached
            )
            cache_hits += 1
            continue

        cache_misses += 1

        cleaned = clean_text(
            message
        )

        intent_result = classifier.predict(
            cleaned
        )

        evidence = retriever.search(
            cleaned,
            top_k=5,
        )

        generation_result = generator.generate(
            cleaned,
            evidence,
        )

        grounding_result = (
            grounding_validator.validate(
                generation_result.get(
                    "reply",
                    "",
                ),
                evidence,
            )
        )

        record = {
            "message": message,
            "intent": intent_result["intent"],
            "intent_confidence": intent_result["confidence"],
            "intent_auto_eligible": (
                classifier.is_auto_eligible(
                    intent_result["intent"]
                )
            ),
            "evidence": evidence,
            "grounding_result": grounding_result,
        }

        records.append(
            record
        )

        cache[key] = record

        _save_cache(
            cache_path,
            cache,
        )

        if args.sleep > 0:
            time.sleep(
                args.sleep
            )

    print(
        f"\n{cache_hits} of {len(sample)} messages "
        f"served from cache "
        f"({cache_misses} required fresh LLM calls this run)."
    )

    print(
        f"\n{'=' * 70}\n"
        "END-TO-END AUTO_HANDLE RATE AT EACH CANDIDATE THRESHOLD\n"
        f"{'=' * 70}"
    )

    original_threshold = (
        SETTINGS.min_intent_confidence
    )

    try:
        for t in CANDIDATE_THRESHOLDS:
            SETTINGS.min_intent_confidence = t

            reason_counter = Counter()
            n_auto = 0

            for r in records:
                decision = policy.decide(
                    intent=r["intent"],
                    intent_confidence=r["intent_confidence"],
                    retrieval_evidence=r["evidence"],
                    grounding_result=r["grounding_result"],
                    customer_message=r["message"],
                    intent_auto_eligible=r[
                        "intent_auto_eligible"
                    ],
                )

                if decision["decision"] == "AUTO_HANDLE":
                    n_auto += 1
                else:
                    if (
                        "confidence"
                        in decision["reason"].lower()
                        or "unrecognized"
                        in decision["reason"].lower()
                    ):
                        reason_counter[
                            "intent_confidence_too_low"
                        ] += 1

                    if (
                        "not a confidently-matched category"
                        in decision["reason"]
                    ):
                        reason_counter[
                            "intent_cluster_not_eligible"
                        ] += 1

                    if (
                        "similarity"
                        in decision["reason"].lower()
                        or "no similar"
                        in decision["reason"].lower()
                    ):
                        reason_counter[
                            "weak_or_no_retrieval"
                        ] += 1

                    if (
                        "grounding"
                        in decision["reason"].lower()
                    ):
                        reason_counter[
                            "grounding_failed"
                        ] += 1

                    if (
                        "risk"
                        in decision["reason"].lower()
                    ):
                        reason_counter[
                            "high_risk_language"
                        ] += 1

                    if (
                        "frustration"
                        in decision["reason"].lower()
                    ):
                        reason_counter[
                            "angry_customer"
                        ] += 1

            pct = (
                n_auto
                / len(records)
                * 100
            )

            print(
                f"\nthreshold {t:.2f}: "
                f"{n_auto}/{len(records)} AUTO_HANDLE "
                f"({pct:.1f}%)"
            )

            for reason, count in (
                reason_counter.most_common()
            ):
                print(
                    f"    escalated due to "
                    f"{reason}: {count}"
                )

    finally:
        SETTINGS.min_intent_confidence = (
            original_threshold
        )

    print(
        "\nThis is the REAL end-to-end rate, not the "
        "intent-classifier-alone rate -- notice how much "
        "of the escalation volume comes from "
        "intent_cluster_not_eligible and "
        "weak_or_no_retrieval rather than the threshold "
        "itself. Pick MIN_INTENT_CONFIDENCE based on where "
        "this table's overall AUTO_HANDLE% and reason "
        "breakdown look sound to you, then validate for "
        "real once src.evaluation.evaluate can run against "
        "your annotated golden set."
    )


if __name__ == "__main__":
    main()