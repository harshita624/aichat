from __future__ import annotations

from src.intents.lexical_signals import (
    base_taxonomy_category,
    normalize_text,
)


CANONICAL_INTENTS = {
    "product_quality",
    "customer_service_complaint",
    "OVERSIZED_MIXED_CLUSTER__5",
    "delivery_issue",
    "stock_availability",
    "uncategorized__dm",
    "technical_problem",
    "uncategorized__meal",
    "uncategorized__free",
    "uncategorized__tesco",
}


def canonicalize_intent(
    intent: str | None,
    text: str = "",
) -> str:
    if not intent:
        return "OVERSIZED_MIXED_CLUSTER__5"

    if intent in CANONICAL_INTENTS:
        return intent

    base = base_taxonomy_category(intent)

    if base in CANONICAL_INTENTS:
        return base

    normalized = normalize_text(text)

    if intent in {
        "uncategorized__sent",
        "uncategorized__dm",
    }:
        if (
            "dm" in normalized
            or "direct message" in normalized
            or "private message" in normalized
        ):
            return "uncategorized__dm"

        return "OVERSIZED_MIXED_CLUSTER__5"

    if intent in {
        "payment_issue",
        "account_issue",
        "payment_account_issue",
    }:
        if any(
            term in normalized
            for term in (
                "clubcard",
                "account",
                "fuel",
                "forecourt",
            )
        ):
            return "uncategorized__tesco"

        return "OVERSIZED_MIXED_CLUSTER__5"

    if intent == "pricing_complaint":
        if (
            "meal deal" in normalized
            or "3 for 2" in normalized
        ):
            return "uncategorized__meal"

        if any(
            term in normalized
            for term in (
                "diesel",
                "fuel",
                "forecourt",
                "clubcard",
            )
        ):
            return "uncategorized__tesco"

        return "OVERSIZED_MIXED_CLUSTER__5"

    if intent == "vouchers_promotions":
        if (
            "meal deal" in normalized
            or "3 for 2" in normalized
        ):
            return "uncategorized__meal"

        return "OVERSIZED_MIXED_CLUSTER__5"

    if intent in {
        "positive_feedback",
        "insufficient_context",
        "unknown",
    }:
        return "OVERSIZED_MIXED_CLUSTER__5"

    return "OVERSIZED_MIXED_CLUSTER__5"