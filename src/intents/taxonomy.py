from __future__ import annotations

import json

from src.config import DATA_PROCESSED, get_logger

logger = get_logger(__name__)

INTENT_CLUSTERS_PATH = DATA_PROCESSED / "intent_clusters.json"
TAXONOMY_PATH = DATA_PROCESSED / "taxonomy.json"

LARGE_CLUSTER_THRESHOLD_PCT = 30.0
MATCH_WINDOW = 5

CANDIDATE_LABELS = {
    "refund": ["refund", "money back", "reimburse", "charged"],
    "payment_issue": [
        "payment",
        "charge",
        "billing",
        "invoice",
        "card declined",
    ],
    "account_issue": [
        "account",
        "login",
        "password",
        "locked",
        "sign in",
    ],
    "delivery_issue": [
        "delivery",
        "home delivery",
        "shipping",
        "package",
        "tracking",
        "arrived",
        "shipped",
        "slot",
        "driver",
    ],
    "cancellation": [
        "cancel",
        "cancellation",
        "unsubscribe",
    ],
    "technical_problem": [
        "error",
        "bug",
        "crash",
        "not working",
        "broken",
        "website",
        "app",
    ],
    "verification": [
        "verify",
        "verification",
        "confirm identity",
        "otp",
    ],
    "pricing_complaint": [
        "price",
        "half price",
        "increase",
        "expensive",
        "cost",
        "overpriced",
    ],
    "vouchers_promotions": [
        "voucher",
        "clubcard",
        "points",
        "coupon",
        "discount",
        "promotion",
        "offer",
    ],
    "customer_service_complaint": [
        "customer service",
        "self service",
        "poor service",
        "staff",
        "rude",
    ],
    "product_quality": [
        "out of date",
        "mouldy",
        "mould",
        "splinter",
        "quality",
        "bought",
    ],
    "stock_availability": [
        "out of stock",
        "not available",
        "sold out",
        "in stock",
        "stock",
    ],
    "packaging_environment": [
        "plastic",
        "packaging",
        "recyclable",
        "bags",
    ],
    "store_experience": [
        "queue",
        "till",
        "local store",
        "road",
    ],
    "complaint": [
        "worst",
        "terrible",
        "angry",
        "disappointed",
        "unacceptable",
    ],
}

DEFAULT_ESCALATION_NOTE = (
    "Escalate if retrieval evidence is weak or the response cannot be grounded."
)


def name_cluster(top_keywords: list[str]) -> tuple[str, bool]:
    window = top_keywords[:MATCH_WINDOW]
    keyword_text = " ".join(window).lower()

    best_label, best_score = None, 0

    for label, hints in CANDIDATE_LABELS.items():
        score = sum(
            1
            for hint in hints
            if hint in keyword_text
        )

        if score > best_score:
            best_label, best_score = label, score

    if best_label is not None:
        return best_label, True

    fallback_keyword = (
        top_keywords[0]
        if top_keywords
        else "unknown"
    )

    return f"uncategorized__{fallback_keyword}", False


def build_taxonomy(clusters_summary: dict) -> dict:
    intents = []
    used_names: set[str] = set()

    for cluster in clusters_summary["clusters"]:
        if cluster["frequency_pct"] >= LARGE_CLUSTER_THRESHOLD_PCT:
            name = (
                f"OVERSIZED_MIXED_CLUSTER__"
                f"{cluster['cluster_id']}"
            )
            matched = False
            auto_eligible = False
        else:
            name, matched = name_cluster(
                cluster["top_keywords"]
            )
            auto_eligible = (
                matched
                and cluster["frequency_pct"] >= 2.0
            )

            if name in used_names:
                original_name = name
                name = (
                    f"{name}__"
                    f"{cluster['cluster_id']}"
                )

                logger.warning(
                    f"Cluster {cluster['cluster_id']} matched the same label "
                    f"('{original_name}') as an earlier, different cluster -- "
                    f"renamed to '{name}' to keep both individually addressable. "
                    f"Consider hand-reviewing whether these two clusters actually "
                    f"represent the same real-world intent (in which case merging "
                    f"them at the clustering level would be cleaner) or genuinely "
                    f"different ones that happen to share vocabulary."
                )

        used_names.add(name)

        intents.append(
            {
                "cluster_id": cluster["cluster_id"],
                "name": name,
                "description": (
                    "Data-driven intent from keywords: "
                    f"{', '.join(cluster['top_keywords'][:5])}"
                ),
                "representative_examples": cluster[
                    "example_messages"
                ],
                "frequency_pct": cluster["frequency_pct"],
                "size": cluster["size"],
                "auto_handle_eligible": auto_eligible,
                "escalation_conditions": DEFAULT_ESCALATION_NOTE,
                "matched_candidate_label": matched,
            }
        )

    return {
        "intents": intents,
        "n_intents": len(intents),
        "total_messages_used": clusters_summary[
            "total_messages_used"
        ],
    }


def main() -> None:
    if not INTENT_CLUSTERS_PATH.exists():
        print(
            "No cluster summary found. "
            "Run python -m src.intents.discover first."
        )
        return

    clusters_summary = json.loads(
        INTENT_CLUSTERS_PATH.read_text(
            encoding="utf-8"
        )
    )

    taxonomy = build_taxonomy(clusters_summary)

    TAXONOMY_PATH.write_text(
        json.dumps(taxonomy, indent=2),
        encoding="utf-8",
    )

    print(
        f"Built taxonomy with "
        f"{taxonomy['n_intents']} intents:\n"
    )

    for intent in taxonomy["intents"]:
        if intent["name"].startswith(
            "OVERSIZED_MIXED_CLUSTER"
        ):
            tag = "!! OVERSIZED CATCH-ALL !!"
        elif intent["matched_candidate_label"]:
            tag = "(matched candidate)"
        else:
            tag = "(NEEDS REVIEW)"

        print(
            f"  {intent['name']:32s} "
            f"{tag:26s} "
            f"freq={intent['frequency_pct']}% "
            f"auto={intent['auto_handle_eligible']}"
        )

    oversized = [
        intent
        for intent in taxonomy["intents"]
        if intent["name"].startswith(
            "OVERSIZED_MIXED_CLUSTER"
        )
    ]

    if oversized:
        print(
            f"\nWARNING: {len(oversized)} cluster(s) exceed "
            f"{LARGE_CLUSTER_THRESHOLD_PCT}% of all messages and were "
            f"NOT given a specific label -- inspect their "
            f"representative_examples in taxonomy.json. This usually "
            f"means the current number of clusters can't separate this "
            f"data well; consider whether the catch-all is genuinely "
            f"one broad 'general inquiry' intent (a legitimate, "
            f"reportable finding) versus needing different preprocessing."
        )

    print(
        f"\nSaved to {TAXONOMY_PATH}. "
        "Hand-edit to rename intents or adjust auto_handle_eligible."
    )


if __name__ == "__main__":
    main()