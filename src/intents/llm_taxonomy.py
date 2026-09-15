from __future__ import annotations

CURATED_INTENTS: dict[str, tuple[str, bool]] = {
    "delivery_issue": (
        "Order, parcel, or delivery has not arrived, arrived late, arrived "
        "damaged/incomplete, or a delivery slot/booking problem.",
        True,
    ),
    "product_quality": (
        "A specific purchased product is faulty, damaged, expired, mouldy, "
        "spoiled, contains a foreign object, or otherwise defective.",
        True,
    ),
    "technical_problem": (
        "The website, app, or an in-store digital system (checkout, scan-"
        "as-you-shop, payment terminal) is broken, erroring, or not "
        "working.",
        True,
    ),
    "stock_availability": (
        "A question about whether a specific product is in stock, where "
        "to find it, or when it will be restocked -- not a complaint "
        "about a product already purchased.",
        True,
    ),
    "customer_service_complaint": (
        "A complaint about how staff behaved, or about the general store "
        "experience (queues, cleanliness, opening hours, till "
        "availability) -- NOT about a specific faulty product or a "
        "missing delivery.",
        True,
    ),
    "vouchers_promotions": (
        "An issue with a Clubcard, voucher, coupon, discount, or "
        "promotional offer not applying correctly.",
        True,
    ),
    "payment_account_issue": (
        "A problem with a charge, refund, payment method, online account "
        "access, or loyalty-point balance -- money or account access is "
        "the core issue, distinct from a specific product or delivery "
        "problem.",
        False,
    ),
    "pricing_complaint": (
        "A complaint that a price is too high, increased unexpectedly, or "
        "differs between channels/stores -- not about a voucher or "
        "promotion failing to apply.",
        False,
    ),
    "positive_feedback": (
        "The customer is thanking the brand, praising staff or service, "
        "or confirming something was already resolved, with NO new "
        "actionable issue in this message.",
        True,
    ),
    "insufficient_context": (
        "The message is too short or is clearly a reply to an earlier "
        "message we cannot see (a bare confirmation, a location name, "
        "'why?', a phone number with no explanation), so the real issue "
        "cannot be determined from this message alone.",
        False,
    ),
    "uncategorized__dm": (
        "The customer is confirming they have sent, are sending, or are "
        "asking about a direct message -- purely administrative, no "
        "support issue is described in THIS message.",
        True,
    ),
    "uncategorized__free": (
        "A question or complaint specifically about a gluten-free, dairy-"
        "free, or other allergen-free product's availability or "
        "labelling.",
        False,
    ),
    "uncategorized__meal": (
        "A question or complaint specifically about a Meal Deal "
        "promotion's contents, pricing, or eligibility.",
        False,
    ),
    "uncategorized__tesco": (
        "Store-level or account-level questions not covered by any "
        "category above (opening hours, fuel pricing, Clubcard/account "
        "mismatches, product-labelling/compliance questions). A "
        "genuinely mixed bucket -- prefer a more specific category above "
        "whenever one fits.",
        False,
    ),
    "OVERSIZED_MIXED_CLUSTER__5": (
        "Use ONLY when truly nothing else above fits and the message "
        "gives no actionable signal at all (e.g. pure social chit-chat, "
        "an isolated word with no discernible topic).",
        False,
    ),
}


def taxonomy_prompt_block() -> str:
    return "\n".join(
        f'- "{name}": {description}'
        for name, (description, _eligible) in CURATED_INTENTS.items()
    )


def is_auto_eligible(intent_name: str) -> bool:
    entry = CURATED_INTENTS.get(intent_name)
    return bool(entry and entry[1])