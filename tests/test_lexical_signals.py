from __future__ import annotations

from src.intents.lexical_signals import (
    LEXICAL_SIGNALS,
    base_taxonomy_category,
    best_category,
    confidence_from_score,
    score_categories,
)


def test_generic_words_alone_never_score():
    for phrases in LEXICAL_SIGNALS.values():
        for phrase in phrases:
            assert phrase not in {"order", "issue", "problem", "working"}


def test_distinctive_keywords_beat_generic_context():
    category, score, scores = best_category(
        "There's a problem with my order -- it's late and never arrived"
    )
    assert category == "delivery_issue"
    assert scores["technical_problem"] == 0.0


def test_payment_signals():
    category, _, _ = best_category(
        "I was charged for an item and my card billing looks wrong"
    )
    assert category == "payment_issue"


def test_account_signals_multiword_phrase():
    category, score, _ = best_category(
        "I can't sign in, my password won't work"
    )
    assert category == "account_issue"
    assert score >= 1.5


def test_vouchers_signals():
    category, _, _ = best_category(
        "The voucher and clubcard points didn't apply"
    )
    assert category == "vouchers_promotions"


def test_product_quality_signals():
    category, _, _ = best_category(
        "The bread arrived mouldy and the packet was damaged"
    )
    assert category == "product_quality"


def test_pricing_signals():
    category, _, _ = best_category(
        "This is overpriced, the shelf price was way more expensive"
    )
    assert category == "pricing_complaint"


def test_word_boundaries_avoid_false_positives():
    scores = score_categories(
        "I bought a costume and a cardigan"
    )
    assert scores["pricing_complaint"] == 0.0
    assert scores["payment_issue"] == 0.0


def test_tied_categories_yield_no_winner():
    category, score, _ = best_category(
        "My card payment failed and the app keeps showing an error"
    )
    assert category is None
    assert score == 0.0


def test_confidence_from_score_is_monotonic_and_bounded():
    values = [
        confidence_from_score(s)
        for s in (0, 1, 2, 3, 5, 10)
    ]

    assert values[0] == 0.0
    assert all(0.0 <= v < 1.0 for v in values)
    assert values == sorted(values)


def test_base_taxonomy_category_strips_disambiguation_suffix():
    assert base_taxonomy_category("payment_issue") == "payment_issue"
    assert base_taxonomy_category("payment_issue__3") == "payment_issue"


def test_base_taxonomy_category_rejects_unrelated_names():
    assert base_taxonomy_category("OVERSIZED_MIXED_CLUSTER__5") is None
    assert base_taxonomy_category("uncategorized__foo") is None
    assert base_taxonomy_category("not_a_real_intent") is None