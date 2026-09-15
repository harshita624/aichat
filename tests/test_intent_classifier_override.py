from __future__ import annotations

import pytest

from src.intents.classifier import IntentClassifier


def make_classifier(intent_metadata: dict) -> IntentClassifier:
    clf = object.__new__(IntentClassifier)
    clf.vectorizer = None
    clf.kmeans = None
    clf.cluster_to_name = {}
    clf.intent_metadata = intent_metadata
    return clf


def stub_semantic(
    clf: IntentClassifier,
    intent: str,
    confidence: float = 0.2,
    raw_similarity: float = 0.3,
    cluster_id: int = 0,
) -> None:
    clf._semantic_predict = lambda text: {  # type: ignore[method-assign]
        "intent": intent,
        "confidence": confidence,
        "raw_similarity": raw_similarity,
        "cluster_id": cluster_id,
    }


OVERSIZED_META = {
    "name": "OVERSIZED_MIXED_CLUSTER__5",
    "matched_candidate_label": False,
    "frequency_pct": 35.0,
}
TECHNICAL_META = {
    "name": "technical_problem",
    "matched_candidate_label": True,
    "frequency_pct": 12.0,
}
DELIVERY_META = {
    "name": "delivery_issue",
    "matched_candidate_label": True,
    "frequency_pct": 9.0,
}
PAYMENT_META = {
    "name": "payment_issue",
    "matched_candidate_label": True,
    "frequency_pct": 8.0,
}
ACCOUNT_META = {
    "name": "account_issue",
    "matched_candidate_label": True,
    "frequency_pct": 7.0,
}
VOUCHERS_META = {
    "name": "vouchers_promotions",
    "matched_candidate_label": True,
    "frequency_pct": 6.0,
}
PRODUCT_META = {
    "name": "product_quality",
    "matched_candidate_label": True,
    "frequency_pct": 10.0,
}
PRICING_META = {
    "name": "pricing_complaint",
    "matched_candidate_label": True,
    "frequency_pct": 5.0,
}


@pytest.mark.parametrize(
    "message, semantic_guess, semantic_meta, expected_intent",
    [
        (
            "My parcel is really late and still hasn't arrived",
            "technical_problem",
            TECHNICAL_META,
            "delivery_issue",
        ),
        (
            "My card was charged twice and the order failed",
            "technical_problem",
            TECHNICAL_META,
            "payment_issue",
        ),
        (
            "I can't log in, it keeps saying my password is wrong",
            "OVERSIZED_MIXED_CLUSTER__5",
            OVERSIZED_META,
            "account_issue",
        ),
        (
            "My clubcard discount didn't come off at checkout",
            "technical_problem",
            TECHNICAL_META,
            "vouchers_promotions",
        ),
        (
            "The item I received was damaged and broken",
            "OVERSIZED_MIXED_CLUSTER__5",
            OVERSIZED_META,
            "product_quality",
        ),
        (
            "Why is this so much more expensive, it feels overpriced now",
            "OVERSIZED_MIXED_CLUSTER__5",
            OVERSIZED_META,
            "pricing_complaint",
        ),
    ],
)
def test_lexical_override_fixes_reported_bugs(
    message,
    semantic_guess,
    semantic_meta,
    expected_intent,
):
    metadata = {
        semantic_meta["name"]: semantic_meta,
        "delivery_issue": DELIVERY_META,
        "payment_issue": PAYMENT_META,
        "account_issue": ACCOUNT_META,
        "vouchers_promotions": VOUCHERS_META,
        "product_quality": PRODUCT_META,
        "pricing_complaint": PRICING_META,
    }

    clf = make_classifier(metadata)
    stub_semantic(clf, semantic_guess)

    result = clf.predict(message)

    assert result["intent"] == expected_intent
    assert result["method"] == "lexical_override"
    assert 0.0 <= result["confidence"] <= 1.0


def test_agreement_keeps_label_and_boosts_confidence():
    clf = make_classifier({
        "technical_problem": TECHNICAL_META
    })

    stub_semantic(
        clf,
        "technical_problem",
        confidence=0.1,
    )

    result = clf.predict(
        "The app crashes with an error every time I open it"
    )

    assert result["intent"] == "technical_problem"
    assert result["method"] == "semantic_lexical_agree"
    assert result["confidence"] >= 0.1


def test_no_clear_lexical_winner_trusts_semantic_prediction():
    meta = {
        "name": "customer_service_complaint",
        "matched_candidate_label": True,
        "frequency_pct": 4.0,
    }

    clf = make_classifier({
        "customer_service_complaint": meta
    })

    stub_semantic(
        clf,
        "customer_service_complaint",
        confidence=0.15,
    )

    result = clf.predict(
        "There is a problem with my order and I am unhappy with the service"
    )

    assert result["intent"] == "customer_service_complaint"
    assert result["method"] == "semantic"
    assert result["lexical_category"] is None


def test_ambiguous_multi_topic_message_does_not_override():
    meta = {
        "name": "customer_service_complaint",
        "matched_candidate_label": True,
        "frequency_pct": 4.0,
    }

    clf = make_classifier({
        "customer_service_complaint": meta
    })

    stub_semantic(
        clf,
        "customer_service_complaint",
        confidence=0.1,
    )

    result = clf.predict(
        "My card payment failed and the app keeps showing an error"
    )

    assert result["lexical_category"] is None
    assert result["method"] == "semantic"


def test_specific_semantic_match_is_not_second_guessed_by_stray_keyword():
    metadata = {
        "cancellation": {
            "name": "cancellation",
            "matched_candidate_label": True,
            "frequency_pct": 6.0,
        },
        "technical_problem": TECHNICAL_META,
    }

    clf = make_classifier(metadata)

    stub_semantic(
        clf,
        "cancellation",
        confidence=0.3,
    )

    result = clf.predict(
        "Please cancel my subscription, I mentioned this on the app once before"
    )

    assert result["intent"] == "cancellation"