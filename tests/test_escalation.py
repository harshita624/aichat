import pytest

from src.config import SETTINGS
from src.escalation.policy import EscalationPolicy

PINNED_MIN_INTENT_CONFIDENCE = 0.5
PINNED_MIN_RETRIEVAL_SIMILARITY = 0.45


@pytest.fixture(autouse=True)
def _pin_thresholds(monkeypatch):
    monkeypatch.setattr(
        SETTINGS,
        "min_intent_confidence",
        PINNED_MIN_INTENT_CONFIDENCE,
    )
    monkeypatch.setattr(
        SETTINGS,
        "min_retrieval_similarity",
        PINNED_MIN_RETRIEVAL_SIMILARITY,
    )


def test_escalates_on_low_intent_confidence():
    result = EscalationPolicy().decide(
        "refund",
        0.10,
        [{"similarity": 0.90}],
        {"grounded": True, "issues": []},
        "ok",
    )

    assert result["decision"] == "ESCALATE"
    assert "confidence" in result["reason"].lower()


def test_escalates_on_no_evidence():
    result = EscalationPolicy().decide(
        "refund",
        0.95,
        [],
        {"grounded": True, "issues": []},
        "My refund has not arrived",
    )

    assert result["decision"] == "ESCALATE"


def test_escalates_on_grounding_failure():
    result = EscalationPolicy().decide(
        "refund",
        0.95,
        [{"similarity": 0.90}],
        {"grounded": False, "issues": ["Unsupported number: 500"]},
        "My refund has not arrived",
    )

    assert result["decision"] == "ESCALATE"
    assert "grounding" in result["reason"].lower()


def test_escalates_on_high_risk_language():
    result = EscalationPolicy().decide(
        "refund",
        0.95,
        [{"similarity": 0.90}],
        {"grounded": True, "issues": []},
        "I'm getting my lawyer involved, this is fraud",
    )

    assert result["decision"] == "ESCALATE"
    assert result["escalation_signals"]["high_risk_language"] is True


def test_escalates_on_repeat_complaint_language():
    result = EscalationPolicy().decide(
        "delivery_issue",
        0.95,
        [{"similarity": 0.90}],
        {"grounded": True, "issues": []},
        "food delivery late yet again, I am yet to receive one within my chosen time!",
    )

    assert result["decision"] == "ESCALATE"
    assert result["escalation_signals"]["repeat_complaint"] is True


def test_auto_handles_when_all_signals_pass():
    result = EscalationPolicy().decide(
        "refund",
        0.95,
        [{"similarity": 0.90}],
        {"grounded": True, "issues": []},
        "When will my refund arrive?",
    )

    assert result["decision"] == "AUTO_HANDLE"