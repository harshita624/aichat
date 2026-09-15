from unittest.mock import MagicMock

from src.agent import SupportAgent


def _make_agent_with_mocks(
    intent_confidence=0.9,
    similarity=0.85,
    grounded=True,
):
    agent = SupportAgent.__new__(SupportAgent)

    agent.intent_classifier = MagicMock()
    agent.intent_classifier.predict.return_value = {
        "intent": "delivery_issue",
        "confidence": intent_confidence,
    }

    agent.retriever = MagicMock()
    agent.retriever.search.return_value = [
        {
            "conversation_id": "brand_123",
            "historical_customer_message": "Where is my refund",
            "historical_agent_response": "Refunds take 5-7 business days.",
            "similarity": similarity,
        }
    ]

    agent.generator = MagicMock()
    agent.generator.generate.return_value = {
        "reply": "Your refund should arrive within 5-7 business days.",
        "grounded": True,
        "confidence": 0.8,
        "reasoning_summary": "Based on similar historical case.",
        "should_escalate": False,
        "escalation_reason": None,
    }

    agent.grounding_validator = MagicMock()
    agent.grounding_validator.validate.return_value = {
        "grounded": grounded,
        "issues": [] if grounded else ["Unsupported number: 999"],
        "unsupported_claims": [],
    }

    from src.escalation.policy import EscalationPolicy

    agent.escalation_policy = EscalationPolicy()

    return agent


def test_agent_auto_handles_with_strong_signals():
    result = _make_agent_with_mocks(
        0.9,
        0.85,
        True,
    ).handle("My refund has not arrived")

    assert result["decision"] == "AUTO_HANDLE"
    assert result["intent"] == "delivery_issue"
    assert result["generated_reply"]


def test_agent_escalates_on_grounding_failure():
    result = _make_agent_with_mocks(
        0.9,
        0.85,
        False,
    ).handle("My refund has not arrived")

    assert result["decision"] == "ESCALATE"
    assert "grounding" in result["escalation_reason"].lower()


def test_agent_escalates_on_low_intent_confidence():
    result = _make_agent_with_mocks(
        0.02,
        0.85,
        True,
    ).handle("something vague")

    assert result["decision"] == "ESCALATE"


def test_agent_passes_exclude_conversation_id_to_retriever():
    agent = _make_agent_with_mocks()

    agent.handle(
        "test message",
        exclude_conversation_id="brand_999",
    )

    _, kwargs = agent.retriever.search.call_args

    assert kwargs.get("exclude_conversation_id") == "brand_999"


def test_agent_output_has_required_keys():
    result = _make_agent_with_mocks().handle(
        "My refund has not arrived"
    )

    required = {
        "customer_message",
        "intent",
        "intent_confidence",
        "retrieved_evidence",
        "generated_reply",
        "generation_self_reported",
        "grounding_check",
        "decision",
        "escalation_reason",
        "escalation_signals",
    }

    assert required.issubset(result.keys())


def test_generator_fallback_when_no_llm_configured(monkeypatch):
    from src.config import SETTINGS
    from src.generation.generator import ResponseGenerator

    monkeypatch.setattr(
        SETTINGS,
        "llm_provider",
        "none",
    )

    evidence = [
        {
            "historical_customer_message": "Where is my refund",
            "historical_agent_response": "Refunds take 5-7 business days.",
            "similarity": 0.85,
        }
    ]

    result = ResponseGenerator().generate(
        "My refund has not arrived",
        evidence,
    )

    assert result["reply"] != evidence[0]["historical_agent_response"]
    assert result["reply"]
    assert result["should_escalate"] is False


def test_generator_fallback_escalates_with_no_evidence(monkeypatch):
    from src.config import SETTINGS
    from src.generation.generator import ResponseGenerator

    monkeypatch.setattr(
        SETTINGS,
        "llm_provider",
        "none",
    )

    result = ResponseGenerator().generate(
        "Some message",
        evidence=[],
    )

    assert result["should_escalate"] is True
    assert result["grounded"] is False