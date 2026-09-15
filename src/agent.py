from __future__ import annotations

import argparse
import json

from src.config import get_logger
from src.data.clean import clean_text
from src.escalation.policy import EscalationPolicy
from src.generation.generator import ResponseGenerator
from src.generation.grounding import GroundingValidator
from src.intents import llm_taxonomy
from src.intents.canonical import canonicalize_intent
from src.intents.classifier import IntentClassifier
from src.retrieval.retriever import Retriever

logger = get_logger(__name__)


class SupportAgent:
    def __init__(self) -> None:
        logger.info("Loading agent components...")
        self.intent_classifier = IntentClassifier()
        self.retriever = Retriever()
        self.generator = ResponseGenerator()
        self.grounding_validator = GroundingValidator()
        self.escalation_policy = EscalationPolicy()

    def _is_auto_eligible(self, intent_name: str, intent_source: str) -> bool:
        if intent_source == "llm":
            return llm_taxonomy.is_auto_eligible(intent_name)
        return self.intent_classifier.is_auto_eligible(intent_name)

    def handle(
        self,
        message: str,
        exclude_conversation_id: str | None = None,
        exclude_message_text: str | None = None,
        top_k: int = 5,
    ) -> dict:
        cleaned = clean_text(message)

        local_intent_result = self.intent_classifier.predict(cleaned)

        evidence = self.retriever.search(
            cleaned,
            top_k=top_k,
            exclude_conversation_id=exclude_conversation_id,
            exclude_message_text=exclude_message_text,
        )
        generation_result = self.generator.generate(cleaned, evidence)
        grounding_result = self.grounding_validator.validate(
            generation_result.get("reply", ""), evidence
        )

        llm_intent = generation_result.get("intent")
        if llm_intent:
            intent_name = canonicalize_intent(llm_intent, cleaned)
            intent_confidence = float(
                generation_result.get("intent_confidence") or 0.0
            )
            intent_source = "llm"
        else:
            intent_name = canonicalize_intent(
                local_intent_result["intent"], cleaned
            )
            intent_confidence = local_intent_result["confidence"]
            intent_source = "local_fallback"

        escalation_result = self.escalation_policy.decide(
            intent=intent_name,
            intent_confidence=intent_confidence,
            retrieval_evidence=evidence,
            grounding_result=grounding_result,
            customer_message=message,
            intent_auto_eligible=self._is_auto_eligible(
                intent_name, intent_source
            ),
        )

        return {
            "customer_message": message,
            "intent": intent_name,
            "intent_confidence": intent_confidence,
            "intent_source": intent_source,
            "intent_local_fallback": local_intent_result["intent"],
            "retrieved_evidence": evidence,
            "generated_reply": generation_result.get("reply", ""),
            "generation_self_reported": {
                "grounded": generation_result.get("grounded"),
                "confidence": generation_result.get("confidence"),
                "reasoning_summary": generation_result.get("reasoning_summary"),
            },
            "grounding_check": grounding_result,
            "decision": escalation_result["decision"],
            "escalation_reason": (
                escalation_result["reason"]
                if escalation_result["decision"] == "ESCALATE"
                else None
            ),
            "escalation_signals": escalation_result["escalation_signals"],
        }


def print_human_readable(result: dict) -> None:
    print(f"Intent: {result['intent']} (source: {result['intent_source']})")
    print(f"Intent confidence: {result['intent_confidence']:.2f}\n")
    print(f"Decision: {result['decision']}\n")
    if result["decision"] == "ESCALATE":
        print(f"Reason:\n{result['escalation_reason']}\n")
    print(f"Suggested reply:\n{result['generated_reply']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Customer Support Agent")
    parser.add_argument("--message", required=True)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--exclude-conversation-id",
        default=None,
        help="Internal: used during evaluation",
    )
    parser.add_argument(
        "--exclude-message-text",
        default=None,
        help="Internal: used during evaluation",
    )
    args = parser.parse_args()

    agent = SupportAgent()
    result = agent.handle(
        args.message,
        exclude_conversation_id=args.exclude_conversation_id,
        exclude_message_text=args.exclude_message_text,
    )
    print(
        json.dumps(result, indent=2)
        if args.json
        else print_human_readable(result)
    )


if __name__ == "__main__":
    main()