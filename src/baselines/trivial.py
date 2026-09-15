from __future__ import annotations

import json

from src.config import DATA_PROCESSED

TAXONOMY_PATH = DATA_PROCESSED / "taxonomy.json"

GENERIC_RESPONSE = (
    "Thank you for reaching out. A member of our support team will "
    "review your message and follow up shortly."
)


class TrivialBaseline:
    def __init__(self) -> None:
        self.majority_intent = "unknown"

        if TAXONOMY_PATH.exists():
            taxonomy = json.loads(
                TAXONOMY_PATH.read_text()
            )

            if taxonomy["intents"]:
                self.majority_intent = max(
                    taxonomy["intents"],
                    key=lambda i: i["size"],
                )["name"]

    def predict(self, message: str) -> dict:
        return {
            "intent": self.majority_intent,
            "intent_confidence": 0.0,
            "reply": GENERIC_RESPONSE,
            "grounded": False,
            "decision": "ESCALATE",
            "escalation_reason": (
                "Trivial baseline always escalates by design."
            ),
        }


if __name__ == "__main__":
    print(
        json.dumps(
            TrivialBaseline().predict(
                "My refund has not arrived"
            ),
            indent=2,
        )
    )