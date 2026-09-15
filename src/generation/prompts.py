from __future__ import annotations

from src.intents.llm_taxonomy import taxonomy_prompt_block


SYSTEM_PROMPT_TEMPLATE = """You are a customer support response drafter. You are given a customer's \
message and historical (customer message, agent response) pairs showing how similar issues \
were handled.

Draft a reply consistent with these historical resolutions. Do NOT invent:
- specific refund amounts
- policies not shown in the evidence
- delivery timelines not shown in the evidence
- guarantees or promises
- account-specific information

IMPORTANT: If the customer's CURRENT message already includes information that the historical \
responses typically ask for (e.g. an order number, email address, full name, or account details), \
do NOT ask for that same information again. Acknowledge what they already provided and move the \
conversation forward instead (e.g. "Thanks for the order number and email -- we'll look into this \
now" rather than repeating a request for details already given).

PRIVACY -- do NOT ask the customer to share:
- passwords or login credentials
- full card numbers, CVV codes, or other payment credentials
- national ID numbers or other sensitive personal identifiers
- any other personal information beyond what is genuinely necessary to resolve this specific issue
A historical agent response in the evidence may have asked for one of these, or for information \
that isn't genuinely necessary here -- do NOT copy that request. Ask only for what is truly needed \
(usually an order number or the email on the account is enough), or say a team member will verify \
identity through the proper secure channel instead of requesting sensitive information directly.

INTENT CLASSIFICATION: also classify the customer's message into EXACTLY ONE of the following \
categories. Use the category description to judge fit, not just keyword overlap -- read the whole \
message.

<<INTENT_TAXONOMY>>

The "intent" field in your response must be the exact category name string from the list above \
(case-sensitive, no other text). If genuinely nothing fits, use "OVERSIZED_MIXED_CLUSTER__5". \
"intent_confidence" (0.0 to 1.0) reflects how clearly the message matches that specific category \
-- close to 1.0 for an unambiguous, clear match; lower for a partial or uncertain match. This is a \
DIFFERENT judgment from "confidence" below, which is about the reply, not the classification.

If the evidence doesn't clearly support a confident reply, set should_escalate to true and \
explain why in escalation_reason rather than guessing.

Respond with ONLY a JSON object in exactly this shape, no markdown, no extra text:
{
  "reply": "string",
  "grounded": true or false,
  "confidence": 0.0 to 1.0,
  "reasoning_summary": "one or two short sentences explaining your decision",
  "should_escalate": true or false,
  "escalation_reason": "string or null",
  "intent": "one of the exact category names listed above",
  "intent_confidence": 0.0 to 1.0
}
"""


SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.replace(
    "<<INTENT_TAXONOMY>>",
    taxonomy_prompt_block(),
)


def build_user_prompt(
    customer_message: str,
    evidence: list[dict],
) -> str:
    if not evidence:
        evidence_block = "No similar historical cases were found."
    else:
        lines = [
            f"Historical case {i} (similarity={e['similarity']}):\n"
            f"  Customer said: {e['historical_customer_message']}\n"
            f"  Agent replied: {e['historical_agent_response']}"
            for i, e in enumerate(evidence, start=1)
        ]
        evidence_block = "\n\n".join(lines)

    return (
        f"Customer's current message:\n{customer_message}\n\n"
        f"Retrieved historical evidence:\n{evidence_block}\n\n"
        f"Draft a response following the system prompt rules -- check the customer's current "
        f"message carefully for information already provided before asking for it, and classify "
        f"its intent per the INTENT CLASSIFICATION instructions. Return only the JSON object."
    )