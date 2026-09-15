from __future__ import annotations

import re

from src.config import SETTINGS


HIGH_RISK_KEYWORDS = re.compile(
    r"\b("
    r"lawyer|legal action|sue|suing|fraud|unauthorized|hacked|stolen|scam|"
    r"report you|authorities|police|trading standards|ombudsman|regulator|"
    r"watchdog|court|"
    r"card details|bank details|account details|personal details"
    r")\b",
    re.IGNORECASE,
)


ANGER_KEYWORDS = re.compile(
    r"\b("
    r"furious|outraged|unacceptable|ridiculous|worst|terrible|"
    r"disgusted|never again|"
    r"disgusting|appalled|appalling|shocking|mortified|"
    r"inappropriate|offensive|disgrace|disgraceful|not good enough|"
    r"sort it out|sortitout|not happy"
    r")\b",
    re.IGNORECASE,
)


REPEAT_COMPLAINT_KEYWORDS = re.compile(
    r"\b("
    r"yet again|still not|still hasn'?t|again and again|over and over|"
    r"multiple times|repeatedly|keeps happening|this keeps|"
    r"not the first time|every time|every single time|"
    r"second time|third time|\d+(?:st|nd|rd|th)\s+time|"
    r"in a row|for years|never got|never received|no apology|"
    r"promised a call|nobody (?:has )?(?:responded|replied|called)"
    r")\b",
    re.IGNORECASE,
)


GENERIC_SUPPORT_KEYWORDS = re.compile(
    r"(?:"
    r"\bi\s+have\s+(?:a\s+)?(?:problem|issue)\b"
    r"|\bthere\s+is\s+(?:a\s+)?(?:problem|issue)\b"
    r"|\bsomething\s+went\s+wrong\b"
    r"|\bplease\s+help\s+me\b"
    r"|\bi\s+need\s+help\b"
    r"|\bhelp\s+with\s+something\b"
    r"|\b(?:an?\s+)?(?:issue|problem)\s+with\s+my\s+"
    r"(?:order|purchase)\b"
    r")",
    re.IGNORECASE,
)


SAFETY_KEYWORDS = re.compile(
    r"("
    r"\ballergic reaction\b|\ballergy\b|\ballergic\b|"
    r"\bbecame sick\b|\bmade me sick\b|\bmade us sick\b|"
    r"\bgot sick\b|\bfelt sick\b|\bfood poisoning\b|"
    r"\bpoisoned\b|\bpoison\b|"
    r"\bdangerous food\b|\bunsafe food\b|\bunsafe to eat\b|"
    r"\bdangerous to eat\b|\bcontaminated\b|\bcontamination\b|"
    r"\bforeign object\b|\bsomething dangerous\b|"
    r"\bout of date\b|\bpast its date\b|\bpast the date\b|"
    r"\bbest before\b|\buse by\b|\bexpired\b|\bexpiry\b|"
    r"\bmould\b|\bmouldy\b|\bmold\b|\bmoldy\b|"
    r"\brotten\b|\bspoiled\b|\bspoilt\b|\bgone off\b|"
    r"\bdefrosted\b|\brusty\b|\bblack bits\b|"
    r"\bhair\b|\bfly\b|\bmaggot\b|\bwasp\b|\bslug\b|"
    r"\bstone\b|\bgravel\b|\bbone\b|"
    r"\bplastic melted\b|\bnot oven proof\b|"
    r"\bhurt my gum\b|\bmystery blue material\b|"
    r"\bfloating around\b|\bleaked all over food\b|"
    r"(?:glass|metal|plastic|chemical|slug|insect|unblocker)"
    r".{0,40}(?:food|pack|packet|product|meal|bread|items)"
    r")",
    re.IGNORECASE,
)


PRODUCT_DEFECT_KEYWORDS = re.compile(
    r"\b("
    r"wonky|cracked|faulty|production problem|poor produce|"
    r"fruit and vegetables.*dreadful|dreadful|not enough|"
    r"missing from|where is the rest|no duck|no bacon|"
    r"wrong items|substitute list|not on my substitute list"
    r"|out of order|old crusty|dry soggy"
    r")\b",
    re.IGNORECASE,
)


FULFILMENT_FAILURE_KEYWORDS = re.compile(
    r"\b("
    r"shopping didn'?t arrive|didn'?t arrive|did not arrive|"
    r"late|cancel an order|cancelled order|not coming at all|"
    r"no show|wrong address|wrong items|showing as collected|"
    r"haven'?t even been in to pick|waiting to speak|"
    r"delivery driver|click and collect.*(?:free|cost|slot)|"
    r"order.*(?:not arrived|hasn'?t arrived|not fulfilled)"
    r")\b",
    re.IGNORECASE,
)


SERVICE_FAILURE_KEYWORDS = re.compile(
    r"\b("
    r"customer service training|staff member|lazy manager|"
    r"no apology|no reply|not heard back|refund is not enough|"
    r"refunded order|no idea why|vat receipt|"
    r"smells like|never worked|assistant button|"
    r"incompetence|know your rights|trading standards|"
    r"couldn'?t provide|how complicated"
    r")\b",
    re.IGNORECASE,
)


TECH_FAILURE_KEYWORDS = re.compile(
    r"\b("
    r"machine has yet to be fixed|still experiencing|"
    r"stopping me from shopping|can'?t order online|cannot order online|"
    r"scanner.*failed|touch screen.*unresponsive|"
    r"email suggests times.*(?:open|closed)|keeps going down"
    r")\b",
    re.IGNORECASE,
)


class EscalationPolicy:
    def decide(
        self,
        intent: str,
        intent_confidence: float,
        retrieval_evidence: list[dict],
        grounding_result: dict,
        customer_message: str,
        intent_auto_eligible: bool = True,
    ) -> dict:
        reasons = []

        if (
            intent == "unknown"
            or intent_confidence
            < SETTINGS.min_intent_confidence
        ):
            reasons.append(
                f"Intent confidence {intent_confidence:.2f} is below "
                f"{SETTINGS.min_intent_confidence} (or unrecognized)."
            )

        if not intent_auto_eligible:
            reasons.append(
                f"Intent '{intent}' is not a confidently-matched category "
                f"(uncategorized cluster) — never auto-handled regardless "
                f"of confidence."
            )

        top_similarity = (
            retrieval_evidence[0]["similarity"]
            if retrieval_evidence
            else 0.0
        )

        if not retrieval_evidence:
            reasons.append(
                "No similar historical cases were found."
            )
        elif (
            top_similarity
            < SETTINGS.min_retrieval_similarity
        ):
            reasons.append(
                f"Best retrieval similarity {top_similarity:.2f} is below "
                f"{SETTINGS.min_retrieval_similarity}."
            )

        if not grounding_result.get(
            "grounded",
            False,
        ):
            reasons.append(
                "Failed grounding validation: "
                + (
                    ", ".join(
                        grounding_result.get(
                            "issues",
                            [],
                        )
                    )
                    or "unspecified"
                )
            )

        high_risk = bool(
            HIGH_RISK_KEYWORDS.search(
                customer_message
            )
        )

        anger = bool(
            ANGER_KEYWORDS.search(
                customer_message
            )
        )

        repeat_complaint = bool(
            REPEAT_COMPLAINT_KEYWORDS.search(
                customer_message
            )
        )

        safety = bool(
            SAFETY_KEYWORDS.search(
                customer_message
            )
        )

        product_defect = bool(
            PRODUCT_DEFECT_KEYWORDS.search(
                customer_message
            )
        )

        fulfilment_failure = bool(
            FULFILMENT_FAILURE_KEYWORDS.search(
                customer_message
            )
        )

        service_failure = bool(
            SERVICE_FAILURE_KEYWORDS.search(
                customer_message
            )
        )

        tech_failure = bool(
            TECH_FAILURE_KEYWORDS.search(
                customer_message
            )
        )

        generic_support = bool(
            GENERIC_SUPPORT_KEYWORDS.search(
                customer_message
            )
        )

        if high_risk:
            reasons.append(
                "Message contains high-risk language (legal/fraud/security)."
            )

        if anger:
            reasons.append(
                "Message indicates a high-frustration/angry customer."
            )

        if repeat_complaint:
            reasons.append(
                "Message indicates a recurring/unresolved issue — escalated "
                "rather than repeating a historical response that may "
                "already have failed once."
            )

        if safety:
            reasons.append(
                "Message contains a safety-sensitive food/product concern "
                "that requires human review."
            )

        if product_defect:
            reasons.append(
                "Message describes a concrete product defect or missing item "
                "that should be reviewed by a human."
            )

        if fulfilment_failure:
            reasons.append(
                "Message describes a delivery/order fulfilment failure that "
                "should be reviewed by a human."
            )

        if service_failure:
            reasons.append(
                "Message describes a staff/service failure or unresolved "
                "complaint that should be reviewed by a human."
            )

        if tech_failure:
            reasons.append(
                "Message describes an unresolved technical failure that is "
                "blocking the customer."
            )

        if generic_support:
            reasons.append(
                "Message is too generic to determine the specific issue "
                "safely; additional intent-specific information is needed."
            )

        decision = (
            "ESCALATE"
            if reasons
            else "AUTO_HANDLE"
        )

        return {
            "decision": decision,
            "reason": (
                " ".join(reasons)
                if reasons
                else "All signals passed thresholds."
            ),
            "escalation_signals": {
                "intent_confidence": intent_confidence,
                "top_retrieval_similarity": top_similarity,
                "grounded": grounding_result.get(
                    "grounded",
                    False,
                ),
                "high_risk_language": high_risk,
                "anger_language": anger,
                "repeat_complaint": repeat_complaint,
                "safety_sensitive": safety,
                "product_defect": product_defect,
                "fulfilment_failure": fulfilment_failure,
                "service_failure": service_failure,
                "tech_failure": tech_failure,
                "generic_support_request": generic_support,
            },
        }


if __name__ == "__main__":
    result = EscalationPolicy().decide(
        "refund",
        0.91,
        [{"similarity": 0.78}],
        {"grounded": True, "issues": []},
        "My refund has not arrived",
    )

    print(result)