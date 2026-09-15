from __future__ import annotations

import re


NUMBER_PATTERN = re.compile(r"\$?\d+(?:\.\d+)?%?")

DATE_PATTERN = re.compile(
    r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b"
    r"|\b(?:january|february|march|april|may|june|july|august|september|october|november|"
    r"december)\s+\d{1,2}(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)

TIMELINE_PATTERN = re.compile(
    r"\b\d+\s*(?:-\s*\d+\s*)?(?:business\s+)?(?:day|days|week|weeks|hour|hours|month|months)\b",
    re.IGNORECASE,
)

PROMISE_PATTERN = re.compile(
    r"\b(guarantee|guaranteed|promise|will definitely|100%|always|never fail)\b",
    re.IGNORECASE,
)


def _extract_claims(
    text: str,
) -> dict[str, list[tuple[str, tuple[int, int]]]]:
    timeline_matches = [
        (m.group(0), m.span())
        for m in TIMELINE_PATTERN.finditer(text)
    ]

    number_matches_raw = [
        (m.group(0), m.span())
        for m in NUMBER_PATTERN.finditer(text)
    ]

    number_matches = [
        (value, span)
        for value, span in number_matches_raw
        if not any(
            span[0] >= timeline_span[0] and span[1] <= timeline_span[1]
            for _, timeline_span in timeline_matches
        )
    ]

    date_matches = [
        (m.group(0), m.span())
        for m in DATE_PATTERN.finditer(text)
    ]

    promise_matches = [
        (m.group(0), m.span())
        for m in PROMISE_PATTERN.finditer(text)
    ]

    return {
        "numbers": number_matches,
        "dates": date_matches,
        "timelines": timeline_matches,
        "promises": promise_matches,
    }


def _claim_supported(claim: str, evidence_text: str) -> bool:
    return claim.lower().strip() in evidence_text.lower()


class GroundingValidator:
    def validate(
        self,
        generated_reply: str,
        evidence: list[dict],
    ) -> dict:
        evidence_text = " ".join(
            e.get("historical_agent_response", "")
            + " "
            + e.get("historical_customer_message", "")
            for e in evidence
        )

        claims = _extract_claims(generated_reply)
        unsupported = []

        for claim_type in (
            "numbers",
            "dates",
            "timelines",
            "promises",
        ):
            for claim_text, _span in claims[claim_type]:
                if not _claim_supported(claim_text, evidence_text):
                    unsupported.append({
                        "type": claim_type[:-1],
                        "claim": claim_text,
                    })

        issues = [
            f"Unsupported {item['type']}: '{item['claim']}'"
            for item in unsupported
        ]

        return {
            "grounded": len(unsupported) == 0,
            "issues": issues,
            "unsupported_claims": unsupported,
        }


if __name__ == "__main__":
    evidence = [
        {
            "historical_agent_response": (
                "If you have a Brand Guarantee voucher applied, can you "
                "remove it & try completing the order again please?"
            ),
            "historical_customer_message": (
                "I'm trying to order food and your website isn't working."
            ),
        }
    ]

    result = GroundingValidator().validate(
        "Clearing your browser cache may help. If you have a Brand Guarantee "
        "voucher applied, please try removing it.",
        evidence,
    )

    print(result)

    assert result["grounded"] is True, (
        f"Expected grounded=True, got: {result}"
    )

    print("Regression check passed: 'may' and 'Guarantee' no longer false-flagged.")