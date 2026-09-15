from __future__ import annotations

from src.generation.prompts import SYSTEM_PROMPT


def test_system_prompt_forbids_requesting_credentials():
    lowered = SYSTEM_PROMPT.lower()

    for term in ("password", "cvv", "card number"):
        assert term in lowered


def test_system_prompt_warns_against_copying_unsafe_historical_requests():
    assert "do not copy that request" in SYSTEM_PROMPT.lower()