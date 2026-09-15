from __future__ import annotations

import collections
import json
import os
import re
import threading
import time

import requests

from src.config import SETTINGS, get_logger
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.intents import llm_taxonomy


logger = get_logger(__name__)


REQUIRED_KEYS = {
    "reply",
    "grounded",
    "confidence",
    "reasoning_summary",
    "should_escalate",
    "escalation_reason",
}


MAX_RATE_LIMIT_RETRIES = 2

MAX_PROVIDER_WAIT_SECONDS = 30.0

_RETRY_SECONDS_PATTERN = re.compile(
    r"try again in ([\d.]+)\s*(?:s|seconds?)",
    re.IGNORECASE,
)


LLM_TPM_BUDGET = int(
    os.getenv("LLM_TPM_BUDGET", "6000")
)

_token_usage_lock = threading.Lock()

_recent_token_usage: collections.deque = collections.deque()


def _estimate_tokens(*texts: str) -> int:
    total_chars = sum(len(text or "") for text in texts)

    return max(1, (total_chars + 3) // 4)


def _wait_for_rate_budget(estimated_tokens: int) -> float:
    while True:
        with _token_usage_lock:
            now = time.monotonic()

            while (
                _recent_token_usage
                and now - _recent_token_usage[0][0] > 60
            ):
                _recent_token_usage.popleft()

            used = sum(
                tokens
                for _, tokens in _recent_token_usage
            )

            if used + estimated_tokens <= LLM_TPM_BUDGET:
                entry_time = now

                _recent_token_usage.append(
                    [entry_time, estimated_tokens]
                )

                return entry_time

            oldest_time = (
                _recent_token_usage[0][0]
                if _recent_token_usage
                else now
            )

            wait_s = max(
                0.0,
                60.0 - (now - oldest_time),
            ) + 0.25

            if wait_s > MAX_PROVIDER_WAIT_SECONDS:
                raise RuntimeError(
                    f"Local token-rate pacing would require waiting "
                    f"{wait_s:.1f}s (budget {LLM_TPM_BUDGET}/min); "
                    f"skipping long wait and using fallback."
                )

            logger.info(
                "Pacing LLM calls: "
                f"{used}+{estimated_tokens} tokens would exceed "
                f"the {LLM_TPM_BUDGET}/min budget; "
                f"waiting {wait_s:.1f}s before this call."
            )

        time.sleep(wait_s)


def _record_actual_tokens(
    entry_time: float,
    actual_tokens: int,
) -> None:
    if actual_tokens is None:
        return

    with _token_usage_lock:
        for entry in _recent_token_usage:
            if entry[0] == entry_time:
                entry[1] = int(actual_tokens)
                return


def _extract_retry_after(res: requests.Response) -> float:
    header_value = res.headers.get("Retry-After")

    if header_value:
        try:
            return max(0.0, float(header_value))
        except (TypeError, ValueError):
            pass

    match = _RETRY_SECONDS_PATTERN.search(res.text or "")

    if match:
        try:
            return max(0.0, float(match.group(1)))
        except (TypeError, ValueError):
            pass

    return 5.0


def _raise_with_body(
    res: requests.Response,
    provider: str,
) -> None:
    if not res.ok:
        raise RuntimeError(
            f"{provider} API error "
            f"{res.status_code}: "
            f"{res.text[:500]}"
        )


def _post_with_rate_limit_retry(
    url: str,
    provider: str,
    **kwargs,
) -> requests.Response:
    attempt = 0

    while True:
        try:
            res = requests.post(
                url,
                **kwargs,
            )

        except requests.exceptions.RequestException as exc:
            attempt += 1

            if attempt > MAX_RATE_LIMIT_RETRIES:
                raise

            wait_s = min(
                2 ** attempt,
                8.0,
            )

            logger.warning(
                f"{provider} network error ({exc}); "
                f"retrying in {wait_s:.1f}s "
                f"(attempt {attempt}/"
                f"{MAX_RATE_LIMIT_RETRIES})."
            )

            time.sleep(wait_s)
            continue

        if res.status_code != 429:
            _raise_with_body(
                res,
                provider,
            )
            return res

        attempt += 1

        wait_s = _extract_retry_after(res)

        if wait_s > MAX_PROVIDER_WAIT_SECONDS:
            raise RuntimeError(
                f"{provider} rate limit requires "
                f"{wait_s:.1f}s; "
                f"skipping long retry and using fallback."
            )

        if attempt > MAX_RATE_LIMIT_RETRIES:
            raise RuntimeError(
                f"{provider} rate limit persisted after "
                f"{MAX_RATE_LIMIT_RETRIES} retries."
            )

        logger.warning(
            f"{provider} rate-limited "
            f"(attempt {attempt}/"
            f"{MAX_RATE_LIMIT_RETRIES}); "
            f"waiting {wait_s:.1f}s before retry."
        )

        time.sleep(
            wait_s + 0.25
        )


_MAX_COMPLETION_TOKENS = int(
    os.getenv(
        "LLM_MAX_COMPLETION_TOKENS",
        "500",
    )
)


MAX_EVIDENCE_FOR_LLM = 2


def _prepare_evidence(
    evidence: list[dict],
) -> list[dict]:
    prepared = []

    for item in evidence[:MAX_EVIDENCE_FOR_LLM]:
        prepared.append(
            {
                "historical_customer_message": str(
                    item.get(
                        "historical_customer_message",
                        "",
                    )
                ),
                "historical_agent_response": str(
                    item.get(
                        "historical_agent_response",
                        "",
                    )
                ),
                "similarity": float(
                    item.get(
                        "similarity",
                        0.0,
                    )
                ),
            }
        )

    return prepared


def _call_groq(
    system_prompt: str,
    user_prompt: str,
) -> str:
    estimated_tokens = (
        _estimate_tokens(
            system_prompt,
            user_prompt,
        )
        + _MAX_COMPLETION_TOKENS
    )

    entry_time = _wait_for_rate_budget(
        estimated_tokens
    )

    res = _post_with_rate_limit_retry(
        "https://api.groq.com/openai/v1/chat/completions",
        "Groq",
        headers={
            "Authorization": (
                f"Bearer {SETTINGS.groq_api_key}"
            ),
            "Content-Type": "application/json",
        },
        json={
            "model": SETTINGS.groq_model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": 0.1,
            "max_tokens": _MAX_COMPLETION_TOKENS,
        },
        timeout=15,
    )

    data = res.json()

    actual = (
        data
        .get("usage", {})
        .get("total_tokens")
    )

    if actual is not None:
        _record_actual_tokens(
            entry_time,
            actual,
        )

    choice = data["choices"][0]

    if choice.get("finish_reason") == "length":
        raise RuntimeError(
            "Groq response truncated by max_tokens."
        )

    return choice["message"]["content"]


def _call_openai(
    system_prompt: str,
    user_prompt: str,
) -> str:
    estimated_tokens = (
        _estimate_tokens(
            system_prompt,
            user_prompt,
        )
        + _MAX_COMPLETION_TOKENS
    )

    entry_time = _wait_for_rate_budget(
        estimated_tokens
    )

    res = _post_with_rate_limit_retry(
        "https://api.openai.com/v1/chat/completions",
        "OpenAI",
        headers={
            "Authorization": (
                f"Bearer {SETTINGS.openai_api_key}"
            ),
            "Content-Type": "application/json",
        },
        json={
            "model": SETTINGS.openai_model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": 0.1,
            "max_tokens": _MAX_COMPLETION_TOKENS,
            "response_format": {
                "type": "json_object"
            },
        },
        timeout=15,
    )

    data = res.json()

    actual = (
        data
        .get("usage", {})
        .get("total_tokens")
    )

    if actual is not None:
        _record_actual_tokens(
            entry_time,
            actual,
        )

    choice = data["choices"][0]

    if choice.get("finish_reason") == "length":
        raise RuntimeError(
            "OpenAI response truncated by max_tokens."
        )

    return choice["message"]["content"]


def _call_gemini(
    system_prompt: str,
    user_prompt: str,
) -> str:
    estimated_tokens = (
        _estimate_tokens(
            system_prompt,
            user_prompt,
        )
        + _MAX_COMPLETION_TOKENS
    )

    entry_time = _wait_for_rate_budget(
        estimated_tokens
    )

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{SETTINGS.gemini_model}"
        f":generateContent?key={SETTINGS.gemini_api_key}"
    )

    res = _post_with_rate_limit_retry(
        url,
        "Gemini",
        json={
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                f"{system_prompt}\n\n"
                                f"{user_prompt}"
                            )
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": (
                    _MAX_COMPLETION_TOKENS
                ),
            },
        },
        timeout=15,
    )

    data = res.json()

    actual = (
        data
        .get("usageMetadata", {})
        .get("totalTokenCount")
    )

    if actual is not None:
        _record_actual_tokens(
            entry_time,
            actual,
        )

    candidate = data["candidates"][0]

    if candidate.get("finishReason") == "MAX_TOKENS":
        raise RuntimeError(
            "Gemini response truncated by maxOutputTokens."
        )

    return candidate["content"]["parts"][0]["text"]


def _parse_llm_json(
    raw: str,
) -> dict | None:
    if not raw:
        return None

    cleaned = raw.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):]

    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)

    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if (
            start == -1
            or end == -1
            or end < start
        ):
            return None

        try:
            parsed = json.loads(
                cleaned[start:end + 1]
            )

        except json.JSONDecodeError:
            return None

    if not isinstance(parsed, dict):
        return None

    missing = REQUIRED_KEYS - parsed.keys()

    if missing:
        logger.warning(
            "LLM response missing keys: "
            f"{missing}"
        )
        return None

    return parsed


def _sanitize_llm_intent_fields(parsed: dict) -> dict:
    intent = parsed.get("intent")

    if intent is not None and intent not in llm_taxonomy.CURATED_INTENTS:
        logger.warning(
            f"LLM returned an intent not in the curated taxonomy: "
            f"{intent!r} -- ignoring; local classifier will be used "
            f"for intent instead."
        )
        intent = None

    parsed["intent"] = intent

    if intent is None:
        parsed["intent_confidence"] = None
    else:
        raw_confidence = parsed.get("intent_confidence")
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        parsed["intent_confidence"] = max(0.0, min(1.0, confidence))

    return parsed


def _deterministic_fallback(
    evidence: list[dict],
    reason: str,
) -> dict:
    if not evidence:
        return {
            "reply": (
                "Thank you for reaching out. "
                "A member of our team will review "
                "your request and get back to you."
            ),
            "grounded": False,
            "confidence": 0.0,
            "reasoning_summary": (
                "No retrieved evidence was available; "
                "deterministic fallback used."
            ),
            "should_escalate": True,
            "escalation_reason": (
                "No LLM generation available and "
                "no similar historical case was found."
            ),
            "intent": None,
            "intent_confidence": None,
        }

    top = evidence[0]

    similarity = float(
        top.get(
            "similarity",
            0.0,
        )
    )

    reply = (
        "Thanks for letting us know about this. "
        "We understand the issue and will review "
        "the relevant details. If the problem continues, "
        "a member of our support team can assist further."
    )

    should_escalate = (
        similarity
        < SETTINGS.min_retrieval_similarity
    )

    return {
        "reply": reply,
        "grounded": True,
        "confidence": similarity,
        "reasoning_summary": (
            "Deterministic privacy-safe fallback used "
            f"because LLM generation was unavailable: "
            f"{reason}"
        ),
        "should_escalate": should_escalate,
        "escalation_reason": (
            None
            if not should_escalate
            else (
                "Best historical match similarity is "
                "below the configured threshold."
            )
        ),
        "intent": None,
        "intent_confidence": None,
    }


class ResponseGenerator:
    def generate(
        self,
        customer_message: str,
        evidence: list[dict],
    ) -> dict:

        if not SETTINGS.llm_configured():
            return _deterministic_fallback(
                evidence,
                (
                    f"LLM provider "
                    f"'{SETTINGS.llm_provider}' "
                    "not configured"
                ),
            )

        prepared_evidence = _prepare_evidence(
            evidence
        )

        user_prompt = build_user_prompt(
            customer_message,
            prepared_evidence,
        )

        try:
            if SETTINGS.llm_provider == "groq":

                raw = _call_groq(
                    SYSTEM_PROMPT,
                    user_prompt,
                )

            elif SETTINGS.llm_provider == "openai":

                raw = _call_openai(
                    SYSTEM_PROMPT,
                    user_prompt,
                )

            elif SETTINGS.llm_provider == "gemini":

                raw = _call_gemini(
                    SYSTEM_PROMPT,
                    user_prompt,
                )

            else:

                return _deterministic_fallback(
                    evidence,
                    (
                        f"Unknown provider "
                        f"'{SETTINGS.llm_provider}'"
                    ),
                )

        except Exception as exc:

            logger.error(
                f"LLM call failed: {exc}"
            )

            return _deterministic_fallback(
                evidence,
                f"LLM call failed: {exc}",
            )

        parsed = _parse_llm_json(raw)

        if parsed is None:

            logger.error(
                "Could not parse LLM response: "
                f"{raw[:300]}"
            )

            return _deterministic_fallback(
                evidence,
                (
                    "LLM response could not be "
                    "parsed as valid JSON"
                ),
            )

        return _sanitize_llm_intent_fields(parsed)


if __name__ == "__main__":

    fake_evidence = [
        {
            "historical_customer_message": (
                "Where is my refund?"
            ),
            "historical_agent_response": (
                "We've processed your refund — "
                "it should appear in 5-7 business days."
            ),
            "similarity": 0.81,
        },
        {
            "historical_customer_message": (
                "My refund hasn't arrived."
            ),
            "historical_agent_response": (
                "Your refund has been processed."
            ),
            "similarity": 0.77,
        },
    ]

    result = ResponseGenerator().generate(
        "My refund still hasn't shown up",
        fake_evidence,
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )