from __future__ import annotations
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
_APOSTROPHE = "['\u2019]"
_CONTRACTION_SUFFIXES = [
    (re.compile(rf"n{_APOSTROPHE}t\b"), " not"),
    (re.compile(rf"{_APOSTROPHE}re\b"), " are"),
    (re.compile(rf"{_APOSTROPHE}ve\b"), " have"),
    (re.compile(rf"{_APOSTROPHE}ll\b"), " will"),
    (re.compile(rf"{_APOSTROPHE}d\b"), " would"),
    (re.compile(rf"{_APOSTROPHE}m\b"), " am"),
    (re.compile(rf"{_APOSTROPHE}s\b"), ""),
]

def tfidf_preprocess(text: str) -> str:
    text = text.lower()
    for pattern, replacement in _CONTRACTION_SUFFIXES:
        text = pattern.sub(replacement, text)
    return text

_DOMAIN_FILLER_WORDS = frozenset({
    "hey", "hi", "hello", "yo", "please",
    "pls","plz","thanks", "thank", "thankyou",  "like",
    "really", "just","yeah", "yep","ok","okay", "lol", "u", "ur",
    "im",
    "ive",
    "youre",
    "gonna",
    "wanna",
    "got",
    "time",
    "today",
    "know",
    "did",
    "going",
    "help",
    "ve",
    "re",
    "don",
})
TFIDF_STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) | _DOMAIN_FILLER_WORDS