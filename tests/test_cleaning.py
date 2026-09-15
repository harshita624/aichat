import pandas as pd

from src.data.clean import clean_text, clean_dataframe


def test_clean_text_strips_urls():
    assert "http" not in clean_text(
        "Check this https://example.com/path?x=1 thanks"
    )


def test_clean_text_strips_mentions():
    result = clean_text("@AmazonHelp my order is late")
    assert "@AmazonHelp" not in result and "order is late" in result


def test_clean_text_unescapes_html_entities():
    assert clean_text("Price &amp; shipping") == "Price & shipping"


def test_clean_text_collapses_whitespace():
    assert clean_text("too   many\n\nspaces") == "too many spaces"


def test_clean_text_handles_non_string_input():
    assert clean_text(None) == "" and clean_text(float("nan")) == ""


def test_clean_dataframe_preserves_exact_duplicates_for_conversation_stage():
    df = pd.DataFrame({
        "text": [
            "hello world",
            "hello world",
            "different message",
        ]
    })

    assert len(clean_dataframe(df, "text")) == 3


def test_clean_dataframe_removes_too_short_messages():
    df = pd.DataFrame({
        "text": [
            "ok",
            "a real customer message here",
            "x",
        ]
    })

    cleaned = clean_dataframe(df, "text")

    assert len(cleaned) == 1
    assert "real customer message" in cleaned.iloc[0]["clean_text"]