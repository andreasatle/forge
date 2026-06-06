"""Tests for the LLM client content extraction."""

import pytest

from forge.llm.client import _extract_content


def test_extract_content_returns_first_nonempty_string():
    """_extract_content returns the first non-empty string value in the message dict."""
    assert _extract_content({"content": "hello world"}) == "hello world"


def test_extract_content_works_with_any_field_name():
    """_extract_content extracts content regardless of the key name used by the model."""
    assert _extract_content({"text": "model response"}) == "model response"
    assert _extract_content({"response": "another format"}) == "another format"


def test_extract_content_raises_on_empty_message():
    """_extract_content raises ValueError when the message dict has no string values."""
    with pytest.raises(ValueError):
        _extract_content({})


def test_extract_content_raises_when_content_is_empty_string():
    """_extract_content raises ValueError when all string values in the message are empty."""
    with pytest.raises(ValueError):
        _extract_content({"content": ""})


def test_extract_content_raises_when_all_strings_empty():
    """_extract_content raises ValueError when every string value is empty."""
    with pytest.raises(ValueError):
        _extract_content({"content": "", "text": ""})


def test_extract_content_skips_non_string_values():
    """_extract_content ignores non-string values and finds the string among mixed types."""
    assert _extract_content({"count": 42, "content": "found it"}) == "found it"
