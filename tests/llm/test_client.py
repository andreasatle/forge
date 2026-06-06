"""Tests for the Ollama LLM client content extraction."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge.llm.client import chat


def _mock_response(data: dict) -> MagicMock:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = data
    return response


async def test_chat_returns_content_field():
    """chat returns the content field from the message."""
    with patch("forge.llm.client.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=_mock_response({"message": {"role": "assistant", "content": "hello world"}})
        )
        result = await chat("model", "prompt")
    assert result == "hello world"


async def test_chat_strips_whitespace_from_content():
    """chat strips leading and trailing whitespace from the content field."""
    with patch("forge.llm.client.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=_mock_response({"message": {"content": "  response text  "}})
        )
        result = await chat("model", "prompt")
    assert result == "response text"


async def test_chat_raises_on_empty_content():
    """chat raises ValueError when the content field is an empty string."""
    with patch("forge.llm.client.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=_mock_response({"message": {"content": ""}})
        )
        with pytest.raises(ValueError):
            await chat("model", "prompt")


async def test_chat_raises_on_whitespace_only_content():
    """chat raises ValueError when the content field contains only whitespace."""
    with patch("forge.llm.client.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=_mock_response({"message": {"content": "   "}})
        )
        with pytest.raises(ValueError):
            await chat("model", "prompt")


async def test_chat_raises_on_missing_content():
    """chat raises ValueError when the message has no content field."""
    with patch("forge.llm.client.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=_mock_response({"message": {}})
        )
        with pytest.raises(ValueError):
            await chat("model", "prompt")
