"""Tests for LLM provider implementations and the make_provider factory."""

from collections.abc import Mapping
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from forge.llm.factory import make_provider
from forge.llm.providers import (
    ChatMessage,
    ClaudeProvider,
    OllamaProvider,
    OpenAIProvider,
    ProviderEmptyOutputError,
)

type CapturedRequest = dict[str, object]

_SEMANTIC_PROVIDER_INJECTION_MARKERS = (
    "valid JSON",
    "No markdown",
    "no explanation",
    "no preamble",
)


def _mock_http_response(data: Mapping[str, object]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = data
    return resp


def _captured_mapping(captured: list[CapturedRequest], key: str) -> dict[str, object]:
    value = captured[0][key]
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _list(value: object) -> list[object]:
    assert isinstance(value, list)
    return cast(list[object], value)


async def _capture_provider_payload(
    provider: object, response_data: Mapping[str, object]
) -> dict[str, object]:
    captured: list[CapturedRequest] = []

    async def fake_post(url: str, **kwargs: object) -> MagicMock:
        captured.append({"url": url, "json": kwargs.get("json"), "headers": kwargs.get("headers")})
        return _mock_http_response(response_data)

    messages: list[ChatMessage] = [
        {"role": "system", "content": "System-owned instruction."},
        {"role": "user", "content": "Do the task."},
        {"role": "assistant", "content": '{"kind": "tool"}'},
        {"role": "user", "content": '{"kind": "tool_response"}'},
    ]
    with patch("forge.llm.providers.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(side_effect=fake_post)
        await provider.chat(messages)  # type: ignore[attr-defined]

    return _captured_mapping(captured, "json")


# --- make_provider factory ---


def test_make_provider_ollama_returns_ollama_provider() -> None:
    """make_provider returns an OllamaProvider for the 'ollama/' prefix."""
    provider = make_provider("ollama/gemma4:e4b", 8192)
    assert isinstance(provider, OllamaProvider)
    assert provider.model == "gemma4:e4b"
    assert provider.max_tokens == 8192


def test_make_provider_claude_returns_claude_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """make_provider returns a ClaudeProvider for the 'claude/' prefix."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = make_provider("claude/claude-sonnet-4-20250514", 4096)
    assert isinstance(provider, ClaudeProvider)
    assert provider.model == "claude-sonnet-4-20250514"
    assert provider.max_tokens == 4096


def test_make_provider_openai_returns_openai_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """make_provider returns an OpenAIProvider for the 'openai/' prefix."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = make_provider("openai/gpt-4o", 2048)
    assert isinstance(provider, OpenAIProvider)
    assert provider.model == "gpt-4o"
    assert provider.max_tokens == 2048


def test_make_provider_unknown_prefix_raises() -> None:
    """make_provider raises ValueError for an unknown provider prefix."""
    with pytest.raises(ValueError, match="unknown provider"):
        make_provider("groq/mixtral", 8192)


def test_make_provider_missing_slash_raises() -> None:
    """make_provider raises ValueError when the model string has no slash."""
    with pytest.raises(ValueError, match="provider/model"):
        make_provider("gemma4:e4b", 8192)


def test_make_provider_claude_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """make_provider raises ValueError when ANTHROPIC_API_KEY is not set."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        make_provider("claude/claude-sonnet-4-20250514", 8192)


def test_make_provider_openai_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """make_provider raises ValueError when OPENAI_API_KEY is not set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        make_provider("openai/gpt-4o", 8192)


# --- OllamaProvider ---


async def test_ollama_provider_chat_formats_request_correctly() -> None:
    """OllamaProvider.chat sends the correct payload to the Ollama REST API."""
    provider = OllamaProvider(model="gemma4:e4b", max_tokens=512, base_url="http://localhost:11434")
    captured: list[CapturedRequest] = []

    async def fake_post(url: str, **kwargs: object) -> MagicMock:
        captured.append({"url": url, "json": kwargs.get("json")})
        return _mock_http_response({"message": {"content": "hello"}})

    with patch("forge.llm.providers.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(side_effect=fake_post)
        result = await provider.chat([{"role": "user", "content": "say hello"}])

    assert result == "hello"
    assert captured[0]["url"] == "http://localhost:11434/api/chat"
    payload = _captured_mapping(captured, "json")
    options = _mapping(payload["options"])
    messages = _list(payload["messages"])
    assert payload["model"] == "gemma4:e4b"
    assert options["num_predict"] == 512
    assert messages[0] == {"role": "user", "content": "say hello"}
    assert payload["format"] == "json"


# --- ClaudeProvider ---


async def test_claude_provider_chat_formats_request_correctly() -> None:
    """ClaudeProvider.chat sends the correct payload and headers to the Anthropic API."""
    provider = ClaudeProvider(model="claude-sonnet-4-20250514", api_key="sk-test", max_tokens=1024)
    captured: list[CapturedRequest] = []

    async def fake_post(url: str, **kwargs: object) -> MagicMock:
        captured.append({"url": url, "json": kwargs.get("json"), "headers": kwargs.get("headers")})
        return _mock_http_response({"content": [{"type": "text", "text": "response"}]})

    with patch("forge.llm.providers.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(side_effect=fake_post)
        result = await provider.chat([{"role": "user", "content": "hello"}])

    assert result == "response"
    assert captured[0]["url"] == "https://api.anthropic.com/v1/messages"
    payload = _captured_mapping(captured, "json")
    headers = _captured_mapping(captured, "headers")
    assert payload["model"] == "claude-sonnet-4-20250514"
    assert payload["max_tokens"] == 1024
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert "system" not in payload
    assert headers["x-api-key"] == "sk-test"
    assert "anthropic-version" in headers


async def test_providers_preserve_prompt_semantics_without_injection() -> None:
    """Providers may package messages differently, but must not add semantic prompt text."""
    messages: list[ChatMessage] = [
        {"role": "system", "content": "System-owned instruction."},
        {"role": "user", "content": "Do the task."},
        {"role": "assistant", "content": '{"kind": "tool"}'},
        {"role": "user", "content": '{"kind": "tool_response"}'},
    ]

    ollama_payload = await _capture_provider_payload(
        OllamaProvider(model="gemma4:e4b", max_tokens=512),
        {"message": {"content": "{}"}},
    )
    openai_payload = await _capture_provider_payload(
        OpenAIProvider(model="gpt-4o", api_key="sk-test", max_tokens=512),
        {"choices": [{"message": {"content": "{}"}}]},
    )
    claude_payload = await _capture_provider_payload(
        ClaudeProvider(model="claude-sonnet-4-20250514", api_key="sk-test", max_tokens=512),
        {"content": [{"type": "text", "text": "{}"}]},
    )

    assert ollama_payload["messages"] == messages
    assert openai_payload["messages"] == messages
    assert claude_payload["system"] == messages[0]["content"]
    assert claude_payload["messages"] == messages[1:]

    semantic_text = "\n".join(
        str(value)
        for payload in (ollama_payload, openai_payload, claude_payload)
        for value in payload.values()
    )
    for marker in _SEMANTIC_PROVIDER_INJECTION_MARKERS:
        assert marker not in semantic_text


async def test_claude_provider_chat_raises_on_empty_response() -> None:
    """ClaudeProvider.chat raises ProviderEmptyOutputError when the API returns no text content."""
    provider = ClaudeProvider(model="claude-sonnet-4-20250514", api_key="sk-test", max_tokens=1024)
    with patch("forge.llm.providers.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=_mock_http_response({"content": []})
        )
        with pytest.raises(ProviderEmptyOutputError, match="empty content"):
            await provider.chat([{"role": "user", "content": "hello"}])


# --- OpenAIProvider ---


async def test_openai_provider_chat_formats_request_correctly() -> None:
    """OpenAIProvider.chat sends the correct payload and headers to the OpenAI API."""
    provider = OpenAIProvider(model="gpt-4o", api_key="sk-test", max_tokens=2048)
    captured: list[CapturedRequest] = []

    async def fake_post(url: str, **kwargs: object) -> MagicMock:
        captured.append({"url": url, "json": kwargs.get("json"), "headers": kwargs.get("headers")})
        return _mock_http_response(
            {"choices": [{"message": {"content": "answer", "role": "assistant"}}]}
        )

    with patch("forge.llm.providers.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(side_effect=fake_post)
        result = await provider.chat([{"role": "user", "content": "question"}])

    assert result == "answer"
    assert captured[0]["url"] == "https://api.openai.com/v1/chat/completions"
    payload = _captured_mapping(captured, "json")
    headers = _captured_mapping(captured, "headers")
    assert payload["model"] == "gpt-4o"
    assert payload["max_tokens"] == 2048
    assert payload["messages"] == [{"role": "user", "content": "question"}]
    assert payload["response_format"] == {"type": "json_object"}
    assert headers["Authorization"] == "Bearer sk-test"


async def test_openai_provider_chat_raises_on_empty_response() -> None:
    """OpenAIProvider.chat raises ProviderEmptyOutputError when the API returns empty content."""
    provider = OpenAIProvider(model="gpt-4o", api_key="sk-test", max_tokens=2048)
    with patch("forge.llm.providers.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=_mock_http_response({"choices": [{"message": {"content": ""}}]})
        )
        with pytest.raises(ProviderEmptyOutputError, match="empty content"):
            await provider.chat([{"role": "user", "content": "question"}])


# --- Retry behaviour (shared _post_with_retry helper) ---


def _make_http_error(status_code: int) -> httpx.HTTPStatusError:
    mock_response = MagicMock()
    mock_response.status_code = status_code
    return httpx.HTTPStatusError("error", request=MagicMock(), response=mock_response)


def _error_response(status_code: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status.side_effect = _make_http_error(status_code)
    return resp


async def test_provider_retries_on_429_then_succeeds() -> None:
    """Provider retries on 429 and returns the result when a subsequent attempt succeeds."""
    provider = OllamaProvider(model="gemma4:e4b", max_tokens=512)

    with patch("forge.llm.providers.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            side_effect=[_error_response(429), _mock_http_response({"message": {"content": "ok"}})]
        )
        with patch("forge.llm.providers.asyncio.sleep") as mock_sleep:
            result = await provider.chat([{"role": "user", "content": "hi"}])

    assert result == "ok"
    mock_sleep.assert_called_once_with(10.0)


async def test_provider_retries_on_503_then_succeeds() -> None:
    """Provider retries on 503 and returns the result when a subsequent attempt succeeds."""
    provider = OllamaProvider(model="gemma4:e4b", max_tokens=512)

    with patch("forge.llm.providers.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            side_effect=[_error_response(503), _mock_http_response({"message": {"content": "ok"}})]
        )
        with patch("forge.llm.providers.asyncio.sleep"):
            result = await provider.chat([{"role": "user", "content": "hi"}])

    assert result == "ok"


async def test_provider_does_not_retry_on_401() -> None:
    """Provider raises immediately on 401 without retrying."""
    provider = OllamaProvider(model="gemma4:e4b", max_tokens=512)

    with patch("forge.llm.providers.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=_error_response(401)
        )
        with patch("forge.llm.providers.asyncio.sleep") as mock_sleep:
            with pytest.raises(httpx.HTTPStatusError):
                await provider.chat([{"role": "user", "content": "hi"}])

    mock_sleep.assert_not_called()


async def test_provider_reraises_after_max_retries_exhausted() -> None:
    """Provider re-raises the last HTTPStatusError after all retry attempts are exhausted."""
    provider = OllamaProvider(model="gemma4:e4b", max_tokens=512)

    with patch("forge.llm.providers.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=_error_response(429)
        )
        with patch("forge.llm.providers.asyncio.sleep"):
            with pytest.raises(httpx.HTTPStatusError):
                await provider.chat([{"role": "user", "content": "hi"}])
