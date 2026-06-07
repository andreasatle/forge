"""Tests for LLM provider implementations and the make_provider factory."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from forge.llm.factory import make_provider
from forge.llm.providers import ClaudeProvider, OllamaProvider, OpenAIProvider


def _mock_http_response(data: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = data
    return resp


# Canonical assistant tool call used across provider wire-format tests.
_CANONICAL_MESSAGES = [
    {"role": "user", "content": "go"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_read_0",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "/tmp/x"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_read_0", "content": "file contents"},
]

_TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "reads a file", "parameters": {}}}
]


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
    captured: list[dict] = []

    async def fake_post(url: str, **kwargs: object) -> MagicMock:
        captured.append({"url": url, "json": kwargs.get("json")})
        return _mock_http_response({"message": {"content": "hello"}})

    with patch("forge.llm.providers.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(side_effect=fake_post)
        result = await provider.chat("say hello", 512)

    assert result == "hello"
    assert captured[0]["url"] == "http://localhost:11434/api/chat"
    payload = captured[0]["json"]
    assert payload["model"] == "gemma4:e4b"
    assert payload["options"]["num_predict"] == 512
    assert payload["messages"][0] == {"role": "user", "content": "say hello"}


async def test_ollama_provider_chat_with_tools_converts_canonical_to_ollama_wire() -> None:
    """OllamaProvider converts canonical messages to Ollama wire format before sending."""
    provider = OllamaProvider(model="gemma4:e4b", max_tokens=512)
    captured: list[dict] = []

    async def fake_post(url: str, **kwargs: object) -> MagicMock:
        captured.append(kwargs.get("json"))
        return _mock_http_response({"message": {"content": "done"}})

    with patch("forge.llm.providers.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(side_effect=fake_post)
        await provider.chat_with_tools(_CANONICAL_MESSAGES, _TOOLS, 512)

    sent = captured[0]["messages"]
    # user message unchanged
    assert sent[0] == {"role": "user", "content": "go"}
    # assistant tool call: no id/type, arguments is a dict
    asst = sent[1]
    assert asst["role"] == "assistant"
    tc = asst["tool_calls"][0]
    assert "id" not in tc
    assert "type" not in tc
    assert tc["function"]["name"] == "read_file"
    assert isinstance(tc["function"]["arguments"], dict)
    assert tc["function"]["arguments"] == {"path": "/tmp/x"}
    # tool result: uses name instead of tool_call_id
    tool_msg = sent[2]
    assert tool_msg["role"] == "tool"
    assert "tool_call_id" not in tool_msg
    assert tool_msg["name"] == "read_file"
    assert tool_msg["content"] == "file contents"


async def test_ollama_provider_chat_with_tools_returns_canonical_tool_calls() -> None:
    """OllamaProvider converts Ollama tool_calls response to canonical format."""
    provider = OllamaProvider(model="gemma4:e4b", max_tokens=512)
    ollama_response = {
        "message": {
            "tool_calls": [{"function": {"name": "write_file", "arguments": {"path": "/out", "content": "hi"}}}]
        }
    }

    with patch("forge.llm.providers.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=_mock_http_response(ollama_response)
        )
        text, tool_calls = await provider.chat_with_tools([{"role": "user", "content": "go"}], [], 512)

    assert text is None
    assert len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc["id"].startswith("call_")
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "write_file"
    assert isinstance(tc["function"]["arguments"], str)
    assert json.loads(tc["function"]["arguments"]) == {"path": "/out", "content": "hi"}


# --- ClaudeProvider ---


async def test_claude_provider_chat_formats_request_correctly() -> None:
    """ClaudeProvider.chat sends the correct payload and headers to the Anthropic API."""
    provider = ClaudeProvider(model="claude-sonnet-4-20250514", api_key="sk-test", max_tokens=1024)
    captured: list[dict] = []

    async def fake_post(url: str, **kwargs: object) -> MagicMock:
        captured.append({"url": url, "json": kwargs.get("json"), "headers": kwargs.get("headers")})
        return _mock_http_response({"content": [{"type": "text", "text": "response"}]})

    with patch("forge.llm.providers.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(side_effect=fake_post)
        result = await provider.chat("hello", 1024)

    assert result == "response"
    assert captured[0]["url"] == "https://api.anthropic.com/v1/messages"
    payload = captured[0]["json"]
    assert payload["model"] == "claude-sonnet-4-20250514"
    assert payload["max_tokens"] == 1024
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    headers = captured[0]["headers"]
    assert headers["x-api-key"] == "sk-test"
    assert "anthropic-version" in headers


async def test_claude_provider_chat_raises_on_empty_response() -> None:
    """ClaudeProvider.chat raises ValueError when the API returns no text content."""
    provider = ClaudeProvider(model="claude-sonnet-4-20250514", api_key="sk-test", max_tokens=1024)
    with patch("forge.llm.providers.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=_mock_http_response({"content": []})
        )
        with pytest.raises(ValueError, match="empty content"):
            await provider.chat("hello", 1024)


async def test_claude_provider_chat_with_tools_converts_canonical_to_claude_wire() -> None:
    """ClaudeProvider converts canonical messages to Claude API wire format before sending."""
    provider = ClaudeProvider(model="claude-sonnet-4-20250514", api_key="sk-test", max_tokens=1024)
    captured: list[dict] = []

    async def fake_post(url: str, **kwargs: object) -> MagicMock:
        captured.append(kwargs.get("json"))
        return _mock_http_response({"content": [{"type": "text", "text": "done"}]})

    with patch("forge.llm.providers.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(side_effect=fake_post)
        await provider.chat_with_tools(_CANONICAL_MESSAGES, _TOOLS, 1024)

    sent = captured[0]["messages"]
    # user message unchanged
    assert sent[0] == {"role": "user", "content": "go"}
    # assistant tool call becomes tool_use content block
    asst = sent[1]
    assert asst["role"] == "assistant"
    tu = asst["content"][0]
    assert tu["type"] == "tool_use"
    assert tu["id"] == "call_read_0"
    assert tu["name"] == "read_file"
    assert tu["input"] == {"path": "/tmp/x"}
    # tool result becomes user message with tool_result block
    tool_msg = sent[2]
    assert tool_msg["role"] == "user"
    tr = tool_msg["content"][0]
    assert tr["type"] == "tool_result"
    assert tr["tool_use_id"] == "call_read_0"
    assert tr["content"] == "file contents"


async def test_claude_provider_chat_with_tools_returns_canonical_tool_calls() -> None:
    """ClaudeProvider converts Claude tool_use response to canonical format."""
    provider = ClaudeProvider(model="claude-sonnet-4-20250514", api_key="sk-test", max_tokens=1024)
    claude_response = {
        "content": [{"type": "tool_use", "id": "toolu_abc123", "name": "write_file", "input": {"path": "/out"}}]
    }

    with patch("forge.llm.providers.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=_mock_http_response(claude_response)
        )
        text, tool_calls = await provider.chat_with_tools([{"role": "user", "content": "go"}], [], 1024)

    assert text is None
    assert len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc["id"] == "toolu_abc123"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "write_file"
    assert isinstance(tc["function"]["arguments"], str)
    assert json.loads(tc["function"]["arguments"]) == {"path": "/out"}


# --- OpenAIProvider ---


async def test_openai_provider_chat_formats_request_correctly() -> None:
    """OpenAIProvider.chat sends the correct payload and headers to the OpenAI API."""
    provider = OpenAIProvider(model="gpt-4o", api_key="sk-test", max_tokens=2048)
    captured: list[dict] = []

    async def fake_post(url: str, **kwargs: object) -> MagicMock:
        captured.append({"url": url, "json": kwargs.get("json"), "headers": kwargs.get("headers")})
        return _mock_http_response({
            "choices": [{"message": {"content": "answer", "role": "assistant"}}]
        })

    with patch("forge.llm.providers.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(side_effect=fake_post)
        result = await provider.chat("question", 2048)

    assert result == "answer"
    assert captured[0]["url"] == "https://api.openai.com/v1/chat/completions"
    payload = captured[0]["json"]
    assert payload["model"] == "gpt-4o"
    assert payload["max_tokens"] == 2048
    assert payload["messages"] == [{"role": "user", "content": "question"}]
    headers = captured[0]["headers"]
    assert headers["Authorization"] == "Bearer sk-test"


async def test_openai_provider_chat_raises_on_empty_response() -> None:
    """OpenAIProvider.chat raises ValueError when the API returns empty content."""
    provider = OpenAIProvider(model="gpt-4o", api_key="sk-test", max_tokens=2048)
    with patch("forge.llm.providers.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=_mock_http_response({"choices": [{"message": {"content": ""}}]})
        )
        with pytest.raises(ValueError, match="empty content"):
            await provider.chat("question", 2048)


async def test_openai_provider_chat_with_tools_passes_canonical_messages_through() -> None:
    """OpenAIProvider sends canonical messages unchanged — canonical IS OpenAI format."""
    provider = OpenAIProvider(model="gpt-4o", api_key="sk-test", max_tokens=2048)
    captured: list[dict] = []

    async def fake_post(url: str, **kwargs: object) -> MagicMock:
        captured.append(kwargs.get("json"))
        return _mock_http_response({"choices": [{"message": {"content": "done", "role": "assistant"}}]})

    with patch("forge.llm.providers.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(side_effect=fake_post)
        await provider.chat_with_tools(_CANONICAL_MESSAGES, _TOOLS, 2048)

    assert captured[0]["messages"] is _CANONICAL_MESSAGES


async def test_openai_provider_chat_with_tools_returns_canonical_tool_calls() -> None:
    """OpenAIProvider returns OpenAI tool_calls response in canonical format."""
    provider = OpenAIProvider(model="gpt-4o", api_key="sk-test", max_tokens=2048)
    openai_response = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_xyz", "type": "function", "function": {"name": "write_file", "arguments": '{"path": "/out"}'}}],
            }
        }]
    }

    with patch("forge.llm.providers.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=_mock_http_response(openai_response)
        )
        text, tool_calls = await provider.chat_with_tools([{"role": "user", "content": "go"}], [], 2048)

    assert text is None
    assert len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc["id"] == "call_xyz"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "write_file"
    assert json.loads(tc["function"]["arguments"]) == {"path": "/out"}


async def test_ollama_provider_sanitizes_trailing_equals_in_argument_keys() -> None:
    """OllamaProvider strips trailing '=' from argument key names returned by the model."""
    provider = OllamaProvider(model="gemma4:e4b", max_tokens=512)
    ollama_response = {
        "message": {
            "tool_calls": [{"function": {"name": "write_file", "arguments": {"path=": "/out", "content=": "hi"}}}]
        }
    }

    with patch("forge.llm.providers.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=_mock_http_response(ollama_response)
        )
        _, tool_calls = await provider.chat_with_tools([{"role": "user", "content": "go"}], [], 512)

    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"path": "/out", "content": "hi"}


async def test_claude_provider_sanitizes_trailing_equals_in_argument_keys() -> None:
    """ClaudeProvider strips trailing '=' from argument key names returned by the model."""
    provider = ClaudeProvider(model="claude-sonnet-4-20250514", api_key="sk-test", max_tokens=1024)
    claude_response = {
        "content": [{"type": "tool_use", "id": "tu_0", "name": "write_file", "input": {"path=": "/out", "content=": "hi"}}]
    }

    with patch("forge.llm.providers.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=_mock_http_response(claude_response)
        )
        _, tool_calls = await provider.chat_with_tools([{"role": "user", "content": "go"}], [], 1024)

    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"path": "/out", "content": "hi"}


async def test_openai_provider_sanitizes_trailing_equals_in_argument_keys() -> None:
    """OpenAIProvider strips trailing '=' from argument key names returned by the model."""
    provider = OpenAIProvider(model="gpt-4o", api_key="sk-test", max_tokens=2048)
    openai_response = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "write_file", "arguments": '{"path=": "/out", "content=": "hi"}'}}],
            }
        }]
    }

    with patch("forge.llm.providers.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=_mock_http_response(openai_response)
        )
        _, tool_calls = await provider.chat_with_tools([{"role": "user", "content": "go"}], [], 2048)

    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"path": "/out", "content": "hi"}


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
            result = await provider.chat("hi", 512)

    assert result == "ok"
    mock_sleep.assert_called_once_with(1.0)


async def test_provider_retries_on_503_then_succeeds() -> None:
    """Provider retries on 503 and returns the result when a subsequent attempt succeeds."""
    provider = OllamaProvider(model="gemma4:e4b", max_tokens=512)

    with patch("forge.llm.providers.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            side_effect=[_error_response(503), _mock_http_response({"message": {"content": "ok"}})]
        )
        with patch("forge.llm.providers.asyncio.sleep"):
            result = await provider.chat("hi", 512)

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
                await provider.chat("hi", 512)

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
                await provider.chat("hi", 512)
