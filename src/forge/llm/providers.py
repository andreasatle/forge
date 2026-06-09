"""LLM provider Protocol and implementations for Ollama, Claude, and OpenAI."""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict, cast

import httpx

_ANTHROPIC_API = "https://api.anthropic.com/v1/messages"
_OPENAI_API = "https://api.openai.com/v1/chat/completions"
_ANTHROPIC_VERSION = "2023-06-01"

_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 10.0
_JSON_INSTRUCTION = (
    "You must respond with valid JSON only. No markdown, no explanation, no preamble."
)


class ChatMessage(TypedDict):
    """A single chat message sent to an LLM provider."""

    role: Literal["system", "user", "assistant"]
    content: str


class ProviderError(Exception):
    """Base exception for provider-level failures."""


class ProviderEmptyOutputError(ProviderError):
    """Raised when a provider response has no usable text content."""


def _as_mapping(value: object) -> Mapping[str, object] | None:
    return cast(Mapping[str, object], value) if isinstance(value, dict) else None


def _as_list(value: object) -> list[object] | None:
    return cast(list[object], value) if isinstance(value, list) else None


async def _post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    **kwargs: object,
) -> httpx.Response:
    """POST to url, retrying on 429/5xx and raising immediately on other errors."""
    last_exc: httpx.HTTPStatusError | None = None
    for attempt in range(_MAX_RETRIES):
        response = await client.post(url, **kwargs)  # type: ignore[arg-type]
        try:
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in _RETRY_STATUSES:
                raise
            last_exc = e
            if attempt < _MAX_RETRIES - 1:
                wait = _RETRY_BASE_DELAY * (2 ** attempt)
                print(
                    f"[debug] HTTP {e.response.status_code} — retry "
                    f"{attempt + 1}/{_MAX_RETRIES - 1}, waiting {wait}s"
                )
                await asyncio.sleep(wait)
    if last_exc is not None:
        raise last_exc
    raise ProviderError("provider retry loop exhausted without a response or HTTP error")


class LLMProvider(Protocol):
    """Protocol for LLM providers — multi-turn chat."""

    max_tokens: int

    async def chat(self, messages: list[ChatMessage]) -> str:
        """Send a multi-turn message list and return the response text."""
        ...


@dataclass
class OllamaProvider:
    """Ollama LLM provider using the local REST API."""

    model: str
    max_tokens: int
    base_url: str = "http://localhost:11434"

    async def chat(self, messages: list[ChatMessage]) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": self.max_tokens},
            "format": "json",
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await _post_with_retry(client, f"{self.base_url}/api/chat", json=payload)
        data = cast(dict[str, object], response.json())
        message = _as_mapping(data.get("message"))
        content = message.get("content") if message is not None else ""
        if not isinstance(content, str) or not content.strip():
            raise ProviderEmptyOutputError("provider returned empty content")
        return content.strip()


@dataclass
class ClaudeProvider:
    """Anthropic Claude LLM provider using the Messages API."""

    model: str
    api_key: str
    max_tokens: int

    async def chat(self, messages: list[ChatMessage]) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        non_system = [m for m in messages if m["role"] != "system"]
        headers: dict[str, str] = {
            "x-api-key": self.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        payload: dict[str, object] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": non_system,
        }
        payload["system"] = f"{system}\n\n{_JSON_INSTRUCTION}" if system else _JSON_INSTRUCTION
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await _post_with_retry(client, _ANTHROPIC_API, headers=headers, json=payload)
        data = cast(dict[str, object], response.json())
        blocks = _as_list(data.get("content"))
        content = ""
        if blocks is not None:
            for block in blocks:
                block_data = _as_mapping(block)
                text = block_data.get("text") if block_data is not None else None
                if (
                    block_data is not None
                    and block_data.get("type") == "text"
                    and isinstance(text, str)
                ):
                    content = text
                    break
        if not content or not content.strip():
            raise ProviderEmptyOutputError("provider returned empty content")
        return content.strip()


@dataclass
class OpenAIProvider:
    """OpenAI LLM provider using the Chat Completions API."""

    model: str
    api_key: str
    max_tokens: int

    async def chat(self, messages: list[ChatMessage]) -> str:
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        payload: dict[str, object] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await _post_with_retry(client, _OPENAI_API, headers=headers, json=payload)
        data = cast(dict[str, object], response.json())
        choices = _as_list(data.get("choices"))
        first_choice = _as_mapping(choices[0]) if choices else None
        message = _as_mapping(first_choice.get("message")) if first_choice is not None else None
        content = message.get("content") if message is not None else ""
        if not isinstance(content, str) or not content.strip():
            raise ProviderEmptyOutputError("provider returned empty content")
        return content.strip()
