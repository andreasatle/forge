"""LLM provider Protocol and implementations for Ollama, Claude, and OpenAI."""

import asyncio
from dataclasses import dataclass
from typing import Protocol

import httpx

_ANTHROPIC_API = "https://api.anthropic.com/v1/messages"
_OPENAI_API = "https://api.openai.com/v1/chat/completions"
_ANTHROPIC_VERSION = "2023-06-01"

_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 10.0
_JSON_INSTRUCTION = "You must respond with valid JSON only. No markdown, no explanation, no preamble."


async def _post_with_retry(client: httpx.AsyncClient, url: str, **kwargs: object) -> httpx.Response:
    """POST to url, retrying on 429/5xx with exponential backoff; raises immediately on other errors."""
    last_exc: httpx.HTTPStatusError
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
                print(f"[debug] HTTP {e.response.status_code} — retry {attempt + 1}/{_MAX_RETRIES - 1}, waiting {wait}s")
                await asyncio.sleep(wait)
    raise last_exc


class LLMProvider(Protocol):
    """Protocol for LLM providers — multi-turn chat."""

    max_tokens: int

    async def chat(self, messages: list[dict]) -> str:  # type: ignore[type-arg]
        """Send a multi-turn message list and return the response text."""


@dataclass
class OllamaProvider:
    """Ollama LLM provider using the local REST API."""

    model: str
    max_tokens: int
    base_url: str = "http://localhost:11434"

    async def chat(self, messages: list[dict]) -> str:  # type: ignore[type-arg]
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": self.max_tokens},
            "format": "json",
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await _post_with_retry(client, f"{self.base_url}/api/chat", json=payload)
        data = response.json()
        content = data.get("message", {}).get("content", "")
        if not content or not content.strip():
            raise ValueError("provider returned empty content")
        return content.strip()


@dataclass
class ClaudeProvider:
    """Anthropic Claude LLM provider using the Messages API."""

    model: str
    api_key: str
    max_tokens: int

    async def chat(self, messages: list[dict]) -> str:  # type: ignore[type-arg]
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        non_system = [m for m in messages if m["role"] != "system"]
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        payload: dict = {  # type: ignore[type-arg]
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": non_system,
        }
        payload["system"] = f"{system}\n\n{_JSON_INSTRUCTION}" if system else _JSON_INSTRUCTION
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await _post_with_retry(client, _ANTHROPIC_API, headers=headers, json=payload)
        data = response.json()
        content = next(
            (b["text"] for b in data.get("content", []) if b.get("type") == "text"), ""
        )
        if not content or not content.strip():
            raise ValueError("provider returned empty content")
        return content.strip()


@dataclass
class OpenAIProvider:
    """OpenAI LLM provider using the Chat Completions API."""

    model: str
    api_key: str
    max_tokens: int

    async def chat(self, messages: list[dict]) -> str:  # type: ignore[type-arg]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await _post_with_retry(client, _OPENAI_API, headers=headers, json=payload)
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content or not content.strip():
            raise ValueError("provider returned empty content")
        return content.strip()
