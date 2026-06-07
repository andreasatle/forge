"""LLM provider Protocol and implementations for Ollama, Claude, and OpenAI.

Canonical message format (OpenAI) — used everywhere outside providers:
  User:        {"role": "user", "content": "..."}
  Assistant:   {"role": "assistant", "content": "..."}
  Asst+tools:  {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "...", "type": "function",
                     "function": {"name": "...", "arguments": "..."}}]}
  Tool result: {"role": "tool", "tool_call_id": "...", "content": "..."}

Each provider converts canonical → its own wire format before sending and
converts its wire response → canonical before returning. No conversion happens
outside of providers.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Protocol

import httpx

_ANTHROPIC_API = "https://api.anthropic.com/v1/messages"
_OPENAI_API = "https://api.openai.com/v1/chat/completions"
_ANTHROPIC_VERSION = "2023-06-01"

_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 10.0


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
    """Protocol for LLM providers — multi-turn chat and tool-augmented chat."""

    max_tokens: int

    async def chat(self, messages: list[dict]) -> str:  # type: ignore[type-arg]
        """Send a multi-turn message list and return the response text."""

    async def chat_with_tools(
        self, messages: list[dict], tools: list[dict], max_tokens: int  # type: ignore[type-arg]
    ) -> tuple[str | None, list[dict]]:  # type: ignore[type-arg]
        """Send canonical messages with tools; return (text, []) or (None, canonical_tool_calls)."""


# --- Ollama wire-format helpers ---


def _canonical_to_ollama_messages(messages: list[dict]) -> list[dict]:  # type: ignore[type-arg]
    """Convert canonical (OpenAI) messages to Ollama wire format."""
    result = []
    id_to_name: dict[str, str] = {}
    for m in messages:
        role = m["role"]
        if role == "user":
            result.append({"role": "user", "content": m.get("content", "")})
        elif role == "assistant" and "tool_calls" not in m:
            result.append({"role": "assistant", "content": m.get("content") or ""})
        elif role == "assistant" and "tool_calls" in m:
            ollama_calls = []
            for tc in m["tool_calls"]:
                fn = tc["function"]
                id_to_name[tc["id"]] = fn["name"]
                args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                ollama_calls.append({"function": {"name": fn["name"], "arguments": args}})
            result.append({"role": "assistant", "content": "", "tool_calls": ollama_calls})
        elif role == "tool":
            name = id_to_name.get(m["tool_call_id"], m["tool_call_id"])
            result.append({"role": "tool", "name": name, "content": m.get("content", "")})
    return result


# --- Claude wire-format helpers ---


def _tools_to_claude_schema(tools: list[dict]) -> list[dict]:  # type: ignore[type-arg]
    """Convert OpenAI-format tool schemas to Claude tool schema format."""
    result = []
    for t in tools:
        fn = t.get("function", t)
        result.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return result


def _canonical_to_claude_messages(messages: list[dict]) -> list[dict]:  # type: ignore[type-arg]
    """Convert canonical (OpenAI) messages to Claude API wire format."""
    result = []
    for m in messages:
        role = m["role"]
        if role == "user":
            result.append({"role": "user", "content": m.get("content", "")})
        elif role == "assistant" and "tool_calls" not in m:
            result.append({"role": "assistant", "content": m.get("content") or ""})
        elif role == "assistant" and "tool_calls" in m:
            content = []
            for tc in m["tool_calls"]:
                fn = tc["function"]
                args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                content.append({"type": "tool_use", "id": tc["id"], "name": fn["name"], "input": args})
            result.append({"role": "assistant", "content": content})
        elif role == "tool":
            result.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m.get("content", "")}],
            })
    return result


def _sanitize_args(raw: dict | str) -> str:
    """Strip trailing '=' from argument keys and return a JSON string."""
    d = json.loads(raw) if isinstance(raw, str) else raw
    return json.dumps({k.rstrip("="): v for k, v in d.items()})


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
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await _post_with_retry(client, f"{self.base_url}/api/chat", json=payload)
        data = response.json()
        content = data.get("message", {}).get("content", "")
        if not content or not content.strip():
            raise ValueError(f"empty content in Ollama response: {data!r}")
        return content.strip()

    async def chat_with_tools(
        self, messages: list[dict], tools: list[dict], max_tokens: int  # type: ignore[type-arg]
    ) -> tuple[str | None, list[dict]]:  # type: ignore[type-arg]
        payload = {
            "model": self.model,
            "messages": _canonical_to_ollama_messages(messages),
            "tools": tools,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await _post_with_retry(client, f"{self.base_url}/api/chat", json=payload)
        data = response.json()
        message = data.get("message", {})
        tool_calls_raw = message.get("tool_calls")
        if tool_calls_raw:
            return None, [
                {
                    "id": f"call_{tc['function']['name']}_{i}",
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": _sanitize_args(tc["function"]["arguments"]),
                    },
                }
                for i, tc in enumerate(tool_calls_raw)
            ]
        content = message.get("content", "")
        if not content or not content.strip():
            raise ValueError(f"empty content in Ollama response: {data!r}")
        return content.strip(), []


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
        if system:
            payload["system"] = system
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await _post_with_retry(client, _ANTHROPIC_API, headers=headers, json=payload)
        data = response.json()
        content = next(
            (b["text"] for b in data.get("content", []) if b.get("type") == "text"), ""
        )
        if not content or not content.strip():
            raise ValueError(f"empty content in Claude response: {data!r}")
        return content.strip()

    async def chat_with_tools(
        self, messages: list[dict], tools: list[dict], max_tokens: int  # type: ignore[type-arg]
    ) -> tuple[str | None, list[dict]]:  # type: ignore[type-arg]
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": _canonical_to_claude_messages(messages),
            "tools": _tools_to_claude_schema(tools),
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await _post_with_retry(client, _ANTHROPIC_API, headers=headers, json=payload)
        data = response.json()
        tool_uses = [b for b in data.get("content", []) if b.get("type") == "tool_use"]
        if tool_uses:
            return None, [
                {
                    "id": tu["id"],
                    "type": "function",
                    "function": {
                        "name": tu["name"],
                        "arguments": _sanitize_args(tu["input"]),
                    },
                }
                for tu in tool_uses
            ]
        text = next(
            (b["text"] for b in data.get("content", []) if b.get("type") == "text"), ""
        )
        if not text or not text.strip():
            raise ValueError(f"empty content in Claude response: {data!r}")
        return text.strip(), []


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
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await _post_with_retry(client, _OPENAI_API, headers=headers, json=payload)
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content or not content.strip():
            raise ValueError(f"empty content in OpenAI response: {data!r}")
        return content.strip()

    async def chat_with_tools(
        self, messages: list[dict], tools: list[dict], max_tokens: int  # type: ignore[type-arg]
    ) -> tuple[str | None, list[dict]]:  # type: ignore[type-arg]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        openai_tools = [{"type": "function", "function": t.get("function", t)} for t in tools]
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,  # canonical IS OpenAI format — no conversion needed
            "tools": openai_tools,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await _post_with_retry(client, _OPENAI_API, headers=headers, json=payload)
        data = response.json()
        message = data.get("choices", [{}])[0].get("message", {})
        tool_calls_raw = message.get("tool_calls")
        if tool_calls_raw:
            return None, [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": _sanitize_args(tc["function"]["arguments"]),
                    },
                }
                for tc in tool_calls_raw
            ]
        content = message.get("content", "")
        if not content or not content.strip():
            raise ValueError(f"empty content in OpenAI response: {data!r}")
        return content.strip(), []
