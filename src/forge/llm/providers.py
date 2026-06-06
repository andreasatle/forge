"""LLM provider Protocol and implementations for Ollama, Claude, and OpenAI."""

import json
from dataclasses import dataclass
from typing import Protocol

import httpx

_ANTHROPIC_API = "https://api.anthropic.com/v1/messages"
_OPENAI_API = "https://api.openai.com/v1/chat/completions"
_ANTHROPIC_VERSION = "2023-06-01"


class LLMProvider(Protocol):
    """Protocol for LLM providers — single-turn chat and tool-augmented chat."""

    max_tokens: int

    async def chat(self, prompt: str, max_tokens: int) -> str:
        """Send a single-turn prompt and return the response text."""

    async def chat_with_tools(
        self, messages: list[dict], tools: list[dict], max_tokens: int
    ) -> tuple[str | None, list[dict]]:
        """Send a multi-turn conversation with tools; return (text, []) or (None, tool_calls)."""


@dataclass
class OllamaProvider:
    """Ollama LLM provider using the local REST API."""

    model: str
    max_tokens: int
    base_url: str = "http://localhost:11434"

    async def chat(self, prompt: str, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content", "")
        if not content or not content.strip():
            print(f"[debug] chat: empty content in response: {data!r}")
            raise ValueError(f"empty content in Ollama response: {data!r}")
        return content.strip()

    async def chat_with_tools(
        self, messages: list[dict], tools: list[dict], max_tokens: int
    ) -> tuple[str | None, list[dict]]:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
        data = response.json()
        message = data.get("message", {})
        tool_calls_raw = message.get("tool_calls")
        if tool_calls_raw:
            return None, [
                {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
                for tc in tool_calls_raw
            ]
        content = message.get("content", "")
        if not content or not content.strip():
            raise ValueError(f"empty content in Ollama response: {data!r}")
        return content.strip(), []


def _tools_to_claude_schema(tools: list[dict]) -> list[dict]:
    """Convert Ollama/OpenAI tool schema to Claude tool schema format."""
    result = []
    for t in tools:
        fn = t.get("function", t)
        result.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return result


def _messages_to_claude(messages: list[dict]) -> list[dict]:
    """Convert Ollama-format message history to Claude API format."""
    result = []
    for m in messages:
        role = m["role"]
        if role in ("user", "assistant") and "tool_calls" not in m:
            result.append({"role": role, "content": m.get("content", "")})
        elif role == "assistant" and "tool_calls" in m:
            content = [
                {
                    "type": "tool_use",
                    "id": f"tu_{i}",
                    "name": tc["function"]["name"],
                    "input": tc["function"]["arguments"]
                    if isinstance(tc["function"]["arguments"], dict)
                    else {},
                }
                for i, tc in enumerate(m["tool_calls"])
            ]
            result.append({"role": "assistant", "content": content})
        elif role == "tool":
            result.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tu_0", "content": m.get("content", "")}],
            })
    return result


@dataclass
class ClaudeProvider:
    """Anthropic Claude LLM provider using the Messages API."""

    model: str
    api_key: str
    max_tokens: int

    async def chat(self, prompt: str, max_tokens: int) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(_ANTHROPIC_API, headers=headers, json=payload)
            response.raise_for_status()
        data = response.json()
        content = next(
            (b["text"] for b in data.get("content", []) if b.get("type") == "text"), ""
        )
        if not content or not content.strip():
            raise ValueError(f"empty content in Claude response: {data!r}")
        return content.strip()

    async def chat_with_tools(
        self, messages: list[dict], tools: list[dict], max_tokens: int
    ) -> tuple[str | None, list[dict]]:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": _messages_to_claude(messages),
            "tools": _tools_to_claude_schema(tools),
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(_ANTHROPIC_API, headers=headers, json=payload)
            response.raise_for_status()
        data = response.json()
        tool_uses = [b for b in data.get("content", []) if b.get("type") == "tool_use"]
        if tool_uses:
            return None, [{"name": b["name"], "arguments": b["input"]} for b in tool_uses]
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

    async def chat(self, prompt: str, max_tokens: int) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(_OPENAI_API, headers=headers, json=payload)
            response.raise_for_status()
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content or not content.strip():
            raise ValueError(f"empty content in OpenAI response: {data!r}")
        return content.strip()

    async def chat_with_tools(
        self, messages: list[dict], tools: list[dict], max_tokens: int
    ) -> tuple[str | None, list[dict]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        openai_tools = [{"type": "function", "function": t.get("function", t)} for t in tools]
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "tools": openai_tools,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(_OPENAI_API, headers=headers, json=payload)
            response.raise_for_status()
        data = response.json()
        message = data.get("choices", [{}])[0].get("message", {})
        tool_calls_raw = message.get("tool_calls")
        if tool_calls_raw:
            return None, [
                {
                    "name": tc["function"]["name"],
                    "arguments": json.loads(tc["function"]["arguments"])
                    if isinstance(tc["function"]["arguments"], str)
                    else tc["function"]["arguments"],
                }
                for tc in tool_calls_raw
            ]
        content = message.get("content", "")
        if not content or not content.strip():
            raise ValueError(f"empty content in OpenAI response: {data!r}")
        return content.strip(), []
