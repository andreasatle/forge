"""Async Ollama HTTP client for single-turn and tool-augmented chat."""

import httpx

OLLAMA_BASE = "http://localhost:11434"


async def chat(model: str, prompt: str) -> str:
    """Send a single-turn prompt to the Ollama model and return the response content."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload)
        response.raise_for_status()
    data = response.json()
    content = data.get("message", {}).get("content")
    if not content:
        raise ValueError(f"no content in Ollama response: {data!r}")
    return content


async def chat_with_tools(
    model: str,
    messages: list[dict],  # type: ignore[type-arg]
    tools: list[dict],  # type: ignore[type-arg]
) -> tuple[str | None, list[dict]]:  # type: ignore[type-arg]
    """Send a multi-turn conversation with tool schemas; return (text, []) or (None, tool_calls)."""
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload)
        response.raise_for_status()
    data = response.json()
    message = data.get("message", {})
    tool_calls_raw = message.get("tool_calls")
    if tool_calls_raw:
        tool_calls = [
            {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
            for tc in tool_calls_raw
        ]
        return None, tool_calls
    content = message.get("content")
    if not content:
        raise ValueError(f"no content in Ollama response: {data!r}")
    return content, []
