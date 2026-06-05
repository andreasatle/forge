import httpx

OLLAMA_BASE = "http://localhost:11434"


async def chat(model: str, prompt: str) -> str:
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
