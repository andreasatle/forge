"""Factory for creating LLM providers from a provider/model string."""

import os
from pathlib import Path

from forge.llm.providers import ClaudeProvider, LLMProvider, OllamaProvider, OpenAIProvider

_dotenv_loaded = False


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        _load_dotenv_manually()


def _load_dotenv_manually() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def make_provider(model_string: str, max_tokens: int) -> LLMProvider:
    """Parse a provider/model string and return the appropriate LLMProvider.

    Format: provider/model — e.g. ollama/gemma4:e4b, claude/claude-sonnet-4-20250514, openai/gpt-4o
    """
    global _dotenv_loaded
    if not _dotenv_loaded:
        _load_dotenv()
        _dotenv_loaded = True

    if "/" not in model_string:
        raise ValueError(f"invalid model string {model_string!r}: expected 'provider/model' format")
    provider, _, model = model_string.partition("/")

    if provider == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaProvider(model=model, max_tokens=max_tokens, base_url=base_url)
    if provider == "claude":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        return ClaudeProvider(model=model, api_key=api_key, max_tokens=max_tokens)
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return OpenAIProvider(model=model, api_key=api_key, max_tokens=max_tokens)

    raise ValueError(f"unknown provider {provider!r}: expected 'ollama', 'claude', or 'openai'")
