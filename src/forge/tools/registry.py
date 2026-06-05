"""Tool dataclass and ToolRegistry for storing and retrieving agent tools."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass


@dataclass
class Tool:
    """A named async callable with a JSON schema description for LLM tool use."""

    name: str
    description: str
    parameters: dict  # type: ignore[type-arg]  # JSON schema
    fn: Callable[..., Awaitable[str]]


class ToolRegistry:
    """Registry that stores tools by name and serializes them to Ollama tool schema format."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add tool to the registry, keyed by tool.name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return the tool with the given name, raising KeyError if not registered."""
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name!r}")
        return self._tools[name]

    def get_many(self, names: list[str]) -> list[Tool]:
        """Return a list of tools for the given names in order, raising KeyError for any missing."""
        return [self.get(name) for name in names]

    def to_ollama_schema(self, names: list[str]) -> list[dict]:  # type: ignore[type-arg]
        """Return the Ollama function-tool schema for the named tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.get_many(names)
        ]
