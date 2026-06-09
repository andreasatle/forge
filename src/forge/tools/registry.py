"""Tool dataclass and ToolRegistry for storing and retrieving agent tools."""

from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass

from pydantic import BaseModel


@dataclass(frozen=True)
class Tool:
    """A named async callable with typed request/response Pydantic models for LLM tool use."""

    name: str
    description: str
    request_type: type[BaseModel]
    response_type: type[BaseModel]
    fn: Callable[[BaseModel], Awaitable[BaseModel]]


class ToolRegistry:
    """Registry that stores tools by name and serializes them to Ollama tool schema format."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def __bool__(self) -> bool:
        """Return True when at least one tool is registered."""
        return bool(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        """Iterate over registered tools."""
        return iter(self._tools.values())

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
