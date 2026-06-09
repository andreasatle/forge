"""Tests for Tool dataclass and ToolRegistry register, get, and schema methods."""

import pytest
from pydantic import BaseModel

from forge.tools.registry import Tool, ToolRegistry


class _SimpleRequest(BaseModel):
    """Minimal request model used in registry unit tests."""

    x: str


class _SimpleResponse(BaseModel):
    """Minimal response model used in registry unit tests."""

    result: str


async def _noop(req: BaseModel) -> BaseModel:
    return _SimpleResponse(result="noop")


def _make_tool(name: str) -> Tool:
    return Tool(
        name=name,
        description=f"tool {name}",
        request_type=_SimpleRequest,
        response_type=_SimpleResponse,
        fn=_noop,
    )


def test_registered_tool_is_retrievable_by_name() -> None:
    """get() returns the same Tool object that was registered under that name."""
    registry = ToolRegistry()
    tool = _make_tool("alpha")
    registry.register(tool)

    assert registry.get("alpha") is tool


def test_get_raises_on_unknown_tool() -> None:
    """get() raises KeyError when no tool with the given name is registered."""
    registry = ToolRegistry()

    with pytest.raises(KeyError, match="unknown tool"):
        registry.get("missing")


def test_get_many_raises_if_any_tool_missing() -> None:
    """get_many() raises KeyError if any requested name is not registered."""
    registry = ToolRegistry()
    registry.register(_make_tool("alpha"))

    with pytest.raises(KeyError, match="unknown tool"):
        registry.get_many(["alpha", "missing"])
