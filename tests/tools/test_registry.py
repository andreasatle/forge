import pytest

from forge.tools.registry import Tool, ToolRegistry


async def _noop(**kwargs: str) -> str:
    return "noop"


def _make_tool(name: str) -> Tool:
    return Tool(
        name=name,
        description=f"tool {name}",
        parameters={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        fn=_noop,
    )


def test_registered_tool_is_retrievable_by_name() -> None:
    registry = ToolRegistry()
    tool = _make_tool("alpha")
    registry.register(tool)

    assert registry.get("alpha") is tool


def test_get_raises_on_unknown_tool() -> None:
    registry = ToolRegistry()

    with pytest.raises(KeyError, match="unknown tool"):
        registry.get("missing")


def test_get_many_raises_if_any_tool_missing() -> None:
    registry = ToolRegistry()
    registry.register(_make_tool("alpha"))

    with pytest.raises(KeyError, match="unknown tool"):
        registry.get_many(["alpha", "missing"])


def test_to_ollama_schema_produces_correct_format() -> None:
    registry = ToolRegistry()
    tool = _make_tool("alpha")
    registry.register(tool)

    schema = registry.to_ollama_schema(["alpha"])

    assert schema == [
        {
            "type": "function",
            "function": {
                "name": "alpha",
                "description": "tool alpha",
                "parameters": tool.parameters,
            },
        }
    ]
