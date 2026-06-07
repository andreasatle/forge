"""Tests for the run_agent base engine — plain chat loop with structured JSON parsing."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from forge.agents.base import _execute_tool, _parse_response, run_agent
from forge.core.models import (
    AgentRequest,
    AgentResponse,
    AgentType,
    DeltaState,
    PlanResponse,
    RequestSource,
    ResponseStatus,
    ToolCallRequest,
    WorkSpec,
)
from forge.tools.registry import Tool, ToolRegistry


class _DoThingRequest(BaseModel):
    pass


class _DoThingResponse(BaseModel):
    result: str


def _work_request() -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="write code",
            success_condition="tests pass",
            adapter="coding",
            artifact="main",
        ),
    )


def _mock_provider(chat_return: str = "{}") -> MagicMock:
    provider = MagicMock()
    provider.max_tokens = 8192
    provider.chat = AsyncMock(return_value=chat_return)
    return provider


def _make_registry() -> tuple[ToolRegistry, AsyncMock]:
    registry = ToolRegistry()
    mock_fn: AsyncMock = AsyncMock(return_value=_DoThingResponse(result="done"))
    registry.register(Tool(
        name="do_thing",
        description="does a thing",
        request_type=_DoThingRequest,
        response_type=_DoThingResponse,
        fn=mock_fn,
    ))
    return registry, mock_fn


# --- _parse_response ---


def test_parse_response_correctly_parses_tool_call_request():
    """_parse_response returns ToolCallRequest when kind == 'tool_call'."""
    raw = '{"kind": "tool_call", "name": "my_tool", "arguments": {}}'
    result = _parse_response(raw, None, DeltaState)
    assert isinstance(result, ToolCallRequest)
    assert result.name == "my_tool"


def test_parse_response_correctly_parses_delta_state_as_final_response():
    """_parse_response returns DeltaState when JSON matches DeltaState schema."""
    raw = '{"edits": [], "new_files": [], "dependencies": []}'
    result = _parse_response(raw, None, DeltaState)
    assert isinstance(result, DeltaState)


def test_parse_response_correctly_parses_plan_response_as_final_response():
    """_parse_response returns PlanResponse when final_response_type is PlanResponse."""
    raw = '{"kind": "plan", "tasks": []}'
    result = _parse_response(raw, None, PlanResponse)
    assert isinstance(result, PlanResponse)
    assert result.tasks == []


def test_parse_response_raises_value_error_on_unknown_format():
    """_parse_response raises ValueError when the response is not valid JSON."""
    with pytest.raises(ValueError, match="not valid JSON"):
        _parse_response("not json at all", None, DeltaState)


# --- _execute_tool ---


async def test_execute_tool_returns_correct_tool_call_response():
    """_execute_tool returns ToolCallResponse with success=True on valid tool call."""
    registry, mock_fn = _make_registry()
    request = ToolCallRequest(kind="tool_call", name="do_thing", arguments={})
    response = await _execute_tool(request, registry)
    assert response.success is True
    assert response.result == {"result": "done"}
    assert mock_fn.call_count == 1


async def test_execute_tool_returns_error_tool_call_response_on_unknown_tool():
    """_execute_tool returns ToolCallResponse with success=False for an unregistered tool."""
    registry = ToolRegistry()
    request = ToolCallRequest(kind="tool_call", name="nonexistent", arguments={})
    response = await _execute_tool(request, registry)
    assert response.success is False
    assert response.error is not None


# --- run_agent ---


async def test_run_agent_routes_tool_calls_correctly():
    """run_agent executes tool calls and feeds results back before the final response."""
    registry, mock_fn = _make_registry()
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(side_effect=[
        '{"kind": "tool_call", "name": "do_thing", "arguments": {}}',
        '{"edits": [], "new_files": [], "dependencies": []}',
    ])

    response = await run_agent(request, WorkSpec, provider, "prompt", tools=registry)

    assert response.status == ResponseStatus.COMPLETED
    assert provider.chat.call_count == 2
    assert mock_fn.call_count == 1


async def test_run_agent_returns_agent_response_on_final_response():
    """run_agent returns COMPLETED AgentResponse when the LLM returns valid final JSON."""
    request = _work_request()
    provider = _mock_provider('{"edits": [], "new_files": [], "dependencies": []}')

    response = await run_agent(request, WorkSpec, provider, "prompt")

    assert isinstance(response, AgentResponse)
    assert response.status == ResponseStatus.COMPLETED


async def test_run_agent_retries_on_invalid_format():
    """run_agent retries when the LLM returns invalid JSON and succeeds on the next attempt."""
    request = _work_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(side_effect=[
        "not valid json",
        '{"edits": [], "new_files": [], "dependencies": []}',
    ])

    response = await run_agent(request, WorkSpec, provider, "prompt", max_retries=3)

    assert response.status == ResponseStatus.COMPLETED
    assert provider.chat.call_count == 2


async def test_run_agent_returns_failed_after_max_iterations():
    """run_agent returns FAILED when the tool iteration limit is exhausted."""
    registry, _ = _make_registry()
    request = _work_request()
    provider = _mock_provider('{"kind": "tool_call", "name": "do_thing", "arguments": {}}')

    response = await run_agent(
        request, WorkSpec, provider, "prompt",
        tools=registry,
        max_tool_iterations=2,
        max_retries=0,
    )

    assert response.status == ResponseStatus.FAILED
    assert "exceeded" in (response.error or "")
