"""Tests for the run_agent base engine — retry, tool loop, and error handling."""

from unittest.mock import AsyncMock, MagicMock

from pydantic import BaseModel

from forge.agents.base import run_agent
from forge.core.models import (
    AgentRequest,
    AgentType,
    PlanSpec,
    RequestSource,
    ResponseStatus,
    WorkSpec,
)
from forge.tools.registry import Tool, ToolRegistry


class _DoThingRequest(BaseModel):
    pass


class _DoThingResponse(BaseModel):
    result: str


def _plan_request() -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(northstar="do something"),
    )


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


def _mock_provider(chat_return: str = "result text") -> MagicMock:
    provider = MagicMock()
    provider.max_tokens = 8192
    provider.chat = AsyncMock(return_value=chat_return)
    provider.chat_with_tools = AsyncMock(return_value=(chat_return, []))
    return provider


async def test_run_agent_wrong_spec_type_returns_failed():
    """run_agent returns FAILED when the request spec does not match the expected type."""
    request = _plan_request()
    response = await run_agent(request, WorkSpec, _mock_provider(), "prompt")
    assert response.status == ResponseStatus.FAILED


async def test_run_agent_no_tools_single_chat_call():
    """run_agent makes exactly one LLM call when no tools are provided."""
    request = _plan_request()
    provider = _mock_provider()
    response = await run_agent(request, PlanSpec, provider, "prompt")
    assert response.status == ResponseStatus.COMPLETED
    assert provider.chat.call_count == 1


async def test_run_agent_tool_loop_executes_and_feeds_back():
    """run_agent executes tool calls and feeds results back until a text response arrives."""
    request = _work_request()
    registry = ToolRegistry()
    mock_fn = AsyncMock(return_value=_DoThingResponse(result="tool_result"))
    registry.register(Tool(
        name="do_thing",
        description="does thing",
        request_type=_DoThingRequest,
        response_type=_DoThingResponse,
        fn=mock_fn,
    ))
    schema = registry.to_tool_schema(["do_thing"])

    provider = _mock_provider()
    provider.chat_with_tools = AsyncMock(side_effect=[
        (None, [{"id": "call_do_thing_0", "type": "function", "function": {"name": "do_thing", "arguments": "{}"}}]),
        ("final answer", []),
    ])

    response = await run_agent(
        request, WorkSpec, provider, "prompt", tools=registry, tool_schema=schema
    )

    assert response.status == ResponseStatus.COMPLETED
    assert provider.chat_with_tools.call_count == 2
    assert mock_fn.call_count == 1


async def test_run_agent_value_error_retries_with_correction_prompt():
    """run_agent calls correction_prompt_fn and retries when ValueError is raised."""
    request = _plan_request()
    correction_fn = MagicMock(return_value="corrected prompt")

    provider = _mock_provider()
    provider.chat = AsyncMock(side_effect=[ValueError("bad format"), "good response"])

    response = await run_agent(
        request, PlanSpec, provider, "prompt", max_retries=3, correction_prompt_fn=correction_fn
    )

    assert response.status == ResponseStatus.COMPLETED
    assert provider.chat.call_count == 2
    assert correction_fn.call_count == 1


async def test_run_agent_non_value_error_returns_failed_immediately():
    """run_agent returns FAILED immediately on non-ValueError without retrying."""
    request = _plan_request()
    correction_fn = MagicMock(return_value="corrected prompt")

    provider = _mock_provider()
    provider.chat = AsyncMock(side_effect=RuntimeError("network error"))

    response = await run_agent(
        request, PlanSpec, provider, "prompt", max_retries=3, correction_prompt_fn=correction_fn
    )

    assert response.status == ResponseStatus.FAILED
    assert provider.chat.call_count == 1
    assert correction_fn.call_count == 0


async def test_run_agent_exhausted_retries_returns_failed():
    """run_agent returns FAILED with an error message after exhausting all retries."""
    provider = _mock_provider()
    provider.chat = AsyncMock(side_effect=ValueError("always fails"))

    response = await run_agent(_plan_request(), PlanSpec, provider, "prompt", max_retries=2)

    assert response.status == ResponseStatus.FAILED
    assert "2 attempts" in (response.error or "")


async def test_run_agent_success_returns_completed():
    """run_agent returns COMPLETED with the result text in delta on success."""
    provider = _mock_provider("great result")
    response = await run_agent(_plan_request(), PlanSpec, provider, "prompt")

    assert response.status == ResponseStatus.COMPLETED
    assert response.delta == {"result": "great result"}
