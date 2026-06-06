"""Tests for the run_agent base engine — retry, tool loop, and error handling."""

from unittest.mock import AsyncMock, MagicMock, patch

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


async def test_run_agent_wrong_spec_type_returns_failed():
    """run_agent returns FAILED when the request spec does not match the expected type."""
    request = _plan_request()
    response = await run_agent(request, WorkSpec, "model", "prompt")
    assert response.status == ResponseStatus.FAILED


async def test_run_agent_no_tools_single_chat_call():
    """run_agent makes exactly one LLM call when no tools are provided."""
    request = _plan_request()
    with patch("forge.agents.base.llm.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = "result text"
        response = await run_agent(request, PlanSpec, "model", "prompt")
    assert response.status == ResponseStatus.COMPLETED
    assert mock_chat.call_count == 1


async def test_run_agent_tool_loop_executes_and_feeds_back():
    """run_agent executes tool calls and feeds results back until a text response arrives."""
    request = _work_request()
    registry = ToolRegistry()
    mock_fn = AsyncMock(return_value="tool_result")
    registry.register(Tool(name="do_thing", description="does thing", parameters={}, fn=mock_fn))
    schema = registry.to_ollama_schema(["do_thing"])

    with patch("forge.agents.base.llm.chat_with_tools", new_callable=AsyncMock) as mock_chat:
        mock_chat.side_effect = [
            (None, [{"name": "do_thing", "arguments": {}}]),
            ("final answer", []),
        ]
        response = await run_agent(
            request, WorkSpec, "model", "prompt", tools=registry, tool_schema=schema
        )

    assert response.status == ResponseStatus.COMPLETED
    assert mock_chat.call_count == 2
    assert mock_fn.call_count == 1


async def test_run_agent_value_error_retries_with_correction_prompt():
    """run_agent calls correction_prompt_fn and retries when ValueError is raised."""
    request = _plan_request()
    correction_fn = MagicMock(return_value="corrected prompt")

    with patch("forge.agents.base.llm.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.side_effect = [ValueError("bad format"), "good response"]
        response = await run_agent(
            request, PlanSpec, "model", "prompt", max_retries=3, correction_prompt_fn=correction_fn
        )

    assert response.status == ResponseStatus.COMPLETED
    assert mock_chat.call_count == 2
    assert correction_fn.call_count == 1


async def test_run_agent_non_value_error_returns_failed_immediately():
    """run_agent returns FAILED immediately on non-ValueError without retrying."""
    request = _plan_request()
    correction_fn = MagicMock(return_value="corrected prompt")

    with patch("forge.agents.base.llm.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.side_effect = RuntimeError("network error")
        response = await run_agent(
            request, PlanSpec, "model", "prompt", max_retries=3, correction_prompt_fn=correction_fn
        )

    assert response.status == ResponseStatus.FAILED
    assert mock_chat.call_count == 1
    assert correction_fn.call_count == 0


async def test_run_agent_exhausted_retries_returns_failed():
    """run_agent returns FAILED with an error message after exhausting all retries."""
    request = _plan_request()

    with patch("forge.agents.base.llm.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.side_effect = ValueError("always fails")
        response = await run_agent(request, PlanSpec, "model", "prompt", max_retries=2)

    assert response.status == ResponseStatus.FAILED
    assert "2 attempts" in (response.error or "")


async def test_run_agent_success_returns_completed():
    """run_agent returns COMPLETED with the result text in delta on success."""
    request = _plan_request()

    with patch("forge.agents.base.llm.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = "great result"
        response = await run_agent(request, PlanSpec, "model", "prompt")

    assert response.status == ResponseStatus.COMPLETED
    assert response.delta == {"result": "great result"}
