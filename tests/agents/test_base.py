"""Tests for the run_agent base helper and its error-handling behaviour."""

import pytest

from forge.agents.base import run_agent
from forge.core.models import (
    AgentRequest,
    AgentResponse,
    AgentType,
    PlanSpec,
    RequestSource,
    ResponseStatus,
    WorkSpec,
)


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
        spec=WorkSpec(objective="write code", success_condition="tests pass", adapter="coding"),
    )


@pytest.mark.asyncio
async def test_run_agent_wrong_spec_type_returns_failed():
    """run_agent returns FAILED when the request spec does not match the expected type."""
    request = _plan_request()
    response = await run_agent(request, WorkSpec, lambda spec: None)
    assert response.status == ResponseStatus.FAILED


@pytest.mark.asyncio
async def test_run_agent_exception_in_build_returns_failed():
    """run_agent returns FAILED when the build_response callback raises an exception."""
    request = _plan_request()

    async def exploding(spec: PlanSpec) -> AgentResponse:
        raise ValueError("something broke")

    response = await run_agent(request, PlanSpec, exploding)
    assert response.status == ResponseStatus.FAILED


@pytest.mark.asyncio
async def test_run_agent_success_returns_build_response():
    """run_agent returns the exact AgentResponse produced by build_response on success."""
    request = _plan_request()
    expected = AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED)

    async def build(spec: PlanSpec) -> AgentResponse:
        return expected

    response = await run_agent(request, PlanSpec, build)
    assert response is expected


@pytest.mark.asyncio
async def test_run_agent_error_message_includes_type_and_message():
    """run_agent error field includes both the exception type name and its message."""
    request = _plan_request()

    async def exploding(spec: PlanSpec) -> AgentResponse:
        raise RuntimeError("oops bad state")

    response = await run_agent(request, PlanSpec, exploding)
    assert response.error is not None
    assert "RuntimeError" in response.error
    assert "oops bad state" in response.error
