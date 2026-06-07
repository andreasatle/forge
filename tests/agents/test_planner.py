"""Tests for the planning agent."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from forge.agents.planner import plan_agent
from forge.core.models import AgentRequest, AgentType, PlanSpec, RequestSource, ResponseStatus


def _make_request() -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(northstar="build a web scraper"),
    )


def _mock_provider(chat_return: str = '{"kind": "plan", "tasks": []}') -> MagicMock:
    provider = MagicMock()
    provider.max_tokens = 8192
    provider.chat = AsyncMock(return_value=chat_return)
    return provider


async def test_planner_succeeds_on_first_attempt() -> None:
    """plan_agent returns COMPLETED when the LLM returns valid PlanResponse JSON."""
    request = _make_request()
    provider = _mock_provider()

    response = await plan_agent(request, ["codebase"], {"codebase": "python"}, provider)

    assert response.status == ResponseStatus.COMPLETED
    assert provider.chat.call_count == 1


async def test_planner_format_failure_triggers_retry() -> None:
    """plan_agent retries when the LLM returns invalid JSON and succeeds on the next attempt."""
    request = _make_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(side_effect=[
        "not valid json",
        '{"kind": "plan", "tasks": []}',
    ])

    response = await plan_agent(request, ["codebase"], {"codebase": "python"}, provider, max_retries=3)

    assert response.status == ResponseStatus.COMPLETED
    assert provider.chat.call_count == 2


async def test_planner_format_failure_exhausts_retries_returns_failed() -> None:
    """plan_agent returns FAILED when the LLM always returns unparseable JSON."""
    request = _make_request()
    provider = _mock_provider("not json")

    response = await plan_agent(request, ["codebase"], {"codebase": "python"}, provider, max_retries=2)

    assert response.status == ResponseStatus.FAILED
    assert provider.chat.call_count == 3  # initial + 2 retries


async def test_planner_llm_error_returns_failed_immediately() -> None:
    """plan_agent returns FAILED immediately when the LLM raises an exception."""
    request = _make_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(side_effect=RuntimeError("network error"))

    response = await plan_agent(request, ["codebase"], {"codebase": "python"}, provider, max_retries=3)

    assert response.status == ResponseStatus.FAILED
    assert provider.chat.call_count == 1


async def test_planner_prints_warning_on_retry(capsys: pytest.CaptureFixture) -> None:
    """plan_agent prints a retry warning when the LLM returns invalid JSON."""
    request = _make_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(side_effect=[
        "not valid json",
        '{"kind": "plan", "tasks": []}',
    ])

    await plan_agent(request, ["codebase"], {"codebase": "python"}, provider, max_retries=3)

    captured = capsys.readouterr()
    assert "agent retry 1" in captured.out
