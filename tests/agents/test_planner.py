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


def _make_registry() -> MagicMock:
    registry = MagicMock()
    registry.names.return_value = ["coding", "document", "audit"]
    return registry


def _mock_provider(chat_return: str = '{"tasks": []}') -> MagicMock:
    provider = MagicMock()
    provider.max_tokens = 8192
    provider.chat = AsyncMock(return_value=chat_return)
    return provider


async def test_planner_succeeds_on_first_attempt() -> None:
    """plan_agent returns COMPLETED when the LLM call and parse both succeed."""
    request = _make_request()
    provider = _mock_provider()
    with MagicMock() as mock_parse:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("forge.agents.planner.parse_plan", MagicMock(return_value=[]))
            response = await plan_agent(request, _make_registry(), ["codebase"], {"codebase": "python"}, provider)

    assert response.status == ResponseStatus.COMPLETED
    assert provider.chat.call_count == 1


async def test_planner_parse_failure_triggers_retry() -> None:
    """plan_agent retries when parse_plan raises and succeeds on the next attempt."""
    request = _make_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(return_value="some response")

    with pytest.MonkeyPatch.context() as mp:
        mock_parse = MagicMock(side_effect=[ValueError("bad json"), []])
        mp.setattr("forge.agents.planner.parse_plan", mock_parse)
        response = await plan_agent(
            request, _make_registry(), ["codebase"], {"codebase": "python"}, provider, max_retries=3
        )

    assert response.status == ResponseStatus.COMPLETED
    assert provider.chat.call_count == 2
    assert mock_parse.call_count == 2


async def test_planner_parse_failure_exhausts_retries_returns_failed() -> None:
    """plan_agent returns FAILED when parse_plan always raises, exhausting all retries."""
    request = _make_request()
    provider = _mock_provider("bad response")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("forge.agents.planner.parse_plan", MagicMock(side_effect=ValueError("bad json")))
        response = await plan_agent(
            request, _make_registry(), ["codebase"], {"codebase": "python"}, provider, max_retries=2
        )

    assert response.status == ResponseStatus.FAILED
    assert provider.chat.call_count == 2


async def test_planner_llm_failure_returns_failed() -> None:
    """plan_agent returns FAILED after all LLM retry attempts are exhausted."""
    request = _make_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(side_effect=ValueError("connection error"))

    response = await plan_agent(
        request, _make_registry(), ["codebase"], {"codebase": "python"}, provider, max_retries=2
    )

    assert response.status == ResponseStatus.FAILED
    assert "2 attempts" in (response.error or "")


async def test_planner_retries_llm_failure_and_succeeds() -> None:
    """plan_agent retries when the LLM call raises ValueError and succeeds on the second attempt."""
    request = _make_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(side_effect=[ValueError("timeout"), '{"tasks": []}'])

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("forge.agents.planner.parse_plan", MagicMock(return_value=[]))
        response = await plan_agent(
            request, _make_registry(), ["codebase"], {"codebase": "python"}, provider, max_retries=3
        )

    assert response.status == ResponseStatus.COMPLETED
    assert provider.chat.call_count == 2


async def test_planner_prints_warning_on_retry(capsys: pytest.CaptureFixture) -> None:
    """plan_agent prints a retry warning on each failure before retrying."""
    request = _make_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(side_effect=[ValueError("oops"), '{"tasks": []}'])

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("forge.agents.planner.parse_plan", MagicMock(return_value=[]))
        await plan_agent(request, _make_registry(), ["codebase"], {"codebase": "python"}, provider, max_retries=3)

    captured = capsys.readouterr()
    assert "agent retry 1" in captured.out
