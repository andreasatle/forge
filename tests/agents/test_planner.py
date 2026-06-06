"""Tests for the planning agent."""

from unittest.mock import AsyncMock, MagicMock, patch

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


async def test_planner_succeeds_on_first_attempt() -> None:
    """plan_agent returns COMPLETED when the LLM call and parse both succeed."""
    request = _make_request()
    with (
        patch("forge.agents.base.llm.chat", new_callable=AsyncMock) as mock_chat,
        patch("forge.agents.planner.parse_plan", return_value=[]) as mock_parse,
    ):
        mock_chat.return_value = '{"tasks": []}'
        response = await plan_agent(request, _make_registry(), ["codebase"], {"codebase": "python"})

    assert response.status == ResponseStatus.COMPLETED
    assert mock_chat.call_count == 1
    assert mock_parse.call_count == 1


async def test_planner_parse_failure_triggers_retry() -> None:
    """plan_agent retries when parse_plan raises and succeeds on the next attempt."""
    request = _make_request()
    with (
        patch("forge.agents.base.llm.chat", new_callable=AsyncMock) as mock_chat,
        patch("forge.agents.planner.parse_plan") as mock_parse,
    ):
        mock_chat.return_value = "some response"
        mock_parse.side_effect = [ValueError("bad json"), []]
        response = await plan_agent(
            request, _make_registry(), ["codebase"], {"codebase": "python"}, max_retries=3
        )

    assert response.status == ResponseStatus.COMPLETED
    assert mock_chat.call_count == 2
    assert mock_parse.call_count == 2


async def test_planner_parse_failure_exhausts_retries_returns_failed() -> None:
    """plan_agent returns FAILED when parse_plan always raises, exhausting all retries."""
    request = _make_request()
    with (
        patch("forge.agents.base.llm.chat", new_callable=AsyncMock) as mock_chat,
        patch("forge.agents.planner.parse_plan", side_effect=ValueError("bad json")),
    ):
        mock_chat.return_value = "bad response"
        response = await plan_agent(
            request, _make_registry(), ["codebase"], {"codebase": "python"}, max_retries=2
        )

    assert response.status == ResponseStatus.FAILED
    assert mock_chat.call_count == 2


async def test_planner_llm_failure_returns_failed() -> None:
    """plan_agent returns FAILED after all LLM retry attempts are exhausted."""
    request = _make_request()
    with patch("forge.agents.base.llm.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.side_effect = ValueError("connection error")
        response = await plan_agent(
            request, _make_registry(), ["codebase"], {"codebase": "python"}, max_retries=2
        )

    assert response.status == ResponseStatus.FAILED
    assert "2 attempts" in (response.error or "")


async def test_planner_retries_llm_failure_and_succeeds() -> None:
    """plan_agent retries when the LLM call raises ValueError and succeeds on the second attempt."""
    request = _make_request()
    with (
        patch("forge.agents.base.llm.chat", new_callable=AsyncMock) as mock_chat,
        patch("forge.agents.planner.parse_plan", return_value=[]),
    ):
        mock_chat.side_effect = [ValueError("timeout"), '{"tasks": []}']
        response = await plan_agent(
            request, _make_registry(), ["codebase"], {"codebase": "python"}, max_retries=3
        )

    assert response.status == ResponseStatus.COMPLETED
    assert mock_chat.call_count == 2


async def test_planner_prints_warning_on_retry(capsys: pytest.CaptureFixture) -> None:
    """plan_agent prints a retry warning on each failure before retrying."""
    request = _make_request()
    with (
        patch("forge.agents.base.llm.chat", new_callable=AsyncMock) as mock_chat,
        patch("forge.agents.planner.parse_plan", return_value=[]),
    ):
        mock_chat.side_effect = [ValueError("oops"), '{"tasks": []}']
        await plan_agent(request, _make_registry(), ["codebase"], {"codebase": "python"}, max_retries=3)

    captured = capsys.readouterr()
    assert "agent retry 1" in captured.out
