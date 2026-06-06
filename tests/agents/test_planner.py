"""Tests for the planning agent retry logic."""

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
    """plan_agent returns COMPLETED on first attempt with no retries."""
    request = _make_request()
    with (
        patch("forge.agents.planner.llm.chat", new_callable=AsyncMock) as mock_chat,
        patch("forge.agents.planner.parse_plan", return_value=[]) as mock_parse,
    ):
        mock_chat.return_value = '{"tasks": []}'
        response = await plan_agent(request, _make_registry(), ["codebase"], {"codebase": "python"})

    assert response.status == ResponseStatus.COMPLETED
    assert mock_chat.call_count == 1
    assert mock_parse.call_count == 1


async def test_planner_retries_on_parse_failure_and_succeeds() -> None:
    """plan_agent retries when parse_plan raises and succeeds on the second attempt."""
    request = _make_request()
    with (
        patch("forge.agents.planner.llm.chat", new_callable=AsyncMock) as mock_chat,
        patch("forge.agents.planner.parse_plan") as mock_parse,
    ):
        mock_chat.return_value = "some response"
        mock_parse.side_effect = [ValueError("bad json"), []]
        response = await plan_agent(request, _make_registry(), ["codebase"], {"codebase": "python"}, max_retries=3)

    assert response.status == ResponseStatus.COMPLETED
    assert mock_chat.call_count == 2
    assert mock_parse.call_count == 2


async def test_planner_fails_after_max_retries() -> None:
    """plan_agent returns FAILED after exhausting all retry attempts."""
    request = _make_request()
    with (
        patch("forge.agents.planner.llm.chat", new_callable=AsyncMock) as mock_chat,
        patch("forge.agents.planner.parse_plan", side_effect=ValueError("bad json")),
    ):
        mock_chat.return_value = "bad response"
        response = await plan_agent(request, _make_registry(), ["codebase"], {"codebase": "python"}, max_retries=2)

    assert response.status == ResponseStatus.FAILED
    assert "planner failed after 2 attempts" in (response.error or "")


async def test_planner_prints_warning_on_retry(capsys: pytest.CaptureFixture) -> None:
    """plan_agent prints a retry warning with the attempt number on each failure."""
    request = _make_request()
    with (
        patch("forge.agents.planner.llm.chat", new_callable=AsyncMock) as mock_chat,
        patch("forge.agents.planner.parse_plan") as mock_parse,
    ):
        mock_chat.return_value = "some response"
        mock_parse.side_effect = [ValueError("oops"), []]
        await plan_agent(request, _make_registry(), ["codebase"], {"codebase": "python"}, max_retries=3)

    captured = capsys.readouterr()
    assert "planner retry 1" in captured.out
