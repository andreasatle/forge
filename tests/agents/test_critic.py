"""Tests for critic_agent."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from forge.agents.critic import critic_agent
from forge.core.models import (
    AgentRequest,
    AgentType,
    CriticDisposition,
    CriticFinding,
    DeltaState,
    FileWrite,
    RequestSource,
    StateView,
    WorkSpec,
)


def _request() -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="write a hello world function",
            success_condition="function prints Hello, World!",
            adapter="coding",
            artifact="codebase",
        ),
    )


def _state_view() -> StateView:
    return StateView(artifact_name="codebase", language="python", files=[], dependencies=[])


def _delta_with_file() -> DeltaState:
    return DeltaState(new_files=[FileWrite(path="main.py", content='print("Hello, World!")')])


def _provider(response_json: str) -> MagicMock:
    provider = MagicMock()
    provider.chat = AsyncMock(return_value=response_json)
    return provider


async def test_critic_agent_returns_accept_finding() -> None:
    """critic_agent returns a CriticFinding with ACCEPT when work meets the success condition."""
    finding_json = json.dumps(
        {"disposition": "accept", "rationale": "The file prints Hello, World!", "hints": []}
    )
    result = await critic_agent(
        _request(), _state_view(), _delta_with_file(), _provider(finding_json)
    )
    assert isinstance(result, CriticFinding)
    assert result.disposition == CriticDisposition.ACCEPT


async def test_critic_agent_returns_revise_finding_with_hints() -> None:
    """critic_agent returns CriticFinding with REVISE and hints when work is incomplete."""
    finding_json = json.dumps(
        {
            "disposition": "revise",
            "rationale": "Missing punctuation.",
            "hints": ["Add an exclamation mark"],
        }
    )
    result = await critic_agent(
        _request(), _state_view(), _delta_with_file(), _provider(finding_json)
    )
    assert result.disposition == CriticDisposition.REVISE
    assert len(result.hints) == 1


async def test_critic_agent_returns_reject_finding() -> None:
    """critic_agent returns CriticFinding with REJECT when no output was produced."""
    finding_json = json.dumps(
        {
            "disposition": "reject",
            "rationale": "No file was produced.",
            "hints": ["Create main.py", "Print Hello, World!"],
        }
    )
    result = await critic_agent(_request(), _state_view(), DeltaState(), _provider(finding_json))
    assert result.disposition == CriticDisposition.REJECT


async def test_critic_agent_retries_on_invalid_json() -> None:
    """critic_agent retries when the provider returns invalid JSON, then succeeds."""
    good_json = json.dumps({"disposition": "accept", "rationale": "Looks good.", "hints": []})
    provider = MagicMock()
    provider.chat = AsyncMock(side_effect=["not valid json", good_json])
    result = await critic_agent(_request(), _state_view(), _delta_with_file(), provider)
    assert isinstance(result, CriticFinding)
    assert provider.chat.call_count == 2


async def test_critic_agent_raises_after_max_retries_exceeded() -> None:
    """critic_agent raises ValueError after exhausting retries on persistent bad JSON."""
    provider = MagicMock()
    provider.chat = AsyncMock(return_value="invalid json always")
    with pytest.raises(ValueError, match="critic_agent failed"):
        await critic_agent(_request(), _state_view(), DeltaState(), provider, max_retries=1)
