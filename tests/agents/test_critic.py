"""Tests for critic_agent."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from forge.adapters.registry import AdapterRegistry
from forge.agents.base import render_files
from forge.agents.critic import critic_agent
from forge.core.models import (
    AgentRequest,
    AgentType,
    CriticDisposition,
    CriticFinding,
    DeltaState,
    FileWrite,
    PlanSpec,
    RequestSource,
    ReviewContext,
    StateView,
    WorkSpec,
)


def _registry() -> AdapterRegistry:
    adapters_dir = Path(__file__).parents[2] / "adapters"
    registry = AdapterRegistry()
    registry.load(adapters_dir)
    return registry


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


def _plan_request() -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(northstar="build a scraper"),
    )


def _state_view() -> StateView:
    return StateView(artifact_name="codebase", language="python", files=[], dependencies=[])


def _rendered_output() -> str:
    delta = DeltaState(new_files=[FileWrite(path="main.py", content='print("Hello, World!")')])
    return render_files(delta, _state_view())


def _rendered_empty() -> str:
    return render_files(DeltaState(), _state_view())


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
        _request(), _state_view(), _rendered_output(), _provider(finding_json), _registry()
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
        _request(), _state_view(), _rendered_output(), _provider(finding_json), _registry()
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
    result = await critic_agent(
        _request(), _state_view(), _rendered_empty(), _provider(finding_json), _registry()
    )
    assert result.disposition == CriticDisposition.REJECT


async def test_critic_prompt_uses_plan_review_context() -> None:
    """Planner validation is framed as plan review, not generic work review."""
    finding_json = json.dumps({"disposition": "accept", "rationale": "Good plan.", "hints": []})
    provider = _provider(finding_json)

    await critic_agent(
        _plan_request(),
        _state_view(),
        "Task 0: fetch pages",
        provider,
        _registry(),
        review_context=ReviewContext(
            output_noun="plan",
            review_focus="whether the task decomposition covers the northstar goal",
            empty_output_guidance="If the plan contains no tasks, reject it.",
        ),
    )

    messages = provider.chat.call_args.args[0]
    prompt = messages[1]["content"]
    assert "assess the plan below" in prompt
    assert "task decomposition covers the northstar goal" in prompt
    assert "If the plan contains no tasks, reject it." in prompt
    assert "work below" not in prompt
    assert "If no files were produced" not in prompt


async def test_critic_agent_retries_on_invalid_json() -> None:
    """critic_agent retries when the provider returns invalid JSON, then succeeds."""
    good_json = json.dumps({"disposition": "accept", "rationale": "Looks good.", "hints": []})
    provider = MagicMock()
    provider.chat = AsyncMock(side_effect=["not valid json", good_json])
    result = await critic_agent(
        _request(), _state_view(), _rendered_output(), provider, _registry()
    )
    assert isinstance(result, CriticFinding)
    assert provider.chat.call_count == 2


async def test_critic_agent_raises_after_max_retries_exceeded() -> None:
    """critic_agent raises ValueError after exhausting retries on persistent bad JSON."""
    provider = MagicMock()
    provider.chat = AsyncMock(return_value="invalid json always")
    with pytest.raises(ValueError, match="critic_agent failed"):
        await critic_agent(
            _request(), _state_view(), _rendered_empty(), provider, _registry(), max_retries=1
        )
