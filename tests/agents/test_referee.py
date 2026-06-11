"""Tests for referee_agent."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from forge.adapters.registry import AdapterRegistry
from forge.agents.base import render_files
from forge.agents.referee import referee_agent
from forge.core.models import (
    AcceptanceCriterion,
    AgentContract,
    AgentRequest,
    AgentType,
    CriticDisposition,
    CriticFinding,
    DeltaState,
    FileWrite,
    PlanSpec,
    RefereeDecision,
    RequestSource,
    ReviewContext,
    StateView,
    WorkSpec,
    render_agent_contract,
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


def _rich_request() -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="write a hello world function",
            success_condition="function prints Hello, World!",
            contract=AgentContract(
                objective="write a hello world function",
                success_condition="function prints Hello, World!",
                acceptance_criteria=[AcceptanceCriterion(id="AC1", text="prints once")],
                constraints=["use Python"],
                non_goals=["CLI flags"],
            ),
            adapter="coding",
            artifact="codebase",
            language="python",
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


def _critic_accept() -> CriticFinding:
    return CriticFinding(disposition=CriticDisposition.ACCEPT, rationale="Looks good.", hints=[])


def _critic_reject() -> CriticFinding:
    return CriticFinding(
        disposition=CriticDisposition.REJECT,
        rationale="Does not meet requirements.",
        hints=["Add output"],
    )


def _provider(response_json: str) -> MagicMock:
    provider = MagicMock()
    provider.chat = AsyncMock(return_value=response_json)
    return provider


async def test_referee_agent_returns_referee_decision() -> None:
    """referee_agent returns a RefereeDecision."""
    decision_json = json.dumps(
        {"disposition": "accept", "rationale": "I agree with the critic.", "override": False}
    )
    result = await referee_agent(
        _request(),
        _state_view(),
        _rendered_output(),
        _critic_accept(),
        _provider(decision_json),
        _registry(),
    )
    assert isinstance(result, RefereeDecision)


async def test_referee_agrees_with_critic_sets_override_false() -> None:
    """referee_agent returns override=False when it agrees with the critic."""
    decision_json = json.dumps({"disposition": "accept", "rationale": "Agreed.", "override": False})
    result = await referee_agent(
        _request(),
        _state_view(),
        _rendered_output(),
        _critic_accept(),
        _provider(decision_json),
        _registry(),
    )
    assert result.override is False
    assert result.disposition == CriticDisposition.ACCEPT


async def test_referee_overrides_critic_sets_override_true() -> None:
    """referee_agent returns override=True when it overrides the critic's disposition."""
    decision_json = json.dumps(
        {"disposition": "accept", "rationale": "Actually meets the bar.", "override": True}
    )
    result = await referee_agent(
        _request(),
        _state_view(),
        _rendered_output(),
        _critic_reject(),
        _provider(decision_json),
        _registry(),
    )
    assert result.override is True
    assert result.disposition == CriticDisposition.ACCEPT


async def test_referee_prompt_uses_plan_review_context() -> None:
    """Planner referee validation is framed as plan review, not generic work review."""
    decision_json = json.dumps(
        {"disposition": "accept", "rationale": "I agree.", "override": False}
    )
    provider = _provider(decision_json)

    await referee_agent(
        _plan_request(),
        _state_view(),
        "Task 0: fetch pages",
        _critic_accept(),
        provider,
        _registry(),
        review_context=ReviewContext(
            output_noun="plan",
            review_focus="whether the task decomposition satisfies the planning contract",
            empty_output_guidance="If the plan contains no tasks, reject it.",
        ),
    )

    messages = provider.chat.call_args.args[0]
    prompt = messages[1]["content"]
    assert "reviewed the plan below" in prompt
    assert "Review focus: whether the task decomposition satisfies the planning contract" in prompt
    assert "Empty output rule: If the plan contains no tasks, reject it." in prompt
    assert render_agent_contract(_plan_request()) in prompt
    assert "work below" not in prompt
    assert "If no files were produced" not in prompt
    assert "fully covers the northstar goal" not in prompt


async def test_referee_prompt_includes_canonical_contract_and_scope_boundary() -> None:
    """Referee prompt uses the canonical contract block and overrides out-of-scope critiques."""
    decision_json = json.dumps(
        {"disposition": "accept", "rationale": "I agree.", "override": False}
    )
    provider = _provider(decision_json)
    request = _rich_request()

    await referee_agent(
        request,
        _state_view(),
        _rendered_output(),
        _critic_reject(),
        provider,
        _registry(),
    )

    messages = provider.chat.call_args.args[0]
    prompt = messages[1]["content"]
    contract_block = render_agent_contract(request)
    assert contract_block in prompt
    assert "- AC1: prints once" in prompt
    assert "- use Python" in prompt
    assert "- CLI flags" in prompt
    assert "Artifact: codebase" in prompt
    assert "Adapter: coding" in prompt
    assert "Language: python" in prompt
    assert "Out-of-scope ideals are not grounds for rejection" in prompt
    assert "outside the contract" in prompt


async def test_referee_agent_retries_on_invalid_json() -> None:
    """referee_agent retries when the provider returns invalid JSON, then succeeds."""
    good_json = json.dumps(
        {"disposition": "revise", "rationale": "Needs minor fix.", "override": False}
    )
    provider = MagicMock()
    provider.chat = AsyncMock(side_effect=["bad json", good_json])
    result = await referee_agent(
        _request(), _state_view(), _rendered_output(), _critic_accept(), provider, _registry()
    )
    assert isinstance(result, RefereeDecision)
    assert provider.chat.call_count == 2


async def test_referee_agent_raises_after_max_retries_exceeded() -> None:
    """referee_agent raises ValueError after exhausting retries on persistent bad JSON."""
    provider = MagicMock()
    provider.chat = AsyncMock(return_value="not json")
    with pytest.raises(ValueError, match="referee_agent failed"):
        await referee_agent(
            _request(),
            _state_view(),
            _rendered_empty(),
            _critic_accept(),
            provider,
            _registry(),
            max_retries=1,
        )
