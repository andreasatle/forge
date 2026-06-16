"""Tests for critic_agent."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from forge.adapters.registry import AdapterRegistry
from forge.agents.attempt import WorkOutputValidator
from forge.agents.critic import critic_agent
from forge.core.models import (
    AcceptanceCriterion,
    AgentContract,
    AgentRequest,
    AgentType,
    CriticDisposition,
    CriticFinding,
    PlanSpec,
    RequestSource,
    ReviewContext,
    StateView,
    WorkOutput,
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


def _plugin_guidance_request() -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="write a module",
            success_condition="tests pass",
            contract=AgentContract(
                objective="write a module",
                success_condition="tests pass",
                acceptance_criteria=[AcceptanceCriterion(id="AC1", text="module imports")],
                constraints=[
                    "Language plugin guidance:\nUse direct imports only.\nNever use forbidden imports."
                ],
            ),
            adapter="coding",
            artifact="codebase",
            language="toy",
        ),
    )


def _plan_request() -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(northstar="build a scraper"),
    )


def _state_view() -> StateView:
    return StateView(artifact_name="codebase", language="python", files=[])


def _rendered_output() -> str:
    work_output = WorkOutput(summary="Changed main.py in the worktree.")
    return WorkOutputValidator(_registry().get("coding"), _state_view()).render_for_critic(
        work_output
    )


def _rendered_empty() -> str:
    return WorkOutputValidator(_registry().get("coding"), _state_view()).render_for_critic(
        WorkOutput()
    )


def _provider(response_json: str) -> MagicMock:
    provider = MagicMock()
    wrapped = f'{{"kind":"final","output":{response_json}}}'
    provider.chat = AsyncMock(return_value=wrapped)
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


async def test_critic_revision_items_preserve_criterion_ids() -> None:
    """critic_agent preserves structured revision items with acceptance criterion ids."""
    finding_json = json.dumps(
        {
            "disposition": "revise",
            "rationale": "Missing required test.",
            "hints": ["Add the test"],
            "revision_items": [
                {
                    "criterion_id": "AC1",
                    "required_change": "Add a test for the greeting output.",
                    "rationale": "AC1 requires the output behavior to be verified.",
                }
            ],
        }
    )
    result = await critic_agent(
        _rich_request(), _state_view(), _rendered_output(), _provider(finding_json), _registry()
    )
    assert result.disposition == CriticDisposition.REVISE
    assert len(result.revision_items) == 1
    assert result.revision_items[0].criterion_id == "AC1"
    assert result.revision_items[0].required_change == "Add a test for the greeting output."


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
            review_focus="whether the task decomposition satisfies the planning contract",
            empty_output_guidance="If the plan contains no tasks, reject it.",
        ),
    )

    messages = provider.chat.call_args.args[0]
    prompt = messages[1]["content"]
    assert "assess the plan below" in prompt
    assert "task decomposition satisfies the planning contract" in prompt
    assert "If the plan contains no tasks, reject it." in prompt
    assert render_agent_contract(_plan_request()) in prompt
    assert "work below" not in prompt
    assert "If no files were produced" not in prompt
    assert "fully covers the northstar goal" not in prompt


async def test_critic_prompt_includes_canonical_contract_and_scope_boundary() -> None:
    """Critic prompt uses the canonical contract block and forbids out-of-scope rejection."""
    finding_json = json.dumps({"disposition": "accept", "rationale": "Good.", "hints": []})
    provider = _provider(finding_json)
    request = _rich_request()

    await critic_agent(request, _state_view(), _rendered_output(), provider, _registry())

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
    assert "Do not revise or reject for unstated ideals" in prompt
    assert "improvements outside the contract" in prompt


async def test_critic_prompt_treats_plugin_guidance_as_binding_contract() -> None:
    """Critic prompt includes plugin guidance and forbids revision items that contradict it."""
    finding_json = json.dumps({"disposition": "accept", "rationale": "Good.", "hints": []})
    provider = _provider(finding_json)
    request = _plugin_guidance_request()

    await critic_agent(request, _state_view(), _rendered_output(), provider, _registry())

    messages = provider.chat.call_args.args[0]
    prompt = messages[1]["content"]
    assert "Language plugin guidance:" in prompt
    assert "Never use forbidden imports." in prompt
    assert "revision_items must stay within the contract" in prompt
    assert "Never request a change that contradicts any constraint" in prompt


async def test_critic_prompt_includes_decomposition_topology_rules_for_planner_output() -> None:
    """Critic prompt includes decomposition topology rules when reviewing planner output."""
    from forge.agents.attempt import PlannerOutputValidator

    finding_json = json.dumps({"disposition": "accept", "rationale": "Good plan.", "hints": []})
    provider = _provider(finding_json)
    review_ctx = PlannerOutputValidator().review_context()

    await critic_agent(
        _plan_request(),
        _state_view(),
        "Decision: split_graph\nNode a: setup\nNode b (depends_on: a): implement scraper\nNode c (depends_on: b): write docs",
        provider,
        _registry(),
        review_context=review_ctx,
    )

    messages = provider.chat.call_args.args[0]
    prompt = messages[1]["content"]
    assert "real artifact or information flow" in prompt
    assert "genuine ordering constraint" in prompt
    assert "split_graph" in prompt
    assert "Maximize safe concurrency" not in prompt
    assert "not a goal" not in prompt


async def test_critic_prompt_topology_rules_cover_all_required_criteria() -> None:
    """Critic topology rules address all required decomposition review criteria."""
    from forge.agents.attempt import PlannerOutputValidator

    finding_json = json.dumps({"disposition": "accept", "rationale": "Good.", "hints": []})
    provider = _provider(finding_json)
    review_ctx = PlannerOutputValidator().review_context()

    await critic_agent(
        _plan_request(),
        _state_view(),
        "Decision: split_graph\nNode a: setup\nNode b (depends_on: a): implement",
        provider,
        _registry(),
        review_context=review_ctx,
    )

    messages = provider.chat.call_args.args[0]
    prompt = messages[1]["content"]
    # Structural validation criteria only — no optimization preferences
    assert "independently" in prompt
    assert "ordering" in prompt
    assert "convention" in prompt or "symmetry" in prompt
    assert "information flow" in prompt


async def test_critic_prompt_excludes_topology_rules_for_work_output() -> None:
    """Critic prompt does not include decomposition topology rules for work output."""
    finding_json = json.dumps({"disposition": "accept", "rationale": "Good.", "hints": []})
    provider = _provider(finding_json)

    await critic_agent(
        _request(),
        _state_view(),
        _rendered_output(),
        provider,
        _registry(),
    )

    messages = provider.chat.call_args.args[0]
    prompt = messages[1]["content"]
    assert "Decomposition topology rules" not in prompt


async def test_critic_prompt_includes_split_graph_topology_rules() -> None:
    """Critic topology rules include split_graph edge minimality guidance."""
    from forge.agents.attempt import PlannerOutputValidator

    finding_json = json.dumps({"disposition": "accept", "rationale": "Good plan.", "hints": []})
    provider = _provider(finding_json)
    review_ctx = PlannerOutputValidator().review_context()

    await critic_agent(
        _plan_request(),
        _state_view(),
        "Decision: split_graph\nNode setup: setup env\nNode scraper (depends_on: setup): implement scraper",
        provider,
        _registry(),
        review_context=review_ctx,
    )

    messages = provider.chat.call_args.args[0]
    prompt = messages[1]["content"]
    assert "split_graph" in prompt
    assert "depends_on" in prompt
    assert "information flow" in prompt


async def test_review_context_topology_excludes_optimization_guidance() -> None:
    """ReviewContext.topology_rules must not inject optimization preferences into critic/referee."""
    from forge.agents.attempt import PlannerOutputValidator

    context = PlannerOutputValidator().review_context()
    rules = context.topology_rules
    assert "Maximize safe concurrency" not in rules
    assert "When in doubt" not in rules
    assert "not a goal" not in rules
    assert "more parallel work" not in rules


async def test_review_context_topology_includes_structural_validation_rules() -> None:
    """ReviewContext.topology_rules contains structural validation authority for planner output."""
    from forge.agents.attempt import PlannerOutputValidator

    context = PlannerOutputValidator().review_context()
    rules = context.topology_rules
    assert "genuine ordering constraint" in rules
    assert "information flow" in rules
    assert "Convention, symmetry" in rules


async def test_critic_agent_retries_on_invalid_json() -> None:
    """critic_agent retries when the provider returns invalid JSON, then succeeds."""
    good_json = json.dumps(
        {
            "kind": "final",
            "output": {"disposition": "accept", "rationale": "Looks good.", "hints": []},
        }
    )
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
