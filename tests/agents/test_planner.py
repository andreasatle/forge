"""Tests for the planning agent."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from forge.agents.planner import PlannerTaskExecutor, plan_agent
from forge.core.models import (
    AcceptanceCriterion,
    AgentContract,
    AgentRequest,
    AgentType,
    PlanResponse,
    PlanSpec,
    RequestSource,
    ResponseStatus,
    render_agent_contract,
)


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


async def test_planner_task_executor_returns_typed_plan_output() -> None:
    """PlannerTaskExecutor preserves PlanResponse as typed producer output."""
    request = _make_request()
    provider = _mock_provider(
        '{"kind": "plan", "tasks": ['
        '{"objective": "Fetch pages", "success_condition": "tests pass", '
        '"adapter": "coding", "artifact": "codebase", "language": "python"}'
        "]}"
    )
    executor = PlannerTaskExecutor(
        provider=provider,
        artifact_names=["codebase"],
        artifact_languages={"codebase": "python"},
    )

    response = await executor.run(request)

    assert response.status == ResponseStatus.COMPLETED
    assert isinstance(response.output, PlanResponse)
    assert len(response.output.tasks) == 1
    task = response.output.tasks[0]
    assert task.objective == "Fetch pages"
    assert task.language == "python"


async def test_planner_task_executor_preserves_artifact_language_context() -> None:
    """PlannerTaskExecutor renders artifact/language context into the planner prompt."""
    request = _make_request()
    provider = _mock_provider()
    executor = PlannerTaskExecutor(
        provider=provider,
        artifact_names=["api", "docs"],
        artifact_languages={"api": "python", "docs": "markdown"},
        artifact_types={"api": "coding", "docs": "document"},
        artifact_descriptions={
            "api": "Backend API implementation.",
            "docs": "User-facing documentation.",
        },
    )

    response = await executor.run(request)

    assert response.status == ResponseStatus.COMPLETED
    messages = provider.chat.call_args.args[0]
    user_prompt = messages[1]["content"]
    assert "artifact must be one of: api, docs" in user_prompt
    assert "Available artifacts:" in user_prompt
    assert "  api:\n    type: coding\n    language: python" in user_prompt
    assert "    description: Backend API implementation." in user_prompt
    assert "  docs:\n    type: document\n    language: markdown" in user_prompt
    assert "    description: User-facing documentation." in user_prompt


async def test_planner_task_executor_renders_plugin_owned_language_guidance() -> None:
    """Planner prompt receives language guidance as plugin-owned artifact metadata."""
    request = _make_request()
    provider = _mock_provider()
    executor = PlannerTaskExecutor(
        provider=provider,
        artifact_names=["api"],
        artifact_languages={"api": "toy"},
        artifact_types={"api": "coding"},
        artifact_language_guidance={"api": "Use toy.mod files only.\nNever emit legacy.toy."},
    )

    response = await executor.run(request)

    assert response.status == ResponseStatus.COMPLETED
    messages = provider.chat.call_args.args[0]
    user_prompt = messages[1]["content"]
    assert "    language guidance:" in user_prompt
    assert "      Use toy.mod files only." in user_prompt
    assert "      Never emit legacy.toy." in user_prompt
    assert "must not contradict artifact-specific language guidance" in user_prompt


async def test_plan_producer_prompt_includes_canonical_contract_block() -> None:
    """Planner producer prompt includes the canonical AgentRequest contract block."""
    request = AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(
            northstar="build a web scraper",
            contract=AgentContract(
                objective="build a web scraper",
                success_condition="planner emits bounded executable tasks",
                acceptance_criteria=[
                    AcceptanceCriterion(id="AC1", text="each task names an artifact")
                ],
                constraints=["at most 5 tasks"],
                non_goals=["browser extension"],
            ),
        ),
    )
    provider = _mock_provider()
    executor = PlannerTaskExecutor(
        provider=provider,
        artifact_names=["codebase"],
        artifact_languages={"codebase": "python"},
    )

    response = await executor.run(request)

    assert response.status == ResponseStatus.COMPLETED
    messages = provider.chat.call_args.args[0]
    user_prompt = messages[1]["content"]
    assert render_agent_contract(request) in user_prompt
    assert "Produce output satisfying this contract." in user_prompt


async def test_planner_prompt_handles_missing_artifact_description() -> None:
    """Planner prompt stays clean when an artifact has no description."""
    request = _make_request()
    provider = _mock_provider()
    executor = PlannerTaskExecutor(
        provider=provider,
        artifact_names=["codebase"],
        artifact_languages={"codebase": "python"},
        artifact_types={"codebase": "coding"},
    )

    response = await executor.run(request)

    assert response.status == ResponseStatus.COMPLETED
    messages = provider.chat.call_args.args[0]
    user_prompt = messages[1]["content"]
    assert "  codebase:\n    type: coding\n    language: python" in user_prompt
    assert "description:" not in user_prompt


async def test_planner_prompt_includes_artifact_description() -> None:
    """Planner prompt includes configured artifact descriptions."""
    request = _make_request()
    provider = _mock_provider()
    executor = PlannerTaskExecutor(
        provider=provider,
        artifact_names=["docs"],
        artifact_languages={},
        artifact_types={"docs": "document"},
        artifact_descriptions={
            "docs": "User-facing documentation for installing and running the scraper."
        },
    )

    response = await executor.run(request)

    assert response.status == ResponseStatus.COMPLETED
    messages = provider.chat.call_args.args[0]
    user_prompt = messages[1]["content"]
    assert "  docs:\n    type: document" in user_prompt
    assert (
        "    description: User-facing documentation for installing and running the scraper."
        in user_prompt
    )


async def test_planner_prompt_only_mentions_configured_artifacts() -> None:
    """Planner prompt derives allowed artifacts only from configured artifact_names."""
    request = _make_request()
    provider = _mock_provider()
    executor = PlannerTaskExecutor(
        provider=provider,
        artifact_names=["codebase"],
        artifact_languages={"codebase": "python"},
        artifact_types={"codebase": "coding"},
    )

    response = await executor.run(request)

    assert response.status == ResponseStatus.COMPLETED
    messages = provider.chat.call_args.args[0]
    user_prompt = messages[1]["content"]
    assert "artifact must be one of: codebase" in user_prompt
    assert "  codebase:\n    type: coding\n    language: python" in user_prompt
    assert "docs" not in user_prompt


async def test_planner_task_executor_format_failure_exhausts_retries() -> None:
    """PlannerTaskExecutor preserves plan_agent retry exhaustion behavior."""
    request = _make_request()
    provider = _mock_provider("not json")
    executor = PlannerTaskExecutor(
        provider=provider,
        artifact_names=["codebase"],
        artifact_languages={"codebase": "python"},
        max_retries=2,
    )

    response = await executor.run(request)

    assert response.status == ResponseStatus.FAILED
    assert provider.chat.call_count == 3  # initial + 2 retries


async def test_planner_format_failure_triggers_retry() -> None:
    """plan_agent retries when the LLM returns invalid JSON and succeeds on the next attempt."""
    request = _make_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            "not valid json",
            '{"kind": "plan", "tasks": []}',
        ]
    )

    response = await plan_agent(
        request, ["codebase"], {"codebase": "python"}, provider, max_retries=3
    )

    assert response.status == ResponseStatus.COMPLETED
    assert provider.chat.call_count == 2


async def test_planner_format_failure_exhausts_retries_returns_failed() -> None:
    """plan_agent returns FAILED when the LLM always returns unparseable JSON."""
    request = _make_request()
    provider = _mock_provider("not json")

    response = await plan_agent(
        request, ["codebase"], {"codebase": "python"}, provider, max_retries=2
    )

    assert response.status == ResponseStatus.FAILED
    assert provider.chat.call_count == 3  # initial + 2 retries


async def test_planner_llm_error_returns_failed_immediately() -> None:
    """plan_agent returns FAILED immediately when the LLM raises an exception."""
    request = _make_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(side_effect=RuntimeError("network error"))

    response = await plan_agent(
        request, ["codebase"], {"codebase": "python"}, provider, max_retries=3
    )

    assert response.status == ResponseStatus.FAILED
    assert provider.chat.call_count == 1


async def test_planner_logs_warning_on_retry(caplog: pytest.LogCaptureFixture) -> None:
    """plan_agent logs a debug retry message when the LLM returns invalid JSON."""
    import logging

    request = _make_request()
    provider = _mock_provider()
    provider.chat = AsyncMock(
        side_effect=[
            "not valid json",
            '{"kind": "plan", "tasks": []}',
        ]
    )

    with caplog.at_level(logging.DEBUG, logger="forge.agents.base"):
        await plan_agent(request, ["codebase"], {"codebase": "python"}, provider, max_retries=3)

    assert any("agent retry 1" in r.message for r in caplog.records)


async def test_plan_agent_never_calls_chat_with_tools() -> None:
    """plan_agent uses provider.chat only, never chat_with_tools."""
    request = _make_request()
    provider = _mock_provider()
    provider.chat_with_tools = AsyncMock(
        side_effect=AssertionError("chat_with_tools must not be called")
    )

    response = await plan_agent(request, ["codebase"], {"codebase": "python"}, provider)

    assert response.status == ResponseStatus.COMPLETED


async def test_planner_user_prompt_does_not_duplicate_final_schema() -> None:
    """Planner-specific prompt leaves final schema ownership to run_agent."""
    request = _make_request()
    provider = _mock_provider()

    response = await plan_agent(request, ["codebase"], {"codebase": "python"}, provider)

    assert response.status == ResponseStatus.COMPLETED
    messages = provider.chat.call_args.args[0]
    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]
    assert "Final response model: PlanResponse" in system_prompt
    assert '"tasks": [' not in user_prompt
    assert '"kind": "plan"' not in user_prompt
    assert "Respond with ONLY a JSON object" not in user_prompt


async def test_planner_output_contains_tasks_not_scheduler_nodes() -> None:
    """plan_agent returns PlanResponse tasks, not scheduler AgentRequests."""
    request = _make_request()
    provider = _mock_provider(
        '{"kind": "plan", "tasks": ['
        '{"objective": "A", "success_condition": "done", '
        '"adapter": "coding", "artifact": "codebase"}'
        "]}"
    )

    response = await plan_agent(request, ["codebase"], {"codebase": "python"}, provider)

    assert response.status == ResponseStatus.COMPLETED
    assert isinstance(response.output, PlanResponse)
    assert len(response.output.tasks) == 1
    assert response.output.tasks[0].objective == "A"


async def test_planner_preserves_task_dependency_indices() -> None:
    """PlanResponse keeps planner task dependency indices for scheduler conversion."""
    request = _make_request()
    provider = _mock_provider(
        '{"kind": "plan", "tasks": ['
        '{"objective": "A", "success_condition": "done", '
        '"adapter": "coding", "artifact": "codebase"},'
        '{"objective": "B", "success_condition": "done", '
        '"adapter": "coding", "artifact": "codebase", "depends_on": [0]}'
        "]}"
    )

    response = await plan_agent(request, ["codebase"], {"codebase": "python"}, provider)

    assert isinstance(response.output, PlanResponse)
    task_b = next(task for task in response.output.tasks if task.objective == "B")
    assert task_b.depends_on == [0]


async def test_planner_prompt_requires_testable_success_conditions() -> None:
    """PLAN_PROMPT requires coding task success conditions to be verifiable by running tests."""
    request = _make_request()
    provider = _mock_provider()

    await plan_agent(request, ["codebase"], {"codebase": "python"}, provider)

    messages = provider.chat.call_args.args[0]
    user_prompt = messages[1]["content"]
    assert "verifiable by running tests" in user_prompt


async def test_planner_prompt_includes_decomposition_context_when_contract_has_constraints() -> (
    None
):
    """Planner prompt includes decomposition context when contract has constraints or non_goals."""
    request = AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(
            northstar="implement a data pipeline module",
            contract=AgentContract(
                objective="implement a data pipeline module",
                success_condition="all tests pass",
                constraints=[
                    "Each subtask must have exactly one concern",
                    "Subtasks must be non-overlapping",
                ],
                non_goals=["Do not combine setup, implementation, and testing in a single task"],
            ),
        ),
    )
    provider = _mock_provider()
    executor = PlannerTaskExecutor(
        provider=provider,
        artifact_names=["codebase"],
        artifact_languages={"codebase": "python"},
    )

    response = await executor.run(request)

    assert response.status == ResponseStatus.COMPLETED
    messages = provider.chat.call_args.args[0]
    user_prompt = messages[1]["content"]
    assert "This task was too broad for a single implementation." in user_prompt
    assert "Decompose it into focused, non-overlapping subtasks" in user_prompt
    assert "each subtask has exactly one concern" in user_prompt


async def test_planner_prompt_omits_decomposition_context_without_constraints() -> None:
    """Planner prompt does not include decomposition context for plans without constraints."""
    request = _make_request()
    provider = _mock_provider()
    executor = PlannerTaskExecutor(
        provider=provider,
        artifact_names=["codebase"],
        artifact_languages={"codebase": "python"},
    )

    response = await executor.run(request)

    assert response.status == ResponseStatus.COMPLETED
    messages = provider.chat.call_args.args[0]
    user_prompt = messages[1]["content"]
    assert "This task was too broad for a single implementation." not in user_prompt


async def test_planner_prompt_omits_python_packaging_policy() -> None:
    """PLAN_PROMPT stays language-agnostic and does not include Python packaging policy."""
    request = _make_request()
    provider = _mock_provider()

    await plan_agent(request, ["codebase"], {"codebase": "python"}, provider)

    messages = provider.chat.call_args.args[0]
    user_prompt = messages[1]["content"]
    assert "requirements.txt" not in user_prompt
    assert "setup.py" not in user_prompt
    assert "legacy Python packaging" not in user_prompt
    assert "observable outcomes" in user_prompt
