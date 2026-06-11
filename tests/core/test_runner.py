"""Tests for Runner routing and test-owned helper handlers."""

# pyright: reportPrivateUsage=false

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from forge.adapters.registry import AdapterRegistry, AdapterSpec
from forge.core.models import (
    AgentRequest,
    AgentResponse,
    AgentType,
    DAGNode,
    FailureKind,
    NodeState,
    PlanSpec,
    RequestSource,
    ResponseStatus,
    RunResult,
    SchedulerState,
    WorkSpec,
)
from forge.core.runner import (
    Runner,
    make_plan_handler,
    make_work_handler,
)
from forge.core.scheduler import Scheduler
from forge.core.state_service import StateService
from forge.core.workspace import Workspace
from forge.languages.registry import LanguageRegistry

# --- Helpers ---


def _plan_request() -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(northstar="test northstar"),
    )


def _work_request() -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="do work",
            success_condition="work done",
            adapter="coding",
            artifact="codebase",
        ),
    )


def _user_sourced_work_request() -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.USER,
        spec=WorkSpec(
            objective="do work", success_condition="done", adapter="coding", artifact="codebase"
        ),
    )


def _mock_registry() -> AdapterRegistry:
    registry = AdapterRegistry()
    registry._adapters["coding"] = AdapterSpec(
        name="coding",
        description="test",
        tools=[],
        prompt_template="do: {objective}",
    )
    return registry


def _make_workspace(tmp_path: Path) -> Workspace:
    ws = Workspace(tmp_path / "ws")
    ws.init()
    ws.init_artifact("codebase")
    return ws


def _mock_provider() -> MagicMock:
    provider = MagicMock()
    provider.max_tokens = 8192
    provider.chat = AsyncMock(return_value="{}")
    return provider


def _mock_ss() -> MagicMock:
    ss = MagicMock(spec=StateService)
    ss.run_tests.return_value = RunResult(passed=True)
    ss.current_version = 0
    return ss


async def stub_plan_handler(request: AgentRequest) -> AgentResponse:
    """Return a completed response with a placeholder plan delta, for testing."""
    return AgentResponse(
        request_id=request.id,
        status=ResponseStatus.COMPLETED,
        delta=None,
    )


async def scripted_plan_handler(request: AgentRequest) -> AgentResponse:
    """Return a hardcoded A→B→C dependency chain for use in integration tests."""
    if request.source == RequestSource.PLANNER:
        return AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, follow_up=[])

    a = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="task A",
            success_condition="A done",
            adapter="coding",
            artifact="codebase",
        ),
    )
    b = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="task B",
            success_condition="B done",
            adapter="coding",
            artifact="codebase",
        ),
        dependencies=frozenset({a.id}),
    )
    c = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="task C",
            success_condition="C done",
            adapter="coding",
            artifact="codebase",
        ),
        dependencies=frozenset({b.id}),
    )
    return AgentResponse(
        request_id=request.id,
        status=ResponseStatus.COMPLETED,
        follow_up=[c, b, a],
    )


# --- Tests ---


async def test_runner_routes_plan_to_plan_handler() -> None:
    """Runner invokes the registered PLAN handler for a plan request."""
    received: list[AgentRequest] = []

    async def handler(request: AgentRequest) -> AgentResponse:
        received.append(request)
        return AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED)

    runner = Runner()
    runner.register(AgentType.PLAN, handler)
    request = _plan_request()
    await runner(request)

    assert len(received) == 1
    assert received[0] is request


async def test_runner_routes_work_to_work_handler() -> None:
    """Runner invokes the registered WORK handler for a work request."""
    received: list[AgentRequest] = []

    async def handler(request: AgentRequest) -> AgentResponse:
        received.append(request)
        return AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED)

    runner = Runner()
    runner.register(AgentType.WORK, handler)
    request = _work_request()
    await runner(request)

    assert len(received) == 1
    assert received[0] is request


async def test_runner_routes_user_sourced_work_to_work_handler() -> None:
    """Runner routes a USER-sourced WORK request to the registered WORK handler."""
    received: list[AgentRequest] = []

    async def handler(request: AgentRequest) -> AgentResponse:
        received.append(request)
        return AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED)

    runner = Runner()
    runner.register(AgentType.WORK, handler)
    request = _user_sourced_work_request()
    await runner(request)

    assert len(received) == 1
    assert received[0] is request


async def test_handler_receives_original_request_unmodified() -> None:
    """The handler receives the exact same request object passed to the runner."""
    received: list[AgentRequest] = []

    async def handler(request: AgentRequest) -> AgentResponse:
        received.append(request)
        return AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED)

    runner = Runner()
    runner.register(AgentType.WORK, handler)
    original = _work_request()
    await runner(original)

    assert received[0] is original


async def test_unregistered_agent_type_returns_failed_response() -> None:
    """Runner returns a FAILED response when no handler is registered for the agent type."""
    runner = Runner()
    request = _plan_request()
    response = await runner(request)

    assert response.status == ResponseStatus.FAILED
    assert response.error == f"no handler registered for: {AgentType.PLAN.value}"


async def test_registering_second_handler_overwrites_first() -> None:
    """Registering a second handler for the same agent type replaces the first."""
    calls: list[str] = []

    async def first(request: AgentRequest) -> AgentResponse:
        calls.append("first")
        return AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED)

    async def second(request: AgentRequest) -> AgentResponse:
        calls.append("second")
        return AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED)

    runner = Runner()
    runner.register(AgentType.PLAN, first)
    runner.register(AgentType.PLAN, second)
    await runner(_plan_request())

    assert calls == ["second"]


async def test_stub_plan_handler_returns_completed() -> None:
    """stub_plan_handler returns a COMPLETED response."""
    response = await stub_plan_handler(_plan_request())

    assert response.status == ResponseStatus.COMPLETED


async def test_work_handler_returns_completed(tmp_path: Path) -> None:
    """make_work_handler returns a COMPLETED response on success."""
    provider = _mock_provider()
    provider.chat = AsyncMock(
        return_value='{"new_files": [{"path": "src/main.py", "content": "x = 1"}], "edits": [], "dependencies": []}'
    )
    handler = make_work_handler(
        _mock_registry(), _make_workspace(tmp_path), LanguageRegistry(), provider
    )
    response = await handler(_work_request())

    assert response.status == ResponseStatus.COMPLETED


async def test_runner_satisfies_agent_runner_type(tmp_path: Path) -> None:
    """A fully registered Runner can be used as an AgentRunner in the Scheduler."""
    runner = Runner()
    runner.register(AgentType.PLAN, stub_plan_handler)
    runner.register(
        AgentType.WORK,
        make_work_handler(
            _mock_registry(), _make_workspace(tmp_path), LanguageRegistry(), _mock_provider()
        ),
    )

    state = SchedulerState(northstar="test northstar")
    final = await Scheduler(runner=runner).run(state, _plan_request())

    assert final is not None


async def test_make_plan_handler_planner_source_returns_completed() -> None:
    """make_plan_handler returns empty follow-up for PLANNER-source requests without calling the LLM."""
    handler = make_plan_handler(
        _mock_registry(),
        artifact_names=["codebase"],
        artifact_languages={},
        provider=_mock_provider(),
    )
    request = AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.PLANNER,
        spec=PlanSpec(northstar="test northstar"),
    )
    response = await handler(request)

    assert response.status == ResponseStatus.COMPLETED
    assert response.follow_up == []


async def test_scripted_plan_handler_user_source_emits_three_follow_ups() -> None:
    """scripted_plan_handler returns exactly three follow-up requests for a USER source."""
    response = await scripted_plan_handler(_plan_request())

    assert len(response.follow_up) == 3


async def test_scripted_plan_handler_follow_ups_form_valid_chain() -> None:
    """scripted_plan_handler follow-ups form a linear A→B→C dependency chain."""
    response = await scripted_plan_handler(_plan_request())

    by_id = {r.id: r for r in response.follow_up}
    work_nodes = [r for r in response.follow_up if r.agent_type == AgentType.WORK]
    no_deps = [r for r in work_nodes if not r.dependencies]
    one_dep = [r for r in work_nodes if len(r.dependencies) == 1]
    two_deps_or_more = [r for r in work_nodes if len(r.dependencies) > 1]

    assert len(no_deps) == 1, "exactly one root node (A)"
    assert len(one_dep) == 2, "B depends on A, C depends on B"
    assert len(two_deps_or_more) == 0

    a = no_deps[0]
    b = next(r for r in one_dep if a.id in r.dependencies)
    c = next(r for r in one_dep if b.id in r.dependencies)

    assert b.id in by_id
    assert c.id in by_id


async def test_scripted_plan_handler_planner_source_emits_empty_follow_up() -> None:
    """scripted_plan_handler returns an empty follow-up list for a PLANNER source request."""
    planner_request = AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.PLANNER,
        spec=PlanSpec(northstar="test northstar"),
    )
    response = await scripted_plan_handler(planner_request)

    assert response.follow_up == []


@pytest.mark.slow
async def test_scripted_plan_handler_end_to_end_produces_four_completed_nodes(
    tmp_path: Path,
) -> None:
    """End-to-end run with scripted_plan_handler produces exactly four INTEGRATED nodes."""
    provider = _mock_provider()
    provider.chat = AsyncMock(
        return_value='{"new_files": [{"path": "src/out.py", "content": "x = 1"}], "edits": [], "dependencies": []}'
    )
    runner = Runner()
    runner.register(AgentType.PLAN, scripted_plan_handler)
    runner.register(
        AgentType.WORK,
        make_work_handler(
            _mock_registry(), _make_workspace(tmp_path), LanguageRegistry(), provider
        ),
    )

    state = SchedulerState(northstar="test northstar")
    final = await Scheduler(runner=runner).run(state, _plan_request())

    completed = [n for n in final.dag.values() if n.node_state.value == "integrated"]
    assert len(completed) == 4


async def test_scheduler_dispatches_global_planner_with_user_source() -> None:
    """Scheduler dispatches the global planner with USER source on the initial run."""
    captured_sources: list[RequestSource] = []

    async def capturing_plan_handler(request: AgentRequest) -> AgentResponse:
        captured_sources.append(request.source)
        return AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, follow_up=[])

    runner = Runner()
    runner.register(AgentType.PLAN, capturing_plan_handler)

    state = SchedulerState(northstar="test northstar")
    await Scheduler(runner=runner).run(state, _plan_request())

    assert RequestSource.USER in captured_sources


async def test_scripted_plan_handler_work_nodes_execute_in_dependency_order() -> None:
    """Work nodes produced by scripted_plan_handler complete in A-then-B-then-C order."""
    completion_order: list[str] = []

    async def tracking_work_handler(request: AgentRequest) -> AgentResponse:
        spec = request.spec
        if isinstance(spec, WorkSpec):
            completion_order.append(spec.objective)
        return AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED)

    runner = Runner()
    runner.register(AgentType.PLAN, scripted_plan_handler)
    runner.register(AgentType.WORK, tracking_work_handler)

    state = SchedulerState(northstar="test northstar")
    await Scheduler(runner=runner).run(state, _plan_request())

    assert completion_order.index("task A") < completion_order.index("task B")
    assert completion_order.index("task B") < completion_order.index("task C")


async def test_make_work_handler_never_calls_chat_with_tools(tmp_path: Path) -> None:
    """make_work_handler uses provider.chat only, never chat_with_tools."""
    provider = MagicMock()
    provider.max_tokens = 8192
    provider.chat = AsyncMock(
        return_value='{"new_files": [{"path": "src/main.py", "content": "x = 1"}], "edits": [], "dependencies": []}'
    )
    provider.chat_with_tools = AsyncMock(
        side_effect=AssertionError("chat_with_tools must not be called")
    )

    handler = make_work_handler(
        _mock_registry(), _make_workspace(tmp_path), LanguageRegistry(), provider
    )
    response = await handler(_work_request())

    assert response.status == ResponseStatus.COMPLETED


async def test_make_plan_handler_never_calls_chat_with_tools() -> None:
    """make_plan_handler uses provider.chat only, never chat_with_tools."""
    provider = MagicMock()
    provider.max_tokens = 8192
    provider.chat = AsyncMock(return_value='{"kind": "plan", "tasks": []}')
    provider.chat_with_tools = AsyncMock(
        side_effect=AssertionError("chat_with_tools must not be called")
    )

    handler = make_plan_handler(
        _mock_registry(),
        artifact_names=["codebase"],
        artifact_languages={"codebase": "python"},
        provider=provider,
    )
    response = await handler(_plan_request())

    assert response.status == ResponseStatus.COMPLETED


async def test_make_plan_handler_passes_critic_and_referee_providers() -> None:
    """make_plan_handler forwards planner critic and referee providers to plan_agent."""
    from unittest.mock import patch

    critic = _mock_provider()
    referee = _mock_provider()
    captured: dict[str, object] = {}

    async def capturing_plan_agent(*args: object, **kwargs: object) -> AgentResponse:
        captured["critic_provider"] = kwargs.get("critic_provider")
        captured["referee_provider"] = kwargs.get("referee_provider")
        captured["registry"] = kwargs.get("registry")
        captured["artifact_types"] = kwargs.get("artifact_types")
        captured["artifact_descriptions"] = kwargs.get("artifact_descriptions")
        captured["artifact_language_guidance"] = kwargs.get("artifact_language_guidance")
        return AgentResponse(request_id=args[0].id, status=ResponseStatus.COMPLETED)  # type: ignore[union-attr]

    registry = _mock_registry()
    with patch("forge.core.runner.plan_agent", capturing_plan_agent):
        handler = make_plan_handler(
            registry,
            artifact_names=["codebase"],
            artifact_languages={"codebase": "python"},
            provider=_mock_provider(),
            critic_provider=critic,
            referee_provider=referee,
            artifact_types={"codebase": "coding"},
            artifact_descriptions={"codebase": "Python implementation."},
            artifact_language_guidance={"codebase": "PLUGIN GUIDANCE"},
        )
        await handler(_plan_request())

    assert captured["critic_provider"] is critic
    assert captured["referee_provider"] is referee
    assert captured["registry"] is registry
    assert captured["artifact_types"] == {"codebase": "coding"}
    assert captured["artifact_descriptions"] == {"codebase": "Python implementation."}
    assert captured["artifact_language_guidance"] == {"codebase": "PLUGIN GUIDANCE"}


async def test_make_work_handler_passes_critic_and_referee_providers(tmp_path: Path) -> None:
    """make_work_handler forwards critic and referee providers to work_agent."""
    from unittest.mock import patch

    critic = _mock_provider()
    referee = _mock_provider()
    captured: dict[str, object] = {}

    async def capturing_work_agent(*args: object, **kwargs: object) -> AgentResponse:
        captured["critic_provider"] = kwargs.get("critic_provider")
        captured["referee_provider"] = kwargs.get("referee_provider")
        return AgentResponse(request_id=args[0].id, status=ResponseStatus.COMPLETED)  # type: ignore[union-attr]

    with patch("forge.core.runner.work_agent", capturing_work_agent):
        handler = make_work_handler(
            _mock_registry(),
            _make_workspace(tmp_path),
            LanguageRegistry(),
            _mock_provider(),
            critic_provider=critic,
            referee_provider=referee,
        )
        await handler(_work_request())

    assert captured["critic_provider"] is critic
    assert captured["referee_provider"] is referee


async def test_make_work_handler_forwards_max_retries(tmp_path: Path) -> None:
    """make_work_handler forwards max_retries to work_agent."""
    from unittest.mock import patch

    captured: dict[str, object] = {}

    async def capturing_work_agent(*args: object, **kwargs: object) -> AgentResponse:
        captured["max_retries"] = kwargs.get("max_retries")
        return AgentResponse(request_id=args[0].id, status=ResponseStatus.COMPLETED)  # type: ignore[union-attr]

    with patch("forge.core.runner.work_agent", capturing_work_agent):
        handler = make_work_handler(
            _mock_registry(),
            _make_workspace(tmp_path),
            LanguageRegistry(),
            _mock_provider(),
            max_retries=7,
        )
        await handler(_work_request())

    assert captured["max_retries"] == 7


async def test_make_work_handler_passes_none_providers_when_omitted(tmp_path: Path) -> None:
    """make_work_handler passes None for critic and referee when not provided."""
    from unittest.mock import patch

    captured: dict[str, object] = {}

    async def capturing_work_agent(*args: object, **kwargs: object) -> AgentResponse:
        captured["critic_provider"] = kwargs.get("critic_provider")
        captured["referee_provider"] = kwargs.get("referee_provider")
        return AgentResponse(request_id=args[0].id, status=ResponseStatus.COMPLETED)  # type: ignore[union-attr]

    with patch("forge.core.runner.work_agent", capturing_work_agent):
        handler = make_work_handler(
            _mock_registry(),
            _make_workspace(tmp_path),
            LanguageRegistry(),
            _mock_provider(),
        )
        await handler(_work_request())

    assert captured["critic_provider"] is None
    assert captured["referee_provider"] is None


async def test_scheduler_uses_provided_state_service_for_integration(tmp_path: Path) -> None:
    """Scheduler calls state_service.apply_delta when a work node completes successfully."""
    provider = _mock_provider()
    provider.chat = AsyncMock(
        return_value='{"new_files": [{"path": "src/main.py", "content": "x = 1"}], "edits": [], "dependencies": []}'
    )

    ss = _mock_ss()
    work = _work_request()
    state = SchedulerState(northstar="test northstar").add_nodes([DAGNode(request=work)])

    runner = Runner()
    runner.register(
        AgentType.WORK,
        make_work_handler(
            _mock_registry(), _make_workspace(tmp_path), LanguageRegistry(), provider
        ),
    )

    await Scheduler(runner=runner, state_services={"codebase": ss}).run(state, _plan_request())

    ss.apply_delta.assert_called_once()


async def test_validation_failed_work_response_does_not_call_apply_delta(tmp_path: Path) -> None:
    """StateService.apply_delta is never called when work_agent returns a validation-rejected FAILED response."""
    ss = _mock_ss()
    work = _work_request()
    state = SchedulerState(northstar="test northstar").add_nodes([DAGNode(request=work)])

    runner = Runner()

    async def validation_failed_handler(request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            failure_kind=FailureKind.VALIDATION_REJECTED,
            error="validation rejected on attempt 1: output does not meet success condition",
        )

    runner.register(AgentType.WORK, validation_failed_handler)

    final = await Scheduler(runner=runner, state_services={"codebase": ss}).run(
        state, _plan_request()
    )

    ss.apply_delta.assert_not_called()
    assert final.dag[work.id].node_state == NodeState.FAILED
