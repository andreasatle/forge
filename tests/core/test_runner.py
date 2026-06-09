"""Tests for Runner routing, built-in handlers, and scripted_plan_handler."""

# pyright: reportPrivateUsage=false

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from forge.adapters.registry import AdapterRegistry, AdapterSpec
from forge.core.models import (
    AgentRequest,
    AgentResponse,
    AgentType,
    DeltaState,
    FileWrite,
    IntegrateSpec,
    PlanSpec,
    Priority,
    RequestSource,
    ResponseStatus,
    SchedulerState,
    WorkSpec,
)
from forge.core.runner import (
    Runner,
    make_integrate_handler,
    make_plan_handler,
    make_work_handler,
    scripted_plan_handler,
    stub_integrate_handler,
    stub_plan_handler,
)
from forge.core.scheduler import Scheduler
from forge.core.workspace import Workspace
from forge.languages.registry import LanguageRegistry

# --- Helpers ---


def _plan_request() -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(northstar="test northstar"),
        priority=Priority.HIGH,
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


def _integrate_request() -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.INTEGRATE,
        source=RequestSource.WORKER,
        spec=IntegrateSpec(objective="integrate work", artifact="codebase", work_request_id=uuid4()),
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


async def test_runner_routes_integrate_to_integrate_handler() -> None:
    """Runner invokes the registered INTEGRATE handler for an integrate request."""
    received: list[AgentRequest] = []

    async def handler(request: AgentRequest) -> AgentResponse:
        received.append(request)
        return AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED)

    runner = Runner()
    runner.register(AgentType.INTEGRATE, handler)
    request = _integrate_request()
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
    provider.chat = AsyncMock(return_value='{"new_files": [{"path": "src/main.py", "content": "x = 1"}], "edits": [], "dependencies": []}')
    handler = make_work_handler(_mock_registry(), _make_workspace(tmp_path), LanguageRegistry(), provider)
    response = await handler(_work_request())

    assert response.status == ResponseStatus.COMPLETED


async def test_stub_integrate_handler_returns_completed() -> None:
    """stub_integrate_handler returns a COMPLETED response."""
    response = await stub_integrate_handler(_integrate_request())

    assert response.status == ResponseStatus.COMPLETED


async def test_runner_satisfies_agent_runner_type(tmp_path: Path) -> None:
    """A fully registered Runner can be used as an AgentRunner in the Scheduler."""
    runner = Runner()
    runner.register(AgentType.PLAN, stub_plan_handler)
    runner.register(AgentType.WORK, make_work_handler(_mock_registry(), _make_workspace(tmp_path), LanguageRegistry(), _mock_provider()))
    runner.register(AgentType.INTEGRATE, stub_integrate_handler)

    state = SchedulerState(northstar="test northstar")
    final = await Scheduler(runner=runner).run(state, _plan_request())

    assert final is not None


async def test_make_plan_handler_planner_source_returns_completed() -> None:
    """make_plan_handler returns empty follow-up for PLANNER-source requests without calling the LLM."""
    handler = make_plan_handler(_mock_registry(), artifact_names=["codebase"], artifact_languages={}, provider=_mock_provider())
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
async def test_scripted_plan_handler_end_to_end_produces_five_completed_nodes(tmp_path: Path) -> None:
    """End-to-end run with scripted_plan_handler produces exactly five COMPLETED nodes."""
    runner = Runner()
    runner.register(AgentType.PLAN, scripted_plan_handler)
    runner.register(AgentType.WORK, make_work_handler(_mock_registry(), _make_workspace(tmp_path), LanguageRegistry(), _mock_provider()))

    state = SchedulerState(northstar="test northstar")
    final = await Scheduler(runner=runner).run(state, _plan_request())

    completed = [n for n in final.dag.values() if n.node_state.value == "completed"]
    assert len(completed) == 5


async def test_scheduler_reinjects_global_planner_with_planner_source() -> None:
    """Scheduler re-injects the global planner with PLANNER source when the DAG goes idle."""
    captured_sources: list[RequestSource] = []

    async def capturing_plan_handler(request: AgentRequest) -> AgentResponse:
        captured_sources.append(request.source)
        return AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED, follow_up=[])

    runner = Runner()
    runner.register(AgentType.PLAN, capturing_plan_handler)

    state = SchedulerState(northstar="test northstar")
    await Scheduler(runner=runner).run(state, _plan_request())

    assert captured_sources[1] == RequestSource.PLANNER


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
    provider.chat = AsyncMock(return_value='{"new_files": [{"path": "src/main.py", "content": "x = 1"}], "edits": [], "dependencies": []}')
    provider.chat_with_tools = AsyncMock(side_effect=AssertionError("chat_with_tools must not be called"))

    handler = make_work_handler(_mock_registry(), _make_workspace(tmp_path), LanguageRegistry(), provider)
    response = await handler(_work_request())

    assert response.status == ResponseStatus.COMPLETED


async def test_make_plan_handler_never_calls_chat_with_tools() -> None:
    """make_plan_handler uses provider.chat only, never chat_with_tools."""
    provider = MagicMock()
    provider.max_tokens = 8192
    provider.chat = AsyncMock(return_value='{"kind": "plan", "tasks": []}')
    provider.chat_with_tools = AsyncMock(side_effect=AssertionError("chat_with_tools must not be called"))

    handler = make_plan_handler(_mock_registry(), artifact_names=["codebase"], artifact_languages={"codebase": "python"}, provider=provider)
    response = await handler(_plan_request())

    assert response.status == ResponseStatus.COMPLETED


async def test_make_integrate_handler_passes_only_requested_deltas(tmp_path: Path) -> None:
    """make_integrate_handler passes the delta for the work_request_id in the spec."""
    from unittest.mock import patch

    wid = uuid4()
    delta = DeltaState(new_files=[FileWrite(path="a.py", content="x = 1")])
    unrelated_wid = uuid4()
    completed = {wid: delta, unrelated_wid: DeltaState()}

    request = AgentRequest(
        agent_type=AgentType.INTEGRATE,
        source=RequestSource.WORKER,
        spec=IntegrateSpec(objective="merge", artifact="codebase", work_request_id=wid),
    )

    with patch("forge.core.runner.integrate_agent", new_callable=AsyncMock) as mock_integrate:
        mock_integrate.return_value = AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED)
        handler = make_integrate_handler(_make_workspace(tmp_path), LanguageRegistry(), _mock_provider(), completed)
        response = await handler(request)

    assert response.status == ResponseStatus.COMPLETED
    mock_integrate.assert_called_once()
    assert mock_integrate.call_args.kwargs["completed_deltas"] == [delta]
