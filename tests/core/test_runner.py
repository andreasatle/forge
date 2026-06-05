from pathlib import Path
from uuid import uuid4

from forge.adapters.registry import AdapterRegistry
from forge.core.models import (
    AgentRequest,
    AgentResponse,
    AgentType,
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
    make_work_handler,
    scripted_plan_handler,
    stub_integrate_handler,
    stub_plan_handler,
)
from forge.core.scheduler import Scheduler

_ADAPTERS_DIR = Path(__file__).parent.parent.parent / "adapters"


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
        ),
    )


def _integrate_request() -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.INTEGRATE,
        source=RequestSource.WORKER,
        spec=IntegrateSpec(source_request_id=uuid4()),
    )


def _registry() -> AdapterRegistry:
    registry = AdapterRegistry()
    registry.load(_ADAPTERS_DIR)
    return registry


# --- Tests ---


async def test_runner_routes_plan_to_plan_handler() -> None:
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
    runner = Runner()
    request = _plan_request()
    response = await runner(request)

    assert response.status == ResponseStatus.FAILED
    assert response.error == f"no handler registered for: {AgentType.PLAN.value}"


async def test_registering_second_handler_overwrites_first() -> None:
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
    response = await stub_plan_handler(_plan_request())

    assert response.status == ResponseStatus.COMPLETED


async def test_work_handler_includes_adapter_in_delta() -> None:
    handler = make_work_handler(_registry())
    response = await handler(_work_request())

    assert response.delta is not None
    assert "coding" in str(response.delta.get("result", ""))


async def test_stub_integrate_handler_returns_completed() -> None:
    response = await stub_integrate_handler(_integrate_request())

    assert response.status == ResponseStatus.COMPLETED


async def test_runner_satisfies_agent_runner_type() -> None:
    runner = Runner()
    runner.register(AgentType.PLAN, stub_plan_handler)
    runner.register(AgentType.WORK, make_work_handler(_registry()))
    runner.register(AgentType.INTEGRATE, stub_integrate_handler)

    state = SchedulerState(northstar="test northstar")
    final = await Scheduler(runner=runner).run(state, _plan_request())

    assert final is not None


async def test_scripted_plan_handler_user_source_emits_three_follow_ups() -> None:
    response = await scripted_plan_handler(_plan_request())

    assert len(response.follow_up) == 3


async def test_scripted_plan_handler_follow_ups_form_valid_chain() -> None:
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
    planner_request = AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.PLANNER,
        spec=PlanSpec(northstar="test northstar"),
    )
    response = await scripted_plan_handler(planner_request)

    assert response.follow_up == []


async def test_scripted_plan_handler_end_to_end_produces_five_completed_nodes() -> None:
    runner = Runner()
    runner.register(AgentType.PLAN, scripted_plan_handler)
    runner.register(AgentType.WORK, make_work_handler(_registry()))

    state = SchedulerState(northstar="test northstar")
    final = await Scheduler(runner=runner).run(state, _plan_request())

    completed = [n for n in final.dag.values() if n.node_state.value == "completed"]
    assert len(completed) == 5


async def test_scheduler_reinjects_global_planner_with_planner_source() -> None:
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
