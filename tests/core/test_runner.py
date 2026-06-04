from uuid import uuid4

from forge.core.models import (
    AdapterType,
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
    stub_integrate_handler,
    stub_plan_handler,
    stub_work_handler,
)
from forge.core.scheduler import Scheduler

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
            adapter_type=AdapterType.CODING,
        ),
    )


def _integrate_request() -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.INTEGRATE,
        source=RequestSource.WORKER,
        spec=IntegrateSpec(source_request_id=uuid4()),
    )


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


async def test_stub_work_handler_includes_adapter_in_delta() -> None:
    response = await stub_work_handler(_work_request())

    assert response.delta is not None
    assert AdapterType.CODING.value in str(response.delta.get("result", ""))


async def test_stub_integrate_handler_returns_completed() -> None:
    response = await stub_integrate_handler(_integrate_request())

    assert response.status == ResponseStatus.COMPLETED


async def test_runner_satisfies_agent_runner_type() -> None:
    runner = Runner()
    runner.register(AgentType.PLAN, stub_plan_handler)
    runner.register(AgentType.WORK, stub_work_handler)
    runner.register(AgentType.INTEGRATE, stub_integrate_handler)

    state = SchedulerState(northstar="test northstar")
    final = await Scheduler(runner=runner).run(state, _plan_request())

    assert final is not None
