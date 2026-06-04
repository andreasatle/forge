import asyncio

from forge.core.models import (
    AdapterType,
    AgentRequest,
    AgentResponse,
    AgentType,
    DAGNode,
    NodeState,
    PlanSpec,
    Priority,
    RequestId,
    RequestSource,
    ResponseStatus,
    SchedulerState,
    WorkSpec,
)
from forge.core.scheduler import Scheduler, SchedulerCallbacks

# --- Helpers ---


def _plan_request() -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(northstar="test northstar"),
        priority=Priority.HIGH,
    )


def _work_request(*, deps: frozenset[RequestId] = frozenset()) -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="do work",
            success_condition="work done",
            adapter_type=AdapterType.CODING,
        ),
        dependencies=deps,
    )


def _ok(request: AgentRequest, *, follow_up: list[AgentRequest] | None = None) -> AgentResponse:
    return AgentResponse(
        request_id=request.id,
        status=ResponseStatus.COMPLETED,
        follow_up=follow_up or [],
    )


def _fail(request: AgentRequest) -> AgentResponse:
    return AgentResponse(request_id=request.id, status=ResponseStatus.FAILED, error="test error")


def _base_state(max_concurrency: int = 1) -> SchedulerState:
    return SchedulerState(northstar="test northstar", max_concurrency=max_concurrency)


# --- Tests ---


async def test_single_work_node_completes() -> None:
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    final = await Scheduler(runner=runner).run(state, _plan_request())

    assert final.dag[work.id].node_state == NodeState.COMPLETED


async def test_follow_up_requests_added_and_executed() -> None:
    work_a = _work_request()
    work_b = _work_request()

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.id == work_a.id:
            return _ok(request, follow_up=[work_b])
        return _ok(request)

    state = _base_state().add_nodes([DAGNode(request=work_a)])
    final = await Scheduler(runner=runner).run(state, _plan_request())

    assert final.dag[work_a.id].node_state == NodeState.COMPLETED
    assert final.dag[work_b.id].node_state == NodeState.COMPLETED


async def test_failed_node_cancels_dependents() -> None:
    work_a = _work_request()
    work_b = _work_request(deps=frozenset({work_a.id}))

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.id == work_a.id:
            return _fail(request)
        return _ok(request)

    state = _base_state().add_nodes([DAGNode(request=work_a), DAGNode(request=work_b)])
    final = await Scheduler(runner=runner).run(state, _plan_request())

    assert final.dag[work_a.id].node_state == NodeState.FAILED
    assert final.dag[work_b.id].node_state == NodeState.CANCELLED


async def test_cancelled_nodes_never_dispatched() -> None:
    work_a = _work_request()
    work_b = _work_request(deps=frozenset({work_a.id}))
    dispatched: list[RequestId] = []

    async def runner(request: AgentRequest) -> AgentResponse:
        dispatched.append(request.id)
        if request.id == work_a.id:
            return _fail(request)
        return _ok(request)

    state = _base_state().add_nodes([DAGNode(request=work_a), DAGNode(request=work_b)])
    await Scheduler(runner=runner).run(state, _plan_request())

    assert work_b.id not in dispatched


async def test_max_concurrency_respected() -> None:
    works = [_work_request() for _ in range(3)]
    running = 0
    max_running = 0

    async def runner(request: AgentRequest) -> AgentResponse:
        nonlocal running, max_running
        if request.agent_type == AgentType.WORK:
            running += 1
            max_running = max(max_running, running)
            await asyncio.sleep(0)
            running -= 1
        return _ok(request)

    state = _base_state(max_concurrency=2).add_nodes([DAGNode(request=w) for w in works])
    await Scheduler(runner=runner).run(state, _plan_request())

    assert max_running <= 2
    assert max_running == 2


async def test_on_node_completed_fires() -> None:
    work = _work_request()
    completed: list[DAGNode] = []

    callbacks = SchedulerCallbacks(on_node_completed=completed.append)
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    await Scheduler(runner=runner, callbacks=callbacks).run(state, _plan_request())

    assert any(n.request.id == work.id for n in completed)


async def test_on_node_failed_fires() -> None:
    work = _work_request()
    failed: list[DAGNode] = []

    callbacks = SchedulerCallbacks(on_node_failed=failed.append)
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.id == work.id:
            return _fail(request)
        return _ok(request)

    await Scheduler(runner=runner, callbacks=callbacks).run(state, _plan_request())

    assert any(n.request.id == work.id for n in failed)


async def test_on_idle_fires() -> None:
    idle_calls: list[SchedulerState] = []
    callbacks = SchedulerCallbacks(on_idle=idle_calls.append)

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    await Scheduler(runner=runner, callbacks=callbacks).run(_base_state(), _plan_request())

    assert len(idle_calls) >= 1


async def test_global_planner_reinjected_on_idle() -> None:
    plan_calls = 0

    async def runner(request: AgentRequest) -> AgentResponse:
        nonlocal plan_calls
        if request.agent_type == AgentType.PLAN:
            plan_calls += 1
        return _ok(request)

    await Scheduler(runner=runner).run(_base_state(), _plan_request())

    assert plan_calls >= 2


async def test_terminates_when_global_planner_returns_empty_follow_up() -> None:
    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    final = await Scheduler(runner=runner).run(_base_state(), _plan_request())

    assert final is not None


async def test_callback_exceptions_do_not_crash() -> None:
    def crashing(node: DAGNode) -> None:
        raise RuntimeError("boom")

    callbacks = SchedulerCallbacks(on_node_completed=crashing)

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    final = await Scheduler(runner=runner, callbacks=callbacks).run(_base_state(), _plan_request())

    assert final is not None


async def test_final_state_reflects_all_node_updates() -> None:
    work_a = _work_request()
    work_b = _work_request()
    work_c = _work_request(deps=frozenset({work_a.id}))

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.id == work_b.id:
            return _fail(request)
        return _ok(request)

    state = _base_state(max_concurrency=2).add_nodes(
        [DAGNode(request=work_a), DAGNode(request=work_b), DAGNode(request=work_c)]
    )
    final = await Scheduler(runner=runner).run(state, _plan_request())

    assert final.dag[work_a.id].node_state == NodeState.COMPLETED
    assert final.dag[work_b.id].node_state == NodeState.FAILED
    assert final.dag[work_c.id].node_state == NodeState.COMPLETED
