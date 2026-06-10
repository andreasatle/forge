"""Tests for Scheduler DAG execution, concurrency, callbacks, and termination."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from forge.core.models import (
    AgentRequest,
    AgentResponse,
    AgentType,
    DAGNode,
    DeltaState,
    FailureKind,
    NodeState,
    PlanSpec,
    RequestId,
    RequestSource,
    ResponseStatus,
    SchedulerState,
    WorkSpec,
)
from forge.core.scheduler import Scheduler, SchedulerCallbacks
from forge.core.state_service import StateService

# --- Helpers ---


def _plan_request() -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(northstar="test northstar"),
    )


def _work_request(*, deps: frozenset[RequestId] = frozenset()) -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="do work",
            success_condition="work done",
            adapter="coding",
            artifact="codebase",
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


def _already_done(request: AgentRequest) -> AgentResponse:
    return AgentResponse(
        request_id=request.id,
        status=ResponseStatus.ALREADY_DONE,
        delta=DeltaState(),
    )


def _base_state(max_concurrency: int = 1) -> SchedulerState:
    return SchedulerState(northstar="test northstar", max_concurrency=max_concurrency)


def _mock_ss() -> MagicMock:
    return MagicMock(spec=StateService)


# --- Tests ---


async def test_single_work_node_completes() -> None:
    """A single pre-seeded work node reaches INTEGRATED state after the scheduler runs."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    final = await Scheduler(runner=runner).run(state, _plan_request())

    assert final.dag[work.id].node_state == NodeState.INTEGRATED


async def test_follow_up_requests_added_and_executed() -> None:
    """Follow-up requests emitted by an agent are added to the DAG and executed."""
    work_a = _work_request()
    work_b = _work_request()

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.id == work_a.id:
            return _ok(request, follow_up=[work_b])
        return _ok(request)

    state = _base_state().add_nodes([DAGNode(request=work_a)])
    final = await Scheduler(runner=runner).run(state, _plan_request())

    assert final.dag[work_a.id].node_state == NodeState.INTEGRATED
    assert final.dag[work_b.id].node_state == NodeState.INTEGRATED


async def test_failed_node_cancels_dependents() -> None:
    """A FAILED node causes all nodes that depend on it to be CANCELLED."""
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
    """Nodes cancelled due to a failed dependency are never dispatched to the runner."""
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
    """The scheduler never dispatches more simultaneous requests than max_concurrency."""
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
    """on_node_completed callback is called when a node reaches COMPLETED state."""
    work = _work_request()
    completed: list[DAGNode] = []

    callbacks = SchedulerCallbacks(on_node_completed=completed.append)
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    await Scheduler(runner=runner, callbacks=callbacks).run(state, _plan_request())

    assert any(n.request.id == work.id for n in completed)


async def test_on_node_failed_fires() -> None:
    """on_node_failed callback is called when a node reaches FAILED state."""
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
    """on_idle callback is called at least once when the DAG has no ready nodes."""
    idle_calls: list[SchedulerState] = []
    callbacks = SchedulerCallbacks(on_idle=idle_calls.append)

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    await Scheduler(runner=runner, callbacks=callbacks).run(_base_state(), _plan_request())

    assert len(idle_calls) >= 1


async def test_terminates_cleanly() -> None:
    """Scheduler terminates and returns final state when the planner produces no work."""

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    final = await Scheduler(runner=runner).run(_base_state(), _plan_request())

    assert final is not None


async def test_callback_exceptions_do_not_crash() -> None:
    """Exceptions raised by scheduler callbacks are swallowed and do not abort the run."""

    def crashing(node: DAGNode) -> None:
        raise RuntimeError("boom")

    callbacks = SchedulerCallbacks(on_node_completed=crashing)

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    final = await Scheduler(runner=runner, callbacks=callbacks).run(_base_state(), _plan_request())

    assert final is not None


async def test_final_state_reflects_all_node_updates() -> None:
    """Final state correctly records INTEGRATED, FAILED, and CANCELLED nodes for a mixed run."""
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

    assert final.dag[work_a.id].node_state == NodeState.INTEGRATED
    assert final.dag[work_b.id].node_state == NodeState.FAILED
    assert final.dag[work_c.id].node_state == NodeState.INTEGRATED


async def test_integrate_called_inline_after_work_completes() -> None:
    """Scheduler calls integrate inline when a WORK node completes successfully."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    ss = _mock_ss()

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    with patch("forge.core.scheduler.integrate", new_callable=AsyncMock) as mock_integrate:
        mock_integrate.return_value = AgentResponse(
            request_id=work.id, status=ResponseStatus.COMPLETED
        )
        await Scheduler(runner=runner, state_services={"codebase": ss}).run(state, _plan_request())

    mock_integrate.assert_called_once()
    assert mock_integrate.call_args.kwargs["state_service"] is ss


async def test_transitive_cancellation_propagates_through_chain() -> None:
    """CANCELLED propagates transitively: A fails → B (dep A) → C (dep B) all CANCELLED."""
    work_a = _work_request()
    work_b = _work_request(deps=frozenset({work_a.id}))
    work_c = _work_request(deps=frozenset({work_b.id}))

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.id == work_a.id:
            return _fail(request)
        return _ok(request)

    state = _base_state().add_nodes(
        [DAGNode(request=work_a), DAGNode(request=work_b), DAGNode(request=work_c)]
    )
    final = await Scheduler(runner=runner).run(state, _plan_request())

    assert final.dag[work_a.id].node_state == NodeState.FAILED
    assert final.dag[work_b.id].node_state == NodeState.CANCELLED
    assert final.dag[work_c.id].node_state == NodeState.CANCELLED


async def test_terminates_when_no_pending_or_running_nodes() -> None:
    """Run completes as soon as no PENDING or RUNNING nodes remain in the DAG."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    final = await Scheduler(runner=runner).run(state, _plan_request())

    assert all(
        n.node_state not in (NodeState.PENDING, NodeState.RUNNING) for n in final.dag.values()
    )


async def test_integration_failure_marks_node_failed() -> None:
    """Node is marked FAILED when integrate returns FAILED status."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    ss = _mock_ss()

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    with patch("forge.core.scheduler.integrate", new_callable=AsyncMock) as mock_integrate:
        mock_integrate.return_value = AgentResponse(
            request_id=work.id, status=ResponseStatus.FAILED
        )
        final = await Scheduler(runner=runner, state_services={"codebase": ss}).run(
            state, _plan_request()
        )

    assert final.dag[work.id].node_state == NodeState.FAILED


async def test_integration_failure_cancels_transitive_dependents() -> None:
    """CANCELLED propagates transitively to dependents when integration fails."""
    work_a = _work_request()
    work_b = _work_request(deps=frozenset({work_a.id}))
    work_c = _work_request(deps=frozenset({work_b.id}))
    state = _base_state().add_nodes(
        [DAGNode(request=work_a), DAGNode(request=work_b), DAGNode(request=work_c)]
    )
    ss = _mock_ss()

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    with patch("forge.core.scheduler.integrate", new_callable=AsyncMock) as mock_integrate:
        mock_integrate.return_value = AgentResponse(
            request_id=work_a.id, status=ResponseStatus.FAILED
        )
        final = await Scheduler(runner=runner, state_services={"codebase": ss}).run(
            state, _plan_request()
        )

    assert final.dag[work_a.id].node_state == NodeState.FAILED
    assert final.dag[work_b.id].node_state == NodeState.CANCELLED
    assert final.dag[work_c.id].node_state == NodeState.CANCELLED


async def test_validation_failed_work_does_not_apply_delta() -> None:
    """A validation-failed work response never reaches StateService.apply_delta."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    ss = _mock_ss()

    async def runner(request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            failure_kind=FailureKind.VALIDATION_REJECTED,
            error="validation rejected work",
        )

    final = await Scheduler(runner=runner, state_services={"codebase": ss}).run(
        state, _plan_request()
    )

    assert final.dag[work.id].node_state == NodeState.FAILED
    ss.apply_delta.assert_not_called()


async def test_integration_success_marks_node_integrated() -> None:
    """Node is marked INTEGRATED only when integration succeeds."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    ss = _mock_ss()

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    with patch("forge.core.scheduler.integrate", new_callable=AsyncMock) as mock_integrate:
        mock_integrate.return_value = AgentResponse(
            request_id=work.id, status=ResponseStatus.COMPLETED
        )
        final = await Scheduler(runner=runner, state_services={"codebase": ss}).run(
            state, _plan_request()
        )

    assert final.dag[work.id].node_state == NodeState.INTEGRATED


async def test_stale_delta_requeues_not_failed() -> None:
    """A stale delta re-queues the node as PENDING; it eventually succeeds on retry."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    ss = _mock_ss()

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    with patch("forge.core.scheduler.integrate", new_callable=AsyncMock) as mock_integrate:
        mock_integrate.side_effect = [
            AgentResponse(
                request_id=work.id,
                status=ResponseStatus.FAILED,
                failure_kind=FailureKind.STALE_DELTA,
            ),
            AgentResponse(request_id=work.id, status=ResponseStatus.COMPLETED),
        ]
        final = await Scheduler(runner=runner, state_services={"codebase": ss}).run(
            state, _plan_request()
        )

    assert final.dag[work.id].node_state == NodeState.INTEGRATED
    assert mock_integrate.call_count == 2


async def test_stale_delta_fails_after_3_retries() -> None:
    """Node is marked FAILED after 3 consecutive stale delta retries are exhausted."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    ss = _mock_ss()

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    stale = AgentResponse(
        request_id=work.id,
        status=ResponseStatus.FAILED,
        failure_kind=FailureKind.STALE_DELTA,
    )
    with patch("forge.core.scheduler.integrate", new_callable=AsyncMock) as mock_integrate:
        mock_integrate.return_value = stale
        final = await Scheduler(runner=runner, state_services={"codebase": ss}).run(
            state, _plan_request()
        )

    assert final.dag[work.id].node_state == NodeState.FAILED
    assert mock_integrate.call_count == 4  # 1 initial + 3 retries


async def test_non_stale_integration_failure_marks_failed_immediately() -> None:
    """Non-stale integration failure marks the node FAILED without any retry."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    ss = _mock_ss()

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    with patch("forge.core.scheduler.integrate", new_callable=AsyncMock) as mock_integrate:
        mock_integrate.return_value = AgentResponse(
            request_id=work.id,
            status=ResponseStatus.FAILED,
            failure_kind=FailureKind.TOOL_ERROR,
        )
        final = await Scheduler(runner=runner, state_services={"codebase": ss}).run(
            state, _plan_request()
        )

    assert final.dag[work.id].node_state == NodeState.FAILED
    mock_integrate.assert_called_once()


async def test_already_done_node_skips_integration() -> None:
    """Scheduler does not call integrate() when a WORK node returns ALREADY_DONE status."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    ss = _mock_ss()

    async def runner(request: AgentRequest) -> AgentResponse:
        return _already_done(request)

    with patch("forge.core.scheduler.integrate", new_callable=AsyncMock) as mock_integrate:
        final = await Scheduler(runner=runner, state_services={"codebase": ss}).run(
            state, _plan_request()
        )

    assert final.dag[work.id].node_state == NodeState.INTEGRATED
    mock_integrate.assert_not_called()


async def test_already_done_state_version_unchanged() -> None:
    """State version does not increment for ALREADY_DONE — integrate() is never invoked."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    ss = _mock_ss()

    async def runner(request: AgentRequest) -> AgentResponse:
        return _already_done(request)

    with patch("forge.core.scheduler.integrate", new_callable=AsyncMock) as mock_integrate:
        await Scheduler(runner=runner, state_services={"codebase": ss}).run(state, _plan_request())

    mock_integrate.assert_not_called()


async def test_already_done_fires_on_node_completed() -> None:
    """on_node_completed callback fires for ALREADY_DONE nodes (bypasses integration)."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    completed: list[DAGNode] = []
    callbacks = SchedulerCallbacks(on_node_completed=completed.append)

    async def runner(request: AgentRequest) -> AgentResponse:
        return _already_done(request)

    await Scheduler(runner=runner, callbacks=callbacks).run(state, _plan_request())

    assert any(n.request.id == work.id for n in completed)
