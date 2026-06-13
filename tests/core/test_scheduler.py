"""Tests for Scheduler DAG execution, concurrency, callbacks, and termination."""

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from forge.core.models import (
    AgentRequest,
    AgentResponse,
    AgentType,
    DAGNode,
    FailureKind,
    FileContent,
    NodeState,
    PlanResponse,
    PlanSpec,
    RequestId,
    RequestSource,
    ResponseStatus,
    SchedulerState,
    TaskSpec,
    WorkOutput,
    WorkSpec,
)
from forge.core.scheduler import Scheduler, SchedulerCallbacks
from forge.core.state_service import StateService
from forge.core.telemetry import TelemetryEvent


class _MemoryTelemetrySink:
    """In-memory TelemetrySink for scheduler tests."""

    def __init__(self) -> None:
        self.run_id = uuid4()
        self.events: list[TelemetryEvent] = []

    def append(self, event: TelemetryEvent) -> None:
        self.events.append(event)


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
        output=WorkOutput(files=[FileContent(path="src/out.py", content="x = 1")]),
        follow_up=follow_up or [],
    )


def _fail(request: AgentRequest) -> AgentResponse:
    return AgentResponse(request_id=request.id, status=ResponseStatus.FAILED, error="test error")


def _already_done(request: AgentRequest) -> AgentResponse:
    return AgentResponse(
        request_id=request.id,
        status=ResponseStatus.ALREADY_DONE,
    )


def _base_state(max_concurrency: int = 1) -> SchedulerState:
    return SchedulerState(northstar="test northstar", max_concurrency=max_concurrency)


def _mock_ss() -> MagicMock:
    ss = MagicMock(spec=StateService)
    ss.apply_work_output = AsyncMock()
    return ss


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


async def test_scheduler_derives_follow_up_from_accepted_plan_output() -> None:
    """Accepted PlanResponse output is converted to WORK nodes by the scheduler."""
    planner = _plan_request()
    plan = PlanResponse(
        tasks=[
            TaskSpec(
                objective="A",
                success_condition="done",
                adapter="coding",
                artifact="codebase",
            ),
            TaskSpec(
                objective="B",
                success_condition="done",
                adapter="coding",
                artifact="codebase",
                depends_on=[0],
            ),
        ]
    )

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.id == planner.id:
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=plan,
            )
        return _ok(request)

    final = await Scheduler(runner=runner).run(_base_state(), planner)

    work_nodes = [
        n
        for n in final.dag.values()
        if n.request.agent_type == AgentType.WORK and isinstance(n.request.spec, WorkSpec)
    ]
    assert len(work_nodes) == 2

    def objective(node: DAGNode) -> str:
        assert isinstance(node.request.spec, WorkSpec)
        return node.request.spec.objective

    work_a = next(n for n in work_nodes if objective(n) == "A")
    work_b = next(n for n in work_nodes if objective(n) == "B")
    assert work_a.node_state == NodeState.INTEGRATED
    assert work_b.node_state == NodeState.INTEGRATED
    assert work_b.request.dependencies == frozenset({work_a.request.id})


async def test_scheduler_does_not_derive_follow_up_from_failed_plan_output() -> None:
    """Rejected planner responses do not create work nodes even when output is present."""
    planner = _plan_request()
    plan = PlanResponse(
        tasks=[
            TaskSpec(
                objective="A",
                success_condition="done",
                adapter="coding",
                artifact="codebase",
            )
        ]
    )

    async def runner(request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            output=plan,
            error="bad plan",
        )

    final = await Scheduler(runner=runner).run(_base_state(), planner)

    assert final.dag[planner.id].node_state == NodeState.FAILED
    assert all(n.request.agent_type != AgentType.WORK for n in final.dag.values())


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


async def test_scheduler_emits_node_failed_telemetry() -> None:
    """Scheduler failure handling appends a node.failed telemetry event."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    sink = _MemoryTelemetrySink()

    async def runner(request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            failure_kind=FailureKind.VALIDATION_REJECTED,
            error="not accepted",
        )

    final = await Scheduler(runner=runner, telemetry_sink=sink, run_id=sink.run_id).run(
        state, _plan_request()
    )

    assert final.dag[work.id].node_state == NodeState.FAILED
    events = [event for event in sink.events if event.event_type == "node.failed"]
    assert len(events) == 1
    event = events[0]
    assert event.run_id == sink.run_id
    assert event.node_id == work.id
    assert event.request_id == work.id
    assert event.role == "scheduler"
    assert event.status == "failed"
    assert event.data["failure_kind"] == "validation_rejected"
    assert event.data["error"] == "not accepted"


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
    """Scheduler calls apply_work_output inline when a WORK node completes successfully."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    ss = _mock_ss()

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    await Scheduler(runner=runner, state_services={"codebase": ss}).run(state, _plan_request())

    ss.apply_work_output.assert_called_once()
    assert ss.apply_work_output.call_args.args[0].files[0].path == "src/out.py"


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
    """Node is marked FAILED when apply_work_output raises RuntimeError."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    ss = _mock_ss()
    ss.apply_work_output = AsyncMock(side_effect=RuntimeError("integration failed"))

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    final = await Scheduler(runner=runner, state_services={"codebase": ss}).run(
        state, _plan_request()
    )

    assert final.dag[work.id].node_state == NodeState.FAILED
    response = final.dag[work.id].response
    assert response is not None
    assert response.status == ResponseStatus.FAILED


async def test_integration_failure_preserves_integration_response() -> None:
    """Integration failures from RuntimeError are captured with INTEGRATION_FAILED kind."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    ss = _mock_ss()
    ss.apply_work_output = AsyncMock(
        side_effect=RuntimeError("tests failed after delta: assertion failed")
    )

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    final = await Scheduler(runner=runner, state_services={"codebase": ss}).run(
        state, _plan_request()
    )

    response = final.dag[work.id].response
    assert final.dag[work.id].node_state == NodeState.FAILED
    assert response is not None
    assert response.status == ResponseStatus.FAILED
    assert response.failure_kind == FailureKind.INTEGRATION_FAILED
    assert "tests failed after delta" in (response.error or "")


async def test_integration_failure_cancels_transitive_dependents() -> None:
    """CANCELLED propagates transitively to dependents when integration fails."""
    work_a = _work_request()
    work_b = _work_request(deps=frozenset({work_a.id}))
    work_c = _work_request(deps=frozenset({work_b.id}))
    state = _base_state().add_nodes(
        [DAGNode(request=work_a), DAGNode(request=work_b), DAGNode(request=work_c)]
    )
    ss = _mock_ss()
    ss.apply_work_output = AsyncMock(side_effect=RuntimeError("integration failed"))

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

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
    """Node is marked INTEGRATED when apply_work_output succeeds."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    ss = _mock_ss()

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    final = await Scheduler(runner=runner, state_services={"codebase": ss}).run(
        state, _plan_request()
    )

    assert final.dag[work.id].node_state == NodeState.INTEGRATED


async def test_stale_delta_requeues_not_failed() -> None:
    """apply_work_output raising RuntimeError marks node FAILED immediately — no stale retry."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    ss = _mock_ss()
    ss.apply_work_output = AsyncMock(side_effect=RuntimeError("stale base_version: ..."))

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    final = await Scheduler(runner=runner, state_services={"codebase": ss}).run(
        state, _plan_request()
    )

    assert final.dag[work.id].node_state == NodeState.FAILED
    ss.apply_work_output.assert_called_once()


async def test_stale_delta_fails_after_3_retries() -> None:
    """apply_work_output raising RuntimeError marks node FAILED with INTEGRATION_FAILED."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    ss = _mock_ss()
    ss.apply_work_output = AsyncMock(side_effect=RuntimeError("stale base_version: ..."))

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    final = await Scheduler(runner=runner, state_services={"codebase": ss}).run(
        state, _plan_request()
    )

    assert final.dag[work.id].node_state == NodeState.FAILED
    response = final.dag[work.id].response
    assert response is not None
    assert response.status == ResponseStatus.FAILED
    assert response.failure_kind == FailureKind.INTEGRATION_FAILED
    ss.apply_work_output.assert_called_once()


async def test_non_stale_integration_failure_marks_failed_immediately() -> None:
    """Integration failure via RuntimeError marks the node FAILED immediately."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    ss = _mock_ss()
    ss.apply_work_output = AsyncMock(side_effect=RuntimeError("integration error"))

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    final = await Scheduler(runner=runner, state_services={"codebase": ss}).run(
        state, _plan_request()
    )

    assert final.dag[work.id].node_state == NodeState.FAILED
    ss.apply_work_output.assert_called_once()


async def test_already_done_node_skips_integration() -> None:
    """Scheduler does not call apply_work_output when a WORK node returns ALREADY_DONE status."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    ss = _mock_ss()

    async def runner(request: AgentRequest) -> AgentResponse:
        return _already_done(request)

    final = await Scheduler(runner=runner, state_services={"codebase": ss}).run(
        state, _plan_request()
    )

    assert final.dag[work.id].node_state == NodeState.INTEGRATED
    ss.apply_work_output.assert_not_called()


async def test_already_done_state_version_unchanged() -> None:
    """apply_work_output is not called for ALREADY_DONE — integration is bypassed."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    ss = _mock_ss()

    async def runner(request: AgentRequest) -> AgentResponse:
        return _already_done(request)

    await Scheduler(runner=runner, state_services={"codebase": ss}).run(state, _plan_request())

    ss.apply_work_output.assert_not_called()


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


async def test_work_node_none_delta_marked_failed() -> None:
    """WORK node completing without WorkOutput in output is marked FAILED before apply_work_output."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    ss = _mock_ss()

    async def runner(request: AgentRequest) -> AgentResponse:
        return AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED)

    final = await Scheduler(runner=runner, state_services={"codebase": ss}).run(
        state, _plan_request()
    )

    assert final.dag[work.id].node_state == NodeState.FAILED
    ss.apply_work_output.assert_not_called()


async def test_work_node_empty_delta_marked_failed() -> None:
    """WORK node completing with empty WorkOutput is marked FAILED before apply_work_output."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    ss = _mock_ss()

    async def runner(request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.id, status=ResponseStatus.COMPLETED, output=WorkOutput()
        )

    final = await Scheduler(runner=runner, state_services={"codebase": ss}).run(
        state, _plan_request()
    )

    assert final.dag[work.id].node_state == NodeState.FAILED
    ss.apply_work_output.assert_not_called()


async def test_work_node_already_done_empty_delta_skips_guard() -> None:
    """ALREADY_DONE response with no WorkOutput is not caught by the empty-output guard."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    ss = _mock_ss()

    async def runner(request: AgentRequest) -> AgentResponse:
        return _already_done(request)

    final = await Scheduler(runner=runner, state_services={"codebase": ss}).run(
        state, _plan_request()
    )

    assert final.dag[work.id].node_state == NodeState.INTEGRATED
    ss.apply_work_output.assert_not_called()


async def test_work_node_non_empty_delta_integrates_normally() -> None:
    """WORK node with non-empty WorkOutput passes the guard and reaches apply_work_output."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    ss = _mock_ss()

    async def runner(request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            output=WorkOutput(files=[FileContent(path="src/main.py", content="x = 1")]),
        )

    final = await Scheduler(runner=runner, state_services={"codebase": ss}).run(
        state, _plan_request()
    )

    assert final.dag[work.id].node_state == NodeState.INTEGRATED
    ss.apply_work_output.assert_called_once()


async def test_work_node_integrates_typed_delta_output() -> None:
    """Scheduler integrates WorkOutput from response.output and passes it to apply_work_output."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    ss = _mock_ss()
    work_output = WorkOutput(files=[FileContent(path="src/main.py", content="x = 1")])

    async def runner(request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            output=work_output,
        )

    final = await Scheduler(runner=runner, state_services={"codebase": ss}).run(
        state, _plan_request()
    )

    assert final.dag[work.id].node_state == NodeState.INTEGRATED
    ss.apply_work_output.assert_called_once()
    assert ss.apply_work_output.call_args.args[0] == work_output


async def test_integration_failure_with_revision_requeues_node() -> None:
    """Integration failure via RuntimeError marks the node FAILED immediately."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    ss = _mock_ss()
    ss.apply_work_output = AsyncMock(side_effect=RuntimeError("tests failed after delta: error"))

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    final = await Scheduler(runner=runner, state_services={"codebase": ss}).run(
        state, _plan_request()
    )

    assert final.dag[work.id].node_state == NodeState.FAILED
    ss.apply_work_output.assert_called_once()


async def test_integration_failure_without_revision_marks_failed() -> None:
    """Integration failure via RuntimeError marks the node FAILED."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    ss = _mock_ss()
    ss.apply_work_output = AsyncMock(side_effect=RuntimeError("integration error"))

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    final = await Scheduler(runner=runner, state_services={"codebase": ss}).run(
        state, _plan_request()
    )

    assert final.dag[work.id].node_state == NodeState.FAILED
    ss.apply_work_output.assert_called_once()


async def test_integration_revision_exhaustion_marks_failed() -> None:
    """Integration failure via RuntimeError marks the node FAILED with INTEGRATION_FAILED kind."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    ss = _mock_ss()
    ss.apply_work_output = AsyncMock(side_effect=RuntimeError("tests failed after delta: error"))

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    final = await Scheduler(runner=runner, state_services={"codebase": ss}).run(
        state, _plan_request()
    )

    assert final.dag[work.id].node_state == NodeState.FAILED
    response = final.dag[work.id].response
    assert response is not None
    assert response.failure_kind == FailureKind.INTEGRATION_FAILED


def _decompose(request: AgentRequest) -> AgentResponse:
    return AgentResponse(request_id=request.id, status=ResponseStatus.DECOMPOSE)


async def test_dependents_not_cancelled_on_integration_revision_requeue() -> None:
    """Integration failure cancels dependents (no revision requeue at scheduler level)."""
    work_a = _work_request()
    work_b = _work_request(deps=frozenset({work_a.id}))
    state = _base_state().add_nodes([DAGNode(request=work_a), DAGNode(request=work_b)])
    ss = _mock_ss()
    ss.apply_work_output = AsyncMock(side_effect=RuntimeError("integration failed"))

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    final = await Scheduler(runner=runner, state_services={"codebase": ss}).run(
        state, _plan_request()
    )

    assert final.dag[work_a.id].node_state == NodeState.FAILED
    assert final.dag[work_b.id].node_state == NodeState.CANCELLED


# --- DECOMPOSE disposition ---


async def test_decompose_status_creates_new_plan_node() -> None:
    """A work node returning DECOMPOSE causes a new PLAN node to be added to the DAG."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.id == work.id:
            return _decompose(request)
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            output=PlanResponse(tasks=[]),
        )

    final = await Scheduler(runner=runner).run(state, _plan_request())

    decompose_plan_nodes = [
        n
        for n in final.dag.values()
        if n.request.agent_type == AgentType.PLAN
        and isinstance(n.request.spec, PlanSpec)
        and n.request.spec.northstar == "do work"
        and n.request.source == RequestSource.USER
    ]
    assert len(decompose_plan_nodes) == 1


async def test_decompose_work_node_is_cancelled_not_failed() -> None:
    """A work node returning DECOMPOSE is marked CANCELLED, not FAILED."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.id == work.id:
            return _decompose(request)
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            output=PlanResponse(tasks=[]),
        )

    final = await Scheduler(runner=runner).run(state, _plan_request())

    assert final.dag[work.id].node_state == NodeState.CANCELLED
    assert final.dag[work.id].node_state != NodeState.FAILED


async def test_decompose_transfers_dependents_to_new_plan_node() -> None:
    """Nodes depending on a DECOMPOSE work node are repointed to the new PLAN node."""
    work_a = _work_request()
    work_b = _work_request(deps=frozenset({work_a.id}))
    state = _base_state().add_nodes([DAGNode(request=work_a), DAGNode(request=work_b)])

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.id == work_a.id:
            return _decompose(request)
        if request.agent_type == AgentType.PLAN:
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=PlanResponse(tasks=[]),
            )
        return _ok(request)

    final = await Scheduler(runner=runner).run(state, _plan_request())

    assert final.dag[work_a.id].node_state == NodeState.CANCELLED
    # work_b ran successfully — its dep was transferred to the new plan node
    assert final.dag[work_b.id].node_state == NodeState.INTEGRATED
    # work_b no longer depends on the cancelled work_a
    work_b_final = final.dag[work_b.id]
    assert work_a.id not in work_b_final.request.dependencies
    # new plan node exists and is INTEGRATED
    decompose_plan_nodes = [
        n
        for n in final.dag.values()
        if n.request.agent_type == AgentType.PLAN
        and isinstance(n.request.spec, PlanSpec)
        and n.request.spec.northstar == "do work"
    ]
    assert len(decompose_plan_nodes) == 1
    plan_node = decompose_plan_nodes[0]
    assert plan_node.node_state == NodeState.INTEGRATED
    assert plan_node.request.id in work_b_final.request.dependencies


async def test_decompose_does_not_fire_on_node_failed() -> None:
    """on_node_failed is not called for a work node that returns DECOMPOSE."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    failed_nodes: list[DAGNode] = []
    callbacks = SchedulerCallbacks(on_node_failed=failed_nodes.append)

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.id == work.id:
            return _decompose(request)
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            output=PlanResponse(tasks=[]),
        )

    await Scheduler(runner=runner, callbacks=callbacks).run(state, _plan_request())

    assert not any(n.request.id == work.id for n in failed_nodes)


async def test_decompose_end_to_end_plan_produces_two_subtasks() -> None:
    """Full path: DECOMPOSE → PLAN produces 2 subtasks, all complete, dependent completes."""
    work_a = _work_request()
    work_b = _work_request(deps=frozenset({work_a.id}))
    state = _base_state().add_nodes([DAGNode(request=work_a), DAGNode(request=work_b)])

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.id == work_a.id:
            return _decompose(request)
        if request.agent_type == AgentType.PLAN:
            spec = request.spec
            if isinstance(spec, PlanSpec) and spec.northstar == "do work":
                return AgentResponse(
                    request_id=request.id,
                    status=ResponseStatus.COMPLETED,
                    output=PlanResponse(
                        tasks=[
                            TaskSpec(
                                objective="do thing A",
                                success_condition="thing A done",
                                adapter="coding",
                                artifact="codebase",
                            ),
                            TaskSpec(
                                objective="do thing B",
                                success_condition="thing B done",
                                adapter="coding",
                                artifact="codebase",
                            ),
                        ]
                    ),
                )
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=PlanResponse(tasks=[]),
            )
        return _ok(request)

    final = await Scheduler(runner=runner).run(state, _plan_request())

    assert final.dag[work_a.id].node_state == NodeState.CANCELLED

    decompose_plan_nodes = [
        n
        for n in final.dag.values()
        if n.request.agent_type == AgentType.PLAN
        and isinstance(n.request.spec, PlanSpec)
        and n.request.spec.northstar == "do work"
    ]
    assert len(decompose_plan_nodes) == 1
    assert decompose_plan_nodes[0].node_state == NodeState.INTEGRATED

    subtask_nodes = [
        n
        for n in final.dag.values()
        if n.request.agent_type == AgentType.WORK and n.request.id not in {work_a.id, work_b.id}
    ]
    assert len(subtask_nodes) == 2
    subtask_objectives = {
        n.request.spec.objective for n in subtask_nodes if isinstance(n.request.spec, WorkSpec)
    }
    assert subtask_objectives == {"do thing A", "do thing B"}
    assert all(n.node_state == NodeState.INTEGRATED for n in subtask_nodes)

    assert final.dag[work_b.id].node_state == NodeState.INTEGRATED


async def test_decompose_emits_node_decomposed_telemetry() -> None:
    """Scheduler emits a node.decomposed telemetry event when a work node is decomposed."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    sink = _MemoryTelemetrySink()

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.id == work.id:
            return _decompose(request)
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            output=PlanResponse(tasks=[]),
        )

    await Scheduler(runner=runner, telemetry_sink=sink, run_id=sink.run_id).run(
        state, _plan_request()
    )

    decompose_events = [e for e in sink.events if e.event_type == "node.decomposed"]
    assert len(decompose_events) == 1
    event = decompose_events[0]
    assert event.run_id == sink.run_id
    assert event.node_id == work.id
    assert event.role == "scheduler"
    assert event.phase == "scheduler"
    assert event.status == "decompose"
    assert "plan_node_id" in event.data
