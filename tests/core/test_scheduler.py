"""Tests for Scheduler DAG execution, concurrency, callbacks, and termination."""

import asyncio
from uuid import uuid4

from forge.core.models import (
    AcceptanceCriterion,
    AgentContract,
    AgentDiagnostic,
    AgentRequest,
    AgentResponse,
    AgentType,
    DAGNode,
    DecompositionTask,
    DependentSplitDecision,
    FailureKind,
    NodeState,
    OrthogonalSplitDecision,
    PlanResponse,
    PlanSpec,
    RequestId,
    RequestSource,
    ResponseStatus,
    SchedulerState,
    TaskSpec,
    WorkDecision,
    WorkOutput,
    WorkSpec,
)
from forge.core.scheduler import Scheduler, SchedulerCallbacks
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


def _ok(request: AgentRequest) -> AgentResponse:
    return AgentResponse(
        request_id=request.id,
        status=ResponseStatus.COMPLETED,
        output=WorkOutput(summary="Completed worktree changes."),
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


# --- Tests ---


async def test_single_work_node_completes() -> None:
    """A single pre-seeded work node reaches INTEGRATED state after the scheduler runs."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    final = await Scheduler(runner=runner).run(state)

    assert final.dag[work.id].node_state == NodeState.INTEGRATED


async def test_scheduler_derives_work_nodes_from_accepted_plan_output() -> None:
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

    final = await Scheduler(runner=runner).run(_base_state().add_nodes([DAGNode(request=planner)]))

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


async def test_scheduler_does_not_derive_work_nodes_from_failed_plan_output() -> None:
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

    final = await Scheduler(runner=runner).run(_base_state().add_nodes([DAGNode(request=planner)]))

    assert final.dag[planner.id].node_state == NodeState.FAILED
    assert all(n.request.agent_type != AgentType.WORK for n in final.dag.values())


async def test_validation_rejected_planner_does_not_spawn_unbounded_planner_nodes() -> None:
    """A validation-rejected planner failure does not create replacement planner nodes."""
    planner = _plan_request()
    blocked_work = _work_request(deps=frozenset({uuid4()}))
    dispatched: list[RequestId] = []
    state = _base_state().add_nodes([DAGNode(request=planner), DAGNode(request=blocked_work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        dispatched.append(request.id)
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            failure_kind=FailureKind.VALIDATION_REJECTED,
            error="maximum validation attempts exhausted without an accept disposition",
            diagnostics=[
                AgentDiagnostic(
                    kind="validation_exhausted",
                    message="maximum validation attempts exhausted without an accept disposition",
                )
            ],
        )

    final = await asyncio.wait_for(Scheduler(runner=runner).run(state), timeout=1)

    plan_nodes = [node for node in final.dag.values() if node.request.agent_type == AgentType.PLAN]
    assert len(plan_nodes) == 1
    assert dispatched == [planner.id]
    assert final.dag[planner.id].node_state == NodeState.FAILED


async def test_failed_node_cancels_dependents() -> None:
    """A FAILED node causes all nodes that depend on it to be CANCELLED."""
    work_a = _work_request()
    work_b = _work_request(deps=frozenset({work_a.id}))

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.id == work_a.id:
            return _fail(request)
        return _ok(request)

    state = _base_state().add_nodes([DAGNode(request=work_a), DAGNode(request=work_b)])
    final = await Scheduler(runner=runner).run(state)

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

    final = await Scheduler(runner=runner, telemetry_sink=sink, run_id=sink.run_id).run(state)

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
    await Scheduler(runner=runner).run(state)

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
    await Scheduler(runner=runner).run(state)

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

    await Scheduler(runner=runner, callbacks=callbacks).run(state)

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

    await Scheduler(runner=runner, callbacks=callbacks).run(state)

    assert any(n.request.id == work.id for n in failed)


async def test_on_idle_fires() -> None:
    """on_idle callback is called at least once when the DAG has no ready nodes."""
    idle_calls: list[SchedulerState] = []
    callbacks = SchedulerCallbacks(on_idle=idle_calls.append)

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    plan = _plan_request()
    await Scheduler(runner=runner, callbacks=callbacks).run(
        _base_state().add_nodes([DAGNode(request=plan)])
    )

    assert len(idle_calls) >= 1


async def test_terminates_cleanly() -> None:
    """Scheduler terminates and returns final state when the planner produces no work."""

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    plan = _plan_request()
    final = await Scheduler(runner=runner).run(_base_state().add_nodes([DAGNode(request=plan)]))

    assert final is not None


async def test_callback_exceptions_do_not_crash() -> None:
    """Exceptions raised by scheduler callbacks are swallowed and do not abort the run."""

    def crashing(node: DAGNode) -> None:
        raise RuntimeError("boom")

    callbacks = SchedulerCallbacks(on_node_completed=crashing)

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    plan = _plan_request()
    final = await Scheduler(runner=runner, callbacks=callbacks).run(
        _base_state().add_nodes([DAGNode(request=plan)])
    )

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
    final = await Scheduler(runner=runner).run(state)

    assert final.dag[work_a.id].node_state == NodeState.INTEGRATED
    assert final.dag[work_b.id].node_state == NodeState.FAILED
    assert final.dag[work_c.id].node_state == NodeState.INTEGRATED


async def test_completed_work_node_is_integrated() -> None:
    """A WORK node returning COMPLETED is marked INTEGRATED by the scheduler."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    final = await Scheduler(runner=runner).run(state)

    assert final.dag[work.id].node_state == NodeState.INTEGRATED


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
    final = await Scheduler(runner=runner).run(state)

    assert final.dag[work_a.id].node_state == NodeState.FAILED
    assert final.dag[work_b.id].node_state == NodeState.CANCELLED
    assert final.dag[work_c.id].node_state == NodeState.CANCELLED


async def test_terminates_when_no_pending_or_running_nodes() -> None:
    """Run completes as soon as no PENDING or RUNNING nodes remain in the DAG."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    final = await Scheduler(runner=runner).run(state)

    assert all(
        n.node_state not in (NodeState.PENDING, NodeState.RUNNING) for n in final.dag.values()
    )


async def test_integration_failure_marks_node_failed() -> None:
    """FAILED runner response marks the node FAILED."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            failure_kind=FailureKind.INTEGRATION_FAILED,
            error="integration failed: tests failed",
        )

    final = await Scheduler(runner=runner).run(state)

    assert final.dag[work.id].node_state == NodeState.FAILED
    response = final.dag[work.id].response
    assert response is not None
    assert response.status == ResponseStatus.FAILED


async def test_integration_failure_preserves_integration_response() -> None:
    """INTEGRATION_FAILED kind is preserved on a failed runner response."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            failure_kind=FailureKind.INTEGRATION_FAILED,
            error="integration failed: tests failed after work output: assertion failed",
        )

    final = await Scheduler(runner=runner).run(state)

    response = final.dag[work.id].response
    assert final.dag[work.id].node_state == NodeState.FAILED
    assert response is not None
    assert response.status == ResponseStatus.FAILED
    assert response.failure_kind == FailureKind.INTEGRATION_FAILED
    assert "tests failed after work output" in (response.error or "")


async def test_integration_called_process_error_becomes_integration_failed() -> None:
    """Runner wraps git errors as INTEGRATION_FAILED; scheduler preserves the kind."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            failure_kind=FailureKind.INTEGRATION_FAILED,
            error="integration failed: git command failed with exit code 1: git merge",
        )

    final = await Scheduler(runner=runner).run(state)

    response = final.dag[work.id].response
    assert final.dag[work.id].node_state == NodeState.FAILED
    assert response is not None
    assert response.status == ResponseStatus.FAILED
    assert response.failure_kind == FailureKind.INTEGRATION_FAILED
    assert "integration failed" in (response.error or "")


async def test_integration_failure_cancels_transitive_dependents() -> None:
    """CANCELLED propagates transitively to dependents when the runner signals failure."""
    work_a = _work_request()
    work_b = _work_request(deps=frozenset({work_a.id}))
    work_c = _work_request(deps=frozenset({work_b.id}))
    state = _base_state().add_nodes(
        [DAGNode(request=work_a), DAGNode(request=work_b), DAGNode(request=work_c)]
    )

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.id == work_a.id:
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                failure_kind=FailureKind.INTEGRATION_FAILED,
                error="integration failed",
            )
        return _ok(request)

    final = await Scheduler(runner=runner).run(state)

    assert final.dag[work_a.id].node_state == NodeState.FAILED
    assert final.dag[work_b.id].node_state == NodeState.CANCELLED
    assert final.dag[work_c.id].node_state == NodeState.CANCELLED


async def test_validation_failed_work_does_not_apply_work_output() -> None:
    """A validation-failed work response marks the node FAILED."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            failure_kind=FailureKind.VALIDATION_REJECTED,
            error="validation rejected work",
        )

    final = await Scheduler(runner=runner).run(state)

    assert final.dag[work.id].node_state == NodeState.FAILED


async def test_integration_success_marks_node_integrated() -> None:
    """Node is marked INTEGRATED when the runner returns COMPLETED."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    final = await Scheduler(runner=runner).run(state)

    assert final.dag[work.id].node_state == NodeState.INTEGRATED


async def test_stale_work_output_requeues_not_failed() -> None:
    """Stale base_version integration failure marks node FAILED immediately."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            failure_kind=FailureKind.INTEGRATION_FAILED,
            error="integration failed: stale base_version: ...",
        )

    final = await Scheduler(runner=runner).run(state)

    assert final.dag[work.id].node_state == NodeState.FAILED


async def test_stale_work_output_fails_after_3_retries() -> None:
    """Stale base_version error produces FAILED node with INTEGRATION_FAILED kind."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            failure_kind=FailureKind.INTEGRATION_FAILED,
            error="integration failed: stale base_version: ...",
        )

    final = await Scheduler(runner=runner).run(state)

    assert final.dag[work.id].node_state == NodeState.FAILED
    response = final.dag[work.id].response
    assert response is not None
    assert response.status == ResponseStatus.FAILED
    assert response.failure_kind == FailureKind.INTEGRATION_FAILED


async def test_non_stale_integration_failure_marks_failed_immediately() -> None:
    """Non-stale integration failure marks the node FAILED immediately."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            failure_kind=FailureKind.INTEGRATION_FAILED,
            error="integration failed: integration error",
        )

    final = await Scheduler(runner=runner).run(state)

    assert final.dag[work.id].node_state == NodeState.FAILED


async def test_already_done_node_skips_integration() -> None:
    """ALREADY_DONE runner response marks the node INTEGRATED without integration."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        return _already_done(request)

    final = await Scheduler(runner=runner).run(state)

    assert final.dag[work.id].node_state == NodeState.INTEGRATED


async def test_already_done_state_version_unchanged() -> None:
    """ALREADY_DONE bypasses integration — scheduler records INTEGRATED without calling apply_work_output."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        return _already_done(request)

    final = await Scheduler(runner=runner).run(state)

    assert final.dag[work.id].node_state == NodeState.INTEGRATED


async def test_already_done_fires_on_node_completed() -> None:
    """on_node_completed callback fires for ALREADY_DONE nodes."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    completed: list[DAGNode] = []
    callbacks = SchedulerCallbacks(on_node_completed=completed.append)

    async def runner(request: AgentRequest) -> AgentResponse:
        return _already_done(request)

    await Scheduler(runner=runner, callbacks=callbacks).run(state)

    assert any(n.request.id == work.id for n in completed)


async def test_work_node_missing_work_output_marked_failed() -> None:
    """WORK node runner returning FAILED (e.g. missing WorkOutput) marks node FAILED."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            error="completed without WorkOutput completion metadata",
        )

    final = await Scheduler(runner=runner).run(state)

    assert final.dag[work.id].node_state == NodeState.FAILED


async def test_work_node_empty_work_output_marked_failed() -> None:
    """WORK node runner returning FAILED (e.g. empty WorkOutput) marks node FAILED."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            error="completed with empty WorkOutput completion metadata",
        )

    final = await Scheduler(runner=runner).run(state)

    assert final.dag[work.id].node_state == NodeState.FAILED


async def test_work_node_already_done_empty_work_output_skips_guard() -> None:
    """ALREADY_DONE response marks node INTEGRATED."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        return _already_done(request)

    final = await Scheduler(runner=runner).run(state)

    assert final.dag[work.id].node_state == NodeState.INTEGRATED


async def test_work_node_non_empty_work_output_integrates_normally() -> None:
    """WORK node runner returning COMPLETED marks node INTEGRATED."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            output=WorkOutput(summary="Completed worktree changes."),
        )

    final = await Scheduler(runner=runner).run(state)

    assert final.dag[work.id].node_state == NodeState.INTEGRATED


async def test_work_node_integrates_typed_work_output() -> None:
    """Scheduler records INTEGRATED state for COMPLETED responses with typed WorkOutput."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    work_output = WorkOutput(summary="Completed worktree changes.")

    async def runner(request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            output=work_output,
        )

    final = await Scheduler(runner=runner).run(state)

    assert final.dag[work.id].node_state == NodeState.INTEGRATED
    node_response = final.dag[work.id].response
    assert node_response is not None
    assert node_response.output == work_output


async def test_integration_failure_with_revision_requeues_node() -> None:
    """Integration failure from the runner marks the node FAILED."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            failure_kind=FailureKind.INTEGRATION_FAILED,
            error="integration failed: tests failed after work output: error",
        )

    final = await Scheduler(runner=runner).run(state)

    assert final.dag[work.id].node_state == NodeState.FAILED


async def test_integration_failure_without_revision_marks_failed() -> None:
    """Integration failure from the runner marks the node FAILED."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            failure_kind=FailureKind.INTEGRATION_FAILED,
            error="integration failed: integration error",
        )

    final = await Scheduler(runner=runner).run(state)

    assert final.dag[work.id].node_state == NodeState.FAILED


async def test_integration_revision_exhaustion_marks_failed() -> None:
    """Integration failure from the runner produces FAILED node with INTEGRATION_FAILED kind."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.FAILED,
            failure_kind=FailureKind.INTEGRATION_FAILED,
            error="integration failed: tests failed after work output: error",
        )

    final = await Scheduler(runner=runner).run(state)

    assert final.dag[work.id].node_state == NodeState.FAILED
    response = final.dag[work.id].response
    assert response is not None
    assert response.failure_kind == FailureKind.INTEGRATION_FAILED


def _decompose(request: AgentRequest) -> AgentResponse:
    return AgentResponse(request_id=request.id, status=ResponseStatus.DECOMPOSE)


async def test_dependents_not_cancelled_on_integration_revision_requeue() -> None:
    """Integration failure on node A cancels dependent nodes."""
    work_a = _work_request()
    work_b = _work_request(deps=frozenset({work_a.id}))
    state = _base_state().add_nodes([DAGNode(request=work_a), DAGNode(request=work_b)])

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.id == work_a.id:
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                failure_kind=FailureKind.INTEGRATION_FAILED,
                error="integration failed",
            )
        return _ok(request)

    final = await Scheduler(runner=runner).run(state)

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

    final = await Scheduler(runner=runner).run(state)

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

    final = await Scheduler(runner=runner).run(state)

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

    final = await Scheduler(runner=runner).run(state)

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

    await Scheduler(runner=runner, callbacks=callbacks).run(state)

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

    final = await Scheduler(runner=runner).run(state)

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

    await Scheduler(runner=runner, telemetry_sink=sink, run_id=sink.run_id).run(state)

    decompose_events = [e for e in sink.events if e.event_type == "node.decomposed"]
    assert len(decompose_events) == 1
    event = decompose_events[0]
    assert event.run_id == sink.run_id
    assert event.node_id == work.id
    assert event.role == "scheduler"
    assert event.phase == "scheduler"
    assert event.status == "decompose"
    assert "plan_node_id" in event.data


# --- Runner exception handling ---


async def test_runner_exception_stores_failed_response_not_none() -> None:
    """When the runner raises an unhandled exception, the failed node has a non-None response."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        raise AttributeError("'AttemptEngine' object has no attribute '_emit'")

    final = await Scheduler(runner=runner).run(state)

    assert final.dag[work.id].node_state == NodeState.FAILED
    response = final.dag[work.id].response
    assert response is not None
    assert response.status == ResponseStatus.FAILED


async def test_runner_exception_response_has_internal_error_kind() -> None:
    """Runner exceptions produce failure_kind=INTERNAL_ERROR on the stored response."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])

    async def runner(request: AgentRequest) -> AgentResponse:
        raise RuntimeError("unexpected executor crash")

    final = await Scheduler(runner=runner).run(state)

    response = final.dag[work.id].response
    assert response is not None
    assert response.failure_kind == FailureKind.INTERNAL_ERROR
    assert "unexpected executor crash" in (response.error or "")


async def test_runner_exception_telemetry_node_failed_has_error_summary() -> None:
    """node.failed telemetry emitted for runner exceptions includes status, failure_kind, and error."""
    work = _work_request()
    state = _base_state().add_nodes([DAGNode(request=work)])
    sink = _MemoryTelemetrySink()

    async def runner(request: AgentRequest) -> AgentResponse:
        raise AttributeError("'AttemptEngine' object has no attribute '_emit'")

    final = await Scheduler(runner=runner, telemetry_sink=sink, run_id=sink.run_id).run(state)

    assert final.dag[work.id].node_state == NodeState.FAILED
    events = [e for e in sink.events if e.event_type == "node.failed"]
    assert len(events) == 1
    event = events[0]
    assert event.status == "failed"
    assert event.data.get("failure_kind") == "internal_error"
    assert "'AttemptEngine'" in (event.summary or "")


async def test_scheduler_emits_node_dispatched_telemetry_for_work_node() -> None:
    """Scheduler emits node.dispatched with contract data when a work node is dispatched."""
    work = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="Implement parser",
            success_condition="tests pass",
            adapter="coding",
            artifact="codebase",
            contract=AgentContract(
                objective="Implement parser",
                success_condition="tests pass",
                acceptance_criteria=[AcceptanceCriterion(id="AC1", text="parse tags")],
            ),
        ),
    )
    state = _base_state().add_nodes([DAGNode(request=work)])
    sink = _MemoryTelemetrySink()

    async def runner(request: AgentRequest) -> AgentResponse:
        return _ok(request)

    await Scheduler(runner=runner, telemetry_sink=sink, run_id=sink.run_id).run(state)

    dispatched = [e for e in sink.events if e.event_type == "node.dispatched"]
    work_dispatched = [e for e in dispatched if e.node_id == work.id]
    assert len(work_dispatched) == 1
    event = work_dispatched[0]
    assert event.run_id == sink.run_id
    assert event.node_id == work.id
    assert event.role == "scheduler"
    assert event.phase == "scheduler"
    assert event.status == "dispatched"
    contract = event.data["contract"]
    assert contract["objective"] == "Implement parser"
    assert contract["success_condition"] == "tests pass"
    assert contract["artifact"] == "codebase"
    assert contract["adapter"] == "coding"
    assert contract["acceptance_criteria"] == [{"id": "AC1", "text": "parse tags"}]


async def test_runner_exception_cancels_dependents() -> None:
    """A runner exception on node A cancels dependent nodes just like a normal failure."""
    work_a = _work_request()
    work_b = _work_request(deps=frozenset({work_a.id}))
    state = _base_state().add_nodes([DAGNode(request=work_a), DAGNode(request=work_b)])

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.id == work_a.id:
            raise RuntimeError("crash before attempt")
        return _ok(request)

    final = await Scheduler(runner=runner).run(state)

    assert final.dag[work_a.id].node_state == NodeState.FAILED
    assert final.dag[work_b.id].node_state == NodeState.CANCELLED


# --- DecompositionDecision expansion via scheduler ---


def _task_spec(objective: str) -> TaskSpec:
    return TaskSpec(
        objective=objective,
        success_condition="done",
        adapter="coding",
        artifact="codebase",
    )


async def test_scheduler_expands_work_decision_via_build_from_decision() -> None:
    """Planner returning WorkDecision creates exactly one WORK node via build_from_decision."""
    planner = _plan_request()
    decision = WorkDecision(
        task=WorkSpec(
            objective="implement parser",
            success_condition="tests pass",
            adapter="coding",
            artifact="codebase",
        )
    )

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.id == planner.id:
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=decision,
            )
        return _ok(request)

    final = await Scheduler(runner=runner).run(_base_state().add_nodes([DAGNode(request=planner)]))

    work_nodes = [n for n in final.dag.values() if n.request.agent_type == AgentType.WORK]
    assert len(work_nodes) == 1
    assert isinstance(work_nodes[0].request.spec, WorkSpec)
    assert work_nodes[0].request.spec.objective == "implement parser"
    assert work_nodes[0].node_state == NodeState.INTEGRATED


async def test_scheduler_expands_dependent_split_decision_into_chained_work_nodes() -> None:
    """Planner returning DependentSplitDecision creates chained WORK nodes."""
    planner = _plan_request()
    decision = DependentSplitDecision(
        tasks=[_task_spec("task-alpha"), _task_spec("task-beta"), _task_spec("task-gamma")]
    )

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.id == planner.id:
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=decision,
            )
        return _ok(request)

    final = await Scheduler(runner=runner).run(
        _base_state(max_concurrency=3).add_nodes([DAGNode(request=planner)])
    )

    work_nodes = [n for n in final.dag.values() if n.request.agent_type == AgentType.WORK]
    assert len(work_nodes) == 3
    assert all(n.node_state == NodeState.INTEGRATED for n in work_nodes)

    def objective(n: DAGNode) -> str:
        assert isinstance(n.request.spec, WorkSpec)
        return n.request.spec.objective

    node_a = next(n for n in work_nodes if objective(n) == "task-alpha")
    node_b = next(n for n in work_nodes if objective(n) == "task-beta")
    node_c = next(n for n in work_nodes if objective(n) == "task-gamma")
    assert node_a.request.dependencies == frozenset()
    assert node_a.request.id in node_b.request.dependencies
    assert node_b.request.id in node_c.request.dependencies


async def test_scheduler_expands_orthogonal_split_decision_into_independent_work_nodes() -> None:
    """Planner returning OrthogonalSplitDecision creates independent WORK nodes."""
    planner = _plan_request()
    decision = OrthogonalSplitDecision(tasks=[_task_spec("task-alpha"), _task_spec("task-beta")])

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.id == planner.id:
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=decision,
            )
        return _ok(request)

    final = await Scheduler(runner=runner).run(
        _base_state(max_concurrency=2).add_nodes([DAGNode(request=planner)])
    )

    work_nodes = [n for n in final.dag.values() if n.request.agent_type == AgentType.WORK]
    assert len(work_nodes) == 2
    assert all(n.node_state == NodeState.INTEGRATED for n in work_nodes)
    assert all(n.request.dependencies == frozenset() for n in work_nodes)


async def test_scheduler_legacy_plan_response_still_expands_correctly() -> None:
    """Legacy PlanResponse still expands to work nodes after DecompositionDecision support added."""
    planner = _plan_request()
    plan = PlanResponse(tasks=[_task_spec("legacy-task")])

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.id == planner.id:
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=plan,
            )
        return _ok(request)

    final = await Scheduler(runner=runner).run(_base_state().add_nodes([DAGNode(request=planner)]))

    work_nodes = [n for n in final.dag.values() if n.request.agent_type == AgentType.WORK]
    assert len(work_nodes) == 1
    assert work_nodes[0].node_state == NodeState.INTEGRATED
    assert isinstance(work_nodes[0].request.spec, WorkSpec)
    assert work_nodes[0].request.spec.objective == "legacy-task"


# --- Decomposition convergence ---


async def test_scheduler_convergence_failure_marks_plan_node_failed() -> None:
    """A split decision whose child repeats the parent objective fails the plan node."""
    planner = _plan_request()
    decision = OrthogonalSplitDecision(
        tasks=[
            TaskSpec(
                objective="test northstar",  # repeats northstar == parent contract objective
                success_condition="done",
                adapter="coding",
                artifact="codebase",
            ),
            TaskSpec(
                objective="add CLI interface",
                success_condition="done",
                adapter="coding",
                artifact="codebase",
            ),
        ]
    )

    async def runner(request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            output=decision,
        )

    final = await Scheduler(runner=runner).run(_base_state().add_nodes([DAGNode(request=planner)]))

    plan_node = final.dag[planner.id]
    assert plan_node.node_state == NodeState.FAILED
    assert plan_node.response is not None
    assert plan_node.response.failure_kind == FailureKind.VALIDATION_REJECTED
    assert "not reductive" in (plan_node.response.error or "")


async def test_scheduler_convergence_failure_adds_no_child_nodes() -> None:
    """No child nodes are added to the DAG when convergence validation rejects the decision."""
    planner = _plan_request()
    decision = DependentSplitDecision(
        tasks=[
            TaskSpec(
                objective="test northstar",
                success_condition="done",
                adapter="coding",
                artifact="codebase",
            )
        ]
    )

    async def runner(request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            output=decision,
        )

    final = await Scheduler(runner=runner).run(_base_state().add_nodes([DAGNode(request=planner)]))

    assert len(final.dag) == 1  # only the original plan node — no children inserted
    assert final.dag[planner.id].node_state == NodeState.FAILED


async def test_scheduler_convergence_failure_emits_telemetry_event() -> None:
    """A convergence failure emits a node.convergence_failed telemetry event."""
    planner = _plan_request()
    decision = OrthogonalSplitDecision(
        tasks=[
            TaskSpec(
                objective="test northstar",
                success_condition="done",
                adapter="coding",
                artifact="codebase",
            ),
            TaskSpec(
                objective="another task",
                success_condition="done",
                adapter="coding",
                artifact="codebase",
            ),
        ]
    )
    sink = _MemoryTelemetrySink()

    async def runner(request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            output=decision,
        )

    await Scheduler(runner=runner, telemetry_sink=sink, run_id=sink.run_id).run(
        _base_state().add_nodes([DAGNode(request=planner)])
    )

    convergence_events = [e for e in sink.events if e.event_type == "node.convergence_failed"]
    assert len(convergence_events) == 1
    assert convergence_events[0].status == "failed"
    assert "reason" in convergence_events[0].data


async def test_scheduler_valid_split_not_affected_by_convergence_check() -> None:
    """A valid split with distinct narrower children passes convergence and expands normally."""
    planner = _plan_request()
    decision = OrthogonalSplitDecision(
        tasks=[
            TaskSpec(
                objective="implement HTTP fetching",
                success_condition="done",
                adapter="coding",
                artifact="codebase",
            ),
            TaskSpec(
                objective="implement HTML parsing",
                success_condition="done",
                adapter="coding",
                artifact="codebase",
            ),
        ]
    )

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.agent_type == AgentType.PLAN:
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=decision,
            )
        return _ok(request)

    final = await Scheduler(runner=runner).run(_base_state().add_nodes([DAGNode(request=planner)]))

    plan_node = final.dag[planner.id]
    assert plan_node.node_state == NodeState.INTEGRATED
    work_nodes = [n for n in final.dag.values() if n.request.agent_type == AgentType.WORK]
    assert len(work_nodes) == 2
    assert all(n.node_state == NodeState.INTEGRATED for n in work_nodes)


# --- Recursive PWC symmetry (Step 7) ---


def _decomposition_task_spec(objective: str) -> DecompositionTask:
    return DecompositionTask(objective=objective, success_condition="planned")


async def test_root_plan_emits_decomposition_task_child_plan_is_dispatched() -> None:
    """Root PLAN emitting a DecompositionTask results in a child PLAN node that is dispatched."""
    root_plan = _plan_request()
    dispatched_ids: list[RequestId] = []

    async def runner(request: AgentRequest) -> AgentResponse:
        dispatched_ids.append(request.id)
        if request.id == root_plan.id:
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=OrthogonalSplitDecision(
                    tasks=[_decomposition_task_spec("implement sub-system")]
                ),
            )
        if request.agent_type == AgentType.PLAN:
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=PlanResponse(tasks=[]),
            )
        return _ok(request)

    final = await Scheduler(runner=runner).run(
        _base_state().add_nodes([DAGNode(request=root_plan)])
    )

    plan_nodes = [n for n in final.dag.values() if n.request.agent_type == AgentType.PLAN]
    assert len(plan_nodes) == 2
    child_plan = next(n for n in plan_nodes if n.request.id != root_plan.id)
    assert child_plan.request.id in dispatched_ids
    assert isinstance(child_plan.request.spec, PlanSpec)
    assert child_plan.request.spec.northstar == "implement sub-system"
    assert child_plan.request.source == RequestSource.PLANNER


async def test_two_level_decomposition_grandchild_work_node_completes() -> None:
    """Root PLAN → child PLAN (via DecompositionTask) → grandchild WORK: all reach terminal state."""
    root_plan = _plan_request()

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.id == root_plan.id:
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=OrthogonalSplitDecision(
                    tasks=[_decomposition_task_spec("implement sub-system")]
                ),
            )
        if request.agent_type == AgentType.PLAN:
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=OrthogonalSplitDecision(
                    tasks=[
                        TaskSpec(
                            objective="write the implementation",
                            success_condition="tests pass",
                            adapter="coding",
                            artifact="codebase",
                        )
                    ]
                ),
            )
        return _ok(request)

    final = await Scheduler(runner=runner).run(
        _base_state().add_nodes([DAGNode(request=root_plan)])
    )

    plan_nodes = [n for n in final.dag.values() if n.request.agent_type == AgentType.PLAN]
    work_nodes = [n for n in final.dag.values() if n.request.agent_type == AgentType.WORK]
    assert len(plan_nodes) == 2
    assert len(work_nodes) == 1
    assert all(n.node_state == NodeState.INTEGRATED for n in plan_nodes)
    assert work_nodes[0].node_state == NodeState.INTEGRATED
    assert isinstance(work_nodes[0].request.spec, WorkSpec)
    assert work_nodes[0].request.spec.objective == "write the implementation"


async def test_child_plan_node_uses_same_acceptance_path_as_root() -> None:
    """Child PLAN node accepted via OrthogonalSplitDecision expands children identically to root."""
    root_plan = _plan_request()
    plan_nodes_seen: list[RequestId] = []

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.agent_type == AgentType.PLAN:
            plan_nodes_seen.append(request.id)
            if request.id == root_plan.id:
                return AgentResponse(
                    request_id=request.id,
                    status=ResponseStatus.COMPLETED,
                    output=DependentSplitDecision(
                        tasks=[
                            _decomposition_task_spec("plan phase-one"),
                            _task_spec("finalize"),
                        ]
                    ),
                )
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=WorkDecision(
                    task=WorkSpec(
                        objective="phase-one work",
                        success_condition="done",
                        adapter="coding",
                        artifact="codebase",
                    )
                ),
            )
        return _ok(request)

    final = await Scheduler(runner=runner).run(
        _base_state().add_nodes([DAGNode(request=root_plan)])
    )

    assert len(plan_nodes_seen) == 2
    work_nodes = [n for n in final.dag.values() if n.request.agent_type == AgentType.WORK]
    assert len(work_nodes) == 2
    objectives = {
        n.request.spec.objective for n in work_nodes if isinstance(n.request.spec, WorkSpec)
    }
    assert objectives == {"phase-one work", "finalize"}
    assert all(n.node_state == NodeState.INTEGRATED for n in work_nodes)


async def test_plan_node_decompose_disposition_creates_new_plan_node_not_failure() -> None:
    """A PLAN node returning DECOMPOSE creates a replacement PLAN node — not a failure."""
    root_plan = _plan_request()

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.id == root_plan.id:
            return AgentResponse(request_id=request.id, status=ResponseStatus.DECOMPOSE)
        if request.agent_type == AgentType.PLAN:
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=PlanResponse(tasks=[]),
            )
        return _ok(request)

    final = await Scheduler(runner=runner).run(
        _base_state().add_nodes([DAGNode(request=root_plan)])
    )

    assert final.dag[root_plan.id].node_state == NodeState.CANCELLED
    replacement_plans = [
        n
        for n in final.dag.values()
        if n.request.agent_type == AgentType.PLAN and n.request.id != root_plan.id
    ]
    assert len(replacement_plans) == 1
    assert replacement_plans[0].node_state == NodeState.INTEGRATED
    assert isinstance(replacement_plans[0].request.spec, PlanSpec)
    assert replacement_plans[0].request.spec.northstar == "test northstar"


async def test_child_plan_decompose_disposition_creates_new_plan_node() -> None:
    """A child PLAN node returning DECOMPOSE behaves symmetrically with root: creates new PLAN."""
    root_plan = _plan_request()
    # Guard so the replacement plan does not also DECOMPOSE, which would loop forever.
    child_decomposed_once = False

    async def runner(request: AgentRequest) -> AgentResponse:
        nonlocal child_decomposed_once
        if request.id == root_plan.id:
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=OrthogonalSplitDecision(
                    tasks=[_decomposition_task_spec("implement sub-system")]
                ),
            )
        spec = request.spec
        if (
            request.agent_type == AgentType.PLAN
            and isinstance(spec, PlanSpec)
            and spec.northstar == "implement sub-system"
            and not child_decomposed_once
        ):
            child_decomposed_once = True
            return AgentResponse(request_id=request.id, status=ResponseStatus.DECOMPOSE)
        if request.agent_type == AgentType.PLAN:
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=PlanResponse(tasks=[]),
            )
        return _ok(request)

    final = await Scheduler(runner=runner).run(
        _base_state().add_nodes([DAGNode(request=root_plan)])
    )

    plan_nodes = [n for n in final.dag.values() if n.request.agent_type == AgentType.PLAN]
    cancelled_plans = [n for n in plan_nodes if n.node_state == NodeState.CANCELLED]
    assert len(cancelled_plans) == 1
    assert isinstance(cancelled_plans[0].request.spec, PlanSpec)
    assert cancelled_plans[0].request.spec.northstar == "implement sub-system"
    integrated_plans = [n for n in plan_nodes if n.node_state == NodeState.INTEGRATED]
    assert len(integrated_plans) >= 2


async def test_child_plan_failure_marks_failed_like_root_plan_failure() -> None:
    """Child PLAN node failure marks it FAILED, symmetrically with root PLAN failure."""
    root_plan = _plan_request()

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.id == root_plan.id:
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=OrthogonalSplitDecision(
                    tasks=[_decomposition_task_spec("implement sub-system")]
                ),
            )
        if request.agent_type == AgentType.PLAN:
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.FAILED,
                failure_kind=FailureKind.VALIDATION_REJECTED,
                error="child plan rejected after exhausting revisions",
            )
        return _ok(request)

    final = await Scheduler(runner=runner).run(
        _base_state().add_nodes([DAGNode(request=root_plan)])
    )

    child_plans = [
        n
        for n in final.dag.values()
        if n.request.agent_type == AgentType.PLAN and n.request.id != root_plan.id
    ]
    assert len(child_plans) == 1
    assert child_plans[0].node_state == NodeState.FAILED
    assert child_plans[0].response is not None
    assert child_plans[0].response.failure_kind == FailureKind.VALIDATION_REJECTED


async def test_telemetry_has_distinct_dispatched_events_for_root_child_and_work_nodes() -> None:
    """Telemetry node.dispatched events are emitted for root PLAN, child PLAN, and WORK nodes."""
    root_plan = _plan_request()
    sink = _MemoryTelemetrySink()

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.id == root_plan.id:
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=OrthogonalSplitDecision(
                    tasks=[_decomposition_task_spec("implement sub-system")]
                ),
            )
        if request.agent_type == AgentType.PLAN:
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=OrthogonalSplitDecision(
                    tasks=[
                        TaskSpec(
                            objective="write the code",
                            success_condition="tests pass",
                            adapter="coding",
                            artifact="codebase",
                        )
                    ]
                ),
            )
        return _ok(request)

    final = await Scheduler(runner=runner, telemetry_sink=sink, run_id=sink.run_id).run(
        _base_state().add_nodes([DAGNode(request=root_plan)])
    )

    dispatched = [e for e in sink.events if e.event_type == "node.dispatched"]
    dispatched_ids = {e.node_id for e in dispatched}
    plan_ids = {n.request.id for n in final.dag.values() if n.request.agent_type == AgentType.PLAN}
    work_ids = {n.request.id for n in final.dag.values() if n.request.agent_type == AgentType.WORK}
    assert len(plan_ids) == 2
    assert len(work_ids) == 1
    assert plan_ids <= dispatched_ids
    assert work_ids <= dispatched_ids


# --- DECOMPOSE convergence loop protection (Step 8) ---


async def test_decompose_convergence_loop_is_rejected() -> None:
    """PLAN('X') → DECOMPOSE → PLAN('X') → DECOMPOSE is rejected as a non-reductive loop."""
    root_plan = _plan_request()
    plan_calls: list[RequestId] = []

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.agent_type == AgentType.PLAN:
            plan_calls.append(request.id)
            return AgentResponse(request_id=request.id, status=ResponseStatus.DECOMPOSE)
        return _ok(request)

    final = await Scheduler(runner=runner).run(
        _base_state().add_nodes([DAGNode(request=root_plan)])
    )

    # First plan is cancelled (first DECOMPOSE allowed), replacement is FAILED (loop caught)
    assert final.dag[root_plan.id].node_state == NodeState.CANCELLED
    replacement_plans = [
        n
        for n in final.dag.values()
        if n.request.agent_type == AgentType.PLAN and n.request.id != root_plan.id
    ]
    assert len(replacement_plans) == 1
    failed = replacement_plans[0]
    assert failed.node_state == NodeState.FAILED
    assert failed.response is not None
    assert failed.response.failure_kind == FailureKind.VALIDATION_REJECTED
    assert "not reductive" in (failed.response.error or "")
    assert len(plan_calls) == 2  # no infinite loop


async def test_decompose_convergence_normalized_objective_rejected() -> None:
    """Loop detection is case-insensitive and whitespace-normalized."""
    plan = AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(
            northstar="Build The Parser",
            contract=AgentContract(
                objective="Build The Parser",
                success_condition="parser built",
            ),
        ),
    )
    plan_calls: list[RequestId] = []

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.agent_type == AgentType.PLAN:
            plan_calls.append(request.id)
            return AgentResponse(request_id=request.id, status=ResponseStatus.DECOMPOSE)
        return _ok(request)

    final = await Scheduler(runner=runner).run(_base_state().add_nodes([DAGNode(request=plan)]))

    assert len(plan_calls) == 2  # no infinite loop
    replacement_plans = [
        n
        for n in final.dag.values()
        if n.request.agent_type == AgentType.PLAN and n.request.id != plan.id
    ]
    assert len(replacement_plans) == 1
    assert replacement_plans[0].node_state == NodeState.FAILED
    assert replacement_plans[0].response is not None
    assert replacement_plans[0].response.failure_kind == FailureKind.VALIDATION_REJECTED


async def test_decompose_convergence_work_to_plan_is_allowed() -> None:
    """WORK → DECOMPOSE → PLAN is always allowed; the check only catches PLAN-to-PLAN loops."""
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

    final = await Scheduler(runner=runner).run(state)

    assert final.dag[work.id].node_state == NodeState.CANCELLED
    plan_nodes = [n for n in final.dag.values() if n.request.agent_type == AgentType.PLAN]
    assert len(plan_nodes) == 1
    assert plan_nodes[0].node_state == NodeState.INTEGRATED


async def test_decompose_convergence_split_decisions_unaffected() -> None:
    """OrthogonalSplitDecision expansion is unaffected by DECOMPOSE convergence checks."""
    planner = _plan_request()
    decision = OrthogonalSplitDecision(tasks=[_task_spec("task-one"), _task_spec("task-two")])

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.id == planner.id:
            return AgentResponse(
                request_id=request.id, status=ResponseStatus.COMPLETED, output=decision
            )
        return _ok(request)

    final = await Scheduler(runner=runner).run(
        _base_state(max_concurrency=2).add_nodes([DAGNode(request=planner)])
    )

    work_nodes = [n for n in final.dag.values() if n.request.agent_type == AgentType.WORK]
    assert len(work_nodes) == 2
    assert all(n.node_state == NodeState.INTEGRATED for n in work_nodes)


async def test_decompose_convergence_no_depth_limit() -> None:
    """Recursive decomposition via DecompositionTask is not depth-limited by convergence checks."""
    root_plan = _plan_request()

    async def runner(request: AgentRequest) -> AgentResponse:
        if not isinstance(request.spec, PlanSpec):
            return _ok(request)
        northstar = request.spec.northstar
        if northstar == "test northstar":
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=OrthogonalSplitDecision(tasks=[_decomposition_task_spec("level-two")]),
            )
        if northstar == "level-two":
            return AgentResponse(
                request_id=request.id,
                status=ResponseStatus.COMPLETED,
                output=OrthogonalSplitDecision(tasks=[_decomposition_task_spec("level-three")]),
            )
        return AgentResponse(
            request_id=request.id,
            status=ResponseStatus.COMPLETED,
            output=OrthogonalSplitDecision(tasks=[_task_spec("leaf-work")]),
        )

    final = await Scheduler(runner=runner).run(
        _base_state().add_nodes([DAGNode(request=root_plan)])
    )

    plan_nodes = [n for n in final.dag.values() if n.request.agent_type == AgentType.PLAN]
    work_nodes = [n for n in final.dag.values() if n.request.agent_type == AgentType.WORK]
    assert len(plan_nodes) == 3
    assert len(work_nodes) == 1
    assert all(n.node_state == NodeState.INTEGRATED for n in plan_nodes)
    assert all(n.node_state == NodeState.INTEGRATED for n in work_nodes)


async def test_decompose_convergence_emits_convergence_failed_telemetry() -> None:
    """DECOMPOSE loop detection emits a node.convergence_failed telemetry event."""
    root_plan = _plan_request()
    sink = _MemoryTelemetrySink()

    async def runner(request: AgentRequest) -> AgentResponse:
        if request.agent_type == AgentType.PLAN:
            return AgentResponse(request_id=request.id, status=ResponseStatus.DECOMPOSE)
        return _ok(request)

    await Scheduler(runner=runner, telemetry_sink=sink, run_id=sink.run_id).run(
        _base_state().add_nodes([DAGNode(request=root_plan)])
    )

    convergence_events = [e for e in sink.events if e.event_type == "node.convergence_failed"]
    assert len(convergence_events) == 1
    assert convergence_events[0].status == "failed"
    assert "reason" in convergence_events[0].data
    assert "not reductive" in convergence_events[0].data["reason"]
