"""Async DAG scheduler that dispatches agent requests up to max_concurrency."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from uuid import UUID, uuid4

from forge.agents.integrator import integrate
from forge.agents.plan_follow_up import PlanFollowUpBuilder
from forge.core.models import (
    AgentContract,
    AgentRequest,
    AgentResponse,
    AgentType,
    DAGNode,
    DeltaState,
    FailureKind,
    NodeState,
    PlanResponse,
    PlanSpec,
    RequestId,
    RequestSource,
    ResponseStatus,
    SchedulerState,
    WorkSpec,
)
from forge.core.state_service import StateService
from forge.core.telemetry import TelemetryEvent, TelemetrySink, safe_append_telemetry

AgentRunner = Callable[[AgentRequest], Awaitable[AgentResponse]]

logger = logging.getLogger(__name__)


@dataclass
class SchedulerCallbacks:
    """Optional lifecycle callbacks fired by the scheduler at key state transitions."""

    on_node_dispatched: Callable[[DAGNode], None] | None = None
    on_node_completed: Callable[[DAGNode], None] | None = None
    on_node_failed: Callable[[DAGNode], None] | None = None
    on_idle: Callable[[SchedulerState], None] | None = None


class Scheduler:
    """Async scheduler that drives a DAG of agent requests to completion."""

    def __init__(
        self,
        runner: AgentRunner,
        state_services: dict[str, StateService] | None = None,
        callbacks: SchedulerCallbacks | None = None,
        telemetry_sink: TelemetrySink | None = None,
        run_id: UUID | None = None,
    ) -> None:
        self._runner = runner
        self._state_services = state_services
        self._callbacks = callbacks or SchedulerCallbacks()
        self._stale_retry_counts: dict[RequestId, int] = {}
        self._integration_retry_counts: dict[RequestId, int] = {}
        self._telemetry_sink = telemetry_sink
        self._run_id = run_id or getattr(telemetry_sink, "run_id", None)

    def _emit_node_decomposed(self, node: DAGNode, plan_request: AgentRequest) -> None:
        if self._run_id is None:
            return
        safe_append_telemetry(
            self._telemetry_sink,
            TelemetryEvent(
                run_id=self._run_id,
                node_id=node.request.id,
                request_id=node.request.id,
                agent_type=node.request.agent_type.value,
                role="scheduler",
                phase="scheduler",
                event_type="node.decomposed",
                status="decompose",
                summary="work node decomposed into plan node",
                data={"plan_node_id": str(plan_request.id)},
            ),
        )

    def _emit_node_failed(self, node: DAGNode) -> None:
        if self._run_id is None:
            return
        response = node.response
        data: dict[str, object] = {}
        status: str | None = None
        summary: str | None = None
        if response is not None:
            status = response.status.value
            summary = response.error
            data = {
                "status": response.status.value,
                "failure_kind": response.failure_kind.value if response.failure_kind else None,
                "error": response.error,
            }
        safe_append_telemetry(
            self._telemetry_sink,
            TelemetryEvent(
                run_id=self._run_id,
                node_id=node.request.id,
                request_id=node.request.id,
                agent_type=node.request.agent_type.value,
                role="scheduler",
                phase="scheduler",
                event_type="node.failed",
                status=status,
                summary=summary,
                data=data,
            ),
        )

    async def run(
        self,
        state: SchedulerState,
        global_planner: AgentRequest,
    ) -> SchedulerState:
        """Drive state forward until no PENDING or RUNNING nodes remain."""
        if not state.dag:
            state = state.add_nodes([DAGNode(request=global_planner)])

        while True:
            ready = state.ready_nodes()

            if not ready:
                self._fire_state(self._callbacks.on_idle, state)
                if not any(
                    n.node_state in (NodeState.PENDING, NodeState.RUNNING)
                    for n in state.dag.values()
                ):
                    break
                new_planner = global_planner.model_copy(
                    update={"id": uuid4(), "source": RequestSource.PLANNER}
                )
                state = state.add_nodes([DAGNode(request=new_planner)])
                continue

            to_dispatch = ready[: state.max_concurrency]

            for node in to_dispatch:
                running = node.with_state(NodeState.RUNNING)
                state = state.update_node(running)
                self._fire_node(self._callbacks.on_node_dispatched, running)

            coros = [self._runner(node.request) for node in to_dispatch]
            raw = await asyncio.gather(*coros, return_exceptions=True)

            for node, result in zip(to_dispatch, raw):
                current = state.dag[node.request.id]

                if isinstance(result, BaseException):
                    failed = current.with_state(NodeState.FAILED)
                    state = state.update_node(failed)
                    self._emit_node_failed(failed)
                    self._fire_node(self._callbacks.on_node_failed, failed)
                    state = self._cancel_dependents(state, node.request.id)
                else:
                    response: AgentResponse = result
                    if (
                        node.request.agent_type == AgentType.PLAN
                        and response.status == ResponseStatus.COMPLETED
                        and isinstance(response.output, PlanResponse)
                    ):
                        response = response.model_copy(
                            update={
                                "follow_up": PlanFollowUpBuilder(node.request).build(
                                    response.output
                                )
                            }
                        )
                    updated = current.with_response(response)
                    state = state.update_node(updated)

                    if updated.node_state == NodeState.INTEGRATED:
                        integration_failed = False
                        stale_retry = False
                        integration_requeued = False
                        if (
                            node.request.agent_type == AgentType.WORK
                            and self._state_services is not None
                            and response.status != ResponseStatus.ALREADY_DONE
                        ):
                            spec = node.request.spec
                            if isinstance(spec, WorkSpec):
                                ss = self._state_services.get(spec.artifact)
                                if ss is not None:
                                    delta = (
                                        response.output
                                        if isinstance(response.output, DeltaState)
                                        else None
                                    )
                                    if delta is None or (
                                        not delta.new_files
                                        and not delta.edits
                                        and not delta.dependencies
                                    ):
                                        updated = current.with_response(
                                            AgentResponse(
                                                request_id=node.request.id,
                                                status=ResponseStatus.FAILED,
                                                error="completed with empty delta — no files, edits, or dependencies produced",
                                            )
                                        )
                                        integration_failed = True
                                    else:
                                        integration_response = await integrate(
                                            request_id=node.request.id,
                                            state_service=ss,
                                            delta=delta,
                                        )
                                        if integration_response.status == ResponseStatus.FAILED:
                                            if (
                                                integration_response.failure_kind
                                                == FailureKind.STALE_DELTA
                                            ):
                                                retry_count = self._stale_retry_counts.get(
                                                    node.request.id, 0
                                                )
                                                if retry_count < 3:
                                                    self._stale_retry_counts[node.request.id] = (
                                                        retry_count + 1
                                                    )
                                                    pending_node = current.with_state(
                                                        NodeState.PENDING
                                                    )
                                                    state = state.update_node(pending_node)
                                                    stale_retry = True
                                                else:
                                                    updated = current.with_response(
                                                        integration_response
                                                    )
                                                    state = state.update_node(updated)
                                                    integration_failed = True
                                            else:
                                                int_retry = self._integration_retry_counts.get(
                                                    node.request.id, 0
                                                )
                                                if (
                                                    integration_response.revision is not None
                                                    and int_retry < 3
                                                ):
                                                    self._integration_retry_counts[
                                                        node.request.id
                                                    ] = int_retry + 1
                                                    revised_request = node.request.model_copy(
                                                        update={
                                                            "integration_revision": integration_response.revision
                                                        }
                                                    )
                                                    pending_node = current.model_copy(
                                                        update={
                                                            "node_state": NodeState.PENDING,
                                                            "request": revised_request,
                                                            "integration_revision": integration_response.revision,
                                                        }
                                                    )
                                                    state = state.update_node(pending_node)
                                                    integration_requeued = True
                                                else:
                                                    updated = current.with_response(
                                                        integration_response
                                                    )
                                                    state = state.update_node(updated)
                                                    integration_failed = True

                        if integration_failed:
                            failed_node = updated.with_state(NodeState.FAILED)
                            state = state.update_node(failed_node)
                            self._emit_node_failed(failed_node)
                            self._fire_node(self._callbacks.on_node_failed, failed_node)
                            state = self._cancel_dependents(state, node.request.id)
                        elif not stale_retry and not integration_requeued:
                            follow_ups = [DAGNode(request=r) for r in response.follow_up]
                            if follow_ups:
                                state = state.add_nodes(follow_ups)
                            self._fire_node(self._callbacks.on_node_completed, updated)
                    elif response.status == ResponseStatus.DECOMPOSE and isinstance(
                        node.request.spec, WorkSpec
                    ):
                        spec = node.request.spec
                        new_plan_request = AgentRequest(
                            agent_type=AgentType.PLAN,
                            source=RequestSource.USER,
                            spec=PlanSpec(
                                northstar=spec.objective,
                                contract=AgentContract(
                                    objective=spec.contract.objective,
                                    success_condition=spec.contract.success_condition,
                                    acceptance_criteria=spec.contract.acceptance_criteria,
                                    constraints=[
                                        "Each subtask must have exactly one concern",
                                        "Subtasks must be non-overlapping",
                                    ],
                                    non_goals=[
                                        "Do not combine setup, implementation, and testing"
                                        " in a single task"
                                    ],
                                ),
                            ),
                        )
                        state = state.add_nodes([DAGNode(request=new_plan_request)])
                        state = self._transfer_dependents(
                            state, node.request.id, new_plan_request.id
                        )
                        self._emit_node_decomposed(updated, new_plan_request)
                    else:
                        self._emit_node_failed(updated)
                        self._fire_node(self._callbacks.on_node_failed, updated)
                        state = self._cancel_dependents(state, node.request.id)

        return state

    def _transfer_dependents(
        self,
        state: SchedulerState,
        from_id: RequestId,
        to_id: RequestId,
    ) -> SchedulerState:
        """Repoint all PENDING nodes depending on from_id to depend on to_id instead."""
        for node in list(state.dag.values()):
            if node.node_state == NodeState.PENDING and from_id in node.request.dependencies:
                new_deps = (node.request.dependencies - {from_id}) | {to_id}
                updated_request = node.request.model_copy(
                    update={"dependencies": frozenset(new_deps)}
                )
                state = state.update_node(node.model_copy(update={"request": updated_request}))
        return state

    def _cancel_dependents(self, state: SchedulerState, failed_id: RequestId) -> SchedulerState:
        failed_ids: set[RequestId] = {failed_id}
        while True:
            to_cancel = [
                node
                for nid, node in state.dag.items()
                if node.node_state == NodeState.PENDING
                and node.request.dependencies & failed_ids
                and nid not in failed_ids
            ]
            if not to_cancel:
                break
            for node in to_cancel:
                state = state.update_node(node.with_state(NodeState.CANCELLED))
                failed_ids.add(node.request.id)
        return state

    def _fire_node(self, callback: Callable[[DAGNode], None] | None, node: DAGNode) -> None:
        if callback is None:
            return
        try:
            callback(node)
        except Exception:
            logger.exception("Scheduler callback raised an exception")

    def _fire_state(
        self, callback: Callable[[SchedulerState], None] | None, state: SchedulerState
    ) -> None:
        if callback is None:
            return
        try:
            callback(state)
        except Exception:
            logger.exception("Scheduler callback raised an exception")
