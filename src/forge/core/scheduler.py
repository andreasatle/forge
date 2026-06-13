"""Async DAG scheduler that dispatches agent requests up to max_concurrency."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from uuid import UUID

from forge.core.models import (
    AgentContract,
    AgentRequest,
    AgentResponse,
    AgentType,
    DAGNode,
    FailureKind,
    NodeState,
    PlanResponse,
    PlanSpec,
    RequestId,
    RequestSource,
    ResponseStatus,
    SchedulerState,
    WorkOutput,
    WorkSpec,
)
from forge.core.plan_expansion import PlanExpansionBuilder
from forge.core.state_service import StateService
from forge.core.telemetry import TelemetryEvent, TelemetrySink, safe_append_telemetry

AgentRunner = Callable[[AgentRequest], Awaitable[AgentResponse]]

logger = logging.getLogger(__name__)
_VALIDATION_EXHAUSTED_DIAGNOSTIC = "validation_exhausted"


@dataclass
class SchedulerCallbacks:
    """Optional lifecycle callbacks fired by the scheduler at key state transitions."""

    on_node_dispatched: Callable[[DAGNode], None] | None = None
    on_node_completed: Callable[[DAGNode], None] | None = None
    on_node_failed: Callable[[DAGNode], None] | None = None
    on_idle: Callable[[SchedulerState], None] | None = None


class TerminalOutcomeKind(Enum):
    """Structured terminal outcome categories consumed by scheduler consequences."""

    ACCEPTED_PLAN = "accepted_plan"
    ACCEPTED_WORK = "accepted_work"
    VALIDATION_EXHAUSTED = "validation_exhausted"
    INTEGRATION_FAILURE = "integration_failure"
    DECOMPOSITION_REQUEST = "decomposition_request"
    TERMINAL_FAILURE = "terminal_failure"


@dataclass(frozen=True)
class TerminalNodeOutcome:
    """Terminal agent result normalized for SchedulerConsequenceHandler."""

    kind: TerminalOutcomeKind
    response: AgentResponse

    @classmethod
    def from_response(cls, node: DAGNode, response: AgentResponse) -> "TerminalNodeOutcome":
        """Classify an agent response into the scheduler's terminal outcome categories."""
        if response.status == ResponseStatus.DECOMPOSE:
            return cls(TerminalOutcomeKind.DECOMPOSITION_REQUEST, response)
        if _has_validation_exhausted_diagnostic(response):
            return cls(TerminalOutcomeKind.VALIDATION_EXHAUSTED, response)
        if response.status in (ResponseStatus.COMPLETED, ResponseStatus.ALREADY_DONE):
            if node.request.agent_type == AgentType.PLAN:
                return cls(TerminalOutcomeKind.ACCEPTED_PLAN, response)
            if node.request.agent_type == AgentType.WORK:
                return cls(TerminalOutcomeKind.ACCEPTED_WORK, response)
        return cls(TerminalOutcomeKind.TERMINAL_FAILURE, response)

    @classmethod
    def from_exception(cls, node: DAGNode, error: BaseException) -> "TerminalNodeOutcome":
        """Represent a runner exception as a terminal scheduler failure outcome."""
        return cls(
            TerminalOutcomeKind.TERMINAL_FAILURE,
            AgentResponse(
                request_id=node.request.id,
                status=ResponseStatus.FAILED,
                failure_kind=FailureKind.INTERNAL_ERROR,
                error=str(error),
            ),
        )


def _has_validation_exhausted_diagnostic(response: AgentResponse) -> bool:
    return any(
        diagnostic.kind == _VALIDATION_EXHAUSTED_DIAGNOSTIC for diagnostic in response.diagnostics
    )


class SchedulerConsequenceHandler:
    """Owns consequences of terminal node responses."""

    def __init__(
        self,
        state_services: dict[str, StateService] | None = None,
        callbacks: SchedulerCallbacks | None = None,
        telemetry_sink: TelemetrySink | None = None,
        run_id: UUID | None = None,
    ) -> None:
        self._state_services = state_services
        self._callbacks = callbacks or SchedulerCallbacks()
        self._telemetry_sink = telemetry_sink
        self._run_id = run_id or getattr(telemetry_sink, "run_id", None)

    async def apply(
        self,
        state: SchedulerState,
        node: DAGNode,
        outcome: TerminalNodeOutcome,
    ) -> SchedulerState:
        """Apply the effects of a terminal response for a DAG node."""
        response = outcome.response
        current = state.dag[node.request.id]
        updated = current.with_response(response)
        state = state.update_node(updated)

        if outcome.kind == TerminalOutcomeKind.ACCEPTED_PLAN:
            plan_expansion = self._build_plan_expansion(node, response)
            return self._handle_accepted_plan(state, updated, plan_expansion)
        if outcome.kind == TerminalOutcomeKind.ACCEPTED_WORK:
            return await self._handle_accepted_work(state, current, updated)
        if outcome.kind == TerminalOutcomeKind.DECOMPOSITION_REQUEST:
            return self._handle_decompose(state, updated)

        return self._handle_failed(state, updated, outcome.kind)

    def _build_plan_expansion(
        self,
        node: DAGNode,
        response: AgentResponse,
    ) -> list[DAGNode]:
        if (
            node.request.agent_type == AgentType.PLAN
            and response.status == ResponseStatus.COMPLETED
            and isinstance(response.output, PlanResponse)
        ):
            return [
                DAGNode(request=request)
                for request in PlanExpansionBuilder(node.request).build(response.output)
            ]
        return []

    def _handle_accepted_plan(
        self,
        state: SchedulerState,
        updated: DAGNode,
        plan_expansion: list[DAGNode],
    ) -> SchedulerState:
        if plan_expansion:
            state = state.add_nodes(plan_expansion)
        self._fire_node(self._callbacks.on_node_completed, updated)
        return state

    async def _handle_accepted_work(
        self,
        state: SchedulerState,
        current: DAGNode,
        updated: DAGNode,
    ) -> SchedulerState:
        integration_failed = False
        response = updated.response
        if response is None:
            return self._handle_failed(state, updated)

        if (
            updated.request.agent_type == AgentType.WORK
            and self._state_services is not None
            and response.status != ResponseStatus.ALREADY_DONE
        ):
            updated, integration_failed = await self._integrate_work_node(current, updated)

        if integration_failed:
            failed_node = updated.with_state(NodeState.FAILED)
            state = state.update_node(failed_node)
            return self._handle_failed(state, failed_node, TerminalOutcomeKind.INTEGRATION_FAILURE)

        self._fire_node(self._callbacks.on_node_completed, updated)
        return state

    async def _integrate_work_node(
        self,
        current: DAGNode,
        updated: DAGNode,
    ) -> tuple[DAGNode, bool]:
        response = updated.response
        if response is None:
            return updated, False

        spec = updated.request.spec
        if not isinstance(spec, WorkSpec):
            return updated, False

        ss = self._state_services.get(spec.artifact) if self._state_services is not None else None
        if ss is None:
            return updated, False

        work_output = response.output if isinstance(response.output, WorkOutput) else None
        if work_output is None or (not work_output.files and not work_output.dependencies):
            return (
                current.with_response(
                    AgentResponse(
                        request_id=updated.request.id,
                        status=ResponseStatus.FAILED,
                        error=(
                            "completed with empty work output — no files or dependencies produced"
                        ),
                    )
                ),
                True,
            )

        try:
            await ss.apply_work_output(work_output, str(updated.request.id))
        except RuntimeError as e:
            return (
                current.with_response(
                    AgentResponse(
                        request_id=updated.request.id,
                        status=ResponseStatus.FAILED,
                        failure_kind=FailureKind.INTEGRATION_FAILED,
                        error=f"integration failed: {e}",
                    )
                ),
                True,
            )

        return updated, False

    def _handle_decompose(self, state: SchedulerState, updated: DAGNode) -> SchedulerState:
        spec = updated.request.spec
        if not isinstance(spec, WorkSpec):
            return self._handle_failed(state, updated, TerminalOutcomeKind.TERMINAL_FAILURE)

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
                        "Do not combine setup, implementation, and testing in a single task"
                    ],
                ),
            ),
        )
        state = state.add_nodes([DAGNode(request=new_plan_request)])
        state = self._transfer_dependents(state, updated.request.id, new_plan_request.id)
        self._emit_node_decomposed(updated, new_plan_request)
        return state

    def _handle_failed(
        self,
        state: SchedulerState,
        updated: DAGNode,
        outcome_kind: TerminalOutcomeKind = TerminalOutcomeKind.TERMINAL_FAILURE,
    ) -> SchedulerState:
        _ = outcome_kind
        self._emit_node_failed(updated)
        self._fire_node(self._callbacks.on_node_failed, updated)
        return self._cancel_dependents(state, updated.request.id)

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

    def _fire_node(self, callback: Callable[[DAGNode], None] | None, node: DAGNode) -> None:
        if callback is None:
            return
        try:
            callback(node)
        except Exception:
            logger.exception("Scheduler callback raised an exception")


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
        self._callbacks = callbacks or SchedulerCallbacks()
        self._consequences = SchedulerConsequenceHandler(
            state_services=state_services,
            callbacks=self._callbacks,
            telemetry_sink=telemetry_sink,
            run_id=run_id,
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
                break

            to_dispatch = ready[: state.max_concurrency]

            for node in to_dispatch:
                running = node.with_state(NodeState.RUNNING)
                state = state.update_node(running)
                self._fire_node(self._callbacks.on_node_dispatched, running)

            coros = [self._runner(node.request) for node in to_dispatch]
            raw = await asyncio.gather(*coros, return_exceptions=True)

            for node, result in zip(to_dispatch, raw):
                if isinstance(result, BaseException):
                    outcome = TerminalNodeOutcome.from_exception(node, result)
                else:
                    outcome = TerminalNodeOutcome.from_response(node, result)
                state = await self._consequences.apply(state, node, outcome)

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
