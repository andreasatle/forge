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
    GraphSplitDecision,
    NodeState,
    PlanSpec,
    RequestId,
    RequestSource,
    ResponseStatus,
    SchedulerState,
    WorkDecision,
    WorkOutput,
    WorkSpec,
)
from forge.core.plan_expansion import (
    DecompositionConvergenceError,
    PlanExpansionBuilder,
    ProfileAssignmentError,
)
from forge.core.profile_assignment import (
    DefaultProfileAssigner,
    ProfileAssigner,
    ProfileAssignmentResult,
)
from forge.core.profile_escalation import (
    NoProfileEscalationPolicy,
    ProfileEscalationPolicy,
)
from forge.core.state_service import StateService
from forge.core.telemetry import TelemetryEvent, TelemetrySink, safe_append_telemetry

AgentRunner = Callable[[AgentRequest], Awaitable[AgentResponse]]

logger = logging.getLogger(__name__)


def _short_rationale(value: str, limit: int = 240) -> str:
    """Return a compact single-line rationale for telemetry."""
    rationale = " ".join(value.split())
    if len(rationale) <= limit:
        return rationale
    return rationale[: limit - 3].rstrip() + "..."


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


class PlanOutputValidationError(ValueError):
    """Raised when a completed PLAN response does not contain a planner decision."""


class DecompositionBudgetError(ValueError):
    """Raised when a planner expansion would exceed scheduler-owned budgets."""

    def __init__(
        self,
        reason: str,
        *,
        current_depth: int,
        max_plan_depth: int,
        dag_size: int,
        max_dag_nodes: int,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.current_depth = current_depth
        self.max_plan_depth = max_plan_depth
        self.dag_size = dag_size
        self.max_dag_nodes = max_dag_nodes


class SchedulerConsequenceHandler:
    """Owns consequences of terminal node responses."""

    def __init__(
        self,
        callbacks: SchedulerCallbacks | None = None,
        telemetry_sink: TelemetrySink | None = None,
        run_id: UUID | None = None,
        state_services: dict[str, StateService] | None = None,
        profile_assigner: ProfileAssigner | None = None,
        profile_escalation_policy: ProfileEscalationPolicy | None = None,
    ) -> None:
        self._callbacks = callbacks or SchedulerCallbacks()
        self._telemetry_sink = telemetry_sink
        self._run_id = run_id or getattr(telemetry_sink, "run_id", None)
        self._state_services = state_services or {}
        self._profile_assigner = profile_assigner or DefaultProfileAssigner()
        self._profile_escalation_policy = profile_escalation_policy or NoProfileEscalationPolicy()

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
            try:
                plan_expansion = await self._build_plan_expansion(node, response)
            except PlanOutputValidationError as exc:
                failed_response = AgentResponse(
                    request_id=node.request.id,
                    status=ResponseStatus.FAILED,
                    failure_kind=FailureKind.VALIDATION_REJECTED,
                    error=str(exc),
                )
                failed_updated = current.with_response(failed_response)
                state = state.update_node(failed_updated)
                return self._handle_failed(
                    state, failed_updated, TerminalOutcomeKind.TERMINAL_FAILURE
                )
            except DecompositionConvergenceError as exc:
                logger.warning(
                    "decomposition convergence check failed for node %s: %s",
                    node.request.id,
                    exc,
                )
                self._emit_convergence_failure(node, str(exc))
                failed_response = AgentResponse(
                    request_id=node.request.id,
                    status=ResponseStatus.FAILED,
                    failure_kind=FailureKind.VALIDATION_REJECTED,
                    error=str(exc),
                )
                failed_updated = current.with_response(failed_response)
                state = state.update_node(failed_updated)
                return self._handle_failed(
                    state, failed_updated, TerminalOutcomeKind.TERMINAL_FAILURE
                )
            except ProfileAssignmentError as exc:
                logger.warning(
                    "profile assignment failed for plan node %s: %s",
                    node.request.id,
                    exc,
                )
                failed_response = AgentResponse(
                    request_id=node.request.id,
                    status=ResponseStatus.FAILED,
                    failure_kind=FailureKind.VALIDATION_REJECTED,
                    error=str(exc),
                )
                failed_updated = current.with_response(failed_response)
                state = state.update_node(failed_updated)
                return self._handle_failed(
                    state, failed_updated, TerminalOutcomeKind.TERMINAL_FAILURE
                )
            try:
                self._validate_plan_expansion_budget(state, updated, plan_expansion)
            except DecompositionBudgetError as exc:
                logger.warning(
                    "decomposition budget check failed for node %s: %s",
                    node.request.id,
                    exc,
                )
                self._emit_convergence_failure(
                    node,
                    exc.reason,
                    current_depth=exc.current_depth,
                    max_plan_depth=exc.max_plan_depth,
                    dag_size=exc.dag_size,
                    max_dag_nodes=exc.max_dag_nodes,
                )
                failed_response = AgentResponse(
                    request_id=node.request.id,
                    status=ResponseStatus.FAILED,
                    failure_kind=FailureKind.VALIDATION_REJECTED,
                    error=exc.reason,
                )
                failed_updated = current.with_response(failed_response)
                state = state.update_node(failed_updated)
                return self._handle_failed(
                    state, failed_updated, TerminalOutcomeKind.TERMINAL_FAILURE
                )
            return self._handle_accepted_plan(state, updated, plan_expansion)
        if outcome.kind == TerminalOutcomeKind.ACCEPTED_WORK:
            return await self._handle_accepted_work(state, updated)
        if outcome.kind == TerminalOutcomeKind.DECOMPOSITION_REQUEST:
            return self._handle_decompose(state, updated)

        return self._handle_failed(state, updated, outcome.kind)

    async def _build_plan_expansion(
        self,
        node: DAGNode,
        response: AgentResponse,
    ) -> list[DAGNode]:
        if (
            node.request.agent_type == AgentType.PLAN
            and response.status == ResponseStatus.COMPLETED
        ):
            output = response.output
            if output is None:
                raise PlanOutputValidationError(
                    "completed PLAN response did not include WorkDecision or GraphSplitDecision output"
                )
            if isinstance(output, (WorkDecision, GraphSplitDecision)):
                builder = PlanExpansionBuilder(
                    node.request, profile_assigner=self._profile_assigner
                )
                requests = await builder.build_from_decision(output)
                self._emit_profile_assigned_events(
                    parent=node,
                    requests=requests,
                    assignments=builder.profile_assignment_results,
                )
                return [DAGNode(request=request) for request in requests]
        return []

    def _validate_plan_expansion_budget(
        self,
        state: SchedulerState,
        parent: DAGNode,
        plan_expansion: list[DAGNode],
    ) -> None:
        dag_size_after_expansion = len(state.dag) + len(plan_expansion)
        if dag_size_after_expansion > state.max_dag_nodes:
            raise DecompositionBudgetError(
                (
                    "Decomposition expansion would exceed max_dag_nodes: "
                    f"{dag_size_after_expansion} > {state.max_dag_nodes}"
                ),
                current_depth=parent.decomposition_depth,
                max_plan_depth=state.max_plan_depth,
                dag_size=len(state.dag),
                max_dag_nodes=state.max_dag_nodes,
            )

        child_plan_depth = parent.decomposition_depth + 1
        if any(
            child.request.agent_type == AgentType.PLAN and child_plan_depth > state.max_plan_depth
            for child in plan_expansion
        ):
            raise DecompositionBudgetError(
                (
                    "Decomposition expansion would exceed max_plan_depth: "
                    f"{child_plan_depth} > {state.max_plan_depth}"
                ),
                current_depth=child_plan_depth,
                max_plan_depth=state.max_plan_depth,
                dag_size=len(state.dag),
                max_dag_nodes=state.max_dag_nodes,
            )

    def _handle_accepted_plan(
        self,
        state: SchedulerState,
        updated: DAGNode,
        plan_expansion: list[DAGNode],
    ) -> SchedulerState:
        if plan_expansion:
            state = state.add_nodes(
                [self._with_child_decomposition_depth(updated, child) for child in plan_expansion]
            )
        self._fire_node(self._callbacks.on_node_completed, updated)
        return state

    @staticmethod
    def _with_child_decomposition_depth(parent: DAGNode, child: DAGNode) -> DAGNode:
        depth = parent.decomposition_depth
        if child.request.agent_type == AgentType.PLAN:
            depth += 1
        return child.model_copy(update={"decomposition_depth": depth})

    async def _handle_accepted_work(
        self,
        state: SchedulerState,
        updated: DAGNode,
    ) -> SchedulerState:
        response = updated.response
        if response is None or response.status == ResponseStatus.ALREADY_DONE:
            self._fire_node(self._callbacks.on_node_completed, updated)
            return state

        spec = updated.request.spec
        if not isinstance(spec, WorkSpec):
            failed = self._integration_failed_node(
                updated,
                FailureKind.INTEGRATION_FAILED,
                f"expected WorkSpec, got {type(spec).__name__}",
            )
            state = state.update_node(failed)
            return self._handle_failed(state, failed, TerminalOutcomeKind.INTEGRATION_FAILURE)

        state_service = self._state_services.get(spec.artifact)
        if state_service is None:
            self._fire_node(self._callbacks.on_node_completed, updated)
            return state

        work_output = response.output if isinstance(response.output, WorkOutput) else None
        if work_output is None:
            failed = self._integration_failed_node(
                updated,
                FailureKind.INTEGRATION_FAILED,
                "completed without WorkOutput completion metadata",
            )
            self._remove_integrated_worktree(state_service, str(updated.request.id))
            state = state.update_node(failed)
            return self._handle_failed(state, failed, TerminalOutcomeKind.INTEGRATION_FAILURE)
        if not work_output.summary.strip():
            failed = self._integration_failed_node(
                updated,
                FailureKind.INTEGRATION_FAILED,
                "completed with empty WorkOutput completion metadata",
            )
            self._remove_integrated_worktree(state_service, str(updated.request.id))
            state = state.update_node(failed)
            return self._handle_failed(state, failed, TerminalOutcomeKind.INTEGRATION_FAILURE)

        try:
            await state_service.apply_work_output(
                work_output,
                str(updated.request.id),
                dispatch_sha=response.dispatch_sha,
            )
        except RuntimeError as exc:
            failed = self._integration_failed_node(
                updated,
                self._classify_integration_error(exc),
                f"integration failed: {exc}",
            )
            state = state.update_node(failed)
            return self._handle_failed(state, failed, TerminalOutcomeKind.INTEGRATION_FAILURE)
        finally:
            self._remove_integrated_worktree(state_service, str(updated.request.id))

        self._fire_node(self._callbacks.on_node_completed, updated)
        return state

    @staticmethod
    def _integration_failed_node(
        node: DAGNode,
        failure_kind: FailureKind,
        error: str,
    ) -> DAGNode:
        response = node.response
        failed_response = AgentResponse(
            request_id=node.request.id,
            status=ResponseStatus.FAILED,
            output=response.output if response else None,
            error=error,
            failure_kind=failure_kind,
            ran_tests_and_passed=response.ran_tests_and_passed if response else False,
            diagnostics=response.diagnostics if response else [],
            revision=response.revision if response else None,
            dispatch_sha=response.dispatch_sha if response else "",
        )
        return node.with_response(failed_response)

    @staticmethod
    def _classify_integration_error(error: RuntimeError) -> FailureKind:
        message = str(error).lower()
        if "stale base_version" in message:
            return FailureKind.STALE_WORK_OUTPUT
        if "tests failed after work output" in message:
            return FailureKind.TEST_FAILED
        if "no worktree changes produced" in message:
            return FailureKind.VALIDATION_REJECTED
        return FailureKind.INTEGRATION_FAILED

    @staticmethod
    def _remove_integrated_worktree(
        state_service: StateService,
        node_id: str,
    ) -> None:
        try:
            state_service.remove_worktree(node_id)
        except Exception:
            logger.exception("failed to remove worktree for integrated node %s", node_id)

    @staticmethod
    def _normalize_objective(text: str) -> str:
        return " ".join(text.lower().strip().split())

    def _prior_decompose_plan_exists(
        self,
        state: SchedulerState,
        current_id: RequestId,
        objective: str,
    ) -> bool:
        """True if another cancelled PLAN already returned DECOMPOSE for this objective."""
        norm = self._normalize_objective(objective)
        return any(
            node.request.id != current_id
            and node.request.agent_type == AgentType.PLAN
            and node.node_state == NodeState.CANCELLED
            and node.response is not None
            and node.response.status == ResponseStatus.DECOMPOSE
            and self._normalize_objective(node.request.spec.contract.objective) == norm
            for node in state.dag.values()
        )

    def _handle_decompose(self, state: SchedulerState, updated: DAGNode) -> SchedulerState:
        spec = updated.request.spec
        if isinstance(spec, WorkSpec):
            northstar = spec.objective
        else:
            northstar = spec.northstar

        replacement_objective = spec.contract.objective
        if self._prior_decompose_plan_exists(state, updated.request.id, replacement_objective):
            error = (
                "DECOMPOSE is not reductive: replacement objective repeats the current objective. "
                "Return WorkDecision if the task is already atomic or propose a narrower decomposition."
            )
            self._emit_convergence_failure(updated, error)
            failed_response = AgentResponse(
                request_id=updated.request.id,
                status=ResponseStatus.FAILED,
                failure_kind=FailureKind.VALIDATION_REJECTED,
                error=error,
            )
            failed_node = updated.model_copy(
                update={"response": failed_response, "node_state": NodeState.FAILED}
            )
            state = state.update_node(failed_node)
            return self._handle_failed(state, failed_node, TerminalOutcomeKind.TERMINAL_FAILURE)

        new_plan_request = AgentRequest(
            agent_type=AgentType.PLAN,
            source=RequestSource.USER,
            spec=PlanSpec(
                northstar=northstar,
                contract=AgentContract(
                    objective=replacement_objective,
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
        state = state.add_nodes(
            [DAGNode(request=new_plan_request, decomposition_depth=updated.decomposition_depth)]
        )
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
        if updated.response is not None:
            retry = self._profile_escalation_retry_node(updated, updated.response)
            if retry is not None:
                state = state.add_nodes([retry])
                state = self._transfer_dependents(state, updated.request.id, retry.request.id)
                self._emit_profile_escalated(updated, retry)
                return state

        self._emit_node_failed(updated)
        self._fire_node(self._callbacks.on_node_failed, updated)
        return self._cancel_dependents(state, updated.request.id)

    def _profile_escalation_retry_node(
        self,
        node: DAGNode,
        response: AgentResponse,
    ) -> DAGNode | None:
        next_profile = self._profile_escalation_policy.next_profile(node, response)
        if next_profile is None:
            return None

        retry_request = AgentRequest(
            agent_type=node.request.agent_type,
            source=node.request.source,
            spec=node.request.spec,
            dependencies=node.request.dependencies,
            model_profile=next_profile,
        )
        return DAGNode(
            request=retry_request,
            decomposition_depth=node.decomposition_depth,
            retry_of=node.request.id,
            profile_escalation_attempt=node.profile_escalation_attempt + 1,
            prior_profiles=(*node.prior_profiles, node.request.model_profile),
        )

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

    def _emit_convergence_failure(
        self,
        node: DAGNode,
        reason: str,
        *,
        current_depth: int | None = None,
        max_plan_depth: int | None = None,
        dag_size: int | None = None,
        max_dag_nodes: int | None = None,
    ) -> None:
        if self._run_id is None:
            return
        data: dict[str, object] = {"reason": reason}
        if current_depth is not None:
            data["depth"] = current_depth
        if max_plan_depth is not None:
            data["max_depth"] = max_plan_depth
        if dag_size is not None:
            data["dag_size"] = dag_size
        if max_dag_nodes is not None:
            data["max_dag_nodes"] = max_dag_nodes
        safe_append_telemetry(
            self._telemetry_sink,
            TelemetryEvent(
                run_id=self._run_id,
                node_id=node.request.id,
                request_id=node.request.id,
                agent_type=node.request.agent_type.value,
                role="scheduler",
                phase="scheduler",
                event_type="node.convergence_failed",
                status="failed",
                summary="Decomposition rejected: not reductive",
                data=data,
            ),
        )

    def _emit_profile_assigned_events(
        self,
        *,
        parent: DAGNode,
        requests: list[AgentRequest],
        assignments: dict[RequestId, ProfileAssignmentResult],
    ) -> None:
        if self._run_id is None:
            return
        for request in requests:
            if request.agent_type is not AgentType.WORK or not isinstance(request.spec, WorkSpec):
                continue
            assignment = assignments.get(request.id)
            if assignment is None:
                assignment = ProfileAssignmentResult(model_profile=request.model_profile)
            data: dict[str, object] = {
                "child_request_id": str(request.id),
                "parent_request_id": str(parent.request.id),
                "artifact": request.spec.artifact,
                "adapter": request.spec.adapter,
                "model_profile": assignment.model_profile,
            }
            if request.spec.language is not None:
                data["language"] = request.spec.language
            if assignment.complexity is not None:
                data["complexity"] = assignment.complexity.value
            if assignment.rationale is not None:
                data["rationale"] = _short_rationale(assignment.rationale)

            safe_append_telemetry(
                self._telemetry_sink,
                TelemetryEvent(
                    run_id=self._run_id,
                    node_id=request.id,
                    request_id=request.id,
                    agent_type=request.agent_type.value,
                    role="scheduler",
                    phase="routing",
                    event_type="node.profile_assigned",
                    status="assigned",
                    summary=f"worker task assigned to profile {assignment.model_profile!r}",
                    data=data,
                ),
            )

    def _emit_profile_escalated(self, failed: DAGNode, retry: DAGNode) -> None:
        if self._run_id is None:
            return
        response = failed.response
        reason = response.failure_kind.value if response and response.failure_kind else None
        error = response.error if response else None
        safe_append_telemetry(
            self._telemetry_sink,
            TelemetryEvent(
                run_id=self._run_id,
                node_id=retry.request.id,
                request_id=retry.request.id,
                agent_type=retry.request.agent_type.value,
                role="scheduler",
                phase="routing",
                event_type="node.profile_escalated",
                status="retry",
                summary=(
                    f"worker profile escalated from {failed.request.model_profile!r} "
                    f"to {retry.request.model_profile!r}"
                ),
                data={
                    "failed_node_id": str(failed.request.id),
                    "retry_node_id": str(retry.request.id),
                    "old_profile": failed.request.model_profile,
                    "new_profile": retry.request.model_profile,
                    "reason": reason,
                    "error": error,
                    "attempt": retry.profile_escalation_attempt,
                },
            ),
        )

    def emit_node_dispatched(self, node: DAGNode) -> None:
        """Emit node.dispatched with the node's contract so traces expose planner intent."""
        if self._run_id is None:
            return
        spec = node.request.spec
        contract_data: dict[str, object] = {
            "objective": spec.contract.objective,
            "success_condition": spec.contract.success_condition,
            "acceptance_criteria": [
                {"id": c.id, "text": c.text} for c in spec.contract.acceptance_criteria
            ],
        }
        if isinstance(spec, WorkSpec):
            contract_data["artifact"] = spec.artifact
            contract_data["adapter"] = spec.adapter
        safe_append_telemetry(
            self._telemetry_sink,
            TelemetryEvent(
                run_id=self._run_id,
                node_id=node.request.id,
                request_id=node.request.id,
                agent_type=node.request.agent_type.value,
                role="scheduler",
                phase="scheduler",
                event_type="node.dispatched",
                status="dispatched",
                summary=spec.contract.objective,
                data={"contract": contract_data, "source": node.request.source.value},
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
        callbacks: SchedulerCallbacks | None = None,
        telemetry_sink: TelemetrySink | None = None,
        run_id: UUID | None = None,
        state_services: dict[str, StateService] | None = None,
        profile_assigner: ProfileAssigner | None = None,
        profile_escalation_policy: ProfileEscalationPolicy | None = None,
    ) -> None:
        self._runner = runner
        self._callbacks = callbacks or SchedulerCallbacks()
        self._consequences = SchedulerConsequenceHandler(
            callbacks=self._callbacks,
            telemetry_sink=telemetry_sink,
            run_id=run_id,
            state_services=state_services,
            profile_assigner=profile_assigner,
            profile_escalation_policy=profile_escalation_policy,
        )

    async def run(
        self,
        state: SchedulerState,
    ) -> SchedulerState:
        """Drive state forward until no PENDING or RUNNING nodes remain."""
        while True:
            ready = state.ready_nodes()

            if not ready:
                self._fire_state(self._callbacks.on_idle, state)
                break

            to_dispatch = ready[: state.max_concurrency]

            for node in to_dispatch:
                running = node.with_state(NodeState.RUNNING)
                state = state.update_node(running)
                self._fire_node(self._callbacks.on_node_dispatched, running)
                self._consequences.emit_node_dispatched(running)

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
