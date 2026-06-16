"""Scheduler-owned conversion from DecompositionDecision to work AgentRequests."""

from forge.core.models import (
    AgentContract,
    AgentRequest,
    AgentType,
    ChildTask,
    DecompositionDecision,
    DecompositionTask,
    GraphSplitDecision,
    PlanSpec,
    RequestId,
    RequestSource,
    TaskSpec,
    WorkDecision,
    WorkSpec,
)
from forge.core.profile_assignment import DefaultProfileAssigner, ProfileAssigner


class DecompositionConvergenceError(ValueError):
    """Raised when a decomposition decision fails semantic convergence checks."""


class DecompositionConvergenceValidator:
    """Validates that decomposition decisions are reductive relative to their parent.

    Invariant: a split decision is valid only if its child tasks are strictly
    narrower, more concrete, or more executable than the parent objective.
    WorkDecision is terminal and exempt from all checks.
    """

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().strip().split())

    def validate(self, parent_objective: str, decision: DecompositionDecision) -> None:
        """Raise DecompositionConvergenceError if decision is not reductive."""
        if isinstance(decision, WorkDecision):
            return

        assert isinstance(decision, GraphSplitDecision)
        parent_norm = self._normalize(parent_objective)
        child_tasks: list[ChildTask] = [node.task for node in decision.nodes]
        child_norms: list[str] = []

        for task in child_tasks:
            obj = task.objective
            norm = self._normalize(obj)

            if len(norm) < 3:
                raise DecompositionConvergenceError(
                    f"Decomposition child has empty or near-empty objective: {obj!r}. "
                    "Return WorkDecision if the task is already atomic."
                )

            if norm == parent_norm:
                raise DecompositionConvergenceError(
                    f"Decomposition is not reductive: child objective {obj!r} repeats "
                    f"parent objective {parent_objective!r}. "
                    "Split children must be narrower or more concrete than the parent, "
                    "or return WorkDecision if the task is already atomic."
                )

            child_norms.append(norm)

        seen_child_norms: dict[str, str] = {}
        for task, norm in zip(child_tasks, child_norms):
            if norm in seen_child_norms:
                raise DecompositionConvergenceError(
                    f"Decomposition is not reductive: sibling objectives "
                    f"{seen_child_norms[norm]!r} and {task.objective!r} normalize to "
                    f"{norm!r}. Each child must address a distinct, narrower concern."
                )
            seen_child_norms[norm] = task.objective


class PlanExpansionBuilder:
    """Build scheduler work requests from an accepted planner response."""

    def __init__(
        self, request: AgentRequest, profile_assigner: ProfileAssigner | None = None
    ) -> None:
        self.request = request
        self.profile_assigner = profile_assigner or DefaultProfileAssigner()

    async def _assign_work_profile(self, request: AgentRequest) -> AgentRequest:
        profile = await self.profile_assigner.assign(request)
        return request.model_copy(update={"model_profile": profile})

    async def _task_spec_to_request(self, task: TaskSpec) -> AgentRequest:
        request = AgentRequest(
            agent_type=AgentType.WORK,
            source=RequestSource.PLANNER,
            spec=WorkSpec(
                objective=task.objective,
                success_condition=task.success_condition,
                contract=AgentContract(
                    objective=task.objective,
                    success_condition=task.success_condition,
                    acceptance_criteria=task.acceptance_criteria,
                    constraints=task.constraints,
                    non_goals=task.non_goals,
                ),
                adapter=task.adapter,
                artifact=task.artifact,
                language=task.language,
            ),
        )
        return await self._assign_work_profile(request)

    def _decomposition_task_to_request(self, task: DecompositionTask) -> AgentRequest:
        return AgentRequest(
            agent_type=AgentType.PLAN,
            source=RequestSource.PLANNER,
            spec=PlanSpec(
                northstar=task.objective,
                contract=AgentContract(
                    objective=task.objective,
                    success_condition=task.success_condition,
                ),
            ),
        )

    async def _child_task_to_request(self, task: TaskSpec | DecompositionTask) -> AgentRequest:
        if isinstance(task, DecompositionTask):
            return self._decomposition_task_to_request(task)
        return await self._task_spec_to_request(task)

    async def build_from_decision(self, decision: DecompositionDecision) -> list[AgentRequest]:
        """Convert a DecompositionDecision into agent requests."""
        parent_objective = self.request.spec.contract.objective
        DecompositionConvergenceValidator().validate(parent_objective, decision)
        if isinstance(decision, WorkDecision):
            request = AgentRequest(
                agent_type=AgentType.WORK,
                source=RequestSource.PLANNER,
                spec=decision.task,
            )
            return [await self._assign_work_profile(request)]
        assert isinstance(decision, GraphSplitDecision)
        return await self._build_from_graph_split(decision)

    async def _build_from_graph_split(self, decision: GraphSplitDecision) -> list[AgentRequest]:
        """Expand a GraphSplitDecision into requests with RequestId dependencies."""
        requests = [await self._child_task_to_request(node.task) for node in decision.nodes]
        str_id_to_request_id: dict[str, RequestId] = {
            node.id: req.id for node, req in zip(decision.nodes, requests)
        }
        return [
            req.model_copy(
                update={
                    "dependencies": frozenset(
                        str_id_to_request_id[ref]
                        for ref in node.depends_on
                        if ref in str_id_to_request_id
                    )
                }
            )
            for node, req in zip(decision.nodes, requests)
        ]
