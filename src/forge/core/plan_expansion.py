"""Scheduler-owned conversion from PlanResponse to work AgentRequests."""

from forge.core.models import (
    AgentContract,
    AgentRequest,
    AgentType,
    DecompositionDecision,
    DecompositionTask,
    DependentSplitDecision,
    OrthogonalSplitDecision,
    PlanResponse,
    PlanSpec,
    RequestSource,
    TaskSpec,
    WorkDecision,
    WorkSpec,
)


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

        parent_norm = self._normalize(parent_objective)
        child_norms: list[str] = []

        for task in decision.tasks:
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

        if len(child_norms) >= 2 and len(set(child_norms)) == 1:
            raise DecompositionConvergenceError(
                f"Decomposition is not reductive: all child objectives normalize to "
                f"{child_norms[0]!r}. Each child must address a distinct, narrower concern."
            )


class PlanExpansionBuilder:
    """Build scheduler work requests from an accepted planner response."""

    def __init__(self, request: AgentRequest) -> None:
        self.request = request

    def _task_spec_to_request(self, task: TaskSpec) -> AgentRequest:
        return AgentRequest(
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

    def _child_task_to_request(self, task: TaskSpec | DecompositionTask) -> AgentRequest:
        if isinstance(task, DecompositionTask):
            return self._decomposition_task_to_request(task)
        return self._task_spec_to_request(task)

    def build(self, plan_response: PlanResponse) -> list[AgentRequest]:
        """Convert a PlanResponse into work requests with remapped dependencies."""
        if not plan_response.tasks:
            return []

        work_nodes = [self._task_spec_to_request(task) for task in plan_response.tasks]

        return [
            work.model_copy(
                update={
                    "dependencies": frozenset(
                        work_nodes[j].id for j in task.depends_on if 0 <= j < len(work_nodes)
                    )
                }
            )
            for work, task in zip(work_nodes, plan_response.tasks)
        ]

    def build_from_decision(self, decision: DecompositionDecision) -> list[AgentRequest]:
        """Convert a DecompositionDecision into agent requests."""
        parent_objective = self.request.spec.contract.objective
        DecompositionConvergenceValidator().validate(parent_objective, decision)
        if isinstance(decision, WorkDecision):
            return [
                AgentRequest(
                    agent_type=AgentType.WORK,
                    source=RequestSource.PLANNER,
                    spec=decision.task,
                )
            ]
        if isinstance(decision, DependentSplitDecision):
            nodes = [self._child_task_to_request(task) for task in decision.tasks]
            result: list[AgentRequest] = [nodes[0]]
            for i in range(1, len(nodes)):
                result.append(
                    nodes[i].model_copy(update={"dependencies": frozenset({nodes[i - 1].id})})
                )
            return result
        # OrthogonalSplitDecision — siblings are independent
        assert isinstance(decision, OrthogonalSplitDecision)
        return [self._child_task_to_request(task) for task in decision.tasks]
