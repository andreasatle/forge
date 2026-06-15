"""Scheduler-owned conversion from PlanResponse to work AgentRequests."""

from forge.core.models import (
    AgentContract,
    AgentRequest,
    AgentType,
    DecompositionDecision,
    DependentSplitDecision,
    OrthogonalSplitDecision,
    PlanResponse,
    RequestSource,
    TaskSpec,
    WorkDecision,
    WorkSpec,
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
        """Convert a DecompositionDecision into work requests."""
        if isinstance(decision, WorkDecision):
            return [
                AgentRequest(
                    agent_type=AgentType.WORK,
                    source=RequestSource.PLANNER,
                    spec=decision.task,
                )
            ]
        if isinstance(decision, DependentSplitDecision):
            nodes = [self._task_spec_to_request(task) for task in decision.tasks]
            result: list[AgentRequest] = [nodes[0]]
            for i in range(1, len(nodes)):
                result.append(
                    nodes[i].model_copy(update={"dependencies": frozenset({nodes[i - 1].id})})
                )
            return result
        # OrthogonalSplitDecision — siblings are independent
        assert isinstance(decision, OrthogonalSplitDecision)
        return [self._task_spec_to_request(task) for task in decision.tasks]
