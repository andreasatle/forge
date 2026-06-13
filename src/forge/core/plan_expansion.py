"""Scheduler-owned conversion from PlanResponse to work AgentRequests."""

from forge.core.models import (
    AgentContract,
    AgentRequest,
    AgentType,
    PlanResponse,
    RequestSource,
    WorkSpec,
)


class PlanExpansionBuilder:
    """Build scheduler work requests from an accepted planner response."""

    def __init__(self, request: AgentRequest) -> None:
        self.request = request

    def build(self, plan_response: PlanResponse) -> list[AgentRequest]:
        """Convert a PlanResponse into work requests with remapped dependencies."""
        if not plan_response.tasks:
            return []

        work_nodes = [
            AgentRequest(
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
            for task in plan_response.tasks
        ]

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
