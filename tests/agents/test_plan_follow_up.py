"""Tests for planner follow-up request construction."""

from forge.agents.plan_follow_up import PlanFollowUpBuilder
from forge.core.models import (
    AcceptanceCriterion,
    AgentRequest,
    AgentType,
    PlanResponse,
    PlanSpec,
    RequestSource,
    TaskSpec,
    WorkSpec,
)


def _plan_request() -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(northstar="build a scraper"),
    )


def test_plan_follow_up_builder_emits_simple_work_node() -> None:
    """A single plan task becomes a single WORK follow-up request."""
    plan = PlanResponse(
        kind="plan",
        tasks=[
            TaskSpec(
                objective="write code",
                success_condition="tests pass",
                adapter="coding",
                artifact="codebase",
            )
        ],
    )

    follow_ups = PlanFollowUpBuilder(_plan_request()).build(plan)

    assert len(follow_ups) == 1
    assert follow_ups[0].agent_type == AgentType.WORK
    assert follow_ups[0].source == RequestSource.PLANNER
    assert isinstance(follow_ups[0].spec, WorkSpec)


def test_plan_follow_up_builder_remaps_task_dependencies_to_work_ids() -> None:
    """depends_on indices become dependencies on the corresponding work request IDs."""
    plan = PlanResponse(
        kind="plan",
        tasks=[
            TaskSpec(
                objective="A", success_condition="done", adapter="coding", artifact="codebase"
            ),
            TaskSpec(
                objective="B",
                success_condition="done",
                adapter="coding",
                artifact="codebase",
                depends_on=[0],
            ),
        ],
    )

    follow_ups = PlanFollowUpBuilder(_plan_request()).build(plan)

    work_a = next(r for r in follow_ups if isinstance(r.spec, WorkSpec) and r.spec.objective == "A")
    work_b = next(r for r in follow_ups if isinstance(r.spec, WorkSpec) and r.spec.objective == "B")
    assert work_a.id in work_b.dependencies
    assert len(work_b.dependencies) == 1


def test_plan_follow_up_builder_propagates_artifact_and_language() -> None:
    """Task artifact and language are copied into the generated WorkSpec."""
    plan = PlanResponse(
        kind="plan",
        tasks=[
            TaskSpec(
                objective="write code",
                success_condition="tests pass",
                adapter="coding",
                artifact="api",
                language="python",
            )
        ],
    )

    follow_up = PlanFollowUpBuilder(_plan_request()).build(plan)[0]

    assert isinstance(follow_up.spec, WorkSpec)
    assert follow_up.spec.artifact == "api"
    assert follow_up.spec.language == "python"


def test_plan_follow_up_builder_preserves_contract_fields() -> None:
    """Planner-emitted contract fields are copied into the generated WorkSpec contract."""
    plan = PlanResponse(
        kind="plan",
        tasks=[
            TaskSpec(
                objective="write code",
                success_condition="tests pass",
                acceptance_criteria=[AcceptanceCriterion(id="AC1", text="unit tests cover parser")],
                constraints=["use stdlib"],
                non_goals=["network UI"],
                adapter="coding",
                artifact="api",
                language="python",
            )
        ],
    )

    follow_up = PlanFollowUpBuilder(_plan_request()).build(plan)[0]

    assert isinstance(follow_up.spec, WorkSpec)
    assert follow_up.spec.contract.acceptance_criteria == [
        AcceptanceCriterion(id="AC1", text="unit tests cover parser")
    ]
    assert follow_up.spec.contract.constraints == ["use stdlib"]
    assert follow_up.spec.contract.non_goals == ["network UI"]


def test_plan_follow_up_builder_ignores_out_of_range_dependency_indices() -> None:
    """Existing manual dependency validation ignores indices outside the task list."""
    plan = PlanResponse(
        kind="plan",
        tasks=[
            TaskSpec(
                objective="A",
                success_condition="done",
                adapter="coding",
                artifact="codebase",
                depends_on=[99],
            )
        ],
    )

    follow_up = PlanFollowUpBuilder(_plan_request()).build(plan)[0]

    assert follow_up.dependencies == frozenset()


def test_plan_follow_up_builder_empty_plan_returns_no_follow_ups() -> None:
    """An empty plan produces no follow-up requests."""
    plan = PlanResponse(kind="plan", tasks=[])

    assert PlanFollowUpBuilder(_plan_request()).build(plan) == []
