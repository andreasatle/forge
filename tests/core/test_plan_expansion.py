"""Tests for scheduler-owned plan expansion."""

import pytest

from forge.core.models import (
    AcceptanceCriterion,
    AgentMessageKind,
    AgentRequest,
    AgentType,
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
from forge.core.plan_expansion import (
    DecompositionConvergenceError,
    DecompositionConvergenceValidator,
    PlanExpansionBuilder,
)


def _plan_request() -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(northstar="build a scraper"),
    )


def test_plan_expansion_builder_emits_simple_work_node() -> None:
    """A single plan task becomes a single WORK request."""
    plan = PlanResponse(
        kind=AgentMessageKind.PLAN,
        tasks=[
            TaskSpec(
                objective="write code",
                success_condition="tests pass",
                adapter="coding",
                artifact="codebase",
            )
        ],
    )

    work_requests = PlanExpansionBuilder(_plan_request()).build(plan)

    assert len(work_requests) == 1
    assert work_requests[0].agent_type == AgentType.WORK
    assert work_requests[0].source == RequestSource.PLANNER
    assert isinstance(work_requests[0].spec, WorkSpec)


def test_plan_expansion_builder_remaps_task_dependencies_to_work_ids() -> None:
    """depends_on indices become dependencies on the corresponding work request IDs."""
    plan = PlanResponse(
        kind=AgentMessageKind.PLAN,
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

    work_requests = PlanExpansionBuilder(_plan_request()).build(plan)

    work_a = next(
        r for r in work_requests if isinstance(r.spec, WorkSpec) and r.spec.objective == "A"
    )
    work_b = next(
        r for r in work_requests if isinstance(r.spec, WorkSpec) and r.spec.objective == "B"
    )
    assert work_a.id in work_b.dependencies
    assert len(work_b.dependencies) == 1


def test_plan_expansion_builder_propagates_artifact_and_language() -> None:
    """Task artifact and language are copied into the generated WorkSpec."""
    plan = PlanResponse(
        kind=AgentMessageKind.PLAN,
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

    work_request = PlanExpansionBuilder(_plan_request()).build(plan)[0]

    assert isinstance(work_request.spec, WorkSpec)
    assert work_request.spec.artifact == "api"
    assert work_request.spec.language == "python"


def test_plan_expansion_builder_preserves_contract_fields() -> None:
    """Planner-emitted contract fields are copied into the generated WorkSpec contract."""
    plan = PlanResponse(
        kind=AgentMessageKind.PLAN,
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

    work_request = PlanExpansionBuilder(_plan_request()).build(plan)[0]

    assert isinstance(work_request.spec, WorkSpec)
    assert work_request.spec.contract.acceptance_criteria == [
        AcceptanceCriterion(id="AC1", text="unit tests cover parser")
    ]
    assert work_request.spec.contract.constraints == ["use stdlib"]
    assert work_request.spec.contract.non_goals == ["network UI"]


def test_plan_expansion_builder_ignores_out_of_range_dependency_indices() -> None:
    """Existing manual dependency validation ignores indices outside the task list."""
    plan = PlanResponse(
        kind=AgentMessageKind.PLAN,
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

    work_request = PlanExpansionBuilder(_plan_request()).build(plan)[0]

    assert work_request.dependencies == frozenset()


def test_plan_expansion_builder_empty_plan_returns_no_work_requests() -> None:
    """An empty plan produces no work requests."""
    plan = PlanResponse(kind=AgentMessageKind.PLAN, tasks=[])

    assert PlanExpansionBuilder(_plan_request()).build(plan) == []


# --- DecompositionDecision expansion ---


def _make_task_spec(objective: str = "task") -> TaskSpec:
    return TaskSpec(
        objective=objective,
        success_condition="done",
        adapter="coding",
        artifact="codebase",
    )


def test_work_decision_expands_to_one_work_node() -> None:
    """WorkDecision produces exactly one WORK request carrying the provided WorkSpec."""
    work_spec = WorkSpec(
        objective="implement parser",
        success_condition="parser passes tests",
        adapter="coding",
        artifact="codebase",
    )
    decision = WorkDecision(task=work_spec)

    requests = PlanExpansionBuilder(_plan_request()).build_from_decision(decision)

    assert len(requests) == 1
    assert requests[0].agent_type == AgentType.WORK
    assert requests[0].source == RequestSource.PLANNER
    assert requests[0].spec == work_spec
    assert requests[0].dependencies == frozenset()


def test_dependent_split_decision_expands_to_chained_work_nodes() -> None:
    """DependentSplitDecision with three tasks produces three WORK nodes chained a->b->c."""
    decision = DependentSplitDecision(
        tasks=[_make_task_spec("task-a"), _make_task_spec("task-b"), _make_task_spec("task-c")]
    )

    requests = PlanExpansionBuilder(_plan_request()).build_from_decision(decision)
    node_a, node_b, node_c = requests

    assert len(requests) == 3
    assert all(r.agent_type == AgentType.WORK for r in requests)
    assert node_a.dependencies == frozenset()
    assert node_a.id in node_b.dependencies
    assert len(node_b.dependencies) == 1
    assert node_b.id in node_c.dependencies
    assert len(node_c.dependencies) == 1


def test_orthogonal_split_decision_expands_to_independent_work_nodes() -> None:
    """OrthogonalSplitDecision with three tasks produces three WORK nodes with no sibling deps."""
    decision = OrthogonalSplitDecision(
        tasks=[_make_task_spec("task-a"), _make_task_spec("task-b"), _make_task_spec("task-c")]
    )

    requests = PlanExpansionBuilder(_plan_request()).build_from_decision(decision)

    assert len(requests) == 3
    assert all(r.agent_type == AgentType.WORK for r in requests)
    assert all(r.dependencies == frozenset() for r in requests)


# --- DecompositionTask expansion ---


def _make_decomposition_task(objective: str = "sub-plan") -> DecompositionTask:
    return DecompositionTask(objective=objective, success_condition="planned")


def test_decomposition_task_inside_split_expands_to_plan_node() -> None:
    """A DecompositionTask inside an orthogonal split produces an AgentType.PLAN request."""
    decision = OrthogonalSplitDecision(tasks=[_make_decomposition_task("plan the sub-system")])

    requests = PlanExpansionBuilder(_plan_request()).build_from_decision(decision)

    assert len(requests) == 1
    assert requests[0].agent_type == AgentType.PLAN
    assert isinstance(requests[0].spec, PlanSpec)
    assert requests[0].spec.northstar == "plan the sub-system"
    assert requests[0].source == RequestSource.PLANNER
    assert requests[0].dependencies == frozenset()


def test_dependent_split_mixed_children_creates_dependency_chain() -> None:
    """DependentSplitDecision with mixed work/decomposition children chains them in order."""
    decision = DependentSplitDecision(
        tasks=[
            _make_task_spec("work-a"),
            _make_decomposition_task("plan-b"),
            _make_task_spec("work-c"),
        ]
    )

    requests = PlanExpansionBuilder(_plan_request()).build_from_decision(decision)
    node_a, node_b, node_c = requests

    assert node_a.agent_type == AgentType.WORK
    assert node_b.agent_type == AgentType.PLAN
    assert node_c.agent_type == AgentType.WORK
    assert node_a.dependencies == frozenset()
    assert node_a.id in node_b.dependencies
    assert len(node_b.dependencies) == 1
    assert node_b.id in node_c.dependencies
    assert len(node_c.dependencies) == 1


def test_orthogonal_split_mixed_children_creates_no_sibling_dependencies() -> None:
    """OrthogonalSplitDecision with mixed children produces no sibling dependencies."""
    decision = OrthogonalSplitDecision(
        tasks=[
            _make_task_spec("work-a"),
            _make_decomposition_task("plan-b"),
        ]
    )

    requests = PlanExpansionBuilder(_plan_request()).build_from_decision(decision)

    assert len(requests) == 2
    assert requests[0].agent_type == AgentType.WORK
    assert requests[1].agent_type == AgentType.PLAN
    assert all(r.dependencies == frozenset() for r in requests)


def test_all_work_expansion_unchanged_with_task_spec() -> None:
    """Legacy all-work expansion via TaskSpec still produces only WORK nodes."""
    decision = DependentSplitDecision(tasks=[_make_task_spec("task-a"), _make_task_spec("task-b")])

    requests = PlanExpansionBuilder(_plan_request()).build_from_decision(decision)

    assert all(r.agent_type == AgentType.WORK for r in requests)
    assert len(requests) == 2


# --- DecompositionConvergenceValidator ---


def _make_validator() -> DecompositionConvergenceValidator:
    return DecompositionConvergenceValidator()


def test_convergence_validator_accepts_narrower_children() -> None:
    """Valid decomposition with distinct, narrower child objectives passes."""
    decision = OrthogonalSplitDecision(
        tasks=[
            _make_task_spec("implement HTTP fetching"),
            _make_task_spec("implement HTML parsing"),
            _make_task_spec("add CLI interface"),
        ]
    )

    _make_validator().validate("implement web scraper", decision)  # must not raise


def test_convergence_validator_rejects_child_identical_to_parent() -> None:
    """A child whose normalized objective matches the parent is rejected."""
    decision = OrthogonalSplitDecision(
        tasks=[
            _make_task_spec("build a web scraper"),
            _make_task_spec("add CLI interface"),
        ]
    )

    with pytest.raises(DecompositionConvergenceError, match="not reductive"):
        _make_validator().validate("build a web scraper", decision)


def test_convergence_validator_rejects_child_identical_to_parent_case_insensitive() -> None:
    """Normalization is case-insensitive — 'Build A Web Scraper' equals 'build a web scraper'."""
    decision = OrthogonalSplitDecision(
        tasks=[_make_task_spec("Build A Web Scraper"), _make_task_spec("add CLI")]
    )

    with pytest.raises(DecompositionConvergenceError, match="not reductive"):
        _make_validator().validate("build a web scraper", decision)


def test_convergence_validator_rejects_all_identical_children() -> None:
    """All children normalizing to the same objective is rejected."""
    decision = DependentSplitDecision(
        tasks=[
            _make_task_spec("write comprehensive tests"),
            _make_task_spec("Write Comprehensive Tests"),
        ]
    )

    with pytest.raises(DecompositionConvergenceError, match="not reductive"):
        _make_validator().validate("implement the system", decision)


def test_convergence_validator_rejects_empty_child_objective() -> None:
    """A child with an empty or near-empty objective is rejected."""
    decision = OrthogonalSplitDecision(tasks=[_make_task_spec("  "), _make_task_spec("add CLI")])

    with pytest.raises(DecompositionConvergenceError, match="near-empty"):
        _make_validator().validate("build a web scraper", decision)


def test_convergence_validator_accepts_work_decision() -> None:
    """WorkDecision is exempt from all convergence checks."""
    work_spec = WorkSpec(
        objective="build a web scraper",
        success_condition="tests pass",
        adapter="coding",
        artifact="codebase",
    )
    decision = WorkDecision(task=work_spec)

    # parent objective equals child objective — still valid for WorkDecision
    _make_validator().validate("build a web scraper", decision)  # must not raise


def test_convergence_validator_single_child_not_identical_passes() -> None:
    """A single child that is distinct from the parent is accepted."""
    decision = OrthogonalSplitDecision(tasks=[_make_task_spec("implement HTTP fetching")])

    _make_validator().validate("implement web scraper", decision)  # must not raise


def test_build_from_decision_raises_on_identical_child_to_parent() -> None:
    """build_from_decision raises DecompositionConvergenceError when a child repeats the parent."""
    parent = AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(northstar="build a web scraper"),
    )
    decision = OrthogonalSplitDecision(
        tasks=[_make_task_spec("build a web scraper"), _make_task_spec("add CLI")]
    )

    with pytest.raises(DecompositionConvergenceError):
        PlanExpansionBuilder(parent).build_from_decision(decision)


def test_build_from_decision_work_decision_not_rejected_by_convergence() -> None:
    """WorkDecision passes through build_from_decision even when objective matches parent."""
    parent = AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(northstar="build a web scraper"),
    )
    decision = WorkDecision(
        task=WorkSpec(
            objective="build a web scraper",
            success_condition="tests pass",
            adapter="coding",
            artifact="codebase",
        )
    )

    requests = PlanExpansionBuilder(parent).build_from_decision(decision)

    assert len(requests) == 1
    assert requests[0].agent_type == AgentType.WORK
