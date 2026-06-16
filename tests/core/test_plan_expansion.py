"""Tests for scheduler-owned plan expansion."""

import pytest

from forge.core.models import (
    AcceptanceCriterion,
    AgentRequest,
    AgentType,
    DecompositionNodeSpec,
    DecompositionTask,
    GraphSplitDecision,
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
from forge.core.profile_assignment import DefaultProfileAssigner


class FakeProfileAssigner:
    """Test double that records assigned requests and returns a fixed profile."""

    def __init__(self, profile: str) -> None:
        self.profile = profile
        self.requests: list[AgentRequest] = []

    async def assign(self, request: AgentRequest) -> str:
        self.requests.append(request)
        return self.profile


def _plan_request() -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(northstar="build a scraper"),
    )


# --- WorkDecision expansion ---


def _make_task_spec(objective: str = "task") -> TaskSpec:
    return TaskSpec(
        objective=objective,
        success_condition="done",
        adapter="coding",
        artifact="codebase",
    )


def _make_decomposition_task(objective: str = "sub-plan") -> DecompositionTask:
    return DecompositionTask(objective=objective, success_condition="planned")


def _make_graph_node(
    node_id: str, objective: str = "task", depends_on: list[str] | None = None
) -> DecompositionNodeSpec:
    return DecompositionNodeSpec(
        id=node_id,
        task=_make_task_spec(objective),
        depends_on=depends_on or [],
    )


def _make_decomp_graph_node(
    node_id: str, objective: str = "sub-plan", depends_on: list[str] | None = None
) -> DecompositionNodeSpec:
    return DecompositionNodeSpec(
        id=node_id,
        task=_make_decomposition_task(objective),
        depends_on=depends_on or [],
    )


async def test_work_decision_expands_to_one_work_node() -> None:
    """WorkDecision produces exactly one WORK request carrying the provided WorkSpec."""
    work_spec = WorkSpec(
        objective="implement parser",
        success_condition="parser passes tests",
        adapter="coding",
        artifact="codebase",
    )
    decision = WorkDecision(task=work_spec)

    requests = await PlanExpansionBuilder(_plan_request()).build_from_decision(decision)

    assert len(requests) == 1
    assert requests[0].agent_type == AgentType.WORK
    assert requests[0].source == RequestSource.PLANNER
    assert requests[0].spec == work_spec
    assert requests[0].dependencies == frozenset()


async def test_default_profile_assigner_returns_default() -> None:
    """DefaultProfileAssigner always returns the default model profile."""
    assert await DefaultProfileAssigner().assign(_plan_request()) == "default"


async def test_work_decision_uses_default_profile_assigner_when_none_provided() -> None:
    """PlanExpansionBuilder preserves default profile behavior without injection."""
    work_spec = WorkSpec(
        objective="implement parser",
        success_condition="parser passes tests",
        adapter="coding",
        artifact="codebase",
    )
    decision = WorkDecision(task=work_spec)

    requests = await PlanExpansionBuilder(_plan_request()).build_from_decision(decision)

    assert requests[0].model_profile == "default"


async def test_work_decision_uses_injected_profile_assigner_for_work_requests() -> None:
    """Injected profile assigners set model_profile on generated WORK requests."""
    assigner = FakeProfileAssigner("fast")
    decision = WorkDecision(
        task=WorkSpec(
            objective="implement parser",
            success_condition="parser passes tests",
            adapter="coding",
            artifact="codebase",
        )
    )

    requests = await PlanExpansionBuilder(
        _plan_request(), profile_assigner=assigner
    ).build_from_decision(decision)

    assert len(assigner.requests) == 1
    assert assigner.requests[0].agent_type == AgentType.WORK
    assert assigner.requests[0].model_profile == "default"
    assert requests[0].model_profile == "fast"


# --- GraphSplitDecision expansion ---


async def test_graph_split_no_edges_expands_to_independent_nodes() -> None:
    """GraphSplitDecision with all depends_on [] produces nodes with empty dependencies."""
    decision = GraphSplitDecision(
        nodes=[
            _make_graph_node("a", "task-a"),
            _make_graph_node("b", "task-b"),
            _make_graph_node("c", "task-c"),
        ]
    )

    requests = await PlanExpansionBuilder(_plan_request()).build_from_decision(decision)

    assert len(requests) == 3
    assert all(r.agent_type == AgentType.WORK for r in requests)
    assert all(r.dependencies == frozenset() for r in requests)


async def test_graph_split_chain_decision_expands_to_chained_work_nodes() -> None:
    """GraphSplitDecision a→b→c produces three WORK nodes chained a->b->c."""
    decision = GraphSplitDecision(
        nodes=[
            _make_graph_node("a", "task-a"),
            _make_graph_node("b", "task-b", depends_on=["a"]),
            _make_graph_node("c", "task-c", depends_on=["b"]),
        ]
    )

    requests = await PlanExpansionBuilder(_plan_request()).build_from_decision(decision)
    node_a, node_b, node_c = requests

    assert len(requests) == 3
    assert all(r.agent_type == AgentType.WORK for r in requests)
    assert node_a.dependencies == frozenset()
    assert node_a.id in node_b.dependencies
    assert len(node_b.dependencies) == 1
    assert node_b.id in node_c.dependencies
    assert len(node_c.dependencies) == 1


async def test_graph_split_mixed_topology_exposes_concurrency() -> None:
    """GraphSplitDecision: setup, docs run immediately; scraper after setup; cli after scraper."""
    decision = GraphSplitDecision(
        nodes=[
            _make_graph_node("setup", "setup environment"),
            _make_graph_node("docs", "write docs"),
            _make_graph_node("scraper", "implement scraper", depends_on=["setup"]),
            _make_graph_node("cli", "implement CLI", depends_on=["scraper"]),
        ]
    )

    requests = await PlanExpansionBuilder(_plan_request()).build_from_decision(decision)
    by_obj = {r.spec.objective: r for r in requests if isinstance(r.spec, WorkSpec)}

    assert by_obj["setup environment"].dependencies == frozenset()
    assert by_obj["write docs"].dependencies == frozenset()
    assert by_obj["setup environment"].id in by_obj["implement scraper"].dependencies
    assert len(by_obj["implement scraper"].dependencies) == 1
    assert by_obj["implement scraper"].id in by_obj["implement CLI"].dependencies
    assert len(by_obj["implement CLI"].dependencies) == 1


async def test_graph_split_multiple_parents() -> None:
    """GraphSplitDecision node with two depends_on parents gets both as dependencies."""
    decision = GraphSplitDecision(
        nodes=[
            _make_graph_node("a", "task-a"),
            _make_graph_node("b", "task-b"),
            _make_graph_node("c", "task-c", depends_on=["a", "b"]),
        ]
    )

    requests = await PlanExpansionBuilder(_plan_request()).build_from_decision(decision)
    node_a, node_b, node_c = requests

    assert node_a.dependencies == frozenset()
    assert node_b.dependencies == frozenset()
    assert node_a.id in node_c.dependencies
    assert node_b.id in node_c.dependencies
    assert len(node_c.dependencies) == 2


async def test_graph_split_propagates_artifact_and_language() -> None:
    """Task artifact and language are copied into the generated WorkSpec."""
    decision = GraphSplitDecision(
        nodes=[
            DecompositionNodeSpec(
                id="a",
                task=TaskSpec(
                    objective="write code",
                    success_condition="tests pass",
                    adapter="coding",
                    artifact="api",
                    language="python",
                ),
                depends_on=[],
            )
        ]
    )

    requests = await PlanExpansionBuilder(_plan_request()).build_from_decision(decision)

    assert isinstance(requests[0].spec, WorkSpec)
    assert requests[0].spec.artifact == "api"
    assert requests[0].spec.language == "python"


async def test_graph_split_preserves_contract_fields() -> None:
    """Planner-emitted contract fields are copied into the generated WorkSpec contract."""
    decision = GraphSplitDecision(
        nodes=[
            DecompositionNodeSpec(
                id="a",
                task=TaskSpec(
                    objective="write code",
                    success_condition="tests pass",
                    acceptance_criteria=[
                        AcceptanceCriterion(id="AC1", text="unit tests cover parser")
                    ],
                    constraints=["use stdlib"],
                    non_goals=["network UI"],
                    adapter="coding",
                    artifact="api",
                    language="python",
                ),
                depends_on=[],
            )
        ]
    )

    requests = await PlanExpansionBuilder(_plan_request()).build_from_decision(decision)

    assert isinstance(requests[0].spec, WorkSpec)
    assert requests[0].spec.contract.acceptance_criteria == [
        AcceptanceCriterion(id="AC1", text="unit tests cover parser")
    ]
    assert requests[0].spec.contract.constraints == ["use stdlib"]
    assert requests[0].spec.contract.non_goals == ["network UI"]


# --- DecompositionTask expansion ---


async def test_decomposition_task_inside_graph_split_expands_to_plan_node() -> None:
    """A DecompositionTask inside a GraphSplitDecision produces an AgentType.PLAN request."""
    decision = GraphSplitDecision(nodes=[_make_decomp_graph_node("sub", "plan the sub-system")])

    requests = await PlanExpansionBuilder(_plan_request()).build_from_decision(decision)

    assert len(requests) == 1
    assert requests[0].agent_type == AgentType.PLAN
    assert isinstance(requests[0].spec, PlanSpec)
    assert requests[0].spec.northstar == "plan the sub-system"
    assert requests[0].source == RequestSource.PLANNER
    assert requests[0].dependencies == frozenset()


async def test_graph_split_chain_mixed_children_creates_dependency_chain() -> None:
    """GraphSplitDecision chain with mixed work/decomposition children produces chained deps."""
    decision = GraphSplitDecision(
        nodes=[
            _make_graph_node("work-a", "work-a"),
            _make_decomp_graph_node("plan-b", "plan-b", depends_on=["work-a"]),
            _make_graph_node("work-c", "work-c", depends_on=["plan-b"]),
        ]
    )

    requests = await PlanExpansionBuilder(_plan_request()).build_from_decision(decision)
    node_a, node_b, node_c = requests

    assert node_a.agent_type == AgentType.WORK
    assert node_b.agent_type == AgentType.PLAN
    assert node_c.agent_type == AgentType.WORK
    assert node_a.dependencies == frozenset()
    assert node_a.id in node_b.dependencies
    assert len(node_b.dependencies) == 1
    assert node_b.id in node_c.dependencies
    assert len(node_c.dependencies) == 1


async def test_graph_split_no_edges_mixed_children_creates_no_sibling_dependencies() -> None:
    """GraphSplitDecision with no edges and mixed children produces no sibling dependencies."""
    decision = GraphSplitDecision(
        nodes=[
            _make_graph_node("work-a", "work-a"),
            _make_decomp_graph_node("plan-b", "plan-b"),
        ]
    )

    requests = await PlanExpansionBuilder(_plan_request()).build_from_decision(decision)

    assert len(requests) == 2
    assert requests[0].agent_type == AgentType.WORK
    assert requests[1].agent_type == AgentType.PLAN
    assert all(r.dependencies == frozenset() for r in requests)


async def test_graph_split_assigns_profiles_to_work_children_only() -> None:
    """Injected profile assignment is applied to WORK children, not PLAN children."""
    assigner = FakeProfileAssigner("fast")
    decision = GraphSplitDecision(
        nodes=[
            _make_graph_node("work-a", "work-a"),
            _make_decomp_graph_node("plan-b", "plan-b"),
        ]
    )

    requests = await PlanExpansionBuilder(
        _plan_request(), profile_assigner=assigner
    ).build_from_decision(decision)
    work_request, plan_request = requests

    assert work_request.agent_type == AgentType.WORK
    assert work_request.model_profile == "fast"
    assert plan_request.agent_type == AgentType.PLAN
    assert plan_request.model_profile == "default"
    assert len(assigner.requests) == 1
    assert assigner.requests[0].agent_type == AgentType.WORK


async def test_graph_split_chain_all_work_produces_only_work_nodes() -> None:
    """GraphSplitDecision chain with all TaskSpec nodes produces only WORK nodes."""
    decision = GraphSplitDecision(
        nodes=[
            _make_graph_node("a", "task-a"),
            _make_graph_node("b", "task-b", depends_on=["a"]),
        ]
    )

    requests = await PlanExpansionBuilder(_plan_request()).build_from_decision(decision)

    assert all(r.agent_type == AgentType.WORK for r in requests)
    assert len(requests) == 2


# --- DecompositionConvergenceValidator ---


def _make_validator() -> DecompositionConvergenceValidator:
    return DecompositionConvergenceValidator()


async def test_convergence_validator_accepts_narrower_children() -> None:
    """Valid decomposition with distinct, narrower child objectives passes."""
    decision = GraphSplitDecision(
        nodes=[
            _make_graph_node("a", "implement HTTP fetching"),
            _make_graph_node("b", "implement HTML parsing"),
            _make_graph_node("c", "add CLI interface"),
        ]
    )

    _make_validator().validate("implement web scraper", decision)  # must not raise


async def test_convergence_validator_rejects_child_identical_to_parent() -> None:
    """A child whose normalized objective matches the parent is rejected."""
    decision = GraphSplitDecision(
        nodes=[
            _make_graph_node("a", "build a web scraper"),
            _make_graph_node("b", "add CLI interface"),
        ]
    )

    with pytest.raises(DecompositionConvergenceError, match="not reductive"):
        _make_validator().validate("build a web scraper", decision)


async def test_convergence_validator_rejects_child_identical_to_parent_case_insensitive() -> None:
    """Normalization is case-insensitive — 'Build A Web Scraper' equals 'build a web scraper'."""
    decision = GraphSplitDecision(
        nodes=[
            _make_graph_node("a", "Build A Web Scraper"),
            _make_graph_node("b", "add CLI"),
        ]
    )

    with pytest.raises(DecompositionConvergenceError, match="not reductive"):
        _make_validator().validate("build a web scraper", decision)


async def test_convergence_validator_rejects_all_identical_children() -> None:
    """All children normalizing to the same objective is rejected."""
    decision = GraphSplitDecision(
        nodes=[
            _make_graph_node("a", "write comprehensive tests"),
            _make_graph_node("b", "Write Comprehensive Tests"),
        ]
    )

    with pytest.raises(DecompositionConvergenceError, match="not reductive"):
        _make_validator().validate("implement the system", decision)


async def test_convergence_validator_rejects_pairwise_identical_siblings() -> None:
    """Any duplicate sibling objective is rejected, even when other siblings are distinct."""
    decision = GraphSplitDecision(
        nodes=[
            _make_graph_node("a", "write comprehensive tests"),
            _make_graph_node("b", "implement parser"),
            _make_graph_node("c", " Write   Comprehensive   Tests "),
        ]
    )

    with pytest.raises(DecompositionConvergenceError, match="sibling objectives"):
        _make_validator().validate("implement the system", decision)


async def test_convergence_validator_rejects_empty_child_objective() -> None:
    """A child with an empty or near-empty objective is rejected."""
    decision = GraphSplitDecision(
        nodes=[
            _make_graph_node("a", "  "),
            _make_graph_node("b", "add CLI"),
        ]
    )

    with pytest.raises(DecompositionConvergenceError, match="near-empty"):
        _make_validator().validate("build a web scraper", decision)


async def test_convergence_validator_accepts_work_decision() -> None:
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


async def test_convergence_validator_single_child_not_identical_passes() -> None:
    """A single child that is distinct from the parent is accepted."""
    decision = GraphSplitDecision(nodes=[_make_graph_node("a", "implement HTTP fetching")])

    _make_validator().validate("implement web scraper", decision)  # must not raise


async def test_build_from_decision_raises_on_identical_child_to_parent() -> None:
    """build_from_decision raises DecompositionConvergenceError when a child repeats the parent."""
    parent = AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(northstar="build a web scraper"),
    )
    decision = GraphSplitDecision(
        nodes=[
            _make_graph_node("a", "build a web scraper"),
            _make_graph_node("b", "add CLI"),
        ]
    )

    with pytest.raises(DecompositionConvergenceError):
        await PlanExpansionBuilder(parent).build_from_decision(decision)


async def test_build_from_decision_work_decision_not_rejected_by_convergence() -> None:
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

    requests = await PlanExpansionBuilder(parent).build_from_decision(decision)

    assert len(requests) == 1
    assert requests[0].agent_type == AgentType.WORK
