"""Tests for core Pydantic models: DAGNode, SchedulerState, and AgentRequest immutability."""

from uuid import UUID

import pytest
from pydantic import TypeAdapter

import forge.core.models as _models
from forge.core.models import (
    AcceptanceCriterion,
    AgentContract,
    AgentMessageKind,
    AgentRequest,
    AgentResponse,
    AgentType,
    ChildTask,
    CriticDisposition,
    CriticFinding,
    DAGNode,
    DecompositionDecision,
    DecompositionNodeSpec,
    DecompositionTask,
    FailureKind,
    FileView,
    FinalTurn,
    GraphSplitDecision,
    NodeState,
    PlanSpec,
    RefereeDecision,
    RequestSource,
    ResponseStatus,
    RevisionItem,
    RevisionRequest,
    SchedulerState,
    StateView,
    TaskSpec,
    ToolCallResponse,
    ToolTurn,
    WorkDecision,
    WorkOutput,
    WorkSpec,
    render_agent_contract,
)

# --- IntegrateSpec removed ---


def test_integrate_spec_no_longer_exists():
    """IntegrateSpec and AgentType.INTEGRATE have been removed from the public API."""
    import forge.core.models as m

    assert not hasattr(m, "IntegrateSpec")
    assert not hasattr(AgentType, "INTEGRATE")


# --- integration_revision removed ---


def test_agent_request_has_no_integration_revision_field():
    """AgentRequest no longer exposes integration_revision after cleanup."""
    req = _make_request()
    assert not hasattr(req, "integration_revision")


def test_dag_node_has_no_integration_revision_field():
    """DAGNode no longer exposes integration_revision after cleanup."""
    node = _make_node()
    assert not hasattr(node, "integration_revision")


def test_agent_request_tolerates_extra_integration_revision_in_json():
    """Old persisted JSON with integration_revision is accepted — Pydantic ignores extra fields."""
    req = _make_request()
    data = req.model_dump()
    data["integration_revision"] = {
        "disposition": "revise",
        "rationale": "old",
        "items": [],
        "prior_attempts": 1,
    }
    restored = AgentRequest.model_validate(data)
    assert not hasattr(restored, "integration_revision")


def test_agent_request_initial_revision_defaults_to_none():
    """AgentRequest.initial_revision is absent by default for backward compatibility."""
    req = _make_request()

    assert req.initial_revision is None


def test_agent_request_old_json_without_initial_revision_loads_as_none():
    """Old serialized AgentRequest payloads without initial_revision load with None."""
    req = _make_request()
    data = req.model_dump()
    del data["initial_revision"]

    restored = AgentRequest.model_validate(data)

    assert restored.initial_revision is None


def test_agent_request_initial_revision_roundtrips():
    """AgentRequest.initial_revision serializes and deserializes as active retry feedback."""
    revision = RevisionRequest(
        rationale="tests failed",
        prior_attempts=1,
        items=[RevisionItem(required_change="fix tests", rationale="pytest failed")],
    )
    req = _make_request().model_copy(update={"initial_revision": revision})

    restored = AgentRequest.model_validate(req.model_dump())

    assert restored.initial_revision == revision


def test_dag_node_tolerates_extra_integration_revision_in_json():
    """Old persisted JSON with integration_revision in DAGNode is accepted — extra field ignored."""
    node = _make_node()
    data = node.model_dump()
    data["integration_revision"] = {
        "disposition": "revise",
        "rationale": "old",
        "items": [],
        "prior_attempts": 1,
    }
    restored = DAGNode.model_validate(data)
    assert not hasattr(restored, "integration_revision")


def _make_request(
    *,
    dependencies: frozenset[UUID] | None = None,
) -> AgentRequest:
    return AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.USER,
        spec=WorkSpec(
            objective="do something",
            success_condition="it is done",
            adapter="coding",
            artifact="codebase",
        ),
        dependencies=dependencies if dependencies is not None else frozenset(),
    )


def _make_node(
    *,
    node_state: NodeState = NodeState.PENDING,
    dependencies: frozenset[UUID] | None = None,
) -> DAGNode:
    return DAGNode(request=_make_request(dependencies=dependencies), node_state=node_state)


def _make_response(
    request_id: UUID,
    *,
    status: ResponseStatus = ResponseStatus.COMPLETED,
) -> AgentResponse:
    return AgentResponse(request_id=request_id, status=status)


def _make_state(nodes: list[DAGNode]) -> SchedulerState:
    return SchedulerState(dag={n.request.id: n for n in nodes}, northstar="test")


# --- DAGNode.with_state ---


def test_with_state_returns_new_instance_with_updated_state():
    """with_state() returns a new DAGNode with the given state without mutating the original."""
    node = _make_node()
    updated = node.with_state(NodeState.RUNNING)
    assert updated.node_state == NodeState.RUNNING
    assert node.node_state == NodeState.PENDING


# --- DAGNode.with_response ---


def test_with_response_sets_integrated_for_completed_status():
    """with_response() sets node_state to INTEGRATED when response status is COMPLETED."""
    node = _make_node()
    response = _make_response(node.request.id, status=ResponseStatus.COMPLETED)
    updated = node.with_response(response)
    assert updated.node_state == NodeState.INTEGRATED
    assert updated.response is response


def test_with_response_sets_failed_for_failed_status():
    """with_response() sets node_state to FAILED when response status is FAILED."""
    node = _make_node()
    response = _make_response(node.request.id, status=ResponseStatus.FAILED)
    updated = node.with_response(response)
    assert updated.node_state == NodeState.FAILED
    assert updated.response is response


def test_with_response_sets_cancelled_for_decompose_status():
    """with_response() sets node_state to CANCELLED when response status is DECOMPOSE."""
    node = _make_node()
    response = _make_response(node.request.id, status=ResponseStatus.DECOMPOSE)
    updated = node.with_response(response)
    assert updated.node_state == NodeState.CANCELLED
    assert updated.response is response


# --- SchedulerState.ready_nodes ---


def test_ready_nodes_returns_pending_node_with_all_deps_integrated():
    """ready_nodes() includes a PENDING node whose only dependency is INTEGRATED."""
    dep = _make_node(node_state=NodeState.INTEGRATED)
    candidate = _make_node(dependencies=frozenset({dep.request.id}))
    state = _make_state([dep, candidate])
    assert candidate in state.ready_nodes()


def test_ready_nodes_excludes_pending_node_with_unsatisfied_dep():
    """ready_nodes() excludes a PENDING node whose dependency is not yet COMPLETED."""
    dep = _make_node(node_state=NodeState.PENDING)
    blocked = _make_node(dependencies=frozenset({dep.request.id}))
    state = _make_state([dep, blocked])
    assert blocked not in state.ready_nodes()


def test_ready_nodes_excludes_running_and_integrated_nodes():
    """ready_nodes() excludes nodes that are already RUNNING or INTEGRATED."""
    running = _make_node(node_state=NodeState.RUNNING)
    integrated = _make_node(node_state=NodeState.INTEGRATED)
    state = _make_state([running, integrated])
    result = state.ready_nodes()
    assert running not in result
    assert integrated not in result


# --- SchedulerState.add_nodes ---


def test_add_nodes_returns_new_instance_original_unchanged():
    """add_nodes() returns a new state containing the new node without mutating the original."""
    state = SchedulerState(northstar="ns")
    node = _make_node()
    new_state = state.add_nodes([node])
    assert node.request.id in new_state.dag
    assert node.request.id not in state.dag


# --- SchedulerState.update_node ---


def test_update_node_returns_new_instance_original_unchanged():
    """update_node() returns a new state with the updated node without mutating the original."""
    node = _make_node()
    state = _make_state([node])
    updated = node.with_state(NodeState.RUNNING)
    new_state = state.update_node(updated)
    assert new_state.dag[node.request.id].node_state == NodeState.RUNNING
    assert state.dag[node.request.id].node_state == NodeState.PENDING


# --- AgentRequest immutability ---


def test_agent_request_raises_on_mutation():
    """AgentRequest raises an exception when any field is mutated directly."""
    request = _make_request()
    with pytest.raises(Exception):
        request.agent_type = AgentType.PLAN  # type: ignore[misc]


# --- WorkSpec.artifact ---


def test_work_spec_requires_artifact_field():
    """WorkSpec raises ValidationError when artifact field is missing."""
    with pytest.raises(Exception):
        WorkSpec(objective="do something", success_condition="it is done", adapter="coding")  # type: ignore[call-arg]


def test_work_spec_with_artifact_serializes_correctly():
    """WorkSpec with artifact field round-trips through model_dump and model_validate."""
    spec = WorkSpec(
        objective="write parser",
        success_condition="parser passes tests",
        adapter="coding",
        artifact="codebase",
    )
    data = spec.model_dump()
    restored = WorkSpec.model_validate(data)
    assert restored.artifact == "codebase"


def test_render_agent_contract_includes_contract_and_routing_fields() -> None:
    """The canonical contract block includes contract fields plus work routing fields."""
    request = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="write parser",
            success_condition="tests pass",
            contract=AgentContract(
                objective="write parser",
                success_condition="tests pass",
                acceptance_criteria=[AcceptanceCriterion(id="AC1", text="parses valid input")],
                constraints=["use stdlib"],
                non_goals=["network fetching"],
            ),
            adapter="coding",
            artifact="codebase",
            language="python",
        ),
    )

    block = render_agent_contract(request)

    assert "AgentRequest contract:" in block
    assert "Objective: write parser" in block
    assert "Success condition: tests pass" in block
    assert "- AC1: parses valid input" in block
    assert "- use stdlib" in block
    assert "- network fetching" in block
    assert "Artifact: codebase" in block
    assert "Adapter: coding" in block
    assert "Language: python" in block


def test_work_spec_has_no_target_entity_field():
    """WorkSpec no longer exposes target_entity."""
    assert not hasattr(WorkSpec, "target_entity")


def test_plan_spec_has_no_goal_field():
    """PlanSpec no longer exposes goal; northstar is the only goal-carrying field."""
    assert not hasattr(PlanSpec, "goal")


# --- StateView ---


def test_state_view_stores_files_as_file_views():
    """StateView accepts a list of FileView objects for the files field."""
    files = [
        FileView(path="src/main.py", content="x = 1"),
        FileView(path="src/utils.py", content="# utils"),
    ]
    view = StateView(
        artifact_name="myapp",
        language="python",
        files=files,
    )
    assert view.files == files


def test_state_view_has_no_dependencies_field():
    """StateView.dependencies and StateView.test_summary have been removed."""
    view = StateView(artifact_name="myapp", language=None, files=[])
    assert not hasattr(view, "dependencies")
    assert not hasattr(view, "test_summary")


# --- TaskSpec ---


def test_task_spec_has_no_depends_on_field():
    """TaskSpec.depends_on has been removed — graph expansion uses string node IDs."""
    ts = TaskSpec(
        objective="write tests",
        success_condition="tests pass",
        adapter="coding",
        artifact="codebase",
        language=None,
    )
    assert not hasattr(ts, "depends_on")


# --- ToolCallResponse ---


def test_tool_call_response_kind_discriminator():
    """ToolCallResponse requires kind to be 'tool_response'."""
    resp = ToolCallResponse(
        kind=AgentMessageKind.TOOL_RESPONSE,
        name="read_file",
        success=True,
        result="content",
    )
    assert resp.kind == "tool_response"


def test_tool_call_response_error_defaults_to_none():
    """ToolCallResponse.error defaults to None when not provided."""
    resp = ToolCallResponse(
        kind=AgentMessageKind.TOOL_RESPONSE,
        name="read_file",
        success=True,
        result="ok",
    )
    assert resp.error is None


# --- FailureKind ---


def test_failure_kind_has_expected_values():
    """FailureKind enum contains exactly the active classification values."""
    values = {k.value for k in FailureKind}
    assert values == {
        "invalid_json",
        "provider_error",
        "timeout",
        "max_iterations",
        "tool_error",
        "stale_work_output",
        "integration_failed",
        "test_failed",
        "validation_rejected",
        "internal_error",
        "unknown",
    }
    assert not hasattr(FailureKind, "TRUNCATED_OUTPUT")


def test_agent_response_with_failure_kind_serializes_correctly():
    """AgentResponse with failure_kind round-trips through model_dump and model_validate."""
    request = _make_request()
    response = AgentResponse(
        request_id=request.id,
        status=ResponseStatus.FAILED,
        error="something went wrong",
        failure_kind=FailureKind.INVALID_JSON,
    )
    data = response.model_dump()
    restored = AgentResponse.model_validate(data)
    assert restored.failure_kind == FailureKind.INVALID_JSON


def test_agent_response_failure_kind_none_when_completed():
    """AgentResponse.failure_kind defaults to None when status is COMPLETED."""
    request = _make_request()
    response = AgentResponse(request_id=request.id, status=ResponseStatus.COMPLETED)
    assert response.failure_kind is None


# --- CriticDisposition ---


def test_critic_disposition_has_exactly_five_values():
    """CriticDisposition has exactly ACCEPT, REVISE, REJECT, ALREADY_DONE, and DECOMPOSE."""
    assert {d.name for d in CriticDisposition} == {
        "ACCEPT",
        "REVISE",
        "REJECT",
        "ALREADY_DONE",
        "DECOMPOSE",
    }


def test_critic_disposition_decompose_value():
    """CriticDisposition.DECOMPOSE has the string value 'decompose'."""
    assert CriticDisposition.DECOMPOSE.value == "decompose"


# --- CriticFinding ---


def test_critic_finding_is_frozen():
    """CriticFinding raises on direct field mutation."""
    finding = CriticFinding(disposition=CriticDisposition.ACCEPT, rationale="looks good")
    with pytest.raises(Exception):
        finding.rationale = "changed"  # type: ignore[misc]


def test_critic_finding_hints_defaults_to_empty_list():
    """CriticFinding.hints defaults to an empty list."""
    finding = CriticFinding(disposition=CriticDisposition.ACCEPT, rationale="looks good")
    assert finding.hints == []


def test_critic_finding_with_hints():
    """CriticFinding stores hints when provided."""
    finding = CriticFinding(
        disposition=CriticDisposition.REVISE,
        rationale="close but needs work",
        hints=["fix the edge case", "add a docstring"],
    )
    assert len(finding.hints) == 2


# --- RefereeDecision ---


def test_referee_decision_is_frozen():
    """RefereeDecision raises on direct field mutation."""
    decision = RefereeDecision(
        disposition=CriticDisposition.ACCEPT, rationale="agreed", override=False
    )
    with pytest.raises(Exception):
        decision.override = True  # type: ignore[misc]


def test_referee_decision_override_true_when_overriding_critic():
    """RefereeDecision.override is True when the referee disagrees with the critic."""
    critic = CriticFinding(
        disposition=CriticDisposition.REJECT, rationale="not good enough", hints=["redo it"]
    )
    decision = RefereeDecision(
        disposition=CriticDisposition.ACCEPT,
        rationale="actually meets the bar",
        override=critic.disposition != CriticDisposition.ACCEPT,
    )
    assert decision.override is True


# --- FileContent removed ---


def test_file_content_no_longer_exists():
    """FileContent has been removed — it was unused after the worktree protocol replaced it."""
    import forge.core.models as m

    assert not hasattr(m, "FileContent")


# --- WorkOutput ---


def test_work_output_defaults_to_empty_summary():
    """WorkOutput defaults to empty completion metadata."""
    wo = WorkOutput()
    assert wo.summary == ""


def test_work_output_ignores_model_supplied_base_version():
    """WorkOutput silently drops base_version from model JSON — framework owns git metadata."""
    wo = WorkOutput.model_validate({"kind": "work_output", "summary": "done", "base_version": "0"})
    assert not hasattr(wo, "base_version")
    assert wo.summary == "done"


def test_work_output_is_frozen():
    """WorkOutput raises on direct field mutation."""
    wo = WorkOutput()
    with pytest.raises(Exception):
        wo.summary = "mutated"  # type: ignore[misc]


def test_work_output_rejects_legacy_files_payload():
    """WorkOutput raises ValueError when a 'files' key is present in the payload."""
    with pytest.raises(Exception, match="files"):
        WorkOutput.model_validate({"files": [{"path": "src/lib.py", "content": "x"}]})


def test_work_output_rejects_legacy_dependencies_payload():
    """WorkOutput raises ValueError when a 'dependencies' key is present in the payload."""
    with pytest.raises(Exception, match="dependencies"):
        WorkOutput.model_validate({"dependencies": ["requests"]})


# --- ToolTurn ---


def test_tool_turn_roundtrip():
    """ToolTurn serializes to JSON and deserializes back with identical field values."""
    turn = ToolTurn(name="write_file", arguments={"path": "src/main.py", "content": "x=1"})
    data = turn.model_dump()
    restored = ToolTurn.model_validate(data)
    assert restored.kind == "tool"
    assert restored.name == "write_file"
    assert restored.arguments == {"path": "src/main.py", "content": "x=1"}


def test_tool_turn_arguments_defaults_to_empty_dict():
    """ToolTurn.arguments defaults to {} when not provided."""
    turn = ToolTurn(name="run_tests")
    assert turn.arguments == {}


def test_tool_turn_kind_is_literal_tool():
    """ToolTurn.kind is always 'tool' and cannot be overridden to another value."""
    turn = ToolTurn(name="read_file")
    assert turn.kind == "tool"


def test_tool_turn_is_frozen():
    """ToolTurn raises on direct field mutation."""
    turn = ToolTurn(name="read_file")
    with pytest.raises(Exception):
        turn.name = "write_file"  # type: ignore[misc]


# --- FinalTurn ---


def test_final_turn_accepts_work_output():
    """FinalTurn.output holds a WorkOutput when the nested kind is 'work_output'."""
    ft = FinalTurn(
        output=WorkOutput(summary="done"),
    )
    assert ft.kind == "final"
    assert isinstance(ft.output, WorkOutput)
    assert ft.output.summary == "done"


def test_final_turn_rejects_unknown_output_kind():
    """FinalTurn raises ValidationError when the nested output kind is unrecognized."""
    with pytest.raises(Exception):
        FinalTurn.model_validate({"kind": "final", "output": {"kind": "unknown_type"}})


def test_final_turn_roundtrip_work_output():
    """FinalTurn with WorkOutput serializes and deserializes back to an equivalent object."""
    ft = FinalTurn(output=WorkOutput(summary="edits applied"))
    data = ft.model_dump()
    restored = FinalTurn.model_validate(data)
    assert isinstance(restored.output, WorkOutput)
    assert restored.output.summary == "edits applied"


# --- DecompositionDecision ---


def _make_work_spec() -> WorkSpec:
    return WorkSpec(
        objective="implement parser",
        success_condition="parser passes tests",
        adapter="coding",
        artifact="codebase",
    )


def test_work_decision_validates_and_serializes():
    """WorkDecision round-trips through model_dump and model_validate."""
    decision = WorkDecision(task=_make_work_spec())
    data = decision.model_dump()
    restored = WorkDecision.model_validate(data)
    assert restored.kind == "work"
    assert restored.task.objective == "implement parser"


def test_decomposition_decision_discriminates_work():
    """TypeAdapter resolves kind='work' to WorkDecision."""
    ta: TypeAdapter[DecompositionDecision] = TypeAdapter(DecompositionDecision)
    result = ta.validate_python({"kind": "work", "task": _make_work_spec().model_dump()})
    assert isinstance(result, WorkDecision)


def test_decomposition_decision_rejects_unknown_kind():
    """TypeAdapter raises ValidationError for an unrecognized kind value."""
    ta: TypeAdapter[DecompositionDecision] = TypeAdapter(DecompositionDecision)
    with pytest.raises(Exception):
        ta.validate_python({"kind": "unknown_kind", "tasks": []})


# --- DecompositionTask ---


def test_decomposition_task_validates_and_serializes():
    """DecompositionTask round-trips through model_dump and model_validate."""
    task = DecompositionTask(objective="plan the sub-system", success_condition="planned")
    data = task.model_dump()
    restored = DecompositionTask.model_validate(data)
    assert restored.kind == "decomposition_task"
    assert restored.objective == "plan the sub-system"
    assert restored.success_condition == "planned"


def test_decomposition_task_kind_is_decomposition_task():
    """DecompositionTask.kind defaults to 'decomposition_task'."""
    task = DecompositionTask(objective="x", success_condition="y")
    assert task.kind == "decomposition_task"


def test_task_spec_kind_defaults_to_work_task():
    """TaskSpec.kind defaults to 'work_task' for backward compatibility."""
    ts = TaskSpec(
        objective="write tests",
        success_condition="tests pass",
        adapter="coding",
        artifact="codebase",
    )
    assert ts.kind == "work_task"


# --- ChildTask discriminated union ---


def test_child_task_discriminates_work_task():
    """TypeAdapter resolves kind='work_task' to TaskSpec."""
    ta: TypeAdapter[ChildTask] = TypeAdapter(ChildTask)
    result = ta.validate_python(
        {
            "kind": "work_task",
            "objective": "write tests",
            "success_condition": "tests pass",
            "adapter": "coding",
            "artifact": "codebase",
        }
    )
    assert isinstance(result, TaskSpec)


def test_child_task_discriminates_decomposition_task():
    """TypeAdapter resolves kind='decomposition_task' to DecompositionTask."""
    ta: TypeAdapter[ChildTask] = TypeAdapter(ChildTask)
    result = ta.validate_python(
        {"kind": "decomposition_task", "objective": "plan sub", "success_condition": "planned"}
    )
    assert isinstance(result, DecompositionTask)


def test_child_task_rejects_unknown_kind():
    """TypeAdapter raises ValidationError for an unrecognized kind value."""
    ta: TypeAdapter[ChildTask] = TypeAdapter(ChildTask)
    with pytest.raises(Exception):
        ta.validate_python({"kind": "unknown_kind"})


# --- Legacy type removal ---


def test_legacy_decomposition_types_no_longer_exist():
    """PlanResponse, DependentSplitDecision, OrthogonalSplitDecision no longer exist in models."""
    assert not hasattr(_models, "PlanResponse")
    assert not hasattr(_models, "DependentSplitDecision")
    assert not hasattr(_models, "OrthogonalSplitDecision")


# --- GraphSplitDecision ---


def _make_graph_node(
    node_id: str, objective: str = "task", depends_on: list[str] | None = None
) -> DecompositionNodeSpec:
    return DecompositionNodeSpec(
        id=node_id,
        task=TaskSpec(
            objective=objective,
            success_condition="done",
            adapter="coding",
            artifact="codebase",
        ),
        depends_on=depends_on or [],
    )


def test_graph_split_decision_validates_and_serializes():
    """GraphSplitDecision round-trips through model_dump and model_validate."""
    decision = GraphSplitDecision(nodes=[_make_graph_node("a"), _make_graph_node("b")])
    data = decision.model_dump()
    restored = GraphSplitDecision.model_validate(data)
    assert restored.kind == "split_graph"
    assert len(restored.nodes) == 2
    assert restored.nodes[0].id == "a"
    assert restored.nodes[1].id == "b"


def test_graph_split_decision_with_dependencies():
    """GraphSplitDecision preserves depends_on references after round-trip."""
    decision = GraphSplitDecision(
        nodes=[
            _make_graph_node("setup"),
            _make_graph_node("scraper", depends_on=["setup"]),
            _make_graph_node("cli", depends_on=["scraper"]),
        ]
    )
    data = decision.model_dump()
    restored = GraphSplitDecision.model_validate(data)
    assert restored.nodes[0].depends_on == []
    assert restored.nodes[1].depends_on == ["setup"]
    assert restored.nodes[2].depends_on == ["scraper"]


def test_graph_split_decision_rejects_unknown_depends_on_ref():
    """GraphSplitDecision raises when depends_on references a non-existent node id."""
    with pytest.raises(Exception, match="not a known node id"):
        GraphSplitDecision(
            nodes=[
                _make_graph_node("a"),
                _make_graph_node("b", depends_on=["nonexistent"]),
            ]
        )


def test_graph_split_decision_rejects_self_loop():
    """GraphSplitDecision raises when a node depends on itself."""
    with pytest.raises(Exception, match="depends on itself"):
        GraphSplitDecision(nodes=[_make_graph_node("a", depends_on=["a"])])


def test_graph_split_decision_rejects_empty_nodes():
    """GraphSplitDecision raises when nodes list is empty."""
    with pytest.raises(Exception):
        GraphSplitDecision(nodes=[])


def test_decomposition_decision_discriminates_split_graph():
    """TypeAdapter resolves kind='split_graph' to GraphSplitDecision."""
    ta: TypeAdapter[DecompositionDecision] = TypeAdapter(DecompositionDecision)
    result = ta.validate_python(
        {
            "kind": "split_graph",
            "nodes": [
                {
                    "id": "a",
                    "task": {
                        "objective": "do a",
                        "success_condition": "done",
                        "adapter": "coding",
                        "artifact": "codebase",
                    },
                    "depends_on": [],
                }
            ],
        }
    )
    assert isinstance(result, GraphSplitDecision)


def test_decomposition_node_spec_defaults_task_kind_to_work_task():
    """DecompositionNodeSpec inserts kind='work_task' when task dict omits it."""
    node = DecompositionNodeSpec.model_validate(
        {
            "id": "x",
            "task": {
                "objective": "do x",
                "success_condition": "done",
                "adapter": "coding",
                "artifact": "codebase",
            },
        }
    )
    assert isinstance(node.task, TaskSpec)


def test_final_turn_accepts_graph_split_decision():
    """FinalTurn.output holds a GraphSplitDecision when nested kind is 'split_graph'."""
    decision = GraphSplitDecision(nodes=[_make_graph_node("a")])
    ft = FinalTurn(output=decision)
    assert ft.kind == "final"
    assert isinstance(ft.output, GraphSplitDecision)


def test_final_turn_roundtrip_graph_split_decision():
    """FinalTurn with GraphSplitDecision serializes and deserializes back equivalently."""
    decision = GraphSplitDecision(
        nodes=[
            _make_graph_node("setup"),
            _make_graph_node("impl", depends_on=["setup"]),
        ]
    )
    ft = FinalTurn(output=decision)
    data = ft.model_dump()
    restored = FinalTurn.model_validate(data)
    assert isinstance(restored.output, GraphSplitDecision)
    assert restored.output.nodes[1].depends_on == ["setup"]


# --- AgentRequest.model_profile ---


def test_agent_request_defaults_model_profile_to_default():
    """AgentRequest without explicit model_profile has model_profile == 'default'."""
    assert _make_request().model_profile == "default"


def test_agent_request_explicit_model_profile_is_preserved():
    """AgentRequest with explicit model_profile stores and returns that value."""
    req = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="do something",
            success_condition="it is done",
            adapter="coding",
            artifact="codebase",
        ),
        model_profile="fast",
    )
    assert req.model_profile == "fast"


def test_agent_request_model_profile_roundtrip():
    """AgentRequest with model_profile='fast' serializes and deserializes back to 'fast'."""
    req = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.USER,
        spec=WorkSpec(
            objective="do something",
            success_condition="it is done",
            adapter="coding",
            artifact="codebase",
        ),
        model_profile="fast",
    )
    data = req.model_dump()
    restored = AgentRequest.model_validate(data)
    assert restored.model_profile == "fast"


def test_agent_request_old_json_without_model_profile_loads_as_default():
    """Old persisted JSON without model_profile deserializes with model_profile == 'default'."""
    req = _make_request()
    data = req.model_dump()
    del data["model_profile"]
    restored = AgentRequest.model_validate(data)
    assert restored.model_profile == "default"


def test_dag_node_carries_request_model_profile_unchanged():
    """DAGNode.request.model_profile is accessible and reflects what was set on the request."""
    req = AgentRequest(
        agent_type=AgentType.WORK,
        source=RequestSource.PLANNER,
        spec=WorkSpec(
            objective="do something",
            success_condition="it is done",
            adapter="coding",
            artifact="codebase",
        ),
        model_profile="fast",
    )
    node = DAGNode(request=req)
    assert node.request.model_profile == "fast"
