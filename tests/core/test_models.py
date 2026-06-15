"""Tests for core Pydantic models: DAGNode, SchedulerState, and AgentRequest immutability."""

from uuid import UUID

import pytest

from forge.core.models import (
    AcceptanceCriterion,
    AgentContract,
    AgentMessageKind,
    AgentRequest,
    AgentResponse,
    AgentType,
    CriticDisposition,
    CriticFinding,
    DAGNode,
    FailureKind,
    FileContent,
    FileView,
    FinalTurn,
    NodeState,
    PlanResponse,
    PlanSpec,
    RefereeDecision,
    RequestSource,
    ResponseStatus,
    SchedulerState,
    StateView,
    TaskSpec,
    ToolCallResponse,
    ToolTurn,
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
        dependencies=["requests"],
    )
    assert view.files == files


# --- PlanResponse ---


def test_plan_response_kind_discriminator():
    """PlanResponse requires kind to be 'plan'."""
    pr = PlanResponse(
        kind=AgentMessageKind.PLAN,
        tasks=[],
    )
    assert pr.kind == "plan"


# --- TaskSpec ---


def test_task_spec_defaults_depends_on_to_empty_list():
    """TaskSpec.depends_on defaults to an empty list."""
    ts = TaskSpec(
        objective="write tests",
        success_condition="tests pass",
        adapter="coding",
        artifact="codebase",
        language=None,
    )
    assert ts.depends_on == []


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


# --- FileContent ---


def test_file_content_stores_path_and_content():
    """FileContent stores path and content correctly."""
    fc = FileContent(path="src/main.py", content="x = 1")
    assert fc.path == "src/main.py"
    assert fc.content == "x = 1"


# --- WorkOutput ---


def test_work_output_defaults_to_empty_summary():
    """WorkOutput defaults to empty completion metadata."""
    wo = WorkOutput()
    assert wo.summary == ""


def test_work_output_base_version_defaults_to_empty_string():
    """WorkOutput.base_version defaults to empty string."""
    wo = WorkOutput()
    assert wo.base_version == ""


def test_work_output_is_frozen():
    """WorkOutput raises on direct field mutation."""
    wo = WorkOutput()
    with pytest.raises(Exception):
        wo.base_version = "abc123"  # type: ignore[misc]


def test_work_output_ignores_legacy_files_and_dependencies():
    """WorkOutput no longer transports file contents or dependencies."""
    fc = FileContent(path="src/lib.py", content="def f(): pass")
    wo = WorkOutput.model_validate(
        {"files": [fc.model_dump()], "dependencies": ["requests"], "base_version": "deadbeef"}
    )
    assert not hasattr(wo, "files")
    assert not hasattr(wo, "dependencies")
    assert wo.summary == "Completed worktree changes."
    assert wo.base_version == "deadbeef"


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
        output=WorkOutput(summary="done", base_version="abc123"),
    )
    assert ft.kind == "final"
    assert isinstance(ft.output, WorkOutput)
    assert ft.output.summary == "done"
    assert ft.output.base_version == "abc123"


def test_final_turn_accepts_plan_response():
    """FinalTurn.output holds a PlanResponse when the nested kind is 'plan'."""
    ft = FinalTurn(output=PlanResponse(tasks=[]))
    assert ft.kind == "final"
    assert isinstance(ft.output, PlanResponse)
    assert ft.output.tasks == []


def test_final_turn_rejects_unknown_output_kind():
    """FinalTurn raises ValidationError when the nested output kind is unrecognized."""
    with pytest.raises(Exception):
        FinalTurn.model_validate({"kind": "final", "output": {"kind": "unknown_type"}})


def test_final_turn_roundtrip_work_output():
    """FinalTurn with WorkOutput serializes and deserializes back to an equivalent object."""
    ft = FinalTurn(output=WorkOutput(summary="edits applied", base_version="sha1"))
    data = ft.model_dump()
    restored = FinalTurn.model_validate(data)
    assert isinstance(restored.output, WorkOutput)
    assert restored.output.summary == "edits applied"


def test_final_turn_roundtrip_plan_response():
    """FinalTurn with PlanResponse serializes and deserializes back to an equivalent object."""
    task = TaskSpec(
        objective="write tests",
        success_condition="tests pass",
        adapter="coding",
        artifact="codebase",
    )
    ft = FinalTurn(output=PlanResponse(tasks=[task]))
    data = ft.model_dump()
    restored = FinalTurn.model_validate(data)
    assert isinstance(restored.output, PlanResponse)
    assert len(restored.output.tasks) == 1
