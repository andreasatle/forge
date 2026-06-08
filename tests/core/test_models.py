"""Tests for core Pydantic models: DAGNode, SchedulerState, and AgentRequest immutability."""

import pytest

from forge.core.models import (
    AgentRequest,
    AgentResponse,
    AgentType,
    DAGNode,
    DeltaState,
    Edit,
    FailureKind,
    NodeState,
    PlanResponse,
    Priority,
    RequestSource,
    ResponseStatus,
    SchedulerState,
    StateView,
    TaskSpec,
    ToolCallRequest,
    ToolCallResponse,
    WorkSpec,
)


def _make_request(
    *,
    dependencies: frozenset | None = None,
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
    dependencies: frozenset | None = None,
) -> DAGNode:
    return DAGNode(request=_make_request(dependencies=dependencies), node_state=node_state)


def _make_response(request_id, *, status: ResponseStatus = ResponseStatus.COMPLETED) -> AgentResponse:
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


def test_with_response_sets_completed_for_completed_status():
    """with_response() sets node_state to COMPLETED when response status is COMPLETED."""
    node = _make_node()
    response = _make_response(node.request.id, status=ResponseStatus.COMPLETED)
    updated = node.with_response(response)
    assert updated.node_state == NodeState.COMPLETED
    assert updated.response is response


def test_with_response_sets_failed_for_failed_status():
    """with_response() sets node_state to FAILED when response status is FAILED."""
    node = _make_node()
    response = _make_response(node.request.id, status=ResponseStatus.FAILED)
    updated = node.with_response(response)
    assert updated.node_state == NodeState.FAILED
    assert updated.response is response


# --- SchedulerState.ready_nodes ---


def test_ready_nodes_returns_pending_node_with_all_deps_completed():
    """ready_nodes() includes a PENDING node whose only dependency is COMPLETED."""
    dep = _make_node(node_state=NodeState.COMPLETED)
    candidate = _make_node(dependencies=frozenset({dep.request.id}))
    state = _make_state([dep, candidate])
    assert candidate in state.ready_nodes()


def test_ready_nodes_excludes_pending_node_with_unsatisfied_dep():
    """ready_nodes() excludes a PENDING node whose dependency is not yet COMPLETED."""
    dep = _make_node(node_state=NodeState.PENDING)
    blocked = _make_node(dependencies=frozenset({dep.request.id}))
    state = _make_state([dep, blocked])
    assert blocked not in state.ready_nodes()


def test_ready_nodes_excludes_running_and_completed_nodes():
    """ready_nodes() excludes nodes that are already RUNNING or COMPLETED."""
    running = _make_node(node_state=NodeState.RUNNING)
    completed = _make_node(node_state=NodeState.COMPLETED)
    state = _make_state([running, completed])
    result = state.ready_nodes()
    assert running not in result
    assert completed not in result


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
        request.priority = Priority.HIGH  # type: ignore[misc]


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


# --- Edit ---


def test_edit_is_frozen():
    """Edit raises on direct field mutation."""
    edit = Edit(path="a.py", old="foo", new="bar")
    with pytest.raises(Exception):
        edit.path = "b.py"  # type: ignore[misc]


# --- DeltaState ---


def test_delta_state_defaults_to_empty_lists():
    """DeltaState fields all default to empty lists."""
    delta = DeltaState()
    assert delta.edits == []
    assert delta.new_files == []
    assert delta.dependencies == []


# --- StateView ---


def test_state_view_stores_files_as_paths():
    """StateView accepts a list of string paths for the files field."""
    view = StateView(
        artifact_name="myapp",
        language="python",
        files=["src/main.py", "src/utils.py"],
        dependencies=["requests"],
    )
    assert view.files == ["src/main.py", "src/utils.py"]


# --- PlanResponse ---


def test_plan_response_kind_discriminator():
    """PlanResponse requires kind to be 'plan'."""
    pr = PlanResponse(
        kind="plan",
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


# --- ToolCallRequest ---


def test_tool_call_request_kind_discriminator():
    """ToolCallRequest requires kind to be 'tool_call'."""
    req = ToolCallRequest(kind="tool_call", name="read_file", arguments={"path": "a.py"})
    assert req.kind == "tool_call"


# --- ToolCallResponse ---


def test_tool_call_response_kind_discriminator():
    """ToolCallResponse requires kind to be 'tool_response'."""
    resp = ToolCallResponse(kind="tool_response", name="read_file", success=True, result="content")
    assert resp.kind == "tool_response"


def test_tool_call_response_error_defaults_to_none():
    """ToolCallResponse.error defaults to None when not provided."""
    resp = ToolCallResponse(kind="tool_response", name="read_file", success=True, result="ok")
    assert resp.error is None


# --- FailureKind ---


def test_failure_kind_has_expected_values():
    """FailureKind enum contains all expected classification values."""
    assert FailureKind.INVALID_JSON.value == "invalid_json"
    assert FailureKind.TRUNCATED_OUTPUT.value == "truncated_output"
    assert FailureKind.PROVIDER_ERROR.value == "provider_error"
    assert FailureKind.TIMEOUT.value == "timeout"
    assert FailureKind.MAX_ITERATIONS.value == "max_iterations"
    assert FailureKind.TOOL_ERROR.value == "tool_error"
    assert FailureKind.UNKNOWN.value == "unknown"


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
