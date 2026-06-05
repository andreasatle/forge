import pytest

from forge.core.models import (
    AgentRequest,
    AgentResponse,
    AgentType,
    DAGNode,
    NodeState,
    Priority,
    RequestSource,
    ResponseStatus,
    SchedulerState,
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
    node = _make_node()
    updated = node.with_state(NodeState.RUNNING)
    assert updated.node_state == NodeState.RUNNING
    assert node.node_state == NodeState.PENDING


# --- DAGNode.with_response ---


def test_with_response_sets_completed_for_completed_status():
    node = _make_node()
    response = _make_response(node.request.id, status=ResponseStatus.COMPLETED)
    updated = node.with_response(response)
    assert updated.node_state == NodeState.COMPLETED
    assert updated.response is response


def test_with_response_sets_failed_for_failed_status():
    node = _make_node()
    response = _make_response(node.request.id, status=ResponseStatus.FAILED)
    updated = node.with_response(response)
    assert updated.node_state == NodeState.FAILED
    assert updated.response is response


# --- SchedulerState.ready_nodes ---


def test_ready_nodes_returns_pending_node_with_all_deps_completed():
    dep = _make_node(node_state=NodeState.COMPLETED)
    candidate = _make_node(dependencies=frozenset({dep.request.id}))
    state = _make_state([dep, candidate])
    assert candidate in state.ready_nodes()


def test_ready_nodes_excludes_pending_node_with_unsatisfied_dep():
    dep = _make_node(node_state=NodeState.PENDING)
    blocked = _make_node(dependencies=frozenset({dep.request.id}))
    state = _make_state([dep, blocked])
    assert blocked not in state.ready_nodes()


def test_ready_nodes_excludes_running_and_completed_nodes():
    running = _make_node(node_state=NodeState.RUNNING)
    completed = _make_node(node_state=NodeState.COMPLETED)
    state = _make_state([running, completed])
    result = state.ready_nodes()
    assert running not in result
    assert completed not in result


# --- SchedulerState.add_nodes ---


def test_add_nodes_returns_new_instance_original_unchanged():
    state = SchedulerState(northstar="ns")
    node = _make_node()
    new_state = state.add_nodes([node])
    assert node.request.id in new_state.dag
    assert node.request.id not in state.dag


# --- SchedulerState.update_node ---


def test_update_node_returns_new_instance_original_unchanged():
    node = _make_node()
    state = _make_state([node])
    updated = node.with_state(NodeState.RUNNING)
    new_state = state.update_node(updated)
    assert new_state.dag[node.request.id].node_state == NodeState.RUNNING
    assert state.dag[node.request.id].node_state == NodeState.PENDING


# --- AgentRequest immutability ---


def test_agent_request_raises_on_mutation():
    request = _make_request()
    with pytest.raises(Exception):
        request.priority = Priority.HIGH  # type: ignore[misc]
