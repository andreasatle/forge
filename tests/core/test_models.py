"""Tests for core Pydantic models: DAGNode, SchedulerState, and AgentRequest immutability."""

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
