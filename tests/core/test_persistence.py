"""Tests for save_run and load_run persistence functions."""

from pathlib import Path

import pytest

from forge.core.models import (
    AgentRequest,
    AgentType,
    DAGNode,
    NodeState,
    PlanSpec,
    RequestSource,
    SchedulerState,
)
from forge.core.persistence import load_run, save_run
from forge.core.workspace import Workspace


def _make_node(node_state: NodeState) -> DAGNode:
    request = AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(northstar="test"),
    )
    return DAGNode(request=request, node_state=node_state)


@pytest.fixture()
def workspace(tmp_path: Path) -> Workspace:
    """Return an initialised Workspace rooted at a temporary path."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    return ws


@pytest.fixture()
def state() -> SchedulerState:
    """Return a minimal SchedulerState with no nodes."""
    return SchedulerState(northstar="test goal")


def test_save_run_writes_state_json(state: SchedulerState, workspace: Workspace) -> None:
    """save_run() writes a state.json file to the workspace."""
    save_run(state, workspace)
    assert workspace.state_path().exists()


def test_save_run_returns_state_path(state: SchedulerState, workspace: Workspace) -> None:
    """save_run() returns the path it wrote to."""
    path = save_run(state, workspace)
    assert path == workspace.state_path()


def test_save_run_writes_valid_json(state: SchedulerState, workspace: Workspace) -> None:
    """save_run() writes JSON that can be deserialized back into a SchedulerState."""
    save_run(state, workspace)
    loaded = SchedulerState.model_validate_json(workspace.state_path().read_text())
    assert loaded.northstar == state.northstar


def test_save_run_overwrites_previous(state: SchedulerState, workspace: Workspace) -> None:
    """save_run() overwrites a previous state file rather than creating a second one."""
    save_run(state, workspace)
    updated = state.model_copy(update={"northstar": "updated goal"})
    save_run(updated, workspace)
    assert len(list(workspace.path.glob("*.json"))) == 1


def test_load_run_deserializes_state(state: SchedulerState, workspace: Workspace) -> None:
    """load_run() returns a SchedulerState matching the one that was saved."""
    save_run(state, workspace)
    loaded = load_run(workspace)
    assert loaded.northstar == state.northstar


def test_load_run_resets_running_to_pending(workspace: Workspace) -> None:
    """load_run() resets any RUNNING node back to PENDING on reload."""
    node = _make_node(NodeState.RUNNING)
    s = SchedulerState(northstar="test").add_nodes([node])
    save_run(s, workspace)
    loaded = load_run(workspace)
    assert loaded.dag[node.request.id].node_state == NodeState.PENDING


def test_load_run_leaves_completed_unchanged(workspace: Workspace) -> None:
    """load_run() preserves COMPLETED node state across a save/load cycle."""
    node = _make_node(NodeState.COMPLETED)
    s = SchedulerState(northstar="test").add_nodes([node])
    save_run(s, workspace)
    loaded = load_run(workspace)
    assert loaded.dag[node.request.id].node_state == NodeState.COMPLETED


def test_load_run_leaves_failed_unchanged(workspace: Workspace) -> None:
    """load_run() preserves FAILED node state across a save/load cycle."""
    node = _make_node(NodeState.FAILED)
    s = SchedulerState(northstar="test").add_nodes([node])
    save_run(s, workspace)
    loaded = load_run(workspace)
    assert loaded.dag[node.request.id].node_state == NodeState.FAILED


def test_round_trip(state: SchedulerState, workspace: Workspace) -> None:
    """A save/load round-trip produces a state equal to the original."""
    save_run(state, workspace)
    loaded = load_run(workspace)
    assert loaded.northstar == state.northstar
    assert loaded.max_concurrency == state.max_concurrency
    assert set(loaded.dag.keys()) == set(state.dag.keys())
