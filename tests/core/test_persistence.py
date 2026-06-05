import re
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


def _make_node(node_state: NodeState) -> DAGNode:
    request = AgentRequest(
        agent_type=AgentType.PLAN,
        source=RequestSource.USER,
        spec=PlanSpec(northstar="test"),
    )
    return DAGNode(request=request, node_state=node_state)


@pytest.fixture()
def state() -> SchedulerState:
    return SchedulerState(northstar="test goal")


def test_save_run_creates_directory(state: SchedulerState, tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    save_run(state, runs_dir)
    assert runs_dir.is_dir()


def test_save_run_writes_valid_json(state: SchedulerState, tmp_path: Path) -> None:
    path = save_run(state, tmp_path / "runs")
    loaded = SchedulerState.model_validate_json(path.read_text())
    assert loaded.northstar == state.northstar


def test_save_run_filename_matches_format(state: SchedulerState, tmp_path: Path) -> None:
    path = save_run(state, tmp_path / "runs")
    assert re.fullmatch(r"\d{8}-\d{6}-\d{6}\.json", path.name)


def test_save_run_returns_correct_path(state: SchedulerState, tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    path = save_run(state, runs_dir)
    assert path.parent == runs_dir
    assert path.exists()


def test_save_run_multiple_calls_produce_separate_files(state: SchedulerState, tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    path1 = save_run(state, runs_dir)
    path2 = save_run(state, runs_dir)
    assert path1 != path2
    assert len(list(runs_dir.iterdir())) == 2


def test_load_run_deserializes_state(state: SchedulerState, tmp_path: Path) -> None:
    path = save_run(state, tmp_path / "runs")
    loaded = load_run(path)
    assert loaded.northstar == state.northstar


def test_load_run_resets_running_to_pending(tmp_path: Path) -> None:
    node = _make_node(NodeState.RUNNING)
    s = SchedulerState(northstar="test").add_nodes([node])
    path = save_run(s, tmp_path / "runs")
    loaded = load_run(path)
    assert loaded.dag[node.request.id].node_state == NodeState.PENDING


def test_load_run_leaves_completed_unchanged(tmp_path: Path) -> None:
    node = _make_node(NodeState.COMPLETED)
    s = SchedulerState(northstar="test").add_nodes([node])
    path = save_run(s, tmp_path / "runs")
    loaded = load_run(path)
    assert loaded.dag[node.request.id].node_state == NodeState.COMPLETED


def test_load_run_leaves_failed_unchanged(tmp_path: Path) -> None:
    node = _make_node(NodeState.FAILED)
    s = SchedulerState(northstar="test").add_nodes([node])
    path = save_run(s, tmp_path / "runs")
    loaded = load_run(path)
    assert loaded.dag[node.request.id].node_state == NodeState.FAILED


def test_round_trip(state: SchedulerState, tmp_path: Path) -> None:
    path = save_run(state, tmp_path / "runs")
    loaded = load_run(path)
    assert loaded.northstar == state.northstar
    assert loaded.max_concurrency == state.max_concurrency
    assert set(loaded.dag.keys()) == set(state.dag.keys())
