"""Tests for Workspace directory initialisation, reset, and path helpers."""

from pathlib import Path

import pytest

from forge.core.models import (
    SchedulerState,
)
from forge.core.persistence import load_run, save_run
from forge.core.workspace import Workspace


def _make_state() -> SchedulerState:
    return SchedulerState(northstar="test goal")


def test_init_creates_workspace_directory(tmp_path: Path) -> None:
    """init() creates the workspace root directory."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    assert ws.path.is_dir()


def test_init_creates_outputs_dir(tmp_path: Path) -> None:
    """init() creates the outputs subdirectory inside the workspace."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    assert ws.outputs_dir().is_dir()


def test_init_creates_logs_dir(tmp_path: Path) -> None:
    """init() creates the logs subdirectory inside the workspace."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    assert ws.logs_dir().is_dir()


def test_init_is_idempotent(tmp_path: Path) -> None:
    """Calling init() twice on the same workspace does not raise."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    ws.init()  # must not raise


def test_init_raises_if_path_is_file(tmp_path: Path) -> None:
    """init() raises NotADirectoryError when the workspace path is an existing file."""
    file_path = tmp_path / "not_a_dir"
    file_path.write_text("occupied")
    ws = Workspace(file_path)
    with pytest.raises(NotADirectoryError):
        ws.init()


def test_reset_deletes_state_json(tmp_path: Path) -> None:
    """reset() removes the state.json file if it exists."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    ws.state_path().write_text("{}")
    ws.reset()
    assert not ws.state_path().exists()


def test_reset_deletes_blackboard_json(tmp_path: Path) -> None:
    """reset() removes the blackboard.json file if it exists."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    ws.blackboard_path().write_text("{}")
    ws.reset()
    assert not ws.blackboard_path().exists()


def test_reset_clears_outputs_contents(tmp_path: Path) -> None:
    """reset() removes all files inside the outputs directory."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    (ws.outputs_dir() / "artifact.txt").write_text("data")
    ws.reset()
    assert list(ws.outputs_dir().iterdir()) == []


def test_reset_clears_logs_contents(tmp_path: Path) -> None:
    """reset() removes all files inside the logs directory."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    (ws.logs_dir() / "run.log").write_text("log")
    ws.reset()
    assert list(ws.logs_dir().iterdir()) == []


def test_reset_keeps_outputs_dir(tmp_path: Path) -> None:
    """reset() preserves the outputs directory itself after clearing its contents."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    ws.reset()
    assert ws.outputs_dir().is_dir()


def test_reset_keeps_logs_dir(tmp_path: Path) -> None:
    """reset() preserves the logs directory itself after clearing its contents."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    ws.reset()
    assert ws.logs_dir().is_dir()


def test_reset_does_not_delete_workspace_dir(tmp_path: Path) -> None:
    """reset() does not remove the workspace root directory."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    ws.reset()
    assert ws.path.is_dir()


def test_save_run_writes_to_state_path(tmp_path: Path) -> None:
    """save_run() writes to the workspace state path and the file exists afterward."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    state = _make_state()
    path = save_run(state, ws)
    assert path == ws.state_path()
    assert ws.state_path().exists()


def test_load_run_reads_from_state_path(tmp_path: Path) -> None:
    """load_run() reads the state previously written by save_run()."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    state = _make_state()
    save_run(state, ws)
    loaded = load_run(ws)
    assert loaded.northstar == state.northstar
