"""Tests for Workspace directory initialisation, reset, and path helpers."""

from pathlib import Path

import pytest

from forge.core.models import (
    SchedulerState,
)
from forge.core.persistence import load_run, save_run
from forge.core.workspace import Workspace
from forge.languages.registry import LanguagePlugin


def _make_plugin(name: str = "python") -> LanguagePlugin:
    return LanguagePlugin(
        name=name,
        package_manager="test-pm",
        init_command="true",
        test_command="test-cmd",
        sync_command="true",
        add_dependency_command="pm add {package}",
        project_structure=["src/", "tests/"],
        prompt_supplement="test",
        delta_example="",
    )


def _make_state() -> SchedulerState:
    return SchedulerState(northstar="test goal")


def test_init_creates_workspace_directory(tmp_path: Path) -> None:
    """init() creates the workspace root directory."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    assert ws.path.is_dir()


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


def test_artifact_dir_returns_workspace_path_slash_name(tmp_path: Path) -> None:
    """artifact_dir() returns workspace.path / name."""
    ws = Workspace(tmp_path / "ws")
    assert ws.artifact_dir("codebase") == ws.path / "codebase"


def test_init_artifact_creates_directory(tmp_path: Path) -> None:
    """init_artifact() creates the artifact directory under the workspace."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    ws.init_artifact("codebase")
    assert ws.artifact_dir("codebase").is_dir()


def test_init_artifact_is_idempotent(tmp_path: Path) -> None:
    """Calling init_artifact() twice for the same name does not raise."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    ws.init_artifact("codebase")
    ws.init_artifact("codebase")  # must not raise


def test_reset_deletes_state_json(tmp_path: Path) -> None:
    """reset() removes the state.json file if it exists."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    ws.state_path().write_text("{}")
    ws.reset([])
    assert not ws.state_path().exists()


def test_reset_deletes_blackboard_json(tmp_path: Path) -> None:
    """reset() removes the blackboard.json file if it exists."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    ws.blackboard_path().write_text("{}")
    ws.reset([])
    assert not ws.blackboard_path().exists()


def test_reset_clears_artifact_directory_contents(tmp_path: Path) -> None:
    """reset() removes all files inside the named artifact directory."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    ws.init_artifact("codebase")
    (ws.artifact_dir("codebase") / "main.py").write_text("code")
    ws.reset(["codebase"])
    assert list(ws.artifact_dir("codebase").iterdir()) == []


def test_reset_keeps_artifact_directory(tmp_path: Path) -> None:
    """reset() preserves the artifact directory itself after clearing its contents."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    ws.init_artifact("codebase")
    ws.reset(["codebase"])
    assert ws.artifact_dir("codebase").is_dir()


def test_reset_handles_multiple_artifact_names(tmp_path: Path) -> None:
    """reset() clears contents of all named artifact directories."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    ws.init_artifact("codebase")
    ws.init_artifact("docs")
    (ws.artifact_dir("codebase") / "main.py").write_text("code")
    (ws.artifact_dir("docs") / "readme.md").write_text("docs")
    ws.reset(["codebase", "docs"])
    assert list(ws.artifact_dir("codebase").iterdir()) == []
    assert list(ws.artifact_dir("docs").iterdir()) == []


def test_reset_keeps_logs_dir(tmp_path: Path) -> None:
    """reset() preserves the logs directory itself after clearing its contents."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    ws.reset([])
    assert ws.logs_dir().is_dir()


def test_reset_does_not_delete_workspace_dir(tmp_path: Path) -> None:
    """reset() does not remove the workspace root directory."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    ws.reset([])
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


def test_init_artifact_skips_init_when_directory_is_not_empty(tmp_path: Path) -> None:
    """init_artifact does not run the language init command when the artifact directory already has content."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    ws.init_artifact("codebase")
    sentinel = ws.artifact_dir("codebase") / "existing.txt"
    sentinel.write_text("already here")
    ws.init_artifact("codebase", _make_plugin())
    assert sentinel.exists()
    assert len(list(ws.artifact_dir("codebase").iterdir())) == 1


def test_init_artifact_skips_init_when_plugin_is_none(tmp_path: Path) -> None:
    """init_artifact does not run any command when plugin is None."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    ws.init_artifact("codebase")
    assert ws.artifact_dir("codebase").is_dir()
    assert list(ws.artifact_dir("codebase").iterdir()) == []
