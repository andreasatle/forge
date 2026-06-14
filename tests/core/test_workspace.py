"""Tests for Workspace directory initialisation, reset, and path helpers."""

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from forge.core.models import (
    SchedulerState,
)
from forge.core.persistence import load_run, save_run
from forge.core.workspace import Workspace, run_git
from forge.languages.registry import LanguagePlugin


def _make_plugin(name: str = "python") -> LanguagePlugin:
    return LanguagePlugin(
        name=name,
        init_command="true",
        test_command="test-cmd",
        sync_command="true",
        prompt_supplement="test",
        work_output_example="",
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


def test_init_creates_telemetry_dir(tmp_path: Path) -> None:
    """init() creates the framework telemetry directory inside the workspace."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    assert ws.telemetry_dir().is_dir()


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
    assert {path.name for path in ws.artifact_dir("codebase").iterdir()} == {
        ".git",
        "existing.txt",
    }


def test_init_artifact_initializes_git_when_plugin_is_none(tmp_path: Path) -> None:
    """init_artifact initializes an empty git repo when plugin is None."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    ws.init_artifact("codebase")
    assert ws.artifact_dir("codebase").is_dir()
    assert (ws.artifact_dir("codebase") / ".git").is_dir()


def test_init_artifact_runs_git_init_for_language_backed_artifacts(tmp_path: Path) -> None:
    """init_artifact initializes a git repo when a plugin is provided and the directory is new."""
    ws = Workspace(tmp_path / "ws")
    ws.init()

    git_cmds: list[Any] = []

    def _run(cmd: Any, **kwargs: Any) -> MagicMock:
        if isinstance(cmd, list) and cmd[0] == "git":
            git_cmds.append(cmd)
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("forge.core.workspace.shutil.which", return_value="/usr/bin/git"):
        with patch("subprocess.run", side_effect=_run):
            ws.init_artifact("codebase", _make_plugin())

    assert ["git", "init", "-b", "main"] in git_cmds
    assert ["git", "add", "-A"] in git_cmds
    assert any(
        c[:3] == ["git", "commit", "--allow-empty"] and "init: codebase" in c[-1] for c in git_cmds
    )


def test_init_artifact_raises_when_git_not_available(tmp_path: Path) -> None:
    """init_artifact raises RuntimeError when git is not found in PATH."""
    ws = Workspace(tmp_path / "ws")
    ws.init()

    with patch("forge.core.workspace.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="git is required"):
            ws.init_artifact("codebase", _make_plugin())


# --- create_worktree ---


def test_create_worktree_returns_expected_path(tmp_path: Path) -> None:
    """create_worktree returns workspace.path / '{artifact}-work-{node_id}'."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    ws.init_artifact("codebase")

    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        result = ws.create_worktree("codebase", "abc123")

    assert result == ws.path / "codebase-work-abc123"


def test_create_worktree_runs_correct_git_command(tmp_path: Path) -> None:
    """create_worktree calls git worktree add with branch and path arguments."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    ws.init_artifact("codebase")

    captured: list[Any] = []

    def _run(cmd: Any, **kwargs: Any) -> MagicMock:
        captured.append(cmd)
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=_run):
        ws.create_worktree("codebase", "abc123")

    assert len(captured) == 1
    cmd = captured[0]
    assert cmd[:4] == ["git", "worktree", "add", "-b"]
    assert cmd[4] == "work/abc123"
    assert "codebase-work-abc123" in cmd[5]
    assert cmd[6] == "main"


def test_workspace_git_commands_ignore_inherited_hook_git_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workspace git commands target the artifact repo even inside a git hook env."""
    outer = tmp_path / "outer"
    outer.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=outer, check=True)
    (outer / "README.md").write_text("outer\n")
    subprocess.run(["git", "add", "-A"], cwd=outer, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Forge Test",
            "-c",
            "user.email=forge-test@example.com",
            "commit",
            "-m",
            "init outer",
        ],
        cwd=outer,
        check=True,
    )
    monkeypatch.setenv("GIT_DIR", str(outer / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(outer))

    ws = Workspace(tmp_path / "ws")
    ws.init()
    ws.init_artifact("codebase")
    worktree = ws.create_worktree("codebase", "abc123")

    assert (ws.artifact_dir("codebase") / ".git").exists()
    branch = run_git(
        ["branch", "--show-current"],
        cwd=worktree,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert branch == "work/abc123"

    outer_work_branches = subprocess.run(
        ["git", "branch", "--list", "work/*"],
        cwd=outer,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert outer_work_branches == ""
    ws.remove_worktree("codebase", "abc123")


# --- remove_worktree ---


def test_remove_worktree_runs_two_git_commands(tmp_path: Path) -> None:
    """remove_worktree issues git worktree remove and git branch -D."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    ws.init_artifact("codebase")

    captured: list[Any] = []

    def _run(cmd: Any, **kwargs: Any) -> MagicMock:
        captured.append(cmd)
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=_run):
        ws.remove_worktree("codebase", "abc123")

    assert len(captured) == 2
    assert captured[0][:3] == ["git", "worktree", "remove"]
    assert "--force" in captured[0]
    assert captured[1] == ["git", "branch", "-D", "work/abc123"]


# --- get_current_sha ---


def test_get_current_sha_returns_stripped_sha(tmp_path: Path) -> None:
    """get_current_sha returns the stdout of git rev-parse HEAD, stripped."""
    ws = Workspace(tmp_path / "ws")
    ws.init()
    ws.init_artifact("codebase")

    mock_result = MagicMock()
    mock_result.stdout = "deadbeefcafe1234\n"

    with patch("subprocess.run", return_value=mock_result):
        sha = ws.get_current_sha("codebase")

    assert sha == "deadbeefcafe1234"
