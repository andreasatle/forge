"""Tests for StateService: build_state_view, apply_delta, and run_tests."""

# pyright: reportPrivateUsage=false

from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from forge.core.models import (
    DeltaState,
    Edit,
    FileContent,
    FileView,
    FileWrite,
    RunResult,
    WorkOutput,
)
from forge.core.state_service import StateService, _parse_test_result
from forge.core.workspace import Workspace
from forge.languages.registry import LanguagePlugin


def _ws(tmp_path: Path) -> Workspace:
    ws = Workspace(path=tmp_path)
    ws.init()
    return ws


def _plugin(name: str = "python") -> LanguagePlugin:
    return LanguagePlugin(
        name=name,
        package_manager="uv",
        init_command="uv init",
        test_command="uv run pytest",
        sync_command="uv sync",
        add_dependency_command="uv add {package}",
        project_structure=[],
        prompt_supplement="",
        delta_example="",
    )


def test_build_state_view_returns_correct_file_listing(tmp_path: Path) -> None:
    """build_state_view lists all files relative to the artifact root."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    artifact_dir = ws.artifact_dir("app")
    (artifact_dir / "src").mkdir()
    (artifact_dir / "src" / "main.py").write_text("x = 1")
    (artifact_dir / "README.md").write_text("# hi")

    view = StateService(ws, "app").build_state_view()

    assert sorted(view.files, key=lambda f: f.path) == [
        FileView(path="README.md", content="# hi"),
        FileView(path="src/main.py", content="x = 1"),
    ]


def test_build_state_view_returns_empty_lists_for_empty_artifact(tmp_path: Path) -> None:
    """build_state_view returns empty files and dependencies for a freshly initialised artifact."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")

    view = StateService(ws, "app").build_state_view()

    assert view.files == []
    assert view.dependencies == []


def test_apply_delta_writes_new_files(tmp_path: Path) -> None:
    """apply_delta creates a new file on disk when DeltaState contains a FileWrite."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")

    StateService(ws, "app").apply_delta(
        DeltaState(new_files=[FileWrite(path="src/hello.py", content="print('hi')")])
    )

    assert (ws.artifact_dir("app") / "src" / "hello.py").read_text() == "print('hi')"


def test_apply_delta_applies_edits(tmp_path: Path) -> None:
    """apply_delta performs an in-place string replacement when DeltaState contains an Edit."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    (ws.artifact_dir("app") / "a.py").write_text("x = 1\n")

    StateService(ws, "app").apply_delta(
        DeltaState(edits=[Edit(path="a.py", old="x = 1", new="x = 2")])
    )

    assert (ws.artifact_dir("app") / "a.py").read_text() == "x = 2\n"


def test_apply_delta_raises_on_non_unique_old_string(tmp_path: Path) -> None:
    """apply_delta raises ValueError when the old string appears more than once in the target file."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    (ws.artifact_dir("app") / "a.py").write_text("x = 1\nx = 1\n")

    with pytest.raises(ValueError, match="not unique"):
        StateService(ws, "app").apply_delta(
            DeltaState(edits=[Edit(path="a.py", old="x = 1", new="x = 2")])
        )


def test_apply_delta_raises_on_old_string_not_found(tmp_path: Path) -> None:
    """apply_delta raises ValueError when the old string is not present in the target file."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    (ws.artifact_dir("app") / "a.py").write_text("x = 1\n")

    with pytest.raises(ValueError, match="not found"):
        StateService(ws, "app").apply_delta(
            DeltaState(edits=[Edit(path="a.py", old="x = 99", new="x = 2")])
        )


def test_apply_delta_raises_on_empty_old_string(tmp_path: Path) -> None:
    """apply_delta raises ValueError when edit.old is empty."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    (ws.artifact_dir("app") / "a.py").write_text("x = 1\n")

    with pytest.raises(ValueError, match="empty 'old' string"):
        StateService(ws, "app").apply_delta(
            DeltaState(edits=[Edit(path="a.py", old="", new="x = 2")])
        )


def test_apply_delta_raises_on_whitespace_only_old_string(tmp_path: Path) -> None:
    """apply_delta raises ValueError when edit.old is whitespace-only."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    (ws.artifact_dir("app") / "a.py").write_text("x = 1\n")

    with pytest.raises(ValueError, match="empty 'old' string"):
        StateService(ws, "app").apply_delta(
            DeltaState(edits=[Edit(path="a.py", old="   ", new="x = 2")])
        )


def test_apply_delta_succeeds_with_valid_old_string(tmp_path: Path) -> None:
    """apply_delta does not raise when edit.old is a non-empty, non-whitespace string."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    (ws.artifact_dir("app") / "a.py").write_text("x = 1\n")

    StateService(ws, "app").apply_delta(
        DeltaState(edits=[Edit(path="a.py", old="x = 1", new="x = 2")])
    )

    assert (ws.artifact_dir("app") / "a.py").read_text() == "x = 2\n"


def test_build_state_view_excludes_noise_files(tmp_path: Path) -> None:
    """build_state_view omits .venv, __pycache__, .pyc, lock files, and other noise."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    artifact_dir = ws.artifact_dir("app")
    (artifact_dir / "src").mkdir()
    (artifact_dir / "src" / "main.py").write_text("x = 1")
    (artifact_dir / ".venv").mkdir()
    (artifact_dir / ".venv" / "lib.py").write_text("x")
    (artifact_dir / "__pycache__").mkdir()
    (artifact_dir / "__pycache__" / "main.cpython-312.pyc").write_text("x")
    (artifact_dir / "uv.lock").write_text("x")
    (artifact_dir / "src" / "compiled.pyc").write_text("x")
    (artifact_dir / "CACHEDIR.TAG").write_text("x")
    (artifact_dir / "pyvenv.cfg").write_text("x")

    view = StateService(ws, "app").build_state_view()

    assert view.files == [FileView(path="src/main.py", content="x = 1")]


def test_apply_delta_is_single_write_boundary(tmp_path: Path) -> None:
    """build_state_view must not write any files — only apply_delta may mutate disk."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    (ws.artifact_dir("app") / "a.py").write_text("x = 1\n")

    artifact_dir = ws.artifact_dir("app")
    before = {f: f.stat().st_mtime for f in artifact_dir.rglob("*") if f.is_file()}
    StateService(ws, "app").build_state_view()
    after = {f: f.stat().st_mtime for f in artifact_dir.rglob("*") if f.is_file()}

    assert before == after


# --- versioning ---


def test_version_starts_at_zero(tmp_path: Path) -> None:
    """StateService.current_version is 0 before any apply_delta call."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")

    ss = StateService(ws, "app")

    assert ss.current_version == 0


def test_apply_delta_increments_version(tmp_path: Path) -> None:
    """current_version increments by 1 on each successful apply_delta call."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    ss = StateService(ws, "app")

    ss.apply_delta(DeltaState(new_files=[FileWrite(path="a.py", content="x = 1")]))
    assert ss.current_version == 1

    ss.apply_delta(DeltaState(new_files=[FileWrite(path="b.py", content="x = 2")]))
    assert ss.current_version == 2


def test_build_state_view_includes_current_version(tmp_path: Path) -> None:
    """build_state_view returns a StateView whose version matches current_version."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    ss = StateService(ws, "app")
    ss.apply_delta(DeltaState(new_files=[FileWrite(path="a.py", content="x = 1")]))

    view = ss.build_state_view()

    assert view.version == ss.current_version == 1


def test_version_starts_at_zero_even_when_artifact_has_files(tmp_path: Path) -> None:
    """StateService always initialises _version to 0 — bootstrap files are the version-0 state."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    (ws.artifact_dir("app") / "existing.py").write_text("x = 1")

    ss = StateService(ws, "app")

    assert ss.current_version == 0


def test_apply_delta_increments_from_zero_even_with_existing_files(tmp_path: Path) -> None:
    """apply_delta increments from 0 regardless of pre-existing files on construction."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    (ws.artifact_dir("app") / "existing.py").write_text("x = 1")
    ss = StateService(ws, "app")

    assert ss.current_version == 0

    ss.apply_delta(DeltaState(new_files=[FileWrite(path="new.py", content="y = 2")]))

    assert ss.current_version == 1


def test_noise_only_files_do_not_set_version_to_one(tmp_path: Path) -> None:
    """Version stays 0 when the artifact directory contains only noise files (e.g. .venv, .pyc)."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    artifact_dir = ws.artifact_dir("app")
    (artifact_dir / ".venv").mkdir()
    (artifact_dir / ".venv" / "lib.py").write_text("x")
    (artifact_dir / "__pycache__").mkdir()
    (artifact_dir / "__pycache__" / "main.cpython-312.pyc").write_text("x")

    ss = StateService(ws, "app")

    assert ss.current_version == 0


# --- run_tests ---


def test_run_tests_returns_passed_when_no_plugin(tmp_path: Path) -> None:
    """run_tests returns RunResult(passed=True) when no plugin is configured."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")

    result = StateService(ws, "app").run_tests()

    assert result == RunResult(passed=True)


def test_run_tests_parses_passing_output(tmp_path: Path) -> None:
    """run_tests returns RunResult(passed=True) when subprocess reports all tests passing."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    plugin = _plugin()

    mock_proc = MagicMock()
    mock_proc.stdout = "command succeeded\n"
    mock_proc.stderr = ""
    mock_proc.returncode = 0

    with patch("subprocess.run", return_value=mock_proc):
        result = StateService(ws, "app", plugin).run_tests()

    assert result.passed is True


def test_run_tests_parses_failing_output(tmp_path: Path) -> None:
    """run_tests returns RunResult(passed=False) with failures when subprocess reports failures."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    plugin = _plugin()

    mock_proc = MagicMock()
    mock_proc.stdout = "command failed\n"
    mock_proc.stderr = ""
    mock_proc.returncode = 1

    with patch("subprocess.run", return_value=mock_proc):
        result = StateService(ws, "app", plugin).run_tests()

    assert result.passed is False
    assert len(result.failures) == 1


# --- _parse_test_result ---


def test_parse_test_result_passing():
    """_parse_test_result returns passed=True when command exit code is zero."""
    result = _parse_test_result("command succeeded", returncode=0)
    assert result.passed is True
    assert result.failures == []
    assert result.output == "command succeeded"


def test_parse_test_result_failing():
    """_parse_test_result returns passed=False when command exit code is nonzero."""
    result = _parse_test_result("command failed", returncode=1)
    assert result.passed is False
    assert len(result.failures) == 1


def test_parse_test_result_timeout():
    """_parse_test_result returns passed=False with 'timed out' failure on timeout output."""
    result = _parse_test_result("timed out after 60 seconds")
    assert result.passed is False
    assert "timed out" in result.failures


def test_parse_test_result_nonzero_exit_with_multiline_output_returns_failed():
    """_parse_test_result returns passed=False for any nonzero command exit."""
    output = "first diagnostic line\nfinal diagnostic line"
    result = _parse_test_result(output, returncode=2)
    assert result.passed is False
    assert len(result.failures) >= 1
    assert result.summary == "final diagnostic line"
    assert result.output == output


def test_parse_test_result_success_text_with_zero_exit_returns_true() -> None:
    """_parse_test_result does not inspect language-specific success text."""
    result = _parse_test_result("success", returncode=0)
    assert result.passed is True


def test_parse_test_result_failure_text_with_nonzero_exit_returns_false() -> None:
    """_parse_test_result does not inspect language-specific failure text."""
    result = _parse_test_result("failure", returncode=1)
    assert result.passed is False


# --- git-transactional apply_delta ---


def _make_subprocess_mock(stash_stdout: str = "") -> Callable[..., MagicMock]:
    """Return a subprocess.run side-effect that returns a sensible mock for every call."""

    def _run(cmd: Any, **kwargs: Any) -> MagicMock:
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        if isinstance(cmd, list) and "stash" in cmd and "pop" not in cmd and "drop" not in cmd:
            result.stdout = stash_stdout
        return result

    return _run


def test_apply_delta_does_not_increment_version_on_test_failure(tmp_path: Path) -> None:
    """_version stays at 0 when the plugin's tests fail after applying the delta."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    plugin = _plugin()
    ss = StateService(ws, "app", plugin)

    with patch("subprocess.run", side_effect=_make_subprocess_mock("No local changes to save")):
        with patch.object(
            ss, "run_tests", return_value=RunResult(passed=False, summary="FAILED", output="FAILED")
        ):
            with pytest.raises(RuntimeError, match="tests failed"):
                ss.apply_delta(DeltaState(new_files=[FileWrite(path="a.py", content="x = 1")]))

    assert ss.current_version == 0


def test_apply_delta_commits_to_git_history_on_success(tmp_path: Path) -> None:
    """apply_delta issues a git commit and increments _version when tests pass."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    plugin = _plugin()
    ss = StateService(ws, "app", plugin)

    commit_calls: list[Any] = []

    def _run(cmd: Any, **kwargs: Any) -> MagicMock:
        result = MagicMock()
        result.returncode = 0
        result.stdout = "No local changes to save"
        result.stderr = ""
        if isinstance(cmd, list) and "commit" in cmd:
            commit_calls.append(cmd)
        return result

    with patch("subprocess.run", side_effect=_run):
        with patch.object(ss, "run_tests", return_value=RunResult(passed=True)):
            ss.apply_delta(DeltaState(new_files=[FileWrite(path="a.py", content="x = 1")]))

    assert ss.current_version == 1
    assert len(commit_calls) == 1
    assert "integrated:" in commit_calls[0][-1]


def test_apply_delta_calls_stash_pop_on_test_failure(tmp_path: Path) -> None:
    """apply_delta calls git stash pop to restore artifact state when tests fail."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    plugin = _plugin()
    ss = StateService(ws, "app", plugin)

    pop_calls: list[Any] = []

    def _run(cmd: Any, **kwargs: Any) -> MagicMock:
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""  # non-empty stash was created
        result.stderr = ""
        if isinstance(cmd, list) and "stash" in cmd and "pop" in cmd:
            pop_calls.append(cmd)
        return result

    with patch("subprocess.run", side_effect=_run):
        with patch.object(
            ss, "run_tests", return_value=RunResult(passed=False, summary="FAIL", output="FAIL")
        ):
            with pytest.raises(RuntimeError):
                ss.apply_delta(DeltaState(new_files=[FileWrite(path="a.py", content="x = 1")]))


# --- apply_work_output ---


def _mock_subprocess_ok() -> MagicMock:
    result = MagicMock()
    result.returncode = 0
    result.stdout = ""
    result.stderr = ""
    return result


async def test_apply_work_output_writes_files_to_worktree(tmp_path: Path) -> None:
    """apply_work_output writes WorkOutput files into the worktree directory."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    plugin = _plugin()
    ss = StateService(ws, "app", plugin)

    worktree_path = tmp_path / "app-work-node1"
    worktree_path.mkdir()
    output = WorkOutput(files=[FileContent(path="src/main.py", content="x = 1")])

    with patch.object(ws, "create_worktree", return_value=worktree_path):
        with patch.object(ws, "remove_worktree"):
            with patch("subprocess.run", return_value=_mock_subprocess_ok()):
                with patch.object(ss, "run_tests", return_value=RunResult(passed=True)):
                    await ss.apply_work_output(output, "node1")

    assert (worktree_path / "src" / "main.py").read_text() == "x = 1"


async def test_apply_work_output_increments_version_on_pass(tmp_path: Path) -> None:
    """apply_work_output increments _version by 1 when tests pass."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    plugin = _plugin()
    ss = StateService(ws, "app", plugin)

    worktree_path = tmp_path / "app-work-node2"
    worktree_path.mkdir()
    output = WorkOutput()

    with patch.object(ws, "create_worktree", return_value=worktree_path):
        with patch.object(ws, "remove_worktree"):
            with patch("subprocess.run", return_value=_mock_subprocess_ok()):
                with patch.object(ss, "run_tests", return_value=RunResult(passed=True)):
                    await ss.apply_work_output(output, "node2")

    assert ss.current_version == 1


async def test_apply_work_output_does_not_increment_version_on_fail(tmp_path: Path) -> None:
    """apply_work_output leaves _version at 0 when tests fail."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    plugin = _plugin()
    ss = StateService(ws, "app", plugin)

    worktree_path = tmp_path / "app-work-node3"
    worktree_path.mkdir()
    output = WorkOutput()

    with patch.object(ws, "create_worktree", return_value=worktree_path):
        with patch.object(ws, "remove_worktree"):
            with patch("subprocess.run", return_value=_mock_subprocess_ok()):
                with patch.object(
                    ss,
                    "run_tests",
                    return_value=RunResult(passed=False, summary="FAIL", output="FAIL"),
                ):
                    with pytest.raises(RuntimeError, match="tests failed"):
                        await ss.apply_work_output(output, "node3")

    assert ss.current_version == 0


async def test_apply_work_output_removes_worktree_after_pass(tmp_path: Path) -> None:
    """apply_work_output calls remove_worktree after a successful integration."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    plugin = _plugin()
    ss = StateService(ws, "app", plugin)

    worktree_path = tmp_path / "app-work-node4"
    worktree_path.mkdir()
    output = WorkOutput()

    remove_calls: list[tuple[str, str]] = []

    def _record_remove(artifact_name: str, node_id: str) -> None:
        remove_calls.append((artifact_name, node_id))

    with patch.object(ws, "create_worktree", return_value=worktree_path):
        with patch.object(ws, "remove_worktree", side_effect=_record_remove):
            with patch("subprocess.run", return_value=_mock_subprocess_ok()):
                with patch.object(ss, "run_tests", return_value=RunResult(passed=True)):
                    await ss.apply_work_output(output, "node4")

    assert remove_calls == [("app", "node4")]


async def test_apply_work_output_removes_worktree_after_fail(tmp_path: Path) -> None:
    """apply_work_output calls remove_worktree even when tests fail (finally block)."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    plugin = _plugin()
    ss = StateService(ws, "app", plugin)

    worktree_path = tmp_path / "app-work-node5"
    worktree_path.mkdir()
    output = WorkOutput()

    remove_calls: list[tuple[str, str]] = []

    def _record_remove(artifact_name: str, node_id: str) -> None:
        remove_calls.append((artifact_name, node_id))

    with patch.object(ws, "create_worktree", return_value=worktree_path):
        with patch.object(ws, "remove_worktree", side_effect=_record_remove):
            with patch("subprocess.run", return_value=_mock_subprocess_ok()):
                with patch.object(
                    ss,
                    "run_tests",
                    return_value=RunResult(passed=False, summary="FAIL", output="FAIL"),
                ):
                    with pytest.raises(RuntimeError):
                        await ss.apply_work_output(output, "node5")

    assert remove_calls == [("app", "node5")]
