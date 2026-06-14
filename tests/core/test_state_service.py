"""Tests for StateService: build_state_view, apply_work_output, and run_tests."""

# pyright: reportPrivateUsage=false

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from forge.core.models import (
    FileView,
    RunResult,
    WorkOutput,
)
from forge.core.state_service import StateService, _parse_test_result
from forge.core.workspace import Workspace, run_git
from forge.languages.registry import LanguagePlugin


def _ws(tmp_path: Path) -> Workspace:
    ws = Workspace(path=tmp_path)
    ws.init()
    return ws


def _plugin(name: str = "python") -> LanguagePlugin:
    return LanguagePlugin(
        name=name,
        init_command="true",
        test_command="true",
        sync_command="true",
        prompt_supplement="",
        work_output_example="",
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


def test_build_state_view_does_not_mutate_disk(tmp_path: Path) -> None:
    """build_state_view must not write any files."""
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
    """StateService.current_version starts at 0."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")

    ss = StateService(ws, "app")

    assert ss.current_version == 0


def test_build_state_view_includes_current_version(tmp_path: Path) -> None:
    """build_state_view returns a StateView whose version matches current_version."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    ss = StateService(ws, "app")
    ss._version = 1

    view = ss.build_state_view()

    assert view.version == ss.current_version == 1


def test_version_starts_at_zero_even_when_artifact_has_files(tmp_path: Path) -> None:
    """StateService always initialises _version to 0 — bootstrap files are the version-0 state."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    (ws.artifact_dir("app") / "existing.py").write_text("x = 1")

    ss = StateService(ws, "app")

    assert ss.current_version == 0


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


# --- apply_work_output ---


def _mock_subprocess_apply(worktree_path: Path):
    def _run(cmd: object, **kwargs: object) -> MagicMock:
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        result.stdout = ""
        if (
            isinstance(cmd, list)
            and cmd[:3] == ["git", "status", "--porcelain"]
            and kwargs.get("cwd") == worktree_path
        ):
            result.stdout = " M src/main.py\n"
        return result

    return _run


async def test_apply_work_output_commits_existing_worktree_changes(tmp_path: Path) -> None:
    """apply_work_output commits changes that already exist in the assigned worktree."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    plugin = _plugin()
    ss = StateService(ws, "app", plugin)

    worktree_path = tmp_path / "app-work-node1"
    worktree_path.mkdir()
    output = WorkOutput(summary="changed src/main.py")

    with patch.object(ws, "worktree_path", return_value=worktree_path):
        with patch.object(ws, "remove_worktree"):
            with patch("subprocess.run", side_effect=_mock_subprocess_apply(worktree_path)):
                with patch.object(ss, "run_tests", return_value=RunResult(passed=True)):
                    await ss.apply_work_output(output, "node1")


async def test_apply_work_output_increments_version_on_pass(tmp_path: Path) -> None:
    """apply_work_output increments _version by 1 when tests pass."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    plugin = _plugin()
    ss = StateService(ws, "app", plugin)

    worktree_path = tmp_path / "app-work-node2"
    worktree_path.mkdir()
    output = WorkOutput()

    with patch.object(ws, "worktree_path", return_value=worktree_path):
        with patch.object(ws, "remove_worktree"):
            with patch("subprocess.run", side_effect=_mock_subprocess_apply(worktree_path)):
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

    with patch.object(ws, "worktree_path", return_value=worktree_path):
        with patch.object(ws, "remove_worktree"):
            with patch("subprocess.run", side_effect=_mock_subprocess_apply(worktree_path)):
                with patch.object(
                    ss,
                    "run_tests",
                    return_value=RunResult(passed=False, summary="FAIL", output="FAIL"),
                ):
                    with pytest.raises(RuntimeError, match="tests failed"):
                        await ss.apply_work_output(output, "node3")

    assert ss.current_version == 0


async def test_apply_work_output_does_not_remove_worktree_after_pass(tmp_path: Path) -> None:
    """apply_work_output does not call remove_worktree; cleanup is WorkTaskExecutor's responsibility."""
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

    with patch.object(ws, "worktree_path", return_value=worktree_path):
        with patch.object(ws, "remove_worktree", side_effect=_record_remove):
            with patch("subprocess.run", side_effect=_mock_subprocess_apply(worktree_path)):
                with patch.object(ss, "run_tests", return_value=RunResult(passed=True)):
                    await ss.apply_work_output(output, "node4")

    assert remove_calls == []


async def test_apply_work_output_does_not_remove_worktree_after_fail(tmp_path: Path) -> None:
    """apply_work_output does not call remove_worktree on failure; cleanup is WorkTaskExecutor's responsibility."""
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

    with patch.object(ws, "worktree_path", return_value=worktree_path):
        with patch.object(ws, "remove_worktree", side_effect=_record_remove):
            with patch("subprocess.run", side_effect=_mock_subprocess_apply(worktree_path)):
                with patch.object(
                    ss,
                    "run_tests",
                    return_value=RunResult(passed=False, summary="FAIL", output="FAIL"),
                ):
                    with pytest.raises(RuntimeError):
                        await ss.apply_work_output(output, "node5")

    assert remove_calls == []


# --- apply_work_output staleness check ---


async def test_apply_work_output_rejects_stale_base_version(tmp_path: Path) -> None:
    """apply_work_output raises RuntimeError when output.base_version does not match HEAD SHA."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    plugin = _plugin()
    ss = StateService(ws, "app", plugin)

    output = WorkOutput(base_version="old-sha-abc")

    with patch.object(ws, "get_current_sha", return_value="current-sha-xyz"):
        with pytest.raises(RuntimeError, match="stale"):
            await ss.apply_work_output(output, "node-stale")


async def test_apply_work_output_accepts_correct_base_version(tmp_path: Path) -> None:
    """apply_work_output proceeds when output.base_version matches HEAD SHA."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    plugin = _plugin()
    ss = StateService(ws, "app", plugin)

    worktree_path = tmp_path / "app-work-node-ok"
    worktree_path.mkdir()
    output = WorkOutput(base_version="matching-sha")

    with patch.object(ws, "get_current_sha", return_value="matching-sha"):
        with patch.object(ws, "worktree_path", return_value=worktree_path):
            with patch.object(ws, "remove_worktree"):
                with patch("subprocess.run", side_effect=_mock_subprocess_apply(worktree_path)):
                    with patch.object(ss, "run_tests", return_value=RunResult(passed=True)):
                        await ss.apply_work_output(output, "node-ok")

    assert ss.current_version == 1


async def test_apply_work_output_accepts_empty_base_version(tmp_path: Path) -> None:
    """apply_work_output skips the staleness check when output.base_version is empty."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    plugin = _plugin()
    ss = StateService(ws, "app", plugin)

    worktree_path = tmp_path / "app-work-node-empty"
    worktree_path.mkdir()
    output = WorkOutput()  # base_version defaults to ""

    mock_get_sha = MagicMock()
    with patch.object(ws, "get_current_sha", mock_get_sha):
        with patch.object(ws, "worktree_path", return_value=worktree_path):
            with patch.object(ws, "remove_worktree"):
                with patch("subprocess.run", side_effect=_mock_subprocess_apply(worktree_path)):
                    with patch.object(ss, "run_tests", return_value=RunResult(passed=True)):
                        await ss.apply_work_output(output, "node-empty")

    mock_get_sha.assert_not_called()
    assert ss.current_version == 1


async def test_apply_work_output_excludes_python_cache_files_from_commit(
    tmp_path: Path,
) -> None:
    """Worker-created __pycache__ files are ignored and excluded from integration commits."""
    ws = _ws(tmp_path)
    ws.init_artifact("app", _plugin("python"))
    ss = StateService(ws, "app", _plugin("python"))
    worktree_path = ws.create_worktree("app", "node-pyc")

    (worktree_path / "src").mkdir()
    (worktree_path / "src" / "scraper.py").write_text("def scrape():\n    return 'ok'\n")
    (worktree_path / "src" / "__pycache__").mkdir()
    (worktree_path / "src" / "__pycache__" / "scraper.cpython-312.pyc").write_bytes(b"pyc")

    await ss.apply_work_output(WorkOutput(summary="added scraper"), "node-pyc")

    tree = run_git(
        ["ls-tree", "-r", "--name-only", "HEAD"],
        cwd=ws.artifact_dir("app"),
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert "src/scraper.py" in tree
    assert all("__pycache__" not in path and not path.endswith(".pyc") for path in tree)


async def test_apply_work_output_cleans_dirty_ignored_main_files_before_merge(
    tmp_path: Path,
) -> None:
    """Dirty generated Python artifacts in main do not crash worker branch merges."""
    ws = _ws(tmp_path)
    ws.init_artifact("app", _plugin("python"))
    artifact_dir = ws.artifact_dir("app")
    (artifact_dir / "tests" / "__pycache__").mkdir(parents=True)
    pyc_path = artifact_dir / "tests" / "__pycache__" / "test_scraper.cpython-312.pyc"
    pyc_path.write_bytes(b"tracked-pyc")
    run_git(["add", "-f", "tests/__pycache__/test_scraper.cpython-312.pyc"], cwd=artifact_dir)
    run_git(["commit", "-m", "fixture: tracked pyc"], cwd=artifact_dir)

    worktree_path = ws.create_worktree("app", "node-dirty-main")
    (worktree_path / "src").mkdir()
    (worktree_path / "src" / "scraper.py").write_text("def scrape():\n    return 'ok'\n")
    pyc_path.write_bytes(b"dirty-generated-change")

    ss = StateService(ws, "app", _plugin("python"))
    await ss.apply_work_output(WorkOutput(summary="added scraper"), "node-dirty-main")

    assert ss.current_version == 1
    assert (artifact_dir / "src" / "scraper.py").exists()


async def test_apply_work_output_real_merge_conflict_raises_runtime_error(
    tmp_path: Path,
) -> None:
    """A git merge conflict is converted from CalledProcessError into RuntimeError."""
    ws = _ws(tmp_path)
    ws.init_artifact("app", _plugin("python"))
    artifact_dir = ws.artifact_dir("app")
    (artifact_dir / "conflict.py").write_text("value = 'base'\n")
    run_git(["add", "conflict.py"], cwd=artifact_dir)
    run_git(["commit", "-m", "fixture: base conflict file"], cwd=artifact_dir)

    worktree_path = ws.create_worktree("app", "node-conflict")
    (worktree_path / "conflict.py").write_text("value = 'worker'\n")
    (artifact_dir / "conflict.py").write_text("value = 'main'\n")
    run_git(["add", "conflict.py"], cwd=artifact_dir)
    run_git(["commit", "-m", "fixture: conflicting main change"], cwd=artifact_dir)

    ss = StateService(ws, "app", _plugin("python"))
    with pytest.raises(RuntimeError, match="git command failed"):
        await ss.apply_work_output(WorkOutput(summary="conflicting change"), "node-conflict")
