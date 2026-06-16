"""Tests for StateService: build_state_view, apply_work_output, and run_tests."""

# pyright: reportPrivateUsage=false

import subprocess
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


# --- remove_worktree ---


def test_remove_worktree_delegates_to_workspace(tmp_path: Path) -> None:
    """StateService.remove_worktree delegates to Workspace.remove_worktree with the artifact name."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    ss = StateService(ws, "app")

    calls: list[tuple[str, str]] = []

    def _capture(artifact_name: str, node_id: str) -> None:
        calls.append((artifact_name, node_id))

    with patch.object(ws, "remove_worktree", side_effect=_capture):
        ss.remove_worktree("node-abc")

    assert calls == [("app", "node-abc")]


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


async def test_apply_work_output_rejects_stale_dispatch_sha(tmp_path: Path) -> None:
    """apply_work_output raises RuntimeError when framework dispatch_sha does not match HEAD SHA."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    plugin = _plugin()
    ss = StateService(ws, "app", plugin)

    output = WorkOutput(summary="done")

    with patch.object(ws, "get_current_sha", return_value="current-sha-xyz"):
        with pytest.raises(RuntimeError, match="stale"):
            await ss.apply_work_output(output, "node-stale", dispatch_sha="old-sha-abc")


async def test_apply_work_output_accepts_matching_dispatch_sha(tmp_path: Path) -> None:
    """apply_work_output proceeds when framework dispatch_sha matches HEAD SHA."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    plugin = _plugin()
    ss = StateService(ws, "app", plugin)

    worktree_path = tmp_path / "app-work-node-ok"
    worktree_path.mkdir()
    output = WorkOutput(summary="done")

    with patch.object(ws, "get_current_sha", return_value="matching-sha"):
        with patch.object(ws, "worktree_path", return_value=worktree_path):
            with patch.object(ws, "remove_worktree"):
                with patch("subprocess.run", side_effect=_mock_subprocess_apply(worktree_path)):
                    with patch.object(ss, "run_tests", return_value=RunResult(passed=True)):
                        await ss.apply_work_output(output, "node-ok", dispatch_sha="matching-sha")

    assert ss.current_version == 1


async def test_apply_work_output_skips_stale_check_when_no_dispatch_sha(tmp_path: Path) -> None:
    """apply_work_output skips the staleness rejection when dispatch_sha is empty."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    plugin = _plugin()
    ss = StateService(ws, "app", plugin)

    worktree_path = tmp_path / "app-work-node-empty"
    worktree_path.mkdir()
    output = WorkOutput()

    # Integration must succeed regardless of what get_current_sha returns, because
    # the staleness check is skipped when dispatch_sha is empty.  (get_current_sha is
    # still called internally for the pre-merge SHA, so we do not assert_not_called.)
    with patch.object(ws, "worktree_path", return_value=worktree_path):
        with patch.object(ws, "remove_worktree"):
            with patch("subprocess.run", side_effect=_mock_subprocess_apply(worktree_path)):
                with patch.object(ss, "run_tests", return_value=RunResult(passed=True)):
                    await ss.apply_work_output(output, "node-empty")

    assert ss.current_version == 1


async def test_model_supplied_zero_cannot_affect_integration(tmp_path: Path) -> None:
    """Model supplying base_version='0' in JSON is silently ignored; framework dispatch_sha governs."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    ss = StateService(ws, "app")

    # Simulate model returning "0" — WorkOutput drops the field; dispatch_sha comes from framework
    output = WorkOutput.model_validate(
        {"kind": "work_output", "summary": "done", "base_version": "0"}
    )
    assert not hasattr(output, "base_version")

    worktree_path = tmp_path / "app-work-node-zero"
    worktree_path.mkdir()
    # Framework dispatch SHA matches HEAD — integration should succeed
    with patch.object(ws, "get_current_sha", return_value="abc123"):
        with patch.object(ws, "worktree_path", return_value=worktree_path):
            with patch.object(ws, "remove_worktree"):
                with patch("subprocess.run", side_effect=_mock_subprocess_apply(worktree_path)):
                    with patch.object(ss, "run_tests", return_value=RunResult(passed=True)):
                        await ss.apply_work_output(output, "node-zero", dispatch_sha="abc123")

    assert ss.current_version == 1


async def test_model_supplied_garbage_cannot_affect_integration(tmp_path: Path) -> None:
    """Model supplying base_version='garbage' is silently dropped; only dispatch_sha matters."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    ss = StateService(ws, "app")

    output = WorkOutput.model_validate(
        {"kind": "work_output", "summary": "done", "base_version": "garbage-sha"}
    )
    assert not hasattr(output, "base_version")

    # Framework provides a stale dispatch_sha — integration must fail regardless of model JSON
    with patch.object(ws, "get_current_sha", return_value="real-head-sha"):
        with pytest.raises(RuntimeError, match="stale"):
            await ss.apply_work_output(output, "node-garbage", dispatch_sha="wrong-sha")


async def test_orthogonal_nodes_with_different_dispatch_shas(tmp_path: Path) -> None:
    """Orthogonal nodes get different dispatch SHAs; the later-integrating node is stale."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    ss = StateService(ws, "app")

    output_a = WorkOutput(summary="node A work")
    output_b = WorkOutput(summary="node B work")

    worktree_path = tmp_path / "app-work-node-a"
    worktree_path.mkdir()

    sha0 = "sha-at-dispatch-time"
    sha_after_a = "sha-after-node-a-merged"

    # Node A integrates successfully (dispatched at sha0, HEAD still sha0)
    with patch.object(ws, "get_current_sha", return_value=sha0):
        with patch.object(ws, "worktree_path", return_value=worktree_path):
            with patch.object(ws, "remove_worktree"):
                with patch("subprocess.run", side_effect=_mock_subprocess_apply(worktree_path)):
                    with patch.object(ss, "run_tests", return_value=RunResult(passed=True)):
                        await ss.apply_work_output(output_a, "node-a", dispatch_sha=sha0)

    # Now HEAD has moved to sha_after_a; node B was also dispatched at sha0 — it is now stale
    with patch.object(ws, "get_current_sha", return_value=sha_after_a):
        with pytest.raises(RuntimeError, match="stale"):
            await ss.apply_work_output(output_b, "node-b", dispatch_sha=sha0)


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


# --- P0 fixes ---


def test_build_state_view_populates_version_sha_without_plugin(tmp_path: Path) -> None:
    """build_state_view returns a non-empty version_sha when the artifact has a git repo, even with no language plugin."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")

    view = StateService(ws, "app").build_state_view()

    assert view.version_sha != ""


async def test_apply_work_output_commit_failure_includes_git_diagnostics(
    tmp_path: Path,
) -> None:
    """apply_work_output includes stdout/stderr from a failed git commit in the RuntimeError message."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    ss = StateService(ws, "app")
    worktree_path = tmp_path / "app-work-commit-diag"
    worktree_path.mkdir()

    commit_error = subprocess.CalledProcessError(
        returncode=1,
        cmd=["git", "commit", "-m", "work: node-commit-diag"],
        output="nothing to commit, working tree clean\n",
        stderr="error: pre-commit hook failed\n",
    )

    def _run(cmd: object, **kwargs: object) -> MagicMock:
        result = MagicMock()
        result.returncode = 0
        result.stdout = (
            " M src/main.py\n"
            if isinstance(cmd, list)
            and cmd[:3] == ["git", "status", "--porcelain"]
            and kwargs.get("cwd") == worktree_path
            else ""
        )
        result.stderr = ""
        if isinstance(cmd, list) and "commit" in cmd:
            raise commit_error
        return result

    with patch.object(ws, "worktree_path", return_value=worktree_path):
        with patch("subprocess.run", side_effect=_run):
            with pytest.raises(RuntimeError) as exc_info:
                await ss.apply_work_output(WorkOutput(summary="test"), "node-commit-diag")

    msg = str(exc_info.value)
    assert "nothing to commit" in msg
    assert "pre-commit hook failed" in msg


async def test_apply_work_output_merge_failure_includes_git_diagnostics(
    tmp_path: Path,
) -> None:
    """apply_work_output includes stdout/stderr from a failed git merge in the RuntimeError message."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    ss = StateService(ws, "app")
    worktree_path = tmp_path / "app-work-merge-diag"
    worktree_path.mkdir()

    merge_error = subprocess.CalledProcessError(
        returncode=1,
        cmd=["git", "merge", "--no-ff", "work/node-merge-diag"],
        output="CONFLICT (content): Merge conflict in foo.py\n",
        stderr="Automatic merge failed; fix conflicts and then commit the result.\n",
    )

    def _run(cmd: object, **kwargs: object) -> MagicMock:
        result = MagicMock()
        result.returncode = 0
        result.stdout = (
            " M src/main.py\n"
            if isinstance(cmd, list)
            and cmd[:3] == ["git", "status", "--porcelain"]
            and kwargs.get("cwd") == worktree_path
            else ""
        )
        result.stderr = ""
        if isinstance(cmd, list) and "merge" in cmd and "--abort" not in cmd:
            raise merge_error
        return result

    with patch.object(ws, "worktree_path", return_value=worktree_path):
        with patch("subprocess.run", side_effect=_run):
            with pytest.raises(RuntimeError) as exc_info:
                await ss.apply_work_output(WorkOutput(summary="test"), "node-merge-diag")

    msg = str(exc_info.value)
    assert "CONFLICT" in msg
    assert "Automatic merge failed" in msg


async def test_apply_work_output_resets_to_pre_merge_sha_on_test_failure(
    tmp_path: Path,
) -> None:
    """apply_work_output resets to the pre-merge SHA (not HEAD~1) when tests fail after a successful merge."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    pre_merge_sha = ws.get_current_sha("app")

    worktree_path = ws.create_worktree("app", "node-rollback")
    (worktree_path / "new_file.py").write_text("x = 1\n")

    ss = StateService(ws, "app")
    with patch.object(
        ss, "run_tests", return_value=RunResult(passed=False, summary="FAIL", output="FAIL")
    ):
        with pytest.raises(RuntimeError, match="tests failed"):
            await ss.apply_work_output(WorkOutput(summary="added file"), "node-rollback")

    assert ws.get_current_sha("app") == pre_merge_sha
    assert ss.current_version == 0


async def test_apply_work_output_does_not_abort_completed_merge_on_test_failure(
    tmp_path: Path,
) -> None:
    """apply_work_output does not call merge --abort when the merge has already committed."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    ss = StateService(ws, "app")
    worktree_path = tmp_path / "app-work-no-abort"
    worktree_path.mkdir()

    abort_called = False

    def _tracking_run(cmd: object, **kwargs: object) -> MagicMock:
        nonlocal abort_called
        result = MagicMock()
        result.returncode = 0
        result.stdout = (
            " M src/main.py\n"
            if isinstance(cmd, list)
            and cmd[:3] == ["git", "status", "--porcelain"]
            and kwargs.get("cwd") == worktree_path
            else ""
        )
        result.stderr = ""
        if isinstance(cmd, list) and "--abort" in cmd:
            abort_called = True
        return result

    with patch.object(ws, "worktree_path", return_value=worktree_path):
        with patch("subprocess.run", side_effect=_tracking_run):
            with patch.object(
                ss, "run_tests", return_value=RunResult(passed=False, summary="FAIL", output="FAIL")
            ):
                with pytest.raises(RuntimeError, match="tests failed"):
                    await ss.apply_work_output(WorkOutput(summary="test"), "no-abort")

    assert not abort_called


# --- Concurrency: two nodes targeting the same artifact ---


async def test_same_artifact_orthogonal_nodes_second_is_stale_or_integrates(
    tmp_path: Path,
) -> None:
    """Two nodes dispatched from the same SHA each write to different files.

    With dispatch_sha provided, the second node is always rejected as stale
    once the first integration advances HEAD.  The 'or integrates' branch of the
    contract applies when dispatch_sha is omitted — orthogonal file changes would
    merge cleanly because there is no three-way conflict.

    Verifies:
    - stale error names both the dispatch SHA and the current HEAD SHA,
    - node A's file is present on main after A integrates,
    - node B's file is absent (node B was rejected before integration),
    - main HEAD does not change after node B is rejected,
    - version counter does not increment after the stale rejection,
    - worktree removal succeeds for both nodes (merged and uncommitted).
    """
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    sha0 = ws.get_current_sha("app")
    artifact_dir = ws.artifact_dir("app")

    # Both worktrees branched from the same base — mirrors parallel dispatch
    worktree_a = ws.create_worktree("app", "node-orth-a")
    worktree_b = ws.create_worktree("app", "node-orth-b")
    (worktree_a / "file_a.py").write_text("a = 1\n")
    (worktree_b / "file_b.py").write_text("b = 2\n")

    ss = StateService(ws, "app")

    # Node A integrates cleanly — HEAD advances from sha0 to sha1
    await ss.apply_work_output(WorkOutput(summary="added file_a"), "node-orth-a", dispatch_sha=sha0)
    sha1 = ws.get_current_sha("app")
    assert sha1 != sha0
    assert ss.current_version == 1

    # Node B was dispatched at sha0; HEAD is now sha1 — stale rejection
    with pytest.raises(RuntimeError, match="stale") as exc_info:
        await ss.apply_work_output(
            WorkOutput(summary="added file_b"), "node-orth-b", dispatch_sha=sha0
        )

    # Diagnostic: error names both SHAs so the caller knows what changed under it
    err = str(exc_info.value)
    assert sha0 in err
    assert sha1 in err

    # Main branch is valid and unchanged by the stale rejection
    assert ws.get_current_sha("app") == sha1
    assert ss.current_version == 1

    tree = run_git(
        ["ls-tree", "-r", "--name-only", "HEAD"],
        cwd=artifact_dir,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert "file_a.py" in tree
    assert "file_b.py" not in tree  # node B was rejected before integration

    # Worktree cleanup is preserved — merged (A) and uncommitted-dirty (B)
    ws.remove_worktree("app", "node-orth-a")
    ws.remove_worktree("app", "node-orth-b")


async def test_same_artifact_conflicting_nodes_second_is_stale_or_fails_with_merge_conflict(
    tmp_path: Path,
) -> None:
    """Two nodes dispatched from the same SHA both modify the same file.

    When the stale check is bypassed (no dispatch_sha), a three-way git merge
    conflict is detected at integration time and raised as RuntimeError with
    git diagnostics.  With dispatch_sha the same scenario is caught earlier
    (stale), but this test exercises the merge-conflict path to confirm git
    output is preserved in the error and that main is not left in a dirty state.

    Verifies:
    - RuntimeError is raised with git failure text,
    - main HEAD stays at sha1 (node A's work) — aborted merge leaves no residue,
    - version counter stays at 1 after the failed integration,
    - worktree removal succeeds for both nodes (merged and committed-but-not-merged).
    """
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    artifact_dir = ws.artifact_dir("app")

    # Commit a base file that both nodes will divergently modify
    (artifact_dir / "conflict.py").write_text("value = 'base'\n")
    run_git(["add", "conflict.py"], cwd=artifact_dir)
    run_git(["commit", "-m", "fixture: base conflict file"], cwd=artifact_dir)
    sha0 = ws.get_current_sha("app")

    # Both worktrees branched from sha0 — mirrors parallel dispatch
    worktree_a = ws.create_worktree("app", "node-conf-a")
    worktree_b = ws.create_worktree("app", "node-conf-b")
    (worktree_a / "conflict.py").write_text("value = 'version_a'\n")
    (worktree_b / "conflict.py").write_text("value = 'version_b'\n")

    ss = StateService(ws, "app")

    # Node A integrates from sha0 — main now has "version_a"
    await ss.apply_work_output(WorkOutput(summary="conflict A"), "node-conf-a", dispatch_sha=sha0)
    sha1 = ws.get_current_sha("app")
    assert sha1 != sha0
    assert ss.current_version == 1

    # Node B bypasses the stale check (no dispatch_sha).  The commit in the
    # worktree succeeds, but the merge into main ("version_a") detects a
    # three-way conflict against the common ancestor ("base").
    with pytest.raises(RuntimeError) as exc_info:
        await ss.apply_work_output(WorkOutput(summary="conflict B"), "node-conf-b")

    err = str(exc_info.value)
    # Either stale (if the stale-detection path fires) or merge conflict — both are
    # explicit, loud failures with enough information to diagnose the problem.
    assert "stale" in err or "git command failed" in err

    # When a merge conflict fires, git output must be present in the error message
    if "git command failed" in err:
        assert any(
            s.lower() in err.lower() for s in ("CONFLICT", "Automatic merge failed", "conflict")
        )

    # Main HEAD must be at sha1 — the failed merge was aborted, not left dirty
    assert ws.get_current_sha("app") == sha1
    assert ss.current_version == 1

    # Worktree cleanup succeeds for both — merged (A) and committed-but-not-merged (B)
    ws.remove_worktree("app", "node-conf-a")
    ws.remove_worktree("app", "node-conf-b")
