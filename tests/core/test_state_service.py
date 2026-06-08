"""Tests for StateService: build_state_view, apply_delta, and run_tests."""

from unittest.mock import MagicMock, patch

import pytest

from forge.core.models import DeltaState, Edit, FileWrite, RunResult
from forge.core.state_service import StateService, _parse_test_result
from forge.core.workspace import Workspace
from forge.languages.registry import LanguagePlugin


def _ws(tmp_path):
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
    )


def test_build_state_view_returns_correct_file_listing(tmp_path):
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    artifact_dir = ws.artifact_dir("app")
    (artifact_dir / "src").mkdir()
    (artifact_dir / "src" / "main.py").write_text("x = 1")
    (artifact_dir / "README.md").write_text("# hi")

    view = StateService(ws, "app").build_state_view()

    assert sorted(view.files) == ["README.md", "src/main.py"]


def test_build_state_view_returns_empty_lists_for_empty_artifact(tmp_path):
    ws = _ws(tmp_path)
    ws.init_artifact("app")

    view = StateService(ws, "app").build_state_view()

    assert view.files == []
    assert view.dependencies == []


def test_apply_delta_writes_new_files(tmp_path):
    ws = _ws(tmp_path)
    ws.init_artifact("app")

    StateService(ws, "app").apply_delta(
        DeltaState(new_files=[FileWrite(path="src/hello.py", content="print('hi')")])
    )

    assert (ws.artifact_dir("app") / "src" / "hello.py").read_text() == "print('hi')"


def test_apply_delta_applies_edits(tmp_path):
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    (ws.artifact_dir("app") / "a.py").write_text("x = 1\n")

    StateService(ws, "app").apply_delta(
        DeltaState(edits=[Edit(path="a.py", old="x = 1", new="x = 2")])
    )

    assert (ws.artifact_dir("app") / "a.py").read_text() == "x = 2\n"


def test_apply_delta_raises_on_non_unique_old_string(tmp_path):
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    (ws.artifact_dir("app") / "a.py").write_text("x = 1\nx = 1\n")

    with pytest.raises(ValueError, match="not unique"):
        StateService(ws, "app").apply_delta(
            DeltaState(edits=[Edit(path="a.py", old="x = 1", new="x = 2")])
        )


def test_apply_delta_raises_on_old_string_not_found(tmp_path):
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    (ws.artifact_dir("app") / "a.py").write_text("x = 1\n")

    with pytest.raises(ValueError, match="not found"):
        StateService(ws, "app").apply_delta(
            DeltaState(edits=[Edit(path="a.py", old="x = 99", new="x = 2")])
        )


def test_build_state_view_excludes_noise_files(tmp_path):
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

    assert view.files == ["src/main.py"]


def test_apply_delta_is_single_write_boundary(tmp_path):
    """build_state_view must not write any files — only apply_delta may mutate disk."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    (ws.artifact_dir("app") / "a.py").write_text("x = 1\n")

    artifact_dir = ws.artifact_dir("app")
    before = {f: f.stat().st_mtime for f in artifact_dir.rglob("*") if f.is_file()}
    StateService(ws, "app").build_state_view()
    after = {f: f.stat().st_mtime for f in artifact_dir.rglob("*") if f.is_file()}

    assert before == after


# --- run_tests ---


def test_run_tests_returns_passed_when_no_plugin(tmp_path):
    """run_tests returns RunResult(passed=True) when no plugin is configured."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")

    result = StateService(ws, "app").run_tests()

    assert result == RunResult(passed=True)


def test_run_tests_parses_passing_output(tmp_path):
    """run_tests returns RunResult(passed=True) when subprocess reports all tests passing."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    plugin = _plugin()

    mock_proc = MagicMock()
    mock_proc.stdout = "1 passed in 0.1s\n"
    mock_proc.stderr = ""

    with patch("subprocess.run", return_value=mock_proc):
        result = StateService(ws, "app", plugin).run_tests()

    assert result.passed is True


def test_run_tests_parses_failing_output(tmp_path):
    """run_tests returns RunResult(passed=False) with failures when subprocess reports failures."""
    ws = _ws(tmp_path)
    ws.init_artifact("app")
    plugin = _plugin()

    mock_proc = MagicMock()
    mock_proc.stdout = "FAILED tests/test_main.py::test_bar - AssertionError\n1 failed in 0.1s\n"
    mock_proc.stderr = ""

    with patch("subprocess.run", return_value=mock_proc):
        result = StateService(ws, "app", plugin).run_tests()

    assert result.passed is False
    assert len(result.failures) == 1


# --- _parse_test_result ---


def test_parse_test_result_passing():
    """_parse_test_result returns passed=True when output contains 'N passed'."""
    result = _parse_test_result("3 passed in 0.2s")
    assert result.passed is True
    assert result.failures == []


def test_parse_test_result_failing():
    """_parse_test_result returns passed=False when output contains 'N failed'."""
    result = _parse_test_result("FAILED tests/test_x.py::test_y\n1 failed in 0.1s")
    assert result.passed is False
    assert len(result.failures) == 1


def test_parse_test_result_timeout():
    """_parse_test_result returns passed=False with 'timed out' failure on timeout output."""
    result = _parse_test_result("timed out after 60 seconds")
    assert result.passed is False
    assert "timed out" in result.failures
