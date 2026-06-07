"""Tests for StateService: build_state_view and apply_delta."""

import pytest

from forge.core.models import DeltaState, Edit, FileWrite
from forge.core.state_service import StateService
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
