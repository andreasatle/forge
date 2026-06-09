"""Tests for builtin registry builders — read vs write tool sets."""

import forge.agents.worker as worker_module
from forge.core.workspace import Workspace
from forge.tools.builtin import build_read_registry, build_write_registry

BLACKBOARD_TOOL_NAMES = {"read_blackboard", "write_blackboard"}

MUTATING_TOOL_NAMES = {
    "write_file",
    "replace_in_file",
    "add_dependency",
    "write_blackboard",
}


def _names(registry) -> set[str]:
    return set(registry._tools.keys())


def test_read_registry_contains_read_tools(tmp_path):
    """build_read_registry registers the expected read-side tools."""
    ws = Workspace(tmp_path)  # type: ignore[arg-type]
    registry = build_read_registry(ws, "myapp")
    names = _names(registry)
    assert "read_file" in names
    assert "list_files" in names


def test_read_registry_excludes_blackboard_tools(tmp_path):
    """build_read_registry must not expose any blackboard tools to agents."""
    ws = Workspace(tmp_path)  # type: ignore[arg-type]
    registry = build_read_registry(ws, "myapp", test_command="pytest")
    assert _names(registry).isdisjoint(BLACKBOARD_TOOL_NAMES)


def test_read_registry_excludes_write_tools(tmp_path):
    """build_read_registry must not register mutating tools."""
    ws = Workspace(tmp_path)  # type: ignore[arg-type]
    registry = build_read_registry(ws, "myapp")
    names = _names(registry)
    assert names.isdisjoint(MUTATING_TOOL_NAMES)


def test_read_registry_exposes_only_non_mutating_tools(tmp_path):
    """build_read_registry exposes only tools allowed for read-only workers."""
    ws = Workspace(tmp_path)  # type: ignore[arg-type]
    registry = build_read_registry(ws, "myapp", test_command="pytest")
    assert _names(registry) == {"read_file", "list_files", "run_tests"}


def test_read_registry_registers_run_tests_when_command_given(tmp_path):
    """build_read_registry adds run_tests only when test_command is provided."""
    ws = Workspace(tmp_path)  # type: ignore[arg-type]
    without = build_read_registry(ws, "myapp")
    with_cmd = build_read_registry(ws, "myapp", test_command="pytest")
    assert "run_tests" not in _names(without)
    assert "run_tests" in _names(with_cmd)


def test_write_registry_contains_all_tools(tmp_path):
    """build_write_registry registers all tools including write_file and replace_in_file."""
    ws = Workspace(tmp_path)  # type: ignore[arg-type]
    registry = build_write_registry(ws, "myapp", test_command="pytest")
    names = _names(registry)
    assert "read_file" in names
    assert "list_files" in names
    assert "write_file" in names
    assert "replace_in_file" in names
    assert "run_tests" in names


def test_write_registry_excludes_blackboard_tools(tmp_path):
    """build_write_registry must not expose any blackboard tools to agents."""
    ws = Workspace(tmp_path)  # type: ignore[arg-type]
    registry = build_write_registry(ws, "myapp", test_command="pytest", add_dependency_command="uv add {package}")
    assert _names(registry).isdisjoint(BLACKBOARD_TOOL_NAMES)


def test_write_registry_registers_add_dependency_when_command_given(tmp_path):
    """build_write_registry adds add_dependency only when add_dependency_command is provided."""
    ws = Workspace(tmp_path)  # type: ignore[arg-type]
    without = build_write_registry(ws, "myapp")
    with_cmd = build_write_registry(ws, "myapp", add_dependency_command="uv add {package}")
    assert "add_dependency" not in _names(without)
    assert "add_dependency" in _names(with_cmd)


def test_worker_uses_read_registry():
    """Worker module must import and use build_read_registry, not the write registry."""
    import inspect

    src = inspect.getsource(worker_module)
    assert "build_read_registry" in src
    assert "build_write_registry" not in src
    assert "build_default_registry" not in src
