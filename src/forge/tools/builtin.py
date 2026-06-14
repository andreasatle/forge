"""Factory functions that build tool registries for worker agents."""

from forge.core.workspace import Workspace
from forge.tools.file_tools import (
    make_list_files_tool,
    make_list_files_tool_for_root,
    make_read_file_tool,
    make_read_file_tool_for_root,
    make_replace_in_file_tool_for_root,
    make_write_file_tool_for_root,
)
from forge.tools.registry import ToolRegistry
from forge.tools.run_tools import make_run_tests_tool, make_run_tests_tool_for_root


def build_read_registry(
    workspace: Workspace,
    artifact_name: str,
    test_command: str | None = None,
) -> ToolRegistry:
    """Read-only tools for worker agents — no disk writes."""
    registry = ToolRegistry()
    registry.register(make_read_file_tool(workspace, artifact_name))
    registry.register(make_list_files_tool(workspace, artifact_name))
    if test_command is not None:
        registry.register(make_run_tests_tool(workspace, artifact_name, test_command))
    return registry


def build_worktree_registry(
    worktree_path: str,
    test_command: str | None = None,
) -> ToolRegistry:
    """Mutating tools scoped to one assigned git worktree."""
    from pathlib import Path

    root = Path(worktree_path)
    registry = ToolRegistry()
    registry.register(make_read_file_tool_for_root(root))
    registry.register(make_list_files_tool_for_root(root))
    registry.register(make_write_file_tool_for_root(root))
    registry.register(make_replace_in_file_tool_for_root(root))
    if test_command is not None:
        registry.register(make_run_tests_tool_for_root(root, test_command))
    return registry
