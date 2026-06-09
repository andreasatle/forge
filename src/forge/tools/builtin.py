"""Factory functions that build tool registries for worker and integrator agents."""

from forge.core.workspace import Workspace
from forge.tools.blackboard import make_read_blackboard_tool, make_write_blackboard_tool
from forge.tools.file_tools import (
    make_list_files_tool,
    make_read_file_tool,
    make_replace_in_file_tool,
    make_write_file_tool,
)
from forge.tools.registry import ToolRegistry
from forge.tools.run_tools import make_add_dependency_tool, make_run_tests_tool


def build_read_registry(
    workspace: Workspace,
    artifact_name: str,
    test_command: str | None = None,
) -> ToolRegistry:
    """Read-only tools for worker agents — no disk writes."""
    registry = ToolRegistry()
    registry.register(make_read_file_tool(workspace, artifact_name))
    registry.register(make_list_files_tool(workspace, artifact_name))
    registry.register(make_read_blackboard_tool(workspace))
    if test_command is not None:
        registry.register(make_run_tests_tool(workspace, artifact_name, test_command))
    return registry


def build_write_registry(
    workspace: Workspace,
    artifact_name: str,
    test_command: str | None = None,
    add_dependency_command: str | None = None,
) -> ToolRegistry:
    """Full tool registry including write tools — for integrator agents."""
    registry = ToolRegistry()
    registry.register(make_read_file_tool(workspace, artifact_name))
    registry.register(make_write_file_tool(workspace, artifact_name))
    registry.register(make_replace_in_file_tool(workspace, artifact_name))
    registry.register(make_list_files_tool(workspace, artifact_name))
    registry.register(make_read_blackboard_tool(workspace))
    registry.register(make_write_blackboard_tool(workspace))
    if test_command is not None:
        registry.register(make_run_tests_tool(workspace, artifact_name, test_command))
    if add_dependency_command is not None:
        registry.register(make_add_dependency_tool(workspace, artifact_name, add_dependency_command))
    return registry


build_default_registry = build_write_registry
