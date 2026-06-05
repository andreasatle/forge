"""Factory function that builds the default tool registry for a workspace."""

from forge.core.workspace import Workspace
from forge.tools.blackboard import make_read_blackboard_tool, make_write_blackboard_tool
from forge.tools.file_tools import (
    make_list_files_tool,
    make_read_file_tool,
    make_replace_in_file_tool,
    make_write_file_tool,
)
from forge.tools.registry import ToolRegistry


def build_default_registry(workspace: Workspace, artifact_name: str) -> ToolRegistry:
    """Create and return a ToolRegistry pre-loaded with all standard workspace tools for the given artifact."""
    registry = ToolRegistry()
    registry.register(make_read_file_tool(workspace, artifact_name))
    registry.register(make_write_file_tool(workspace, artifact_name))
    registry.register(make_replace_in_file_tool(workspace, artifact_name))
    registry.register(make_list_files_tool(workspace, artifact_name))
    registry.register(make_read_blackboard_tool(workspace))
    registry.register(make_write_blackboard_tool(workspace))
    return registry
