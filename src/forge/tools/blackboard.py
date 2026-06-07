"""Blackboard tools for reading and writing shared key-value state in the workspace."""

import json

from forge.core.workspace import Workspace
from forge.tools.registry import Tool
from forge.tools.schemas import (
    ReadBlackboardRequest,
    ReadBlackboardResponse,
    WriteBlackboardRequest,
    WriteBlackboardResponse,
)


async def read_blackboard(key: str, workspace: Workspace) -> str:
    """Return the string value for key from the blackboard, or 'empty'/'key not found'."""
    bb_path = workspace.blackboard_path()
    if not bb_path.exists():
        return "empty"
    data = json.loads(bb_path.read_text())
    return str(data[key]) if key in data else "key not found"


def make_read_blackboard_tool(workspace: Workspace) -> Tool:
    """Return a Tool that reads a single key from the workspace blackboard."""
    async def fn(req: ReadBlackboardRequest) -> ReadBlackboardResponse:  # type: ignore[misc]
        raw = await read_blackboard(req.key, workspace)
        value = None if raw in ("empty", "key not found") else raw
        return ReadBlackboardResponse(key=req.key, value=value)

    return Tool(
        name="read_blackboard",
        description="Read a value from the shared blackboard by key",
        request_type=ReadBlackboardRequest,
        response_type=ReadBlackboardResponse,
        fn=fn,  # type: ignore[arg-type]
    )


def make_write_blackboard_tool(workspace: Workspace) -> Tool:
    """Return a Tool that writes a single key-value pair to the workspace blackboard."""
    async def fn(req: WriteBlackboardRequest) -> WriteBlackboardResponse:  # type: ignore[misc]
        path = workspace.blackboard_path()
        data = json.loads(path.read_text()) if path.exists() else {}
        data[req.key] = req.value
        path.write_text(json.dumps(data, indent=2))
        return WriteBlackboardResponse(key=req.key)

    return Tool(
        name="write_blackboard",
        description="Write a value to the shared blackboard by key.",
        request_type=WriteBlackboardRequest,
        response_type=WriteBlackboardResponse,
        fn=fn,  # type: ignore[arg-type]
    )
