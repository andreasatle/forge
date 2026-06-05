import json

from forge.core.workspace import Workspace
from forge.tools.registry import Tool


async def read_blackboard(key: str, workspace: Workspace) -> str:
    bb_path = workspace.blackboard_path()
    if not bb_path.exists():
        return "empty"
    data = json.loads(bb_path.read_text())
    return str(data[key]) if key in data else "key not found"


def make_read_blackboard_tool(workspace: Workspace) -> Tool:
    async def fn(key: str) -> str:
        return await read_blackboard(key, workspace)

    return Tool(
        name="read_blackboard",
        description="Read a value from the shared blackboard by key",
        parameters={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "The blackboard key to read"},
            },
            "required": ["key"],
        },
        fn=fn,
    )


def make_write_blackboard_tool(workspace: Workspace) -> Tool:
    async def write_blackboard(key: str, value: str) -> str:
        import json

        path = workspace.blackboard_path()
        data = json.loads(path.read_text()) if path.exists() else {}
        data[key] = value
        path.write_text(json.dumps(data, indent=2))
        return f"wrote blackboard key: {key}"

    return Tool(
        name="write_blackboard",
        description="Write a value to the shared blackboard by key.",
        parameters={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Blackboard key"},
                "value": {"type": "string", "description": "Value to store"},
            },
            "required": ["key", "value"],
        },
        fn=write_blackboard,
    )
