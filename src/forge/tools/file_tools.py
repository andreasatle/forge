"""File tools for reading, writing, listing, and patching files in a workspace artifact directory."""

from forge.core.workspace import Workspace
from forge.tools.registry import Tool


async def read_file(path: str, workspace: Workspace, artifact_name: str) -> str:
    """Read and return the contents of a file from the named artifact directory."""
    file_path = workspace.artifact_dir(artifact_name) / path
    if not file_path.exists():
        raise FileNotFoundError(f"file not found in workspace outputs: {path!r}")
    return file_path.read_text()


async def list_files(directory: str, workspace: Workspace, artifact_name: str) -> str:
    """Return newline-separated relative paths of all files under directory, or 'empty'."""
    root = workspace.artifact_dir(artifact_name)
    dir_path = root / directory
    if not dir_path.exists() or not dir_path.is_dir():
        return "empty"
    files = sorted(str(f.relative_to(root)) for f in dir_path.rglob("*") if f.is_file())
    return "\n".join(files) if files else "empty"


def make_read_file_tool(workspace: Workspace, artifact_name: str) -> Tool:
    """Return a Tool that reads a file from the named artifact directory."""
    async def fn(path: str) -> str:
        return await read_file(path, workspace, artifact_name)

    return Tool(
        name="read_file",
        description="Read a file from the workspace outputs directory",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to workspace outputs/"},
            },
            "required": ["path"],
        },
        fn=fn,
    )


def make_list_files_tool(workspace: Workspace, artifact_name: str) -> Tool:
    """Return a Tool that lists files in a directory within the named artifact directory."""
    async def fn(directory: str) -> str:
        return await list_files(directory, workspace, artifact_name)

    return Tool(
        name="list_files",
        description="List files in a directory within the workspace outputs",
        parameters={
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory path relative to workspace outputs/",
                },
            },
            "required": ["directory"],
        },
        fn=fn,
    )


def make_write_file_tool(workspace: Workspace, artifact_name: str) -> Tool:
    """Return a Tool that creates or overwrites a file in the named artifact directory."""
    async def write_file(path: str, content: str) -> str:
        target = workspace.artifact_dir(artifact_name) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {path}"

    return Tool(
        name="write_file",
        description="Create or overwrite a file in the workspace outputs directory.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to write"},
                "content": {"type": "string", "description": "File content"},
            },
            "required": ["path", "content"],
        },
        fn=write_file,
    )


def make_replace_in_file_tool(workspace: Workspace, artifact_name: str) -> Tool:
    """Return a Tool that surgically replaces a unique string in a file in the named artifact directory."""
    async def replace_in_file(path: str, old: str, new: str) -> str:
        import re

        target = workspace.artifact_dir(artifact_name) / path
        if not target.exists():
            raise FileNotFoundError(f"file not found: {path}")
        content = target.read_text(encoding="utf-8")
        matches = re.findall(re.escape(old), content)
        if len(matches) == 0:
            raise ValueError(f"pattern not found in {path}")
        if len(matches) > 1:
            raise ValueError(f"pattern not unique in {path} — found {len(matches)} occurrences")
        target.write_text(content.replace(old, new, 1), encoding="utf-8")
        return f"replaced in {path}"

    return Tool(
        name="replace_in_file",
        description="Surgically replace a unique string in a file. Read the file first to pick a unique old string.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to file"},
                "old": {"type": "string", "description": "Unique string to replace"},
                "new": {"type": "string", "description": "Replacement string"},
            },
            "required": ["path", "old", "new"],
        },
        fn=replace_in_file,
    )
