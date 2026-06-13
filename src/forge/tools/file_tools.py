"""File tools for reading and listing files in a workspace artifact directory."""

from forge.core.workspace import Workspace
from forge.tools.registry import Tool
from forge.tools.schemas import (
    ListFilesRequest,
    ListFilesResponse,
    ReadFileRequest,
    ReadFileResponse,
)


async def read_file(path: str, workspace: Workspace, artifact_name: str) -> str:
    """Read and return the contents of a file from the named artifact directory."""
    file_path = workspace.artifact_dir(artifact_name) / path
    if not file_path.exists():
        return f'file not found: {path} — use list_files("") to see what files exist'
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

    async def fn(req: ReadFileRequest) -> ReadFileResponse:  # type: ignore[misc]
        content = await read_file(req.path, workspace, artifact_name)
        return ReadFileResponse(content=content)

    return Tool(
        name="read_file",
        description="Read a file from the artifact root",
        request_type=ReadFileRequest,
        response_type=ReadFileResponse,
        fn=fn,  # type: ignore[arg-type]
    )


def make_list_files_tool(workspace: Workspace, artifact_name: str) -> Tool:
    """Return a Tool that lists files in a directory within the named artifact directory."""

    async def fn(req: ListFilesRequest) -> ListFilesResponse:  # type: ignore[misc]
        raw = await list_files(req.directory, workspace, artifact_name)
        paths = raw.split("\n") if raw != "empty" else []
        return ListFilesResponse(paths=paths)

    return Tool(
        name="list_files",
        description="List all files in a directory. Always call this before read_file to discover what files exist.",
        request_type=ListFilesRequest,
        response_type=ListFilesResponse,
        fn=fn,  # type: ignore[arg-type]
    )
