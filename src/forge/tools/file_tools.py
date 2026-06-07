"""File tools for reading, writing, listing, and patching files in a workspace artifact directory."""

from forge.core.workspace import Workspace
from forge.tools.registry import Tool
from forge.tools.schemas import (
    ListFilesRequest,
    ListFilesResponse,
    ReadFileRequest,
    ReadFileResponse,
    ReplaceInFileRequest,
    ReplaceInFileResponse,
    WriteFileRequest,
    WriteFileResponse,
)


async def read_file(path: str, workspace: Workspace, artifact_name: str) -> str:
    """Read and return the contents of a file from the named artifact directory."""
    file_path = workspace.artifact_dir(artifact_name) / path
    if not file_path.exists():
        return f'file not found: {path} — use list_files("") to see what files exist, or write_file to create it'
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
        description="Read a file from the workspace outputs directory",
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


def make_write_file_tool(workspace: Workspace, artifact_name: str) -> Tool:
    """Return a Tool that creates or overwrites a file in the named artifact directory."""
    async def fn(req: WriteFileRequest) -> WriteFileResponse:  # type: ignore[misc]
        if req.path.endswith("/"):
            raise ValueError(f"skipped: {req.path} is a directory")
        target = workspace.artifact_dir(artifact_name) / req.path
        if target.exists() and target.is_dir():
            raise ValueError(f"skipped: {req.path} is a directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(req.content, encoding="utf-8")
        return WriteFileResponse(path=req.path)

    return Tool(
        name="write_file",
        description="Create or overwrite a file in the workspace outputs directory.",
        request_type=WriteFileRequest,
        response_type=WriteFileResponse,
        fn=fn,  # type: ignore[arg-type]
    )


def make_replace_in_file_tool(workspace: Workspace, artifact_name: str) -> Tool:
    """Return a Tool that surgically replaces a unique string in a file in the named artifact directory."""
    async def fn(req: ReplaceInFileRequest) -> ReplaceInFileResponse:  # type: ignore[misc]
        import re

        target = workspace.artifact_dir(artifact_name) / req.path
        if not target.exists():
            raise FileNotFoundError(f"file not found: {req.path}")
        content = target.read_text(encoding="utf-8")
        matches = re.findall(re.escape(req.old), content)
        if len(matches) == 0:
            raise ValueError(f"pattern not found in {req.path}")
        if len(matches) > 1:
            raise ValueError(f"pattern not unique in {req.path} — found {len(matches)} occurrences")
        target.write_text(content.replace(req.old, req.new, 1), encoding="utf-8")
        return ReplaceInFileResponse(path=req.path)

    return Tool(
        name="replace_in_file",
        description="Surgically replace a unique string in a file. Read the file first to pick a unique old string.",
        request_type=ReplaceInFileRequest,
        response_type=ReplaceInFileResponse,
        fn=fn,  # type: ignore[arg-type]
    )
