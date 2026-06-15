"""File tools for reading, listing, and mutating files in scoped artifact roots."""

from pathlib import Path

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


def _safe_path(root: Path, path: str) -> Path:
    resolved_root = root.resolve()
    artifact_name = resolved_root.name
    if path == artifact_name or path.startswith(artifact_name + "/"):
        corrected = path[len(artifact_name) + 1 :] if path.startswith(artifact_name + "/") else ""
        hint = (
            f"\n\nUse:\n\n{corrected}\n\nnot:\n\n{path}"
            if corrected
            else f"\n\nUse the artifact root directly, not: {path}"
        )
        raise ValueError(f"Paths are relative to the artifact root.{hint}")
    candidate = (root / path).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"path escapes artifact root: {path}")
    return candidate


def _is_visible_artifact_file(path: Path, root: Path) -> bool:
    return not any(part.startswith(".") for part in path.relative_to(root).parts)


async def read_file(path: str, workspace: Workspace, artifact_name: str) -> str:
    """Read and return the contents of a file from the named artifact directory."""
    return await read_file_from_root(path, workspace.artifact_dir(artifact_name))


async def read_file_from_root(path: str, root: Path) -> str:
    """Read and return the contents of a file from a scoped root directory."""
    file_path = _safe_path(root, path)
    if not file_path.exists():
        return f'file not found: {path} — use list_files("") to see what files exist'
    return file_path.read_text()


async def list_files(directory: str, workspace: Workspace, artifact_name: str) -> str:
    """Return newline-separated relative paths of all files under directory, or 'empty'."""
    return await list_files_from_root(directory, workspace.artifact_dir(artifact_name))


async def list_files_from_root(directory: str, root: Path) -> str:
    """Return newline-separated relative paths under a scoped root directory."""
    root = root.resolve()
    dir_path = _safe_path(root, directory)
    if not dir_path.exists() or not dir_path.is_dir():
        return "empty"
    files = sorted(
        str(f.relative_to(root))
        for f in dir_path.rglob("*")
        if f.is_file() and _is_visible_artifact_file(f, root)
    )
    return "\n".join(files) if files else "empty"


async def write_file_to_root(path: str, content: str, root: Path) -> WriteFileResponse:
    """Write complete text content to a file under a scoped root directory."""
    file_path = _safe_path(root, path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)
    return WriteFileResponse(path=path, bytes_written=len(content.encode()))


async def replace_in_file_at_root(
    path: str,
    old: str,
    new: str,
    root: Path,
) -> ReplaceInFileResponse:
    """Replace one exact text occurrence in a file under a scoped root directory."""
    file_path = _safe_path(root, path)
    text = file_path.read_text()
    if old not in text:
        raise ValueError(f"old text not found in {path}")
    file_path.write_text(text.replace(old, new, 1))
    return ReplaceInFileResponse(path=path, replacements=1)


def make_read_file_tool(workspace: Workspace, artifact_name: str) -> Tool:
    """Return a Tool that reads a file from the named artifact directory."""
    return make_read_file_tool_for_root(workspace.artifact_dir(artifact_name))


def make_read_file_tool_for_root(root: Path) -> Tool:
    """Return a Tool that reads a file from a scoped root directory."""

    async def fn(req: ReadFileRequest) -> ReadFileResponse:  # type: ignore[misc]
        content = await read_file_from_root(req.path, root)
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
    return make_list_files_tool_for_root(workspace.artifact_dir(artifact_name))


def make_list_files_tool_for_root(root: Path) -> Tool:
    """Return a Tool that lists files under a scoped root directory."""

    async def fn(req: ListFilesRequest) -> ListFilesResponse:  # type: ignore[misc]
        raw = await list_files_from_root(req.directory, root)
        paths = raw.split("\n") if raw != "empty" else []
        return ListFilesResponse(paths=paths)

    return Tool(
        name="list_files",
        description="List all files in a directory. Always call this before read_file to discover what files exist.",
        request_type=ListFilesRequest,
        response_type=ListFilesResponse,
        fn=fn,  # type: ignore[arg-type]
    )


def make_write_file_tool_for_root(root: Path) -> Tool:
    """Return a Tool that writes complete file content under a scoped root directory."""

    async def fn(req: WriteFileRequest) -> WriteFileResponse:  # type: ignore[misc]
        return await write_file_to_root(req.path, req.content, root)

    return Tool(
        name="write_file",
        description="Write complete text content to a file in the assigned worktree.",
        request_type=WriteFileRequest,
        response_type=WriteFileResponse,
        fn=fn,  # type: ignore[arg-type]
    )


def make_replace_in_file_tool_for_root(root: Path) -> Tool:
    """Return a Tool that replaces exact text in a file under a scoped root directory."""

    async def fn(req: ReplaceInFileRequest) -> ReplaceInFileResponse:  # type: ignore[misc]
        return await replace_in_file_at_root(req.path, req.old, req.new, root)

    return Tool(
        name="replace_in_file",
        description="Replace one exact text occurrence in a file in the assigned worktree.",
        request_type=ReplaceInFileRequest,
        response_type=ReplaceInFileResponse,
        fn=fn,  # type: ignore[arg-type]
    )
