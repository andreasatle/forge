"""Tests for workspace file tool functions: read and list."""

from pathlib import Path
from typing import cast

import pytest

from forge.core.workspace import Workspace
from forge.tools.file_tools import (
    list_files,
    make_list_files_tool,
    make_replace_in_file_tool_for_root,
    read_file,
    write_file_to_root,
)
from forge.tools.schemas import (
    ListFilesRequest,
    ListFilesResponse,
    ReplaceInFileRequest,
    ReplaceInFileResponse,
)

_ARTIFACT = "test-artifact"


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    """Return an initialised Workspace with a test-artifact directory."""
    ws = Workspace(tmp_path)
    ws.init()
    ws.init_artifact(_ARTIFACT)
    return ws


async def test_read_file_returns_contents(workspace: Workspace) -> None:
    """read_file() returns the full text content of an existing workspace output file."""
    (workspace.artifact_dir(_ARTIFACT) / "hello.txt").write_text("world")

    result = await read_file("hello.txt", workspace, _ARTIFACT)

    assert result == "world"


async def test_read_file_returns_helpful_message_for_missing_file(workspace: Workspace) -> None:
    """read_file() returns a recovery hint instead of raising when the file does not exist."""
    result = await read_file("no_such_file.txt", workspace, _ARTIFACT)

    assert "file not found: no_such_file.txt" in result
    assert "list_files" in result


async def test_list_files_returns_newline_separated_paths(workspace: Workspace) -> None:
    """list_files() returns newline-separated relative paths for all files in the directory."""
    sub = workspace.artifact_dir(_ARTIFACT) / "sub"
    sub.mkdir()
    (sub / "a.txt").write_text("a")
    (sub / "b.txt").write_text("b")

    result = await list_files("sub", workspace, _ARTIFACT)

    assert result == "sub/a.txt\nsub/b.txt"


async def test_list_files_returns_empty_for_missing_directory(workspace: Workspace) -> None:
    """list_files() returns 'empty' when the requested directory does not exist."""
    result = await list_files("nonexistent", workspace, _ARTIFACT)

    assert result == "empty"


async def test_list_files_tool_returns_paths_as_list(workspace: Workspace) -> None:
    """make_list_files_tool fn returns ListFilesResponse with paths as a list."""
    sub = workspace.artifact_dir(_ARTIFACT) / "sub"
    sub.mkdir()
    (sub / "a.txt").write_text("a")
    (sub / "b.txt").write_text("b")
    tool = make_list_files_tool(workspace, _ARTIFACT)

    result = await tool.fn(ListFilesRequest(directory="sub"))

    assert isinstance(result, ListFilesResponse)
    assert result.paths == ["sub/a.txt", "sub/b.txt"]


async def test_list_files_tool_returns_empty_list_for_missing_directory(
    workspace: Workspace,
) -> None:
    """make_list_files_tool fn returns ListFilesResponse with empty paths for missing directory."""
    tool = make_list_files_tool(workspace, _ARTIFACT)

    result = await tool.fn(ListFilesRequest(directory="nonexistent"))

    assert isinstance(result, ListFilesResponse)
    assert result.paths == []


async def test_write_file_to_root_writes_inside_scoped_root(tmp_path: Path) -> None:
    """write_file_to_root writes content under the provided root."""
    result = await write_file_to_root("src/main.py", "x = 1", tmp_path)

    assert (tmp_path / "src" / "main.py").read_text() == "x = 1"
    assert result.path == "src/main.py"


async def test_write_file_to_root_rejects_path_escape(tmp_path: Path) -> None:
    """write_file_to_root rejects paths outside the scoped root."""
    with pytest.raises(ValueError, match="escapes"):
        await write_file_to_root("../outside.py", "x", tmp_path)


async def test_replace_in_file_tool_replaces_exact_text(tmp_path: Path) -> None:
    """replace_in_file tool replaces one exact text occurrence."""
    (tmp_path / "main.py").write_text("x = 1\n")
    tool = make_replace_in_file_tool_for_root(tmp_path)

    result = cast(
        ReplaceInFileResponse,
        await tool.fn(ReplaceInFileRequest(path="main.py", old="1", new="2")),
    )

    assert (tmp_path / "main.py").read_text() == "x = 2\n"
    assert result.replacements == 1
