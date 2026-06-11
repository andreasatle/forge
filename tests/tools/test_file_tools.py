"""Tests for workspace file tool functions: read, list, write, and replace."""

import pytest

from forge.core.workspace import Workspace
from forge.tools.file_tools import (
    list_files,
    make_list_files_tool,
    make_replace_in_file_tool,
    make_write_file_tool,
    read_file,
)
from forge.tools.schemas import (
    ListFilesRequest,
    ListFilesResponse,
    ReplaceInFileRequest,
    ReplaceInFileResponse,
    WriteFileRequest,
    WriteFileResponse,
)

_ARTIFACT = "test-artifact"


@pytest.fixture
def workspace(tmp_path: pytest.TempPathFactory) -> Workspace:
    """Return an initialised Workspace with a test-artifact directory."""
    ws = Workspace(tmp_path)  # type: ignore[arg-type]
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


async def test_write_file_creates_file_at_correct_path(workspace: Workspace) -> None:
    """write_file tool creates the file at the given path with the given content."""
    tool = make_write_file_tool(workspace, _ARTIFACT)

    result = await tool.fn(WriteFileRequest(path="out.txt", content="hello"))

    assert isinstance(result, WriteFileResponse)
    assert result.path == "out.txt"
    assert (workspace.artifact_dir(_ARTIFACT) / "out.txt").read_text() == "hello"


async def test_write_file_creates_parent_directories(workspace: Workspace) -> None:
    """write_file tool creates intermediate parent directories when they do not exist."""
    tool = make_write_file_tool(workspace, _ARTIFACT)

    await tool.fn(WriteFileRequest(path="a/b/c.txt", content="deep"))

    assert (workspace.artifact_dir(_ARTIFACT) / "a" / "b" / "c.txt").read_text() == "deep"


async def test_write_file_overwrites_existing_file(workspace: Workspace) -> None:
    """write_file tool overwrites an existing file with the new content."""
    target = workspace.artifact_dir(_ARTIFACT) / "f.txt"
    target.write_text("old")
    tool = make_write_file_tool(workspace, _ARTIFACT)

    result = await tool.fn(WriteFileRequest(path="f.txt", content="new"))

    assert isinstance(result, WriteFileResponse)
    assert target.read_text() == "new"


async def test_write_file_uses_artifact_root_not_outputs_dir(workspace: Workspace) -> None:
    """write_file tool writes to workspace.path / artifact_name / path, not a generic outputs dir."""
    tool = make_write_file_tool(workspace, _ARTIFACT)

    await tool.fn(WriteFileRequest(path="main.py", content="# code"))

    assert (workspace.path / _ARTIFACT / "main.py").exists()
    assert not (workspace.path / "outputs" / "main.py").exists()


async def test_replace_in_file_replaces_unique_occurrence(workspace: Workspace) -> None:
    """replace_in_file tool replaces a unique string and returns a ReplaceInFileResponse."""
    (workspace.artifact_dir(_ARTIFACT) / "r.txt").write_text("hello world")
    tool = make_replace_in_file_tool(workspace, _ARTIFACT)

    result = await tool.fn(ReplaceInFileRequest(path="r.txt", old="world", new="there"))

    assert isinstance(result, ReplaceInFileResponse)
    assert result.path == "r.txt"
    assert (workspace.artifact_dir(_ARTIFACT) / "r.txt").read_text() == "hello there"


async def test_replace_in_file_raises_on_pattern_not_found(workspace: Workspace) -> None:
    """replace_in_file tool raises ValueError when the old string is not found in the file."""
    (workspace.artifact_dir(_ARTIFACT) / "r.txt").write_text("hello world")
    tool = make_replace_in_file_tool(workspace, _ARTIFACT)

    with pytest.raises(ValueError, match="pattern not found"):
        await tool.fn(ReplaceInFileRequest(path="r.txt", old="missing", new="x"))


async def test_replace_in_file_raises_on_non_unique_pattern(workspace: Workspace) -> None:
    """replace_in_file tool raises ValueError when the old string appears more than once."""
    (workspace.artifact_dir(_ARTIFACT) / "r.txt").write_text("aa aa")
    tool = make_replace_in_file_tool(workspace, _ARTIFACT)

    with pytest.raises(ValueError, match="pattern not unique"):
        await tool.fn(ReplaceInFileRequest(path="r.txt", old="aa", new="bb"))


async def test_replace_in_file_fails_on_missing_file(workspace: Workspace) -> None:
    """replace_in_file tool raises FileNotFoundError when the target file does not exist."""
    tool = make_replace_in_file_tool(workspace, _ARTIFACT)

    with pytest.raises(FileNotFoundError, match="file not found"):
        await tool.fn(ReplaceInFileRequest(path="no_such.txt", old="x", new="y"))


async def test_write_file_skips_trailing_slash_path(workspace: Workspace) -> None:
    """write_file tool raises ValueError when path ends with '/' (directory path)."""
    tool = make_write_file_tool(workspace, _ARTIFACT)

    with pytest.raises(ValueError, match="is a directory"):
        await tool.fn(WriteFileRequest(path="src/", content="# code"))


async def test_write_file_skips_existing_directory(workspace: Workspace) -> None:
    """write_file tool raises ValueError when path resolves to an existing directory."""
    (workspace.artifact_dir(_ARTIFACT) / "src").mkdir()
    tool = make_write_file_tool(workspace, _ARTIFACT)

    with pytest.raises(ValueError, match="is a directory"):
        await tool.fn(WriteFileRequest(path="src", content="# code"))
