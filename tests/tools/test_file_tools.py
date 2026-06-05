import pytest

from forge.core.workspace import Workspace
from forge.tools.file_tools import (
    list_files,
    make_replace_in_file_tool,
    make_write_file_tool,
    read_file,
)


@pytest.fixture
def workspace(tmp_path: pytest.TempPathFactory) -> Workspace:
    ws = Workspace(tmp_path)  # type: ignore[arg-type]
    ws.init()
    return ws


async def test_read_file_returns_contents(workspace: Workspace) -> None:
    (workspace.outputs_dir() / "hello.txt").write_text("world")

    result = await read_file("hello.txt", workspace)

    assert result == "world"


async def test_read_file_fails_clearly_on_missing_file(workspace: Workspace) -> None:
    with pytest.raises(FileNotFoundError, match="file not found in workspace outputs"):
        await read_file("no_such_file.txt", workspace)


async def test_list_files_returns_newline_separated_paths(workspace: Workspace) -> None:
    sub = workspace.outputs_dir() / "sub"
    sub.mkdir()
    (sub / "a.txt").write_text("a")
    (sub / "b.txt").write_text("b")

    result = await list_files("sub", workspace)

    assert result == "sub/a.txt\nsub/b.txt"


async def test_list_files_returns_empty_for_missing_directory(workspace: Workspace) -> None:
    result = await list_files("nonexistent", workspace)

    assert result == "empty"


async def test_write_file_creates_file_at_correct_path(workspace: Workspace) -> None:
    tool = make_write_file_tool(workspace)

    result = await tool.fn(path="out.txt", content="hello")

    assert result == "wrote out.txt"
    assert (workspace.outputs_dir() / "out.txt").read_text() == "hello"


async def test_write_file_creates_parent_directories(workspace: Workspace) -> None:
    tool = make_write_file_tool(workspace)

    await tool.fn(path="a/b/c.txt", content="deep")

    assert (workspace.outputs_dir() / "a" / "b" / "c.txt").read_text() == "deep"


async def test_write_file_overwrites_existing_file(workspace: Workspace) -> None:
    target = workspace.outputs_dir() / "f.txt"
    target.write_text("old")
    tool = make_write_file_tool(workspace)

    await tool.fn(path="f.txt", content="new")

    assert target.read_text() == "new"


async def test_replace_in_file_replaces_unique_occurrence(workspace: Workspace) -> None:
    (workspace.outputs_dir() / "r.txt").write_text("hello world")
    tool = make_replace_in_file_tool(workspace)

    result = await tool.fn(path="r.txt", old="world", new="there")

    assert result == "replaced in r.txt"
    assert (workspace.outputs_dir() / "r.txt").read_text() == "hello there"


async def test_replace_in_file_raises_on_pattern_not_found(workspace: Workspace) -> None:
    (workspace.outputs_dir() / "r.txt").write_text("hello world")
    tool = make_replace_in_file_tool(workspace)

    with pytest.raises(ValueError, match="pattern not found"):
        await tool.fn(path="r.txt", old="missing", new="x")


async def test_replace_in_file_raises_on_non_unique_pattern(workspace: Workspace) -> None:
    (workspace.outputs_dir() / "r.txt").write_text("aa aa")
    tool = make_replace_in_file_tool(workspace)

    with pytest.raises(ValueError, match="pattern not unique"):
        await tool.fn(path="r.txt", old="aa", new="bb")


async def test_replace_in_file_fails_on_missing_file(workspace: Workspace) -> None:
    tool = make_replace_in_file_tool(workspace)

    with pytest.raises(FileNotFoundError, match="file not found"):
        await tool.fn(path="no_such.txt", old="x", new="y")
