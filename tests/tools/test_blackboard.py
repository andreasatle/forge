"""Tests for blackboard read and write tool functions."""

import json

import pytest

from forge.core.workspace import Workspace
from forge.tools.blackboard import make_write_blackboard_tool, read_blackboard


@pytest.fixture
def workspace(tmp_path: pytest.TempPathFactory) -> Workspace:
    """Return an initialised Workspace rooted at a temporary path."""
    ws = Workspace(tmp_path)  # type: ignore[arg-type]
    ws.init()
    return ws


async def test_read_blackboard_returns_value_for_existing_key(workspace: Workspace) -> None:
    """read_blackboard() returns the stored string value for a key that exists."""
    workspace.blackboard_path().write_text(json.dumps({"key1": "value1"}))

    result = await read_blackboard("key1", workspace)

    assert result == "value1"


async def test_read_blackboard_returns_key_not_found_for_missing_key(workspace: Workspace) -> None:
    """read_blackboard() returns 'key not found' when the key is absent from the blackboard."""
    workspace.blackboard_path().write_text(json.dumps({"other": "x"}))

    result = await read_blackboard("missing", workspace)

    assert result == "key not found"


async def test_read_blackboard_returns_empty_if_blackboard_does_not_exist(
    workspace: Workspace,
) -> None:
    """read_blackboard() returns 'empty' when the blackboard file does not exist."""
    result = await read_blackboard("any", workspace)

    assert result == "empty"


async def test_write_blackboard_creates_file_if_missing(workspace: Workspace) -> None:
    """write_blackboard tool creates the blackboard file and stores the key when it is missing."""
    tool = make_write_blackboard_tool(workspace)

    result = await tool.fn(key="k", value="v")

    assert result == "wrote blackboard key: k"
    data = json.loads(workspace.blackboard_path().read_text())
    assert data["k"] == "v"


async def test_write_blackboard_updates_existing_key(workspace: Workspace) -> None:
    """write_blackboard tool overwrites an existing key with the new value."""
    workspace.blackboard_path().write_text(json.dumps({"k": "old"}))
    tool = make_write_blackboard_tool(workspace)

    await tool.fn(key="k", value="new")

    data = json.loads(workspace.blackboard_path().read_text())
    assert data["k"] == "new"


async def test_write_blackboard_preserves_other_keys(workspace: Workspace) -> None:
    """write_blackboard tool leaves other keys unchanged when updating one key."""
    workspace.blackboard_path().write_text(json.dumps({"a": "1", "b": "2"}))
    tool = make_write_blackboard_tool(workspace)

    await tool.fn(key="a", value="updated")

    data = json.loads(workspace.blackboard_path().read_text())
    assert data["b"] == "2"


async def test_write_then_read_blackboard_round_trip(workspace: Workspace) -> None:
    """A write followed by a read returns the value that was written."""
    write_tool = make_write_blackboard_tool(workspace)
    await write_tool.fn(key="x", value="42")

    result = await read_blackboard("x", workspace)

    assert result == "42"
