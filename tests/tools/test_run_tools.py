"""Tests for run_tests function and make_run_tests_tool factory."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge.core.workspace import Workspace
from forge.tools.run_tools import make_run_tests_tool, run_tests

_ARTIFACT = "test-artifact"


@pytest.fixture
def workspace(tmp_path: pytest.TempPathFactory) -> Workspace:
    """Return an initialised Workspace with a test-artifact directory."""
    ws = Workspace(tmp_path)  # type: ignore[arg-type]
    ws.init()
    ws.init_artifact(_ARTIFACT)
    return ws


async def test_run_tests_returns_output_from_successful_command(workspace: Workspace) -> None:
    """run_tests returns stdout output when the command exits successfully."""
    result = await run_tests(workspace, _ARTIFACT, "echo hello")
    assert "hello" in result


async def test_run_tests_returns_output_from_failing_command(workspace: Workspace) -> None:
    """run_tests returns combined output even when the command exits with a non-zero code."""
    result = await run_tests(workspace, _ARTIFACT, "echo failure_marker; exit 1")
    assert "failure_marker" in result


async def test_run_tests_returns_timeout_message_on_timeout(workspace: Workspace) -> None:
    """run_tests returns a timeout message when asyncio.wait_for raises TimeoutError."""
    mock_proc = MagicMock()
    with (
        patch("asyncio.create_subprocess_shell", new_callable=AsyncMock, return_value=mock_proc),
        patch("asyncio.wait_for", side_effect=TimeoutError()),
    ):
        result = await run_tests(workspace, _ARTIFACT, "sleep 100")

    assert "timed out" in result
    assert "60" in result


async def test_make_run_tests_tool_returns_tool_with_correct_name(workspace: Workspace) -> None:
    """make_run_tests_tool returns a Tool named 'run_tests'."""
    tool = make_run_tests_tool(workspace, _ARTIFACT, "echo test")
    assert tool.name == "run_tests"
