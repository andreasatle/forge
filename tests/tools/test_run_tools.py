"""Tests for run_tests function and make_run_tests_tool factory."""

# pyright: reportPrivateUsage=false
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge.core.workspace import Workspace
from forge.tools.run_tools import (
    _parse_test_output,
    make_run_tests_tool,
    run_tests,
)
from forge.tools.schemas import (
    RunTestsRequest,
    RunTestsResponse,
)

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
    output, returncode = await run_tests(workspace, _ARTIFACT, "echo hello")
    assert "hello" in output
    assert returncode == 0


async def test_run_tests_returns_output_from_failing_command(workspace: Workspace) -> None:
    """run_tests returns combined output even when the command exits with a non-zero code."""
    output, returncode = await run_tests(workspace, _ARTIFACT, "echo failure_marker; exit 1")
    assert "failure_marker" in output
    assert returncode == 1


async def test_run_tests_returns_timeout_message_on_timeout(workspace: Workspace) -> None:
    """run_tests returns a timeout message when asyncio.wait_for raises TimeoutError."""
    mock_proc = MagicMock()
    with (
        patch("asyncio.create_subprocess_shell", new_callable=AsyncMock, return_value=mock_proc),
        patch("asyncio.wait_for", side_effect=TimeoutError()),
    ):
        result = await run_tests(workspace, _ARTIFACT, "sleep 100")

    output, returncode = result
    assert "timed out" in output
    assert returncode == 124
    assert "60" in output


async def test_make_run_tests_tool_returns_tool_with_correct_name(workspace: Workspace) -> None:
    """make_run_tests_tool returns a Tool named 'run_tests'."""
    tool = make_run_tests_tool(workspace, _ARTIFACT, "echo test")
    assert tool.name == "run_tests"


async def test_run_tests_tool_returns_run_tests_response(workspace: Workspace) -> None:
    """make_run_tests_tool fn returns a RunTestsResponse."""
    tool = make_run_tests_tool(workspace, _ARTIFACT, "echo hello")

    result = await tool.fn(RunTestsRequest())

    assert isinstance(result, RunTestsResponse)


async def test_run_tests_tool_marks_zero_exit_as_passing(workspace: Workspace) -> None:
    """make_run_tests_tool fn sets passed=True when the command exits zero."""
    tool = make_run_tests_tool(workspace, _ARTIFACT, "echo 'command succeeded'")

    result = await tool.fn(RunTestsRequest())

    assert isinstance(result, RunTestsResponse)
    assert result.passed is True
    assert result.failures == []
    assert "command succeeded" in result.output


async def test_run_tests_tool_marks_nonzero_exit_as_failing(workspace: Workspace) -> None:
    """make_run_tests_tool fn sets passed=False when the command exits nonzero."""
    cmd = "sh -c \"printf 'command failed\\n'; exit 1\""
    tool = make_run_tests_tool(workspace, _ARTIFACT, cmd)

    result = await tool.fn(RunTestsRequest())

    assert isinstance(result, RunTestsResponse)
    assert result.passed is False
    assert result.failures == ["command failed"]
    assert "command failed" in result.output


async def test_run_tests_tool_marks_timeout_as_failed(workspace: Workspace) -> None:
    """make_run_tests_tool fn sets passed=False and records timeout in failures."""
    mock_proc = MagicMock()
    with (
        patch("asyncio.create_subprocess_shell", new_callable=AsyncMock, return_value=mock_proc),
        patch("asyncio.wait_for", side_effect=TimeoutError()),
    ):
        tool = make_run_tests_tool(workspace, _ARTIFACT, "sleep 100")
        result = await tool.fn(RunTestsRequest())

    assert isinstance(result, RunTestsResponse)
    assert result.passed is False
    assert "timed out" in result.failures


# --- _parse_test_output ---


def test_parse_test_output_zero_exit_returns_true() -> None:
    """_parse_test_output returns passed=True when command exit code is zero."""
    result = _parse_test_output("success", returncode=0)
    assert result.passed is True
    assert result.output == "success"


def test_parse_test_output_nonzero_exit_returns_false() -> None:
    """_parse_test_output returns passed=False when command exit code is nonzero."""
    result = _parse_test_output("failure details", returncode=1)
    assert result.passed is False
    assert result.output == "failure details"
