"""Tests for run_tests function and make_run_tests_tool factory."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge.core.workspace import Workspace
from forge.tools.run_tools import (
    make_add_dependency_tool,
    make_run_tests_tool,
    run_tests,
)
from forge.tools.schemas import (
    AddDependencyRequest,
    AddDependencyResponse,
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


async def test_run_tests_tool_returns_run_tests_response(workspace: Workspace) -> None:
    """make_run_tests_tool fn returns a RunTestsResponse."""
    tool = make_run_tests_tool(workspace, _ARTIFACT, "echo hello")

    result = await tool.fn(RunTestsRequest())

    assert isinstance(result, RunTestsResponse)


async def test_run_tests_tool_parses_passing_pytest_output(workspace: Workspace) -> None:
    """make_run_tests_tool fn sets passed=True when output contains '1 passed'."""
    tool = make_run_tests_tool(workspace, _ARTIFACT, "echo '1 passed in 0.1s'")

    result = await tool.fn(RunTestsRequest())

    assert isinstance(result, RunTestsResponse)
    assert result.passed is True
    assert result.failures == []


async def test_run_tests_tool_parses_failing_pytest_output(workspace: Workspace) -> None:
    """make_run_tests_tool fn sets passed=False and extracts FAILED lines when tests fail."""
    cmd = "printf 'FAILED tests/foo.py::bar\\n1 failed in 0.1s\\n'"
    tool = make_run_tests_tool(workspace, _ARTIFACT, cmd)

    result = await tool.fn(RunTestsRequest())

    assert isinstance(result, RunTestsResponse)
    assert result.passed is False
    assert any("FAILED tests/foo.py::bar" in f for f in result.failures)


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


async def test_make_add_dependency_tool_returns_tool_with_correct_name(
    workspace: Workspace,
) -> None:
    """make_add_dependency_tool returns a Tool named 'add_dependency'."""
    tool = make_add_dependency_tool(workspace, _ARTIFACT, "echo {package}")
    assert tool.name == "add_dependency"


async def test_add_dependency_tool_substitutes_package_in_command(workspace: Workspace) -> None:
    """The add_dependency tool runs the command with the package name substituted."""
    tool = make_add_dependency_tool(workspace, _ARTIFACT, "echo {package}")

    result = await tool.fn(AddDependencyRequest(package="requests"))

    assert isinstance(result, AddDependencyResponse)
    assert "requests" in result.output


async def test_add_dependency_tool_returns_command_output(workspace: Workspace) -> None:
    """The add_dependency tool returns combined stdout+stderr output in the response."""
    tool = make_add_dependency_tool(workspace, _ARTIFACT, "echo installed_{package}")

    result = await tool.fn(AddDependencyRequest(package="numpy"))

    assert isinstance(result, AddDependencyResponse)
    assert "installed_numpy" in result.output


async def test_add_dependency_tool_sets_success_true_on_normal_output(workspace: Workspace) -> None:
    """The add_dependency tool sets success=True when the command completes without timing out."""
    tool = make_add_dependency_tool(workspace, _ARTIFACT, "echo {package}")

    result = await tool.fn(AddDependencyRequest(package="requests"))

    assert isinstance(result, AddDependencyResponse)
    assert result.success is True


async def test_add_dependency_tool_sets_package_field(workspace: Workspace) -> None:
    """The add_dependency tool response includes the package name that was installed."""
    tool = make_add_dependency_tool(workspace, _ARTIFACT, "echo {package}")

    result = await tool.fn(AddDependencyRequest(package="numpy"))

    assert isinstance(result, AddDependencyResponse)
    assert result.package == "numpy"
