"""Tool for running tests in a workspace artifact directory."""

import asyncio

from forge.core.workspace import Workspace
from forge.tools.registry import Tool
from forge.tools.schemas import (
    AddDependencyRequest,
    AddDependencyResponse,
    RunTestsRequest,
    RunTestsResponse,
)

_TIMEOUT_SECONDS = 60


async def run_tests(workspace: Workspace, artifact_name: str, test_command: str) -> tuple[str, int]:
    """Run test_command in the artifact directory and return combined output plus exit code."""
    cwd = workspace.artifact_dir(artifact_name)
    try:
        proc = await asyncio.create_subprocess_shell(
            test_command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_TIMEOUT_SECONDS)
        return (stdout + stderr).decode(errors="replace"), proc.returncode or 0
    except TimeoutError:
        return f"run_tests timed out after {_TIMEOUT_SECONDS} seconds", 124


def _parse_test_output(raw: str, returncode: int = 0) -> RunTestsResponse:
    """Parse raw test command output into a structured RunTestsResponse."""
    if "timed out" in raw:
        return RunTestsResponse(passed=False, failures=["timed out"], summary=raw.strip())

    lines = raw.splitlines()
    non_empty = [line.strip() for line in lines if line.strip()]
    summary = non_empty[-1] if non_empty else raw.strip()
    passed = returncode == 0
    failures = [] if passed else [summary]

    return RunTestsResponse(passed=passed, failures=failures, summary=summary)


def make_run_tests_tool(workspace: Workspace, artifact_name: str, test_command: str) -> Tool:
    """Return a Tool that runs test_command in the artifact directory with no LLM-facing parameters."""

    async def fn(req: RunTestsRequest) -> RunTestsResponse:  # type: ignore[misc]
        raw, returncode = await run_tests(workspace, artifact_name, test_command)
        return _parse_test_output(raw, returncode)

    return Tool(
        name="run_tests",
        description="Run the test suite by calling this tool directly. Do NOT try to read a file called 'run_tests' — call this tool instead. Returns test output.",
        request_type=RunTestsRequest,
        response_type=RunTestsResponse,
        fn=fn,  # type: ignore[arg-type]
    )


async def add_dependency(
    workspace: Workspace, artifact_name: str, add_dependency_command: str, package_name: str
) -> str:
    """Run add_dependency_command with package_name substituted in the artifact directory and return combined stdout+stderr."""
    cwd = workspace.artifact_dir(artifact_name)
    cmd = add_dependency_command.format(package=package_name)
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_TIMEOUT_SECONDS)
        return (stdout + stderr).decode(errors="replace")
    except TimeoutError:
        return f"add_dependency timed out after {_TIMEOUT_SECONDS} seconds"


def make_add_dependency_tool(
    workspace: Workspace, artifact_name: str, add_dependency_command: str
) -> Tool:
    """Return a Tool that installs a package using add_dependency_command."""

    async def fn(req: AddDependencyRequest) -> AddDependencyResponse:  # type: ignore[misc]
        output = await add_dependency(workspace, artifact_name, add_dependency_command, req.package)
        success = "timed out" not in output
        return AddDependencyResponse(package=req.package, success=success, output=output)

    return Tool(
        name="add_dependency",
        description=f"Install a package using: {add_dependency_command}. Pass the package name as the argument.",
        request_type=AddDependencyRequest,
        response_type=AddDependencyResponse,
        fn=fn,  # type: ignore[arg-type]
    )
