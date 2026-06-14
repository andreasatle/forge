"""Tool for running tests in a workspace artifact directory."""

import asyncio
from pathlib import Path

from forge.core.workspace import Workspace
from forge.tools.registry import Tool
from forge.tools.schemas import (
    RunTestsRequest,
    RunTestsResponse,
)

_TIMEOUT_SECONDS = 60


async def run_tests(workspace: Workspace, artifact_name: str, test_command: str) -> tuple[str, int]:
    """Run test_command in the artifact directory and return combined output plus exit code."""
    return await run_tests_in_root(workspace.artifact_dir(artifact_name), test_command)


async def run_tests_in_root(root: Path, test_command: str) -> tuple[str, int]:
    """Run test_command in a scoped root directory and return output plus exit code."""
    try:
        proc = await asyncio.create_subprocess_shell(
            test_command,
            cwd=root,
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
        return RunTestsResponse(
            passed=False,
            failures=["timed out"],
            summary=raw.strip(),
            output=raw,
        )

    lines = raw.splitlines()
    non_empty = [line.strip() for line in lines if line.strip()]
    summary = non_empty[-1] if non_empty else raw.strip()
    passed = returncode == 0
    failures = [] if passed else [summary]

    return RunTestsResponse(passed=passed, failures=failures, summary=summary, output=raw)


def make_run_tests_tool(workspace: Workspace, artifact_name: str, test_command: str) -> Tool:
    """Return a Tool that runs test_command in the artifact directory with no LLM-facing parameters."""
    return make_run_tests_tool_for_root(workspace.artifact_dir(artifact_name), test_command)


def make_run_tests_tool_for_root(root: Path, test_command: str) -> Tool:
    """Return a Tool that runs test_command in a scoped root directory."""

    async def fn(req: RunTestsRequest) -> RunTestsResponse:  # type: ignore[misc]
        raw, returncode = await run_tests_in_root(root, test_command)
        return _parse_test_output(raw, returncode)

    return Tool(
        name="run_tests",
        description="Run the test suite by calling this tool directly. Do NOT try to read a file called 'run_tests' — call this tool instead. Returns test output.",
        request_type=RunTestsRequest,
        response_type=RunTestsResponse,
        fn=fn,  # type: ignore[arg-type]
    )
