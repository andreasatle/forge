"""Tool for running tests in a workspace artifact directory."""

import asyncio

from forge.core.workspace import Workspace
from forge.tools.registry import Tool

_TIMEOUT_SECONDS = 60


async def run_tests(workspace: Workspace, artifact_name: str, test_command: str) -> str:
    """Run test_command in the artifact directory and return combined stdout+stderr."""
    cwd = workspace.artifact_dir(artifact_name)
    try:
        proc = await asyncio.create_subprocess_shell(
            test_command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_TIMEOUT_SECONDS)
        return (stdout + stderr).decode(errors="replace")
    except TimeoutError:
        return f"run_tests timed out after {_TIMEOUT_SECONDS} seconds"


def make_run_tests_tool(workspace: Workspace, artifact_name: str, test_command: str) -> Tool:
    """Return a Tool that runs test_command in the artifact directory with no LLM-facing parameters."""
    async def fn() -> str:
        return await run_tests(workspace, artifact_name, test_command)

    return Tool(
        name="run_tests",
        description=f"Run tests with: {test_command}. Use after writing or modifying code. Iterate until tests pass.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        fn=fn,
    )
